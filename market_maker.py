#!/usr/bin/env python3
"""
Bracket Market Maker for Hyperliquid
Uses dual accounts with automated bracket orders (TP + SL) for risk management
Places fixed spread orders and protects each position with exchange-level brackets
"""

import logging
import time
import yaml
import asyncio
import signal
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple

from hyperliquid.utils import constants
from utils.utils import setup
from utils.trade_logger import TradeLogger
from utils.HL_executor import HyperliquidExecutor, Price

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class BracketMarketMaker:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the bracket market maker"""
        self.config = self._load_config(config_path)
        
        # Validate required configuration
        self._validate_config()
        
       
        # Initialize HyperLiquid clients
        self.network = self.config['universal']['network']

        if(self.network == "PROD"):   
            accounts, info = setup(
                base_url=constants.MAINNET_API_URL,
                skip_ws=True
            )
        else:
            accounts, info = setup(
                base_url=constants.TESTNET_API_URL,
                skip_ws=True
            )
            
        self.executor = HyperliquidExecutor(
            accounts['buy']['exchange'],
            accounts['sell']['exchange'], 
            accounts['buy']['address'],
            accounts['sell']['address'],
            info
        )

        self.info = info
        logger.info("Initialized in live trading mode with dual accounts")
        
        # Trading parameters from config
        self.coin = self.config['universal']['asset']
        self.order_size = self.config['trading']['order_size']
        
        # Simple spread configuration for bracket trading
        self.spread_bps = self.config['trading']['base_spread_bps']
        
        # State tracking
        self.current_price = 0.0
        self.current_position = 0.0
        self.bid_order_id = None
        self.ask_order_id = None
        self.running = True
        
        # Order tracking for fill detection
        self.active_orders = {}  # {order_id: {'side': 'buy'/'sell', 'size': float, 'price': float, 'timestamp': float}}
        self.active_stop_orders = {}  # {order_id: {'original_fill': dict, 'stop_price': float, 'size': float, 'side': str}}
        
        # Single-loop coordination - now configurable
        self.trading_lock = asyncio.Lock()     # Protects all trading state
        self.needs_order_refresh = False      # Signal to refresh orders
        self.last_order_refresh = 0.0         # Last time orders were refreshed
        
        # Logging configuration (optional parameters with defaults)  
        trading_config = self.config['trading']
        self.enable_debug_logging = trading_config.get('enable_debug_logging', False)
        self.log_fill_checks = trading_config.get('log_fill_checks', False)
        self.log_strategy_skips = trading_config.get('log_strategy_skips', False)
        
        # Timing configuration (after trading_config is defined)
        self.trading_interval = trading_config.get('trading_interval', 2.0)       # Main trading loop interval (seconds)
        self.order_refresh_interval = trading_config.get('order_refresh_interval', 30.0)  # Minimum refresh interval
        
        # Bracket trading configuration
        self.enable_bracket_protection = trading_config.get('enable_bracket_protection', True)
        self.profit_target_pct = trading_config.get('profit_target_pct', 0.005)  # 0.5% take-profit
        self.stop_loss_pct = trading_config.get('stop_loss_pct', 0.0025)  # 0.25% stop-loss
        self.bracket_buffer = trading_config.get('bracket_buffer', 0.001)  # 0.1% buffer
        
        
        # Single lock for all trading operations - no deadlock concerns
        
        # Profit and PnL tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_fees_paid = 0.0
        self.trade_count = 0
        self.avg_entry_price = 0.0  # Volume-weighted average entry price
        
        # Generate session ID
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.network == "TEST":
            session_id = f"testnet_{session_id}"
        
        # Initialize trade logger
        self.trade_logger = TradeLogger(session_id=session_id, config=self.config)
        
        logger.info(f"Bracket Market Maker initialized - Asset: {self.coin}")
        logger.info(f"Spread: {self.spread_bps} bps, Order size: {self.order_size}")
        logger.info(f"Trading interval: {self.trading_interval}s, Order refresh: {self.order_refresh_interval}s")

    def _validate_config(self):
        """Validate required configuration parameters"""
        required_keys = {
            'universal': ['asset', 'network'],
            'trading': ['order_size', 'base_spread_bps']
        }
        
        for section, keys in required_keys.items():
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
            
            for key in keys:
                if key not in self.config[section]:
                    raise ValueError(f"Missing required config parameter: {section}.{key}")
        
        # Validate parameter ranges
        trading = self.config['trading']
        if trading['order_size'] <= 0:
            raise ValueError("order_size must be positive")
        if trading['base_spread_bps'] <= 0:
            raise ValueError("base_spread_bps must be positive")
        
        # Validate timing parameters
        trading_interval = trading.get('trading_interval', 2.0)
        order_refresh_interval = trading.get('order_refresh_interval', 30.0)
        if trading_interval <= 0:
            raise ValueError("trading_interval must be positive")
        if order_refresh_interval <= 0:
            raise ValueError("order_refresh_interval must be positive")
        if trading_interval >= order_refresh_interval:
            raise ValueError("trading_interval must be less than order_refresh_interval")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {config_path}: {e}")
            return {}

    async def update_position(self):
        """Update current position from exchange"""
        try:
            # Atomically update position under lock protection
            async with self.trading_lock:
                self.current_position = self.executor.get_position(self.coin)
                if self.enable_debug_logging:
                    logger.debug(f"Current position: {self.current_position}")
        except Exception as e:
            logger.error(f"Error updating position: {e}")

    async def check_fills(self) -> bool:
        """Check for order fills and update position/profit tracking. Returns True if any fills occurred."""
        try:
            filled_orders = []      # Orders that were filled - need processing
            cancelled_orders = []   # Orders that were cancelled - just cleanup
            stop_fills = []         # Stop-loss orders that were filled
            
            # Create atomic snapshot of active orders under lock protection
            async with self.trading_lock:
                active_orders_snapshot = dict(self.active_orders)
                active_stop_orders_snapshot = dict(self.active_stop_orders)
            
            # Check regular market-making orders
            for order_id, order_info in active_orders_snapshot.items():
                # Check order status using executor
                is_buy = order_info['side'] == 'buy'
                status = self.executor.check_order_status(self.coin, order_id, is_buy)
                
                if status['status'] == 'filled':
                    # Process the fill for profit tracking
                    await self._process_fill(order_id, order_info, status)
                    filled_orders.append(order_id)
                elif status['status'] == 'cancelled':
                    # Just track for cleanup (no fill processing)
                    cancelled_orders.append(order_id)
                    logger.info(f"Order {order_id} was cancelled")
            
            # Check stop-loss orders
            for order_id, stop_info in active_stop_orders_snapshot.items():
                is_buy = stop_info['side'] == 'buy'
                status = self.executor.check_order_status(self.coin, order_id, is_buy)
                
                if status['status'] == 'filled':
                    # Process bracket order fill (TP or SL)
                    await self._process_bracket_fill(order_id, stop_info, status)
                    stop_fills.append(order_id)
                elif status['status'] == 'cancelled':
                    # Clean up cancelled stop orders
                    async with self.trading_lock:
                        if order_id in self.active_stop_orders:
                            del self.active_stop_orders[order_id]
                    logger.info(f"Stop-loss order {order_id} was cancelled")
            
            # If any fills occurred, clean up and signal order refresh
            if filled_orders or stop_fills:
                await self._cleanup_filled_orders(filled_orders, stop_fills)
                self.needs_order_refresh = True
                return True
            
            # Clean up cancelled orders only
            if cancelled_orders:
                async with self.trading_lock:
                    for order_id in cancelled_orders:
                        if order_id in self.active_orders:
                            del self.active_orders[order_id]
                        # Clear order IDs if they match
                        if order_id == self.bid_order_id:
                            self.bid_order_id = None
                        if order_id == self.ask_order_id:
                            self.ask_order_id = None
            
            return False  # No fills occurred
                    
        except Exception as e:
            logger.error(f"Error checking fills: {e}")
            return False

    async def _cleanup_filled_orders(self, filled_orders, stop_fills):
        """Clean up filled orders and prepare for position-aware order refresh"""
        async with self.trading_lock:
            # 1. Cancel any remaining live orders on exchange
            await self._cancel_orders_unsafe()
            await self._cancel_stop_orders_unsafe()
            
            # 2. Clear memory after cancellations complete
            self.active_orders.clear()
            self.active_stop_orders.clear()
            self.bid_order_id = None
            self.ask_order_id = None
            
            logger.info("🔄 Orders cleaned up after fills - ready for position-aware refresh")
    
    async def _process_fill(self, order_id: str, order_info: Dict, fill_status: Dict):
        """Process a filled order and update profit tracking"""
        try:
            fill_price = fill_status.get('fill_price', order_info['price'])
            fill_size = fill_status.get('filled_size', order_info['size'])
            fee = fill_status.get('fee', 0.0)
            side = order_info['side']
            
            # Update position and PnL atomically
            async with self.trading_lock:
                # Update total fees
                self.total_fees_paid += fee
            
                # Update position tracking with volume-weighted average entry price
                if side == 'buy':
                    # Buying - update average entry price
                    if self.current_position >= 0:  # Adding to long or starting long
                        old_value = self.current_position * self.avg_entry_price
                        new_value = fill_size * fill_price
                        total_position = self.current_position + fill_size
                        self.avg_entry_price = (old_value + new_value) / total_position if total_position > 0 else 0
                    else:  # Covering short position
                        if abs(self.current_position) <= fill_size:  # Closing short position completely
                            # Calculate realized PnL for the closed portion
                            closed_size = abs(self.current_position)
                            pnl = closed_size * (self.avg_entry_price - fill_price)  # Short PnL
                            self.realized_pnl += pnl
                            self.trade_count += 1
                            
                            # Reset for any remaining long position
                            remaining_size = fill_size - closed_size
                            if remaining_size > 0:
                                self.avg_entry_price = fill_price
                            else:
                                self.avg_entry_price = 0
                            
                            logger.info(f"Closed short position: PnL {pnl:.4f}, Remaining size: {remaining_size}")
                        else:  # Partially covering short
                            # No PnL realization, just reduce position
                            pass
                else:  # sell
                    # Selling - opposite logic
                    if self.current_position <= 0:  # Adding to short or starting short
                        old_value = abs(self.current_position) * self.avg_entry_price
                        new_value = fill_size * fill_price
                        total_position = abs(self.current_position) + fill_size
                        self.avg_entry_price = (old_value + new_value) / total_position if total_position > 0 else 0
                    else:  # Covering long position
                        if self.current_position <= fill_size:  # Closing long position completely
                            # Calculate realized PnL for the closed portion
                            closed_size = self.current_position
                            pnl = closed_size * (fill_price - self.avg_entry_price)  # Long PnL
                            self.realized_pnl += pnl
                            self.trade_count += 1
                            
                            # Reset for any remaining short position
                            remaining_size = fill_size - closed_size
                            if remaining_size > 0:
                                self.avg_entry_price = fill_price
                            else:
                                self.avg_entry_price = 0
                            
                            logger.info(f"Closed long position: PnL {pnl:.4f}, Remaining size: {remaining_size}")
                        else:  # Partially covering long
                            # No PnL realization, just reduce position
                            pass
                
                # Update position last (after all calculations)
                if side == 'buy':
                    self.current_position += fill_size
                else:
                    self.current_position -= fill_size
                
                # Log the fill
                logger.info(f"🎯 FILL: {side.upper()} {fill_size:.4f} @ {fill_price:.2f}, "
                           f"Fee: {fee:.4f}, Position: {self.current_position:.4f}, PnL: {self.realized_pnl:.4f}")
                
                # Log trade data
                trade_data = {
                    'timestamp': datetime.now().timestamp(),
                    'order_id': order_id,
                    'asset': self.coin,
                    'side': side,
                    'size': fill_size,
                    'price': fill_price,
                    'position_after': self.current_position,
                    'pnl': self.realized_pnl,
                    'pnl_after': self.realized_pnl - fee  # PnL after fees
                }
                self.trade_logger.log_trade(trade_data)
                
                # Set protective bracket (TP + SL) after successful fill processing
                if self.enable_bracket_protection:
                    await self._set_protective_bracket(fill_price, fill_size, side)
            
        except Exception as e:
            logger.error(f"Error processing fill for order {order_id}: {e}")
    
    async def _set_protective_bracket(self, fill_price: float, fill_size: float, side: str):
        """Set both take-profit and stop-loss bracket orders after fill"""
        try:
            # Calculate TP and SL prices based on configuration
            if side == 'buy':  # Long position
                tp_price = fill_price * (1 + self.profit_target_pct)
                sl_price = fill_price * (1 - self.stop_loss_pct)
                
                # Place sell TP and SL orders to close long position
                tp_order_id = await self.executor.place_take_profit_sell_order(
                    self.coin, fill_size, tp_price
                )
                sl_order_id = await self.executor.place_stop_sell_order(
                    self.coin, fill_size, sl_price
                )
                
            else:  # Short position  
                tp_price = fill_price * (1 - self.profit_target_pct)
                sl_price = fill_price * (1 + self.stop_loss_pct)
                
                # Place buy TP and SL orders to close short position
                tp_order_id = await self.executor.place_take_profit_buy_order(
                    self.coin, fill_size, tp_price
                )
                sl_order_id = await self.executor.place_stop_buy_order(
                    self.coin, fill_size, sl_price
                )
            
            # Track bracket orders if both were placed successfully
            async with self.trading_lock:
                bracket_info = {
                    'original_fill_price': fill_price,
                    'original_fill_size': fill_size,
                    'original_side': side,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'timestamp': time.time()
                }
                
                if tp_order_id:
                    self.active_stop_orders[tp_order_id] = {
                        **bracket_info,
                        'order_type': 'take_profit',
                        'side': 'sell' if side == 'buy' else 'buy',
                        'size': fill_size,
                        'stop_price': tp_price  # For consistent interface
                    }
                
                if sl_order_id:
                    self.active_stop_orders[sl_order_id] = {
                        **bracket_info,
                        'order_type': 'stop_loss', 
                        'side': 'sell' if side == 'buy' else 'buy',
                        'size': fill_size,
                        'stop_price': sl_price
                    }
            
            # Log bracket placement
            if tp_order_id and sl_order_id:
                logger.info(f"🎯🛡️ Set bracket: TP @ {tp_price:.2f} (+{self.profit_target_pct:.1%}) | "
                           f"SL @ {sl_price:.2f} (-{self.stop_loss_pct:.1%}) for {side} fill @ {fill_price:.2f}")
            elif tp_order_id:
                logger.warning(f"🎯 Only TP placed @ {tp_price:.2f} - SL failed")
            elif sl_order_id:
                logger.warning(f"🛡️ Only SL placed @ {sl_price:.2f} - TP failed")
            else:
                logger.error(f"❌ Failed to place bracket orders for {side} fill @ {fill_price:.2f}")
                           
        except Exception as e:
            logger.error(f"Failed to set protective bracket: {e}")
    
    async def _process_bracket_fill(self, order_id: str, stop_info: Dict, fill_status: Dict):
        """Process a filled bracket order (take-profit or stop-loss)"""
        try:
            fill_price = fill_status.get('fill_price', stop_info['stop_price'])
            fill_size = fill_status.get('filled_size', stop_info['size'])
            fee = fill_status.get('fee', 0.0)
            side = stop_info['side']
            
            # Update position and PnL atomically
            async with self.trading_lock:
                # Update total fees
                self.total_fees_paid += fee
                
                # Update position tracking
                if side == 'buy':
                    self.current_position += fill_size
                else:
                    self.current_position -= fill_size
                
                # Log the bracket execution
                order_type = stop_info.get('order_type', 'unknown')
                if order_type == 'take_profit':
                    logger.info(f"🎯 TAKE-PROFIT EXECUTED: {side.upper()} {fill_size:.4f} @ {fill_price:.2f}, "
                               f"Fee: {fee:.4f}, Position: {self.current_position:.4f}, "
                               f"Original entry: {stop_info['original_fill_price']:.2f}")
                elif order_type == 'stop_loss':
                    logger.warning(f"🛡️ STOP-LOSS EXECUTED: {side.upper()} {fill_size:.4f} @ {fill_price:.2f}, "
                                  f"Fee: {fee:.4f}, Position: {self.current_position:.4f}, "
                                  f"Original entry: {stop_info['original_fill_price']:.2f}")
                else:
                    logger.info(f"📊 BRACKET ORDER EXECUTED: {side.upper()} {fill_size:.4f} @ {fill_price:.2f}, "
                               f"Fee: {fee:.4f}, Position: {self.current_position:.4f}")
                
                # Calculate realized PnL from stop-loss execution
                original_side = stop_info['original_side']
                entry_price = stop_info['original_fill_price']
                
                if original_side == 'buy':  # Was long, now stopped out
                    stop_pnl = fill_size * (fill_price - entry_price)
                else:  # Was short, now stopped out
                    stop_pnl = fill_size * (entry_price - fill_price)
                
                self.realized_pnl += stop_pnl
                self.trade_count += 1
                
                # Log trade data for stop-loss
                trade_data = {
                    'timestamp': datetime.now().timestamp(),
                    'order_id': order_id,
                    'asset': self.coin,
                    'side': side,
                    'size': fill_size,
                    'price': fill_price,
                    'position_after': self.current_position,
                    'pnl': self.realized_pnl,
                    'pnl_after': self.realized_pnl - fee  # PnL after fees
                }
                self.trade_logger.log_trade(trade_data)
                
                if order_type == 'take_profit':
                    logger.info(f"🎯 Take-profit PnL: {stop_pnl:.4f}, Total PnL: {self.realized_pnl:.4f}")
                elif order_type == 'stop_loss':
                    logger.info(f"🛡️ Stop-loss PnL: {stop_pnl:.4f}, Total PnL: {self.realized_pnl:.4f}")
                else:
                    logger.info(f"📊 Bracket PnL: {stop_pnl:.4f}, Total PnL: {self.realized_pnl:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing bracket fill for order {order_id}: {e}")

    def calculate_simple_spread_orders(self, current_price: float) -> Tuple[float, float, float, float]:
        """
        Calculate simple spread orders for bracket market making
        Returns: (bid_price, ask_price, bid_size, ask_size)
        No position skewing - brackets handle risk management
        """
        base_spread = current_price * (self.spread_bps / 10000)
        
        # Simple spread calculation - brackets provide risk management
        bid_price = current_price - (base_spread / 2)
        ask_price = current_price + (base_spread / 2)
        
        # Fixed order sizes - brackets provide risk control
        bid_size = self.order_size
        ask_size = self.order_size
        
        return bid_price, ask_price, bid_size, ask_size

    async def cancel_orders(self):
        """Cancel existing orders using executor (with locking)"""
        async with self.trading_lock:
            await self._cancel_orders_unsafe()

    async def get_current_price(self) -> float: 
        try:
            return await self.executor.get_current_price(self.coin)
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
    
    async def _cancel_orders_unsafe(self):
        """Cancel existing orders without additional locking (for use within existing locks)"""
        try:
            if self.bid_order_id:
                await self.executor.cancel_order(self.coin, self.bid_order_id, True)
                if self.bid_order_id in self.active_orders:
                    del self.active_orders[self.bid_order_id]
                self.bid_order_id = None
            
            if self.ask_order_id:
                await self.executor.cancel_order(self.coin, self.ask_order_id, False)
                if self.ask_order_id in self.active_orders:
                    del self.active_orders[self.ask_order_id]
                self.ask_order_id = None
                
        except Exception as e:
            logger.error(f"Error canceling orders: {e}")
    
    async def _cancel_stop_orders_unsafe(self):
        """Cancel existing stop orders without additional locking (for use within existing locks)"""
        try:
            for stop_order_id, stop_info in list(self.active_stop_orders.items()):
                is_buy = stop_info['side'] == 'buy'
                await self.executor.cancel_order(self.coin, stop_order_id, is_buy)
                if stop_order_id in self.active_stop_orders:
                    del self.active_stop_orders[stop_order_id]
                logger.debug(f"Cancelled stop order {stop_order_id}")
                
        except Exception as e:
            logger.error(f"Error canceling stop orders: {e}")

    async def place_position_aware_orders(self, current_position: float, current_price: float):
        """Place orders based on current position - position-aware bracket trading"""
        async with self.trading_lock:
            try:
                # Calculate spread orders
                bid_price, ask_price, bid_size, ask_size = self.calculate_simple_spread_orders(current_price)
                
                # Position-aware order placement logic
                if abs(current_position) < 0.01:  # Essentially flat position
                    # Market making mode - place both bid and ask
                    await self._place_both_orders(current_price, bid_price, ask_price, bid_size, ask_size)
                    logger.info(f"🎯 Market making mode - Position: {current_position:.4f} (flat)")
                    
                elif current_position > 0.01:  # Long position
                    # Only place sell orders to reduce exposure
                    await self._place_sell_order_only(current_price, ask_price, ask_size)
                    logger.info(f"📉 Long position mode - Only placing sell orders. Position: {current_position:.4f}")
                    
                elif current_position < -0.01:  # Short position  
                    # Only place buy orders to reduce exposure
                    await self._place_buy_order_only(current_price, bid_price, bid_size)
                    logger.info(f"📈 Short position mode - Only placing buy orders. Position: {current_position:.4f}")
                
                self.last_order_refresh = time.time()
                
            except Exception as e:
                logger.error(f"Error placing position-aware orders: {e}")
    
    async def _place_both_orders(self, current_price: float, bid_price: float, ask_price: float, bid_size: float, ask_size: float):
        """Place both bid and ask orders (market making mode)"""
        # Place bid order
        if bid_price is not None and bid_size > 0:
            self.bid_order_id = await self.executor.place_buy_order(self.coin, bid_size, bid_price)
            if self.bid_order_id:
                self.active_orders[self.bid_order_id] = {
                    'side': 'buy', 'size': bid_size, 'price': bid_price, 'timestamp': time.time()
                }
                bid_bps_from_mid = ((current_price - bid_price) / current_price) * 10000
                logger.info(f"📈 BUY Order: {bid_size:.4f} @ {bid_price:.2f} (-{bid_bps_from_mid:.1f} bps)")
        
        # Place ask order
        if ask_price is not None and ask_size > 0:
            self.ask_order_id = await self.executor.place_sell_order(self.coin, ask_size, ask_price)
            if self.ask_order_id:
                self.active_orders[self.ask_order_id] = {
                    'side': 'sell', 'size': ask_size, 'price': ask_price, 'timestamp': time.time()
                }
                ask_bps_from_mid = ((ask_price - current_price) / current_price) * 10000
                logger.info(f"📉 SELL Order: {ask_size:.4f} @ {ask_price:.2f} (+{ask_bps_from_mid:.1f} bps)")
    
    async def _place_buy_order_only(self, current_price: float, bid_price: float, bid_size: float):
        """Place only buy orders (to reduce short position)"""
        if bid_price is not None and bid_size > 0:
            self.bid_order_id = await self.executor.place_buy_order(self.coin, bid_size, bid_price)
            if self.bid_order_id:
                self.active_orders[self.bid_order_id] = {
                    'side': 'buy', 'size': bid_size, 'price': bid_price, 'timestamp': time.time()
                }
                bid_bps_from_mid = ((current_price - bid_price) / current_price) * 10000
                logger.info(f"📈 BUY Order (reduce short): {bid_size:.4f} @ {bid_price:.2f} (-{bid_bps_from_mid:.1f} bps)")
    
    async def _place_sell_order_only(self, current_price: float, ask_price: float, ask_size: float):
        """Place only sell orders (to reduce long position)"""
        if ask_price is not None and ask_size > 0:
            self.ask_order_id = await self.executor.place_sell_order(self.coin, ask_size, ask_price)
            if self.ask_order_id:
                self.active_orders[self.ask_order_id] = {
                    'side': 'sell', 'size': ask_size, 'price': ask_price, 'timestamp': time.time()
                }
                ask_bps_from_mid = ((ask_price - current_price) / current_price) * 10000
                logger.info(f"📉 SELL Order (reduce long): {ask_size:.4f} @ {ask_price:.2f} (+{ask_bps_from_mid:.1f} bps)")

    def calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL based on current position and price"""
        if self.current_position == 0 or self.avg_entry_price == 0:
            return 0.0
        
        if self.current_position > 0:  # Long position
            return self.current_position * (self.current_price - self.avg_entry_price)
        else:  # Short position
            return abs(self.current_position) * (self.avg_entry_price - self.current_price)

    async def trading_loop(self):
        """Single efficient trading loop with position-aware order management"""
        logger.info(f"Started position-aware trading loop (interval: {self.trading_interval}s)")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while self.running and not self.shutdown_requested:
                try:
                    # 1. Check for shutdown signal
                    if self.shutdown_requested:
                        logger.info("🛑 Shutdown signal detected in trading loop")
                        break
                    
                    # 2. Get current market price
                    current_price = await self.get_current_price()
                    if current_price == 0:
                        logger.warning("Invalid price, skipping cycle")
                        await asyncio.sleep(self.trading_interval)
                        continue
                    
                    # 4. Check for fills and get updated position
                    fills_occurred = await self.check_fills()
                    await self.update_position()
                    
                    # 5. Update PnL calculations
                    async with self.trading_lock:
                        self.unrealized_pnl = self.calculate_unrealized_pnl()
                        current_position = self.current_position
                        
                    # 6. Position-aware order placement
                    should_refresh = (fills_occurred or 
                                    self.needs_order_refresh or
                                    self._should_refresh_orders())
                    
                    if should_refresh and not self.shutdown_requested:
                        await self.place_position_aware_orders(current_position, current_price)
                        self.needs_order_refresh = False
                        
                    # 7. Logging and status
                    if fills_occurred:
                        logger.info("🚀 Fills detected - orders refreshed")
                    elif self.log_fill_checks:
                        logger.debug(f"Trading cycle complete - Price: {current_price:.2f}, Position: {current_position:.4f}")
                    
                    # 8. Reset error counter and wait (with shutdown check)
                    consecutive_errors = 0
                    
                    # Check for shutdown during sleep to be more responsive
                    sleep_time = self.trading_interval
                    sleep_increment = 0.1  # Check every 100ms
                    slept = 0.0
                    
                    while slept < sleep_time and not self.shutdown_requested:
                        await asyncio.sleep(min(sleep_increment, sleep_time - slept))
                        slept += sleep_increment
                    
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error in trading loop iteration #{consecutive_errors}: {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Trading loop: {consecutive_errors} consecutive errors - terminating")
                        raise e
                    
                    # Cancel problematic orders before retry
                    try:
                        await self.cancel_orders()
                        logger.info("Cancelled orders during error recovery")
                    except Exception as cancel_error:
                        logger.error(f"Failed to cancel orders during error recovery: {cancel_error}")
                    
                    # Exponential backoff on errors
                    error_delay = min(self.trading_interval * (2 ** consecutive_errors), 30.0)
                    logger.warning(f"Trading loop error recovery: waiting {error_delay:.1f}s before retry")
                    await asyncio.sleep(error_delay)
                
        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}")
            raise
    
    def _should_refresh_orders(self) -> bool:
        """Check if orders should be refreshed based on age or market conditions"""
        current_time = time.time()
        
        # Refresh orders based on configurable interval
        if current_time - self.last_order_refresh > self.order_refresh_interval:
            return True
            
        # Check if orders are very old
        trading_config = self.config['trading']
        max_order_age = trading_config.get('max_order_age', 150)
        
        if ((self.bid_order_id and self.bid_order_id in self.active_orders and
             current_time - self.active_orders[self.bid_order_id]['timestamp'] > max_order_age) or
            (self.ask_order_id and self.ask_order_id in self.active_orders and
             current_time - self.active_orders[self.ask_order_id]['timestamp'] > max_order_age)):
            return True
            
        return False


    async def run(self):
        """Run the bracket market maker with single position-aware trading loop"""
        logger.info("🚀 Starting bracket market maker with position-aware architecture...")
        logger.info(f"Trading interval: {self.trading_interval}s")
        logger.info("Position-aware order management: Only place orders that reduce risk")

        self.shutdown_requested = False
        
        try:
            # Start single efficient trading loop
            await self.trading_loop()
                
        except KeyboardInterrupt:
            # This should rarely happen now due to signal handling
            logger.info("KeyboardInterrupt caught - initiating intelligent shutdown")
            self.shutdown_requested = True
            self.shutdown_reason = "KeyboardInterrupt exception"
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            self.shutdown_requested = True
            self.shutdown_reason = f"Trading loop exception: {e}"
        finally:
            if self.shutdown_requested:
                logger.info(f"🛱 Initiating intelligent shutdown - Reason: {self.shutdown_reason}")
            else:
                logger.info("🛱 Initiating intelligent shutdown - Normal termination")
            
            self.running = False
            
            # Intelligent shutdown with position management
            await self._intelligent_shutdown()
            
            logger.info("📊 Bracket market maker stopped")

    async def _intelligent_shutdown(self):
        """Intelligent shutdown with position closure and comprehensive risk analysis"""
        try:
            # 1. Get current market price for calculations
            current_price = await self.get_current_price()
            
            # 2. Update final position and calculate PnL
            await self.update_position()
            
            async with self.trading_lock:
                final_position = self.current_position
                self.unrealized_pnl = self.calculate_unrealized_pnl()
                total_pnl = self.realized_pnl + self.unrealized_pnl
                
                logger.info("=" * 60)
                logger.info("🔍 INTELLIGENT SHUTDOWN ANALYSIS")
                logger.info("=" * 60)
                
                # 3. Current Status Report
                logger.info(f"📊 Current Status:")
                logger.info(f"   Price: {current_price:.4f}")
                logger.info(f"   Position: {final_position:.4f}")
                logger.info(f"   Realized PnL: {self.realized_pnl:.4f}")
                logger.info(f"   Unrealized PnL: {self.unrealized_pnl:.4f}")
                logger.info(f"   Total PnL: {total_pnl:.4f}")
                logger.info(f"   Trades: {self.trade_count}")
                logger.info(f"   Fees Paid: {self.total_fees_paid:.4f}")
                
                # 4. Risk Analysis if position is open
                if abs(final_position) > 0.01:
                    await self._analyze_position_risk(final_position, current_price)
                    
                    # 5. Attempt to close position intelligently
                    closure_success = await self._attempt_position_closure(final_position, current_price)
                    
                    if closure_success:
                        logger.info("✅ Position successfully closed")
                    else:
                        logger.warning("⚠️ Position remains open - monitor manually!")
                        await self._log_manual_closure_instructions(final_position, current_price)
                else:
                    logger.info("✅ No open position - clean shutdown")
                
                # 6. Cancel all remaining orders
                await self.cancel_orders()
                await self._cancel_stop_orders_unsafe()
                
                # 7. Final session summary
                await self._log_final_session_summary(total_pnl)
                
        except Exception as e:
            logger.error(f"Error during intelligent shutdown: {e}")
            # Fallback to basic shutdown
            await self.cancel_orders()
            async with self.trading_lock:
                await self._cancel_stop_orders_unsafe()
    
    async def _analyze_position_risk(self, position: float, current_price: float):
        """Analyze maximum risk and potential outcomes for open position"""
        logger.info("🎯 POSITION RISK ANALYSIS:")
        
        if position > 0:  # Long position
            # Calculate distances to TP and SL based on average entry price
            if self.avg_entry_price > 0:
                entry_price = self.avg_entry_price
                tp_price = entry_price * (1 + self.profit_target_pct)
                sl_price = entry_price * (1 - self.stop_loss_pct)
                
                max_gain = position * (tp_price - entry_price)
                max_loss = position * (entry_price - sl_price)
                current_unrealized = position * (current_price - entry_price)
                
                logger.info(f"   📈 Long Position Analysis:")
                logger.info(f"   Entry Price: {entry_price:.4f}")
                logger.info(f"   Take Profit: {tp_price:.4f} (max gain: +{max_gain:.4f})")
                logger.info(f"   Stop Loss: {sl_price:.4f} (max loss: -{max_loss:.4f})")
                logger.info(f"   Current P&L: {current_unrealized:.4f}")
                
                # Distance analysis
                tp_distance = abs(current_price - tp_price) / current_price * 10000
                sl_distance = abs(current_price - sl_price) / current_price * 10000
                logger.info(f"   Distance to TP: {tp_distance:.1f} bps")
                logger.info(f"   Distance to SL: {sl_distance:.1f} bps")
                
        else:  # Short position
            if self.avg_entry_price > 0:
                entry_price = self.avg_entry_price
                tp_price = entry_price * (1 - self.profit_target_pct)
                sl_price = entry_price * (1 + self.stop_loss_pct)
                
                max_gain = abs(position) * (entry_price - tp_price)
                max_loss = abs(position) * (sl_price - entry_price)
                current_unrealized = abs(position) * (entry_price - current_price)
                
                logger.info(f"   📉 Short Position Analysis:")
                logger.info(f"   Entry Price: {entry_price:.4f}")
                logger.info(f"   Take Profit: {tp_price:.4f} (max gain: +{max_gain:.4f})")
                logger.info(f"   Stop Loss: {sl_price:.4f} (max loss: -{max_loss:.4f})")
                logger.info(f"   Current P&L: {current_unrealized:.4f}")
                
                # Distance analysis
                tp_distance = abs(current_price - tp_price) / current_price * 10000
                sl_distance = abs(current_price - sl_price) / current_price * 10000
                logger.info(f"   Distance to TP: {tp_distance:.1f} bps")
                logger.info(f"   Distance to SL: {sl_distance:.1f} bps")
    
    async def _attempt_position_closure(self, position: float, current_price: float) -> bool:
        """Attempt to close position with market order"""
        try:
            logger.info("🎯 ATTEMPTING POSITION CLOSURE:")
            
            if abs(position) < 0.01:
                return True  # Already flat
            
            if position > 0:  # Close long position
                logger.info(f"   Placing market sell order for {position:.4f} to close long position")
                
                # For live trading, place market sell order
                order_id = await self.executor.place_market_sell_order(self.coin, position)
                if order_id:
                    logger.info(f"   Market sell order placed: {order_id}")
                    return True
                else:
                    logger.error("   Failed to place market sell order")
                    return False
                        
            else:  # Close short position
                position_size = abs(position)
                logger.info(f"   Placing market buy order for {position_size:.4f} to close short position")
                
                # For live trading, place market buy order
                order_id = await self.executor.place_market_buy_order(self.coin, position_size)
                if order_id:
                    logger.info(f"   Market buy order placed: {order_id}")
                    return True
                else:
                    logger.error("   Failed to place market buy order")
                    return False
                        
        except Exception as e:
            logger.error(f"Error attempting position closure: {e}")
            return False
    
    async def _log_manual_closure_instructions(self, position: float, current_price: float):
        """Log instructions for manual position closure"""
        logger.info("📋 MANUAL CLOSURE INSTRUCTIONS:")
        
        if position > 0:
            logger.info(f"   🔴 LONG POSITION OPEN: {position:.4f}")
            logger.info(f"   📝 Manual Action: SELL {position:.4f} {self.coin} at market")
            logger.info(f"   💰 Current Value: {position * current_price:.4f}")
        else:
            position_size = abs(position)
            logger.info(f"   🔴 SHORT POSITION OPEN: {position_size:.4f}")
            logger.info(f"   📝 Manual Action: BUY {position_size:.4f} {self.coin} at market")
            logger.info(f"   💰 Current Cost: {position_size * current_price:.4f}")
        
        logger.info(f"   ⚠️ MONITOR: Stop-loss and take-profit orders may still be active")
        logger.info(f"   📊 Current Price: {current_price:.4f}")
    
    async def _log_final_session_summary(self, total_pnl: float):
        """Log comprehensive final session summary"""
        logger.info("=" * 60)
        logger.info("📋 FINAL SESSION SUMMARY")
        logger.info("=" * 60)
        
        # Performance metrics
        win_rate = (self.trade_count - sum(1 for t in [] if t < 0)) / max(self.trade_count, 1) * 100 if self.trade_count > 0 else 0
        avg_trade = self.realized_pnl / max(self.trade_count, 1) if self.trade_count > 0 else 0
        
        logger.info(f"🎯 Performance:")
        logger.info(f"   Total PnL: {total_pnl:.4f}")
        logger.info(f"   Realized PnL: {self.realized_pnl:.4f}")
        logger.info(f"   Unrealized PnL: {self.unrealized_pnl:.4f}")
        logger.info(f"   Total Trades: {self.trade_count}")
        logger.info(f"   Average Trade: {avg_trade:.4f}")
        logger.info(f"   Total Fees: {self.total_fees_paid:.4f}")
        logger.info(f"   Net PnL (after fees): {total_pnl - self.total_fees_paid:.4f}")
        
        # Update trade logger with final session data
        self.trade_logger.update_metadata('end_time', datetime.now().isoformat())
        self.trade_logger.update_metadata('final_pnl', total_pnl)
        self.trade_logger.update_metadata('final_position', self.current_position)
        self.trade_logger.update_metadata('total_trades', self.trade_count)
        
        logger.info(f"💾 Session data saved to: {self.trade_logger.session_dir}")
        logger.info("=" * 60)
    
    def stop(self):
        """Stop the market maker"""
        self.running = False


async def main():
    """Main entry point with comprehensive signal handling"""
    
    logger.info("🚀 Starting Bracket Market Maker...")
    
    market_maker = BracketMarketMaker()
    #await market_maker.run()
    
    try:
        market_maker = BracketMarketMaker()
        # Run the market maker (signal handling is done in the class)
        await market_maker.run()
        
    except KeyboardInterrupt:
        # This should rarely happen due to signal handling in the class
        logger.info("🛑 Main: KeyboardInterrupt caught")
        if market_maker:
            market_maker.shutdown_requested = True
            market_maker.shutdown_reason = "Main KeyboardInterrupt"
    except Exception as e:
        logger.error(f"❌ Failed to run market maker: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        if market_maker:
            market_maker.shutdown_requested = True
            market_maker.shutdown_reason = f"Main exception: {e}"
    finally:
        if market_maker:
            # Ensure the market maker is properly stopped
            if market_maker.running or not market_maker.shutdown_requested:
                logger.info("🔧 Ensuring proper shutdown in main cleanup")
                market_maker.stop()
        
        logger.info("🏁 Main function cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())