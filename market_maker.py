import logging
import time
import yaml
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import os
import asyncio
import threading
from queue import Queue
import signal

import torch
import numpy as np
from hyperliquid.utils import constants
from utils.data_fetcher import DataFetcher
from utils.utils import setup
from utils.trade_logger import TradeLogger
from utils.tui import TradingTUI
from utils.analytics import TradingAnalytics
from models.spread_predictor import SpreadPredictor, SpreadDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OrderBook:
    def __init__(self):
        self.bids: List[Dict] = []  # List of bid orders
        self.asks: List[Dict] = []  # List of ask orders
        self.last_price: float = 0.0
        self.last_update: datetime = datetime.now()
        self.base_spread: float = 0.001  # Base spread of 0.1%
        self.inventory_skew: float = 0.0  # Skew factor based on inventory

    def update(self, mid_price: float, position: float, max_position: float):
        """Update the order book with new mid price and position-based adjustments."""
        self.last_price = mid_price
        self.last_update = datetime.now()
        
        # Calculate inventory skew factor (-1 to 1)
        self.inventory_skew = position / max_position
        
        # Calculate dynamic spread based on inventory
        # Wider spread when inventory is skewed
        spread_multiplier = 1.0 + abs(self.inventory_skew)
        current_spread = self.base_spread * spread_multiplier
        
        # Calculate bid and ask prices with inventory skew
        # When long, we want to sell more aggressively (lower ask)
        # When short, we want to buy more aggressively (higher bid)
        bid_skew = -self.inventory_skew * current_spread * 0.5
        ask_skew = self.inventory_skew * current_spread * 0.5
        
        bid_price = mid_price - (current_spread / 2) + bid_skew
        ask_price = mid_price + (current_spread / 2) + ask_skew
        
        # Clear existing orders
        self.bids = []
        self.asks = []
        
        # Add new orders with size based on inventory
        # When long, reduce bid size and increase ask size
        # When short, increase bid size and reduce ask size
        bid_size = 1.0 * (1 - self.inventory_skew)
        ask_size = 1.0 * (1 + self.inventory_skew)
        
        self.bids.append({
            'price': bid_price,
            'size': max(0.1, bid_size),  # Minimum size of 0.1
            'timestamp': self.last_update
        })
        
        self.asks.append({
            'price': ask_price,
            'size': max(0.1, ask_size),  # Minimum size of 0.1
            'timestamp': self.last_update
        })

    def get_best_bid(self) -> Dict:
        return self.bids[0] if self.bids else None

    def get_best_ask(self) -> Dict:
        return self.asks[0] if self.asks else None


class MarketMaker:
    def __init__(self, config_path: str = "config.yaml", session_id: str = None):
        """Initialize the market maker with configuration"""
        self.config = self._load_config(config_path)
        self.trading_params = self.config.get('trading', {})
        
        # Initialize HyperLiquid client
        self.account_address, self.info, self.exchange = setup(
            base_url=constants.MAINNET_API_URL,
            skip_ws=True
        )
        self.coin = self.config.get('universal', {}).get('asset', 'BTC')
        
        # Initialize trading parameters from config with defaults
        self.max_position_size = self.trading_params.get('max_position_size', 1.0)
        self.leverage = self.trading_params.get('leverage', 1.0)
        self.profit_target = self.trading_params.get('profit_target', 0.002)  # 0.2% default
        self.stop_loss = self.trading_params.get('stop_loss', 0.001)  # 0.1% default
        
        # Initialize volatility parameters
        self.volatility_window = self.trading_params.get('volatility_window', 20)
        self.min_volatility_threshold = self.trading_params.get('min_volatility_threshold', 0.0001)
        self.max_volatility_threshold = self.trading_params.get('max_volatility_threshold', 0.01)
        self.volatility_scaling_threshold = self.trading_params.get('volatility_scaling_threshold', 0.005)
        self.min_volatility_factor = self.trading_params.get('min_volatility_factor', 0.5)
        self.max_volatility_factor = self.trading_params.get('max_volatility_factor', 1.0)
        
        # Set leverage at the start of the session
        try:
            self.exchange.update_leverage(self.leverage, self.coin)
            logger.info(f"Set leverage to {self.leverage}x for {self.coin}")
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
        
        # Get session ID and directory from environment or use default
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.getenv("SESSION_DIR", os.path.join("logs", "sessions", "default_session"))
        os.makedirs(self.session_dir, exist_ok=True)
        print(f"[MarketMaker] Using session directory: {self.session_dir}")
        
        # Set session directory in environment for TradeLogger
        os.environ["SESSION_DIR"] = self.session_dir
        
        # Initialize components
        self.trade_logger = TradeLogger(session_id=self.session_id, config=self.config)
        
        # Initialize state
        self.current_data = []
        self.current_price = 0.0
        self.current_position = 0.0
        self.current_pnl = 0.0
        self.last_trade_time = 0
        self.trade_count = 0
        self.total_pnl = 0.0
        self.is_running = True
        
        # Initialize performance metrics
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_trade_pnl': 0.0
        }
        
        # Initialize trade frequency tracking
        self.minute_start = datetime.now()
        self.trades_per_minute = 0
        
        # Initialize ML model and data processor
        self.model = SpreadPredictor(input_size=4, hidden_size=64)
        self.data_processor = SpreadDataProcessor(sequence_length=10)
        
        # Test log entry to verify file creation
        self.trade_logger.log_trade({
            'timestamp': int(time.time()),
            'order_id': 'test',
            'asset': 'ETH',
            'side': 'test',
            'size': 0,
            'price': 0,
            'position_after': 0,
            'pnl': 0,
            'pnl_after': 0
        })
        
        logger.info("Market Maker initialized with configuration:")
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Session Directory: {self.session_dir}")
        logger.info(f"Coin: {self.coin}")
        logger.info(f"Max Position Size: {self.max_position_size}")
        logger.info(f"Leverage: {self.leverage}")
        logger.info(f"Profit Target: {self.profit_target}")
        logger.info(f"Stop Loss: {self.stop_loss}")
        logger.info(f"Volatility Window: {self.volatility_window}")
        logger.info(f"Volatility Thresholds: {self.min_volatility_threshold} - {self.max_volatility_threshold}")

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

    def update_performance_metrics(self, trade_pnl: float):
        """Update performance metrics after each trade."""
        self.performance_metrics['total_trades'] += 1
        if trade_pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
        
        # Update average trade PnL
        total_pnl = self.performance_metrics['avg_trade_pnl'] * (self.performance_metrics['total_trades'] - 1)
        self.performance_metrics['avg_trade_pnl'] = (total_pnl + trade_pnl) / self.performance_metrics['total_trades']
        # logger.info(f"Trade PnL: {trade_pnl:.6f}, Cumulative PnL: {self.pnl:.6f}")

    def update_trade_frequency(self):
        """Update trade frequency metrics."""
        current_time = datetime.now()
        time_diff = (current_time - self.minute_start).total_seconds()
        
        if time_diff >= 60:  # New minute
            self.trades_per_minute = self.trade_count
            logger.info(f"Trading frequency: {self.trades_per_minute} trades per minute")
            self.trade_count = 0
            self.minute_start = current_time

    def round_price(self, price: float) -> float:
        """
        Round the price according to Hyperliquid's specifications:
        - Up to 5 significant figures
        - No more than MAX_DECIMALS - szDecimals decimal places
        - Integer prices always allowed
        """
        # If price is an integer, return as is
        if price.is_integer():
            return int(price)
            
        # For prices >= 1000, round to nearest integer
        if price >= 1000:
            return round(price)
            
        # For prices < 1000, round to 5 significant figures
        # First convert to scientific notation to count significant figures
        str_price = f"{price:.10f}"
        # Remove trailing zeros
        str_price = str_price.rstrip('0').rstrip('.')
        
        # If we have more than 5 significant figures, round appropriately
        if len(str_price.replace('.', '')) > 5:
            # For prices < 1, we need to be more careful with decimal places
            if price < 1:
                # Count leading zeros after decimal point
                decimal_part = str_price.split('.')[1]
                leading_zeros = len(decimal_part) - len(decimal_part.lstrip('0'))
                # Round to appropriate decimal places
                return round(price, leading_zeros + 5)
            else:
                # For prices >= 1, round to 5 significant figures
                return round(price, 5 - len(str(int(price))))
        return float(str_price)

    async def execute_trade(self, side, size, price=None):
        """Execute a trade with the given parameters."""
        try:
            is_buy = side if isinstance(side, bool) else side.lower() == 'buy'
            
            # Log position before trade
            position_data = {
                'timestamp': datetime.now().timestamp(),
                'asset': self.coin,
                'size': self.current_position,
                'value': self.current_position * (price if price else self.current_price),
                'entry_price': self.current_price
            }
            self.trade_logger.log_position(position_data)
            
            # Place order using self.exchange
            if price:
                # Limit order
                rounded_price = self.round_price(price)
                order_result = self.exchange.order(
                    self.coin,
                    is_buy,
                    size,
                    rounded_price,
                    {
                        "limit": {
                            "tif": "Gtc"
                        },
                        "leverage": self.leverage
                    }
                )
                order_price = rounded_price
            else:
                # Market order
                order_result = self.exchange.market_open(
                    self.coin,
                    is_buy,
                    size,
                    None,  # No price for market orders
                    0.01   # Slippage tolerance
                )
                order_price = 'MARKET'
            
            # Update position
            if is_buy:
                self.current_position += size
            else:
                self.current_position -= size
            
            # Calculate trade PnL (only for limit orders with known price)
            if price:
                trade_pnl = size * (rounded_price - self.current_price) * (1 if is_buy else -1)
                self.current_pnl += trade_pnl
            else:
                trade_pnl = 0  # Market orders will update PnL on next price update
            
            # Log the trade
            trade_data = {
                'timestamp': datetime.now().timestamp(),
                'order_id': order_result.get('orderId'),
                'asset': self.coin,
                'side': 'buy' if is_buy else 'sell',
                'size': size,
                'price': order_price,
                'position_after': self.current_position,
                'pnl': trade_pnl,
                'pnl_after': self.current_pnl
            }
            self.trade_logger.log_trade(trade_data)
            
            # Log updated position
            position_data = {
                'timestamp': datetime.now().timestamp(),
                'asset': self.coin,
                'size': self.current_position,
                'value': self.current_position * (price if price else self.current_price),
                'entry_price': self.current_price
            }
            self.trade_logger.log_position(position_data)
            
            return {
                'side': side,
                'size': size,
                'price': order_price,
                'order_id': order_result.get('orderId')
            }
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            raise

    def train_model(self, candle_data: List[Dict], epochs: int = 10):
        """Train the model on historical data."""
        logger.info("Training model on historical data...")
        
        # Prepare training data
        X, y = self.data_processor.prepare_data(candle_data)
        if len(X) == 0:
            logger.error("No training data available!")
            return False
        
        # Convert to numpy arrays first, then to tensors
        X_np = np.array(X)
        y_np = np.array(y)
        X_tensor = torch.FloatTensor(X_np)
        y_tensor = torch.FloatTensor(y_np)
        
        # Use the proper training function
        from models.spread_predictor import train_model as train_model_fn
        losses = train_model_fn(
            model=self.model,
            X=X_tensor,
            y=y_tensor,
            epochs=epochs,
            batch_size=32,
            learning_rate=0.001
        )
        
        logger.info("Model training completed!")
        return True

    def calculate_position_size(self, current_price: float, spread: float) -> float:
        """
        Calculate appropriate position size based on current conditions.
        Leverage is applied to max_position_size to determine actual position capacity.
        """
        # Base size on current position and max position (adjusted for leverage)
        leveraged_max_size = self.max_position_size * self.leverage
        remaining_capacity = leveraged_max_size - abs(self.current_position)
        
        # Calculate volatility from recent candles
        recent_candles = self.current_data[-5:]  # Last 5 candles
        if len(recent_candles) >= 2:
            price_changes = []
            for i in range(1, len(recent_candles)):
                prev_close = (float(recent_candles[i-1]['h']) + float(recent_candles[i-1]['l'])) / 2
                curr_close = (float(recent_candles[i]['h']) + float(recent_candles[i]['l'])) / 2
                price_changes.append(abs(curr_close - prev_close) / prev_close)
            volatility = sum(price_changes) / len(price_changes)
        else:
            volatility = 0.001  # Default to 0.1% if not enough data
        
        # Adjust size based on spread width and volatility
        spread_factor = min(1.0, spread / (self.min_spread_threshold * 2))
        volatility_factor = max(0.2, min(1.0, 0.001 / (volatility + 1e-10)))  # Inverse relationship with volatility
        
        # Calculate base size with dynamic adjustment
        base_size = self.max_position_size * spread_factor * volatility_factor
        
        # Ensure we don't exceed remaining capacity
        return min(base_size, remaining_capacity)

    def should_enter_trade(self, current_price: float, volatility: float) -> Tuple[bool, str, float]:
        """
        Determine if we should enter a trade based on market conditions
        Returns: (should_trade, side, size)
        """
        # Calculate position size based on volatility
        size = self._calculate_position_size()
        
        # If we have no position, look for entry opportunities
        if self.current_position == 0:
            # Simple market making strategy: alternate between buy and sell
            if self.trade_count % 2 == 0:
                return True, "buy", size
            else:
                return True, "sell", size
        else:
            # If we have a position, look for exit opportunities using ML-predicted TP/SL
            X, _ = self.data_processor.prepare_data(self.current_data[-self.data_processor.sequence_length:])
            take_profit_spread, stop_loss_spread = self._calculate_take_profit_stop_loss(X)
            
            if self.current_position > 0:  # Long position
                if (current_price > self.current_price * (1 + take_profit_spread)):
                    logger.info(f"Exit LONG: TP hit | Entry: {self.current_price:.2f}, Current: {current_price:.2f}, TP: {take_profit_spread:.6f}")
                    return True, "sell", abs(self.current_position)
                elif (current_price < self.current_price * (1 - stop_loss_spread)):
                    logger.info(f"Exit LONG: SL hit | Entry: {self.current_price:.2f}, Current: {current_price:.2f}, SL: {stop_loss_spread:.6f}")
                    return True, "sell", abs(self.current_position)
            else:  # Short position
                if (current_price < self.current_price * (1 - take_profit_spread)):
                    logger.info(f"Exit SHORT: TP hit | Entry: {self.current_price:.2f}, Current: {current_price:.2f}, TP: {take_profit_spread:.6f}")
                    return True, "buy", abs(self.current_position)
                elif (current_price > self.current_price * (1 + stop_loss_spread)):
                    logger.info(f"Exit SHORT: SL hit | Entry: {self.current_price:.2f}, Current: {current_price:.2f}, SL: {stop_loss_spread:.6f}")
                    return True, "buy", abs(self.current_position)
        return False, '', 0.0

    def should_exit_position(self, current_price: float) -> Tuple[bool, str, float]:
        """
        Determine if we should exit an existing position.
        Returns: (should_exit, side, size)
        """
        if self.current_position == 0:
            return False, '', 0.0
            
        # Check for stop loss
        if self.current_position > 0:  # Long position
            if current_price <= self.current_price * (1 - self.stop_loss):
                return True, 'sell', abs(self.current_position)
        else:  # Short position
            if current_price >= self.current_price * (1 + self.stop_loss):
                return True, 'buy', abs(self.current_position)
                
        return False, '', 0.0

    async def cancel_all_orders(self):
        """Cancel all active orders"""
        try:
            # Get open orders using info client with account address
            open_orders = self.info.open_orders(self.account_address)
            if not open_orders:
                logger.info("No open orders to cancel")
                return
                
            for order in open_orders:
                try:
                    # Convert order ID to integer
                    order_id = int(order['oid'])
                    # Cancel order synchronously
                    result = self.exchange.cancel(self.coin, order_id)
                    logger.info(f"Canceled order {order_id}: {result}")
                except (ValueError, KeyError) as e:
                    logger.warning(f"Invalid order ID format: {e}")
                except Exception as e:
                    logger.warning(f"Error canceling order {order.get('oid')}: {e}")
        except Exception as e:
            logger.error(f"Error in cancel_all_orders: {e}")
            # Log the full error details for debugging
            logger.error(f"Error details: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response: {e.response.text if hasattr(e.response, 'text') else e.response}")

    async def set_exit_orders(self):
        """Set stop loss and take profit orders for current position"""
        if self.current_position == 0:
            return
            
        try:
            # Use a conservative exit percentage for both TP and SL
            exit_percentage = self.stop_loss  # Use stop loss percentage for both TP and SL
            
            # Calculate exit prices based on current position
            if self.current_position > 0:  # Long position
                stop_loss_price = self.round_price(self.current_price * (1 - exit_percentage))
                take_profit_price = self.round_price(self.current_price * (1 + exit_percentage))  # Same percentage as SL
                
                # Set take profit order
                tp_order_type = {
                    "trigger": {
                        "triggerPx": take_profit_price,
                        "isMarket": True,
                        "tpsl": "tp"
                    }
                }
                tp_result = self.exchange.order(
                    self.coin,
                    False,  # sell for long position
                    abs(self.current_position),
                    take_profit_price,
                    tp_order_type,
                    reduce_only=True
                )
                logger.info(f"Set take profit order at {take_profit_price}: {tp_result}")
                
                # Set stop loss order
                sl_order_type = {
                    "trigger": {
                        "triggerPx": stop_loss_price,
                        "isMarket": True,
                        "tpsl": "sl"
                    }
                }
                sl_result = self.exchange.order(
                    self.coin,
                    False,  # sell for long position
                    abs(self.current_position),
                    stop_loss_price,
                    sl_order_type,
                    reduce_only=True
                )
                logger.info(f"Set stop loss order at {stop_loss_price}: {sl_result}")
                
            else:  # Short position
                stop_loss_price = self.round_price(self.current_price * (1 + exit_percentage))
                take_profit_price = self.round_price(self.current_price * (1 - exit_percentage))  # Same percentage as SL
                
                # Set take profit order
                tp_order_type = {
                    "trigger": {
                        "triggerPx": take_profit_price,
                        "isMarket": True,
                        "tpsl": "tp"
                    }
                }
                tp_result = self.exchange.order(
                    self.coin,
                    True,  # buy for short position
                    abs(self.current_position),
                    take_profit_price,
                    tp_order_type,
                    reduce_only=True
                )
                logger.info(f"Set take profit order at {take_profit_price}: {tp_result}")
                
                # Set stop loss order
                sl_order_type = {
                    "trigger": {
                        "triggerPx": stop_loss_price,
                        "isMarket": True,
                        "tpsl": "sl"
                    }
                }
                sl_result = self.exchange.order(
                    self.coin,
                    True,  # buy for short position
                    abs(self.current_position),
                    stop_loss_price,
                    sl_order_type,
                    reduce_only=True
                )
                logger.info(f"Set stop loss order at {stop_loss_price}: {sl_result}")
                
        except Exception as e:
            logger.error(f"Error setting exit orders: {e}")
            # Log the full error details for debugging
            logger.error(f"Error details: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response: {e.response.text if hasattr(e.response, 'text') else e.response}")

    async def run_live_trading(self, update_interval: float = 1.0):
        """
        Run live trading using Hyperliquid API.
        
        Args:
            update_interval: Time between updates in seconds (default: 1.0s for 1-second updates)
        """
        data_fetcher = DataFetcher()
        logger.info("Starting live trading...")
        
        try:
            # First, fetch initial data and train the model
            logger.info("Fetching initial data for training...")
            initial_data = data_fetcher.fetch_price_data()
            if not initial_data:
                logger.error("Failed to fetch initial data for training!")
                return
            
            logger.info(f"Successfully fetched {len(initial_data)} candles for training")
            if not self.train_model(initial_data):
                logger.error("Failed to train model!")
                return
            
            logger.info("Starting live trading loop...")
            
            # Initialize current_data with the last sequence_length candles
            self.current_data = initial_data[-self.data_processor.sequence_length:]
            logger.info(f"Initialized with {len(self.current_data)} candles for sequence")
            self.current_price = (float(self.current_data[-1]['h']) + float(self.current_data[-1]['l'])) / 2
            logger.info(f"Initial price set to: {self.current_price}")
            
            # Cancel any existing orders before starting
            try:
                await self.cancel_all_orders()
            except Exception as e:
                logger.warning(f"Error during initial order cancellation: {e}")
            
            # # Place a test market order to create an open position
            # try:
            #     logger.info("Placing test market order...")
            #     test_size = 0.1  # Small test size
            #     test_trade = await self.execute_trade("buy", test_size)  # No price parameter means market order
            #     logger.info(f"Test market order placed: {test_trade}")
                
            #     # Wait a moment for the order to fill
            #     await asyncio.sleep(2)
                
            #     # Check if we have an open position
            #     if self.current_position != 0:
            #         logger.info(f"Test order filled, current position: {self.current_position}")
            #     else:
            #         logger.info("Test order not filled, continuing with no position")
            # except Exception as e:
            #     logger.error(f"Error placing test order: {e}")
            
            while self.is_running:
                try:
                    # Get latest market data
                    latest_candle = data_fetcher.fetch_latest_candle()
                    if latest_candle:
                        self.current_data = self.current_data[1:] + [latest_candle]
                        self.current_price = (float(latest_candle['h']) + float(latest_candle['l'])) / 2
                        self.current_price = self.round_price(self.current_price)
                        
                        # Calculate volatility
                        volatility = self._calculate_volatility()
                        
                        # Check if we should enter or exit a position
                        should_trade, side, size = self.should_enter_trade(self.current_price, volatility)
                        
                        if should_trade:
                            try:
                                # Cancel existing orders before placing new ones
                                await self.cancel_all_orders()
                                
                                # Execute the trade
                                trade = await self.execute_trade(side, size, self.current_price)
                                logger.info(f"Executed trade: {trade}")
                                
                            except Exception as e:
                                logger.error(f"Error executing trade: {e}")
                    
                    await asyncio.sleep(update_interval)
                    
                except asyncio.CancelledError:
                    logger.info("Trading loop cancelled, cleaning up...")
                    break
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    continue
                
        except KeyboardInterrupt:
            logger.info("Trading stopped by user.")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            self.is_running = False
            try:
                # Cancel any existing orders
                await self.cancel_all_orders()
                
                # Set stop loss and take profit orders for any open position
                if self.current_position != 0:
                    logger.info(f"Setting exit orders for open position of {self.current_position}")
                    await self.set_exit_orders()
                
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
            logger.info(f"Final PnL: {self.current_pnl:.2f}")
            if hasattr(self, 'analytics'):
                logger.info(self.analytics.generate_report())
            logger.info("Trading session ended.")

    def _calculate_volatility(self) -> float:
        """Calculate current price volatility"""
        recent_candles = self.current_data[-self.volatility_window:]
        if len(recent_candles) < 2:
            return 0.0
            
        # Calculate mid prices
        mid_prices = [(float(candle['h']) + float(candle['l'])) / 2 for candle in recent_candles]
        
        # Calculate price changes
        price_changes = []
        for i in range(1, len(mid_prices)):
            change = (mid_prices[i] - mid_prices[i-1]) / mid_prices[i-1]
            price_changes.append(change)
        
        # Calculate volatility as standard deviation of price changes
        if len(price_changes) < 2:
            return 0.0
            
        mean = sum(price_changes) / len(price_changes)
        variance = sum((x - mean) ** 2 for x in price_changes) / len(price_changes)
        volatility = (variance ** 0.5) * 100  # Convert to percentage
        
        # Log detailed volatility information
        logger.info(
            f"Volatility Analysis - "
            f"Window: {self.volatility_window} candles, "
            f"Raw Volatility: {volatility:.6f}, "
            f"Price Changes: {[f'{x:.4f}' for x in price_changes]}, "
            f"Current Price: {mid_prices[-1]:.2f}"
        )
        
        return volatility

    def _log_market_state(self, price: float, volatility: float):
        """Log current market state"""
        logger.info(
            f"Market State - "
            f"Price: {price:.2f}, "
            f"Volatility: {volatility:.4f}, "
            f"Position: {self.current_position:.4f}, "
            f"PnL: {self.current_pnl:.2f}"
        )

    def _log_no_trade_reason(self, volatility: float):
        """Log why we're not trading"""
        reasons = []
        
        if volatility < self.min_volatility_threshold:
            reasons.append(f"Low volatility ({volatility:.4f} < {self.min_volatility_threshold:.4f})")
        
        if volatility > self.max_volatility_threshold:
            reasons.append(f"High volatility ({volatility:.4f} > {self.max_volatility_threshold:.4f})")
        
        if reasons:
            logger.info(f"No trade: {' and '.join(reasons)}")

    def _calculate_position_size(self):
        """Calculate position size based on volatility and current position"""
        # Base size on current position and max position (adjusted for leverage)
        leveraged_max_size = self.max_position_size * self.leverage
        remaining_capacity = leveraged_max_size - abs(self.current_position)
        
        # Calculate volatility from recent candles
        volatility = self._calculate_volatility()
        
        # Scale position size based on volatility
        if volatility > self.volatility_scaling_threshold:
            # Reduce position size as volatility increases
            scale_factor = self.min_volatility_factor
        else:
            # Increase position size as volatility decreases
            scale_factor = self.max_volatility_factor
            
        position_size = self.max_position_size * scale_factor
        
        # Log position sizing decision
        logger.info(
            f"Position Sizing - "
            f"Volatility: {volatility:.4f}, "
            f"Scale Factor: {scale_factor:.2f}, "
            f"Position Size: {position_size:.4f}"
        )
        
        return min(position_size, remaining_capacity)

    def _calculate_take_profit_stop_loss(self, X):
        """
        Use ML model to predict optimal take-profit and stop-loss spreads
        Returns: (take_profit_spread, stop_loss_spread)
        """
        if len(X) > 0:
            current_sequence = X[-1:]
            with torch.no_grad():
                model_outputs = self.model(torch.FloatTensor(current_sequence))
                take_profit_spread = model_outputs[0][0].item()
                stop_loss_spread = model_outputs[0][2].item()
                
                # Log ML predictions
                logger.info(
                    f"ML Risk Management - "
                    f"TP Spread: {take_profit_spread:.6f}, "
                    f"SL Spread: {stop_loss_spread:.6f}"
                )
        else:
            # Fallback to static config values if not enough data
            take_profit_spread = self.profit_target
            stop_loss_spread = self.stop_loss
            logger.info(
                f"Using Static Risk Management - "
                f"TP: {take_profit_spread:.6f}, "
                f"SL: {stop_loss_spread:.6f}"
            )
        return take_profit_spread, stop_loss_spread

    async def cleanup(self):
        """Clean up resources before exiting"""
        logger.info("Cleaning up before exiting...")
        try:
            # Cancel all orders
            await self.cancel_all_orders()
            
            # Set exit orders for any open position
            if self.current_position != 0:
                logger.info(f"Setting exit orders for open position of {self.current_position}")
                await self.set_exit_orders()
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        logger.info(f"Final PnL: {self.current_pnl:.2f}")
        if hasattr(self, 'analytics'):
            logger.info(self.analytics.generate_report())
        logger.info("Trading session ended.")


if __name__ == "__main__":
    # Create and run the market maker
    market_maker = MarketMaker()
    
    def handle_exit():
        """Handle exit signals"""
        logger.info("Received exit signal, cleaning up...")
        market_maker.is_running = False
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, lambda s, f: handle_exit())
    signal.signal(signal.SIGTSTP, lambda s, f: handle_exit())
    
    try:
        asyncio.run(market_maker.run_live_trading())
    except KeyboardInterrupt:
        logger.info("Trading stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        # Clean up signal handlers
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTSTP, signal.SIG_DFL) 