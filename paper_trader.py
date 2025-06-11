import time
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
from market_maker import MarketMaker, OrderBook
from utils.trade_logger import TradeLogger
import os

class PaperTrader(MarketMaker):
    def __init__(self, 
                 config_path: str = "config.yaml",
                 asset: str = "ETH",
                 max_position_size: float = 1.0,
                 leverage: float = 1.0,
                 profit_target: float = 0.002,
                 stop_loss: float = 0.005,
                 order_size: float = 0.1,
                 spread: float = 0.001,
                 update_interval: float = 1.0,
                 log_dir: str = "logs"):
        # Initialize parent class with config
        super().__init__(config_path=config_path)
        
        # Override trading parameters if provided
        self.coin = asset
        self.max_position_size = max_position_size
        self.leverage = leverage
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.order_size = order_size
        self.spread = spread
        self.update_interval = update_interval
        
        # Paper trading specific attributes
        self.paper_orders: Dict[str, Dict] = {}  # Track open paper orders
        self.paper_position: float = 0.0  # Current paper position size
        self.paper_entry_price: float = 0.0  # Average entry price for paper position
        self.paper_pnl: float = 0.0  # Current paper PnL
        
        # Get session directory from environment
        self.session_dir = os.getenv("SESSION_DIR", os.path.join("logs", "sessions", "default_session"))
        os.makedirs(self.session_dir, exist_ok=True)
        print(f"[PaperTrader] Using session directory: {self.session_dir}")
        
        # Set session directory in environment for TradeLogger
        os.environ["SESSION_DIR"] = self.session_dir
        
        # Reinitialize trade logger with correct session directory
        self.trade_logger = TradeLogger(session_id=self.session_id, config=self.config)
        
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
        
    def _place_order(self, side: str, size: float, price: float) -> Optional[str]:
        """Simulate placing an order and return a paper order ID"""
        order_id = f"paper_{int(time.time() * 1000)}_{len(self.paper_orders)}"
        
        # Store the paper order
        self.paper_orders[order_id] = {
            'side': side,
            'size': size,
            'price': price,
            'timestamp': datetime.now().timestamp(),
            'filled': False
        }
        
        return order_id
    
    def _cancel_order(self, order_id: str) -> bool:
        """Simulate canceling an order"""
        if order_id in self.paper_orders and not self.paper_orders[order_id]['filled']:
            del self.paper_orders[order_id]
            return True
        return False
    
    def _check_order_fills(self) -> None:
        """Simulate order fills based on market conditions"""
        current_price = self.orderbook.mid_price
        
        # Check each open order for potential fills
        for order_id, order in list(self.paper_orders.items()):
            if order['filled']:
                continue
                
            # Simulate fill based on price movement
            if order['side'] == 'buy' and current_price <= order['price']:
                self._simulate_fill(order_id, order)
            elif order['side'] == 'sell' and current_price >= order['price']:
                self._simulate_fill(order_id, order)
    
    def _simulate_fill(self, order_id: str, order: Dict) -> None:
        """Simulate an order fill and update paper position"""
        # Mark order as filled
        self.paper_orders[order_id]['filled'] = True
        current_price = self.orderbook.mid_price
        
        # Update paper position
        if order['side'] == 'buy':
            self.paper_position += order['size']
            # Update average entry price
            if self.paper_position > 0:
                self.paper_entry_price = (
                    (self.paper_entry_price * (self.paper_position - order['size']) +
                     order['price'] * order['size']) / self.paper_position
                )
        else:  # sell
            self.paper_position -= order['size']
            # Calculate PnL for closing position
            if self.paper_position == 0:
                pnl = (order['price'] - self.paper_entry_price) * order['size']
                self.paper_pnl += pnl
                self.paper_entry_price = 0.0
        
        # Log the simulated trade with more detailed information
        self.trade_logger.log_trade(
            timestamp=datetime.now().timestamp(),
            order_id=order_id,
            side=order['side'],
            size=order['size'],
            price=order['price'],
            asset=self.coin,
            pnl=self.paper_pnl if self.paper_position == 0 else 0.0,
            position_after=self.paper_position,
            entry_price=self.paper_entry_price,
            exit_price=order['price'] if self.paper_position == 0 else None,
            trade_duration=time.time() - order['timestamp'] if self.paper_position == 0 else None
        )
        
        # Log position update with more detailed information
        self.trade_logger.log_position(
            timestamp=datetime.now().timestamp(),
            size=self.paper_position,
            value=abs(self.paper_position * current_price),
            asset=self.coin,
            entry_price=self.paper_entry_price,
            unrealized_pnl=(current_price - self.paper_entry_price) * self.paper_position if self.paper_position != 0 else 0.0
        )
        
        # Update performance metrics
        self._update_paper_pnl()
    
    def _update_paper_pnl(self) -> None:
        """Update paper PnL based on current market price"""
        current_price = self.orderbook.mid_price
        unrealized_pnl = (current_price - self.paper_entry_price) * self.paper_position if self.paper_position != 0 else 0.0
        total_pnl = self.paper_pnl + unrealized_pnl
        
        # Calculate additional metrics
        filled_orders = [o for o in self.paper_orders.values() if o['filled']]
        winning_trades = len([o for o in filled_orders if o['price'] > self.paper_entry_price])
        total_trades = len(filled_orders)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        # Log performance metrics with more detailed information
        self.trade_logger.log_performance(
            timestamp=datetime.now().timestamp(),
            cumulative_pnl=total_pnl,
            drawdown=self._calculate_drawdown(total_pnl),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=total_trades - winning_trades,
            win_rate=win_rate,
            current_position=self.paper_position,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.paper_pnl,
            sharpe_ratio=self._calculate_sharpe_ratio(),
            volatility=self._calculate_volatility()
        )
    
    def _calculate_drawdown(self, current_pnl: float) -> float:
        """Calculate current drawdown"""
        if not hasattr(self, '_peak_pnl'):
            self._peak_pnl = current_pnl
        else:
            self._peak_pnl = max(self._peak_pnl, current_pnl)
        
        if self._peak_pnl == 0:
            return 0.0
        
        return (self._peak_pnl - current_pnl) / abs(self._peak_pnl) * 100
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from PnL series"""
        if not hasattr(self, '_pnl_history'):
            self._pnl_history = []
        
        current_pnl = self.paper_pnl + (self.orderbook.mid_price - self.paper_entry_price) * self.paper_position
        self._pnl_history.append(current_pnl)
        
        if len(self._pnl_history) < 2:
            return 0.0
        
        returns = np.diff(self._pnl_history)
        if len(returns) == 0:
            return 0.0
        
        return np.sqrt(252) * (np.mean(returns) / np.std(returns)) if np.std(returns) != 0 else 0.0
    
    def _calculate_volatility(self) -> float:
        """Calculate volatility from price changes"""
        if not hasattr(self, '_price_history'):
            self._price_history = []
        
        self._price_history.append(self.orderbook.mid_price)
        
        if len(self._price_history) < 2:
            return 0.0
        
        returns = np.diff(self._price_history) / self._price_history[:-1]
        return np.std(returns) * np.sqrt(252) * 100  # Annualized volatility in percentage
    
    def run(self):
        """Run the paper trading bot"""
        print(f"Starting paper trading bot for {self.coin}")
        print(f"Initial position: {self.paper_position}")
        print(f"Initial PnL: {self.paper_pnl}")
        
        # Initialize performance tracking
        self._pnl_history = []
        self._price_history = []
        
        while True:
            try:
                # Update order book
                self._update_orderbook()
                
                # Check for order fills
                self._check_order_fills()
                
                # Update PnL and metrics
                self._update_paper_pnl()
                
                # Place new orders if needed
                self._place_orders()
                
                # Check for profit target or stop loss
                if self._check_exit_conditions():
                    print("Exit conditions met. Closing all positions...")
                    self._close_all_positions()
                    break
                
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                print("\nStopping paper trading bot...")
                self._close_all_positions()
                break
            except Exception as e:
                print(f"Error in paper trading loop: {e}")
                time.sleep(self.update_interval)
    
    def _close_all_positions(self):
        """Close all paper positions"""
        if self.paper_position != 0:
            current_price = self.orderbook.mid_price
            side = 'sell' if self.paper_position > 0 else 'buy'
            size = abs(self.paper_position)
            
            # Simulate closing position
            order_id = self._place_order(side, size, current_price)
            if order_id:
                self._simulate_fill(order_id, {
                    'side': side,
                    'size': size,
                    'price': current_price,
                    'filled': False
                })
        
        print(f"Final paper position: {self.paper_position}")
        print(f"Final paper PnL: {self.paper_pnl}")

if __name__ == "__main__":
    # Example usage
    paper_trader = PaperTrader(
        config_path="config.yaml",
        asset="ETH",
        max_position_size=1.0,
        leverage=1.0,
        profit_target=0.002,
        stop_loss=0.005,
        order_size=0.1,
        spread=0.001,
        update_interval=1.0
    )
    paper_trader.run() 