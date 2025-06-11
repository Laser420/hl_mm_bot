import numpy as np
from typing import List, Dict
from datetime import datetime

class TradingAnalytics:
    def __init__(self):
        self.trades: List[Dict] = []
        self.positions: List[Dict] = []
        self.prices: List[float] = []
        self.timestamps: List[datetime] = []
        
    def add_trade(self, trade: Dict):
        """Add a new trade to the analytics"""
        self.trades.append(trade)
        self.prices.append(trade['price'])
        self.timestamps.append(datetime.fromtimestamp(trade['timestamp']))
        
    def add_position(self, position: Dict):
        """Add a new position update"""
        self.positions.append(position)
        
    def calculate_metrics(self) -> Dict:
        """Calculate all performance metrics"""
        if not self.trades:
            return self._empty_metrics()
            
        metrics = {}
        
        # Basic trade statistics
        metrics['total_trades'] = len(self.trades)
        metrics['winning_trades'] = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        metrics['losing_trades'] = sum(1 for t in self.trades if t.get('pnl', 0) <= 0)
        
        # PnL metrics
        pnls = [t.get('pnl', 0) for t in self.trades]
        metrics['total_pnl'] = sum(pnls)
        metrics['avg_trade_pnl'] = np.mean(pnls) if pnls else 0
        
        # Win rate
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
        
        # Risk metrics
        if len(pnls) > 1:
            returns = np.diff(pnls) / np.array(pnls[:-1])
            metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
            
            # Calculate drawdown
            cumulative_returns = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (running_max - cumulative_returns) / running_max
            metrics['max_drawdown'] = np.max(drawdowns) if len(drawdowns) > 0 else 0
        else:
            metrics['sharpe_ratio'] = 0
            metrics['max_drawdown'] = 0
            
        # Current state
        if self.positions:
            metrics['current_position'] = self.positions[-1]['size']
        if self.prices:
            metrics['current_price'] = self.prices[-1]
            
        # Volatility
        if len(self.prices) > 1:
            returns = np.diff(self.prices) / self.prices[:-1]
            metrics['volatility'] = np.std(returns) if len(returns) > 0 else 0
        else:
            metrics['volatility'] = 0
            
        # Last trade info
        if self.trades:
            last_trade = self.trades[-1]
            metrics['last_trade_time'] = datetime.fromtimestamp(last_trade['timestamp'])
            metrics['last_trade_price'] = last_trade['price']
            metrics['last_trade_size'] = last_trade['size']
            metrics['last_trade_side'] = last_trade['side']
            
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "current_position": 0.0,
            "current_price": 0.0,
            "volatility": 0.0,
            "last_trade_time": None,
            "last_trade_price": 0.0,
            "last_trade_size": 0.0,
            "last_trade_side": None,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
    def generate_report(self) -> str:
        """Generate a detailed performance report"""
        metrics = self.calculate_metrics()
        
        report = []
        report.append("=== Trading Performance Report ===")
        report.append(f"Total Trades: {metrics['total_trades']}")
        report.append(f"Win Rate: {metrics['win_rate']:.2%}")
        report.append(f"Total PnL: ${metrics['total_pnl']:.2f}")
        report.append(f"Average Trade PnL: ${metrics['avg_trade_pnl']:.2f}")
        report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        report.append(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        report.append(f"Current Position: {metrics['current_position']:.4f}")
        report.append(f"Current Price: ${metrics['current_price']:.2f}")
        report.append(f"Volatility: {metrics['volatility']:.4%}")
        
        if metrics['last_trade_time']:
            report.append("\nLast Trade:")
            report.append(f"Time: {metrics['last_trade_time']}")
            report.append(f"Side: {metrics['last_trade_side']}")
            report.append(f"Size: {metrics['last_trade_size']:.4f}")
            report.append(f"Price: ${metrics['last_trade_price']:.2f}")
            
        return "\n".join(report) 