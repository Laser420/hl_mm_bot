import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from typing import Dict, Tuple
import argparse

class TradeAnalyzer:
    def __init__(self, session_id: str = None, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.session_id = session_id
        self.trades_df = None
        self.positions_df = None
        self.performance_df = None
        self.load_data()
    
    def load_data(self):
        """Load all log files into pandas DataFrames"""
        if not self.session_id:
            return
            
        session_dir = os.path.join(self.log_dir, "sessions", self.session_id)
        
        # Load trades
        trades_path = os.path.join(session_dir, "trades.csv")
        if os.path.exists(trades_path):
            self.trades_df = pd.read_csv(trades_path)
            self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'], unit='s')
        
        # Load positions
        positions_path = os.path.join(session_dir, "positions.csv")
        if os.path.exists(positions_path):
            self.positions_df = pd.read_csv(positions_path)
            self.positions_df['timestamp'] = pd.to_datetime(self.positions_df['timestamp'], unit='s')
        
        # Load performance
        performance_path = os.path.join(session_dir, "performance.csv")
        if os.path.exists(performance_path):
            self.performance_df = pd.read_csv(performance_path)
            self.performance_df['timestamp'] = pd.to_datetime(self.performance_df['timestamp'], unit='s')
    
    def calculate_basic_metrics(self) -> Dict:
        """Calculate basic trading metrics"""
        # Initialize default metrics
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_trade_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'avg_position_size': 0.0,
            'avg_trade_duration': 0.0,
            'volatility': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_holding_time': 0.0
        }
        
        if self.trades_df is None or len(self.trades_df) == 0:
            return metrics
        
        # Calculate volatility from price changes
        if len(self.trades_df) > 1:
            price_changes = self.trades_df['price'].pct_change().dropna()
            volatility = price_changes.std() * np.sqrt(252) * 100  # Annualized volatility in percentage
        else:
            volatility = 0.0
        
        # Update metrics with actual data
        metrics.update({
            'total_trades': len(self.trades_df),
            'winning_trades': len(self.trades_df[self.trades_df['pnl'] > 0]),
            'losing_trades': len(self.trades_df[self.trades_df['pnl'] < 0]),
            'win_rate': len(self.trades_df[self.trades_df['pnl'] > 0]) / len(self.trades_df) * 100 if len(self.trades_df) > 0 else 0.0,
            'avg_trade_pnl': self.trades_df['pnl'].mean() if len(self.trades_df) > 0 else 0.0,
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'avg_position_size': self.trades_df['size'].mean() if len(self.trades_df) > 0 else 0.0,
            'avg_trade_duration': self._calculate_avg_trade_duration(),
            'volatility': volatility,
            'profit_factor': self._calculate_profit_factor(),
            'avg_win': self.trades_df[self.trades_df['pnl'] > 0]['pnl'].mean() if len(self.trades_df[self.trades_df['pnl'] > 0]) > 0 else 0.0,
            'avg_loss': self.trades_df[self.trades_df['pnl'] < 0]['pnl'].mean() if len(self.trades_df[self.trades_df['pnl'] < 0]) > 0 else 0.0,
            'max_consecutive_wins': self._calculate_max_consecutive_wins(),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(),
            'avg_holding_time': self._calculate_avg_holding_time()
        })
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from PnL series"""
        if self.trades_df is None or len(self.trades_df) == 0:
            return 0.0
        
        pnl_series = self.trades_df['pnl']
        rolling_max = pnl_series.expanding().max()
        drawdowns = (pnl_series - rolling_max) / rolling_max
        return abs(drawdowns.min()) * 100  # Return as percentage
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from PnL series"""
        if self.trades_df is None or len(self.trades_df) == 0:
            return 0.0
        
        pnl_returns = self.trades_df['pnl'].pct_change().dropna()
        if len(pnl_returns) == 0:
            return 0.0
        
        return np.sqrt(252) * (pnl_returns.mean() / pnl_returns.std())
    
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average time between trades in minutes"""
        if self.trades_df is None or len(self.trades_df) < 2:
            return 0.0
        
        time_diffs = self.trades_df['timestamp'].diff().dropna()
        return time_diffs.mean().total_seconds() / 60
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if self.trades_df is None or len(self.trades_df) == 0:
            return 0.0
        
        gross_profit = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(self.trades_df[self.trades_df['pnl'] < 0]['pnl'].sum())
        
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def _calculate_max_consecutive_wins(self) -> int:
        """Calculate maximum number of consecutive winning trades"""
        if self.trades_df is None or len(self.trades_df) == 0:
            return 0
        
        wins = self.trades_df['pnl'] > 0
        consecutive_wins = 0
        max_consecutive_wins = 0
        
        for is_win in wins:
            if is_win:
                consecutive_wins += 1
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_wins = 0
        
        return max_consecutive_wins
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum number of consecutive losing trades"""
        if self.trades_df is None or len(self.trades_df) == 0:
            return 0
        
        losses = self.trades_df['pnl'] < 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for is_loss in losses:
            if is_loss:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive_losses
    
    def _calculate_avg_holding_time(self) -> float:
        """Calculate average holding time in minutes"""
        if self.trades_df is None or len(self.trades_df) < 2:
            return 0.0
        
        # Calculate time between trades for the same position
        holding_times = []
        current_position = 0
        entry_time = None
        
        for _, trade in self.trades_df.iterrows():
            if current_position == 0:  # New position
                entry_time = trade['timestamp']
                current_position = trade['position_after']
            elif current_position * trade['position_after'] <= 0:  # Position closed
                if entry_time is not None:
                    holding_time = (trade['timestamp'] - entry_time).total_seconds() / 60
                    holding_times.append(holding_time)
                entry_time = None
                current_position = trade['position_after']
        
        return np.mean(holding_times) if holding_times else 0.0
    
    def plot_pnl_over_time(self, save_path: str = None):
        """Plot PnL over time"""
        if self.trades_df is None or len(self.trades_df) == 0:
            print("No trade data available for plotting")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.trades_df['timestamp'], self.trades_df['pnl'], label='PnL')
        plt.title('PnL Over Time')
        plt.xlabel('Time')
        plt.ylabel('PnL')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_position_size_distribution(self, save_path: str = None):
        """Plot distribution of position sizes"""
        if self.trades_df is None or len(self.trades_df) == 0:
            print("No trade data available for plotting")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.trades_df['size'], bins=30, alpha=0.7)
        plt.title('Position Size Distribution')
        plt.xlabel('Position Size')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_win_loss_ratio(self, save_path: str = None):
        """Plot win/loss ratio over time"""
        if self.trades_df is None or len(self.trades_df) == 0:
            print("No trade data available for plotting")
            return
        
        # Calculate rolling win rate
        window = min(50, len(self.trades_df))
        rolling_win_rate = self.trades_df['pnl'].rolling(window=window).apply(
            lambda x: (x > 0).mean() * 100
        )
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.trades_df['timestamp'], rolling_win_rate, label=f'{window}-trade Win Rate')
        plt.axhline(y=50, color='r', linestyle='--', label='50% Win Rate')
        plt.title('Rolling Win Rate Over Time')
        plt.xlabel('Time')
        plt.ylabel('Win Rate (%)')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_volatility_over_time(self, save_path: str = None):
        """Plot volatility over time"""
        if self.trades_df is None or len(self.trades_df) < 2:
            print("No trade data available for plotting")
            return
        
        # Calculate rolling volatility
        window = min(50, len(self.trades_df))
        rolling_volatility = self.trades_df['price'].pct_change().rolling(window=window).std() * np.sqrt(252) * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.trades_df['timestamp'], rolling_volatility, label=f'{window}-trade Rolling Volatility')
        plt.title('Volatility Over Time')
        plt.xlabel('Time')
        plt.ylabel('Volatility (%)')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_holding_time_distribution(self, save_path: str = None):
        """Plot distribution of holding times"""
        if self.trades_df is None or len(self.trades_df) < 2:
            print("No trade data available for plotting")
            return
        
        holding_times = []
        current_position = 0
        entry_time = None
        
        for _, trade in self.trades_df.iterrows():
            if current_position == 0:  # New position
                entry_time = trade['timestamp']
                current_position = trade['position_after']
            elif current_position * trade['position_after'] <= 0:  # Position closed
                if entry_time is not None:
                    holding_time = (trade['timestamp'] - entry_time).total_seconds() / 60
                    holding_times.append(holding_time)
                entry_time = None
                current_position = trade['position_after']
        
        if not holding_times:
            print("No holding time data available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(holding_times, bins=30, alpha=0.7)
        plt.title('Holding Time Distribution')
        plt.xlabel('Holding Time (minutes)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def generate_report(self, output_dir: str = "analysis"):
        """Generate a comprehensive trading report"""
        if not self.session_id:
            print("No session ID provided!")
            return
            
        # Create session-specific output directory
        session_output_dir = os.path.join(output_dir, self.session_id)
        os.makedirs(session_output_dir, exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_basic_metrics()
        
        # Generate plots only if we have trade data
        if self.trades_df is not None and len(self.trades_df) > 0:
            self.plot_pnl_over_time(os.path.join(session_output_dir, 'pnl_over_time.png'))
            self.plot_position_size_distribution(os.path.join(session_output_dir, 'position_distribution.png'))
            self.plot_win_loss_ratio(os.path.join(session_output_dir, 'win_loss_ratio.png'))
            self.plot_volatility_over_time(os.path.join(session_output_dir, 'volatility_over_time.png'))
            self.plot_holding_time_distribution(os.path.join(session_output_dir, 'holding_time_distribution.png'))
        
        # Save metrics to CSV
        pd.DataFrame([metrics]).to_csv(os.path.join(session_output_dir, 'trading_metrics.csv'), index=False)
        
        # Print summary
        print(f"\nTrading Performance Summary for Session {self.session_id}:")
        if self.trades_df is None or len(self.trades_df) == 0:
            print("No trading data available for this session.")
            return
            
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Average Trade PnL: {metrics['avg_trade_pnl']:.6f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Average Position Size: {metrics['avg_position_size']:.6f}")
        print(f"Average Trade Duration: {metrics['avg_trade_duration']:.2f} minutes")
        print(f"Volatility: {metrics['volatility']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Average Win: {metrics['avg_win']:.6f}")
        print(f"Average Loss: {metrics['avg_loss']:.6f}")
        print(f"Max Consecutive Wins: {metrics['max_consecutive_wins']}")
        print(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        print(f"Average Holding Time: {metrics['avg_holding_time']:.2f} minutes")

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze trading performance from logs')
    
    # Main actions
    parser.add_argument('--report', action='store_true', help='Generate full report with all metrics and plots')
    parser.add_argument('--plot', choices=['pnl', 'position', 'winrate', 'all'], 
                       help='Generate specific plot(s)')
    parser.add_argument('--metrics', type=str, 
                       help='Comma-separated list of metrics to display (e.g., "win_rate,sharpe_ratio")')
    
    # Session selection
    parser.add_argument('--session', type=str, help='Session ID to analyze')
    
    # Data filtering
    parser.add_argument('--start-date', type=str, help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for analysis (YYYY-MM-DD)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='analysis',
                       help='Directory to save analysis outputs')
    parser.add_argument('--no-display', action='store_true',
                       help='Save plots without displaying them')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get available sessions
    sessions_dir = os.path.join("logs", "sessions")
    if not os.path.exists(sessions_dir):
        print("No trading sessions found!")
        return
        
    sessions = [d for d in os.listdir(sessions_dir) 
               if os.path.isdir(os.path.join(sessions_dir, d)) and 
               os.path.exists(os.path.join(sessions_dir, d, "metadata.json"))]
    
    if not sessions:
        print("No trading sessions found!")
        return
    
    # Use specified session or most recent
    session_id = args.session or sessions[0]
    if session_id not in sessions:
        print(f"Session {session_id} not found!")
        return
    
    analyzer = TradeAnalyzer(session_id=session_id)
    
    # Filter data by date if specified
    if args.start_date or args.end_date:
        start_date = pd.to_datetime(args.start_date) if args.start_date else None
        end_date = pd.to_datetime(args.end_date) if args.end_date else None
        
        if analyzer.trades_df is not None:
            analyzer.trades_df = analyzer.trades_df[
                (analyzer.trades_df['timestamp'] >= start_date) if start_date else True &
                (analyzer.trades_df['timestamp'] <= end_date) if end_date else True
            ]
    
    # Generate full report
    if args.report:
        analyzer.generate_report(args.output_dir)
        return
    
    # Generate specific plots
    if args.plot:
        if args.plot in ['pnl', 'all']:
            analyzer.plot_pnl_over_time(
                os.path.join(args.output_dir, session_id, 'pnl_over_time.png') if args.no_display else None
            )
        if args.plot in ['position', 'all']:
            analyzer.plot_position_size_distribution(
                os.path.join(args.output_dir, session_id, 'position_distribution.png') if args.no_display else None
            )
        if args.plot in ['winrate', 'all']:
            analyzer.plot_win_loss_ratio(
                os.path.join(args.output_dir, session_id, 'win_loss_ratio.png') if args.no_display else None
            )
    
    # Display specific metrics
    if args.metrics:
        metrics = analyzer.calculate_basic_metrics()
        requested_metrics = args.metrics.split(',')
        print(f"\nRequested Metrics for Session {session_id}:")
        for metric in requested_metrics:
            if metric in metrics:
                print(f"{metric}: {metrics[metric]:.6f}")
            else:
                print(f"Metric '{metric}' not found")

if __name__ == "__main__":
    main() 