from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from datetime import datetime
import time
import os
import json
import pandas as pd
from typing import Dict, Optional

class TradingTUI:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header"),
            Layout(name="main"),
            Layout(name="footer")
        )
        self.layout["main"].split_row(
            Layout(name="trading_info"),
            Layout(name="performance")
        )
        
        # Initialize metrics
        self.metrics = self._empty_metrics()
        self.start_time = datetime.now()
        
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
        
    def _get_latest_session(self) -> Optional[str]:
        """Get the most recent session ID"""
        sessions_dir = os.path.join("logs", "sessions")
        if not os.path.exists(sessions_dir):
            return None
            
        sessions = []
        for session_id in os.listdir(sessions_dir):
            session_path = os.path.join(sessions_dir, session_id)
            if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "metadata.json")):
                sessions.append(session_id)
        return sorted(sessions, reverse=True)[0] if sessions else None
        
    def _read_latest_metrics(self):
        """Read latest metrics from log files"""
        session_id = self._get_latest_session()
        if not session_id:
            return
            
        session_dir = os.path.join("logs", "sessions", session_id)
        
        # Read trades
        trades_file = os.path.join(session_dir, "trades.csv")
        if os.path.exists(trades_file):
            trades_df = pd.read_csv(trades_file)
            if not trades_df.empty:
                self.metrics["total_trades"] = len(trades_df)
                self.metrics["winning_trades"] = len(trades_df[trades_df["pnl"] > 0])
                self.metrics["losing_trades"] = len(trades_df[trades_df["pnl"] <= 0])
                self.metrics["total_pnl"] = trades_df["pnl"].sum()
                self.metrics["avg_trade_pnl"] = trades_df["pnl"].mean()
                self.metrics["win_rate"] = self.metrics["winning_trades"] / self.metrics["total_trades"] if self.metrics["total_trades"] > 0 else 0
                
                # Get last trade info
                last_trade = trades_df.iloc[-1]
                self.metrics["last_trade_time"] = datetime.fromtimestamp(last_trade["timestamp"])
                self.metrics["last_trade_price"] = last_trade["price"]
                self.metrics["last_trade_size"] = last_trade["size"]
                self.metrics["last_trade_side"] = last_trade["side"]
        
        # Read positions
        positions_file = os.path.join(session_dir, "positions.csv")
        if os.path.exists(positions_file):
            positions_df = pd.read_csv(positions_file)
            if not positions_df.empty:
                self.metrics["current_position"] = positions_df.iloc[-1]["size"]
                # Try to get price from either mark_price or price column
                if "mark_price" in positions_df.columns:
                    self.metrics["current_price"] = positions_df.iloc[-1]["mark_price"]
                elif "price" in positions_df.columns:
                    self.metrics["current_price"] = positions_df.iloc[-1]["price"]
                else:
                    self.metrics["current_price"] = 0.0  # Default if no price column found
        
        # Read performance
        performance_file = os.path.join(session_dir, "performance.csv")
        if os.path.exists(performance_file):
            performance_df = pd.read_csv(performance_file)
            if not performance_df.empty:
                self.metrics["max_drawdown"] = abs(performance_df["drawdown"].min()) / 100  # Convert to decimal
                
                # Calculate Sharpe ratio
                if len(performance_df) > 1:
                    returns = performance_df["cumulative_pnl"].pct_change().dropna()
                    if len(returns) > 0 and returns.std() != 0:
                        self.metrics["sharpe_ratio"] = (returns.mean() / returns.std()) * (252 ** 0.5)  # Annualized
                
                # Calculate volatility
                if len(performance_df) > 1:
                    price_changes = performance_df["cumulative_pnl"].pct_change().dropna()
                    self.metrics["volatility"] = price_changes.std() * (252 ** 0.5)  # Annualized
        
    def create_header(self) -> Panel:
        """Create header panel with bot status"""
        header_text = Text()
        header_text.append("HyperLiquid Market Maker Bot", style="bold blue")
        header_text.append(f" | Running since: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        
        return Panel(header_text, title="Status", border_style="blue")
    
    def create_trading_info(self) -> Table:
        """Create trading information table"""
        table = Table(title="Trading Information")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Current Position", f"{self.metrics['current_position']:.4f}")
        table.add_row("Current Price", f"${self.metrics['current_price']:.2f}")
        table.add_row("Volatility", f"{self.metrics['volatility']:.4%}")
        table.add_row("Last Trade", f"{self.metrics['last_trade_side']} {self.metrics['last_trade_size']:.4f} @ ${self.metrics['last_trade_price']:.2f}")
        table.add_row("Last Trade Time", str(self.metrics['last_trade_time'] or 'N/A'))
        
        return table
    
    def create_performance(self) -> Table:
        """Create performance metrics table"""
        table = Table(title="Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Trades", str(self.metrics['total_trades']))
        table.add_row("Win Rate", f"{self.metrics['win_rate']:.2%}")
        table.add_row("Total PnL", f"${self.metrics['total_pnl']:.2f}")
        table.add_row("Avg Trade PnL", f"${self.metrics['avg_trade_pnl']:.2f}")
        table.add_row("Max Drawdown", f"{self.metrics['max_drawdown']:.2%}")
        table.add_row("Sharpe Ratio", f"{self.metrics['sharpe_ratio']:.2f}")
        
        return table
    
    def create_footer(self) -> Panel:
        """Create footer with controls information"""
        footer_text = Text()
        footer_text.append("Controls: ", style="bold")
        footer_text.append("Ctrl+C: Quit | Space: Pause/Resume", style="dim")
        
        return Panel(footer_text, title="Controls", border_style="blue")
    
    def render(self):
        """Render the complete TUI"""
        self._read_latest_metrics()  # Update metrics before rendering
        self.layout["header"].update(self.create_header())
        self.layout["trading_info"].update(self.create_trading_info())
        self.layout["performance"].update(self.create_performance())
        self.layout["footer"].update(self.create_footer())
        
        self.console.clear()
        self.console.print(self.layout)
    
    def start(self):
        """Start the TUI with live updates"""
        is_paused = False
        
        try:
            with Live(self.layout, refresh_per_second=1) as live:
                while True:
                    if not is_paused:
                        self.render()
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting TUI...") 