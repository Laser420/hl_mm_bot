#!/usr/bin/env python3
"""
Simple Session Analysis Tool
Reviews market maker trading sessions and displays key metrics
"""

import os
import json
import csv
import sys
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class SessionAnalyzer:
    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.session_id = os.path.basename(session_dir)
        
        # File paths
        self.trades_file = os.path.join(session_dir, "trades.csv")
        self.positions_file = os.path.join(session_dir, "positions.csv")
        self.performance_file = os.path.join(session_dir, "performance.csv")
        self.metadata_file = os.path.join(session_dir, "metadata.json")
        
        # Load data
        self.metadata = self._load_metadata()
        self.trades = self._load_trades()
        self.positions = self._load_positions()
    
    def _load_metadata(self) -> Dict:
        """Load session metadata"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
        return {}
    
    def _load_trades(self) -> pd.DataFrame:
        """Load trades data"""
        try:
            if os.path.exists(self.trades_file):
                return pd.read_csv(self.trades_file)
        except Exception as e:
            print(f"Warning: Could not load trades: {e}")
        return pd.DataFrame()
    
    def _load_positions(self) -> pd.DataFrame:
        """Load positions data"""
        try:
            if os.path.exists(self.positions_file):
                return pd.read_csv(self.positions_file)
        except Exception as e:
            print(f"Warning: Could not load positions: {e}")
        return pd.DataFrame()
    
    def get_session_summary(self) -> Dict:
        """Get basic session information"""
        is_paper = self.session_id.startswith('paper_')
        
        # Parse session timestamp
        try:
            if is_paper:
                timestamp_str = self.session_id.replace('paper_', '')
            else:
                timestamp_str = self.session_id
            session_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except:
            session_time = datetime.now()
        
        # Calculate session duration
        if not self.trades.empty:
            start_time = self.trades['timestamp'].min()
            end_time = self.trades['timestamp'].max()
            duration_seconds = end_time - start_time
            duration_minutes = duration_seconds / 60
        else:
            duration_minutes = 0
        
        return {
            'session_id': self.session_id,
            'trading_mode': 'Paper Trading' if is_paper else 'Live Trading',
            'session_start': session_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_minutes': duration_minutes,
            'asset': self.metadata.get('config', {}).get('asset', 'Unknown'),
            'has_trades': not self.trades.empty,
            'has_positions': not self.positions.empty
        }
    
    def get_trading_metrics(self) -> Dict:
        """Calculate key trading metrics"""
        if self.trades.empty:
            return {
                'total_trades': 0,
                'completed_round_trips': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_fees': 0.0,
                'avg_trade_pnl': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'volume_traded': 0.0
            }
        
        # Calculate metrics
        total_trades = len(self.trades)
        
        # PnL analysis (only count closed positions)
        trade_pnls = []
        if 'pnl' in self.trades.columns:
            # Look for completed round trips (position changes from non-zero back to zero)
            for i in range(1, len(self.trades)):
                prev_pos = self.trades.iloc[i-1]['position_after'] if 'position_after' in self.trades.columns else 0
                curr_pos = self.trades.iloc[i]['position_after'] if 'position_after' in self.trades.columns else 0
                
                # If we went from having a position to flat, this completed a trade
                if prev_pos != 0 and curr_pos == 0:
                    pnl_change = self.trades.iloc[i]['pnl'] - self.trades.iloc[i-1]['pnl']
                    trade_pnls.append(pnl_change)
        
        winning_trades = len([pnl for pnl in trade_pnls if pnl > 0])
        losing_trades = len([pnl for pnl in trade_pnls if pnl < 0])
        win_rate = (winning_trades / len(trade_pnls)) * 100 if trade_pnls else 0
        
        total_pnl = self.trades['pnl'].iloc[-1] if 'pnl' in self.trades.columns and not self.trades.empty else 0
        total_fees = self.trades['fee'].sum() if 'fee' in self.trades.columns else 0
        avg_trade_pnl = sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0
        largest_win = max(trade_pnls) if trade_pnls else 0
        largest_loss = min(trade_pnls) if trade_pnls else 0
        
        # Volume calculation
        volume_traded = 0
        if 'size' in self.trades.columns and 'price' in self.trades.columns:
            volume_traded = (self.trades['size'] * self.trades['price']).sum()
        
        return {
            'total_trades': total_trades,
            'completed_round_trips': len(trade_pnls),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_fees': total_fees,
            'avg_trade_pnl': avg_trade_pnl,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'volume_traded': volume_traded
        }
    
    def get_position_analysis(self) -> Dict:
        """Analyze position data"""
        if self.positions.empty:
            return {
                'max_position': 0.0,
                'min_position': 0.0,
                'avg_position': 0.0,
                'time_in_position': 0.0
            }
        
        if 'size' in self.positions.columns:
            max_position = self.positions['size'].max()
            min_position = self.positions['size'].min()
            avg_position = self.positions['size'].mean()
        else:
            max_position = min_position = avg_position = 0.0
        
        # Calculate time spent in position vs flat
        flat_time = len(self.positions[self.positions.get('size', pd.Series([0])) == 0])
        total_time = len(self.positions)
        time_in_position = ((total_time - flat_time) / total_time) * 100 if total_time > 0 else 0
        
        return {
            'max_position': max_position,
            'min_position': min_position,
            'avg_position': avg_position,
            'time_in_position': time_in_position
        }
    
    def print_analysis(self):
        """Print comprehensive session analysis"""
        summary = self.get_session_summary()
        metrics = self.get_trading_metrics()
        position_data = self.get_position_analysis()
        
        print("=" * 80)
        print(f"SESSION ANALYSIS: {summary['session_id']}")
        print("=" * 80)
        
        # Session Info
        print(f"\nSESSION INFORMATION:")
        print(f"  Trading Mode:     {summary['trading_mode']}")
        print(f"  Asset:            {summary['asset']}")
        print(f"  Started:          {summary['session_start']}")
        print(f"  Duration:         {summary['duration_minutes']:.1f} minutes")
        print(f"  Data Available:   Trades: {'✓' if summary['has_trades'] else '✗'}, Positions: {'✓' if summary['has_positions'] else '✗'}")
        
        # Trading Metrics
        print(f"\nTRADING PERFORMANCE:")
        print(f"  Total Orders:     {metrics['total_trades']}")
        print(f"  Completed Trades: {metrics['completed_round_trips']}")
        print(f"  Winning Trades:   {metrics['winning_trades']}")
        print(f"  Losing Trades:    {metrics['losing_trades']}")
        print(f"  Win Rate:         {metrics['win_rate']:.1f}%")
        print(f"  Total PnL:        ${metrics['total_pnl']:.4f}")
        print(f"  Total Fees:       ${metrics['total_fees']:.4f}")
        print(f"  Net PnL:          ${metrics['total_pnl'] - metrics['total_fees']:.4f}")
        print(f"  Avg Trade PnL:    ${metrics['avg_trade_pnl']:.4f}")
        print(f"  Largest Win:      ${metrics['largest_win']:.4f}")
        print(f"  Largest Loss:     ${metrics['largest_loss']:.4f}")
        print(f"  Volume Traded:    ${metrics['volume_traded']:.2f}")
        
        # Position Analysis
        print(f"\nPOSITION ANALYSIS:")
        print(f"  Max Long:         {position_data['max_position']:.4f}")
        print(f"  Max Short:        {position_data['min_position']:.4f}")
        print(f"  Avg Position:     {position_data['avg_position']:.4f}")
        print(f"  Time in Position: {position_data['time_in_position']:.1f}%")
        
        # Recent Trades
        if not self.trades.empty:
            print(f"\nRECENT TRADES (Last 5):")
            recent_trades = self.trades.tail(5)
            print(f"  {'Time':<12} {'Side':<4} {'Size':<8} {'Price':<10} {'PnL':<10}")
            print(f"  {'-' * 50}")
            for _, trade in recent_trades.iterrows():
                trade_time = datetime.fromtimestamp(trade['timestamp']).strftime('%H:%M:%S')
                side = trade.get('side', 'N/A')
                size = trade.get('size', 0)
                price = trade.get('price', 0)
                pnl = trade.get('pnl', 0)
                print(f"  {trade_time:<12} {side:<4} {size:<8.4f} ${price:<9.2f} ${pnl:<9.4f}")
        
        print("=" * 80)


def get_available_sessions() -> List[str]:
    """Get list of available trading sessions"""
    sessions_dir = "logs/sessions"
    if not os.path.exists(sessions_dir):
        return []
    
    sessions = []
    for session_id in os.listdir(sessions_dir):
        session_path = os.path.join(sessions_dir, session_id)
        if os.path.isdir(session_path):
            # Check if it has any data files
            has_data = any(
                os.path.exists(os.path.join(session_path, f))
                for f in ["trades.csv", "positions.csv", "metadata.json"]
            )
            if has_data:
                sessions.append(session_id)
    
    return sorted(sessions, reverse=True)  # Most recent first


def select_session_interactive() -> Optional[str]:
    """Interactive session selection"""
    sessions = get_available_sessions()
    
    if not sessions:
        print("No trading sessions found!")
        print("Run the market maker first to generate session data.")
        return None
    
    print("\nAvailable Trading Sessions:")
    print("=" * 40)
    
    for i, session in enumerate(sessions, 1):
        session_type = "Paper" if session.startswith('paper_') else "Live"
        print(f"{i:2d}. {session_type:<6} {session}")
    
    print(f"{len(sessions) + 1:2d}. Exit")
    
    while True:
        try:
            choice = input(f"\nSelect session (1-{len(sessions) + 1}): ").strip()
            
            if not choice:
                continue
                
            choice_num = int(choice)
            
            if choice_num == len(sessions) + 1:
                return None
            elif 1 <= choice_num <= len(sessions):
                return sessions[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(sessions) + 1}")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def main():
    """Main entry point"""
    print("Market Maker Session Analysis Tool")
    
    # Check for command line session ID
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
        session_dir = os.path.join("logs", "sessions", session_id)
        
        if not os.path.exists(session_dir):
            print(f"Session '{session_id}' not found!")
            return
            
        analyzer = SessionAnalyzer(session_dir)
        analyzer.print_analysis()
        return
    
    # Interactive mode
    while True:
        session_id = select_session_interactive()
        
        if session_id is None:
            print("Goodbye!")
            break
            
        session_dir = os.path.join("logs", "sessions", session_id)
        analyzer = SessionAnalyzer(session_dir)
        analyzer.print_analysis()
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()