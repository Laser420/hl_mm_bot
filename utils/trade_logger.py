import os
import json
import csv
from datetime import datetime
from typing import Dict, Any

class TradeLogger:
    def __init__(self, session_id: str, config: Dict[str, Any]):
        """Initialize the trade logger with session ID and configuration"""
        self.session_id = session_id
        self.config = config
        
        # Get session directory from environment variable or use default
        self.session_dir = os.getenv("SESSION_DIR", os.path.join("logs", "sessions", session_id))
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Initialize log files
        self.trades_file = os.path.join(self.session_dir, "trades.csv")
        self.positions_file = os.path.join(self.session_dir, "positions.csv")
        self.performance_file = os.path.join(self.session_dir, "performance.csv")
        self.metadata_file = os.path.join(self.session_dir, "metadata.json")
        
        # Create CSV files with headers if they don't exist
        self._initialize_csv_files()
        
        # Save session metadata
        self._save_metadata()
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        # Trades file
        if not os.path.exists(self.trades_file):
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'order_id', 'asset', 'side', 'size',
                    'price', 'position_after', 'pnl', 'pnl_after'
                ])
        
        # Positions file
        if not os.path.exists(self.positions_file):
            with open(self.positions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'asset', 'size', 'value', 'entry_price'
                ])
        
        # Performance file
        if not os.path.exists(self.performance_file):
            with open(self.performance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'total_trades', 'winning_trades', 'losing_trades',
                    'win_rate', 'total_pnl', 'current_position', 'current_price'
                ])
    
    def _save_metadata(self):
        """Save session metadata"""
        metadata = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'config': self.config
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log a trade to the trades CSV file"""
        with open(self.trades_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'order_id', 'asset', 'side', 'size',
                'price', 'position_after', 'pnl', 'pnl_after'
            ])
            writer.writerow(trade_data)
    
    def log_position(self, position_data: Dict[str, Any]):
        """Log a position update to the positions CSV file"""
        with open(self.positions_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'asset', 'size', 'value', 'entry_price'
            ])
            writer.writerow(position_data)
    
    def log_performance(self, performance_data: Dict[str, Any]):
        """Log performance metrics to the performance CSV file"""
        with open(self.performance_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'total_trades', 'winning_trades', 'losing_trades',
                'win_rate', 'total_pnl', 'current_position', 'current_price'
            ])
            writer.writerow(performance_data)
    
    def update_metadata(self, key: str, value: Any):
        """Update session metadata"""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata[key] = value
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            print(f"Error updating metadata: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the trading session"""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Add end time if session is complete
            if 'end_time' not in metadata:
                metadata['end_time'] = datetime.now().isoformat()
            
            return metadata
        except Exception as e:
            print(f"Error getting session summary: {e}")
            return {} 