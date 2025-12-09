import yaml
from typing import Dict, Any, List
import logging
from datetime import datetime, timedelta
import time
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the DataFetcher with configuration settings.
        
        Args:
            config_path (str): Path to the config.yaml file
        """
        self.config = self._load_config(config_path)
        self.timeframe = self.config['data']['timeframe']
        self.lookback_period = self.config['data']['lookback_period']
        self.sequence_length = self.config['data']['sequence_length']
        
        # Initialize Hyperliquid info client
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)  # Skip websocket connection

    def __del__(self):
        """
        Cleanup when the object is destroyed
        """
        if hasattr(self, 'info'):
            del self.info

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the config file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise

    def fetch_price_data(self) -> List[Dict[str, float]]:
        """
        Fetch price data from Hyperliquid API and calculate volatility.
        
        Returns:
            List[Dict[str, float]]: List of dictionaries containing timestamp and volatility data
        """
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(seconds=self.timeframe * self.lookback_period)
            
            # Time the API call
            api_start = time.time()
            historical_data: List[Dict] = self.info.candles_snapshot(
                self.config['universal']['asset'],
                "1m",
                int(start_time.timestamp() * 1000),
                int(end_time.timestamp() * 1000)
            )
            api_duration = time.time() - api_start
            
            logger.info(f"API call took {api_duration:.2f} seconds to fetch {len(historical_data)} candles")
            return historical_data
            
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            raise

    def fetch_latest_candle(self) -> Dict:
        """
        Fetch only the most recent closed 1m candle from Hyperliquid.
        Returns a single candle dictionary or None if failed.
        """
        try:
            # Align to the last closed minute
            now = int(time.time())
            last_minute = now - (now % 60)
            start_time = (last_minute - 60) * 1000  # previous minute
            end_time = last_minute * 1000           # last closed minute

            logger.debug(f"Fetching latest closed candle from {start_time} to {end_time}")
            response = self.info.candles_snapshot(
                self.config['universal']['asset'],
                "1m",
                start_time,
                end_time
            )
            if response and len(response) > 0:
                logger.debug(f"Fetched {len(response)} candles, returning most recent")
                return response[-1]
            logger.warning("No candles returned from API")
            return None
        except Exception as e:
            logger.error(f"Error fetching latest candle: {e}")
            return None