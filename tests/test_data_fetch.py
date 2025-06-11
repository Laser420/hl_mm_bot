from utils.data_fetcher import DataFetcher
import time
import sys

def main():
    start_time = time.time()
    fetcher = None
    
    try:
        # Initialize the data fetcher
        fetcher = DataFetcher()
        
        # Fetch the data
        candle_data = fetcher.fetch_price_data()
        
        duration = time.time() - start_time
        print(f"Total execution time: {duration:.2f} seconds")
        print(f"Fetched {len(candle_data)} candles")
        
        return len(candle_data)
    finally:
        # Cleanup
        if fetcher:
            del fetcher
        sys.exit(0)  # Force exit

if __name__ == "__main__":
    main() 