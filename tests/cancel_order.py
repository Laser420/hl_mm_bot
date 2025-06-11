import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import setup
from hyperliquid.utils import constants

if __name__ == "__main__":
    # Get order ID from command line or prompt
    if len(sys.argv) > 1:
        order_id = int(sys.argv[1])
    else:
        order_id = int(input("Enter the order ID to cancel: "))

    # Setup exchange
    account_address, info, exchange = setup(base_url=constants.MAINNET_API_URL, skip_ws=True)
    print(f"Using account: {account_address}")
    print(f"Attempting to cancel order {order_id} for BTC...")
    try:
        result = exchange.cancel("BTC", order_id)
        print(f"Cancel result: {result}")
    except Exception as e:
        print(f"Error canceling order: {e}") 