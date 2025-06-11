import logging
import time
from hyperliquid.utils import constants
from utils.utils import setup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_order_placement():
    # Initialize Hyperliquid clients
    account_address, info, exchange = setup(
        base_url=constants.MAINNET_API_URL,
        skip_ws=True
    )
    
    logger.info(f"Testing orders for account: {account_address}")
    
    try:
        # First check if we have any existing orders
        logger.info("Checking for existing orders...")
        open_orders = info.open_orders("BTC")
        if open_orders:
            logger.info(f"Found {len(open_orders)} existing orders. Canceling them...")
            for order in open_orders:
                try:
                    exchange.cancel("BTC", order['oid'])
                    logger.info(f"Canceled order {order['oid']}")
                except Exception as e:
                    logger.error(f"Error canceling order {order['oid']}: {e}")
        
        # Get current market price
        market_data = info.l2_snapshot("BTC")
        if not market_data:
            raise Exception("Could not fetch market data")
        
        current_price = float(market_data['levels'][0]['px'])
        logger.info(f"Current market price: {current_price}")
        
        # Place a small buy order slightly below market price
        test_price = current_price * 0.99  # 1% below market
        test_size = 0.001  # Very small size for testing
        
        logger.info(f"Placing test buy order at {test_price} for {test_size} BTC...")
        order_result = exchange.order(
            "BTC",
            True,  # is_buy
            test_size,
            test_price,
            {
                "limit": {
                    "tif": "Gtc",
                    "leverage": 1
                }
            }
        )
        
        logger.info(f"Order placed successfully: {order_result}")
        order_id = order_result.get('orderId')
        
        if not order_id:
            raise Exception("No order ID returned from order placement")
        
        # Wait a few seconds to see the order on the website
        logger.info("Waiting 5 seconds to verify order on website...")
        time.sleep(5)
        
        # Verify the order exists
        open_orders = info.open_orders("BTC")
        order_exists = any(order['oid'] == order_id for order in open_orders)
        logger.info(f"Order {order_id} exists: {order_exists}")
        
        # Cancel the order
        logger.info(f"Canceling order {order_id}...")
        cancel_result = exchange.cancel("BTC", order_id)
        logger.info(f"Cancel result: {cancel_result}")
        
    except Exception as e:
        logger.error(f"Error during order test: {e}")
        raise

if __name__ == "__main__":
    test_order_placement() 