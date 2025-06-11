import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import setup

from hyperliquid.utils import constants

async def place_test_order():
    try:
        # Setup exchange using utils with mainnet URL
        address, info, exchange = setup(base_url=constants.MAINNET_API_URL, skip_ws=True)
        
        # Define order parameters
        symbol = "BTC"  # Hyperliquid uses "BTC" not "BTC-PERP"
        side = True  # True for buy, False for sell
        amount = 0.002  # Amount in BTC
        price = 50000.0  # Price in USD
        leverage = 50  # 50x leverage
        
        print(f"Placing order for {amount} BTC at ${price} with {leverage}x leverage")
        print(f"Total position value: ${amount * price * leverage}")

        # Place the order using the correct method
        order_result = exchange.order(
            symbol,
            side,
            amount,
            price,
            {
                "limit": {
                    "tif": "Gtc"
                },
                "leverage": leverage
            }
        )
        print("Order placed successfully:", order_result)
    except Exception as e:
        print("Error placing order:", str(e))

if __name__ == "__main__":
    import asyncio
    asyncio.run(place_test_order())