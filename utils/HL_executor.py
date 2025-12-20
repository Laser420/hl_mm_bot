"""
Order execution implementations for different trading modes
"""
import time
import logging
import aiohttp
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional

logger = logging.getLogger(__name__)

@dataclass
class Price:
    symbol: str
    price: float
    timestamp: int  # Unix timestamp

class HyperliquidExecutor():
    """Hyperliquid exchange implementation"""
    def __init__(self, api_base_url):
        self.api_base_url = api_base_url

    async def _make_request(self, endpoint: str, data: Dict[str, any] = None, signed: bool = False) -> Dict[str, any]:
        """Make HTTP request to Hyperliquid API"""
        url = f"{self.api_base_url}{endpoint}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = data if data else {}
        
        if signed and not self.account:
            raise ValueError("Private key required for signed requests")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"API request failed: {response.status} - {await response.text()}")
                return await response.json()
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price using the allMids endpoint"""
        try:
            data = {"type": "allMids"}
            response = await self._make_request("/info", data, signed=False)
            
            if symbol not in response:
                raise Exception(f"Symbol {symbol} not found in market data")
            
            return float(response[symbol])
            
           # return Price(
            ##    symbol=symbol,
           #     price=mid_price,
            #    timestamp=int(time.time())
           # )
        except Exception as e:
            raise Exception(f"Failed to get current price for {symbol}: {e}")
    
    def __init__(self, buy_exchange, sell_exchange, buy_address: str, sell_address: str, info):
        self.buy_exchange = buy_exchange
        self.sell_exchange = sell_exchange
        self.buy_address = buy_address
        self.sell_address = sell_address
        self.info = info
    
    async def place_buy_order(self, coin: str, size: float, price: float) -> Optional[str]:
        """Place buy order using buy account"""
        try:
            result = self.buy_exchange.order(
                coin,
                True,  # is_buy
                size,
                price,
                {"limit": {"tif": "Gtc"}}
            )
            
            if 'orderId' in result:
                logger.info(f"Placed buy order: {size} @ {price:.2f} (ID: {result['orderId']})")
                return result['orderId']
            else:
                logger.warning(f"Buy order failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to place buy order: {e}")
            return None
    
    async def place_sell_order(self, coin: str, size: float, price: float) -> Optional[str]:
        """Place sell order using sell account"""
        try:
            result = self.sell_exchange.order(
                coin,
                False,  # is_buy
                size,
                price,
                {"limit": {"tif": "Gtc"}}
            )
            
            if 'orderId' in result:
                logger.info(f"Placed sell order: {size} @ {price:.2f} (ID: {result['orderId']})")
                return result['orderId']
            else:
                logger.warning(f"Sell order failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to place sell order: {e}")
            return None
    
    async def place_market_buy_order(self, coin: str, size: float) -> Optional[str]:
        """Place market buy order using buy account"""
        try:
            result = self.buy_exchange.order(
                coin,
                True,  # is_buy
                size,
                None,  # No price for market orders
                {"market": {}}
            )
            
            if 'orderId' in result:
                logger.info(f"Placed market buy order: {size} (ID: {result['orderId']})")
                return result['orderId']
            else:
                logger.warning(f"Market buy order failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to place market buy order: {e}")
            return None
    
    async def place_market_sell_order(self, coin: str, size: float) -> Optional[str]:
        """Place market sell order using sell account"""
        try:
            result = self.sell_exchange.order(
                coin,
                False,  # is_buy
                size,
                None,  # No price for market orders
                {"market": {}}
            )
            
            if 'orderId' in result:
                logger.info(f"Placed market sell order: {size} (ID: {result['orderId']})")
                return result['orderId']
            else:
                logger.warning(f"Market sell order failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to place market sell order: {e}")
            return None
    
    async def place_stop_sell_order(self, coin: str, size: float, stop_price: float) -> Optional[str]:
        """Place stop-loss sell order using sell account"""
        try:
            result = self.sell_exchange.order(
                coin,
                False,  # is_buy = False (sell)
                size,
                stop_price,
                {"trigger": {"triggerPx": str(stop_price), "isMarket": True, "tpsl": "sl"}},
                reduce_only=True
            )
            
            if 'orderId' in result:
                logger.info(f"🛡️ Placed stop-sell order: {size} @ {stop_price:.2f} (ID: {result['orderId']})")
                return result['orderId']
            else:
                logger.warning(f"Stop-sell order failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to place stop-sell order: {e}")
            return None
    
    async def place_stop_buy_order(self, coin: str, size: float, stop_price: float) -> Optional[str]:
        """Place stop-loss buy order using buy account"""
        try:
            result = self.buy_exchange.order(
                coin,
                True,  # is_buy = True
                size,
                stop_price,
                {"trigger": {"triggerPx": str(stop_price), "isMarket": True, "tpsl": "sl"}},
                reduce_only=True
            )
            
            if 'orderId' in result:
                logger.info(f"🛡️ Placed stop-buy order: {size} @ {stop_price:.2f} (ID: {result['orderId']})")
                return result['orderId']
            else:
                logger.warning(f"Stop-buy order failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to place stop-buy order: {e}")
            return None
    
    async def place_take_profit_sell_order(self, coin: str, size: float, tp_price: float) -> Optional[str]:
        """Place take-profit sell order using sell account"""
        try:
            result = self.sell_exchange.order(
                coin,
                False,  # is_buy = False (sell)
                size,
                tp_price,
                {"trigger": {"triggerPx": str(tp_price), "isMarket": True, "tpsl": "tp"}},
                reduce_only=True
            )
            
            if 'orderId' in result:
                logger.info(f"🎯 Placed take-profit sell order: {size} @ {tp_price:.2f} (ID: {result['orderId']})")
                return result['orderId']
            else:
                logger.warning(f"Take-profit sell order failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to place take-profit sell order: {e}")
            return None
    
    async def place_take_profit_buy_order(self, coin: str, size: float, tp_price: float) -> Optional[str]:
        """Place take-profit buy order using buy account"""
        try:
            result = self.buy_exchange.order(
                coin,
                True,  # is_buy = True
                size,
                tp_price,
                {"trigger": {"triggerPx": str(tp_price), "isMarket": True, "tpsl": "tp"}},
                reduce_only=True
            )
            
            if 'orderId' in result:
                logger.info(f"🎯 Placed take-profit buy order: {size} @ {tp_price:.2f} (ID: {result['orderId']})")
                return result['orderId']
            else:
                logger.warning(f"Take-profit buy order failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to place take-profit buy order: {e}")
            return None
    
    async def cancel_order(self, coin: str, order_id: str, is_buy: bool) -> bool:
        """Cancel order using appropriate account"""
        try:
            exchange = self.buy_exchange if is_buy else self.sell_exchange
            result = exchange.cancel(coin, order_id)
            logger.info(f"Canceled {'buy' if is_buy else 'sell'} order {order_id}: {result}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_position(self, coin: str) -> float:
        """Get current position from both accounts"""
        try:
            # Get position from buy account
            buy_state = self.info.user_state(self.buy_address)
            buy_positions = buy_state.get('assetPositions', [])
            
            # Get position from sell account  
            sell_state = self.info.user_state(self.sell_address)
            sell_positions = sell_state.get('assetPositions', [])
            
            total_position = 0.0
            
            # Sum positions from both accounts
            for pos in buy_positions:
                if pos['position']['coin'] == coin:
                    total_position += float(pos['position']['szi'])
            
            for pos in sell_positions:
                if pos['position']['coin'] == coin:
                    total_position += float(pos['position']['szi'])
            
            return total_position
            
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return 0.0
    
    def check_order_status(self, order_id: str, is_buy: bool) -> Dict:
        """Check the status of a specific order"""
        try:
            # Get open orders from appropriate account
            account_address = self.buy_address if is_buy else self.sell_address
            open_orders = self.info.open_orders(account_address)
            
            # Find our order
            for order in open_orders:
                if str(order['oid']) == str(order_id):
                    return {
                        'status': 'open',
                        'filled_size': 0.0,
                        'remaining_size': float(order['sz']),
                        'order': order
                    }
            
            # If not in open orders, check fills
            fills = self.info.user_fills(account_address)
            for fill in fills:
                if str(fill['oid']) == str(order_id):
                    return {
                        'status': 'filled',
                        'filled_size': float(fill['sz']),
                        'remaining_size': 0.0,
                        'fill_price': float(fill['px']),
                        'fee': float(fill['fee']),
                        'fill': fill
                    }
            
            # Order not found - likely cancelled or very old
            return {
                'status': 'cancelled',
                'filled_size': 0.0,
                'remaining_size': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error checking order status for {order_id}: {e}")
            return {'status': 'error', 'filled_size': 0.0, 'remaining_size': 0.0}
    
    def check_stop_order_status(self, order_id: str, is_buy: bool) -> Dict:
        """Check the status of a stop-loss order (same logic as regular orders)"""
        return self.check_order_status(order_id, is_buy)