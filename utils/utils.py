import json
import os
from dotenv import load_dotenv
import eth_account
from eth_account.signers.local import LocalAccount

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

# Load environment variables from .env.local file
load_dotenv(dotenv_path='.env.local')

def setup(base_url=None, skip_ws=False, perp_dexs=None):
    """Setup accounts with smart key detection"""
    
    # Check for dual account keys first
    buy_secret_key = os.getenv("HYPERLIQUID_BUY_SECRET_KEY")
    sell_secret_key = os.getenv("HYPERLIQUID_SELL_SECRET_KEY")
    
    buy_account_address = os.getenv("HYPERLIQUID_BUY_ACCOUNT_ADDRESS")
    sell_account_address = os.getenv("HYPERLIQUID_SELL_ACCOUNT_ADDRESS")
    
    # Smart key detection logic
    if buy_secret_key and sell_secret_key:
        # Dual account mode - optimal for live trading
        print("Dual account mode: separate buy/sell accounts for live trading")
        mode = "dual"
    else:
        raise ValueError(
            "No valid keys found. Provide Both key pairs"
        )

    # Setup buy account
    buy_account: LocalAccount = eth_account.Account.from_key(buy_secret_key)
    if not buy_account_address:
        buy_account_address = buy_account.address
    print("Buy account address:", buy_account_address)
    
    # Setup sell account
    sell_account: LocalAccount = eth_account.Account.from_key(sell_secret_key)
    if not sell_account_address:
        sell_account_address = sell_account.address
    print("Sell account address:", sell_account_address)

    # Create info client (use buy account)
    info = Info(base_url, skip_ws, perp_dexs=perp_dexs)
    
    # Create exchanges
    buy_exchange = Exchange(buy_account, base_url, account_address=buy_account_address, perp_dexs=perp_dexs)
    sell_exchange = Exchange(sell_account, base_url, account_address=sell_account_address, perp_dexs=perp_dexs)
    
    # Return structured data
    accounts = {
        'buy': {
            'address': buy_account_address,
            'account': buy_account,
            'exchange': buy_exchange
        },
        'sell': {
            'address': sell_account_address,
            'account': sell_account,
            'exchange': sell_exchange
        },
        'mode': mode,
        'is_dual': mode == "dual"
    }
    
    return accounts, info