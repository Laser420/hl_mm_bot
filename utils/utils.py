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
    # Retrieve secret key and account address from environment variables
    secret_key = os.getenv("HYPERLIQUID_SECRET_KEY")
    account_address = os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS")

    if not secret_key:
        raise ValueError("HYPERLIQUID_SECRET_KEY must be set in the .env.local file.")

    account: LocalAccount = eth_account.Account.from_key(secret_key)
    if not account_address:
        account_address = account.address
    print("Running with account address:", account_address)
    if account_address != account.address:
        print("Running with agent address:", account.address)

    info = Info(base_url, skip_ws, perp_dexs=perp_dexs)
    user_state = info.user_state(account_address)
    spot_user_state = info.spot_user_state(account_address)
    margin_summary = user_state["marginSummary"]
    if float(margin_summary["accountValue"]) == 0 and len(spot_user_state["balances"]) == 0:
        print("Not running the example because the provided account has no equity.")
        url = info.base_url.split(".", 1)[1]
        error_string = f"No accountValue:\nIf you think this is a mistake, make sure that {account_address} has a balance on {url}.\nIf address shown is your API wallet address, update the config to specify the address of your account, not the address of the API wallet."
        raise Exception(error_string)
    exchange = Exchange(account, base_url, account_address=account_address, perp_dexs=perp_dexs)
    return account_address, info, exchange