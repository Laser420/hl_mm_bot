from hyperliquid.exchange import Exchange

def test_sdk_import():
    print("Successfully imported Hyperliquid SDK")
    print(f"Exchange class available: {Exchange is not None}")

if __name__ == "__main__":
    test_sdk_import() 