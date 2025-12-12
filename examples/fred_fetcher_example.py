"""Example usage of the FRED data fetcher.

Before running this example:
1. Get a free FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html
2. Set the FRED_API_KEY environment variable or create a .env file with:
   FRED_API_KEY=your_api_key_here
"""

from datetime import date, timedelta

from src.data.fetchers.fred import FREDFetcher


def main() -> None:
    """Demonstrate FRED fetcher usage."""
    print("FRED Data Fetcher Example")
    print("=" * 60)

    # Initialize the fetcher (API key loaded from environment)
    try:
        fetcher = FREDFetcher()
        print("[OK] FRED fetcher initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize FRED fetcher: {e}")
        print("\nPlease set FRED_API_KEY environment variable")
        print("Get a free API key at:")
        print("https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    # Test connection
    if fetcher.validate_connection():
        print("[OK] Connection to FRED validated")
    else:
        print("[ERROR] Failed to connect to FRED")
        return

    # Set date range for data fetch (last 30 days)
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    print(f"\nFetching data from {start_date} to {end_date}")
    print("-" * 60)

    # Fetch VIX data
    print("\n1. Fetching VIX (Volatility Index)...")
    try:
        vix_df = fetcher.fetch_vix(start_date, end_date)
        print(f"   [OK] Retrieved {len(vix_df)} VIX observations")
        print(f"   Latest VIX: {vix_df['vix'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"   [ERROR] {e}")

    # Fetch Treasury yields
    print("\n2. Fetching Treasury Yields...")
    try:
        treasury_df = fetcher.fetch_treasury_yields(start_date, end_date)
        print(f"   [OK] Retrieved {len(treasury_df)} Treasury observations")
        latest_2y = treasury_df["treasury_2y"].iloc[-1]
        latest_10y = treasury_df["treasury_10y"].iloc[-1]
        print(f"   Latest 2Y: {latest_2y:.2f}%")
        print(f"   Latest 10Y: {latest_10y:.2f}%")
    except Exception as e:
        print(f"   [ERROR] {e}")

    # Fetch credit spreads
    print("\n3. Fetching Credit Spreads...")
    try:
        spreads_df = fetcher.fetch_credit_spreads(start_date, end_date)
        print(f"   [OK] Retrieved {len(spreads_df)} spread observations")
        latest_2s10s = spreads_df["spread_2s10s"].iloc[-1]
        latest_hy = spreads_df["hy_oas_spread"].iloc[-1]
        print(f"   Latest 2s10s spread: {latest_2s10s:.2f} bps")
        print(f"   Latest HY OAS spread: {latest_hy:.2f} bps")
    except Exception as e:
        print(f"   [ERROR] {e}")

    # Fetch all indicators as Pydantic models
    print("\n4. Fetching All Macro Indicators (Pydantic Models)...")
    try:
        indicators = fetcher.fetch_macro_indicators(start_date, end_date)
        print(f"   [OK] Retrieved {len(indicators)} indicator observations")

        # Group by indicator name
        by_name = {}
        for ind in indicators:
            if ind.indicator_name not in by_name:
                by_name[ind.indicator_name] = []
            by_name[ind.indicator_name].append(ind)

        print("\n   Indicator Summary:")
        for name, obs in by_name.items():
            print(f"   - {name}: {len(obs)} observations")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
