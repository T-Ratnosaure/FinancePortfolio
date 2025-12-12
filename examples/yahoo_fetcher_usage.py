"""Example usage of Yahoo Finance fetcher for PEA Portfolio system.

This script demonstrates how to use the YahooFinanceFetcher to retrieve
market data for PEA-compatible ETFs and other financial instruments.
"""

from datetime import date, timedelta

import pandas as pd

from src.data.fetchers.yahoo import YahooFinanceFetcher


def main() -> None:
    """Demonstrate Yahoo Finance fetcher usage."""
    fetcher = YahooFinanceFetcher(
        delay_between_requests=0.5,
        max_retries=3,
    )

    print("Validating Yahoo Finance connection...")
    if fetcher.validate_connection():
        print("Connection successful!")
    else:
        print("Connection failed!")
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    print(f"Fetching data from {start_date} to {end_date}")

    # Example 1: Fetch PEA ETF prices
    pea_symbols = ["LQQ.PA", "CL2.PA", "WPEA.PA"]
    try:
        prices = fetcher.fetch_etf_prices(pea_symbols, start_date, end_date)
        print(f"Fetched {len(prices)} price records")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Fetch using ETFSymbol enum values directly
    from src.data.models import ETFSymbol

    enum_symbols: list[ETFSymbol | str] = [ETFSymbol.CL2, ETFSymbol.WPEA]
    try:
        prices = fetcher.fetch_etf_prices(enum_symbols, start_date, end_date)
        print(f"Fetched {len(prices)} price records using enum symbols")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Fetch VIX
    try:
        vix_df = fetcher.fetch_vix(start_date, end_date)
        print(f"Fetched VIX data: {len(vix_df)} records")
    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Fetch index prices
    indices = ["^GSPC", "^DJI", "^IXIC"]
    try:
        index_df = fetcher.fetch_index_prices(indices, start_date, end_date)
        print(f"Fetched index data: {len(index_df)} records")
        if not index_df.empty:
            if isinstance(index_df.columns, pd.MultiIndex):
                available = list(index_df.columns.get_level_values(0).unique())
            else:
                available = list(index_df.columns)
            print(f"Available indices: {available}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 5: Batch download of multiple PEA ETFs
    batch_symbols = ["LQQ.PA", "CL2.PA", "WPEA.PA"]
    try:
        batch_df = fetcher.fetch_multiple_symbols(batch_symbols, start_date, end_date)
        print(f"Batch download complete: {len(batch_df)} records")
        if isinstance(batch_df.columns, pd.MultiIndex):
            symbols = list(batch_df.columns.get_level_values(0).unique())
        else:
            symbols = list(batch_df.columns)
        print(f"PEA ETFs downloaded: {symbols}")
    except Exception as e:
        print(f"Error: {e}")

    print("Examples complete!")


if __name__ == "__main__":
    main()
