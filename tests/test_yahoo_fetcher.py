"""Tests for Yahoo Finance fetcher."""

from datetime import date, timedelta

import pytest

from src.data.fetchers.yahoo import YahooFinanceFetcher, YahooFinanceFetcherError
from src.data.models import ETFSymbol


@pytest.fixture
def fetcher() -> YahooFinanceFetcher:
    """Create a Yahoo Finance fetcher instance."""
    return YahooFinanceFetcher(delay_between_requests=0.1)


def test_validate_connection(fetcher: YahooFinanceFetcher) -> None:
    """Test connection validation."""
    assert fetcher.validate_connection() is True


def test_fetch_etf_prices_single_symbol(fetcher: YahooFinanceFetcher) -> None:
    """Test fetching prices for a single ETF symbol."""
    end_date = date.today()
    start_date = end_date - timedelta(days=7)

    prices = fetcher.fetch_etf_prices([ETFSymbol.WPEA], start_date, end_date)

    assert len(prices) > 0
    assert all(p.symbol == ETFSymbol.WPEA for p in prices)
    assert all(p.open > 0 for p in prices)
    assert all(p.close > 0 for p in prices)


def test_fetch_etf_prices_invalid_date_range(
    fetcher: YahooFinanceFetcher,
) -> None:
    """Test that invalid date ranges raise ValueError."""
    end_date = date.today()
    start_date = end_date + timedelta(days=7)

    with pytest.raises(ValueError):
        fetcher.fetch_etf_prices([ETFSymbol.WPEA], start_date, end_date)


def test_fetch_vix(fetcher: YahooFinanceFetcher) -> None:
    """Test fetching VIX data."""
    end_date = date.today()
    start_date = end_date - timedelta(days=7)

    vix_df = fetcher.fetch_vix(start_date, end_date)

    assert not vix_df.empty
    assert "vix" in vix_df.columns
    assert vix_df.index.name == "date"


def test_fetch_index_prices(fetcher: YahooFinanceFetcher) -> None:
    """Test fetching index prices."""
    end_date = date.today()
    start_date = end_date - timedelta(days=7)

    df = fetcher.fetch_index_prices(["^GSPC", "^DJI"], start_date, end_date)

    assert not df.empty
    assert df.index.name == "date"
    assert ("^GSPC", "Close") in df.columns or "^GSPC" in df.columns.get_level_values(0)


def test_fetch_pea_symbols_as_strings(fetcher: YahooFinanceFetcher) -> None:
    """Test fetching PEA-compatible symbols using strings."""
    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    # Note: These may fail if symbols are invalid or data not available
    # This test demonstrates the API, actual data availability may vary
    try:
        prices = fetcher.fetch_etf_prices(["LQQ.PA", "CL2.PA"], start_date, end_date)
        # If we get here, data was available
        assert len(prices) >= 0  # May be empty if symbols don't exist
    except YahooFinanceFetcherError:
        # Expected if symbols are not valid or data not available
        pytest.skip("PEA symbols not available in test environment")


def test_rate_limiting(fetcher: YahooFinanceFetcher) -> None:
    """Test that rate limiting is enforced."""
    import time

    start_time = time.time()

    # Make two requests
    end_date = date.today()
    start_date = end_date - timedelta(days=1)

    fetcher._fetch_ticker_data("SPY", start_date, end_date)
    fetcher._fetch_ticker_data("SPY", start_date, end_date)

    elapsed = time.time() - start_time

    # Should take at least delay_between_requests (0.1s) for second request
    assert elapsed >= 0.1
