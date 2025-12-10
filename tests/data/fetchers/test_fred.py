"""Tests for FRED data fetcher."""

import os
from datetime import date, timedelta

import pandas as pd
import pytest

from src.data.fetchers.fred import FREDFetcher, FREDFetcherError
from src.data.models import MacroIndicator


class TestFREDFetcher:
    """Tests for FREDFetcher class."""

    def test_init_without_api_key_raises_error(self):
        """Test that initialization fails without an API key."""
        # Temporarily remove API key from environment
        old_key = os.environ.get("FRED_API_KEY")
        if old_key:
            del os.environ["FRED_API_KEY"]

        with pytest.raises(FREDFetcherError, match="FRED API key not found"):
            FREDFetcher()

        # Restore old key if it existed
        if old_key:
            os.environ["FRED_API_KEY"] = old_key

    def test_init_with_api_key_succeeds(self):
        """Test that initialization succeeds with an API key."""
        # Skip if no API key available
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            pytest.skip("FRED_API_KEY not set in environment")

        fetcher = FREDFetcher(api_key=api_key)
        assert fetcher is not None

    def test_validate_connection(self):
        """Test connection validation."""
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            pytest.skip("FRED_API_KEY not set in environment")

        fetcher = FREDFetcher(api_key=api_key)
        assert fetcher.validate_connection() is True

    def test_fetch_vix(self):
        """Test fetching VIX data."""
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            pytest.skip("FRED_API_KEY not set in environment")

        fetcher = FREDFetcher(api_key=api_key)
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        df = fetcher.fetch_vix(start_date, end_date)

        assert isinstance(df, pd.DataFrame)
        assert "vix" in df.columns
        assert len(df) > 0

    def test_fetch_treasury_yields(self):
        """Test fetching Treasury yield data."""
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            pytest.skip("FRED_API_KEY not set in environment")

        fetcher = FREDFetcher(api_key=api_key)
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        df = fetcher.fetch_treasury_yields(start_date, end_date)

        assert isinstance(df, pd.DataFrame)
        assert "treasury_2y" in df.columns
        assert "treasury_10y" in df.columns
        assert len(df) > 0

    def test_fetch_credit_spreads(self):
        """Test fetching credit spread data."""
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            pytest.skip("FRED_API_KEY not set in environment")

        fetcher = FREDFetcher(api_key=api_key)
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        df = fetcher.fetch_credit_spreads(start_date, end_date)

        assert isinstance(df, pd.DataFrame)
        assert "spread_2s10s" in df.columns
        assert "hy_oas_spread" in df.columns
        assert len(df) > 0

    def test_fetch_macro_indicators(self):
        """Test fetching all macro indicators as Pydantic models."""
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            pytest.skip("FRED_API_KEY not set in environment")

        fetcher = FREDFetcher(api_key=api_key)
        end_date = date.today()
        start_date = end_date - timedelta(days=10)

        indicators = fetcher.fetch_macro_indicators(start_date, end_date)

        assert isinstance(indicators, list)
        assert len(indicators) > 0
        assert all(isinstance(ind, MacroIndicator) for ind in indicators)

        # Check that we have all expected indicator types
        indicator_names = {ind.indicator_name for ind in indicators}
        expected_names = {
            "VIX",
            "TREASURY_2Y",
            "TREASURY_10Y",
            "SPREAD_2S10S",
            "HY_OAS_SPREAD",
        }
        # At least some of the expected indicators should be present
        assert len(indicator_names & expected_names) > 0

    def test_invalid_date_range_raises_error(self):
        """Test that invalid date ranges raise an error."""
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            pytest.skip("FRED_API_KEY not set in environment")

        fetcher = FREDFetcher(api_key=api_key)
        end_date = date(2024, 1, 1)
        start_date = date(2024, 1, 31)

        with pytest.raises(ValueError, match="must be before"):
            fetcher.fetch_vix(start_date, end_date)
