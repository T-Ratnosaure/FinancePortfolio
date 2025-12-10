"""Tests for data fetchers."""

from datetime import date

import pytest

from src.data.fetchers.base import (
    BaseFetcher,
    DataNotAvailableError,
    FetchError,
    RateLimitError,
)


class TestBaseErrors:
    """Tests for base fetcher error classes."""

    def test_fetch_error_with_source(self) -> None:
        """Test FetchError with source."""
        error = FetchError("Something went wrong", source="Yahoo Finance")
        assert "[Yahoo Finance]" in str(error)
        assert "Something went wrong" in str(error)

    def test_fetch_error_without_source(self) -> None:
        """Test FetchError without source."""
        error = FetchError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_rate_limit_error(self) -> None:
        """Test RateLimitError inherits from FetchError."""
        error = RateLimitError("Rate limit exceeded", source="Yahoo")
        assert isinstance(error, FetchError)
        assert "[Yahoo]" in str(error)

    def test_data_not_available_error(self) -> None:
        """Test DataNotAvailableError inherits from FetchError."""
        error = DataNotAvailableError("No data for symbol", source="FRED")
        assert isinstance(error, FetchError)


class TestBaseFetcherDateValidation:
    """Tests for date validation in base fetcher."""

    class ConcreteFetcher(BaseFetcher):
        """Concrete implementation for testing."""

        def __init__(self) -> None:
            pass

        def validate_connection(self) -> bool:
            return True

    def test_valid_date_range(self) -> None:
        """Test valid date range passes validation."""
        fetcher = self.ConcreteFetcher()
        # Should not raise
        fetcher._validate_date_range(date(2024, 1, 1), date(2024, 12, 31))

    def test_invalid_date_range(self) -> None:
        """Test invalid date range raises ValueError."""
        fetcher = self.ConcreteFetcher()
        with pytest.raises(ValueError, match="must be before"):
            fetcher._validate_date_range(date(2024, 12, 31), date(2024, 1, 1))


class TestYahooFinanceFetcher:
    """Tests for Yahoo Finance fetcher (requires network)."""

    @pytest.mark.skip(reason="Requires network access")
    def test_validate_connection(self) -> None:
        """Test connection validation."""
        from src.data.fetchers.yahoo import YahooFinanceFetcher

        fetcher = YahooFinanceFetcher()
        assert fetcher.validate_connection() is True

    @pytest.mark.skip(reason="Requires network access")
    def test_fetch_vix(self) -> None:
        """Test fetching VIX data."""
        from src.data.fetchers.yahoo import YahooFinanceFetcher

        fetcher = YahooFinanceFetcher()
        vix_df = fetcher.fetch_vix(date(2024, 1, 1), date(2024, 1, 31))
        assert "vix" in vix_df.columns
        assert not vix_df.empty


class TestFREDFetcher:
    """Tests for FRED fetcher (requires API key)."""

    @pytest.mark.skip(reason="Requires FRED API key")
    def test_validate_connection(self) -> None:
        """Test connection validation."""
        from src.data.fetchers.fred import FREDFetcher

        fetcher = FREDFetcher()
        assert fetcher.validate_connection() is True

    def test_missing_api_key_raises_error(self) -> None:
        """Test that missing API key raises error."""
        import os

        from src.data.fetchers.fred import FREDFetcher, FREDFetcherError

        # Temporarily remove API key from environment
        original = os.environ.pop("FRED_API_KEY", None)
        try:
            with pytest.raises(FREDFetcherError, match="API key not found"):
                FREDFetcher()
        finally:
            if original:
                os.environ["FRED_API_KEY"] = original
