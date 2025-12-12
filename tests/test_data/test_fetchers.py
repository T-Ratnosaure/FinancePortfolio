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

    def test_rate_limit_enforcement(self) -> None:
        """Test that rate limiting is enforced between requests."""
        import time

        from src.data.fetchers.fred import FREDFetcher

        # Create fetcher with very short delay for testing
        fetcher = FREDFetcher(api_key="test_key", delay_between_requests=0.1)

        # First call should set the timestamp
        fetcher._rate_limit()

        # Second call should be delayed
        start_time = time.time()
        fetcher._rate_limit()
        second_call_time = time.time() - start_time

        # Second call should take at least the delay time
        assert second_call_time >= 0.1

    def test_retry_configuration(self) -> None:
        """Test that retry configuration is stored correctly."""
        from src.data.fetchers.fred import FREDFetcher

        fetcher = FREDFetcher(
            api_key="test_key",
            delay_between_requests=0.3,
            max_retries=5,
        )

        assert fetcher._delay == 0.3
        assert fetcher._max_retries == 5

    @pytest.mark.skip(reason="Requires FRED API key and network")
    def test_retry_on_rate_limit(self) -> None:
        """Test that retry logic works for rate limit errors."""
        from datetime import date
        from unittest.mock import MagicMock, patch

        from src.data.fetchers.fred import FREDFetcher

        fetcher = FREDFetcher()

        # Mock the client to raise rate limit error first, then succeed
        with patch.object(fetcher._client, "get_series") as mock_get:
            mock_get.side_effect = [
                Exception("429 Too Many Requests"),
                MagicMock(empty=False, __iter__=lambda x: iter([1, 2, 3])),
            ]

            # Should succeed after retry
            result = fetcher._fetch_series(
                "VIXCLS", date(2024, 1, 1), date(2024, 1, 31)
            )
            assert result is not None

    @pytest.mark.skip(reason="Requires FRED API key and network")
    def test_data_not_available_error(self) -> None:
        """Test that DataNotAvailableError is raised for empty data."""
        from datetime import date
        from unittest.mock import patch

        import pandas as pd

        from src.data.fetchers.fred import FREDFetcher

        fetcher = FREDFetcher()

        # Mock the client to return empty series
        with patch.object(fetcher._client, "get_series") as mock_get:
            mock_get.return_value = pd.Series([], dtype=float)

            with pytest.raises(DataNotAvailableError, match="No data returned"):
                fetcher._fetch_series(
                    "INVALID_SERIES", date(2024, 1, 1), date(2024, 1, 31)
                )
