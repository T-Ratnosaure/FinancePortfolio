"""FRED (Federal Reserve Economic Data) fetcher for macroeconomic indicators."""

import logging
import os
import time
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.data.fetchers.base import (
    BaseFetcher,
    DataNotAvailableError,
    FetchError,
    RateLimitError,
)
from src.data.models import MacroIndicator

logger = logging.getLogger(__name__)


class FREDFetcherError(FetchError):
    """Custom exception for FRED fetcher errors."""

    pass


class FREDFetcher(BaseFetcher):
    """Fetcher for FRED macroeconomic data.

    This fetcher retrieves economic indicators from the Federal Reserve Economic
    Database (FRED) API. Requires a FRED API key from:
    https://fred.stlouisfed.org/docs/api/api_key.html

    The API key should be set in the FRED_API_KEY environment variable.

    FRED series codes used:
    - VIXCLS: CBOE Volatility Index (VIX)
    - DGS10: 10-Year Treasury Constant Maturity Rate
    - DGS2: 2-Year Treasury Constant Maturity Rate
    - T10Y2Y: 10-Year Treasury Minus 2-Year Treasury (2s10s spread)
    - BAMLH0A0HYM2: ICE BofA High Yield Option-Adjusted Spread

    Rate limiting: FRED API has a limit of 120 requests per minute. This
    implementation includes exponential backoff and configurable delays
    between requests.
    """

    SERIES_VIX = "VIXCLS"
    SERIES_TREASURY_10Y = "DGS10"
    SERIES_TREASURY_2Y = "DGS2"
    SERIES_YIELD_SPREAD_2S10S = "T10Y2Y"
    SERIES_HY_SPREAD = "BAMLH0A0HYM2"

    def __init__(
        self,
        api_key: str | None = None,
        delay_between_requests: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        """Initialize the FRED fetcher.

        Args:
            api_key: FRED API key. If None, will be loaded from FRED_API_KEY
                environment variable.
            delay_between_requests: Delay in seconds between API requests
                to respect rate limits (default: 0.5s)
            max_retries: Maximum number of retry attempts for failed
                requests (default: 3)

        Raises:
            FREDFetcherError: If API key is not provided or found in environment
        """
        load_dotenv()

        self._api_key = api_key or os.getenv("FRED_API_KEY")

        if not self._api_key:
            raise FREDFetcherError(
                "FRED API key not found. Please set FRED_API_KEY environment "
                "variable or pass api_key parameter. Get a free API key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html",
                source="FRED",
            )

        self._delay = delay_between_requests
        self._max_retries = max_retries
        self._last_request_time: float = 0.0

        try:
            self._client = Fred(api_key=self._api_key)
        except Exception as e:
            raise FREDFetcherError(
                f"Failed to initialize FRED client: {e}", source="FRED"
            ) from e

    def validate_connection(self) -> bool:
        """Validate that the connection to FRED is working.

        Returns:
            True if connection is valid, False otherwise.
        """
        try:
            self._client.get_series(self.SERIES_VIX, limit=1)
            return True
        except Exception:
            return False

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time

        if time_since_last_request < self._delay:
            time.sleep(self._delay - time_since_last_request)

        self._last_request_time = time.time()

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _fetch_series(
        self, series_id: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Fetch a single FRED series with retry logic.

        Args:
            series_id: FRED series identifier
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            DataFrame with date index and single value column

        Raises:
            FREDFetcherError: If fetch fails
            RateLimitError: If rate limit is exceeded
            DataNotAvailableError: If no data is available for the series
        """
        self._validate_date_range(start_date, end_date)
        self._rate_limit()

        try:
            series = self._client.get_series(
                series_id,
                observation_start=start_date.isoformat(),
                observation_end=end_date.isoformat(),
            )

            if series is None or series.empty:
                raise DataNotAvailableError(
                    f"No data returned for series {series_id}",
                    source="FRED",
                )

            # Create DataFrame from series
            df = pd.DataFrame({series_id: series})
            df.index.name = "date"

            return df

        except Exception as e:
            if isinstance(e, (FetchError, RateLimitError, DataNotAvailableError)):
                raise

            # Check if it's a rate limit issue
            error_msg = str(e).lower()
            if (
                "429" in error_msg
                or "rate limit" in error_msg
                or "too many requests" in error_msg
            ):
                raise RateLimitError(
                    f"Rate limit exceeded for series {series_id}", source="FRED"
                ) from e

            # Check for network issues
            if "connection" in error_msg or "timeout" in error_msg:
                raise RateLimitError(
                    f"Network error for series {series_id}: {e!s}", source="FRED"
                ) from e

            # Check for API unavailability
            if "503" in error_msg or "500" in error_msg or "unavailable" in error_msg:
                raise RateLimitError(
                    f"FRED API temporarily unavailable for series {series_id}",
                    source="FRED",
                ) from e

            raise FREDFetcherError(
                f"Failed to fetch series {series_id}: {e!s}", source="FRED"
            ) from e

    def fetch_vix(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch VIX (CBOE Volatility Index) data.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            DataFrame with date index and VIX values

        Raises:
            FREDFetcherError: If fetch fails
        """
        try:
            df = self._fetch_series(self.SERIES_VIX, start_date, end_date)
            df.columns = ["vix"]
            return df
        except RetryError as e:
            raise FREDFetcherError(
                "Max retries exceeded for VIX data", source="FRED"
            ) from e

    def fetch_treasury_yields(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch US Treasury yield curve data.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            DataFrame with date index and columns: treasury_2y, treasury_10y

        Raises:
            FREDFetcherError: If fetch fails
        """
        try:
            df_2y = self._fetch_series(self.SERIES_TREASURY_2Y, start_date, end_date)
            df_10y = self._fetch_series(self.SERIES_TREASURY_10Y, start_date, end_date)

            df = pd.concat([df_2y, df_10y], axis=1)
            df.columns = ["treasury_2y", "treasury_10y"]

            return df
        except RetryError as e:
            raise FREDFetcherError(
                "Max retries exceeded for treasury yield data", source="FRED"
            ) from e

    def fetch_credit_spreads(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch credit spread indicators.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            DataFrame with date index and columns: spread_2s10s, hy_oas_spread

        Raises:
            FREDFetcherError: If fetch fails
        """
        try:
            df_2s10s = self._fetch_series(
                self.SERIES_YIELD_SPREAD_2S10S, start_date, end_date
            )
            df_hy = self._fetch_series(self.SERIES_HY_SPREAD, start_date, end_date)

            df = pd.concat([df_2s10s, df_hy], axis=1)
            df.columns = ["spread_2s10s", "hy_oas_spread"]

            return df
        except RetryError as e:
            raise FREDFetcherError(
                "Max retries exceeded for credit spread data", source="FRED"
            ) from e

    def _create_indicator(
        self, indicator_name: str, current_date: date, value: float
    ) -> MacroIndicator:
        """Create a MacroIndicator model.

        Args:
            indicator_name: Name of the indicator
            current_date: Date of the observation
            value: Value of the indicator

        Returns:
            MacroIndicator model
        """
        return MacroIndicator(
            indicator_name=indicator_name,
            date=current_date,
            value=value,
            source="FRED",
        )

    def fetch_macro_indicators(
        self, start_date: date, end_date: date
    ) -> list[MacroIndicator]:
        """Fetch all macro indicators as Pydantic models.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            List of MacroIndicator models

        Raises:
            FREDFetcherError: If fetch fails
        """
        indicators: list[MacroIndicator] = []

        try:
            vix_df = self.fetch_vix(start_date, end_date)
            treasury_df = self.fetch_treasury_yields(start_date, end_date)
            spreads_df = self.fetch_credit_spreads(start_date, end_date)

            combined_df = pd.concat([vix_df, treasury_df, spreads_df], axis=1)

            # Mapping of column names to indicator names
            indicator_map = {
                "vix": "VIX",
                "treasury_2y": "TREASURY_2Y",
                "treasury_10y": "TREASURY_10Y",
                "spread_2s10s": "SPREAD_2S10S",
                "hy_oas_spread": "HY_OAS_SPREAD",
            }

            for date_idx, row in combined_df.iterrows():
                if not isinstance(date_idx, pd.Timestamp):
                    continue

                current_date = date_idx.date()

                # Create indicators for all non-null values
                for col_name, indicator_name in indicator_map.items():
                    if pd.notna(row[col_name]):
                        indicators.append(
                            self._create_indicator(
                                indicator_name, current_date, float(row[col_name])
                            )
                        )

        except Exception as e:
            if isinstance(e, (FREDFetcherError, RetryError)):
                if isinstance(e, RetryError):
                    raise FREDFetcherError(
                        "Max retries exceeded for macro indicators", source="FRED"
                    ) from e
                raise
            raise FREDFetcherError(
                f"Failed to fetch macro indicators: {e}", source="FRED"
            ) from e

        return indicators
