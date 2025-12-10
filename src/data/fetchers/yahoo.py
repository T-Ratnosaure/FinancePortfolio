"""Yahoo Finance fetcher for market data."""

import time
from datetime import date, datetime
from typing import Any

import pandas as pd
import yfinance as yf
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
from src.data.models import DailyPrice, ETFSymbol


class YahooFinanceFetcherError(FetchError):
    """Custom exception for Yahoo Finance fetcher errors."""

    pass


class YahooFinanceFetcher(BaseFetcher):
    """Fetcher for Yahoo Finance market data.

    This fetcher retrieves market data from Yahoo Finance using the yfinance
    library. It includes retry logic for handling transient failures and
    rate limiting considerations.

    Supports PEA-compatible ETFs:
    - LQQ.PA: Lyxor Nasdaq-100 2x Leveraged
    - CL2.PA: Lyxor MSCI USA 2x Leveraged
    - WPEA.PA: World PEA ETF

    Rate limiting: Yahoo Finance has undocumented rate limits. This
    implementation includes exponential backoff and configurable delays
    between requests.
    """

    def __init__(
        self, delay_between_requests: float = 0.5, max_retries: int = 3
    ) -> None:
        """Initialize the Yahoo Finance fetcher.

        Args:
            delay_between_requests: Delay in seconds between API requests
                to respect rate limits (default: 0.5s)
            max_retries: Maximum number of retry attempts for failed
                requests (default: 3)
        """
        self._delay = delay_between_requests
        self._max_retries = max_retries
        self._last_request_time: float = 0.0

    def validate_connection(self) -> bool:
        """Validate that the connection to Yahoo Finance is working.

        Returns:
            True if connection is valid, False otherwise.
        """
        try:
            # Try to fetch a single day of data for a reliable symbol
            test_ticker = yf.Ticker("SPY")
            data = test_ticker.history(period="1d")
            return not data.empty
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
    def _fetch_ticker_data(
        self,
        symbol: str,
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> pd.DataFrame:
        """Fetch data for a single ticker with retry logic.

        Args:
            symbol: Ticker symbol to fetch
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with OHLCV data

        Raises:
            YahooFinanceFetcherError: If fetch fails
            RateLimitError: If rate limit is exceeded
        """
        self._rate_limit()

        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=False)

            if data.empty:
                raise DataNotAvailableError(
                    f"No data available for {symbol} "
                    f"between {start_date} and {end_date}",
                    source="Yahoo Finance",
                )

            return data

        except Exception as e:
            if isinstance(e, (FetchError, RateLimitError, DataNotAvailableError)):
                raise

            # Check if it's a rate limit issue
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                raise RateLimitError(
                    f"Rate limit exceeded for {symbol}", source="Yahoo Finance"
                ) from e

            raise YahooFinanceFetcherError(
                f"Failed to fetch data for {symbol}: {e!s}",
                source="Yahoo Finance",
            ) from e

    def fetch_etf_prices(
        self,
        symbols: list[ETFSymbol | str],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> list[DailyPrice]:
        """Fetch ETF price data for given symbols.

        Args:
            symbols: List of ETF symbols (can be ETFSymbol enum or strings)
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            List of DailyPrice models for all symbols and dates

        Raises:
            YahooFinanceFetcherError: If fetch fails
            ValueError: If date range is invalid
        """
        self._validate_date_range(start_date, end_date)

        prices: list[DailyPrice] = []

        for symbol in symbols:
            # Convert ETFSymbol enum to string if needed
            symbol_str = symbol.value if isinstance(symbol, ETFSymbol) else symbol

            try:
                df = self._fetch_ticker_data(symbol_str, start_date, end_date)

                # Convert DataFrame to DailyPrice models
                for date_idx, row in df.iterrows():
                    if isinstance(date_idx, pd.Timestamp):
                        price_date = date_idx.to_pydatetime()
                    else:
                        continue

                    # Handle potential missing data
                    if pd.isna(row["Close"]) or pd.isna(row["Adj Close"]):
                        continue

                    # Try to find matching enum value, fallback to string
                    etf_symbol = self._get_etf_symbol(symbol_str)

                    prices.append(
                        DailyPrice(
                            symbol=etf_symbol,
                            date=price_date.date(),
                            open=float(row["Open"]),
                            high=float(row["High"]),
                            low=float(row["Low"]),
                            close=float(row["Close"]),
                            volume=int(row["Volume"]),
                            adjusted_close=float(row["Adj Close"]),
                        )
                    )

            except DataNotAvailableError as e:
                # Log warning but continue with other symbols
                print(f"Warning: {e}")
                continue
            except RetryError as e:
                raise YahooFinanceFetcherError(
                    f"Max retries exceeded for {symbol_str}",
                    source="Yahoo Finance",
                ) from e

        if not prices:
            raise DataNotAvailableError(
                f"No price data available for any symbols "
                f"between {start_date} and {end_date}",
                source="Yahoo Finance",
            )

        return prices

    def fetch_index_prices(
        self,
        symbols: list[str],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> pd.DataFrame:
        """Fetch index price data for given symbols.

        Args:
            symbols: List of index ticker symbols (e.g., '^GSPC', '^DJI')
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            DataFrame with multi-index columns (symbol, price_type)
            and date index

        Raises:
            YahooFinanceFetcherError: If fetch fails
            ValueError: If date range is invalid
        """
        self._validate_date_range(start_date, end_date)

        dfs: list[pd.DataFrame] = []

        for symbol in symbols:
            try:
                df = self._fetch_ticker_data(symbol, start_date, end_date)
                # Add symbol as column prefix
                df.columns = pd.MultiIndex.from_product(
                    [[symbol], df.columns], names=["symbol", "price_type"]
                )
                dfs.append(df)
            except DataNotAvailableError as e:
                print(f"Warning: {e}")
                continue
            except RetryError as e:
                raise YahooFinanceFetcherError(
                    f"Max retries exceeded for {symbol}",
                    source="Yahoo Finance",
                ) from e

        if not dfs:
            raise DataNotAvailableError(
                f"No index data available for any symbols "
                f"between {start_date} and {end_date}",
                source="Yahoo Finance",
            )

        # Concatenate all dataframes
        result = pd.concat(dfs, axis=1)
        result.index.name = "date"

        return result

    def fetch_vix(
        self, start_date: date | datetime, end_date: date | datetime
    ) -> pd.DataFrame:
        """Fetch VIX (CBOE Volatility Index) data from Yahoo Finance.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            DataFrame with date index and 'vix' column containing
            closing values

        Raises:
            YahooFinanceFetcherError: If fetch fails
            ValueError: If date range is invalid
        """
        self._validate_date_range(start_date, end_date)

        try:
            df = self._fetch_ticker_data("^VIX", start_date, end_date)

            # Extract close prices and rename column
            vix_df = pd.DataFrame(df["Close"])
            vix_df.columns = ["vix"]
            vix_df.index.name = "date"

            return vix_df

        except RetryError as e:
            raise YahooFinanceFetcherError(
                "Max retries exceeded for VIX data", source="Yahoo Finance"
            ) from e

    def _get_etf_symbol(self, symbol_str: str) -> ETFSymbol:
        """Convert string symbol to ETFSymbol enum if possible.

        Args:
            symbol_str: Symbol as string

        Returns:
            Matching ETFSymbol enum value

        Raises:
            ValueError: If symbol not found in ETFSymbol enum
        """
        try:
            # Try to find matching enum by value
            for etf_symbol in ETFSymbol:
                if etf_symbol.value == symbol_str:
                    return etf_symbol

            # If not found, raise error
            raise ValueError(
                f"Symbol {symbol_str} not found in ETFSymbol enum. "
                f"Available symbols: {[e.value for e in ETFSymbol]}"
            )
        except Exception as e:
            raise YahooFinanceFetcherError(
                f"Failed to convert symbol {symbol_str} to ETFSymbol: {e!s}",
                source="Yahoo Finance",
            ) from e

    def fetch_multiple_symbols(
        self,
        symbols: list[str],
        start_date: date | datetime,
        end_date: date | datetime,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch data for multiple symbols at once (batch request).

        This is more efficient than individual requests for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            **kwargs: Additional arguments passed to yfinance

        Returns:
            DataFrame with multi-index columns (symbol, price_type)
            and date index

        Raises:
            YahooFinanceFetcherError: If fetch fails
            ValueError: If date range is invalid
        """
        self._validate_date_range(start_date, end_date)

        self._rate_limit()

        try:
            # Download data for all symbols at once
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                group_by="ticker",
                **kwargs,
            )

            if data.empty:
                raise DataNotAvailableError(
                    f"No data available for symbols {symbols} "
                    f"between {start_date} and {end_date}",
                    source="Yahoo Finance",
                )

            data.index.name = "date"
            return data

        except Exception as e:
            if isinstance(e, (FetchError, DataNotAvailableError)):
                raise

            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                raise RateLimitError(
                    "Rate limit exceeded for batch download",
                    source="Yahoo Finance",
                ) from e

            raise YahooFinanceFetcherError(
                f"Failed to fetch batch data: {e!s}", source="Yahoo Finance"
            ) from e
