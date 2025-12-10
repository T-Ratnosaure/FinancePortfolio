"""FRED (Federal Reserve Economic Data) fetcher for macroeconomic indicators."""

import os
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from src.data.fetchers.base import BaseFetcher
from src.data.models import MacroIndicator


class FREDFetcherError(Exception):
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
    """

    SERIES_VIX = "VIXCLS"
    SERIES_TREASURY_10Y = "DGS10"
    SERIES_TREASURY_2Y = "DGS2"
    SERIES_YIELD_SPREAD_2S10S = "T10Y2Y"
    SERIES_HY_SPREAD = "BAMLH0A0HYM2"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the FRED fetcher.

        Args:
            api_key: FRED API key. If None, will be loaded from FRED_API_KEY
                environment variable.

        Raises:
            FREDFetcherError: If API key is not provided or found in environment
        """
        load_dotenv()

        self._api_key = api_key or os.getenv("FRED_API_KEY")

        if not self._api_key:
            raise FREDFetcherError(
                "FRED API key not found. Please set FRED_API_KEY environment "
                "variable or pass api_key parameter. Get a free API key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        try:
            self._client = Fred(api_key=self._api_key)
        except Exception as e:
            raise FREDFetcherError(f"Failed to initialize FRED client: {e}") from e

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

    def _fetch_series(
        self, series_id: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Fetch a single FRED series.

        Args:
            series_id: FRED series identifier
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            DataFrame with date index and single value column

        Raises:
            FREDFetcherError: If fetch fails
        """
        self._validate_date_range(start_date, end_date)

        try:
            series = self._client.get_series(
                series_id,
                observation_start=start_date.isoformat(),
                observation_end=end_date.isoformat(),
            )

            if series is None or series.empty:
                raise FREDFetcherError(f"No data returned for series {series_id}")

            df = pd.DataFrame(series, columns=[series_id])
            df.index.name = "date"

            return df

        except Exception as e:
            if isinstance(e, FREDFetcherError):
                raise
            raise FREDFetcherError(f"Failed to fetch series {series_id}: {e}") from e

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
        df = self._fetch_series(self.SERIES_VIX, start_date, end_date)
        df.columns = ["vix"]
        return df

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
        df_2y = self._fetch_series(self.SERIES_TREASURY_2Y, start_date, end_date)
        df_10y = self._fetch_series(self.SERIES_TREASURY_10Y, start_date, end_date)

        df = pd.concat([df_2y, df_10y], axis=1)
        df.columns = ["treasury_2y", "treasury_10y"]

        return df

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
        df_2s10s = self._fetch_series(
            self.SERIES_YIELD_SPREAD_2S10S, start_date, end_date
        )
        df_hy = self._fetch_series(self.SERIES_HY_SPREAD, start_date, end_date)

        df = pd.concat([df_2s10s, df_hy], axis=1)
        df.columns = ["spread_2s10s", "hy_oas_spread"]

        return df

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

            for date_idx, row in combined_df.iterrows():
                if isinstance(date_idx, pd.Timestamp):
                    current_date = date_idx.date()
                else:
                    continue

                if pd.notna(row["vix"]):
                    indicators.append(
                        MacroIndicator(
                            indicator_name="VIX",
                            date=current_date,
                            value=float(row["vix"]),
                            source="FRED",
                        )
                    )

                if pd.notna(row["treasury_2y"]):
                    indicators.append(
                        MacroIndicator(
                            indicator_name="TREASURY_2Y",
                            date=current_date,
                            value=float(row["treasury_2y"]),
                            source="FRED",
                        )
                    )

                if pd.notna(row["treasury_10y"]):
                    indicators.append(
                        MacroIndicator(
                            indicator_name="TREASURY_10Y",
                            date=current_date,
                            value=float(row["treasury_10y"]),
                            source="FRED",
                        )
                    )

                if pd.notna(row["spread_2s10s"]):
                    indicators.append(
                        MacroIndicator(
                            indicator_name="SPREAD_2S10S",
                            date=current_date,
                            value=float(row["spread_2s10s"]),
                            source="FRED",
                        )
                    )

                if pd.notna(row["hy_oas_spread"]):
                    indicators.append(
                        MacroIndicator(
                            indicator_name="HY_OAS_SPREAD",
                            date=current_date,
                            value=float(row["hy_oas_spread"]),
                            source="FRED",
                        )
                    )

        except Exception as e:
            if isinstance(e, FREDFetcherError):
                raise
            raise FREDFetcherError(f"Failed to fetch macro indicators: {e}") from e

        return indicators
