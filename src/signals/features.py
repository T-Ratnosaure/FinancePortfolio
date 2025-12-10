"""Feature engineering module for regime detection.

This module provides feature calculations for the HMM regime detector.
Features are designed to capture market conditions across three dimensions:
volatility, trend, and macro/credit conditions.

All features are calculated with proper handling of missing data and
validation for sufficient historical data.
"""

from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


class InsufficientDataError(Exception):
    """Raised when there is insufficient historical data for feature calculation."""

    def __init__(self, feature_name: str, required: int, available: int) -> None:
        """Initialize the error.

        Args:
            feature_name: Name of the feature that failed calculation
            required: Number of data points required
            available: Number of data points available
        """
        self.feature_name = feature_name
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient data for {feature_name}: "
            f"required {required} points, got {available}"
        )


class FeatureSet(BaseModel):
    """Complete feature set for regime detection.

    This model contains all features used by the HMM regime detector to
    classify market conditions. Features are grouped by category:
    volatility, trend, and macro/credit indicators.

    Attributes:
        date: Date for which features are calculated
        vix_level: Current VIX value (implied volatility)
        vix_percentile_20d: VIX percentile over last 20 days (0-1)
        realized_vol_20d: 20-day realized volatility of reference ETF
        price_vs_ma200: Current price / 200-day MA ratio
        ma50_vs_ma200: 50-day MA / 200-day MA ratio (golden/death cross)
        momentum_3m: 3-month return (63 trading days)
        yield_curve_slope: 10Y - 2Y Treasury spread (percentage points)
        hy_spread: High yield OAS spread (percentage points)
        hy_spread_change_1m: 1-month change in HY spread (percentage points)
    """

    date: date

    # Volatility features (most important for regime detection)
    vix_level: float = Field(ge=0.0, description="Current VIX value")
    vix_percentile_20d: float = Field(
        ge=0.0, le=1.0, description="VIX percentile over last 20 days"
    )
    realized_vol_20d: float = Field(
        ge=0.0, description="20-day annualized realized volatility"
    )

    # Trend features
    price_vs_ma200: float = Field(
        gt=0.0, description="Current price / 200-day MA ratio"
    )
    ma50_vs_ma200: float = Field(
        gt=0.0, description="50-day MA / 200-day MA ratio"
    )
    momentum_3m: float = Field(description="3-month return (decimal)")

    # Credit/Macro features
    yield_curve_slope: float = Field(
        description="10Y - 2Y Treasury spread in percentage points"
    )
    hy_spread: float = Field(ge=0.0, description="High yield OAS spread")
    hy_spread_change_1m: float = Field(
        description="1-month change in HY spread"
    )

    @model_validator(mode="after")
    def validate_feature_consistency(self) -> "FeatureSet":
        """Validate feature values are within reasonable bounds.

        Financial features should not have extreme values that would
        indicate data quality issues.
        """
        # VIX sanity check - historically between 9 and 90
        if self.vix_level > 100:
            raise ValueError(
                f"VIX level {self.vix_level} exceeds historical maximum (~90)"
            )

        # Realized vol sanity check - rarely exceeds 100% annualized
        if self.realized_vol_20d > 2.0:
            raise ValueError(
                f"Realized vol {self.realized_vol_20d:.2%} exceeds "
                "reasonable bounds (200% annualized)"
            )

        # Price ratio sanity check - extreme deviation from MA200 unlikely
        if self.price_vs_ma200 > 2.0 or self.price_vs_ma200 < 0.3:
            raise ValueError(
                f"Price/MA200 ratio {self.price_vs_ma200:.2f} outside "
                "reasonable bounds [0.3, 2.0]"
            )

        return self

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for model input.

        Returns:
            1D numpy array with features in consistent order.
        """
        return np.array([
            self.vix_level,
            self.vix_percentile_20d,
            self.realized_vol_20d,
            self.price_vs_ma200,
            self.ma50_vs_ma200,
            self.momentum_3m,
            self.yield_curve_slope,
            self.hy_spread,
            self.hy_spread_change_1m,
        ])

    @classmethod
    def feature_names(cls) -> list[str]:
        """Get ordered list of feature names.

        Returns:
            List of feature names matching to_array() order.
        """
        return [
            "vix_level",
            "vix_percentile_20d",
            "realized_vol_20d",
            "price_vs_ma200",
            "ma50_vs_ma200",
            "momentum_3m",
            "yield_curve_slope",
            "hy_spread",
            "hy_spread_change_1m",
        ]


class FeatureCalculator:
    """Calculator for regime detection features.

    This class encapsulates all feature calculation logic, ensuring
    consistent computation across training and inference. All methods
    handle missing data gracefully and validate data sufficiency.

    The calculator is stateless - all required data must be passed to
    each method call.

    Attributes:
        lookback_days: Number of historical days required for calculations
    """

    # Trading days constants
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_MONTH = 21
    TRADING_DAYS_3_MONTHS = 63

    # Minimum data requirements for each feature category
    MIN_DAYS_VOLATILITY = 20
    MIN_DAYS_TREND = 200
    MIN_DAYS_MACRO = 21

    def __init__(self, lookback_days: int = 252) -> None:
        """Initialize the feature calculator.

        Args:
            lookback_days: Number of historical days to maintain for
                calculations. Default is 252 (one trading year).
        """
        self._lookback_days = lookback_days

    @property
    def lookback_days(self) -> int:
        """Get the lookback period in days."""
        return self._lookback_days

    def _validate_series_length(
        self, series: pd.Series, min_length: int, feature_name: str
    ) -> None:
        """Validate that a series has sufficient data.

        Args:
            series: Pandas series to validate
            min_length: Minimum required length
            feature_name: Name of feature for error message

        Raises:
            InsufficientDataError: If series is too short
        """
        valid_count = series.dropna().shape[0]
        if valid_count < min_length:
            raise InsufficientDataError(feature_name, min_length, valid_count)

    def _prepare_series(self, series: pd.Series) -> pd.Series:
        """Prepare a series for calculation by forward-filling NaN values.

        Args:
            series: Raw input series

        Returns:
            Series with NaN values forward-filled
        """
        return series.ffill()

    def calculate_volatility_features(
        self, vix_series: pd.Series, price_series: pd.Series
    ) -> dict[str, float]:
        """Calculate volatility-related features.

        Volatility features capture market fear and realized risk:
        - VIX level: Current implied volatility (forward-looking)
        - VIX percentile: Relative VIX level vs recent history
        - Realized vol: Backward-looking actual price volatility

        Args:
            vix_series: Series of VIX values with datetime index
            price_series: Series of adjusted close prices with datetime index

        Returns:
            Dictionary with keys: vix_level, vix_percentile_20d, realized_vol_20d

        Raises:
            InsufficientDataError: If insufficient data for calculation
        """
        vix_clean = self._prepare_series(vix_series)
        price_clean = self._prepare_series(price_series)

        self._validate_series_length(
            vix_clean, self.MIN_DAYS_VOLATILITY, "vix_level"
        )
        self._validate_series_length(
            price_clean, self.MIN_DAYS_VOLATILITY, "realized_vol"
        )

        # Current VIX level
        vix_level = float(vix_clean.iloc[-1])

        # VIX percentile over last 20 days
        vix_20d = vix_clean.iloc[-self.MIN_DAYS_VOLATILITY:]
        vix_percentile = float(
            (vix_20d < vix_level).sum() / len(vix_20d)
        )

        # 20-day realized volatility (annualized)
        returns = price_clean.pct_change().dropna()
        if len(returns) < self.MIN_DAYS_VOLATILITY - 1:
            raise InsufficientDataError(
                "realized_vol_20d",
                self.MIN_DAYS_VOLATILITY - 1,
                len(returns),
            )

        returns_20d = returns.iloc[-self.MIN_DAYS_VOLATILITY:]
        realized_vol = float(
            returns_20d.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        )

        return {
            "vix_level": vix_level,
            "vix_percentile_20d": vix_percentile,
            "realized_vol_20d": realized_vol,
        }

    def calculate_trend_features(
        self, price_series: pd.Series
    ) -> dict[str, float]:
        """Calculate trend-related features.

        Trend features capture market direction and momentum:
        - Price vs MA200: Current price relative to long-term average
        - MA50 vs MA200: Golden/death cross indicator
        - 3-month momentum: Recent return performance

        Args:
            price_series: Series of adjusted close prices with datetime index

        Returns:
            Dictionary with keys: price_vs_ma200, ma50_vs_ma200, momentum_3m

        Raises:
            InsufficientDataError: If insufficient data for calculation
        """
        price_clean = self._prepare_series(price_series)

        self._validate_series_length(
            price_clean, self.MIN_DAYS_TREND, "trend_features"
        )

        current_price = float(price_clean.iloc[-1])

        # 200-day moving average
        ma200 = float(price_clean.iloc[-200:].mean())

        # 50-day moving average
        ma50 = float(price_clean.iloc[-50:].mean())

        # 3-month momentum (63 trading days return)
        if len(price_clean) >= self.TRADING_DAYS_3_MONTHS:
            price_3m_ago = float(price_clean.iloc[-self.TRADING_DAYS_3_MONTHS])
            momentum_3m = (current_price - price_3m_ago) / price_3m_ago
        else:
            # Fallback to available data
            oldest_price = float(price_clean.iloc[0])
            momentum_3m = (current_price - oldest_price) / oldest_price

        return {
            "price_vs_ma200": current_price / ma200,
            "ma50_vs_ma200": ma50 / ma200,
            "momentum_3m": float(momentum_3m),
        }

    def calculate_macro_features(
        self,
        treasury_2y: pd.Series,
        treasury_10y: pd.Series,
        hy_spread: pd.Series,
    ) -> dict[str, float]:
        """Calculate macro/credit-related features.

        Macro features capture credit conditions and economic outlook:
        - Yield curve slope: Inversion signals recession risk
        - HY spread: Credit stress indicator
        - HY spread change: Acceleration of credit conditions

        Args:
            treasury_2y: Series of 2-year Treasury yields (%)
            treasury_10y: Series of 10-year Treasury yields (%)
            hy_spread: Series of high yield OAS spread values (%)

        Returns:
            Dictionary with keys: yield_curve_slope, hy_spread, hy_spread_change_1m

        Raises:
            InsufficientDataError: If insufficient data for calculation
        """
        t2y_clean = self._prepare_series(treasury_2y)
        t10y_clean = self._prepare_series(treasury_10y)
        hy_clean = self._prepare_series(hy_spread)

        self._validate_series_length(
            t2y_clean, self.MIN_DAYS_MACRO, "treasury_2y"
        )
        self._validate_series_length(
            t10y_clean, self.MIN_DAYS_MACRO, "treasury_10y"
        )
        self._validate_series_length(
            hy_clean, self.MIN_DAYS_MACRO, "hy_spread"
        )

        # Yield curve slope (10Y - 2Y)
        yield_curve_slope = float(t10y_clean.iloc[-1] - t2y_clean.iloc[-1])

        # Current HY spread
        current_hy = float(hy_clean.iloc[-1])

        # 1-month change in HY spread
        hy_1m_ago = float(hy_clean.iloc[-self.TRADING_DAYS_PER_MONTH])
        hy_spread_change = current_hy - hy_1m_ago

        return {
            "yield_curve_slope": yield_curve_slope,
            "hy_spread": current_hy,
            "hy_spread_change_1m": float(hy_spread_change),
        }

    def calculate_all_features(
        self,
        feature_date: date,
        vix_series: pd.Series,
        price_series: pd.Series,
        treasury_2y: pd.Series,
        treasury_10y: pd.Series,
        hy_spread: pd.Series,
    ) -> FeatureSet:
        """Calculate all features for a single date.

        This method combines volatility, trend, and macro features into
        a complete FeatureSet for regime detection.

        Args:
            feature_date: Date for which features are calculated
            vix_series: Series of VIX values with datetime index
            price_series: Series of adjusted close prices with datetime index
            treasury_2y: Series of 2-year Treasury yields
            treasury_10y: Series of 10-year Treasury yields
            hy_spread: Series of high yield OAS spread values

        Returns:
            FeatureSet with all calculated features

        Raises:
            InsufficientDataError: If insufficient data for any feature
            ValueError: If feature values fail validation
        """
        vol_features = self.calculate_volatility_features(vix_series, price_series)
        trend_features = self.calculate_trend_features(price_series)
        macro_features = self.calculate_macro_features(
            treasury_2y, treasury_10y, hy_spread
        )

        return FeatureSet(
            date=feature_date,
            **vol_features,
            **trend_features,
            **macro_features,
        )

    def calculate_feature_history(
        self,
        vix_df: pd.DataFrame,
        price_df: pd.DataFrame,
        treasury_df: pd.DataFrame,
        hy_spread_df: pd.DataFrame,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Calculate features over a historical time period.

        This method is used for training the HMM regime detector. It calculates
        features for each date in the specified range, ensuring proper temporal
        ordering and no look-ahead bias.

        Args:
            vix_df: DataFrame with 'vix' column and datetime index
            price_df: DataFrame with 'close' or 'adjusted_close' column
                and datetime index
            treasury_df: DataFrame with 'treasury_2y' and 'treasury_10y'
                columns and datetime index
            hy_spread_df: DataFrame with 'hy_spread' or 'hy_oas_spread' column
                and datetime index
            start_date: Start date for feature calculation (optional)
            end_date: End date for feature calculation (optional)

        Returns:
            DataFrame with features for each date, indexed by date.
            Columns match FeatureSet.feature_names()

        Raises:
            InsufficientDataError: If insufficient data for calculations
            ValueError: If required columns are missing
        """
        # Validate input columns
        self._validate_dataframe_columns(vix_df, ["vix"], "vix_df")
        price_col = self._get_price_column(price_df)
        self._validate_dataframe_columns(
            treasury_df, ["treasury_2y", "treasury_10y"], "treasury_df"
        )
        hy_col = self._get_hy_spread_column(hy_spread_df)

        # Align all dataframes to common index
        all_dates = (
            vix_df.index
            .intersection(price_df.index)
            .intersection(treasury_df.index)
            .intersection(hy_spread_df.index)
        )

        if len(all_dates) == 0:
            raise InsufficientDataError(
                "feature_history", self.MIN_DAYS_TREND, 0
            )

        # Apply date filters
        if start_date is not None:
            all_dates = all_dates[all_dates >= pd.Timestamp(start_date)]
        if end_date is not None:
            all_dates = all_dates[all_dates <= pd.Timestamp(end_date)]

        all_dates = all_dates.sort_values()

        # Minimum offset for first valid calculation
        min_offset = self.MIN_DAYS_TREND

        if len(all_dates) < min_offset:
            raise InsufficientDataError(
                "feature_history", min_offset, len(all_dates)
            )

        # Calculate features for each date
        feature_records: list[dict[str, float | date]] = []

        for i in range(min_offset, len(all_dates)):
            current_date = all_dates[i]

            # Get data up to current date (no look-ahead)
            mask = all_dates <= current_date
            valid_dates = all_dates[mask]

            try:
                vix_series = vix_df.loc[valid_dates, "vix"]
                price_series = price_df.loc[valid_dates, price_col]
                t2y_series = treasury_df.loc[valid_dates, "treasury_2y"]
                t10y_series = treasury_df.loc[valid_dates, "treasury_10y"]
                hy_series = hy_spread_df.loc[valid_dates, hy_col]

                features = self.calculate_all_features(
                    feature_date=current_date.date()
                    if hasattr(current_date, "date")
                    else current_date,
                    vix_series=vix_series,
                    price_series=price_series,
                    treasury_2y=t2y_series,
                    treasury_10y=t10y_series,
                    hy_spread=hy_series,
                )

                record = features.model_dump()
                feature_records.append(record)

            except InsufficientDataError:
                # Skip dates with insufficient data
                continue
            except ValueError:
                # Skip dates with invalid feature values
                continue

        if not feature_records:
            raise InsufficientDataError(
                "feature_history",
                min_offset,
                len(all_dates),
            )

        result_df = pd.DataFrame(feature_records)
        result_df.set_index("date", inplace=True)
        result_df.index = pd.to_datetime(result_df.index)

        return result_df

    def _validate_dataframe_columns(
        self, df: pd.DataFrame, required_cols: list[str], df_name: str
    ) -> None:
        """Validate that a DataFrame has required columns.

        Args:
            df: DataFrame to validate
            required_cols: List of required column names
            df_name: Name of DataFrame for error message

        Raises:
            ValueError: If required columns are missing
        """
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"{df_name} missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def _get_price_column(self, price_df: pd.DataFrame) -> str:
        """Get the price column name from DataFrame.

        Prefers 'adjusted_close' over 'close' for accuracy.

        Args:
            price_df: DataFrame with price data

        Returns:
            Column name to use for prices

        Raises:
            ValueError: If no valid price column found
        """
        if "adjusted_close" in price_df.columns:
            return "adjusted_close"
        if "close" in price_df.columns:
            return "close"
        if "Close" in price_df.columns:
            return "Close"
        if "Adj Close" in price_df.columns:
            return "Adj Close"

        raise ValueError(
            "price_df must contain 'adjusted_close', 'close', 'Close', "
            f"or 'Adj Close' column. Found: {list(price_df.columns)}"
        )

    def _get_hy_spread_column(self, hy_df: pd.DataFrame) -> str:
        """Get the HY spread column name from DataFrame.

        Args:
            hy_df: DataFrame with HY spread data

        Returns:
            Column name to use for HY spread

        Raises:
            ValueError: If no valid HY spread column found
        """
        if "hy_spread" in hy_df.columns:
            return "hy_spread"
        if "hy_oas_spread" in hy_df.columns:
            return "hy_oas_spread"

        raise ValueError(
            "hy_spread_df must contain 'hy_spread' or 'hy_oas_spread' column. "
            f"Found: {list(hy_df.columns)}"
        )


# Alias for backwards compatibility with __init__.py
FeatureEngineer = FeatureCalculator
