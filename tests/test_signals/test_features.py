"""Tests for feature engineering module."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.signals.features import (
    FeatureCalculator,
    FeatureSet,
    InsufficientDataError,
)


class TestFeatureSet:
    """Tests for FeatureSet model."""

    def test_valid_feature_set(self) -> None:
        """Test creating a valid feature set."""
        fs = FeatureSet(
            date=date(2024, 1, 15),
            vix_level=15.0,
            vix_percentile_20d=0.5,
            realized_vol_20d=0.15,
            price_vs_ma200=1.05,
            ma50_vs_ma200=1.02,
            momentum_3m=0.08,
            yield_curve_slope=0.5,
            hy_spread=3.5,
            hy_spread_change_1m=0.1,
        )
        assert fs.vix_level == 15.0
        assert fs.date == date(2024, 1, 15)

    def test_to_array(self) -> None:
        """Test converting feature set to numpy array."""
        fs = FeatureSet(
            date=date(2024, 1, 15),
            vix_level=15.0,
            vix_percentile_20d=0.5,
            realized_vol_20d=0.15,
            price_vs_ma200=1.05,
            ma50_vs_ma200=1.02,
            momentum_3m=0.08,
            yield_curve_slope=0.5,
            hy_spread=3.5,
            hy_spread_change_1m=0.1,
        )
        arr = fs.to_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 9
        assert arr[0] == 15.0  # vix_level

    def test_feature_names(self) -> None:
        """Test feature names list."""
        names = FeatureSet.feature_names()
        assert len(names) == 9
        assert "vix_level" in names
        assert "hy_spread" in names

    def test_invalid_vix_level(self) -> None:
        """Test that negative VIX is rejected."""
        with pytest.raises(ValueError):
            FeatureSet(
                date=date(2024, 1, 15),
                vix_level=-5.0,  # Invalid
                vix_percentile_20d=0.5,
                realized_vol_20d=0.15,
                price_vs_ma200=1.05,
                ma50_vs_ma200=1.02,
                momentum_3m=0.08,
                yield_curve_slope=0.5,
                hy_spread=3.5,
                hy_spread_change_1m=0.1,
            )

    def test_invalid_percentile(self) -> None:
        """Test that percentile outside [0, 1] is rejected."""
        with pytest.raises(ValueError):
            FeatureSet(
                date=date(2024, 1, 15),
                vix_level=15.0,
                vix_percentile_20d=1.5,  # Invalid
                realized_vol_20d=0.15,
                price_vs_ma200=1.05,
                ma50_vs_ma200=1.02,
                momentum_3m=0.08,
                yield_curve_slope=0.5,
                hy_spread=3.5,
                hy_spread_change_1m=0.1,
            )


class TestFeatureCalculator:
    """Tests for FeatureCalculator class."""

    @pytest.fixture
    def calculator(self) -> FeatureCalculator:
        """Create a feature calculator instance."""
        return FeatureCalculator(lookback_days=252)

    @pytest.fixture
    def sample_vix_series(self) -> pd.Series:
        """Create sample VIX data."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
        # VIX ranging from 12 to 25
        values = 15 + 5 * np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100)
        return pd.Series(values, index=dates)

    @pytest.fixture
    def sample_price_series(self) -> pd.Series:
        """Create sample price data with trend."""
        dates = pd.date_range(start="2022-01-01", periods=300, freq="B")
        # Trending price with noise
        base = 100 * (1 + 0.0003) ** np.arange(300)  # ~8% annual return
        noise = np.random.randn(300) * 0.5
        return pd.Series(base + noise, index=dates)

    @pytest.fixture
    def sample_treasury_data(self) -> tuple[pd.Series, pd.Series]:
        """Create sample Treasury yield data."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
        t2y = pd.Series(4.5 + 0.2 * np.random.randn(100), index=dates)
        t10y = pd.Series(4.0 + 0.2 * np.random.randn(100), index=dates)
        return t2y, t10y

    @pytest.fixture
    def sample_hy_spread(self) -> pd.Series:
        """Create sample HY spread data."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
        return pd.Series(3.5 + 0.3 * np.random.randn(100), index=dates)

    def test_calculate_volatility_features(
        self,
        calculator: FeatureCalculator,
        sample_vix_series: pd.Series,
        sample_price_series: pd.Series,
    ) -> None:
        """Test volatility feature calculation."""
        features = calculator.calculate_volatility_features(
            sample_vix_series, sample_price_series.iloc[-100:]
        )
        assert "vix_level" in features
        assert "vix_percentile_20d" in features
        assert "realized_vol_20d" in features
        assert 0 <= features["vix_percentile_20d"] <= 1
        assert features["realized_vol_20d"] >= 0

    def test_calculate_trend_features(
        self,
        calculator: FeatureCalculator,
        sample_price_series: pd.Series,
    ) -> None:
        """Test trend feature calculation."""
        features = calculator.calculate_trend_features(sample_price_series)
        assert "price_vs_ma200" in features
        assert "ma50_vs_ma200" in features
        assert "momentum_3m" in features
        assert features["price_vs_ma200"] > 0
        assert features["ma50_vs_ma200"] > 0

    def test_calculate_macro_features(
        self,
        calculator: FeatureCalculator,
        sample_treasury_data: tuple[pd.Series, pd.Series],
        sample_hy_spread: pd.Series,
    ) -> None:
        """Test macro feature calculation."""
        t2y, t10y = sample_treasury_data
        features = calculator.calculate_macro_features(t2y, t10y, sample_hy_spread)
        assert "yield_curve_slope" in features
        assert "hy_spread" in features
        assert "hy_spread_change_1m" in features

    def test_insufficient_data_raises_error(
        self,
        calculator: FeatureCalculator,
    ) -> None:
        """Test that insufficient data raises InsufficientDataError."""
        short_vix = pd.Series([15.0] * 10)
        short_prices = pd.Series([100.0] * 10)

        with pytest.raises(InsufficientDataError):
            calculator.calculate_volatility_features(short_vix, short_prices)

    def test_calculate_all_features(
        self,
        calculator: FeatureCalculator,
        sample_price_series: pd.Series,
    ) -> None:
        """Test calculating all features at once."""
        # Create aligned data with enough history for all calculations
        # Price series already has 300 points from fixture
        dates = sample_price_series.index

        # Create VIX data aligned to price dates
        np.random.seed(42)
        vix_values = 15 + 5 * np.sin(np.linspace(0, 6 * np.pi, len(dates)))
        vix = pd.Series(vix_values, index=dates)

        # Create treasury data
        t2y = pd.Series(4.5 + 0.2 * np.random.randn(len(dates)), index=dates)
        t10y = pd.Series(4.0 + 0.2 * np.random.randn(len(dates)), index=dates)

        # Create HY spread data
        hy = pd.Series(3.5 + 0.3 * np.random.randn(len(dates)), index=dates)

        feature_set = calculator.calculate_all_features(
            feature_date=date(2024, 1, 15),
            vix_series=vix,
            price_series=sample_price_series,
            treasury_2y=t2y,
            treasury_10y=t10y,
            hy_spread=hy,
        )

        assert isinstance(feature_set, FeatureSet)
        assert feature_set.date == date(2024, 1, 15)

    def test_prepare_series_fills_nan(
        self,
        calculator: FeatureCalculator,
    ) -> None:
        """Test that NaN values are forward filled."""
        series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        prepared = calculator._prepare_series(series)
        assert not prepared.isna().any()
        assert prepared.iloc[1] == 1.0  # Forward filled
        assert prepared.iloc[3] == 3.0  # Forward filled
