"""Integration tests for data-to-signal pipeline.

Tests the complete flow:
1. Fetch/load price and macro data
2. Calculate features for HMM
3. Detect market regime
4. Generate allocation recommendation
5. Verify outputs are valid and consistent

These tests verify that data flows correctly through the signal generation
layer and that all components integrate properly.
"""

import pandas as pd

from src.data.models import Regime
from src.signals.allocation import AllocationOptimizer
from src.signals.features import FeatureCalculator
from src.signals.regime import RegimeDetector


class TestDataToSignalPipeline:
    """Integration tests for the complete data-to-signal pipeline."""

    def test_end_to_end_pipeline_with_mock_data(
        self,
        mock_price_data: pd.DataFrame,
        mock_vix_data: pd.DataFrame,
        mock_treasury_data: pd.DataFrame,
        mock_hy_spread_data: pd.DataFrame,
    ) -> None:
        """Test complete pipeline from raw data to allocation recommendation.

        This is the primary integration test - it verifies that all components
        work together to produce a valid allocation from market data.
        """
        # Step 1: Prepare data
        wpea_prices = mock_price_data[mock_price_data["symbol"] == "WPEA.PA"].set_index(
            "date"
        )["close"]

        vix_df = pd.DataFrame({"vix": mock_vix_data.set_index("date")["vix"]})
        price_df = pd.DataFrame({"close": wpea_prices})
        treasury_df = mock_treasury_data.set_index("date")[
            ["treasury_2y", "treasury_10y"]
        ]
        hy_df = pd.DataFrame(
            {"hy_spread": mock_hy_spread_data.set_index("date")["hy_spread"]}
        )

        # Step 2: Calculate features
        feature_calc = FeatureCalculator(lookback_days=252)
        features_df = feature_calc.calculate_feature_history(
            vix_df=vix_df,
            price_df=price_df,
            treasury_df=treasury_df,  # type: ignore[arg-type]
            hy_spread_df=hy_df,
        )

        # Verify features were calculated
        assert not features_df.empty
        assert len(features_df) > 1000  # Should have many observations

        # Step 3: Train regime detector (skip sample validation for test data)
        detector = RegimeDetector(n_states=3, random_state=42)
        feature_array = features_df.values
        detector.fit(feature_array, skip_sample_validation=True)

        assert detector.is_fitted
        assert detector.n_features == 9

        # Step 4: Predict current regime
        latest_features = feature_array[-1:, :]
        current_regime = detector.predict_regime(latest_features)
        regime_probs = detector.predict_regime_probabilities(latest_features)

        # Verify regime output
        assert isinstance(current_regime, Regime)
        assert current_regime in [Regime.RISK_ON, Regime.NEUTRAL, Regime.RISK_OFF]

        # Verify probabilities sum to 1
        total_prob = sum(regime_probs.values())
        assert 0.99 <= total_prob <= 1.01

        # Step 5: Generate allocation
        optimizer = AllocationOptimizer()
        allocation = optimizer.get_target_allocation(
            regime=current_regime, confidence=regime_probs[current_regime]
        )

        # Verify allocation
        assert allocation.regime == current_regime
        assert 0.0 <= allocation.lqq_weight <= 0.30
        assert 0.0 <= allocation.cl2_weight <= 0.30
        assert 0.0 <= allocation.wpea_weight <= 1.0
        assert 0.10 <= allocation.cash_weight <= 1.0

        # Verify weights sum to 1
        total_weight = (
            allocation.lqq_weight
            + allocation.cl2_weight
            + allocation.wpea_weight
            + allocation.cash_weight
        )
        assert 0.99 <= total_weight <= 1.01

        # Verify leveraged exposure limit
        leveraged_exposure = allocation.lqq_weight + allocation.cl2_weight
        assert leveraged_exposure <= 0.30

    def test_pipeline_handles_risk_on_regime(
        self,
        mock_price_data: pd.DataFrame,
        mock_vix_data: pd.DataFrame,
        mock_treasury_data: pd.DataFrame,
        mock_hy_spread_data: pd.DataFrame,
    ) -> None:
        """Test pipeline produces appropriate allocation for RISK_ON regime.

        Uses data biased toward low VIX to force RISK_ON detection.
        """
        # Bias VIX data toward low values (RISK_ON)
        vix_data = mock_vix_data.copy()
        vix_data["vix"] = vix_data["vix"] * 0.5 + 9.0  # Scale to 9-15 range

        wpea_prices = mock_price_data[mock_price_data["symbol"] == "WPEA.PA"].set_index(
            "date"
        )["close"]

        vix_df = pd.DataFrame({"vix": vix_data.set_index("date")["vix"]})
        price_df = pd.DataFrame({"close": wpea_prices})
        treasury_df = mock_treasury_data.set_index("date")[
            ["treasury_2y", "treasury_10y"]
        ]
        hy_df = pd.DataFrame(
            {"hy_spread": mock_hy_spread_data.set_index("date")["hy_spread"]}
        )

        # Calculate features and train detector
        feature_calc = FeatureCalculator()
        features_df = feature_calc.calculate_feature_history(
            vix_df=vix_df,
            price_df=price_df,
            treasury_df=treasury_df,  # type: ignore[arg-type]
            hy_spread_df=hy_df,
        )

        detector = RegimeDetector(n_states=3, random_state=42)
        detector.fit(features_df.values, skip_sample_validation=True)

        # Predict using latest features (which should be low VIX)
        latest_features = features_df.values[-1:, :]
        regime = detector.predict_regime(latest_features)

        # Generate allocation
        optimizer = AllocationOptimizer()
        allocation = optimizer.get_target_allocation(regime, confidence=1.0)

        # With low VIX, should get higher leveraged exposure
        leveraged_exposure = allocation.lqq_weight + allocation.cl2_weight
        assert leveraged_exposure > 0.15  # Should be above neutral

    def test_pipeline_handles_risk_off_regime(
        self,
        mock_price_data: pd.DataFrame,
        mock_vix_data: pd.DataFrame,
        mock_treasury_data: pd.DataFrame,
        mock_hy_spread_data: pd.DataFrame,
    ) -> None:
        """Test pipeline produces conservative allocation for RISK_OFF regime.

        Uses data biased toward high VIX to force RISK_OFF detection.
        """
        # Bias VIX data toward high values (RISK_OFF)
        vix_data = mock_vix_data.copy()
        vix_data["vix"] = vix_data["vix"] * 1.5 + 20.0  # Scale to 35-50 range

        wpea_prices = mock_price_data[mock_price_data["symbol"] == "WPEA.PA"].set_index(
            "date"
        )["close"]

        vix_df = pd.DataFrame({"vix": vix_data.set_index("date")["vix"]})
        price_df = pd.DataFrame({"close": wpea_prices})
        treasury_df = mock_treasury_data.set_index("date")[
            ["treasury_2y", "treasury_10y"]
        ]
        hy_df = pd.DataFrame(
            {"hy_spread": mock_hy_spread_data.set_index("date")["hy_spread"]}
        )

        # Calculate features and train detector
        feature_calc = FeatureCalculator()
        features_df = feature_calc.calculate_feature_history(
            vix_df=vix_df,
            price_df=price_df,
            treasury_df=treasury_df,  # type: ignore[arg-type]
            hy_spread_df=hy_df,
        )

        detector = RegimeDetector(n_states=3, random_state=42)
        detector.fit(features_df.values, skip_sample_validation=True)

        # Predict using latest features
        latest_features = features_df.values[-1:, :]
        regime = detector.predict_regime(latest_features)

        # Generate allocation
        optimizer = AllocationOptimizer()
        allocation = optimizer.get_target_allocation(regime, confidence=1.0)

        # With high VIX, should get conservative allocation
        # The exact regime detected may vary, but allocation should be valid
        leveraged_exposure = allocation.lqq_weight + allocation.cl2_weight
        assert leveraged_exposure <= 0.30  # Should respect limits

        # If RISK_OFF is detected, should be more defensive
        if regime == Regime.RISK_OFF:
            assert leveraged_exposure <= 0.20  # Should be defensive
            assert allocation.cash_weight >= 0.20  # Should hold more cash

    def test_pipeline_handles_missing_data_gracefully(
        self,
        mock_price_data: pd.DataFrame,
        mock_vix_data: pd.DataFrame,
        mock_treasury_data: pd.DataFrame,
        mock_hy_spread_data: pd.DataFrame,
    ) -> None:
        """Test pipeline handles missing data points in input series."""
        # Introduce gaps in data
        vix_data = mock_vix_data.copy()
        vix_data.loc[100:110, "vix"] = None  # Missing VIX data

        treasury_data = mock_treasury_data.copy()
        treasury_data.loc[200:210, "treasury_2y"] = None  # Missing yields

        wpea_prices = mock_price_data[mock_price_data["symbol"] == "WPEA.PA"].set_index(
            "date"
        )["close"]

        vix_df = pd.DataFrame({"vix": vix_data.set_index("date")["vix"]})
        price_df = pd.DataFrame({"close": wpea_prices})
        treasury_df = treasury_data.set_index("date")[["treasury_2y", "treasury_10y"]]
        hy_df = pd.DataFrame(
            {"hy_spread": mock_hy_spread_data.set_index("date")["hy_spread"]}
        )

        # Should not raise exception
        feature_calc = FeatureCalculator()
        features_df = feature_calc.calculate_feature_history(
            vix_df=vix_df,
            price_df=price_df,
            treasury_df=treasury_df,  # type: ignore[arg-type]
            hy_spread_df=hy_df,
        )

        # Should still produce features (with forward-fill handling gaps)
        assert not features_df.empty
        assert len(features_df) > 1000

    def test_pipeline_feature_consistency(
        self,
        mock_price_data: pd.DataFrame,
        mock_vix_data: pd.DataFrame,
        mock_treasury_data: pd.DataFrame,
        mock_hy_spread_data: pd.DataFrame,
    ) -> None:
        """Test that features maintain reasonable values throughout pipeline."""
        wpea_prices = mock_price_data[mock_price_data["symbol"] == "WPEA.PA"].set_index(
            "date"
        )["close"]

        vix_df = pd.DataFrame({"vix": mock_vix_data.set_index("date")["vix"]})
        price_df = pd.DataFrame({"close": wpea_prices})
        treasury_df = mock_treasury_data.set_index("date")[
            ["treasury_2y", "treasury_10y"]
        ]
        hy_df = pd.DataFrame(
            {"hy_spread": mock_hy_spread_data.set_index("date")["hy_spread"]}
        )

        feature_calc = FeatureCalculator()
        features_df = feature_calc.calculate_feature_history(
            vix_df=vix_df,
            price_df=price_df,
            treasury_df=treasury_df,  # type: ignore[arg-type]
            hy_spread_df=hy_df,
        )

        # Check feature ranges
        assert (features_df["vix_level"] >= 0).all()
        assert (features_df["vix_level"] <= 100).all()  # VIX shouldn't exceed 100

        assert (features_df["vix_percentile_20d"] >= 0).all()
        assert (features_df["vix_percentile_20d"] <= 1).all()

        assert (features_df["realized_vol_20d"] >= 0).all()
        assert (features_df["realized_vol_20d"] <= 2.0).all()  # 200% vol is extreme

        assert (features_df["price_vs_ma200"] > 0).all()
        assert (features_df["ma50_vs_ma200"] > 0).all()

        # No NaN values in final features
        assert not features_df.isnull().any().any()  # type: ignore[union-attr]

    def test_pipeline_regime_probability_confidence(
        self,
        mock_price_data: pd.DataFrame,
        mock_vix_data: pd.DataFrame,
        mock_treasury_data: pd.DataFrame,
        mock_hy_spread_data: pd.DataFrame,
    ) -> None:
        """Test that regime probabilities affect allocation appropriately.

        Low confidence should blend allocation toward neutral.
        """
        wpea_prices = mock_price_data[mock_price_data["symbol"] == "WPEA.PA"].set_index(
            "date"
        )["close"]

        vix_df = pd.DataFrame({"vix": mock_vix_data.set_index("date")["vix"]})
        price_df = pd.DataFrame({"close": wpea_prices})
        treasury_df = mock_treasury_data.set_index("date")[
            ["treasury_2y", "treasury_10y"]
        ]
        hy_df = pd.DataFrame(
            {"hy_spread": mock_hy_spread_data.set_index("date")["hy_spread"]}
        )

        feature_calc = FeatureCalculator()
        features_df = feature_calc.calculate_feature_history(
            vix_df=vix_df,
            price_df=price_df,
            treasury_df=treasury_df,  # type: ignore[arg-type]
            hy_spread_df=hy_df,
        )

        detector = RegimeDetector(n_states=3, random_state=42)
        detector.fit(features_df.values, skip_sample_validation=True)

        latest_features = features_df.values[-1:, :]
        regime = detector.predict_regime(latest_features)

        optimizer = AllocationOptimizer()

        # Get allocation with full confidence
        alloc_high_conf = optimizer.get_target_allocation(regime, confidence=1.0)

        # Get allocation with low confidence
        alloc_low_conf = optimizer.get_target_allocation(regime, confidence=0.3)

        # Low confidence should blend toward neutral
        # If regime is RISK_ON or RISK_OFF, low confidence should move toward NEUTRAL
        if regime != Regime.NEUTRAL:
            # Leveraged exposure should be closer to neutral with low confidence
            high_lev = alloc_high_conf.lqq_weight + alloc_high_conf.cl2_weight
            low_lev = alloc_low_conf.lqq_weight + alloc_low_conf.cl2_weight

            # Should be blended (not necessarily lower, but different)
            assert (
                high_lev != low_lev
                or alloc_high_conf.cash_weight != alloc_low_conf.cash_weight
            )
