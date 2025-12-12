# Sprint 5 ML Code - Production Engineering Action Plan

**Created**: 2025-12-12
**Owner**: Development Team
**Priority**: Complete HIGH items before merge

## Overview

The Sprint 5 ML code (HMM regime detector + feature engineering) is **production ready** from a code quality perspective. However, to ensure smooth deployment and operational excellence, we need to address several production engineering items.

This document provides a concrete action plan with implementation guidance.

---

## Action Items by Priority

### HIGH Priority (Complete Before Merge)

#### 1. Add Example Script

**File**: `examples/regime_detector_example.py`

**Why**: Critical for usability. Other modules have examples; this should too.

**Implementation**:
```python
"""Example usage of HMM regime detector for market regime classification.

This example demonstrates the complete workflow:
1. Loading historical market data
2. Calculating regime detection features
3. Training the HMM regime detector
4. Making regime predictions
5. Saving and loading trained models
6. Interpreting results

Note: This uses synthetic data for demonstration. In production,
replace with real market data from your data pipeline.
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd

from src.signals import (
    RegimeDetector,
    FeatureCalculator,
    FeatureSet,
)
from src.data.models import Regime


def generate_synthetic_data(n_days: int = 2000) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Generate synthetic market data for demonstration.

    In production, replace this with real data from:
    - VIX: From CBOE or via data provider
    - SPY: From Yahoo Finance or market data feed
    - Treasury yields: From FRED (Federal Reserve Economic Data)
    - HY spreads: From FRED or Bloomberg

    Args:
        n_days: Number of trading days to generate

    Returns:
        Tuple of (vix_df, spy_df, treasury_df, hy_spread_df)
    """
    dates = pd.date_range(end=date.today(), periods=n_days, freq="B")
    rng = np.random.default_rng(42)

    # Simulate VIX with regime structure
    regime_periods = n_days // 3
    vix_low = rng.normal(loc=12, scale=2, size=regime_periods)
    vix_medium = rng.normal(loc=18, scale=3, size=regime_periods)
    vix_high = rng.normal(loc=30, scale=5, size=n_days - 2 * regime_periods)
    vix_values = np.concatenate([vix_low, vix_medium, vix_high])

    vix_df = pd.DataFrame({"vix": vix_values}, index=dates)

    # Simulate SPY with trend
    spy_returns = rng.normal(loc=0.0003, scale=0.01, size=n_days)
    spy_prices = 100 * np.exp(np.cumsum(spy_returns))
    spy_df = pd.DataFrame({"close": spy_prices}, index=dates)

    # Simulate Treasury yields
    t2y = 4.5 + rng.normal(loc=0, scale=0.2, size=n_days)
    t10y = 4.0 + rng.normal(loc=0, scale=0.2, size=n_days)
    treasury_df = pd.DataFrame(
        {"treasury_2y": t2y, "treasury_10y": t10y},
        index=dates,
    )

    # Simulate HY spreads
    hy_spread = 3.5 + rng.normal(loc=0, scale=0.5, size=n_days)
    hy_spread_df = pd.DataFrame({"hy_spread": hy_spread}, index=dates)

    return vix_df, spy_df, treasury_df, hy_spread_df


def main() -> None:
    """Run complete regime detection workflow."""
    print("=" * 60)
    print("HMM Regime Detector - Complete Example")
    print("=" * 60)

    # Step 1: Generate synthetic data (replace with real data in production)
    print("\n[Step 1] Generating synthetic market data...")
    vix_df, spy_df, treasury_df, hy_spread_df = generate_synthetic_data(n_days=2000)
    print(f"  Generated {len(vix_df)} trading days of data")
    print(f"  Date range: {vix_df.index[0]} to {vix_df.index[-1]}")

    # Step 2: Calculate features
    print("\n[Step 2] Calculating regime detection features...")
    calculator = FeatureCalculator(lookback_days=252)

    features_df = calculator.calculate_feature_history(
        vix_df=vix_df,
        price_df=spy_df,
        treasury_df=treasury_df,
        hy_spread_df=hy_spread_df,
    )
    print(f"  Calculated {len(features_df)} feature observations")
    print(f"  Features: {', '.join(FeatureSet.feature_names())}")

    # Step 3: Train regime detector
    print("\n[Step 3] Training HMM regime detector...")
    detector = RegimeDetector(n_states=3, random_state=42)

    feature_array = features_df[FeatureSet.feature_names()].values
    detector.fit(feature_array)

    print(f"  Model fitted with {len(feature_array)} samples")
    print(f"  Number of features: {detector.n_features}")
    print(f"  Number of states: {detector.n_states}")

    # Step 4: Analyze learned regimes
    print("\n[Step 4] Analyzing learned regime characteristics...")
    characteristics = detector.get_state_characteristics()

    for regime in [Regime.RISK_ON, Regime.NEUTRAL, Regime.RISK_OFF]:
        chars = characteristics[regime]
        print(f"\n  {regime.value}:")
        print(f"    VIX level: {chars['feature_0']:.1f}")
        print(f"    VIX percentile: {chars['feature_1']:.2f}")
        print(f"    Realized vol: {chars['feature_2']:.2%}")

    # Step 5: Get transition matrix
    print("\n[Step 5] Regime transition probabilities...")
    trans_matrix = detector.get_transition_matrix()
    print(f"  Transition matrix shape: {trans_matrix.shape}")
    print(f"  Diagonal (persistence): {np.diag(trans_matrix)}")

    stationary = detector.get_stationary_distribution()
    print(f"\n  Long-run regime distribution:")
    for regime, prob in stationary.items():
        print(f"    {regime.value}: {prob:.1%}")

    # Step 6: Make predictions on recent data
    print("\n[Step 6] Predicting current market regime...")
    latest_features = feature_array[-1:]

    current_regime = detector.predict_regime(latest_features)
    regime_probs = detector.predict_regime_probabilities(latest_features)

    print(f"  Current regime: {current_regime.value}")
    print(f"  Regime probabilities:")
    for regime, prob in regime_probs.items():
        print(f"    {regime.value}: {prob:.1%}")

    # Step 7: Save model
    print("\n[Step 7] Saving trained model...")
    model_path = "models/regime_detector"
    detector.save(model_path)
    print(f"  Model saved to: {model_path}")
    print(f"  Files created:")
    print(f"    - {model_path}.joblib (HMM model)")
    print(f"    - {model_path}_config.json (configuration)")
    print(f"    - {model_path}_arrays.npz (feature statistics)")

    # Step 8: Load model and verify
    print("\n[Step 8] Loading model and verifying...")
    loaded_detector = RegimeDetector.load(model_path)

    loaded_regime = loaded_detector.predict_regime(latest_features)
    assert loaded_regime == current_regime, "Loaded model produces different results!"
    print(f"  Verification: PASSED")
    print(f"  Loaded model predicts: {loaded_regime.value}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

**Acceptance Criteria**:
- [ ] Script runs without errors
- [ ] Output is informative and clear
- [ ] Demonstrates all key API features
- [ ] Has proper docstrings
- [ ] Uses realistic-looking synthetic data

**Estimated Effort**: 2 hours

---

#### 2. Add Integration Test

**File**: `tests/test_signals/test_integration.py`

**Why**: Verifies complete workflow works end-to-end.

**Implementation**:
```python
"""Integration tests for regime detection workflow.

These tests verify the complete pipeline from data to predictions,
ensuring all components work together correctly.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.models import Regime
from src.signals import (
    FeatureCalculator,
    FeatureSet,
    RegimeDetector,
)


class TestRegimeDetectionWorkflow:
    """Integration tests for complete regime detection workflow."""

    @pytest.fixture
    def synthetic_data(self) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
    ]:
        """Generate synthetic market data for testing."""
        n_days = 1500  # About 6 years of daily data
        dates = pd.date_range(end="2024-01-15", periods=n_days, freq="B")
        rng = np.random.default_rng(42)

        # VIX with regime structure
        vix_values = 15 + 10 * np.sin(np.linspace(0, 6 * np.pi, n_days))
        vix_values += rng.normal(scale=2, size=n_days)
        vix_df = pd.DataFrame({"vix": vix_values}, index=dates)

        # SPY with trend
        spy_returns = rng.normal(loc=0.0003, scale=0.01, size=n_days)
        spy_prices = 100 * np.exp(np.cumsum(spy_returns))
        spy_df = pd.DataFrame({"close": spy_prices}, index=dates)

        # Treasury yields
        t2y = 4.5 + rng.normal(scale=0.2, size=n_days)
        t10y = 4.0 + rng.normal(scale=0.2, size=n_days)
        treasury_df = pd.DataFrame(
            {"treasury_2y": t2y, "treasury_10y": t10y},
            index=dates,
        )

        # HY spreads
        hy_spread = 3.5 + rng.normal(scale=0.5, size=n_days)
        hy_spread_df = pd.DataFrame({"hy_spread": hy_spread}, index=dates)

        return vix_df, spy_df, treasury_df, hy_spread_df

    def test_complete_workflow(
        self, synthetic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test complete workflow from data to prediction."""
        vix_df, spy_df, treasury_df, hy_spread_df = synthetic_data

        # Step 1: Calculate features
        calculator = FeatureCalculator(lookback_days=252)
        features_df = calculator.calculate_feature_history(
            vix_df=vix_df,
            price_df=spy_df,
            treasury_df=treasury_df,
            hy_spread_df=hy_spread_df,
        )

        assert len(features_df) > 0, "Should generate features"
        assert all(
            col in features_df.columns for col in FeatureSet.feature_names()
        ), "Should have all features"

        # Step 2: Train detector
        detector = RegimeDetector(n_states=3, random_state=42)
        feature_array = features_df[FeatureSet.feature_names()].values
        detector.fit(feature_array)

        assert detector.is_fitted, "Detector should be fitted"

        # Step 3: Make prediction
        latest_features = feature_array[-1:]
        regime = detector.predict_regime(latest_features)

        assert isinstance(regime, Regime), "Should return Regime enum"

        # Step 4: Get probabilities
        probs = detector.predict_regime_probabilities(latest_features)

        assert len(probs) == 3, "Should have 3 regime probabilities"
        assert abs(sum(probs.values()) - 1.0) < 1e-6, "Probabilities should sum to 1"

    def test_save_load_preserves_workflow(
        self, synthetic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test that save/load cycle preserves predictions."""
        vix_df, spy_df, treasury_df, hy_spread_df = synthetic_data

        # Calculate features and train
        calculator = FeatureCalculator(lookback_days=252)
        features_df = calculator.calculate_feature_history(
            vix_df=vix_df,
            price_df=spy_df,
            treasury_df=treasury_df,
            hy_spread_df=hy_spread_df,
        )

        detector = RegimeDetector(n_states=3, random_state=42)
        feature_array = features_df[FeatureSet.feature_names()].values
        detector.fit(feature_array)

        # Get original prediction
        test_features = feature_array[-10:]
        original_regime = detector.predict_regime(test_features)
        original_probs = detector.predict_regime_probabilities(test_features)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            detector.save(str(model_path))

            loaded_detector = RegimeDetector.load(str(model_path))

        # Verify loaded model produces same results
        loaded_regime = loaded_detector.predict_regime(test_features)
        loaded_probs = loaded_detector.predict_regime_probabilities(test_features)

        assert loaded_regime == original_regime, "Regime should be preserved"
        for regime in Regime:
            assert (
                abs(loaded_probs[regime] - original_probs[regime]) < 1e-6
            ), f"Probabilities for {regime} should be preserved"

    def test_feature_calculation_reproducibility(
        self, synthetic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test that feature calculation is reproducible."""
        vix_df, spy_df, treasury_df, hy_spread_df = synthetic_data

        calculator1 = FeatureCalculator(lookback_days=252)
        calculator2 = FeatureCalculator(lookback_days=252)

        features1 = calculator1.calculate_feature_history(
            vix_df=vix_df,
            price_df=spy_df,
            treasury_df=treasury_df,
            hy_spread_df=hy_spread_df,
        )

        features2 = calculator2.calculate_feature_history(
            vix_df=vix_df,
            price_df=spy_df,
            treasury_df=treasury_df,
            hy_spread_df=hy_spread_df,
        )

        pd.testing.assert_frame_equal(features1, features2)

    def test_regime_transitions_make_sense(
        self, synthetic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test that regime transitions are reasonable."""
        vix_df, spy_df, treasury_df, hy_spread_df = synthetic_data

        # Calculate features and train
        calculator = FeatureCalculator(lookback_days=252)
        features_df = calculator.calculate_feature_history(
            vix_df=vix_df,
            price_df=spy_df,
            treasury_df=treasury_df,
            hy_spread_df=hy_spread_df,
        )

        detector = RegimeDetector(n_states=3, random_state=42)
        feature_array = features_df[FeatureSet.feature_names()].values
        detector.fit(feature_array)

        # Get transition matrix
        trans_matrix = detector.get_transition_matrix()

        # Verify reasonable transition properties
        # 1. Diagonal should be largest (regime persistence)
        for i in range(3):
            diagonal = trans_matrix[i, i]
            off_diagonal = [trans_matrix[i, j] for j in range(3) if j != i]
            assert diagonal >= max(off_diagonal), (
                f"State {i} should have highest probability of staying in same state"
            )

        # 2. Should have some regime switching (not 100% persistence)
        assert np.all(np.diag(trans_matrix) < 0.99), "Should allow regime transitions"
```

**Acceptance Criteria**:
- [ ] All integration tests pass
- [ ] Tests use realistic data volumes (1500+ samples)
- [ ] Tests verify complete workflows
- [ ] Tests check reproducibility

**Estimated Effort**: 3 hours

---

### MEDIUM Priority (Complete in Sprint 5)

#### 3. Add Feature Importance Method

**File**: `src/signals/regime.py`

**Implementation**:
```python
def get_feature_importance(self) -> dict[str, float]:
    """Calculate feature importance based on state separation.

    Returns a measure of how much each feature contributes to
    distinguishing between regimes. Calculated as the variance
    of feature means across states (normalized).

    Higher values indicate features that differ more across regimes,
    suggesting they are more important for regime classification.

    Returns:
        Dictionary mapping feature names to importance scores.
        Scores sum to 1.0.

    Raises:
        NotFittedError: If the model has not been fitted.

    Example:
        >>> detector = RegimeDetector()
        >>> detector.fit(features)
        >>> importance = detector.get_feature_importance()
        >>> print(f"VIX importance: {importance['feature_0']:.2%}")
    """
    self._validate_fitted()

    if self._model is None:
        raise NotFittedError("Model must be fitted before calculating importance.")

    # Get mean values for each state (standardized)
    state_means = self._model.means_  # shape: (n_states, n_features)

    # Calculate variance of each feature across states
    # Higher variance = more discriminative
    feature_variance = np.var(state_means, axis=0)

    # Normalize to sum to 1
    total_variance = feature_variance.sum()
    if total_variance < 1e-10:
        # All features have same mean across states (unlikely but possible)
        normalized = np.ones(len(feature_variance)) / len(feature_variance)
    else:
        normalized = feature_variance / total_variance

    # Convert to dictionary with feature names
    feature_names = FeatureSet.feature_names() if self._n_features == 9 else None

    if feature_names and len(feature_names) == self._n_features:
        return {name: float(imp) for name, imp in zip(feature_names, normalized)}
    else:
        return {f"feature_{i}": float(imp) for i, imp in enumerate(normalized)}
```

**Test**:
```python
def test_get_feature_importance(fitted_detector: RegimeDetector) -> None:
    """Test feature importance calculation."""
    importance = fitted_detector.get_feature_importance()

    # Should have importance for each feature
    assert len(importance) == fitted_detector.n_features

    # Should sum to 1
    assert abs(sum(importance.values()) - 1.0) < 1e-6

    # All values should be non-negative
    assert all(v >= 0 for v in importance.values())
```

**Estimated Effort**: 1 hour

---

#### 4. Add Model Diagnostics Method

**File**: `src/signals/regime.py`

**Implementation**:
```python
def get_diagnostics(self) -> dict[str, float | bool]:
    """Get model diagnostic metrics for monitoring.

    Returns metrics useful for assessing model quality and
    detecting potential issues in production:

    - converged: Whether EM algorithm converged during training
    - n_iterations: Number of EM iterations until convergence
    - state_persistence: Average diagonal of transition matrix
      (higher = regimes more persistent, lower = more switching)

    Returns:
        Dictionary of diagnostic metrics.

    Raises:
        NotFittedError: If the model has not been fitted.

    Example:
        >>> detector = RegimeDetector()
        >>> detector.fit(features)
        >>> diag = detector.get_diagnostics()
        >>> if not diag['converged']:
        ...     logger.warning("HMM did not converge - consider more iterations")
    """
    self._validate_fitted()

    if self._model is None:
        raise NotFittedError("Model must be fitted to get diagnostics.")

    # Get transition matrix
    trans_matrix = self.get_transition_matrix()

    diagnostics = {
        "converged": bool(self._model.monitor_.converged),
        "n_iterations": int(self._model.monitor_.iter),
        "state_persistence": float(np.diag(trans_matrix).mean()),
        "n_samples_trained": int(self._model.n_features_in_),  # Actually n_samples
        "n_features": int(self._n_features) if self._n_features else 0,
        "n_states": int(self.n_states),
    }

    return diagnostics
```

**Test**:
```python
def test_get_diagnostics(fitted_detector: RegimeDetector) -> None:
    """Test model diagnostics."""
    diag = fitted_detector.get_diagnostics()

    assert "converged" in diag
    assert "n_iterations" in diag
    assert "state_persistence" in diag

    # State persistence should be between 0 and 1
    assert 0 <= diag["state_persistence"] <= 1
```

**Estimated Effort**: 1 hour

---

#### 5. Add Prediction Metadata for Monitoring

**File**: `src/signals/regime.py`

**Implementation**:
```python
def get_prediction_metadata(self, features: np.ndarray) -> dict:
    """Get prediction with metadata for production monitoring.

    Returns prediction along with metadata useful for logging,
    monitoring, and debugging in production systems.

    Args:
        features: Feature array for prediction

    Returns:
        Dictionary containing:
        - regime: Predicted regime (str)
        - confidence: Maximum regime probability
        - probabilities: All regime probabilities
        - model_version: Model format version
        - n_features: Number of features used
        - feature_values: Input feature values (for drift detection)

    Raises:
        NotFittedError: If the model has not been fitted.
        FeatureDimensionError: If feature dimensions don't match.

    Example:
        >>> metadata = detector.get_prediction_metadata(features)
        >>> logger.info(f"Regime: {metadata['regime']}, "
        ...             f"Confidence: {metadata['confidence']:.1%}")
    """
    self._validate_fitted()
    features = self._validate_features(features)

    regime = self.predict_regime(features)
    regime_probs = self.predict_regime_probabilities(features)

    # Get feature values (last sample if multiple provided)
    feature_vals = features[-1].tolist() if features.ndim > 1 else features.tolist()

    return {
        "regime": regime.value,
        "confidence": max(regime_probs.values()),
        "probabilities": {r.value: p for r, p in regime_probs.items()},
        "model_version": MODEL_FORMAT_VERSION,
        "n_features": self._n_features,
        "feature_values": feature_vals,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
```

**Test**:
```python
def test_get_prediction_metadata(fitted_detector: RegimeDetector) -> None:
    """Test prediction metadata generation."""
    features = np.array([[10, 0.05, 0.5]])
    metadata = fitted_detector.get_prediction_metadata(features)

    assert "regime" in metadata
    assert "confidence" in metadata
    assert "probabilities" in metadata
    assert "feature_values" in metadata
    assert 0 <= metadata["confidence"] <= 1
```

**Estimated Effort**: 1 hour

---

### LOW Priority (Future Sprint)

#### 6. Extract Validation Logic

**Why**: Nice-to-have refactoring, but current code is already clean.

**When**: Defer until we have multiple models needing similar validation.

**Estimated Effort**: 2 hours

---

## Testing Checklist

Before considering Sprint 5 complete, verify:

- [ ] All existing tests pass (114 tests)
- [ ] New integration tests added and passing
- [ ] Example script runs without errors
- [ ] `pyrefly check` passes (0 errors)
- [ ] `ruff check .` passes
- [ ] `bandit -r src/signals/` passes
- [ ] Documentation updated (README, docstrings)

---

## Timeline Estimate

| Item | Priority | Effort | Cumulative |
|------|----------|--------|------------|
| Example script | HIGH | 2h | 2h |
| Integration tests | HIGH | 3h | 5h |
| Feature importance | MEDIUM | 1h | 6h |
| Model diagnostics | MEDIUM | 1h | 7h |
| Prediction metadata | MEDIUM | 1h | 8h |

**Total**: 8 hours (1 developer day)

**Recommended**: Complete HIGH items (5 hours) before merge, MEDIUM items (3 hours) before Sprint 5 end.

---

## Success Criteria

Sprint 5 ML code is **production deployment ready** when:

1. [ ] All HIGH priority items completed
2. [ ] Example script demonstrates complete workflow
3. [ ] Integration tests verify end-to-end correctness
4. [ ] All tests pass (existing + new)
5. [ ] Code quality checks pass (ruff, pyrefly, bandit)
6. [ ] Documentation updated
7. [ ] Code reviewed by at least one other developer

Once these criteria are met, the ML code can be confidently deployed to production.
