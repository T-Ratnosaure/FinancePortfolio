"""Signal generation for regime detection and allocation."""

from src.signals.allocation import (
    REGIME_ALLOCATIONS,
    AllocationError,
    AllocationOptimizer,
    RiskLimits,
)
from src.signals.features import (
    FeatureCalculator,
    FeatureEngineer,
    FeatureSet,
    InsufficientDataError,
)
from src.signals.regime import (
    ABSOLUTE_MIN_SAMPLES,
    MIN_SAMPLES_PER_PARAMETER,
    FeatureDimensionError,
    InsufficientSamplesError,
    NotFittedError,
    RegimeDetector,
    RegimeDetectorConfig,
    RegimeDetectorError,
    calculate_hmm_parameters,
    calculate_min_samples,
)

__all__ = [
    # Features
    "FeatureSet",
    "FeatureCalculator",
    "FeatureEngineer",
    "InsufficientDataError",
    # Regime
    "RegimeDetector",
    "RegimeDetectorConfig",
    "RegimeDetectorError",
    "NotFittedError",
    "FeatureDimensionError",
    "InsufficientSamplesError",
    "calculate_hmm_parameters",
    "calculate_min_samples",
    "MIN_SAMPLES_PER_PARAMETER",
    "ABSOLUTE_MIN_SAMPLES",
    # Allocation
    "AllocationOptimizer",
    "RiskLimits",
    "AllocationError",
    "REGIME_ALLOCATIONS",
]
