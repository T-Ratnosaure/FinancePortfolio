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
    FeatureDimensionError,
    NotFittedError,
    RegimeDetector,
    RegimeDetectorConfig,
    RegimeDetectorError,
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
    # Allocation
    "AllocationOptimizer",
    "RiskLimits",
    "AllocationError",
    "REGIME_ALLOCATIONS",
]
