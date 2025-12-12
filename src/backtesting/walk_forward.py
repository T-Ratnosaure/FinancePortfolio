"""Walk-forward validation for backtesting with proper bias prevention.

This module implements walk-forward validation methodology for testing the
regime-based allocation strategy. Walk-forward validation prevents look-ahead
bias by ensuring the model is always trained on historical data only.

Key principles:
1. Training window precedes test window with no overlap
2. Model is frozen during test period (no parameter updates)
3. Signal at close day t, execute at open day t+1
4. Minimum sample size enforced for HMM training (1,700+ samples)

Walk-Forward Timeline Example:
```
Train: 2015-01 to 2020-01 -> Test: 2020-01 to 2021-01
Train: 2015-07 to 2020-07 -> Test: 2020-07 to 2021-07
Train: 2016-01 to 2021-01 -> Test: 2021-01 to 2022-01
... continue through 2024
```
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from src.signals.allocation import AllocationOptimizer
from src.signals.regime import (
    RegimeDetector,
    calculate_min_samples,
)

logger = logging.getLogger(__name__)

# Trading days per year (approximate)
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21

# Default walk-forward parameters
# Training needs at least 1700 samples (7 years) for HMM stability
DEFAULT_TRAIN_YEARS = 7
DEFAULT_TEST_YEARS = 1
DEFAULT_STEP_MONTHS = 6

# Minimum training samples for HMM (per P0-04 requirements)
# This ensures approximately 7 years of daily data for 3-state, 9-feature HMM
MIN_HMM_TRAINING_SAMPLES = 1700


class LookaheadBiasError(Exception):
    """Raised when potential look-ahead bias is detected.

    Look-ahead bias occurs when information from the future is used to make
    decisions in the past. This is a critical error in backtesting that leads
    to unrealistically optimistic performance estimates.

    Common causes:
    - Using test period data in training
    - Features calculated with future information
    - Prediction date before feature calculation date
    """

    def __init__(self, message: str, details: dict[str, str] | None = None) -> None:
        """Initialize look-ahead bias error.

        Args:
            message: Error description
            details: Optional dictionary with specific dates/values causing the issue
        """
        self.details = details or {}
        super().__init__(message)


@dataclass(frozen=True)
class WalkForwardWindow:
    """A single walk-forward validation window.

    This dataclass represents one window in the walk-forward validation process.
    It is immutable (frozen) to prevent accidental modification after creation.

    The window defines:
    - Training period: Historical data used to fit the model
    - Testing period: Forward data used to evaluate model performance

    CRITICAL: The training period must end before the testing period begins.
    There must be no overlap between training and testing data.

    Attributes:
        window_id: Unique identifier for this window (0-indexed)
        train_start: First date of training period (inclusive)
        train_end: Last date of training period (inclusive)
        test_start: First date of testing period (inclusive)
        test_end: Last date of testing period (inclusive)
    """

    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    _validated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate window dates after initialization."""
        # Validate date ordering
        if self.train_start >= self.train_end:
            raise ValueError(
                f"train_start ({self.train_start}) must be before "
                f"train_end ({self.train_end})"
            )
        if self.test_start >= self.test_end:
            raise ValueError(
                f"test_start ({self.test_start}) must be before "
                f"test_end ({self.test_end})"
            )
        if self.train_end > self.test_start:
            raise ValueError(
                f"train_end ({self.train_end}) must be on or before "
                f"test_start ({self.test_start}) to prevent data leakage"
            )

        # Mark as validated (use object.__setattr__ because dataclass is frozen)
        object.__setattr__(self, "_validated", True)

    @property
    def train_days(self) -> int:
        """Calculate number of calendar days in training period."""
        return (self.train_end - self.train_start).days + 1

    @property
    def test_days(self) -> int:
        """Calculate number of calendar days in testing period."""
        return (self.test_end - self.test_start).days + 1

    @property
    def estimated_train_samples(self) -> int:
        """Estimate number of trading days in training period.

        Assumes approximately 252 trading days per 365 calendar days.
        """
        calendar_days = self.train_days
        return int(calendar_days * TRADING_DAYS_PER_YEAR / 365)

    @property
    def estimated_test_samples(self) -> int:
        """Estimate number of trading days in testing period.

        Assumes approximately 252 trading days per 365 calendar days.
        """
        calendar_days = self.test_days
        return int(calendar_days * TRADING_DAYS_PER_YEAR / 365)

    def has_sufficient_training_samples(
        self, min_samples: int = MIN_HMM_TRAINING_SAMPLES
    ) -> bool:
        """Check if training period has sufficient samples for HMM.

        Args:
            min_samples: Minimum required samples (default: 1,700 for HMM)

        Returns:
            True if estimated samples >= min_samples
        """
        return self.estimated_train_samples >= min_samples

    def to_dict(self) -> dict[str, str | int]:
        """Convert window to dictionary representation.

        Returns:
            Dictionary with window details for serialization
        """
        return {
            "window_id": self.window_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "estimated_train_samples": self.estimated_train_samples,
            "estimated_test_samples": self.estimated_test_samples,
        }


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward validation.

    Attributes:
        train_years: Number of years for training window (default: 5)
        test_years: Number of years for testing window (default: 1)
        step_months: Number of months to step forward between windows (default: 6)
        min_training_samples: Minimum samples required for HMM training
        execution_delay_days: Days between signal and execution (default: 1)
    """

    train_years: int = Field(default=DEFAULT_TRAIN_YEARS, ge=1, le=20)
    test_years: int = Field(default=DEFAULT_TEST_YEARS, ge=1, le=5)
    step_months: int = Field(default=DEFAULT_STEP_MONTHS, ge=1, le=24)
    min_training_samples: int = Field(default=MIN_HMM_TRAINING_SAMPLES, ge=100)
    execution_delay_days: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Days between signal (close day t) and execution (day t+n)",
    )

    @model_validator(mode="after")
    def validate_window_sizes(self) -> "WalkForwardConfig":
        """Validate that training window can accumulate sufficient samples."""
        # Estimate training samples
        estimated_samples = self.train_years * TRADING_DAYS_PER_YEAR
        if estimated_samples < self.min_training_samples:
            raise ValueError(
                f"Training window of {self.train_years} years "
                f"(~{estimated_samples} samples) is insufficient. "
                f"Need at least {self.min_training_samples} samples. "
                f"Increase train_years or reduce min_training_samples."
            )
        return self


class WalkForwardValidator:
    """Walk-forward validation manager for backtesting.

    This class manages the walk-forward validation process, ensuring proper
    temporal separation between training and testing periods to prevent
    look-ahead bias.

    Key responsibilities:
    1. Generate non-overlapping train/test windows
    2. Validate minimum sample requirements for HMM training
    3. Prevent look-ahead bias through date validation
    4. Integrate with RegimeDetector and AllocationOptimizer

    Example usage:
        >>> validator = WalkForwardValidator()
        >>> windows = validator.generate_windows(
        ...     start_date=date(2015, 1, 1),
        ...     end_date=date(2024, 12, 31)
        ... )
        >>> for window in windows:
        ...     # Train model on window.train_start to window.train_end
        ...     # Test model on window.test_start to window.test_end
        ...     pass

    Attributes:
        config: Walk-forward configuration parameters
    """

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
        regime_detector: RegimeDetector | None = None,
        allocation_optimizer: AllocationOptimizer | None = None,
    ) -> None:
        """Initialize walk-forward validator.

        Args:
            config: Optional configuration. Uses defaults if not provided.
            regime_detector: Optional pre-configured RegimeDetector
            allocation_optimizer: Optional pre-configured AllocationOptimizer
        """
        self.config = config or WalkForwardConfig()
        self._regime_detector = regime_detector
        self._allocation_optimizer = allocation_optimizer or AllocationOptimizer()

    @property
    def regime_detector(self) -> RegimeDetector | None:
        """Get the current regime detector (may be None if not set)."""
        return self._regime_detector

    @property
    def allocation_optimizer(self) -> AllocationOptimizer:
        """Get the allocation optimizer."""
        return self._allocation_optimizer

    def generate_windows(
        self,
        start_date: date,
        end_date: date,
        train_years: int | None = None,
        test_years: int | None = None,
        step_months: int | None = None,
    ) -> list[WalkForwardWindow]:
        """Generate walk-forward validation windows.

        Creates a sequence of non-overlapping train/test windows stepping
        forward through time. Each window uses only historical data for
        training and evaluates on forward data.

        Timeline Example (default parameters):
        ```
        Window 0: Train 2015-01 to 2020-01 -> Test 2020-01 to 2021-01
        Window 1: Train 2015-07 to 2020-07 -> Test 2020-07 to 2021-07
        Window 2: Train 2016-01 to 2021-01 -> Test 2021-01 to 2022-01
        ...
        ```

        Args:
            start_date: Earliest date for training data (inclusive)
            end_date: Latest date for testing data (inclusive)
            train_years: Override config train_years (optional)
            test_years: Override config test_years (optional)
            step_months: Override config step_months (optional)

        Returns:
            List of WalkForwardWindow objects ordered chronologically

        Raises:
            ValueError: If date range is insufficient for even one window
        """
        # Use overrides or config defaults
        train_yrs = train_years if train_years is not None else self.config.train_years
        test_yrs = test_years if test_years is not None else self.config.test_years
        step_mo = step_months if step_months is not None else self.config.step_months

        # Validate date range
        total_days = (end_date - start_date).days
        min_days_needed = (train_yrs + test_yrs) * 365
        if total_days < min_days_needed:
            raise ValueError(
                f"Date range ({total_days} days) is insufficient for "
                f"{train_yrs} year training + {test_yrs} year testing "
                f"({min_days_needed} days needed)."
            )

        windows: list[WalkForwardWindow] = []
        window_id = 0

        # Start with first window
        current_train_start = start_date

        while True:
            # Calculate window boundaries
            train_end = self._add_years(current_train_start, train_yrs)
            test_start = train_end  # No gap, but also no overlap
            test_end = self._add_years(test_start, test_yrs)

            # Check if test window fits within date range
            if test_end > end_date:
                # Try to fit a final window by adjusting test_end
                test_end = end_date
                # Only create window if test period is at least 6 months
                test_duration = (test_end - test_start).days
                if test_duration < 180:  # Less than ~6 months
                    logger.info(
                        f"Skipping final window: test period only {test_duration} days"
                    )
                    break

            # Create window
            try:
                window = WalkForwardWindow(
                    window_id=window_id,
                    train_start=current_train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
                windows.append(window)
                logger.debug(
                    f"Created window {window_id}: "
                    f"Train {current_train_start} to {train_end}, "
                    f"Test {test_start} to {test_end}"
                )
                window_id += 1
            except ValueError as e:
                logger.warning(f"Invalid window configuration: {e}")
                break

            # Step forward
            current_train_start = self._add_months(current_train_start, step_mo)

            # Safety check: don't create infinite windows
            if window_id > 100:
                logger.warning("Generated 100 windows, stopping to prevent runaway")
                break

            # Check if next window would exceed date range
            next_test_end = self._add_years(
                self._add_years(current_train_start, train_yrs), test_yrs
            )
            if next_test_end > end_date and test_end >= end_date:
                break

        if not windows:
            raise ValueError(
                f"Could not generate any valid windows for date range "
                f"{start_date} to {end_date} with current configuration."
            )

        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows

    def validate_no_lookahead(
        self,
        window: WalkForwardWindow,
        features_date: date,
        prediction_date: date,
    ) -> bool:
        """Validate that no look-ahead bias exists in feature/prediction dates.

        This method enforces the critical rule: features used for prediction
        must be calculated from data available BEFORE the prediction is made.

        Execution timing rule:
        - Signal at close day t (using data up to and including day t)
        - Execute at open day t+1 (or with configured delay)

        Args:
            window: The walk-forward window being validated
            features_date: Date of the most recent data used in features
            prediction_date: Date when the prediction/signal is generated

        Returns:
            True if no look-ahead bias detected

        Raises:
            LookaheadBiasError: If look-ahead bias is detected
        """
        # Check 1: Features date must be on or before prediction date
        if features_date > prediction_date:
            raise LookaheadBiasError(
                f"Features date ({features_date}) is after prediction date "
                f"({prediction_date}). This indicates look-ahead bias.",
                details={
                    "features_date": features_date.isoformat(),
                    "prediction_date": prediction_date.isoformat(),
                    "window_id": str(window.window_id),
                },
            )

        # Check 2: Prediction date must be within test period
        if prediction_date < window.test_start or prediction_date > window.test_end:
            raise LookaheadBiasError(
                f"Prediction date ({prediction_date}) is outside test period "
                f"({window.test_start} to {window.test_end}).",
                details={
                    "prediction_date": prediction_date.isoformat(),
                    "test_start": window.test_start.isoformat(),
                    "test_end": window.test_end.isoformat(),
                    "window_id": str(window.window_id),
                },
            )

        # Check 3: Features must not use data from future relative to prediction
        # The features_date should be at most prediction_date
        # (we calculate features at close, then make prediction)
        if features_date > prediction_date:
            raise LookaheadBiasError(
                f"Features calculated with future data. "
                f"Features date: {features_date}, Prediction date: {prediction_date}",
                details={
                    "features_date": features_date.isoformat(),
                    "prediction_date": prediction_date.isoformat(),
                },
            )

        return True

    def validate_execution_timing(
        self,
        signal_date: date,
        execution_date: date,
    ) -> bool:
        """Validate proper execution timing (signal before execution).

        The rule is: Signal at close day t, execute at open day t+N
        where N is the configured execution_delay_days (default: 1).

        Args:
            signal_date: Date when signal was generated (at market close)
            execution_date: Date when trade was executed (at market open)

        Returns:
            True if timing is valid

        Raises:
            LookaheadBiasError: If execution precedes or equals signal
        """
        min_delay = self.config.execution_delay_days
        actual_delay = (execution_date - signal_date).days

        if actual_delay < min_delay:
            raise LookaheadBiasError(
                f"Execution date ({execution_date}) is too close to signal date "
                f"({signal_date}). Minimum delay is {min_delay} day(s), "
                f"but actual delay is {actual_delay} day(s).",
                details={
                    "signal_date": signal_date.isoformat(),
                    "execution_date": execution_date.isoformat(),
                    "min_delay_days": str(min_delay),
                    "actual_delay_days": str(actual_delay),
                },
            )

        return True

    def validate_training_data(
        self,
        window: WalkForwardWindow,
        data_start_date: date,
        data_end_date: date,
        n_features: int = 9,
        covariance_type: Literal["spherical", "diag", "full", "tied"] = "full",
    ) -> tuple[bool, list[str]]:
        """Validate training data for a window meets HMM requirements.

        Checks:
        1. Data covers the training period
        2. Sufficient samples for HMM parameter estimation
        3. No overlap with test period

        Args:
            window: Walk-forward window to validate
            data_start_date: Earliest date in training data
            data_end_date: Latest date in training data
            n_features: Number of features for HMM (default: 9)
            covariance_type: HMM covariance type (default: 'full')

        Returns:
            Tuple of (is_valid, list of issues found)
        """
        issues: list[str] = []

        # Check data covers training period
        if data_start_date > window.train_start:
            issues.append(
                f"Training data starts ({data_start_date}) after window "
                f"train_start ({window.train_start})"
            )

        if data_end_date < window.train_end:
            issues.append(
                f"Training data ends ({data_end_date}) before window "
                f"train_end ({window.train_end})"
            )

        # Check no test period overlap
        if data_end_date >= window.test_start:
            issues.append(
                f"Training data ({data_end_date}) extends into test period "
                f"(starts {window.test_start}). This causes look-ahead bias!"
            )

        # Check sample size requirements
        min_samples = calculate_min_samples(
            n_states=3, n_features=n_features, covariance_type=covariance_type
        )
        estimated_samples = window.estimated_train_samples

        if estimated_samples < min_samples:
            issues.append(
                f"Insufficient training samples. Window has ~{estimated_samples} "
                f"samples, but HMM requires {min_samples} samples minimum. "
                f"Training period needs to be longer."
            )

        return len(issues) == 0, issues

    def get_execution_date(self, signal_date: date) -> date:
        """Calculate execution date from signal date.

        Following the rule: Signal at close day t, execute at open day t+N.

        Args:
            signal_date: Date when signal was generated

        Returns:
            Date when trade should be executed
        """
        return signal_date + timedelta(days=self.config.execution_delay_days)

    def filter_valid_windows(
        self,
        windows: list[WalkForwardWindow],
        n_features: int = 9,
        covariance_type: Literal["spherical", "diag", "full", "tied"] = "full",
    ) -> list[WalkForwardWindow]:
        """Filter windows to only those with sufficient training samples.

        Args:
            windows: List of windows to filter
            n_features: Number of features for HMM
            covariance_type: HMM covariance type

        Returns:
            List of windows with sufficient training data
        """
        min_samples = calculate_min_samples(
            n_states=3, n_features=n_features, covariance_type=covariance_type
        )

        valid_windows = []
        for window in windows:
            if window.has_sufficient_training_samples(min_samples):
                valid_windows.append(window)
            else:
                logger.warning(
                    f"Window {window.window_id} has insufficient samples "
                    f"({window.estimated_train_samples} < {min_samples})"
                )

        return valid_windows

    def summarize_windows(
        self, windows: list[WalkForwardWindow]
    ) -> dict[str, int | float | str]:
        """Generate summary statistics for walk-forward windows.

        Args:
            windows: List of windows to summarize

        Returns:
            Dictionary with summary statistics
        """
        if not windows:
            return {"n_windows": 0, "message": "No windows to summarize"}

        train_samples = [w.estimated_train_samples for w in windows]
        test_samples = [w.estimated_test_samples for w in windows]

        return {
            "n_windows": len(windows),
            "first_train_start": windows[0].train_start.isoformat(),
            "last_test_end": windows[-1].test_end.isoformat(),
            "avg_train_samples": int(np.mean(train_samples)),
            "min_train_samples": min(train_samples),
            "max_train_samples": max(train_samples),
            "avg_test_samples": int(np.mean(test_samples)),
            "min_test_samples": min(test_samples),
            "max_test_samples": max(test_samples),
            "total_test_coverage_days": sum(w.test_days for w in windows),
        }

    @staticmethod
    def _add_years(d: date, years: int) -> date:
        """Add years to a date, handling leap years.

        Args:
            d: Starting date
            years: Number of years to add

        Returns:
            Date with years added
        """
        try:
            return d.replace(year=d.year + years)
        except ValueError:
            # Handle Feb 29 -> Feb 28 for non-leap years
            return d.replace(year=d.year + years, day=28)

    @staticmethod
    def _add_months(d: date, months: int) -> date:
        """Add months to a date, handling month length variations.

        Args:
            d: Starting date
            months: Number of months to add

        Returns:
            Date with months added
        """
        # Calculate new year and month
        new_month = d.month + months
        new_year = d.year + (new_month - 1) // 12
        new_month = ((new_month - 1) % 12) + 1

        # Handle day overflow (e.g., Jan 31 + 1 month = Feb 28)
        import calendar

        max_day = calendar.monthrange(new_year, new_month)[1]
        new_day = min(d.day, max_day)

        return date(new_year, new_month, new_day)
