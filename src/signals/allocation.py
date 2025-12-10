"""Allocation optimizer for PEA Portfolio regime-based allocation.

This module provides allocation recommendations based on detected market regimes,
enforcing hard-coded risk limits to ensure portfolio safety.
"""

from datetime import date

from pydantic import BaseModel, Field, model_validator

from src.data.models import (
    MAX_LEVERAGED_EXPOSURE,
    MAX_SINGLE_POSITION,
    MIN_CASH_BUFFER,
    REBALANCE_THRESHOLD,
    AllocationRecommendation,
    Regime,
)


class AllocationError(Exception):
    """Exception raised for allocation validation failures.

    Attributes:
        message: Description of the validation failure
        violations: List of specific constraint violations
    """

    def __init__(self, message: str, violations: list[str] | None = None) -> None:
        """Initialize allocation error.

        Args:
            message: Error description
            violations: List of specific constraint violations
        """
        self.violations = violations or []
        super().__init__(message)


class RiskLimits(BaseModel):
    """Risk limit configuration for portfolio allocation.

    Attributes:
        max_leveraged_exposure: Maximum combined weight for leveraged ETFs (LQQ + CL2)
        max_single_position: Maximum weight for any single position
        min_cash_buffer: Minimum required cash allocation
        rebalance_threshold: Minimum drift from target to trigger rebalancing
    """

    max_leveraged_exposure: float = Field(
        default=MAX_LEVERAGED_EXPOSURE,
        ge=0.0,
        le=1.0,
        description="Maximum combined weight for leveraged ETFs",
    )
    max_single_position: float = Field(
        default=MAX_SINGLE_POSITION,
        ge=0.0,
        le=1.0,
        description="Maximum weight for any single position",
    )
    min_cash_buffer: float = Field(
        default=MIN_CASH_BUFFER,
        ge=0.0,
        le=1.0,
        description="Minimum required cash allocation",
    )
    rebalance_threshold: float = Field(
        default=REBALANCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum drift from target to trigger rebalancing",
    )

    @model_validator(mode="after")
    def validate_limits_consistency(self) -> "RiskLimits":
        """Validate that risk limits are internally consistent."""
        if self.min_cash_buffer + self.max_leveraged_exposure > 1.0:
            raise ValueError(
                "min_cash_buffer + max_leveraged_exposure cannot exceed 1.0"
            )
        return self


# Regime-based target allocations (conservative by design)
REGIME_ALLOCATIONS: dict[Regime, dict[str, float]] = {
    Regime.RISK_ON: {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10},
    Regime.NEUTRAL: {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20},
    Regime.RISK_OFF: {"LQQ": 0.05, "CL2": 0.05, "WPEA": 0.60, "CASH": 0.30},
}

# Valid allocation symbols
VALID_SYMBOLS = {"LQQ", "CL2", "WPEA", "CASH"}
LEVERAGED_SYMBOLS = {"LQQ", "CL2"}


class AllocationOptimizer:
    """Optimizer for portfolio allocation based on market regime.

    Generates allocation recommendations that respect hard-coded risk limits
    and provides rebalancing calculations.

    Attributes:
        risk_limits: Risk limit configuration
    """

    def __init__(self, risk_limits: RiskLimits | None = None) -> None:
        """Initialize the allocation optimizer.

        Args:
            risk_limits: Optional custom risk limits. Uses defaults if not provided.
        """
        self.risk_limits = risk_limits or RiskLimits()

    def get_target_allocation(
        self,
        regime: Regime,
        confidence: float = 1.0,
        as_of_date: date | None = None,
    ) -> AllocationRecommendation:
        """Get target allocation for the given market regime.

        When confidence is less than 1.0, the allocation is blended toward
        NEUTRAL to reduce risk during uncertain periods.

        Args:
            regime: Detected market regime
            confidence: Confidence in regime detection (0.0 to 1.0).
                       Lower confidence scales allocation toward NEUTRAL.
            as_of_date: Date for the recommendation. Defaults to today.

        Returns:
            AllocationRecommendation with validated target weights

        Raises:
            AllocationError: If the resulting allocation violates risk limits
            ValueError: If confidence is not in [0, 1]
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {confidence}")

        if as_of_date is None:
            as_of_date = date.today()

        # Get base allocation for the regime
        target = REGIME_ALLOCATIONS[regime].copy()
        neutral = REGIME_ALLOCATIONS[Regime.NEUTRAL]

        # Blend toward neutral based on confidence
        if confidence < 1.0:
            blended = self._blend_allocations(target, neutral, confidence)
        else:
            blended = target

        # Validate the allocation
        is_valid, violations = self.validate_allocation(blended)
        if not is_valid:
            raise AllocationError(
                f"Target allocation violates risk limits: {', '.join(violations)}",
                violations=violations,
            )

        # Generate reasoning
        reasoning = self._generate_reasoning(regime, confidence, blended)

        return AllocationRecommendation(
            date=as_of_date,
            regime=regime,
            lqq_weight=blended["LQQ"],
            cl2_weight=blended["CL2"],
            wpea_weight=blended["WPEA"],
            cash_weight=blended["CASH"],
            confidence=confidence,
            reasoning=reasoning,
        )

    def calculate_rebalance_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
    ) -> list[dict[str, str | float]]:
        """Calculate trades needed to rebalance portfolio.

        Args:
            current_weights: Current position weights by symbol
            target_weights: Target position weights by symbol
            portfolio_value: Total portfolio value in currency units

        Returns:
            List of trade dictionaries with keys:
                - symbol: Asset symbol
                - action: 'BUY' or 'SELL'
                - amount: Trade amount in currency units (absolute value)

        Raises:
            AllocationError: If current or target weights are invalid
            ValueError: If portfolio_value is not positive
        """
        if portfolio_value <= 0:
            raise ValueError(f"Portfolio value must be positive, got {portfolio_value}")

        # Validate inputs
        self._validate_weight_dict(current_weights, "current_weights")
        self._validate_weight_dict(target_weights, "target_weights")

        # Validate target allocation against risk limits
        is_valid, violations = self.validate_allocation(target_weights)
        if not is_valid:
            raise AllocationError(
                f"Target allocation violates risk limits: {', '.join(violations)}",
                violations=violations,
            )

        trades: list[dict[str, str | float]] = []

        # Calculate trades for each symbol
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in sorted(all_symbols):  # Sort for deterministic order
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            diff = target - current

            if abs(diff) > 1e-6:  # Ignore negligible differences
                amount = abs(diff) * portfolio_value
                action = "BUY" if diff > 0 else "SELL"
                trades.append(
                    {
                        "symbol": symbol,
                        "action": action,
                        "amount": round(amount, 2),
                    }
                )

        return trades

    def needs_rebalancing(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> bool:
        """Check if portfolio needs rebalancing.

        Rebalancing is needed if any position drifts more than
        REBALANCE_THRESHOLD from its target weight.

        Args:
            current_weights: Current position weights by symbol
            target_weights: Target position weights by symbol

        Returns:
            True if any position exceeds the drift threshold
        """
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            drift = abs(current - target)

            if drift > self.risk_limits.rebalance_threshold:
                return True

        return False

    def validate_allocation(
        self,
        weights: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """Validate allocation against risk limits.

        Checks:
        1. All weights are non-negative
        2. Weights sum to 1.0 (within tolerance)
        3. Combined leveraged exposure does not exceed limit
        4. No single position exceeds maximum position size
        5. Cash buffer meets minimum requirement

        Args:
            weights: Position weights by symbol

        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations: list[str] = []

        # Check for negative weights
        for symbol, weight in weights.items():
            if weight < 0:
                violations.append(f"Negative weight for {symbol}: {weight:.4f}")

        # Check weights sum to 1.0
        total_weight = sum(weights.values())
        if not (0.99 <= total_weight <= 1.01):
            violations.append(f"Weights sum to {total_weight:.4f}, expected 1.0")

        # Check leveraged exposure
        leveraged_exposure = sum(
            weights.get(symbol, 0.0) for symbol in LEVERAGED_SYMBOLS
        )
        if leveraged_exposure > self.risk_limits.max_leveraged_exposure + 1e-6:
            violations.append(
                f"Leveraged exposure {leveraged_exposure:.4f} exceeds "
                f"limit {self.risk_limits.max_leveraged_exposure:.4f}"
            )

        # Check single position limits (only for leveraged ETFs)
        # WPEA is the core safe holding - not subject to max single position limit
        for symbol, weight in weights.items():
            if (
                symbol in LEVERAGED_SYMBOLS
                and weight > self.risk_limits.max_single_position + 1e-6
            ):
                violations.append(
                    f"Position {symbol} weight {weight:.4f} exceeds "
                    f"limit {self.risk_limits.max_single_position:.4f}"
                )

        # Check cash buffer
        cash_weight = weights.get("CASH", 0.0)
        if cash_weight < self.risk_limits.min_cash_buffer - 1e-6:
            violations.append(
                f"Cash buffer {cash_weight:.4f} below "
                f"minimum {self.risk_limits.min_cash_buffer:.4f}"
            )

        return len(violations) == 0, violations

    def _blend_allocations(
        self,
        target: dict[str, float],
        neutral: dict[str, float],
        confidence: float,
    ) -> dict[str, float]:
        """Blend target allocation toward neutral based on confidence.

        Formula: blended = confidence * target + (1 - confidence) * neutral

        Args:
            target: Target regime allocation
            neutral: Neutral regime allocation
            confidence: Confidence weight (0 to 1)

        Returns:
            Blended allocation dictionary
        """
        blended: dict[str, float] = {}
        for symbol in VALID_SYMBOLS:
            target_weight = target.get(symbol, 0.0)
            neutral_weight = neutral.get(symbol, 0.0)
            blended[symbol] = (
                confidence * target_weight + (1 - confidence) * neutral_weight
            )

        return blended

    def _validate_weight_dict(
        self,
        weights: dict[str, float],
        name: str,
    ) -> None:
        """Validate a weight dictionary structure.

        Args:
            weights: Weight dictionary to validate
            name: Name for error messages

        Raises:
            AllocationError: If weights are invalid
        """
        if not weights:
            raise AllocationError(f"{name} cannot be empty")

        for symbol, weight in weights.items():
            if not isinstance(weight, (int, float)):
                raise AllocationError(
                    f"{name}[{symbol}] must be numeric, got {type(weight).__name__}"
                )

    def _generate_reasoning(
        self,
        regime: Regime,
        confidence: float,
        allocation: dict[str, float],
    ) -> str:
        """Generate human-readable reasoning for the allocation.

        Args:
            regime: Detected market regime
            confidence: Regime confidence
            allocation: Final allocation weights

        Returns:
            Explanation string for the recommendation
        """
        leveraged_total = allocation.get("LQQ", 0.0) + allocation.get("CL2", 0.0)

        if regime == Regime.RISK_ON:
            regime_description = "bullish conditions with low volatility"
        elif regime == Regime.RISK_OFF:
            regime_description = "defensive positioning due to elevated risk"
        else:
            regime_description = "mixed signals suggesting balanced positioning"

        confidence_note = ""
        if confidence < 1.0:
            confidence_note = (
                f" Allocation blended toward neutral due to "
                f"{confidence:.0%} confidence."
            )

        return (
            f"Regime: {regime.value} - {regime_description}. "
            f"Leveraged exposure: {leveraged_total:.0%}, "
            f"Cash buffer: {allocation.get('CASH', 0.0):.0%}.{confidence_note}"
        )
