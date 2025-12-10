"""Tests for allocation optimizer module."""

from datetime import date

import pytest
from pydantic import ValidationError

from src.data.models import Regime
from src.signals.allocation import (
    REGIME_ALLOCATIONS,
    AllocationError,
    AllocationOptimizer,
    RiskLimits,
)


class TestRiskLimits:
    """Tests for RiskLimits model."""

    def test_default_values(self) -> None:
        """Test default risk limits match expected values."""
        limits = RiskLimits()
        assert limits.max_leveraged_exposure == 0.30
        assert limits.max_single_position == 0.25
        assert limits.min_cash_buffer == 0.10
        assert limits.rebalance_threshold == 0.05

    def test_custom_limits(self) -> None:
        """Test creating custom risk limits."""
        limits = RiskLimits(
            max_leveraged_exposure=0.20,
            max_single_position=0.15,
            min_cash_buffer=0.15,
            rebalance_threshold=0.03,
        )
        assert limits.max_leveraged_exposure == 0.20
        assert limits.max_single_position == 0.15
        assert limits.min_cash_buffer == 0.15
        assert limits.rebalance_threshold == 0.03

    def test_invalid_limits_negative(self) -> None:
        """Test that negative limits are rejected."""
        with pytest.raises(ValidationError):
            RiskLimits(max_leveraged_exposure=-0.10)

    def test_invalid_limits_over_one(self) -> None:
        """Test that limits over 1.0 are rejected."""
        with pytest.raises(ValidationError):
            RiskLimits(max_single_position=1.5)

    def test_inconsistent_limits_rejected(self) -> None:
        """Test that inconsistent limits are rejected."""
        with pytest.raises(
            ValidationError, match="min_cash_buffer.*max_leveraged_exposure"
        ):
            RiskLimits(
                min_cash_buffer=0.60,
                max_leveraged_exposure=0.50,  # Sum > 1.0
            )


class TestRegimeAllocations:
    """Tests for REGIME_ALLOCATIONS constant."""

    def test_all_regimes_defined(self) -> None:
        """Test all regimes have allocations defined."""
        assert Regime.RISK_ON in REGIME_ALLOCATIONS
        assert Regime.NEUTRAL in REGIME_ALLOCATIONS
        assert Regime.RISK_OFF in REGIME_ALLOCATIONS

    def test_allocations_sum_to_one(self) -> None:
        """Test each regime allocation sums to 1.0."""
        for regime, allocation in REGIME_ALLOCATIONS.items():
            total = sum(allocation.values())
            assert abs(total - 1.0) < 0.001, f"{regime} allocation sums to {total}"

    def test_risk_on_is_most_aggressive(self) -> None:
        """Test RISK_ON has highest leveraged exposure."""
        risk_on_leveraged = (
            REGIME_ALLOCATIONS[Regime.RISK_ON]["LQQ"]
            + REGIME_ALLOCATIONS[Regime.RISK_ON]["CL2"]
        )
        neutral_leveraged = (
            REGIME_ALLOCATIONS[Regime.NEUTRAL]["LQQ"]
            + REGIME_ALLOCATIONS[Regime.NEUTRAL]["CL2"]
        )
        risk_off_leveraged = (
            REGIME_ALLOCATIONS[Regime.RISK_OFF]["LQQ"]
            + REGIME_ALLOCATIONS[Regime.RISK_OFF]["CL2"]
        )

        assert risk_on_leveraged > neutral_leveraged
        assert neutral_leveraged > risk_off_leveraged

    def test_risk_off_has_highest_cash(self) -> None:
        """Test RISK_OFF has highest cash allocation."""
        assert (
            REGIME_ALLOCATIONS[Regime.RISK_OFF]["CASH"]
            > REGIME_ALLOCATIONS[Regime.NEUTRAL]["CASH"]
        )
        assert (
            REGIME_ALLOCATIONS[Regime.NEUTRAL]["CASH"]
            > REGIME_ALLOCATIONS[Regime.RISK_ON]["CASH"]
        )


class TestAllocationOptimizer:
    """Tests for AllocationOptimizer class."""

    def test_init_default_limits(self) -> None:
        """Test optimizer initializes with default limits."""
        optimizer = AllocationOptimizer()
        assert optimizer.risk_limits.max_leveraged_exposure == 0.30

    def test_init_custom_limits(self) -> None:
        """Test optimizer accepts custom limits."""
        custom_limits = RiskLimits(max_leveraged_exposure=0.25)
        optimizer = AllocationOptimizer(risk_limits=custom_limits)
        assert optimizer.risk_limits.max_leveraged_exposure == 0.25


class TestGetTargetAllocation:
    """Tests for get_target_allocation method."""

    def test_risk_on_allocation(self) -> None:
        """Test RISK_ON regime returns expected allocation."""
        optimizer = AllocationOptimizer()
        rec = optimizer.get_target_allocation(Regime.RISK_ON)

        assert rec.regime == Regime.RISK_ON
        assert rec.lqq_weight == 0.15
        assert rec.cl2_weight == 0.15
        assert rec.wpea_weight == 0.60
        assert rec.cash_weight == 0.10
        assert rec.confidence == 1.0

    def test_neutral_allocation(self) -> None:
        """Test NEUTRAL regime returns expected allocation."""
        optimizer = AllocationOptimizer()
        rec = optimizer.get_target_allocation(Regime.NEUTRAL)

        assert rec.regime == Regime.NEUTRAL
        assert rec.lqq_weight == 0.10
        assert rec.cl2_weight == 0.10
        assert rec.wpea_weight == 0.60
        assert rec.cash_weight == 0.20

    def test_risk_off_allocation(self) -> None:
        """Test RISK_OFF regime returns expected allocation."""
        optimizer = AllocationOptimizer()
        rec = optimizer.get_target_allocation(Regime.RISK_OFF)

        assert rec.regime == Regime.RISK_OFF
        assert rec.lqq_weight == 0.05
        assert rec.cl2_weight == 0.05
        assert rec.wpea_weight == 0.60
        assert rec.cash_weight == 0.30

    def test_custom_date(self) -> None:
        """Test allocation with custom date."""
        optimizer = AllocationOptimizer()
        test_date = date(2024, 6, 15)
        rec = optimizer.get_target_allocation(Regime.NEUTRAL, as_of_date=test_date)

        assert rec.date == test_date

    def test_confidence_blending_full_confidence(self) -> None:
        """Test that confidence=1.0 returns unblended allocation."""
        optimizer = AllocationOptimizer()
        rec = optimizer.get_target_allocation(Regime.RISK_ON, confidence=1.0)

        assert rec.lqq_weight == REGIME_ALLOCATIONS[Regime.RISK_ON]["LQQ"]
        assert rec.cl2_weight == REGIME_ALLOCATIONS[Regime.RISK_ON]["CL2"]

    def test_confidence_blending_zero_confidence(self) -> None:
        """Test that confidence=0.0 returns neutral allocation."""
        optimizer = AllocationOptimizer()
        rec = optimizer.get_target_allocation(Regime.RISK_ON, confidence=0.0)

        # With 0 confidence, should be fully neutral
        assert rec.lqq_weight == REGIME_ALLOCATIONS[Regime.NEUTRAL]["LQQ"]
        assert rec.cl2_weight == REGIME_ALLOCATIONS[Regime.NEUTRAL]["CL2"]
        assert rec.cash_weight == REGIME_ALLOCATIONS[Regime.NEUTRAL]["CASH"]

    def test_confidence_blending_half_confidence(self) -> None:
        """Test that confidence=0.5 blends allocation halfway."""
        optimizer = AllocationOptimizer()
        rec = optimizer.get_target_allocation(Regime.RISK_ON, confidence=0.5)

        # Blended = 0.5 * risk_on + 0.5 * neutral
        expected_lqq = (0.15 + 0.10) / 2  # 0.125
        expected_cash = (0.10 + 0.20) / 2  # 0.15

        assert abs(rec.lqq_weight - expected_lqq) < 0.001
        assert abs(rec.cash_weight - expected_cash) < 0.001

    def test_invalid_confidence_raises_error(self) -> None:
        """Test that invalid confidence values raise ValueError."""
        optimizer = AllocationOptimizer()

        with pytest.raises(ValueError, match="Confidence must be in"):
            optimizer.get_target_allocation(Regime.NEUTRAL, confidence=-0.1)

        with pytest.raises(ValueError, match="Confidence must be in"):
            optimizer.get_target_allocation(Regime.NEUTRAL, confidence=1.5)

    def test_recommendation_has_reasoning(self) -> None:
        """Test that recommendation includes reasoning."""
        optimizer = AllocationOptimizer()
        rec = optimizer.get_target_allocation(Regime.RISK_ON)

        assert rec.reasoning is not None
        assert "risk_on" in rec.reasoning.lower()


class TestValidateAllocation:
    """Tests for validate_allocation method."""

    def test_valid_allocation(self) -> None:
        """Test valid allocation passes validation."""
        optimizer = AllocationOptimizer()
        weights = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}

        is_valid, violations = optimizer.validate_allocation(weights)

        assert is_valid is True
        assert violations == []

    def test_weights_not_summing_to_one(self) -> None:
        """Test weights not summing to 1.0 are invalid."""
        optimizer = AllocationOptimizer()
        weights = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.50, "CASH": 0.10}  # 0.90

        is_valid, violations = optimizer.validate_allocation(weights)

        assert is_valid is False
        assert any("sum" in v.lower() for v in violations)

    def test_negative_weights_invalid(self) -> None:
        """Test negative weights are invalid."""
        optimizer = AllocationOptimizer()
        weights = {"LQQ": -0.05, "CL2": 0.15, "WPEA": 0.80, "CASH": 0.10}

        is_valid, violations = optimizer.validate_allocation(weights)

        assert is_valid is False
        assert any("negative" in v.lower() for v in violations)

    def test_leveraged_exposure_exceeded(self) -> None:
        """Test leveraged exposure over limit is invalid."""
        optimizer = AllocationOptimizer()
        weights = {"LQQ": 0.20, "CL2": 0.20, "WPEA": 0.50, "CASH": 0.10}  # 40%

        is_valid, violations = optimizer.validate_allocation(weights)

        assert is_valid is False
        assert any("leveraged" in v.lower() for v in violations)

    def test_single_position_exceeded(self) -> None:
        """Test single position over limit is invalid."""
        optimizer = AllocationOptimizer()
        weights = {"LQQ": 0.30, "CL2": 0.00, "WPEA": 0.60, "CASH": 0.10}

        is_valid, violations = optimizer.validate_allocation(weights)

        assert is_valid is False
        assert any("position" in v.lower() and "lqq" in v.lower() for v in violations)

    def test_cash_buffer_insufficient(self) -> None:
        """Test insufficient cash buffer is invalid."""
        optimizer = AllocationOptimizer()
        weights = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.75, "CASH": 0.05}

        is_valid, violations = optimizer.validate_allocation(weights)

        assert is_valid is False
        assert any("cash" in v.lower() for v in violations)


class TestNeedsRebalancing:
    """Tests for needs_rebalancing method."""

    def test_no_rebalance_needed(self) -> None:
        """Test no rebalance when positions are close to target."""
        optimizer = AllocationOptimizer()
        current = {"LQQ": 0.14, "CL2": 0.14, "WPEA": 0.60, "CASH": 0.12}
        target = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}

        # Max drift is 0.02 < 0.05 threshold
        assert optimizer.needs_rebalancing(current, target) is False

    def test_rebalance_needed_single_position_drift(self) -> None:
        """Test rebalance needed when single position drifts too far."""
        optimizer = AllocationOptimizer()
        current = {"LQQ": 0.10, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.15}
        target = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}

        # LQQ drift is 0.05 == threshold, but cash drift is also 0.05
        assert optimizer.needs_rebalancing(current, target) is False

    def test_rebalance_needed_large_drift(self) -> None:
        """Test rebalance needed when position drifts significantly."""
        optimizer = AllocationOptimizer()
        current = {"LQQ": 0.08, "CL2": 0.08, "WPEA": 0.64, "CASH": 0.20}
        target = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}

        # LQQ drift is 0.07 > 0.05 threshold
        assert optimizer.needs_rebalancing(current, target) is True

    def test_rebalance_missing_position(self) -> None:
        """Test rebalance handles missing positions correctly."""
        optimizer = AllocationOptimizer()
        current = {"WPEA": 0.90, "CASH": 0.10}
        target = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}

        # LQQ missing means drift of 0.15 > 0.05
        assert optimizer.needs_rebalancing(current, target) is True

    def test_custom_threshold(self) -> None:
        """Test rebalance with custom threshold."""
        limits = RiskLimits(rebalance_threshold=0.10)
        optimizer = AllocationOptimizer(risk_limits=limits)

        current = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}
        target = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}

        # Drift of 0.05 and 0.10 - not exceeding 0.10 threshold
        assert optimizer.needs_rebalancing(current, target) is False


class TestCalculateRebalanceTrades:
    """Tests for calculate_rebalance_trades method."""

    def test_basic_rebalance_trades(self) -> None:
        """Test basic rebalance trade calculation."""
        optimizer = AllocationOptimizer()
        current = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}
        target = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}
        portfolio_value = 10000.0

        trades = optimizer.calculate_rebalance_trades(current, target, portfolio_value)

        # Should have trades for LQQ (buy 0.05), CL2 (buy 0.05), CASH (sell 0.10)
        assert len(trades) == 3

        lqq_trade = next(t for t in trades if t["symbol"] == "LQQ")
        assert lqq_trade["action"] == "BUY"
        assert lqq_trade["amount"] == 500.0  # 0.05 * 10000

        cash_trade = next(t for t in trades if t["symbol"] == "CASH")
        assert cash_trade["action"] == "SELL"
        assert cash_trade["amount"] == 1000.0  # 0.10 * 10000

    def test_no_trades_when_balanced(self) -> None:
        """Test no trades generated when already balanced."""
        optimizer = AllocationOptimizer()
        weights = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}

        trades = optimizer.calculate_rebalance_trades(weights, weights, 10000.0)

        assert len(trades) == 0

    def test_invalid_portfolio_value(self) -> None:
        """Test that non-positive portfolio value raises error."""
        optimizer = AllocationOptimizer()
        weights = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}

        with pytest.raises(ValueError, match="positive"):
            optimizer.calculate_rebalance_trades(weights, weights, 0.0)

        with pytest.raises(ValueError, match="positive"):
            optimizer.calculate_rebalance_trades(weights, weights, -1000.0)

    def test_invalid_target_allocation_rejected(self) -> None:
        """Test that invalid target allocation raises AllocationError."""
        optimizer = AllocationOptimizer()
        current = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}
        invalid_target = {"LQQ": 0.40, "CL2": 0.40, "WPEA": 0.10, "CASH": 0.10}

        with pytest.raises(AllocationError):
            optimizer.calculate_rebalance_trades(current, invalid_target, 10000.0)

    def test_empty_weights_rejected(self) -> None:
        """Test that empty weight dicts are rejected."""
        optimizer = AllocationOptimizer()

        with pytest.raises(AllocationError, match="empty"):
            optimizer.calculate_rebalance_trades(
                {}, {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}, 10000.0
            )

    def test_new_position_creation(self) -> None:
        """Test trade calculation when adding new position."""
        optimizer = AllocationOptimizer()
        current = {"WPEA": 0.80, "CASH": 0.20}
        target = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}

        trades = optimizer.calculate_rebalance_trades(current, target, 10000.0)

        # Should buy LQQ and CL2, sell some WPEA
        lqq_trade = next(t for t in trades if t["symbol"] == "LQQ")
        assert lqq_trade["action"] == "BUY"
        assert lqq_trade["amount"] == 1000.0

        wpea_trade = next(t for t in trades if t["symbol"] == "WPEA")
        assert wpea_trade["action"] == "SELL"
        assert wpea_trade["amount"] == 2000.0


class TestAllocationError:
    """Tests for AllocationError exception."""

    def test_basic_error(self) -> None:
        """Test creating basic allocation error."""
        error = AllocationError("Test error")
        assert str(error) == "Test error"
        assert error.violations == []

    def test_error_with_violations(self) -> None:
        """Test error with violation list."""
        violations = ["Violation 1", "Violation 2"]
        error = AllocationError("Multiple violations", violations=violations)
        assert error.violations == violations

    def test_error_raised_on_invalid_allocation(self) -> None:
        """Test AllocationError is raised for invalid allocations."""
        optimizer = AllocationOptimizer()

        with pytest.raises(AllocationError) as exc_info:
            optimizer.calculate_rebalance_trades(
                {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10},
                {"LQQ": 0.25, "CL2": 0.25, "WPEA": 0.40, "CASH": 0.10},  # Invalid
                10000.0,
            )

        assert len(exc_info.value.violations) > 0
