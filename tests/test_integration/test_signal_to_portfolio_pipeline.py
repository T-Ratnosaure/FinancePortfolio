"""Integration tests for signal-to-portfolio pipeline.

Tests the flow from regime detection to portfolio rebalancing:
1. Given a regime detection result
2. Generate target allocation
3. Check allocation respects risk limits
4. Verify rebalancing logic triggers correctly
5. Calculate actual trade orders

These tests verify the portfolio management layer responds correctly to signals.
"""

import pytest

from src.data.models import MAX_LEVERAGED_EXPOSURE, Regime
from src.signals.allocation import AllocationError, AllocationOptimizer, RiskLimits


class TestSignalToPortfolioPipeline:
    """Integration tests for signal-to-portfolio rebalancing pipeline."""

    def test_signal_to_rebalance_full_pipeline(self) -> None:
        """Test complete flow from regime signal to rebalance trades.

        Scenario: Portfolio drifts from target, system detects need to rebalance.
        """
        optimizer = AllocationOptimizer()

        # Step 1: Regime detector outputs RISK_ON
        regime = Regime.RISK_ON
        confidence = 0.85

        # Step 2: Generate target allocation
        target_alloc = optimizer.get_target_allocation(regime, confidence=confidence)

        assert target_alloc.regime == Regime.RISK_ON
        assert target_alloc.confidence == confidence

        # Step 3: Current portfolio has drifted from target
        current_weights = {
            "LQQ": 0.10,  # Target is higher
            "CL2": 0.10,  # Target is higher
            "WPEA": 0.65,  # Target is 60%
            "CASH": 0.15,  # Target is lower
        }

        target_weights = {
            "LQQ": target_alloc.lqq_weight,
            "CL2": target_alloc.cl2_weight,
            "WPEA": target_alloc.wpea_weight,
            "CASH": target_alloc.cash_weight,
        }

        # Step 4: Check if rebalancing is needed
        needs_rebalance = optimizer.needs_rebalancing(current_weights, target_weights)

        # With RISK_ON, leveraged exposure should be higher than current 20%
        expected_leveraged = target_alloc.lqq_weight + target_alloc.cl2_weight
        if expected_leveraged > 0.25:  # Significant difference
            assert needs_rebalance

        # Step 5: Calculate rebalance trades
        if needs_rebalance:
            portfolio_value = 10000.0
            trades = optimizer.calculate_rebalance_trades(
                current_weights, target_weights, portfolio_value
            )

            # Verify trades are generated
            assert len(trades) > 0

            # Verify trade structure
            for trade in trades:
                assert "symbol" in trade
                assert "action" in trade
                assert "amount" in trade
                assert trade["action"] in ["BUY", "SELL"]
                assert isinstance(trade["amount"], (int, float))
                assert trade["amount"] > 0

    def test_regime_change_triggers_rebalance(self) -> None:
        """Test that regime changes from RISK_ON to RISK_OFF trigger rebalancing.

        This simulates a market shift requiring defensive positioning.
        """
        optimizer = AllocationOptimizer()

        # Current allocation: RISK_ON positioning
        current_weights = {
            "LQQ": 0.15,
            "CL2": 0.15,
            "WPEA": 0.60,
            "CASH": 0.10,
        }

        # New regime signal: RISK_OFF
        new_regime = Regime.RISK_OFF
        target_alloc = optimizer.get_target_allocation(new_regime, confidence=1.0)

        target_weights = {
            "LQQ": target_alloc.lqq_weight,
            "CL2": target_alloc.cl2_weight,
            "WPEA": target_alloc.wpea_weight,
            "CASH": target_alloc.cash_weight,
        }

        # Should need rebalancing (reducing leveraged exposure, increasing cash)
        needs_rebalance = optimizer.needs_rebalancing(current_weights, target_weights)
        assert needs_rebalance

        # Calculate trades
        trades = optimizer.calculate_rebalance_trades(
            current_weights, target_weights, 10000.0
        )

        # Should sell leveraged ETFs and raise cash
        lqq_trades = [t for t in trades if t["symbol"] == "LQQ"]

        # Leveraged exposure should decrease (likely sell trades)
        if lqq_trades:
            # If there's a trade, it should be a sell (reducing exposure)
            assert (
                lqq_trades[0]["action"] == "SELL"
                or target_alloc.lqq_weight > current_weights["LQQ"]
            )

    def test_risk_limits_enforcement(self) -> None:
        """Test that risk limits are enforced during allocation.

        Ensures leveraged exposure never exceeds 30% limit.
        """
        optimizer = AllocationOptimizer()

        # Try to create allocation that violates limits
        invalid_weights = {
            "LQQ": 0.20,
            "CL2": 0.20,  # Total leveraged: 40% > 30% limit
            "WPEA": 0.50,
            "CASH": 0.10,
        }

        # Validation should fail
        is_valid, violations = optimizer.validate_allocation(invalid_weights)
        assert not is_valid
        assert any("leveraged" in v.lower() for v in violations)

        # Trying to rebalance to invalid allocation should raise error
        current_weights = {
            "LQQ": 0.15,
            "CL2": 0.15,
            "WPEA": 0.60,
            "CASH": 0.10,
        }

        with pytest.raises(AllocationError):
            optimizer.calculate_rebalance_trades(
                current_weights, invalid_weights, 10000.0
            )

    def test_max_leveraged_exposure_constant_respected(self) -> None:
        """Test that MAX_LEVERAGED_EXPOSURE constant is enforced.

        All regime allocations must respect the 30% limit.
        """
        optimizer = AllocationOptimizer()

        for regime in [Regime.RISK_ON, Regime.NEUTRAL, Regime.RISK_OFF]:
            allocation = optimizer.get_target_allocation(regime, confidence=1.0)

            leveraged_exposure = allocation.lqq_weight + allocation.cl2_weight

            # Should never exceed limit
            assert leveraged_exposure <= MAX_LEVERAGED_EXPOSURE
            assert leveraged_exposure <= 0.30  # Explicit value check

    def test_rebalance_threshold_prevents_excessive_trading(self) -> None:
        """Test that small drifts don't trigger rebalancing.

        Prevents excessive trading costs from minor market movements.
        """
        optimizer = AllocationOptimizer()

        # Current weights very close to target
        current_weights = {
            "LQQ": 0.148,  # Target: 0.15 (drift: 0.002)
            "CL2": 0.152,  # Target: 0.15 (drift: 0.002)
            "WPEA": 0.595,  # Target: 0.60 (drift: 0.005)
            "CASH": 0.105,  # Target: 0.10 (drift: 0.005)
        }

        target_weights = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}

        # Small drifts should not trigger rebalance (all < 5% threshold)
        needs_rebalance = optimizer.needs_rebalancing(current_weights, target_weights)
        assert not needs_rebalance

        # Larger drift should trigger rebalance
        drifted_weights = {
            "LQQ": 0.10,  # Target: 0.15 (drift: 0.05 - exactly at threshold)
            "CL2": 0.15,
            "WPEA": 0.60,
            "CASH": 0.15,
        }

        # Drift of exactly 5% should not trigger (only drift > threshold)
        needs_rebalance = optimizer.needs_rebalancing(drifted_weights, target_weights)
        assert not needs_rebalance

        # Drift exceeding threshold should trigger
        large_drift_weights = {
            "LQQ": 0.08,  # Target: 0.15 (drift: 0.07 > 0.05)
            "CL2": 0.15,
            "WPEA": 0.62,
            "CASH": 0.15,
        }

        needs_rebalance = optimizer.needs_rebalancing(
            large_drift_weights, target_weights
        )
        assert needs_rebalance

    def test_confidence_blending_affects_portfolio(self) -> None:
        """Test that low confidence in regime leads to more conservative allocation.

        Low confidence should blend allocation toward NEUTRAL.
        """
        optimizer = AllocationOptimizer()

        # RISK_ON with high confidence
        alloc_high = optimizer.get_target_allocation(Regime.RISK_ON, confidence=1.0)

        # RISK_ON with low confidence (should blend toward NEUTRAL)
        alloc_low = optimizer.get_target_allocation(Regime.RISK_ON, confidence=0.2)

        # Low confidence should reduce leveraged exposure (toward neutral)
        lev_high = alloc_high.lqq_weight + alloc_high.cl2_weight
        lev_low = alloc_low.lqq_weight + alloc_low.cl2_weight

        # With 20% confidence, should be much closer to NEUTRAL (20% leveraged)
        assert lev_low < lev_high
        assert alloc_low.cash_weight > alloc_high.cash_weight

    def test_cash_buffer_enforcement(self) -> None:
        """Test that minimum 10% cash buffer is always maintained.

        Even in RISK_ON, must maintain liquidity.
        """
        optimizer = AllocationOptimizer()

        for regime in [Regime.RISK_ON, Regime.NEUTRAL, Regime.RISK_OFF]:
            allocation = optimizer.get_target_allocation(regime, confidence=1.0)

            # Cash should always be >= 10%
            assert allocation.cash_weight >= 0.10

        # Validation should reject allocations with insufficient cash
        invalid_weights = {
            "LQQ": 0.15,
            "CL2": 0.15,
            "WPEA": 0.65,
            "CASH": 0.05,  # Below 10% minimum
        }

        is_valid, violations = optimizer.validate_allocation(invalid_weights)
        assert not is_valid
        assert any("cash" in v.lower() for v in violations)

    def test_custom_risk_limits(self) -> None:
        """Test that custom risk limits are properly enforced.

        Verifies that institutional users can set tighter limits.
        """
        # Create optimizer with tighter limits
        custom_limits = RiskLimits(
            max_leveraged_exposure=0.20,  # Tighter than default 30%
            max_single_position=0.15,  # Tighter than default 25%
            min_cash_buffer=0.15,  # Higher than default 10%
            rebalance_threshold=0.03,  # Tighter than default 5%
        )

        optimizer = AllocationOptimizer(risk_limits=custom_limits)

        # Standard RISK_ON allocation (30% leveraged) should be rejected
        standard_risk_on = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}

        is_valid, violations = optimizer.validate_allocation(standard_risk_on)
        assert not is_valid
        # Should violate both leveraged exposure (30% > 20%) and cash (10% < 15%)
        assert any("leveraged" in v.lower() for v in violations)
        assert any("cash" in v.lower() for v in violations)

    def test_rebalance_trades_are_executable(self) -> None:
        """Test that calculated trades result in target allocation.

        Verifies trade calculations are mathematically correct.
        """
        optimizer = AllocationOptimizer()

        current_weights = {
            "LQQ": 0.10,
            "CL2": 0.10,
            "WPEA": 0.70,
            "CASH": 0.10,
        }

        target_weights = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}

        portfolio_value = 10000.0
        trades = optimizer.calculate_rebalance_trades(
            current_weights, target_weights, portfolio_value
        )

        # Simulate executing trades
        executed_weights = current_weights.copy()

        for trade in trades:
            symbol = str(trade["symbol"])
            action = str(trade["action"])
            amount = float(trade["amount"])

            current_value = executed_weights[symbol] * portfolio_value

            if action == "BUY":
                new_value = current_value + amount
            else:  # SELL
                new_value = current_value - amount

            executed_weights[symbol] = new_value / portfolio_value

        # After executing trades, should match target (within rounding)
        for symbol in target_weights:
            assert abs(executed_weights[symbol] - target_weights[symbol]) < 0.01

    def test_no_trades_when_already_balanced(self) -> None:
        """Test that no trades are generated when portfolio matches target.

        Avoids unnecessary trading costs.
        """
        optimizer = AllocationOptimizer()

        # Current matches target
        weights = {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10}

        needs_rebalance = optimizer.needs_rebalancing(weights, weights)
        assert not needs_rebalance

        trades = optimizer.calculate_rebalance_trades(weights, weights, 10000.0)
        assert len(trades) == 0
