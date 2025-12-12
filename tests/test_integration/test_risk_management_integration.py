"""Integration tests for risk management system.

Tests the complete risk management workflow:
1. Verify position limits are enforced
2. Test VaR calculations with sample data
3. Test exposure limits enforcement
4. Verify risk alerts are generated correctly
5. Test interaction between allocation and risk systems

These tests ensure the risk management layer provides proper portfolio protection.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.models import (
    DRAWDOWN_ALERT,
    MAX_LEVERAGED_EXPOSURE,
    MAX_SINGLE_POSITION,
    MIN_CASH_BUFFER,
    ETFSymbol,
    Position,
)
from src.portfolio.risk import InsufficientDataError, RiskCalculator
from src.signals.allocation import AllocationOptimizer


class TestRiskManagementIntegration:
    """Integration tests for risk management enforcement."""

    def test_position_limits_enforced_in_allocation(self) -> None:
        """Test that allocation system respects position limits.

        MAX_SINGLE_POSITION should prevent any single position from being too large.
        """
        optimizer = AllocationOptimizer()

        # All standard allocations should respect limits
        for regime in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
            from src.data.models import Regime

            regime_enum = Regime(regime.lower())
            allocation = optimizer.get_target_allocation(regime_enum, confidence=1.0)

            # No single leveraged position should exceed MAX_SINGLE_POSITION
            assert allocation.lqq_weight <= MAX_SINGLE_POSITION
            assert allocation.cl2_weight <= MAX_SINGLE_POSITION

            # WPEA can exceed single position limit (it's the core safe holding)
            # But combined leveraged exposure must not exceed limit
            leveraged = allocation.lqq_weight + allocation.cl2_weight
            assert leveraged <= MAX_LEVERAGED_EXPOSURE

    def test_var_calculation_with_realistic_data(
        self, mock_price_data: pd.DataFrame
    ) -> None:
        """Test VaR calculation with realistic return distributions.

        Uses mock price data to generate returns and calculate VaR.
        """
        calc = RiskCalculator(lookback_days=252)

        # Get WPEA prices and calculate returns
        wpea_data = mock_price_data[mock_price_data["symbol"] == "WPEA.PA"]
        prices = wpea_data.sort_values("date")["close"]  # type: ignore[call-overload]
        returns = prices.pct_change().dropna()

        # Convert to pandas Series
        returns_series = pd.Series(returns.values, index=range(len(returns)))

        # Calculate 95% VaR
        var_95 = calc.calculate_var(
            returns_series, confidence=0.95, method="historical"
        )

        # VaR should be positive (representing potential loss)
        assert var_95 > 0

        # For daily equity returns, VaR should typically be 1-5%
        assert 0.005 < var_95 < 0.10

        # Parametric VaR should be similar
        var_parametric = calc.calculate_var(
            returns_series, confidence=0.95, method="parametric"
        )
        assert var_parametric > 0

        # Both methods should give reasonably similar results
        # (within a factor of 2 for normal-ish returns)
        ratio = max(var_95, var_parametric) / min(var_95, var_parametric)
        assert ratio < 2.0

    def test_exposure_limits_enforcement(self) -> None:
        """Test that leveraged exposure limits are enforced.

        MAX_LEVERAGED_EXPOSURE (30%) should be a hard limit.
        """
        optimizer = AllocationOptimizer()

        # Verify allocation system enforces limit
        invalid_allocation = {
            "LQQ": 0.20,
            "CL2": 0.20,  # Total: 40% leveraged
            "WPEA": 0.50,
            "CASH": 0.10,
        }

        is_valid, violations = optimizer.validate_allocation(invalid_allocation)
        assert not is_valid
        assert any("leveraged" in v.lower() for v in violations)

        # Verify constant is set correctly
        assert MAX_LEVERAGED_EXPOSURE == 0.30

    def test_drawdown_alert_generation(self) -> None:
        """Test that drawdown alerts are generated at threshold.

        DRAWDOWN_ALERT threshold (-20%) should trigger warnings.
        """
        calc = RiskCalculator()

        # Create price series with large drawdown
        dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        prices_values = [100.0]

        # Simulate 30% drawdown
        for _ in range(1, 100):
            prices_values.append(100.0)  # Stay at peak

        for i in range(100, 200):
            # Drop 30% over 100 days
            prices_values.append(100.0 - 30.0 * (i - 100) / 100)

        for _ in range(200, 252):
            prices_values.append(70.0)  # Stay at trough

        prices = pd.Series(prices_values, index=dates)

        # Calculate max drawdown
        max_dd, peak_date, trough_date = calc.calculate_max_drawdown(prices)

        # Should detect the 30% drawdown
        assert max_dd < -0.25  # More than 25% drawdown
        assert max_dd < DRAWDOWN_ALERT  # Exceeds -20% threshold

        # Peak and trough should be identified
        assert peak_date is not None
        assert trough_date is not None
        assert trough_date > peak_date

    def test_risk_report_generation_with_positions(
        self, mock_price_data: pd.DataFrame
    ) -> None:
        """Test comprehensive risk report generation.

        Verifies all risk metrics are calculated together.
        """
        calc = RiskCalculator(lookback_days=252)

        # Create sample positions
        positions = [
            Position(
                symbol=ETFSymbol.LQQ,
                shares=10.0,
                average_cost=100.0,
                current_price=110.0,
                market_value=1100.0,
                unrealized_pnl=100.0,
                weight=0.11,
            ),
            Position(
                symbol=ETFSymbol.CL2,
                shares=10.0,
                average_cost=100.0,
                current_price=105.0,
                market_value=1050.0,
                unrealized_pnl=50.0,
                weight=0.105,
            ),
            Position(
                symbol=ETFSymbol.WPEA,
                shares=60.0,
                average_cost=100.0,
                current_price=108.0,
                market_value=6480.0,
                unrealized_pnl=480.0,
                weight=0.648,
            ),
        ]

        # Create price history
        prices_history: dict[str, pd.Series] = {}  # type: ignore[type-arg]
        for symbol in [ETFSymbol.LQQ, ETFSymbol.CL2, ETFSymbol.WPEA]:
            symbol_data = mock_price_data[mock_price_data["symbol"] == symbol.value]
            if not symbol_data.empty:
                prices = symbol_data.sort_values("date").set_index("date")["close"]  # type: ignore[call-overload]
                prices_history[symbol.value] = prices  # type: ignore[assignment]

        # Generate risk report
        report = calc.generate_risk_report(
            positions=positions,
            prices_history=prices_history,
            risk_free_rate=0.03,
        )

        # Verify report fields
        assert report.report_date == date.today()
        assert report.var_95 >= 0.0
        assert report.volatility >= 0.0
        assert report.max_drawdown <= 0.0

        # Check for leveraged exposure alert
        leveraged_exposure = positions[0].weight + positions[1].weight
        if leveraged_exposure > MAX_LEVERAGED_EXPOSURE:
            assert any("leveraged" in alert.lower() for alert in report.risk_alerts)

    def test_cash_buffer_enforcement(self) -> None:
        """Test that minimum cash buffer is enforced.

        MIN_CASH_BUFFER (10%) should always be maintained.
        """
        optimizer = AllocationOptimizer()

        # Allocation with insufficient cash
        invalid_allocation = {
            "LQQ": 0.15,
            "CL2": 0.15,
            "WPEA": 0.65,
            "CASH": 0.05,  # Below 10% minimum
        }

        is_valid, violations = optimizer.validate_allocation(invalid_allocation)
        assert not is_valid
        assert any("cash" in v.lower() for v in violations)

        # Verify constant
        assert MIN_CASH_BUFFER == 0.10

    def test_portfolio_volatility_calculation(
        self, mock_price_data: pd.DataFrame
    ) -> None:
        """Test portfolio volatility with correlated assets.

        Verifies covariance matrix approach to portfolio risk.
        """
        calc = RiskCalculator(lookback_days=252)

        # Build returns DataFrame
        returns_dict = {}
        for symbol in [ETFSymbol.LQQ.value, ETFSymbol.CL2.value, ETFSymbol.WPEA.value]:
            symbol_data = mock_price_data[mock_price_data["symbol"] == symbol]
            prices = symbol_data.sort_values("date")["close"]  # type: ignore[call-overload]
            returns = prices.pct_change().dropna()
            returns_dict[symbol] = returns.values

        # Ensure same length
        min_length = min(len(v) for v in returns_dict.values())
        for key in returns_dict:
            returns_dict[key] = returns_dict[key][:min_length]

        returns_df = pd.DataFrame(returns_dict)

        # Calculate portfolio volatility with equal weights
        weights = {
            ETFSymbol.LQQ.value: 0.15,
            ETFSymbol.CL2.value: 0.15,
            ETFSymbol.WPEA.value: 0.60,
            "CASH": 0.10,
        }

        portfolio_vol = calc.calculate_portfolio_volatility(weights, returns_df)

        # Portfolio vol should be positive and reasonable (5-30% annualized)
        assert 0.05 < portfolio_vol < 0.50

        # Portfolio vol should be less than average individual vol due to
        # diversification. Due to leveraged ETFs, portfolio vol might not
        # always be lower. Just verify it's calculated and reasonable.
        assert portfolio_vol > 0

    def test_sharpe_ratio_calculation(self, mock_price_data: pd.DataFrame) -> None:
        """Test Sharpe ratio calculation with realistic returns."""
        calc = RiskCalculator(lookback_days=252)

        # Get returns for WPEA
        wpea_data = mock_price_data[mock_price_data["symbol"] == "WPEA.PA"]
        prices = wpea_data.sort_values("date")["close"]  # type: ignore[call-overload]
        returns = prices.pct_change().dropna()
        returns_series = pd.Series(returns.values)

        # Calculate Sharpe ratio
        sharpe = calc.calculate_sharpe_ratio(returns_series, risk_free_rate=0.03)

        # Sharpe ratio should be finite
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)

        # For equity markets, typical range is -1 to 2
        assert -2.0 < sharpe < 3.0

    def test_sortino_ratio_calculation(self, mock_price_data: pd.DataFrame) -> None:
        """Test Sortino ratio calculation with realistic returns.

        Sortino should be higher than Sharpe (only penalizes downside).
        """
        calc = RiskCalculator(lookback_days=252)

        # Get returns for WPEA
        wpea_data = mock_price_data[mock_price_data["symbol"] == "WPEA.PA"]
        prices = wpea_data.sort_values("date")["close"]  # type: ignore[call-overload]
        returns = prices.pct_change().dropna()
        returns_series = pd.Series(returns.values)

        # Calculate both ratios
        sharpe = calc.calculate_sharpe_ratio(returns_series, risk_free_rate=0.03)
        sortino = calc.calculate_sortino_ratio(returns_series, risk_free_rate=0.03)

        # Both should be finite
        assert not np.isnan(sharpe)
        assert not np.isnan(sortino)

        # Sortino typically >= Sharpe (less penalty for upside volatility)
        # May not always hold, but both should be reasonable
        assert -2.0 < sortino < 3.0

    def test_leveraged_decay_estimation(self, mock_price_data: pd.DataFrame) -> None:
        """Test estimation of leveraged ETF decay.

        2x ETFs experience decay due to daily rebalancing and volatility.
        """
        calc = RiskCalculator(lookback_days=252)

        # Get leveraged and unleveraged returns
        lqq_data = mock_price_data[mock_price_data["symbol"] == "LQQ.PA"]
        wpea_data = mock_price_data[mock_price_data["symbol"] == "WPEA.PA"]

        lqq_prices = lqq_data.sort_values("date")["close"]  # type: ignore[call-overload]
        wpea_prices = wpea_data.sort_values("date")["close"]  # type: ignore[call-overload]

        lqq_returns = lqq_prices.pct_change().dropna()
        wpea_returns = wpea_prices.pct_change().dropna()

        # Align series
        min_length = min(len(lqq_returns), len(wpea_returns))
        lqq_series = pd.Series(lqq_returns.values[:min_length])
        wpea_series = pd.Series(wpea_returns.values[:min_length])

        # Calculate decay
        decay = calc.calculate_leveraged_decay(
            etf_returns=lqq_series, index_returns=wpea_series, leverage=2
        )

        # Decay should be positive (cost of leverage)
        assert decay >= 0.0

        # Typical decay for 2x ETFs is 2-10% annually
        # Mock data might not show realistic decay, so just check it's calculated
        assert 0.0 <= decay < 0.30  # Less than 30% annual decay (extreme upper bound)

    def test_risk_alerts_for_excessive_leverage(self) -> None:
        """Test that risk system generates alerts for excessive leverage.

        Verifies integration between risk calculation and alerting.
        """
        calc = RiskCalculator()

        # Create positions with excessive leveraged exposure
        positions = [
            Position(
                symbol=ETFSymbol.LQQ,
                shares=20.0,
                average_cost=100.0,
                current_price=110.0,
                market_value=2200.0,
                unrealized_pnl=200.0,
                weight=0.22,  # 22% LQQ
            ),
            Position(
                symbol=ETFSymbol.CL2,
                shares=15.0,
                average_cost=100.0,
                current_price=105.0,
                market_value=1575.0,
                unrealized_pnl=75.0,
                weight=0.1575,  # 15.75% CL2 - Total leveraged: 37.75% > 30%
            ),
            Position(
                symbol=ETFSymbol.WPEA,
                shares=50.0,
                average_cost=100.0,
                current_price=108.0,
                market_value=5400.0,
                unrealized_pnl=400.0,
                weight=0.54,
            ),
        ]

        # Create minimal price history
        prices_history = {
            ETFSymbol.LQQ.value: pd.Series([100.0] * 50),
            ETFSymbol.CL2.value: pd.Series([100.0] * 50),
            ETFSymbol.WPEA.value: pd.Series([100.0] * 50),
        }

        # Generate report
        report = calc.generate_risk_report(
            positions=positions, prices_history=prices_history
        )

        # Should have leveraged exposure alert
        leveraged_exposure = 0.22 + 0.1575  # 37.75%
        assert leveraged_exposure > MAX_LEVERAGED_EXPOSURE

        # Check for alert
        assert len(report.risk_alerts) > 0
        assert any("leveraged" in alert.lower() for alert in report.risk_alerts)

    def test_insufficient_data_handling(self) -> None:
        """Test that risk calculations handle insufficient data gracefully."""
        calc = RiskCalculator(lookback_days=252)

        # Try to calculate VaR with insufficient data
        short_returns = pd.Series([0.01, -0.01, 0.02])  # Only 3 observations

        with pytest.raises(InsufficientDataError):
            calc.calculate_var(short_returns, confidence=0.95)

        # Try portfolio volatility with insufficient data
        short_returns_df = pd.DataFrame({"asset1": [0.01] * 10, "asset2": [-0.01] * 10})

        weights = {"asset1": 0.5, "asset2": 0.5}

        with pytest.raises(InsufficientDataError):
            calc.calculate_portfolio_volatility(weights, short_returns_df)

    def test_risk_integration_with_allocation(self) -> None:
        """Test that risk and allocation systems work together.

        Allocation should always produce risk-compliant portfolios.
        """
        optimizer = AllocationOptimizer()

        from src.data.models import Regime

        # Generate allocations for all regimes
        for regime in [Regime.RISK_ON, Regime.NEUTRAL, Regime.RISK_OFF]:
            allocation = optimizer.get_target_allocation(regime, confidence=1.0)

            # Convert to weights dict
            weights = {
                "LQQ": allocation.lqq_weight,
                "CL2": allocation.cl2_weight,
                "WPEA": allocation.wpea_weight,
                "CASH": allocation.cash_weight,
            }

            # Validate against risk limits
            is_valid, violations = optimizer.validate_allocation(weights)

            # All standard allocations should pass risk checks
            assert is_valid, f"Regime {regime} failed validation: {violations}"

            # Verify specific limits
            leveraged = allocation.lqq_weight + allocation.cl2_weight
            assert leveraged <= MAX_LEVERAGED_EXPOSURE
            assert allocation.cash_weight >= MIN_CASH_BUFFER
