"""Tests for portfolio rebalancer module."""

from decimal import Decimal

import pytest

from src.data.models import ETFSymbol, Position, TradeAction
from src.portfolio.rebalancer import (
    Rebalancer,
    RebalancerConfig,
    TradePriority,
    TradeRecommendation,
)
from src.signals.allocation import AllocationError, AllocationOptimizer, RiskLimits


@pytest.fixture
def rebalancer() -> Rebalancer:
    """Create a rebalancer with default config."""
    return Rebalancer()


@pytest.fixture
def custom_config() -> RebalancerConfig:
    """Create custom rebalancer configuration."""
    return RebalancerConfig(
        min_trade_value=Decimal("100.0"),
        default_commission_rate=Decimal("0.002"),
        fixed_commission=Decimal("1.99"),
        spread_cost=Decimal("0.0015"),
    )


class TestRebalancerInit:
    """Tests for Rebalancer initialization."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        rebalancer = Rebalancer()
        assert rebalancer.optimizer is not None
        assert rebalancer.config is not None

    def test_init_with_custom_optimizer(self) -> None:
        """Test initialization with custom optimizer."""
        optimizer = AllocationOptimizer(
            risk_limits=RiskLimits(rebalance_threshold=0.03)
        )
        rebalancer = Rebalancer(optimizer=optimizer)
        assert rebalancer.optimizer.risk_limits.rebalance_threshold == 0.03

    def test_init_with_custom_config(self, custom_config: RebalancerConfig) -> None:
        """Test initialization with custom config."""
        rebalancer = Rebalancer(config=custom_config)
        assert rebalancer.config.min_trade_value == Decimal("100.0")


class TestCalculateTrades:
    """Tests for calculate_trades method."""

    def test_calculate_trades_no_rebalancing_needed(
        self, rebalancer: Rebalancer
    ) -> None:
        """Test that no trades are generated when allocation is within threshold."""
        current = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}
        target = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}

        trades = rebalancer.calculate_trades(current, target, Decimal("100000.0"))

        assert len(trades) == 0

    def test_calculate_trades_rebalancing_needed(self, rebalancer: Rebalancer) -> None:
        """Test that trades are generated when drift exceeds threshold."""
        current = {"LQQ": 0.20, "CL2": 0.10, "WPEA": 0.50, "CASH": 0.20}
        target = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}

        trades = rebalancer.calculate_trades(current, target, Decimal("100000.0"))

        # Should have trades for LQQ (sell) and WPEA (buy)
        assert len(trades) >= 1
        symbols = [t.symbol for t in trades]
        assert "LQQ" in symbols or "WPEA" in symbols

    def test_calculate_trades_validates_target_allocation(
        self, rebalancer: Rebalancer
    ) -> None:
        """Test that invalid target allocation raises error."""
        current = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}
        # Invalid: leveraged exposure exceeds limit
        target = {"LQQ": 0.25, "CL2": 0.25, "WPEA": 0.40, "CASH": 0.10}

        with pytest.raises(AllocationError):
            rebalancer.calculate_trades(current, target, Decimal("100000.0"))

    def test_calculate_trades_invalid_portfolio_value(
        self, rebalancer: Rebalancer
    ) -> None:
        """Test that zero/negative portfolio value raises error."""
        current = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}
        target = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}

        with pytest.raises(ValueError, match="must be positive"):
            rebalancer.calculate_trades(current, target, Decimal("0.0"))

    def test_calculate_trades_filters_small_trades(
        self, rebalancer: Rebalancer
    ) -> None:
        """Test that trades below minimum value are filtered out."""
        # Very small drift that would result in < min trade value
        current = {"LQQ": 0.101, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.199}
        target = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}

        trades = rebalancer.calculate_trades(
            current,
            target,
            Decimal("1000.0"),  # Small portfolio
        )

        assert len(trades) == 0  # Drift too small to trade

    def test_calculate_trades_with_prices(self, rebalancer: Rebalancer) -> None:
        """Test trades calculation with price data for share calculation."""
        current = {"LQQ": 0.20, "CL2": 0.05, "WPEA": 0.55, "CASH": 0.20}
        target = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}
        prices = {
            "LQQ": Decimal("100.0"),
            "CL2": Decimal("200.0"),
            "WPEA": Decimal("50.0"),
        }

        trades = rebalancer.calculate_trades(
            current, target, Decimal("100000.0"), prices
        )

        # Verify shares are calculated based on prices
        for trade in trades:
            if trade.symbol in prices:
                expected_shares = trade.estimated_value / prices[trade.symbol]
                assert abs(float(trade.shares) - float(expected_shares)) < 0.1


class TestTradeOrdering:
    """Tests for trade ordering/prioritization."""

    def test_optimize_trade_order_sells_first(self, rebalancer: Rebalancer) -> None:
        """Test that sells are ordered before buys."""
        trades = [
            TradeRecommendation(
                symbol="WPEA",
                action=TradeAction.BUY,
                shares=Decimal("100"),
                estimated_value=Decimal("5000"),
                reason="Test buy",
                priority=TradePriority.BUY_REGULAR,
                current_weight=0.55,
                target_weight=0.60,
                drift=0.05,
            ),
            TradeRecommendation(
                symbol="LQQ",
                action=TradeAction.SELL,
                shares=Decimal("50"),
                estimated_value=Decimal("5000"),
                reason="Test sell",
                priority=TradePriority.SELL_LEVERAGED,
                current_weight=0.15,
                target_weight=0.10,
                drift=0.05,
            ),
        ]

        ordered = rebalancer.optimize_trade_order(trades)

        # First trade should be the sell
        assert ordered[0].action == TradeAction.SELL
        assert ordered[1].action == TradeAction.BUY

    def test_optimize_trade_order_leveraged_sells_first(
        self, rebalancer: Rebalancer
    ) -> None:
        """Test that leveraged sell comes before regular sell."""
        trades = [
            TradeRecommendation(
                symbol="WPEA",
                action=TradeAction.SELL,
                shares=Decimal("100"),
                estimated_value=Decimal("5000"),
                reason="Test sell WPEA",
                priority=TradePriority.SELL_REGULAR,
                current_weight=0.65,
                target_weight=0.60,
                drift=0.05,
            ),
            TradeRecommendation(
                symbol="LQQ",
                action=TradeAction.SELL,
                shares=Decimal("50"),
                estimated_value=Decimal("5000"),
                reason="Test sell LQQ",
                priority=TradePriority.SELL_LEVERAGED,
                current_weight=0.15,
                target_weight=0.10,
                drift=0.05,
            ),
        ]

        ordered = rebalancer.optimize_trade_order(trades)

        # LQQ (leveraged) should be first
        assert ordered[0].symbol == "LQQ"
        assert ordered[1].symbol == "WPEA"

    def test_optimize_trade_order_empty_list(self, rebalancer: Rebalancer) -> None:
        """Test that empty trade list returns empty."""
        ordered = rebalancer.optimize_trade_order([])
        assert ordered == []


class TestTransactionCosts:
    """Tests for transaction cost estimation."""

    def test_estimate_transaction_costs(self, rebalancer: Rebalancer) -> None:
        """Test transaction cost estimation."""
        trades = [
            TradeRecommendation(
                symbol="LQQ",
                action=TradeAction.BUY,
                shares=Decimal("100"),
                estimated_value=Decimal("10000"),
                reason="Test",
                priority=TradePriority.BUY_LEVERAGED,
                current_weight=0.05,
                target_weight=0.10,
                drift=0.05,
            ),
        ]

        costs = rebalancer.estimate_transaction_costs(trades)

        # Should have some costs > 0
        assert costs > Decimal("0")

    def test_estimate_transaction_costs_empty(self, rebalancer: Rebalancer) -> None:
        """Test cost estimation for empty trade list."""
        costs = rebalancer.estimate_transaction_costs([])
        assert costs == Decimal("0.0")

    def test_estimate_transaction_costs_with_custom_config(
        self, custom_config: RebalancerConfig
    ) -> None:
        """Test cost estimation with custom config."""
        rebalancer = Rebalancer(config=custom_config)

        trades = [
            TradeRecommendation(
                symbol="LQQ",
                action=TradeAction.BUY,
                shares=Decimal("100"),
                estimated_value=Decimal("10000"),
                reason="Test",
                priority=TradePriority.BUY_LEVERAGED,
                current_weight=0.05,
                target_weight=0.10,
                drift=0.05,
            ),
        ]

        costs = rebalancer.estimate_transaction_costs(trades)

        # Should include commission + spread + fixed fee
        # Commission: 10000 * 0.002 = 20
        # Spread: 10000 * 0.0015 = 15
        # Fixed: 1.99
        expected_min = Decimal("36.0")
        assert costs >= expected_min


class TestRebalanceReport:
    """Tests for rebalance report generation."""

    def test_generate_report_no_rebalancing(self, rebalancer: Rebalancer) -> None:
        """Test report when no rebalancing needed."""
        current = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}
        target = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}

        report = rebalancer.generate_rebalance_report(
            current, target, Decimal("100000.0")
        )

        assert not report.needs_rebalancing
        assert report.total_trades == 0
        assert len(report.notes) > 0  # Should have note about being within tolerance

    def test_generate_report_with_trades(self, rebalancer: Rebalancer) -> None:
        """Test report when rebalancing is needed."""
        current = {"LQQ": 0.20, "CL2": 0.10, "WPEA": 0.50, "CASH": 0.20}
        target = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}

        report = rebalancer.generate_rebalance_report(
            current, target, Decimal("100000.0")
        )

        assert report.needs_rebalancing
        assert report.total_trades >= 1
        assert report.portfolio_value == Decimal("100000.0")

    def test_generate_report_with_risk_violations(self, rebalancer: Rebalancer) -> None:
        """Test report includes risk violations."""
        current = {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20}
        # Invalid target - exceeds leveraged limit
        target = {"LQQ": 0.25, "CL2": 0.25, "WPEA": 0.40, "CASH": 0.10}

        report = rebalancer.generate_rebalance_report(
            current, target, Decimal("100000.0")
        )

        assert len(report.risk_violations) > 0


class TestCashAdjustment:
    """Tests for cash constraint handling."""

    def test_adjust_for_available_cash_sufficient(self, rebalancer: Rebalancer) -> None:
        """Test no adjustment when cash is sufficient."""
        trades = [
            TradeRecommendation(
                symbol="LQQ",
                action=TradeAction.BUY,
                shares=Decimal("100"),
                estimated_value=Decimal("10000"),
                reason="Test",
                priority=TradePriority.BUY_LEVERAGED,
                current_weight=0.05,
                target_weight=0.10,
                drift=0.05,
            ),
        ]

        adjusted = rebalancer.adjust_for_available_cash(
            trades,
            Decimal("20000"),  # Plenty of cash
        )

        assert len(adjusted) == len(trades)
        assert adjusted[0].estimated_value == trades[0].estimated_value

    def test_adjust_for_available_cash_insufficient(
        self, rebalancer: Rebalancer
    ) -> None:
        """Test adjustment when cash is insufficient."""
        trades = [
            TradeRecommendation(
                symbol="LQQ",
                action=TradeAction.BUY,
                shares=Decimal("100"),
                estimated_value=Decimal("10000"),
                reason="Test",
                priority=TradePriority.BUY_LEVERAGED,
                current_weight=0.05,
                target_weight=0.10,
                drift=0.05,
            ),
        ]

        adjusted = rebalancer.adjust_for_available_cash(
            trades,
            Decimal("5000"),  # Not enough cash
        )

        # Should have reduced the buy or removed it
        total_buys = sum(
            t.estimated_value for t in adjusted if t.action == TradeAction.BUY
        )
        assert total_buys <= Decimal("5000")


class TestShareSufficiency:
    """Tests for share sufficiency checking."""

    def test_check_sufficient_shares_ok(self, rebalancer: Rebalancer) -> None:
        """Test check passes when sufficient shares."""
        positions = {
            "LQQ": Position(
                symbol=ETFSymbol.LQQ,
                shares=100.0,
                average_cost=100.0,
                current_price=100.0,
                market_value=10000.0,
                unrealized_pnl=0.0,
                weight=0.1,
            ),
        }

        trades = [
            TradeRecommendation(
                symbol="LQQ",
                action=TradeAction.SELL,
                shares=Decimal("50"),
                estimated_value=Decimal("5000"),
                reason="Test",
                priority=TradePriority.SELL_LEVERAGED,
                current_weight=0.15,
                target_weight=0.10,
                drift=0.05,
            ),
        ]

        ok, issues = rebalancer.check_sufficient_shares(positions, trades)

        assert ok is True
        assert len(issues) == 0

    def test_check_sufficient_shares_insufficient(self, rebalancer: Rebalancer) -> None:
        """Test check fails when insufficient shares."""
        positions = {
            "LQQ": Position(
                symbol=ETFSymbol.LQQ,
                shares=30.0,  # Only 30 shares
                average_cost=100.0,
                current_price=100.0,
                market_value=3000.0,
                unrealized_pnl=0.0,
                weight=0.1,
            ),
        }

        trades = [
            TradeRecommendation(
                symbol="LQQ",
                action=TradeAction.SELL,
                shares=Decimal("50"),  # Want to sell 50
                estimated_value=Decimal("5000"),
                reason="Test",
                priority=TradePriority.SELL_LEVERAGED,
                current_weight=0.15,
                target_weight=0.10,
                drift=0.05,
            ),
        ]

        ok, issues = rebalancer.check_sufficient_shares(positions, trades)

        assert ok is False
        assert len(issues) == 1
        assert "Insufficient" in issues[0]

    def test_check_sufficient_shares_no_position(self, rebalancer: Rebalancer) -> None:
        """Test check fails when position doesn't exist."""
        positions: dict[str, Position] = {}

        trades = [
            TradeRecommendation(
                symbol="LQQ",
                action=TradeAction.SELL,
                shares=Decimal("50"),
                estimated_value=Decimal("5000"),
                reason="Test",
                priority=TradePriority.SELL_LEVERAGED,
                current_weight=0.15,
                target_weight=0.10,
                drift=0.05,
            ),
        ]

        ok, issues = rebalancer.check_sufficient_shares(positions, trades)

        assert ok is False
        assert len(issues) == 1
        assert "no position" in issues[0]
