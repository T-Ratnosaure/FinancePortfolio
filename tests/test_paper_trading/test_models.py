"""Tests for paper trading models."""

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.paper_trading.models import (
    FillSimulationConfig,
    OrderAction,
    OrderResult,
    OrderStatus,
    OrderType,
    PaperOrder,
    PortfolioSnapshot,
    SessionConfig,
    VirtualCashPosition,
    VirtualPosition,
)


class TestSessionConfig:
    """Tests for SessionConfig model."""

    def test_default_config(self) -> None:
        """Test creating config with defaults."""
        config = SessionConfig(session_name="Test")
        assert config.session_name == "Test"
        assert config.initial_capital == Decimal("10000.00")
        assert config.currency == "EUR"
        assert config.auto_rebalance is True
        assert config.rebalance_threshold == 0.05

    def test_custom_config(self) -> None:
        """Test creating config with custom values."""
        config = SessionConfig(
            session_name="Custom Session",
            initial_capital=Decimal("50000.00"),
            currency="USD",
            auto_rebalance=False,
            rebalance_threshold=0.10,
        )
        assert config.session_name == "Custom Session"
        assert config.initial_capital == Decimal("50000.00")
        assert config.auto_rebalance is False

    def test_invalid_capital(self) -> None:
        """Test validation of initial capital."""
        with pytest.raises(ValidationError):
            SessionConfig(
                session_name="Test",
                initial_capital=Decimal("50.00"),
            )

    def test_invalid_threshold(self) -> None:
        """Test validation of rebalance threshold."""
        with pytest.raises(ValidationError):
            SessionConfig(
                session_name="Test",
                rebalance_threshold=0.5,
            )


class TestPaperOrder:
    """Tests for PaperOrder model."""

    def test_market_order(self) -> None:
        """Test creating a market order."""
        order = PaperOrder(
            session_id="session-001",
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
        )
        assert order.order_type == OrderType.MARKET
        assert order.limit_price is None
        assert order.time_in_force == "DAY"

    def test_limit_order(self) -> None:
        """Test creating a limit order."""
        order = PaperOrder(
            session_id="session-001",
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            limit_price=Decimal("100.00"),
        )
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == Decimal("100.00")

    def test_limit_order_without_price(self) -> None:
        """Test that limit orders require a price."""
        with pytest.raises(ValueError, match="Limit price required"):
            PaperOrder(
                session_id="session-001",
                symbol="LQQ.PA",
                action=OrderAction.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("10"),
            )

    def test_order_has_unique_id(self) -> None:
        """Test that orders get unique IDs."""
        order1 = PaperOrder(
            session_id="session-001",
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
        )
        order2 = PaperOrder(
            session_id="session-001",
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
        )
        assert order1.order_id != order2.order_id


class TestVirtualPosition:
    """Tests for VirtualPosition model."""

    def test_create_position(self) -> None:
        """Test creating a position with calculated fields."""
        position = VirtualPosition.create(
            symbol="LQQ.PA",
            shares=Decimal("10"),
            average_cost=Decimal("100.00"),
            current_price=Decimal("110.00"),
        )

        assert position.symbol == "LQQ.PA"
        assert position.shares == Decimal("10")
        assert position.average_cost == Decimal("100.00")
        assert position.current_price == Decimal("110.00")
        assert position.market_value == Decimal("1100.00")
        assert position.unrealized_pnl == Decimal("100.00")

    def test_position_with_loss(self) -> None:
        """Test position with unrealized loss."""
        position = VirtualPosition.create(
            symbol="CL2.PA",
            shares=Decimal("5"),
            average_cost=Decimal("200.00"),
            current_price=Decimal("180.00"),
        )

        assert position.market_value == Decimal("900.00")
        assert position.unrealized_pnl == Decimal("-100.00")


class TestOrderResult:
    """Tests for OrderResult model."""

    def test_filled_order(self) -> None:
        """Test filled order result."""
        result = OrderResult(
            order_id="order-001",
            status=OrderStatus.FILLED,
            fill_price=Decimal("100.50"),
            fill_quantity=Decimal("10"),
            fill_timestamp=datetime.now(),
            transaction_cost=Decimal("1.00"),
            slippage_cost=Decimal("0.50"),
        )

        assert result.status == OrderStatus.FILLED
        assert result.fill_price == Decimal("100.50")
        assert result.rejection_reason is None

    def test_rejected_order(self) -> None:
        """Test rejected order result."""
        result = OrderResult(
            order_id="order-001",
            status=OrderStatus.REJECTED,
            rejection_reason="Insufficient funds",
        )

        assert result.status == OrderStatus.REJECTED
        assert result.fill_price is None
        assert result.rejection_reason == "Insufficient funds"


class TestFillSimulationConfig:
    """Tests for FillSimulationConfig model."""

    def test_default_config(self) -> None:
        """Test default fill configuration."""
        config = FillSimulationConfig()
        assert config.base_slippage_bps == 3.0
        assert config.volatility_slippage_factor == 0.5
        assert config.enable_partial_fills is False

    def test_custom_config(self) -> None:
        """Test custom fill configuration."""
        config = FillSimulationConfig(
            base_slippage_bps=5.0,
            volatility_slippage_factor=1.0,
            max_price_deviation_pct=0.05,
        )
        assert config.base_slippage_bps == 5.0
        assert config.max_price_deviation_pct == 0.05


class TestPortfolioSnapshot:
    """Tests for PortfolioSnapshot model."""

    def test_create_snapshot(self) -> None:
        """Test creating a portfolio snapshot."""
        positions = [
            VirtualPosition.create(
                symbol="LQQ.PA",
                shares=Decimal("10"),
                average_cost=Decimal("100.00"),
                current_price=Decimal("110.00"),
            )
        ]
        cash = VirtualCashPosition(amount=Decimal("5000.00"))

        snapshot = PortfolioSnapshot(
            session_id="session-001",
            positions=positions,
            cash=cash,
            total_value=Decimal("6100.00"),
            weights={"LQQ.PA": 0.18, "CASH": 0.82},
        )

        assert snapshot.session_id == "session-001"
        assert len(snapshot.positions) == 1
        assert snapshot.total_value == Decimal("6100.00")
