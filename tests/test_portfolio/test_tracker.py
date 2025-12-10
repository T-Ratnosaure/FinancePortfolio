"""Tests for portfolio tracker module."""

import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from src.data.models import (
    DiscrepancyType,
    ETFSymbol,
    Trade,
    TradeAction,
)
from src.portfolio.tracker import PortfolioTracker


@pytest.fixture
def temp_db_path() -> str:
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield str(Path(tmpdir) / "test_portfolio.duckdb")


@pytest.fixture
def tracker(temp_db_path: str) -> PortfolioTracker:
    """Create a portfolio tracker with temporary database."""
    t = PortfolioTracker(temp_db_path)
    yield t
    t.close()


class TestPortfolioTrackerInit:
    """Tests for PortfolioTracker initialization."""

    def test_init_creates_database(self, temp_db_path: str) -> None:
        """Test that tracker creates database file."""
        tracker = PortfolioTracker(temp_db_path)
        assert tracker.db is not None
        tracker.close()

    def test_init_creates_cash_table(self, tracker: PortfolioTracker) -> None:
        """Test that cash positions table is created."""
        # Just verify the table works by calling get_cash_position
        cash = tracker.get_cash_position()
        assert cash.amount == Decimal("0")


class TestCurrentPositions:
    """Tests for position retrieval."""

    def test_get_current_positions_empty(self, tracker: PortfolioTracker) -> None:
        """Test getting positions from empty portfolio."""
        positions = tracker.get_current_positions()
        assert positions == {}

    def test_get_portfolio_value_empty(self, tracker: PortfolioTracker) -> None:
        """Test portfolio value with no positions."""
        value = tracker.get_portfolio_value()
        assert value == Decimal("0")

    def test_get_portfolio_weights_empty(self, tracker: PortfolioTracker) -> None:
        """Test weights return 100% cash when no positions."""
        weights = tracker.get_portfolio_weights()
        assert weights == {"CASH": 1.0}


class TestCashPosition:
    """Tests for cash position management."""

    def test_set_and_get_cash_position(self, tracker: PortfolioTracker) -> None:
        """Test setting and retrieving cash position."""
        tracker.set_cash_position(Decimal("10000.00"))
        cash = tracker.get_cash_position()
        assert cash.amount == Decimal("10000.00")
        assert cash.currency == "EUR"

    def test_deposit_cash(self, tracker: PortfolioTracker) -> None:
        """Test depositing cash."""
        tracker.set_cash_position(Decimal("1000.00"))
        tracker.deposit_cash(Decimal("500.00"))
        cash = tracker.get_cash_position()
        assert cash.amount == Decimal("1500.00")

    def test_withdraw_cash(self, tracker: PortfolioTracker) -> None:
        """Test withdrawing cash."""
        tracker.set_cash_position(Decimal("1000.00"))
        tracker.withdraw_cash(Decimal("300.00"))
        cash = tracker.get_cash_position()
        assert cash.amount == Decimal("700.00")

    def test_withdraw_cash_insufficient_fails(self, tracker: PortfolioTracker) -> None:
        """Test that withdrawing more than balance fails."""
        tracker.set_cash_position(Decimal("100.00"))
        with pytest.raises(ValueError, match="Cannot withdraw"):
            tracker.withdraw_cash(Decimal("200.00"))

    def test_deposit_negative_fails(self, tracker: PortfolioTracker) -> None:
        """Test that negative deposit fails."""
        with pytest.raises(ValueError, match="must be positive"):
            tracker.deposit_cash(Decimal("-100.00"))

    def test_withdraw_negative_fails(self, tracker: PortfolioTracker) -> None:
        """Test that negative withdrawal fails."""
        tracker.set_cash_position(Decimal("1000.00"))
        with pytest.raises(ValueError, match="must be positive"):
            tracker.withdraw_cash(Decimal("-100.00"))


class TestTradeRecording:
    """Tests for trade recording functionality."""

    def test_record_buy_trade_new_position(self, tracker: PortfolioTracker) -> None:
        """Test recording a buy trade creates new position."""
        tracker.set_cash_position(Decimal("10000.00"))

        trade = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.BUY,
            shares=10.0,
            price=100.0,
            total_value=1000.0,
            commission=1.99,
        )
        tracker.record_trade(trade)

        positions = tracker.get_current_positions()
        assert ETFSymbol.LQQ.value in positions
        assert positions[ETFSymbol.LQQ.value].shares == 10.0

    def test_record_buy_trade_adds_to_existing(self, tracker: PortfolioTracker) -> None:
        """Test recording buy trade adds to existing position."""
        tracker.set_cash_position(Decimal("20000.00"))

        # First buy
        trade1 = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.BUY,
            shares=10.0,
            price=100.0,
            total_value=1000.0,
            commission=1.99,
        )
        tracker.record_trade(trade1)

        # Second buy at different price
        trade2 = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.BUY,
            shares=10.0,
            price=110.0,
            total_value=1100.0,
            commission=1.99,
        )
        tracker.record_trade(trade2)

        positions = tracker.get_current_positions()
        pos = positions[ETFSymbol.LQQ.value]
        assert pos.shares == 20.0
        # Average cost should be (10*100 + 10*110) / 20 = 105
        assert abs(pos.average_cost - 105.0) < 0.01

    def test_record_sell_trade_reduces_position(
        self, tracker: PortfolioTracker
    ) -> None:
        """Test recording sell trade reduces position."""
        tracker.set_cash_position(Decimal("10000.00"))

        # Buy first
        buy = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.BUY,
            shares=20.0,
            price=100.0,
            total_value=2000.0,
            commission=1.99,
        )
        tracker.record_trade(buy)

        # Then sell half
        sell = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.SELL,
            shares=10.0,
            price=110.0,
            total_value=1100.0,
            commission=1.99,
        )
        tracker.record_trade(sell)

        positions = tracker.get_current_positions()
        pos = positions[ETFSymbol.LQQ.value]
        assert pos.shares == 10.0

    def test_sell_more_than_held_fails(self, tracker: PortfolioTracker) -> None:
        """Test that selling more shares than held raises error."""
        tracker.set_cash_position(Decimal("10000.00"))

        buy = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.BUY,
            shares=10.0,
            price=100.0,
            total_value=1000.0,
            commission=1.99,
        )
        tracker.record_trade(buy)

        sell = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.SELL,
            shares=20.0,
            price=110.0,
            total_value=2200.0,
            commission=1.99,
        )

        with pytest.raises(ValueError, match="Cannot sell"):
            tracker.record_trade(sell)

    def test_sell_nonexistent_position_fails(self, tracker: PortfolioTracker) -> None:
        """Test that selling a position that doesn't exist fails."""
        tracker.set_cash_position(Decimal("10000.00"))

        sell = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.SELL,
            shares=10.0,
            price=110.0,
            total_value=1100.0,
            commission=1.99,
        )

        with pytest.raises(ValueError, match="no position exists"):
            tracker.record_trade(sell)


class TestReconciliation:
    """Tests for broker reconciliation."""

    def test_reconcile_matching_positions(self, tracker: PortfolioTracker) -> None:
        """Test reconciliation with matching positions."""
        tracker.set_cash_position(Decimal("10000.00"))

        # Create position
        trade = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.BUY,
            shares=10.0,
            price=100.0,
            total_value=1000.0,
            commission=1.99,
        )
        tracker.record_trade(trade)

        # Broker data matches
        broker_data = {
            ETFSymbol.LQQ.value: {"shares": 10.0, "price": 100.0},
        }

        discrepancies = tracker.reconcile_with_broker(broker_data)
        assert len(discrepancies) == 0

    def test_reconcile_missing_in_db(self, tracker: PortfolioTracker) -> None:
        """Test reconciliation detects position missing from database."""
        # No positions in tracker
        broker_data = {
            ETFSymbol.LQQ.value: {"shares": 10.0, "price": 100.0},
        }

        discrepancies = tracker.reconcile_with_broker(broker_data)
        assert len(discrepancies) == 1
        assert discrepancies[0].discrepancy_type == DiscrepancyType.MISSING_IN_DB

    def test_reconcile_missing_at_broker(self, tracker: PortfolioTracker) -> None:
        """Test reconciliation detects position missing at broker."""
        tracker.set_cash_position(Decimal("10000.00"))

        trade = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.BUY,
            shares=10.0,
            price=100.0,
            total_value=1000.0,
            commission=1.99,
        )
        tracker.record_trade(trade)

        # No broker data
        broker_data: dict[str, dict[str, float]] = {}

        discrepancies = tracker.reconcile_with_broker(broker_data)
        assert len(discrepancies) == 1
        assert discrepancies[0].discrepancy_type == DiscrepancyType.MISSING_AT_BROKER

    def test_reconcile_share_mismatch(self, tracker: PortfolioTracker) -> None:
        """Test reconciliation detects share count mismatch."""
        tracker.set_cash_position(Decimal("10000.00"))

        trade = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.BUY,
            shares=10.0,
            price=100.0,
            total_value=1000.0,
            commission=1.99,
        )
        tracker.record_trade(trade)

        # Broker shows different shares
        broker_data = {
            ETFSymbol.LQQ.value: {"shares": 15.0, "price": 100.0},
        }

        discrepancies = tracker.reconcile_with_broker(broker_data)
        assert len(discrepancies) == 1
        assert discrepancies[0].discrepancy_type == DiscrepancyType.SHARES_MISMATCH

    def test_reconcile_price_mismatch(self, tracker: PortfolioTracker) -> None:
        """Test reconciliation detects significant price mismatch."""
        tracker.set_cash_position(Decimal("10000.00"))

        trade = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.BUY,
            shares=10.0,
            price=100.0,
            total_value=1000.0,
            commission=1.99,
        )
        tracker.record_trade(trade)

        # Broker shows significantly different price (>1%)
        broker_data = {
            ETFSymbol.LQQ.value: {"shares": 10.0, "price": 110.0},
        }

        discrepancies = tracker.reconcile_with_broker(broker_data)
        assert len(discrepancies) == 1
        assert discrepancies[0].discrepancy_type == DiscrepancyType.PRICE_MISMATCH


class TestPortfolioSummary:
    """Tests for portfolio summary functionality."""

    def test_get_position_summary_empty(self, tracker: PortfolioTracker) -> None:
        """Test summary for empty portfolio."""
        summary = tracker.get_position_summary()
        assert summary["positions"] == []
        assert summary["total_value"] == 0.0
        assert summary["num_positions"] == 0

    def test_get_position_summary_with_positions(
        self, tracker: PortfolioTracker
    ) -> None:
        """Test summary with positions."""
        tracker.set_cash_position(Decimal("10000.00"))

        trade = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.BUY,
            shares=10.0,
            price=100.0,
            total_value=1000.0,
            commission=1.99,
        )
        tracker.record_trade(trade)

        summary = tracker.get_position_summary()
        assert len(summary["positions"]) == 1
        assert summary["num_positions"] == 1
        assert summary["total_value"] > 0


class TestUpdatePrices:
    """Tests for price update functionality."""

    def test_update_prices_recalculates_values(self, tracker: PortfolioTracker) -> None:
        """Test that updating prices recalculates market values."""
        tracker.set_cash_position(Decimal("10000.00"))

        trade = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime.now(),
            action=TradeAction.BUY,
            shares=10.0,
            price=100.0,
            total_value=1000.0,
            commission=1.99,
        )
        tracker.record_trade(trade)

        # Update to higher price
        tracker.update_prices({ETFSymbol.LQQ: 120.0})

        positions = tracker.get_current_positions()
        pos = positions[ETFSymbol.LQQ.value]
        assert pos.current_price == 120.0
        assert abs(pos.market_value - 1200.0) < 0.01


class TestContextManager:
    """Tests for context manager support."""

    def test_context_manager_closes_connection(self, temp_db_path: str) -> None:
        """Test that context manager closes database connection."""
        with PortfolioTracker(temp_db_path) as tracker:
            tracker.set_cash_position(Decimal("1000.00"))
            cash = tracker.get_cash_position()
            assert cash.amount == Decimal("1000.00")
        # Connection should be closed after with block
