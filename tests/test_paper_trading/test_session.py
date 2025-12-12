"""Tests for paper trading session management."""

import tempfile
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

from src.paper_trading.models import (
    OrderAction,
    OrderStatus,
    SessionConfig,
    SessionStatus,
)
from src.paper_trading.session import (
    PaperTradingSession,
    SessionNotActiveError,
    SessionNotFoundError,
    delete_session,
    list_sessions,
)


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary database path (path only, not the file)."""
    temp_dir = Path(tempfile.gettempdir())
    return temp_dir / f"test_session_{uuid.uuid4().hex}.duckdb"


@pytest.fixture
def session_config() -> SessionConfig:
    """Create a test session configuration."""
    return SessionConfig(
        session_name="Test Trading Session",
        initial_capital=Decimal("10000.00"),
        auto_rebalance=False,
    )


class TestPaperTradingSession:
    """Tests for PaperTradingSession class."""

    def test_create_new_session(
        self, temp_db: Path, session_config: SessionConfig
    ) -> None:
        """Test creating a new session."""
        session = PaperTradingSession.create_new(session_config, temp_db)

        assert session.is_active
        assert session.status == SessionStatus.ACTIVE
        assert session.get_portfolio_value() == Decimal("10000.00")

        session.stop()

    def test_session_lifecycle(
        self, temp_db: Path, session_config: SessionConfig
    ) -> None:
        """Test full session lifecycle."""
        session = PaperTradingSession.create_new(session_config, temp_db)

        assert session.is_active

        session.pause()
        assert session.status == SessionStatus.PAUSED

        resumed = PaperTradingSession.resume(session.session_id, temp_db)
        assert resumed.status == SessionStatus.PAUSED

        resumed.stop()
        assert resumed.status == SessionStatus.STOPPED

    def test_execute_buy_order(
        self, temp_db: Path, session_config: SessionConfig
    ) -> None:
        """Test executing a buy order."""
        session = PaperTradingSession.create_new(session_config, temp_db)

        result, trade = session.execute_order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            current_price=Decimal("100.00"),
        )

        assert result.status == OrderStatus.FILLED
        assert result.fill_quantity == Decimal("10")
        assert trade is not None
        assert trade.symbol == "LQQ.PA"

        positions = session.get_current_positions()
        assert "LQQ.PA" in positions
        assert positions["LQQ.PA"].shares == Decimal("10")

        cash = session.get_cash_balance()
        assert cash is not None
        assert cash.amount < Decimal("10000.00")

        session.stop()

    def test_execute_sell_order(
        self, temp_db: Path, session_config: SessionConfig
    ) -> None:
        """Test executing a sell order."""
        session = PaperTradingSession.create_new(session_config, temp_db)

        session.execute_order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            current_price=Decimal("100.00"),
        )

        result, trade = session.execute_order(
            symbol="LQQ.PA",
            action=OrderAction.SELL,
            quantity=Decimal("5"),
            current_price=Decimal("110.00"),
        )

        assert result.status == OrderStatus.FILLED
        assert result.fill_quantity == Decimal("5")
        assert trade is not None

        positions = session.get_current_positions()
        assert positions["LQQ.PA"].shares == Decimal("5")

        session.stop()

    def test_reject_insufficient_funds(
        self, temp_db: Path, session_config: SessionConfig
    ) -> None:
        """Test rejecting order with insufficient funds."""
        session = PaperTradingSession.create_new(session_config, temp_db)

        result, trade = session.execute_order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("200"),
            current_price=Decimal("100.00"),
        )

        assert result.status == OrderStatus.REJECTED
        # Pre-trade validation may catch this as multiple violations
        # Check for the funds error message (case-insensitive substring)
        rejection = (result.rejection_reason or "").lower()
        assert "insufficient" in rejection or "cash" in rejection
        assert trade is None

        session.stop()

    def test_reject_insufficient_shares(
        self, temp_db: Path, session_config: SessionConfig
    ) -> None:
        """Test rejecting sell order with insufficient shares."""
        session = PaperTradingSession.create_new(session_config, temp_db)

        result, trade = session.execute_order(
            symbol="LQQ.PA",
            action=OrderAction.SELL,
            quantity=Decimal("10"),
            current_price=Decimal("100.00"),
        )

        assert result.status == OrderStatus.REJECTED
        # Pre-trade validation catches this as "Cannot sell more than current position"
        rejection = (result.rejection_reason or "").lower()
        assert (
            "insufficient" in rejection
            or "sell" in rejection
            or "position" in rejection
        )
        assert trade is None

        session.stop()

    def test_portfolio_weights(
        self, temp_db: Path, session_config: SessionConfig
    ) -> None:
        """Test portfolio weight calculation."""
        session = PaperTradingSession.create_new(session_config, temp_db)

        session.execute_order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            current_price=Decimal("100.00"),
        )

        weights = session.get_portfolio_weights()

        assert "LQQ.PA" in weights
        assert "CASH" in weights
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)

        session.stop()

    def test_take_snapshot(self, temp_db: Path, session_config: SessionConfig) -> None:
        """Test taking portfolio snapshots."""
        session = PaperTradingSession.create_new(session_config, temp_db)

        session.execute_order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            current_price=Decimal("100.00"),
        )

        snapshot = session.take_snapshot()
        assert snapshot is not None
        assert len(snapshot.positions) == 1
        assert "CASH" in snapshot.weights

        snapshots = session.get_snapshots()
        assert len(snapshots) >= 1

        session.stop()

    def test_performance_summary(
        self, temp_db: Path, session_config: SessionConfig
    ) -> None:
        """Test getting performance summary."""
        session = PaperTradingSession.create_new(session_config, temp_db)

        session.execute_order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            current_price=Decimal("100.00"),
        )

        summary = session.get_performance_summary()
        assert summary is not None
        assert summary.session_name == "Test Trading Session"
        assert summary.initial_capital == Decimal("10000.00")
        assert summary.total_trades == 1

        session.stop()

    def test_trade_log(self, temp_db: Path, session_config: SessionConfig) -> None:
        """Test getting trade log."""
        session = PaperTradingSession.create_new(session_config, temp_db)

        session.execute_order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            current_price=Decimal("100.00"),
        )

        trade_log = session.get_trade_log()
        assert len(trade_log) == 1
        assert trade_log[0]["symbol"] == "LQQ.PA"
        assert trade_log[0]["action"] == "BUY"

        session.stop()

    def test_update_prices(self, temp_db: Path, session_config: SessionConfig) -> None:
        """Test updating market prices."""
        session = PaperTradingSession.create_new(session_config, temp_db)

        session.execute_order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            current_price=Decimal("100.00"),
        )

        initial_value = session.get_portfolio_value()

        session.update_prices({"LQQ.PA": Decimal("120.00")})

        positions = session.get_current_positions()
        assert positions["LQQ.PA"].current_price == Decimal("120.00")

        new_value = session.get_portfolio_value()
        assert new_value > initial_value

        session.stop()

    def test_operations_on_inactive_session(
        self, temp_db: Path, session_config: SessionConfig
    ) -> None:
        """Test that operations fail on inactive sessions."""
        session = PaperTradingSession.create_new(session_config, temp_db)
        session.stop()

        with pytest.raises(SessionNotActiveError):
            session.execute_order(
                symbol="LQQ.PA",
                action=OrderAction.BUY,
                quantity=Decimal("10"),
                current_price=Decimal("100.00"),
            )

    def test_resume_nonexistent_session(self, temp_db: Path) -> None:
        """Test resuming a session that doesn't exist."""
        with pytest.raises(SessionNotFoundError):
            PaperTradingSession.resume("nonexistent-session", temp_db)


class TestSessionManagement:
    """Tests for session management functions."""

    def test_list_sessions(self, temp_db: Path, session_config: SessionConfig) -> None:
        """Test listing all sessions."""
        session1 = PaperTradingSession.create_new(session_config, temp_db)
        session2 = PaperTradingSession.create_new(session_config, temp_db)

        sessions = list_sessions(temp_db)

        session_ids = [s["session_id"] for s in sessions]
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids

        session1.stop()
        session2.stop()

    def test_delete_session(self, temp_db: Path, session_config: SessionConfig) -> None:
        """Test deleting a session."""
        session = PaperTradingSession.create_new(session_config, temp_db)
        session_id = session.session_id
        session.stop()

        delete_session(session_id, temp_db)

        sessions = list_sessions(temp_db)
        session_ids = [s["session_id"] for s in sessions]
        assert session_id not in session_ids
