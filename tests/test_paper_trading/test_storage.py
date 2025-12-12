"""Tests for paper trading storage layer."""

from datetime import date, datetime
from decimal import Decimal

from src.paper_trading.models import (
    DailyPnLSnapshot,
    OrderAction,
    OrderResult,
    OrderStatus,
    OrderType,
    PaperOrder,
    PortfolioSnapshot,
    SessionConfig,
    SessionStatus,
    TradeRecord,
    VirtualCashPosition,
    VirtualPosition,
)
from src.paper_trading.storage import PaperTradingStorage


class TestPaperTradingStorage:
    """Tests for PaperTradingStorage class."""

    def test_create_session(
        self, storage: PaperTradingStorage, sample_config: SessionConfig
    ) -> None:
        """Test creating a new session."""
        session_id = "test-session-001"
        storage.create_session(session_id, sample_config)

        result = storage.get_session(session_id)
        assert result is not None

        config, status, created_at, ended_at = result
        assert config.session_name == "Test Session"
        assert config.initial_capital == Decimal("10000.00")
        assert status == SessionStatus.ACTIVE
        assert ended_at is None

    def test_get_nonexistent_session(self, storage: PaperTradingStorage) -> None:
        """Test getting a session that doesn't exist."""
        result = storage.get_session("nonexistent")
        assert result is None

    def test_update_session_status(
        self, storage: PaperTradingStorage, sample_config: SessionConfig
    ) -> None:
        """Test updating session status."""
        session_id = "test-session-002"
        storage.create_session(session_id, sample_config)

        storage.update_session_status(session_id, SessionStatus.STOPPED)

        result = storage.get_session(session_id)
        assert result is not None
        _, status, _, ended_at = result
        assert status == SessionStatus.STOPPED
        assert ended_at is not None

    def test_list_sessions(
        self, storage: PaperTradingStorage, sample_config: SessionConfig
    ) -> None:
        """Test listing all sessions."""
        storage.create_session("session-1", sample_config)
        storage.create_session("session-2", sample_config)

        sessions = storage.list_sessions()
        assert len(sessions) >= 2

        session_ids = [s["session_id"] for s in sessions]
        assert "session-1" in session_ids
        assert "session-2" in session_ids

    def test_save_and_get_order(
        self, storage: PaperTradingStorage, sample_config: SessionConfig
    ) -> None:
        """Test saving and retrieving an order."""
        session_id = "test-session-orders"
        storage.create_session(session_id, sample_config)

        order = PaperOrder(
            session_id=session_id,
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
        )
        storage.save_order(order)

        result = OrderResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            fill_price=Decimal("100.00"),
            fill_quantity=Decimal("10"),
            fill_timestamp=datetime.now(),
            transaction_cost=Decimal("1.00"),
            slippage_cost=Decimal("0.50"),
        )
        storage.update_order_result(order.order_id, result)

    def test_save_and_get_positions(
        self, storage: PaperTradingStorage, sample_config: SessionConfig
    ) -> None:
        """Test saving and retrieving positions."""
        session_id = "test-session-positions"
        storage.create_session(session_id, sample_config)

        position = VirtualPosition.create(
            symbol="LQQ.PA",
            shares=Decimal("10"),
            average_cost=Decimal("100.00"),
            current_price=Decimal("110.00"),
        )
        storage.save_position(session_id, position)

        positions = storage.get_current_positions(session_id)
        assert "LQQ.PA" in positions
        assert positions["LQQ.PA"].shares == Decimal("10")
        assert positions["LQQ.PA"].average_cost == Decimal("100.00")

    def test_update_position(
        self, storage: PaperTradingStorage, sample_config: SessionConfig
    ) -> None:
        """Test updating an existing position."""
        session_id = "test-session-update-pos"
        storage.create_session(session_id, sample_config)

        position1 = VirtualPosition.create(
            symbol="LQQ.PA",
            shares=Decimal("10"),
            average_cost=Decimal("100.00"),
            current_price=Decimal("100.00"),
        )
        storage.save_position(session_id, position1)

        position2 = VirtualPosition.create(
            symbol="LQQ.PA",
            shares=Decimal("15"),
            average_cost=Decimal("105.00"),
            current_price=Decimal("110.00"),
        )
        storage.save_position(session_id, position2)

        positions = storage.get_current_positions(session_id)
        assert positions["LQQ.PA"].shares == Decimal("15")
        assert positions["LQQ.PA"].average_cost == Decimal("105.00")

    def test_cash_position(
        self, storage: PaperTradingStorage, sample_config: SessionConfig
    ) -> None:
        """Test cash position management."""
        session_id = "test-session-cash"
        storage.create_session(session_id, sample_config)

        cash = storage.get_cash_position(session_id)
        assert cash is not None
        assert cash.amount == Decimal("10000.00")

        new_cash = VirtualCashPosition(amount=Decimal("8000.00"))
        storage.save_cash_position(session_id, new_cash)

        cash = storage.get_cash_position(session_id)
        assert cash is not None
        assert cash.amount == Decimal("8000.00")

    def test_save_and_get_snapshot(
        self, storage: PaperTradingStorage, sample_config: SessionConfig
    ) -> None:
        """Test saving and retrieving snapshots."""
        session_id = "test-session-snapshots"
        storage.create_session(session_id, sample_config)

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
            session_id=session_id,
            positions=positions,
            cash=cash,
            total_value=Decimal("6100.00"),
            weights={"LQQ.PA": 0.18, "CASH": 0.82},
        )
        storage.save_snapshot(snapshot)

        snapshots = storage.get_snapshots(session_id)
        assert len(snapshots) == 1
        assert snapshots[0].total_value == Decimal("6100.00")
        assert len(snapshots[0].positions) == 1

    def test_save_and_get_daily_pnl(
        self, storage: PaperTradingStorage, sample_config: SessionConfig
    ) -> None:
        """Test saving and retrieving daily PnL."""
        session_id = "test-session-pnl"
        storage.create_session(session_id, sample_config)

        pnl = DailyPnLSnapshot(
            pnl_date=date.today(),
            session_id=session_id,
            starting_value=Decimal("10000.00"),
            ending_value=Decimal("10100.00"),
            realized_pnl=Decimal("50.00"),
            unrealized_pnl=Decimal("50.00"),
            total_pnl=Decimal("100.00"),
            transaction_costs=Decimal("2.00"),
            net_pnl=Decimal("98.00"),
            trades_executed=2,
        )
        storage.save_daily_pnl(pnl)

        history = storage.get_daily_pnl_history(session_id)
        assert len(history) == 1
        assert history[0].total_pnl == Decimal("100.00")
        assert history[0].trades_executed == 2

    def test_save_and_get_trades(
        self, storage: PaperTradingStorage, sample_config: SessionConfig
    ) -> None:
        """Test saving and retrieving trades."""
        session_id = "test-session-trades"
        storage.create_session(session_id, sample_config)

        trade = TradeRecord(
            order_id="order-001",
            session_id=session_id,
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            price=Decimal("100.00"),
            total_value=Decimal("1000.00"),
            transaction_cost=Decimal("1.00"),
            slippage_cost=Decimal("0.50"),
            realized_pnl=Decimal("0"),
        )
        storage.save_trade(trade)

        trades = storage.get_trades(session_id)
        assert len(trades) == 1
        assert trades[0].symbol == "LQQ.PA"
        assert trades[0].quantity == Decimal("10")

    def test_delete_session(
        self, storage: PaperTradingStorage, sample_config: SessionConfig
    ) -> None:
        """Test deleting a session and all its data."""
        session_id = "test-session-delete"
        storage.create_session(session_id, sample_config)

        position = VirtualPosition.create(
            symbol="LQQ.PA",
            shares=Decimal("10"),
            average_cost=Decimal("100.00"),
            current_price=Decimal("110.00"),
        )
        storage.save_position(session_id, position)

        storage.delete_session(session_id)

        result = storage.get_session(session_id)
        assert result is None

        positions = storage.get_current_positions(session_id)
        assert len(positions) == 0
