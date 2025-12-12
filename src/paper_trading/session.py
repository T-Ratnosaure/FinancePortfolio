"""Paper Trading Session Manager.

This module provides the main orchestrator for paper trading sessions,
coordinating portfolio management, order execution, and performance tracking.
"""

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from src.backtesting.costs import TransactionCostModel
from src.paper_trading.executor import VirtualOrderExecutor
from src.paper_trading.models import (
    FillSimulationConfig,
    OrderAction,
    OrderResult,
    OrderType,
    PaperOrder,
    PerformanceSummary,
    PortfolioSnapshot,
    SessionConfig,
    SessionStatus,
    TradeRecord,
    VirtualCashPosition,
    VirtualPosition,
)
from src.paper_trading.performance import PerformanceTracker
from src.paper_trading.portfolio_state import PortfolioStateManager
from src.paper_trading.storage import PaperTradingStorage


class SessionNotFoundError(Exception):
    """Raised when a session is not found."""

    pass


class SessionNotActiveError(Exception):
    """Raised when an operation requires an active session."""

    pass


class PaperTradingSession:
    """Main orchestrator for paper trading simulation.

    Manages the complete lifecycle of a paper trading session including:
    - Portfolio state initialization and tracking
    - Order execution simulation
    - Performance tracking and reporting

    Example:
        ```python
        config = SessionConfig(
            session_name="Test Strategy",
            initial_capital=Decimal("10000.00"),
        )
        session = PaperTradingSession(config)
        session.start()

        # Execute a trade
        result = session.execute_order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            current_price=Decimal("100.00"),
        )

        # Get performance
        summary = session.get_performance_summary()
        session.stop()
        ```
    """

    def __init__(
        self,
        config: SessionConfig,
        session_id: str | None = None,
        db_path: str | Path = "data/portfolio.duckdb",
        cost_model: TransactionCostModel | None = None,
        fill_config: FillSimulationConfig | None = None,
    ) -> None:
        """Initialize paper trading session.

        Args:
            config: Session configuration.
            session_id: Optional session ID (auto-generated if None).
            db_path: Path to DuckDB database.
            cost_model: Transaction cost model (default if None).
            fill_config: Fill simulation configuration.
        """
        self.config = config
        self.session_id = session_id or str(uuid4())
        self.db_path = Path(db_path)

        self.storage = PaperTradingStorage(self.db_path)
        self.executor = VirtualOrderExecutor(cost_model, fill_config)

        self._status = SessionStatus.ACTIVE
        self._started_at: datetime | None = None
        self._portfolio_manager: PortfolioStateManager | None = None
        self._performance_tracker: PerformanceTracker | None = None

    @classmethod
    def create_new(
        cls,
        config: SessionConfig,
        db_path: str | Path = "data/portfolio.duckdb",
    ) -> "PaperTradingSession":
        """Create and start a new paper trading session.

        Args:
            config: Session configuration.
            db_path: Path to DuckDB database.

        Returns:
            Started paper trading session.
        """
        session = cls(config, db_path=db_path)
        session.start()
        return session

    @classmethod
    def resume(
        cls,
        session_id: str,
        db_path: str | Path = "data/portfolio.duckdb",
    ) -> "PaperTradingSession":
        """Resume an existing paper trading session.

        Args:
            session_id: Session ID to resume.
            db_path: Path to DuckDB database.

        Returns:
            Resumed paper trading session.

        Raises:
            SessionNotFoundError: If session not found.
        """
        storage = PaperTradingStorage(db_path)
        session_data = storage.get_session(session_id)

        if session_data is None:
            raise SessionNotFoundError(f"Session {session_id} not found")

        config, status, created_at, ended_at = session_data

        if status not in (SessionStatus.ACTIVE, SessionStatus.PAUSED):
            raise SessionNotActiveError(
                f"Session {session_id} is {status.value}, cannot resume"
            )

        session = cls(config, session_id=session_id, db_path=db_path)
        session._status = status
        session._started_at = created_at
        session._init_managers()

        if session._performance_tracker:
            session._performance_tracker.load_trades_from_storage()

        return session

    def start(self) -> None:
        """Start the paper trading session."""
        if self._started_at is not None:
            return

        self._started_at = datetime.now()
        self._status = SessionStatus.ACTIVE

        self.storage.create_session(self.session_id, self.config)

        self._init_managers()

        if self._portfolio_manager:
            self._portfolio_manager.take_snapshot()

    def _init_managers(self) -> None:
        """Initialize portfolio and performance managers."""
        self._portfolio_manager = PortfolioStateManager(
            session_id=self.session_id,
            initial_cash=self.config.initial_capital,
            storage=self.storage,
            currency=self.config.currency,
        )

        start_date = self._started_at.date() if self._started_at else date.today()

        self._performance_tracker = PerformanceTracker(
            session_id=self.session_id,
            session_name=self.config.session_name,
            initial_capital=self.config.initial_capital,
            start_date=start_date,
            storage=self.storage,
        )

    def stop(self) -> None:
        """Stop the session and persist final state."""
        self._ensure_not_stopped()

        if self._portfolio_manager:
            self._portfolio_manager.take_snapshot()

        self._status = SessionStatus.STOPPED
        self.storage.update_session_status(self.session_id, SessionStatus.STOPPED)

    def pause(self) -> None:
        """Pause the session (can be resumed later)."""
        self._ensure_active()

        if self._portfolio_manager:
            self._portfolio_manager.take_snapshot()

        self._status = SessionStatus.PAUSED
        self.storage.update_session_status(self.session_id, SessionStatus.PAUSED)

    def _ensure_active(self) -> None:
        """Ensure the session is active."""
        if self._status != SessionStatus.ACTIVE:
            raise SessionNotActiveError(f"Session is {self._status.value}, not ACTIVE")

    def _ensure_not_stopped(self) -> None:
        """Ensure the session is not already stopped."""
        if self._status == SessionStatus.STOPPED:
            raise SessionNotActiveError("Session is already STOPPED")

    @property
    def status(self) -> SessionStatus:
        """Get current session status."""
        return self._status

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self._status == SessionStatus.ACTIVE

    def execute_order(
        self,
        symbol: str,
        action: OrderAction,
        quantity: Decimal,
        current_price: Decimal,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Decimal | None = None,
        market_volatility: float = 0.02,
    ) -> tuple[OrderResult, TradeRecord | None]:
        """Execute an order in the paper trading session.

        Args:
            symbol: ETF symbol (e.g., "LQQ.PA").
            action: Buy or sell.
            quantity: Number of shares.
            current_price: Current market price.
            order_type: Market or limit order.
            limit_price: Limit price (required for limit orders).
            market_volatility: Current market volatility.

        Returns:
            Tuple of (OrderResult, TradeRecord or None if rejected).
        """
        self._ensure_active()

        if self._portfolio_manager is None:
            raise SessionNotActiveError("Session not properly initialized")

        order = PaperOrder(
            session_id=self.session_id,
            symbol=symbol,
            action=action,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
        )

        self.storage.save_order(order)

        result, trade = self.executor.execute_order(
            order=order,
            current_price=current_price,
            portfolio_manager=self._portfolio_manager,
            market_volatility=market_volatility,
        )

        self.storage.update_order_result(order.order_id, result)

        if trade and self._performance_tracker:
            self._performance_tracker.record_trade(trade)

        return result, trade

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        """Update current prices for all positions.

        Args:
            prices: Dictionary of symbol to current price.
        """
        self._ensure_active()

        if self._portfolio_manager:
            self._portfolio_manager.update_prices(prices)

    def get_current_positions(self) -> dict[str, VirtualPosition]:
        """Get current positions.

        Returns:
            Dictionary of symbol to position.
        """
        if self._portfolio_manager is None:
            return {}
        return self._portfolio_manager.get_current_positions()

    def get_cash_balance(self) -> VirtualCashPosition | None:
        """Get current cash balance.

        Returns:
            Cash position or None.
        """
        if self._portfolio_manager is None:
            return None
        return self._portfolio_manager.get_cash_balance()

    def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value.

        Returns:
            Total value including cash.
        """
        if self._portfolio_manager is None:
            return Decimal("0")
        return self._portfolio_manager.get_portfolio_value()

    def get_portfolio_weights(self) -> dict[str, float]:
        """Get current portfolio weights.

        Returns:
            Dictionary of symbol to weight (including 'CASH').
        """
        if self._portfolio_manager is None:
            return {"CASH": 1.0}
        return self._portfolio_manager.get_portfolio_weights()

    def take_snapshot(self) -> PortfolioSnapshot | None:
        """Take a portfolio snapshot.

        Returns:
            Portfolio snapshot or None.
        """
        if self._portfolio_manager is None:
            return None
        return self._portfolio_manager.take_snapshot()

    def get_snapshots(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[PortfolioSnapshot]:
        """Get historical snapshots.

        Args:
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            List of snapshots.
        """
        if self._portfolio_manager is None:
            return []
        return self._portfolio_manager.get_snapshots(start_time, end_time)

    def get_performance_summary(self) -> PerformanceSummary | None:
        """Get performance summary.

        Returns:
            Performance summary or None.
        """
        if self._performance_tracker is None:
            return None
        return self._performance_tracker.get_performance_summary(
            self.get_portfolio_value()
        )

    def get_trade_log(self) -> list[dict]:
        """Get trade log.

        Returns:
            List of trade dictionaries.
        """
        if self._performance_tracker is None:
            return []
        return self._performance_tracker.get_trade_log()

    def record_end_of_day(self) -> None:
        """Record end-of-day snapshot and PnL.

        Should be called at the end of each trading day.
        """
        self._ensure_active()

        if self._portfolio_manager is None or self._performance_tracker is None:
            return

        snapshot = self._portfolio_manager.take_snapshot()

        today = date.today()

        snapshots = self._portfolio_manager.get_snapshots()
        if len(snapshots) >= 2:
            start_value = snapshots[-2].total_value
        else:
            start_value = self.config.initial_capital

        trades_today = self.storage.get_trades(self.session_id)
        today_trades = [t for t in trades_today if t.executed_at.date() == today]

        realized_pnl_today = sum(
            (t.realized_pnl for t in today_trades), start=Decimal("0")
        )
        costs_today = sum(
            (t.transaction_cost + t.slippage_cost for t in today_trades),
            start=Decimal("0"),
        )

        self._performance_tracker.record_daily_pnl(
            pnl_date=today,
            starting_value=start_value,
            ending_value=snapshot.total_value,
            realized_pnl_today=realized_pnl_today,
            transaction_costs_today=costs_today,
            trades_today=len(today_trades),
        )


def list_sessions(
    db_path: str | Path = "data/portfolio.duckdb",
) -> list[dict]:
    """List all paper trading sessions.

    Args:
        db_path: Path to DuckDB database.

    Returns:
        List of session summaries.
    """
    storage = PaperTradingStorage(db_path)
    return storage.list_sessions()


def delete_session(
    session_id: str,
    db_path: str | Path = "data/portfolio.duckdb",
) -> None:
    """Delete a paper trading session and all its data.

    Args:
        session_id: Session ID to delete.
        db_path: Path to DuckDB database.
    """
    storage = PaperTradingStorage(db_path)
    storage.delete_session(session_id)
