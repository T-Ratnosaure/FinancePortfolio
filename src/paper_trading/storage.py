"""DuckDB storage layer for Paper Trading Engine.

This module handles persistence of all paper trading data including
sessions, orders, positions, snapshots, and performance metrics.
"""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from src.paper_trading.models import (
    DailyPnLSnapshot,
    OrderResult,
    OrderStatus,
    PaperOrder,
    PortfolioSnapshot,
    SessionConfig,
    SessionStatus,
    TradeRecord,
    VirtualCashPosition,
    VirtualPosition,
)


class PaperTradingStorage:
    """DuckDB storage for paper trading data.

    Manages persistence of sessions, orders, positions, and performance data
    in a dedicated schema within the project's DuckDB database.

    Note: SQL queries use f-strings with SCHEMA constant (not user input).
    This is safe because SCHEMA is a class constant, not user-controlled.
    """

    SCHEMA = "paper_trading"  # noqa: S608 - constant, not user input

    def __init__(self, db_path: str | Path = "data/portfolio.duckdb") -> None:
        """Initialize storage with database path.

        Args:
            db_path: Path to the DuckDB database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get a database connection."""
        return duckdb.connect(str(self.db_path))

    def _init_schema(self) -> None:
        """Initialize the paper trading schema and tables."""
        with self._get_connection() as conn:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.SCHEMA}")

            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.sessions (
                    session_id VARCHAR PRIMARY KEY,
                    session_name VARCHAR NOT NULL,
                    initial_capital DECIMAL(18, 6) NOT NULL,
                    currency VARCHAR DEFAULT 'EUR',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    status VARCHAR DEFAULT 'ACTIVE',
                    config_json TEXT
                )
            """)

            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.orders (
                    order_id VARCHAR PRIMARY KEY,
                    session_id VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    action VARCHAR NOT NULL,
                    order_type VARCHAR NOT NULL,
                    quantity DECIMAL(18, 8) NOT NULL,
                    limit_price DECIMAL(18, 6),
                    created_at TIMESTAMP NOT NULL,
                    status VARCHAR NOT NULL,
                    fill_price DECIMAL(18, 6),
                    fill_quantity DECIMAL(18, 8),
                    fill_timestamp TIMESTAMP,
                    transaction_cost DECIMAL(18, 6),
                    slippage_cost DECIMAL(18, 6),
                    rejection_reason TEXT
                )
            """)

            conn.execute(f"""
                CREATE SEQUENCE IF NOT EXISTS {self.SCHEMA}.positions_seq
            """)

            seq = f"{self.SCHEMA}.positions_seq"
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.positions (
                    id INTEGER DEFAULT nextval('{seq}') PRIMARY KEY,
                    session_id VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    shares DECIMAL(18, 8) NOT NULL,
                    average_cost DECIMAL(18, 6) NOT NULL,
                    current_price DECIMAL(18, 6) NOT NULL,
                    market_value DECIMAL(18, 6) NOT NULL,
                    unrealized_pnl DECIMAL(18, 6) NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_current BOOLEAN DEFAULT TRUE
                )
            """)

            conn.execute(f"""
                CREATE SEQUENCE IF NOT EXISTS {self.SCHEMA}.cash_positions_seq
            """)

            seq = f"{self.SCHEMA}.cash_positions_seq"
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.cash_positions (
                    id INTEGER DEFAULT nextval('{seq}') PRIMARY KEY,
                    session_id VARCHAR NOT NULL,
                    amount DECIMAL(18, 6) NOT NULL,
                    currency VARCHAR DEFAULT 'EUR',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_current BOOLEAN DEFAULT TRUE
                )
            """)

            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.snapshots (
                    snapshot_id VARCHAR PRIMARY KEY,
                    session_id VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    total_value DECIMAL(18, 6) NOT NULL,
                    positions_json TEXT NOT NULL,
                    weights_json TEXT NOT NULL,
                    cash_balance DECIMAL(18, 6) NOT NULL
                )
            """)

            conn.execute(f"""
                CREATE SEQUENCE IF NOT EXISTS {self.SCHEMA}.daily_pnl_seq
            """)

            seq = f"{self.SCHEMA}.daily_pnl_seq"
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.daily_pnl (
                    id INTEGER DEFAULT nextval('{seq}') PRIMARY KEY,
                    session_id VARCHAR NOT NULL,
                    date DATE NOT NULL,
                    starting_value DECIMAL(18, 6) NOT NULL,
                    ending_value DECIMAL(18, 6) NOT NULL,
                    realized_pnl DECIMAL(18, 6) NOT NULL,
                    unrealized_pnl DECIMAL(18, 6) NOT NULL,
                    total_pnl DECIMAL(18, 6) NOT NULL,
                    transaction_costs DECIMAL(18, 6) NOT NULL,
                    net_pnl DECIMAL(18, 6) NOT NULL,
                    trades_executed INTEGER NOT NULL,
                    UNIQUE(session_id, date)
                )
            """)

            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.trades (
                    trade_id VARCHAR PRIMARY KEY,
                    order_id VARCHAR NOT NULL,
                    session_id VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    action VARCHAR NOT NULL,
                    quantity DECIMAL(18, 8) NOT NULL,
                    price DECIMAL(18, 6) NOT NULL,
                    total_value DECIMAL(18, 6) NOT NULL,
                    transaction_cost DECIMAL(18, 6) NOT NULL,
                    slippage_cost DECIMAL(18, 6) NOT NULL,
                    realized_pnl DECIMAL(18, 6) NOT NULL,
                    executed_at TIMESTAMP NOT NULL
                )
            """)

            conn.execute(f"""
                CREATE SEQUENCE IF NOT EXISTS {self.SCHEMA}.signals_seq
            """)

            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.signals (
                    id INTEGER DEFAULT nextval('{self.SCHEMA}.signals_seq') PRIMARY KEY,
                    session_id VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    regime VARCHAR NOT NULL,
                    confidence FLOAT NOT NULL,
                    target_weights_json TEXT NOT NULL,
                    triggered_rebalance BOOLEAN DEFAULT FALSE
                )
            """)

    def create_session(self, session_id: str, config: SessionConfig) -> None:
        """Create a new paper trading session.

        Args:
            session_id: Unique session identifier.
            config: Session configuration.
        """
        config_json = config.model_dump_json()
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.SCHEMA}.sessions
                (session_id, session_name, initial_capital, currency, config_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    session_id,
                    config.session_name,
                    float(config.initial_capital),
                    config.currency,
                    config_json,
                ],
            )

            conn.execute(
                f"""
                INSERT INTO {self.SCHEMA}.cash_positions
                (session_id, amount, currency)
                VALUES (?, ?, ?)
                """,
                [session_id, float(config.initial_capital), config.currency],
            )

    def get_session(
        self, session_id: str
    ) -> tuple[SessionConfig, SessionStatus, datetime, datetime | None] | None:
        """Get session configuration and status.

        Args:
            session_id: Session identifier.

        Returns:
            Tuple of (config, status, created_at, ended_at) or None if not found.
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT config_json, status, created_at, ended_at
                FROM {self.SCHEMA}.sessions
                WHERE session_id = ?
                """,
                [session_id],
            ).fetchone()

        if result is None:
            return None

        config = SessionConfig.model_validate_json(result[0])
        status = SessionStatus(result[1])
        created_at = result[2]
        ended_at = result[3]

        return config, status, created_at, ended_at

    def update_session_status(self, session_id: str, status: SessionStatus) -> None:
        """Update session status.

        Args:
            session_id: Session identifier.
            status: New status.
        """
        with self._get_connection() as conn:
            if status in (SessionStatus.STOPPED, SessionStatus.COMPLETED):
                conn.execute(
                    f"""
                    UPDATE {self.SCHEMA}.sessions
                    SET status = ?, ended_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                    """,
                    [status.value, session_id],
                )
            else:
                conn.execute(
                    f"""
                    UPDATE {self.SCHEMA}.sessions
                    SET status = ?
                    WHERE session_id = ?
                    """,
                    [status.value, session_id],
                )

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all paper trading sessions.

        Returns:
            List of session summaries.
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT session_id, session_name, initial_capital,
                       currency, created_at, ended_at, status
                FROM {self.SCHEMA}.sessions
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [
            {
                "session_id": row[0],
                "session_name": row[1],
                "initial_capital": Decimal(str(row[2])),
                "currency": row[3],
                "created_at": row[4],
                "ended_at": row[5],
                "status": SessionStatus(row[6]),
            }
            for row in result
        ]

    def save_order(self, order: PaperOrder) -> None:
        """Save an order to the database.

        Args:
            order: Order to save.
        """
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.SCHEMA}.orders
                (order_id, session_id, symbol, action, order_type,
                 quantity, limit_price, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    order.order_id,
                    order.session_id,
                    order.symbol,
                    order.action.value,
                    order.order_type.value,
                    float(order.quantity),
                    float(order.limit_price) if order.limit_price else None,
                    order.created_at,
                    OrderStatus.PENDING.value,
                ],
            )

    def update_order_result(self, order_id: str, result: OrderResult) -> None:
        """Update order with execution result.

        Args:
            order_id: Order identifier.
            result: Execution result.
        """
        with self._get_connection() as conn:
            conn.execute(
                f"""
                UPDATE {self.SCHEMA}.orders
                SET status = ?,
                    fill_price = ?,
                    fill_quantity = ?,
                    fill_timestamp = ?,
                    transaction_cost = ?,
                    slippage_cost = ?,
                    rejection_reason = ?
                WHERE order_id = ?
                """,
                [
                    result.status.value,
                    float(result.fill_price) if result.fill_price else None,
                    float(result.fill_quantity) if result.fill_quantity else None,
                    result.fill_timestamp,
                    float(result.transaction_cost),
                    float(result.slippage_cost),
                    result.rejection_reason,
                    order_id,
                ],
            )

    def save_trade(self, trade: TradeRecord) -> None:
        """Save a trade record.

        Args:
            trade: Trade record to save.
        """
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.SCHEMA}.trades
                (trade_id, order_id, session_id, symbol, action,
                 quantity, price, total_value, transaction_cost,
                 slippage_cost, realized_pnl, executed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    trade.trade_id,
                    trade.order_id,
                    trade.session_id,
                    trade.symbol,
                    trade.action.value,
                    float(trade.quantity),
                    float(trade.price),
                    float(trade.total_value),
                    float(trade.transaction_cost),
                    float(trade.slippage_cost),
                    float(trade.realized_pnl),
                    trade.executed_at,
                ],
            )

    def get_trades(self, session_id: str) -> list[TradeRecord]:
        """Get all trades for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of trade records.
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT trade_id, order_id, session_id, symbol, action,
                       quantity, price, total_value, transaction_cost,
                       slippage_cost, realized_pnl, executed_at
                FROM {self.SCHEMA}.trades
                WHERE session_id = ?
                ORDER BY executed_at
                """,
                [session_id],
            ).fetchall()

        from src.paper_trading.models import OrderAction

        return [
            TradeRecord(
                trade_id=row[0],
                order_id=row[1],
                session_id=row[2],
                symbol=row[3],
                action=OrderAction(row[4]),
                quantity=Decimal(str(row[5])),
                price=Decimal(str(row[6])),
                total_value=Decimal(str(row[7])),
                transaction_cost=Decimal(str(row[8])),
                slippage_cost=Decimal(str(row[9])),
                realized_pnl=Decimal(str(row[10])),
                executed_at=row[11],
            )
            for row in result
        ]

    def save_position(self, session_id: str, position: VirtualPosition) -> None:
        """Save or update a position.

        Args:
            session_id: Session identifier.
            position: Position to save.
        """
        with self._get_connection() as conn:
            conn.execute(
                f"""
                UPDATE {self.SCHEMA}.positions
                SET is_current = FALSE
                WHERE session_id = ? AND symbol = ? AND is_current = TRUE
                """,
                [session_id, position.symbol],
            )

            conn.execute(
                f"""
                INSERT INTO {self.SCHEMA}.positions
                (session_id, symbol, shares, average_cost, current_price,
                 market_value, unrealized_pnl, updated_at, is_current)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, TRUE)
                """,
                [
                    session_id,
                    position.symbol,
                    float(position.shares),
                    float(position.average_cost),
                    float(position.current_price),
                    float(position.market_value),
                    float(position.unrealized_pnl),
                    position.last_updated,
                ],
            )

    def get_current_positions(self, session_id: str) -> dict[str, VirtualPosition]:
        """Get current positions for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Dictionary of symbol to position.
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT symbol, shares, average_cost, current_price,
                       market_value, unrealized_pnl, updated_at
                FROM {self.SCHEMA}.positions
                WHERE session_id = ? AND is_current = TRUE AND shares > 0
                """,
                [session_id],
            ).fetchall()

        return {
            row[0]: VirtualPosition(
                symbol=row[0],
                shares=Decimal(str(row[1])),
                average_cost=Decimal(str(row[2])),
                current_price=Decimal(str(row[3])),
                market_value=Decimal(str(row[4])),
                unrealized_pnl=Decimal(str(row[5])),
                last_updated=row[6],
            )
            for row in result
        }

    def save_cash_position(self, session_id: str, cash: VirtualCashPosition) -> None:
        """Save or update cash position.

        Args:
            session_id: Session identifier.
            cash: Cash position to save.
        """
        with self._get_connection() as conn:
            conn.execute(
                f"""
                UPDATE {self.SCHEMA}.cash_positions
                SET is_current = FALSE
                WHERE session_id = ? AND is_current = TRUE
                """,
                [session_id],
            )

            conn.execute(
                f"""
                INSERT INTO {self.SCHEMA}.cash_positions
                (session_id, amount, currency, updated_at, is_current)
                VALUES (?, ?, ?, ?, TRUE)
                """,
                [session_id, float(cash.amount), cash.currency, cash.last_updated],
            )

    def get_cash_position(self, session_id: str) -> VirtualCashPosition | None:
        """Get current cash position.

        Args:
            session_id: Session identifier.

        Returns:
            Cash position or None.
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT amount, currency, updated_at
                FROM {self.SCHEMA}.cash_positions
                WHERE session_id = ? AND is_current = TRUE
                """,
                [session_id],
            ).fetchone()

        if result is None:
            return None

        return VirtualCashPosition(
            amount=Decimal(str(result[0])),
            currency=result[1],
            last_updated=result[2],
        )

    def save_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Save a portfolio snapshot.

        Args:
            snapshot: Snapshot to save.
        """
        positions_json = json.dumps(
            [p.model_dump(mode="json") for p in snapshot.positions]
        )
        weights_json = json.dumps(snapshot.weights)

        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.SCHEMA}.snapshots
                (snapshot_id, session_id, timestamp, total_value,
                 positions_json, weights_json, cash_balance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    snapshot.snapshot_id,
                    snapshot.session_id,
                    snapshot.timestamp,
                    float(snapshot.total_value),
                    positions_json,
                    weights_json,
                    float(snapshot.cash.amount),
                ],
            )

    def get_snapshots(
        self,
        session_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[PortfolioSnapshot]:
        """Get portfolio snapshots for a session.

        Args:
            session_id: Session identifier.
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            List of snapshots.
        """
        query = f"""
            SELECT snapshot_id, session_id, timestamp, total_value,
                   positions_json, weights_json, cash_balance
            FROM {self.SCHEMA}.snapshots
            WHERE session_id = ?
        """
        params: list[Any] = [session_id]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp"

        with self._get_connection() as conn:
            result = conn.execute(query, params).fetchall()

        snapshots = []
        for row in result:
            positions_data = json.loads(row[4])
            positions = [VirtualPosition.model_validate(p) for p in positions_data]
            weights = json.loads(row[5])

            snapshots.append(
                PortfolioSnapshot(
                    snapshot_id=row[0],
                    session_id=row[1],
                    timestamp=row[2],
                    total_value=Decimal(str(row[3])),
                    positions=positions,
                    weights=weights,
                    cash=VirtualCashPosition(amount=Decimal(str(row[6]))),
                )
            )

        return snapshots

    def save_daily_pnl(self, pnl: DailyPnLSnapshot) -> None:
        """Save daily PnL snapshot.

        Args:
            pnl: Daily PnL to save.
        """
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.SCHEMA}.daily_pnl
                (session_id, date, starting_value, ending_value,
                 realized_pnl, unrealized_pnl, total_pnl,
                 transaction_costs, net_pnl, trades_executed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (session_id, date) DO UPDATE SET
                    starting_value = EXCLUDED.starting_value,
                    ending_value = EXCLUDED.ending_value,
                    realized_pnl = EXCLUDED.realized_pnl,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    total_pnl = EXCLUDED.total_pnl,
                    transaction_costs = EXCLUDED.transaction_costs,
                    net_pnl = EXCLUDED.net_pnl,
                    trades_executed = EXCLUDED.trades_executed
                """,
                [
                    pnl.session_id,
                    pnl.pnl_date,
                    float(pnl.starting_value),
                    float(pnl.ending_value),
                    float(pnl.realized_pnl),
                    float(pnl.unrealized_pnl),
                    float(pnl.total_pnl),
                    float(pnl.transaction_costs),
                    float(pnl.net_pnl),
                    pnl.trades_executed,
                ],
            )

    def get_daily_pnl_history(self, session_id: str) -> list[DailyPnLSnapshot]:
        """Get daily PnL history for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of daily PnL snapshots.
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT session_id, date, starting_value, ending_value,
                       realized_pnl, unrealized_pnl, total_pnl,
                       transaction_costs, net_pnl, trades_executed
                FROM {self.SCHEMA}.daily_pnl
                WHERE session_id = ?
                ORDER BY date
                """,
                [session_id],
            ).fetchall()

        return [
            DailyPnLSnapshot(
                session_id=row[0],
                pnl_date=row[1],
                starting_value=Decimal(str(row[2])),
                ending_value=Decimal(str(row[3])),
                realized_pnl=Decimal(str(row[4])),
                unrealized_pnl=Decimal(str(row[5])),
                total_pnl=Decimal(str(row[6])),
                transaction_costs=Decimal(str(row[7])),
                net_pnl=Decimal(str(row[8])),
                trades_executed=row[9],
            )
            for row in result
        ]

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its data.

        Args:
            session_id: Session identifier.
        """
        with self._get_connection() as conn:
            for table in [
                "signals",
                "daily_pnl",
                "snapshots",
                "trades",
                "orders",
                "cash_positions",
                "positions",
                "sessions",
            ]:
                conn.execute(
                    f"DELETE FROM {self.SCHEMA}.{table} WHERE session_id = ?",
                    [session_id],
                )
