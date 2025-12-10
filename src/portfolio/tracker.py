"""Portfolio tracking and position management.

This module provides the PortfolioTracker class for managing portfolio positions,
recording trades, reconciling with broker data, and calculating performance metrics.
"""

import logging
import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.data.models import (
    CashPosition,
    Discrepancy,
    DiscrepancyType,
    ETFSymbol,
    PerformanceMetrics,
    Position,
    Trade,
    TradeAction,
)
from src.data.storage.duckdb import DuckDBStorage

logger = logging.getLogger(__name__)

# Constants for reconciliation
PRICE_TOLERANCE_PERCENT = 0.01  # 1% tolerance for price discrepancies
SHARES_TOLERANCE = 0.0001  # Tolerance for fractional share differences


class PortfolioTracker:
    """Portfolio tracking and position management.

    This class provides functionality to:
    - Track current portfolio positions
    - Record executed trades
    - Reconcile positions with broker exports
    - Calculate portfolio value and weights
    - Track cash positions separately
    - Calculate performance metrics

    Attributes:
        db: DuckDB storage instance for persistence

    Example:
        >>> tracker = PortfolioTracker("data/portfolio.duckdb")
        >>> positions = tracker.get_current_positions()
        >>> portfolio_value = tracker.get_portfolio_value()
    """

    def __init__(self, db_path: str) -> None:
        """Initialize the portfolio tracker with database connection.

        Args:
            db_path: Path to the DuckDB database file
        """
        self.db = DuckDBStorage(db_path)
        self._ensure_cash_table()
        logger.info(f"PortfolioTracker initialized with database at {db_path}")

    def _ensure_cash_table(self) -> None:
        """Ensure the cash positions table exists in the database."""
        self.db.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS analytics.seq_cash_positions_id START 1
        """)

        self.db.conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics.cash_positions (
                id INTEGER PRIMARY KEY
                    DEFAULT nextval('analytics.seq_cash_positions_id'),
                amount DECIMAL(18, 6) NOT NULL,
                currency VARCHAR NOT NULL DEFAULT 'EUR',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_current BOOLEAN DEFAULT TRUE
            )
        """)

    def get_current_positions(self) -> dict[str, Position]:
        """Load current holdings from database.

        Returns:
            Dictionary mapping ETF symbol string to Position object.
            Returns empty dict if no positions exist (new portfolio).

        Example:
            >>> positions = tracker.get_current_positions()
            >>> for symbol, pos in positions.items():
            ...     print(f"{symbol}: {pos.shares} shares")
        """
        positions_list = self.db.get_positions()
        return {pos.symbol.value: pos for pos in positions_list}

    def record_trade(self, trade: Trade) -> None:
        """Record an executed trade and update positions.

        This method records the trade in the database and updates the
        corresponding position. For BUY trades, it increases shares and
        recalculates average cost. For SELL trades, it decreases shares
        and adjusts cash position.

        Args:
            trade: Trade object containing execution details

        Raises:
            ValueError: If attempting to sell more shares than currently held

        Example:
            >>> trade = Trade(
            ...     symbol=ETFSymbol.LQQ,
            ...     date=datetime.now(),
            ...     action=TradeAction.BUY,
            ...     shares=10.0,
            ...     price=100.0,
            ...     total_value=1000.0,
            ...     commission=1.99
            ... )
            >>> tracker.record_trade(trade)
        """
        # Record the trade
        self.db.insert_trade(trade)

        # Update positions based on trade
        self._update_position_from_trade(trade)

        # Update cash position
        self._update_cash_from_trade(trade)

        logger.info(
            f"Recorded {trade.action.value} trade: {trade.shares} shares of "
            f"{trade.symbol.value} @ {trade.price}"
        )

    def _update_position_from_trade(self, trade: Trade) -> None:
        """Update position based on executed trade.

        Args:
            trade: The executed trade
        """
        current_positions = self.get_current_positions()
        symbol_str = trade.symbol.value

        if trade.action == TradeAction.BUY:
            if symbol_str in current_positions:
                existing = current_positions[symbol_str]
                new_shares = existing.shares + trade.shares
                # Calculate new average cost using weighted average
                total_cost = (
                    existing.shares * existing.average_cost + trade.shares * trade.price
                )
                new_avg_cost = total_cost / new_shares
                new_market_value = new_shares * existing.current_price
                new_unrealized_pnl = new_market_value - (new_shares * new_avg_cost)

                new_position = Position(
                    symbol=trade.symbol,
                    shares=new_shares,
                    average_cost=new_avg_cost,
                    current_price=existing.current_price,
                    market_value=new_market_value,
                    unrealized_pnl=new_unrealized_pnl,
                    weight=0.0,  # Will be recalculated
                )
            else:
                # New position
                new_position = Position(
                    symbol=trade.symbol,
                    shares=trade.shares,
                    average_cost=trade.price,
                    current_price=trade.price,
                    market_value=trade.total_value,
                    unrealized_pnl=0.0,
                    weight=0.0,
                )

            self.db.insert_position(new_position)

        elif trade.action == TradeAction.SELL:
            if symbol_str not in current_positions:
                raise ValueError(
                    f"Cannot sell {trade.symbol.value}: no position exists"
                )

            existing = current_positions[symbol_str]
            if trade.shares > existing.shares:
                raise ValueError(
                    f"Cannot sell {trade.shares} shares of {trade.symbol.value}: "
                    f"only {existing.shares} shares held"
                )

            new_shares = existing.shares - trade.shares

            if new_shares > 0:
                new_market_value = new_shares * existing.current_price
                new_unrealized_pnl = new_market_value - (
                    new_shares * existing.average_cost
                )

                new_position = Position(
                    symbol=trade.symbol,
                    shares=new_shares,
                    average_cost=existing.average_cost,  # Avg cost unchanged on sell
                    current_price=existing.current_price,
                    market_value=new_market_value,
                    unrealized_pnl=new_unrealized_pnl,
                    weight=0.0,
                )
                self.db.insert_position(new_position)
            else:
                # Position closed - mark as not current
                self.db.conn.execute(
                    """
                    UPDATE analytics.portfolio_positions
                    SET is_current = FALSE
                    WHERE symbol = ? AND is_current = TRUE
                """,
                    [symbol_str],
                )

        # Recalculate all weights
        self._recalculate_weights()

    def _update_cash_from_trade(self, trade: Trade) -> None:
        """Update cash position based on executed trade.

        Args:
            trade: The executed trade
        """
        current_cash = self.get_cash_position()
        total_cost = trade.total_value + trade.commission

        if trade.action == TradeAction.BUY:
            new_amount = current_cash.amount - Decimal(str(total_cost))
        else:  # SELL
            net_proceeds = trade.total_value - trade.commission
            new_amount = current_cash.amount + Decimal(str(net_proceeds))

        self.set_cash_position(new_amount)

    def _recalculate_weights(self) -> None:
        """Recalculate portfolio weights for all positions."""
        portfolio_value = self.get_portfolio_value()

        if portfolio_value <= 0:
            return

        positions = self.get_current_positions()
        for _symbol_str, pos in positions.items():
            weight = float(Decimal(str(pos.market_value)) / portfolio_value)
            # Clamp weight to valid range
            weight = max(0.0, min(1.0, weight))

            updated_position = Position(
                symbol=pos.symbol,
                shares=pos.shares,
                average_cost=pos.average_cost,
                current_price=pos.current_price,
                market_value=pos.market_value,
                unrealized_pnl=pos.unrealized_pnl,
                weight=weight,
            )
            self.db.insert_position(updated_position)

    def reconcile_with_broker(
        self, broker_positions: dict[str, dict[str, Any]]
    ) -> list[Discrepancy]:
        """Compare database positions with Boursobank export.

        This method identifies discrepancies between the internal database
        and broker-provided position data, including:
        - Positions in broker but missing from database
        - Positions in database but missing from broker
        - Share count mismatches
        - Price mismatches (beyond tolerance)

        Args:
            broker_positions: Dictionary mapping symbol to position info.
                Expected format:
                {
                    "LQQ.PA": {"shares": 10.0, "price": 105.50},
                    "CL2.PA": {"shares": 5.0, "price": 200.00}
                }

        Returns:
            List of Discrepancy objects describing any differences found.
            Empty list indicates positions are reconciled.

        Example:
            >>> broker_data = {
            ...     "LQQ.PA": {"shares": 10.0, "price": 105.50}
            ... }
            >>> discrepancies = tracker.reconcile_with_broker(broker_data)
            >>> for d in discrepancies:
            ...     print(f"{d.symbol}: {d.description}")
        """
        discrepancies: list[Discrepancy] = []
        db_positions = self.get_current_positions()

        # Check for positions in broker but not in DB
        for symbol, broker_data in broker_positions.items():
            if symbol not in db_positions:
                discrepancies.append(
                    Discrepancy(
                        symbol=symbol,
                        discrepancy_type=DiscrepancyType.MISSING_IN_DB,
                        db_value=None,
                        broker_value=broker_data.get("shares"),
                        difference=broker_data.get("shares"),
                        description=(
                            f"Position {symbol} exists at broker "
                            f"({broker_data.get('shares')} shares) "
                            "but not in database"
                        ),
                    )
                )
                continue

            db_pos = db_positions[symbol]

            # Check share count mismatch
            broker_shares = broker_data.get("shares", 0.0)
            share_diff = abs(db_pos.shares - broker_shares)
            if share_diff > SHARES_TOLERANCE:
                discrepancies.append(
                    Discrepancy(
                        symbol=symbol,
                        discrepancy_type=DiscrepancyType.SHARES_MISMATCH,
                        db_value=db_pos.shares,
                        broker_value=broker_shares,
                        difference=share_diff,
                        description=(
                            f"Share count mismatch for {symbol}: "
                            f"DB has {db_pos.shares}, broker has {broker_shares}"
                        ),
                    )
                )

            # Check price mismatch (beyond tolerance)
            broker_price = broker_data.get("price")
            if broker_price is not None and db_pos.current_price > 0:
                price_diff_pct = (
                    abs(db_pos.current_price - broker_price) / db_pos.current_price
                )
                if price_diff_pct > PRICE_TOLERANCE_PERCENT:
                    discrepancies.append(
                        Discrepancy(
                            symbol=symbol,
                            discrepancy_type=DiscrepancyType.PRICE_MISMATCH,
                            db_value=db_pos.current_price,
                            broker_value=broker_price,
                            difference=abs(db_pos.current_price - broker_price),
                            description=(
                                f"Price mismatch for {symbol}: "
                                f"DB has {db_pos.current_price}, "
                                f"broker has {broker_price} "
                                f"({price_diff_pct * 100:.2f}% difference)"
                            ),
                        )
                    )

        # Check for positions in DB but not at broker
        for symbol in db_positions:
            if symbol not in broker_positions:
                db_pos = db_positions[symbol]
                discrepancies.append(
                    Discrepancy(
                        symbol=symbol,
                        discrepancy_type=DiscrepancyType.MISSING_AT_BROKER,
                        db_value=db_pos.shares,
                        broker_value=None,
                        difference=db_pos.shares,
                        description=(
                            f"Position {symbol} exists in database "
                            f"({db_pos.shares} shares) "
                            "but not found at broker"
                        ),
                    )
                )

        if discrepancies:
            logger.warning(f"Found {len(discrepancies)} discrepancies with broker")
        else:
            logger.info("Portfolio reconciled successfully with broker")

        return discrepancies

    def get_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value including cash.

        Returns:
            Total portfolio value as Decimal (ETF positions + cash)

        Example:
            >>> value = tracker.get_portfolio_value()
            >>> print(f"Portfolio value: {value:.2f} EUR")
        """
        positions = self.get_current_positions()
        etf_value = sum(Decimal(str(pos.market_value)) for pos in positions.values())
        cash = self.get_cash_position()
        return etf_value + cash.amount

    def get_portfolio_weights(self) -> dict[str, float]:
        """Calculate current portfolio weights including cash.

        Returns:
            Dictionary mapping symbol/cash to weight (0-1).
            Weights sum to 1.0.

        Example:
            >>> weights = tracker.get_portfolio_weights()
            >>> print(f"LQQ weight: {weights.get('LQQ.PA', 0) * 100:.1f}%")
        """
        portfolio_value = self.get_portfolio_value()

        if portfolio_value <= 0:
            return {"CASH": 1.0}

        weights: dict[str, float] = {}
        positions = self.get_current_positions()

        for symbol, pos in positions.items():
            weights[symbol] = float(Decimal(str(pos.market_value)) / portfolio_value)

        cash = self.get_cash_position()
        weights["CASH"] = float(cash.amount / portfolio_value)

        return weights

    def update_prices(self, prices: dict[ETFSymbol, float]) -> None:
        """Update current prices for positions and recalculate values.

        This method updates the current price, market value, and unrealized
        P&L for each position that has a price update.

        Args:
            prices: Dictionary mapping ETFSymbol to current price

        Example:
            >>> prices = {
            ...     ETFSymbol.LQQ: 105.50,
            ...     ETFSymbol.CL2: 200.00
            ... }
            >>> tracker.update_prices(prices)
        """
        positions = self.get_current_positions()

        for etf_symbol, new_price in prices.items():
            symbol_str = etf_symbol.value
            if symbol_str not in positions:
                continue

            pos = positions[symbol_str]
            new_market_value = pos.shares * new_price
            new_unrealized_pnl = new_market_value - (pos.shares * pos.average_cost)

            updated_position = Position(
                symbol=etf_symbol,
                shares=pos.shares,
                average_cost=pos.average_cost,
                current_price=new_price,
                market_value=new_market_value,
                unrealized_pnl=new_unrealized_pnl,
                weight=pos.weight,  # Will be recalculated
            )
            self.db.insert_position(updated_position)

        # Recalculate weights after price updates
        self._recalculate_weights()
        logger.info(f"Updated prices for {len(prices)} ETFs")

    def get_trade_history(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[Trade]:
        """Get trade history for a date range.

        Args:
            start_date: Start date (inclusive). None means no lower bound.
            end_date: End date (inclusive). None means no upper bound.

        Returns:
            List of Trade objects ordered by date descending

        Example:
            >>> from datetime import date
            >>> trades = tracker.get_trade_history(
            ...     start_date=date(2024, 1, 1),
            ...     end_date=date(2024, 12, 31)
            ... )
        """
        return self.db.get_trades(start_date=start_date, end_date=end_date)

    def calculate_performance(
        self,
        start_date: date,
        end_date: date,
    ) -> PerformanceMetrics:
        """Calculate portfolio performance metrics for a period.

        Calculates key performance metrics including total return,
        annualized return, volatility, Sharpe ratio, and max drawdown.

        Note: This method requires historical position snapshots or
        trade history to calculate accurate metrics. If insufficient
        data is available, some metrics may be None.

        Args:
            start_date: Period start date
            end_date: Period end date

        Returns:
            PerformanceMetrics object with calculated metrics

        Example:
            >>> from datetime import date
            >>> metrics = tracker.calculate_performance(
            ...     start_date=date(2024, 1, 1),
            ...     end_date=date(2024, 6, 30)
            ... )
            >>> print(f"Total return: {metrics.total_return * 100:.2f}%")
        """
        # Get trade history for the period
        trades = self.get_trade_history(start_date=start_date, end_date=end_date)
        num_trades = len(trades)

        # Calculate days in period
        days_in_period = (end_date - start_date).days
        if days_in_period <= 0:
            raise ValueError("end_date must be after start_date")

        # Get current portfolio value as end value
        end_value = self.get_portfolio_value()

        # Estimate start value from current value and trades
        # This is a simplified calculation - production would use snapshots
        start_value = self._estimate_start_value(
            end_value, trades, start_date, end_date
        )

        # Calculate returns
        if start_value > 0:
            total_return = float((end_value - start_value) / start_value)
        else:
            total_return = 0.0

        # Calculate annualized return
        years = days_in_period / 365.0
        annualized_return: float | None = None
        if years > 0 and start_value > 0:
            if total_return > -1.0:  # Avoid math domain error
                annualized_return = math.pow(1 + total_return, 1 / years) - 1

        # Volatility and Sharpe would require daily return data
        # For now, return None - would be implemented with price history
        volatility: float | None = None
        sharpe_ratio: float | None = None
        max_drawdown: float | None = None

        return PerformanceMetrics(
            start_date=start_date,
            end_date=end_date,
            start_value=start_value,
            end_value=end_value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            num_trades=num_trades,
        )

    def _estimate_start_value(
        self,
        end_value: Decimal,
        trades: list[Trade],
        start_date: date,
        end_date: date,
    ) -> Decimal:
        """Estimate portfolio start value from end value and trades.

        This is a simplified estimation. Production systems should use
        historical position snapshots for accuracy.

        Args:
            end_value: Current portfolio value
            trades: Trades in the period
            start_date: Period start date
            end_date: Period end date

        Returns:
            Estimated start value as Decimal
        """
        # Sum net cash flows from trades
        net_flow = Decimal("0")
        for trade in trades:
            trade_date = (
                trade.date.date() if isinstance(trade.date, datetime) else trade.date
            )
            if start_date <= trade_date <= end_date:
                trade_value = Decimal(str(trade.total_value))
                commission = Decimal(str(trade.commission))
                if trade.action == TradeAction.BUY:
                    net_flow += trade_value + commission  # Cash outflow
                else:
                    net_flow -= trade_value - commission  # Cash inflow

        # Estimated start = end value + net outflows
        # (if we bought more, start value was higher by that amount)
        estimated_start = end_value + net_flow

        # Ensure non-negative
        return max(Decimal("0"), estimated_start)

    def get_cash_position(self) -> CashPosition:
        """Get current cash position.

        Returns:
            CashPosition object with current cash balance

        Example:
            >>> cash = tracker.get_cash_position()
            >>> print(f"Cash: {cash.amount:.2f} {cash.currency}")
        """
        result = self.db.conn.execute(
            """
            SELECT amount, currency, updated_at
            FROM analytics.cash_positions
            WHERE is_current = TRUE
            ORDER BY updated_at DESC
            LIMIT 1
        """
        ).fetchone()

        if result is None:
            # No cash position exists - return zero
            return CashPosition(amount=Decimal("0"), currency="EUR", updated_at=None)

        return CashPosition(
            amount=Decimal(str(result[0])),
            currency=result[1],
            updated_at=result[2],
        )

    def set_cash_position(
        self,
        amount: Decimal,
        currency: str = "EUR",
    ) -> None:
        """Set the current cash position.

        Args:
            amount: Cash amount
            currency: Currency code (default: EUR)

        Example:
            >>> from decimal import Decimal
            >>> tracker.set_cash_position(Decimal("10000.00"))
        """
        # Mark existing cash positions as not current
        self.db.conn.execute("""
            UPDATE analytics.cash_positions
            SET is_current = FALSE
            WHERE is_current = TRUE
        """)

        # Insert new cash position
        self.db.conn.execute(
            """
            INSERT INTO analytics.cash_positions
            (amount, currency, updated_at, is_current)
            VALUES (?, ?, ?, TRUE)
        """,
            [float(amount), currency, datetime.now()],
        )

        logger.info(f"Set cash position to {amount} {currency}")

    def deposit_cash(self, amount: Decimal) -> None:
        """Add cash to the portfolio.

        Args:
            amount: Amount to deposit (must be positive)

        Raises:
            ValueError: If amount is not positive
        """
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")

        current = self.get_cash_position()
        new_amount = current.amount + amount
        self.set_cash_position(new_amount, current.currency)
        logger.info(f"Deposited {amount} {current.currency}")

    def withdraw_cash(self, amount: Decimal) -> None:
        """Remove cash from the portfolio.

        Args:
            amount: Amount to withdraw (must be positive)

        Raises:
            ValueError: If amount is not positive or exceeds balance
        """
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")

        current = self.get_cash_position()
        if amount > current.amount:
            raise ValueError(
                f"Cannot withdraw {amount}: only {current.amount} available"
            )

        new_amount = current.amount - amount
        self.set_cash_position(new_amount, current.currency)
        logger.info(f"Withdrew {amount} {current.currency}")

    def get_position_summary(self) -> dict[str, Any]:
        """Get a summary of all positions including cash.

        Returns:
            Dictionary with position details and portfolio totals

        Example:
            >>> summary = tracker.get_position_summary()
            >>> print(f"Total value: {summary['total_value']}")
        """
        positions = self.get_current_positions()
        cash = self.get_cash_position()
        total_value = self.get_portfolio_value()
        weights = self.get_portfolio_weights()

        position_details = []
        for symbol, pos in positions.items():
            position_details.append(
                {
                    "symbol": symbol,
                    "shares": pos.shares,
                    "average_cost": pos.average_cost,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "weight": weights.get(symbol, 0.0),
                }
            )

        return {
            "positions": position_details,
            "cash": {
                "amount": float(cash.amount),
                "currency": cash.currency,
                "weight": weights.get("CASH", 0.0),
            },
            "total_value": float(total_value),
            "num_positions": len(positions),
        }

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()
        logger.info("PortfolioTracker closed")

    def __enter__(self) -> "PortfolioTracker":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit."""
        self.close()
