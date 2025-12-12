"""Portfolio State Manager for Paper Trading Engine.

This module manages the virtual portfolio state throughout a paper trading session,
tracking positions, cash, and portfolio value over time.
"""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from src.paper_trading.models import (
    OrderAction,
    OrderResult,
    OrderStatus,
    PaperOrder,
    PortfolioSnapshot,
    VirtualCashPosition,
    VirtualPosition,
)

if TYPE_CHECKING:
    from src.paper_trading.storage import PaperTradingStorage


class InsufficientFundsError(Exception):
    """Raised when there are insufficient funds for a trade."""

    pass


class InsufficientSharesError(Exception):
    """Raised when there are insufficient shares for a sell order."""

    pass


class PortfolioStateManager:
    """Manages virtual portfolio state for paper trading.

    Tracks positions, cash, and portfolio value over time.
    Provides snapshots for PnL calculation and reporting.
    """

    def __init__(
        self,
        session_id: str,
        initial_cash: Decimal,
        storage: "PaperTradingStorage",
        currency: str = "EUR",
    ) -> None:
        """Initialize portfolio state manager.

        Args:
            session_id: Session identifier.
            initial_cash: Initial cash balance.
            storage: Storage backend.
            currency: Base currency (default EUR).
        """
        self.session_id = session_id
        self.storage = storage
        self.currency = currency

        self._positions: dict[str, VirtualPosition] = {}
        self._cash = VirtualCashPosition(
            amount=initial_cash,
            currency=currency,
            last_updated=datetime.now(),
        )
        self._load_state()

    def _load_state(self) -> None:
        """Load current state from storage."""
        stored_positions = self.storage.get_current_positions(self.session_id)
        if stored_positions:
            self._positions = stored_positions

        stored_cash = self.storage.get_cash_position(self.session_id)
        if stored_cash:
            self._cash = stored_cash

    def get_current_positions(self) -> dict[str, VirtualPosition]:
        """Get current virtual positions.

        Returns:
            Dictionary of symbol to position.
        """
        return self._positions.copy()

    def get_position(self, symbol: str) -> VirtualPosition | None:
        """Get a specific position.

        Args:
            symbol: ETF symbol.

        Returns:
            Position or None if not held.
        """
        return self._positions.get(symbol)

    def get_cash_balance(self) -> VirtualCashPosition:
        """Get current cash balance.

        Returns:
            Cash position.
        """
        return self._cash

    def get_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value.

        Returns:
            Total portfolio value including cash.
        """
        positions_value = sum(pos.market_value for pos in self._positions.values())
        return positions_value + self._cash.amount

    def get_portfolio_weights(self) -> dict[str, float]:
        """Calculate current portfolio weights including cash.

        Returns:
            Dictionary of symbol to weight (including 'CASH').
        """
        total_value = self.get_portfolio_value()
        if total_value == Decimal("0"):
            return {"CASH": 1.0}

        weights: dict[str, float] = {}

        for symbol, position in self._positions.items():
            weights[symbol] = float(position.market_value / total_value)

        weights["CASH"] = float(self._cash.amount / total_value)

        return weights

    def validate_order(self, order: PaperOrder, current_price: Decimal) -> str | None:
        """Validate an order can be executed.

        Args:
            order: Order to validate.
            current_price: Current market price.

        Returns:
            Error message if invalid, None if valid.
        """
        if order.action == OrderAction.BUY:
            required_cash = order.quantity * current_price
            if required_cash > self._cash.amount:
                return (
                    f"Insufficient funds: need {required_cash:.2f} EUR, "
                    f"have {self._cash.amount:.2f} EUR"
                )
        else:
            position = self._positions.get(order.symbol)
            if position is None or position.shares < order.quantity:
                available = position.shares if position else Decimal("0")
                return f"Insufficient shares: need {order.quantity}, have {available}"

        return None

    def update_position_from_order(
        self,
        order: PaperOrder,
        result: OrderResult,
    ) -> VirtualPosition | None:
        """Update position after order execution.

        Args:
            order: The executed order.
            result: Execution result.

        Returns:
            Updated position or None if position closed.

        Raises:
            InsufficientFundsError: If buy order exceeds cash.
            InsufficientSharesError: If sell order exceeds shares.
        """
        if result.status != OrderStatus.FILLED:
            return self._positions.get(order.symbol)

        if result.fill_price is None or result.fill_quantity is None:
            return self._positions.get(order.symbol)

        fill_price = result.fill_price
        fill_quantity = result.fill_quantity
        total_cost = fill_price * fill_quantity + result.transaction_cost

        current_position = self._positions.get(order.symbol)

        if order.action == OrderAction.BUY:
            return self._process_buy(
                order.symbol,
                fill_quantity,
                fill_price,
                total_cost,
                current_position,
            )
        else:
            return self._process_sell(
                order.symbol,
                fill_quantity,
                fill_price,
                total_cost,
                current_position,
            )

    def _process_buy(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        total_cost: Decimal,
        current_position: VirtualPosition | None,
    ) -> VirtualPosition:
        """Process a buy order.

        Args:
            symbol: ETF symbol.
            quantity: Shares to buy.
            price: Execution price.
            total_cost: Total cost including fees.
            current_position: Existing position if any.

        Returns:
            Updated position.

        Raises:
            InsufficientFundsError: If insufficient cash.
        """
        if total_cost > self._cash.amount:
            raise InsufficientFundsError(
                f"Insufficient funds: need {total_cost:.2f}, "
                f"have {self._cash.amount:.2f}"
            )

        self._cash = VirtualCashPosition(
            amount=self._cash.amount - total_cost,
            currency=self.currency,
            last_updated=datetime.now(),
        )
        self.storage.save_cash_position(self.session_id, self._cash)

        if current_position is None:
            new_position = VirtualPosition.create(
                symbol=symbol,
                shares=quantity,
                average_cost=price,
                current_price=price,
            )
        else:
            total_shares = current_position.shares + quantity
            new_avg_cost = (
                (current_position.average_cost * current_position.shares)
                + (price * quantity)
            ) / total_shares

            new_position = VirtualPosition.create(
                symbol=symbol,
                shares=total_shares,
                average_cost=new_avg_cost,
                current_price=price,
            )

        self._positions[symbol] = new_position
        self.storage.save_position(self.session_id, new_position)

        return new_position

    def _process_sell(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        total_cost: Decimal,
        current_position: VirtualPosition | None,
    ) -> VirtualPosition | None:
        """Process a sell order.

        Args:
            symbol: ETF symbol.
            quantity: Shares to sell.
            price: Execution price.
            total_cost: Total cost (negative for sells).
            current_position: Existing position.

        Returns:
            Updated position or None if fully closed.

        Raises:
            InsufficientSharesError: If insufficient shares.
        """
        if current_position is None or current_position.shares < quantity:
            available = current_position.shares if current_position else Decimal("0")
            raise InsufficientSharesError(
                f"Insufficient shares: need {quantity}, have {available}"
            )

        proceeds = price * quantity - (total_cost - price * quantity)
        self._cash = VirtualCashPosition(
            amount=self._cash.amount + proceeds,
            currency=self.currency,
            last_updated=datetime.now(),
        )
        self.storage.save_cash_position(self.session_id, self._cash)

        remaining_shares = current_position.shares - quantity

        if remaining_shares <= Decimal("0"):
            del self._positions[symbol]
            closed_position = VirtualPosition.create(
                symbol=symbol,
                shares=Decimal("0"),
                average_cost=current_position.average_cost,
                current_price=price,
            )
            self.storage.save_position(self.session_id, closed_position)
            return None

        new_position = VirtualPosition.create(
            symbol=symbol,
            shares=remaining_shares,
            average_cost=current_position.average_cost,
            current_price=price,
        )
        self._positions[symbol] = new_position
        self.storage.save_position(self.session_id, new_position)

        return new_position

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        """Update current prices and recalculate values.

        Args:
            prices: Dictionary of symbol to current price.
        """
        now = datetime.now()

        for symbol, price in prices.items():
            if symbol in self._positions:
                position = self._positions[symbol]
                updated_position = VirtualPosition.create(
                    symbol=symbol,
                    shares=position.shares,
                    average_cost=position.average_cost,
                    current_price=price,
                )
                updated_position.last_updated = now
                self._positions[symbol] = updated_position
                self.storage.save_position(self.session_id, updated_position)

    def take_snapshot(self) -> PortfolioSnapshot:
        """Create and persist a portfolio snapshot.

        Returns:
            Portfolio snapshot.
        """
        snapshot = PortfolioSnapshot(
            session_id=self.session_id,
            timestamp=datetime.now(),
            positions=list(self._positions.values()),
            cash=self._cash,
            total_value=self.get_portfolio_value(),
            weights=self.get_portfolio_weights(),  # pyrefly: ignore[bad-argument-type]
        )

        self.storage.save_snapshot(snapshot)
        return snapshot

    def get_snapshots(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[PortfolioSnapshot]:
        """Retrieve historical snapshots.

        Args:
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            List of snapshots.
        """
        return self.storage.get_snapshots(self.session_id, start_time, end_time)

    def calculate_realized_pnl(
        self,
        symbol: str,
        quantity: Decimal,
        sell_price: Decimal,
        average_cost: Decimal,
    ) -> Decimal:
        """Calculate realized P&L for a sell trade.

        Args:
            symbol: ETF symbol.
            quantity: Shares sold.
            sell_price: Sell execution price.
            average_cost: Average cost basis.

        Returns:
            Realized P&L.
        """
        return (sell_price - average_cost) * quantity
