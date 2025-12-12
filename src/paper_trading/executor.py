"""Virtual Order Executor for Paper Trading Engine.

This module simulates order execution with realistic fills, including
slippage modeling and transaction cost calculation.
"""

import random
import time
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from src.backtesting.costs import TransactionCostModel
from src.paper_trading.models import (
    FillSimulationConfig,
    OrderAction,
    OrderResult,
    OrderStatus,
    OrderType,
    PaperOrder,
    TradeRecord,
)
from src.paper_trading.portfolio_state import PortfolioStateManager


class VirtualOrderExecutor:
    """Simulates order execution with realistic market behavior.

    Uses the existing TransactionCostModel for cost calculations
    and adds slippage simulation based on market conditions.
    """

    def __init__(
        self,
        cost_model: TransactionCostModel | None = None,
        fill_config: FillSimulationConfig | None = None,
    ) -> None:
        """Initialize the virtual order executor.

        Args:
            cost_model: Transaction cost model (uses default if None).
            fill_config: Fill simulation configuration.
        """
        self.cost_model = cost_model or TransactionCostModel()
        self.fill_config = fill_config or FillSimulationConfig()

    def execute_order(
        self,
        order: PaperOrder,
        current_price: Decimal,
        portfolio_manager: PortfolioStateManager,
        market_volatility: float = 0.02,
    ) -> tuple[OrderResult, TradeRecord | None]:
        """Execute an order with simulated fill.

        Args:
            order: Order to execute.
            current_price: Current market price.
            portfolio_manager: Portfolio state manager.
            market_volatility: Current market volatility (default 2%).

        Returns:
            Tuple of (OrderResult, TradeRecord or None if rejected).
        """
        validation_error = portfolio_manager.validate_order(order, current_price)
        if validation_error:
            return (
                OrderResult(
                    order_id=order.order_id,
                    status=OrderStatus.REJECTED,
                    rejection_reason=validation_error,
                ),
                None,
            )

        if order.order_type == OrderType.LIMIT:
            if order.action == OrderAction.BUY:
                if order.limit_price is not None and current_price > order.limit_price:
                    msg = f"Price {current_price} > limit {order.limit_price}"
                    return (
                        OrderResult(
                            order_id=order.order_id,
                            status=OrderStatus.REJECTED,
                            rejection_reason=msg,
                        ),
                        None,
                    )
            else:
                if order.limit_price is not None and current_price < order.limit_price:
                    msg = f"Price {current_price} < limit {order.limit_price}"
                    return (
                        OrderResult(
                            order_id=order.order_id,
                            status=OrderStatus.REJECTED,
                            rejection_reason=msg,
                        ),
                        None,
                    )

        self._simulate_execution_delay()

        fill_price = self._simulate_fill_price(order, current_price, market_volatility)

        price_deviation = abs(float(fill_price - current_price) / float(current_price))
        if price_deviation > self.fill_config.max_price_deviation_pct:
            return (
                OrderResult(
                    order_id=order.order_id,
                    status=OrderStatus.REJECTED,
                    rejection_reason=(
                        f"Price deviation {price_deviation:.2%} exceeds max "
                        f"{self.fill_config.max_price_deviation_pct:.2%}"
                    ),
                ),
                None,
            )

        transaction_cost, slippage_cost = self._calculate_costs(
            order, fill_price, current_price, market_volatility
        )

        fill_timestamp = datetime.now()
        fill_quantity = order.quantity

        result = OrderResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            fill_timestamp=fill_timestamp,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
        )

        position_before = portfolio_manager.get_position(order.symbol)
        average_cost = position_before.average_cost if position_before else Decimal("0")

        portfolio_manager.update_position_from_order(order, result)

        realized_pnl = Decimal("0")
        if order.action == OrderAction.SELL and position_before:
            realized_pnl = portfolio_manager.calculate_realized_pnl(
                order.symbol,
                fill_quantity,
                fill_price,
                average_cost,
            )

        trade = TradeRecord(
            order_id=order.order_id,
            session_id=order.session_id,
            symbol=order.symbol,
            action=order.action,
            quantity=fill_quantity,
            price=fill_price,
            total_value=fill_price * fill_quantity,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
            realized_pnl=realized_pnl,
            executed_at=fill_timestamp,
        )

        return result, trade

    def _simulate_execution_delay(self) -> None:
        """Simulate execution delay."""
        delay_ms = random.randint(  # noqa: S311 - simulation, not crypto
            self.fill_config.min_execution_delay_ms,
            self.fill_config.max_execution_delay_ms,
        )
        time.sleep(delay_ms / 1000.0)

    def _simulate_fill_price(
        self,
        order: PaperOrder,
        current_price: Decimal,
        market_volatility: float,
    ) -> Decimal:
        """Calculate simulated fill price with slippage.

        Args:
            order: Order being executed.
            current_price: Current market price.
            market_volatility: Market volatility factor.

        Returns:
            Simulated fill price.
        """
        base_slippage = self.fill_config.base_slippage_bps / 10000.0

        vol_adjustment = self.fill_config.volatility_slippage_factor * market_volatility

        random_factor = random.uniform(0.5, 1.5)  # noqa: S311 - simulation

        total_slippage = (base_slippage + vol_adjustment) * random_factor

        if order.action == OrderAction.BUY:
            fill_price = current_price * Decimal(str(1 + total_slippage))
        else:
            fill_price = current_price * Decimal(str(1 - total_slippage))

        return fill_price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_costs(
        self,
        order: PaperOrder,
        fill_price: Decimal,
        current_price: Decimal,
        market_volatility: float,
    ) -> tuple[Decimal, Decimal]:
        """Calculate transaction and slippage costs.

        Args:
            order: Order being executed.
            fill_price: Simulated fill price.
            current_price: Original market price.
            market_volatility: Market volatility.

        Returns:
            Tuple of (transaction_cost, slippage_cost).
        """
        trade_value = float(fill_price * order.quantity)

        total_cost = self.cost_model.calculate_cost(
            trade_value=trade_value,
            symbol=order.symbol,
            market_volatility=market_volatility,
        )

        transaction_cost = Decimal(str(total_cost)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        price_diff = abs(fill_price - current_price)
        slippage_cost = (price_diff * order.quantity).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return transaction_cost, slippage_cost

    def preview_order(
        self,
        order: PaperOrder,
        current_price: Decimal,
        market_volatility: float = 0.02,
    ) -> dict[str, Decimal]:
        """Preview estimated costs for an order without executing.

        Args:
            order: Order to preview.
            current_price: Current market price.
            market_volatility: Market volatility.

        Returns:
            Dictionary with estimated costs.
        """
        trade_value = float(current_price * order.quantity)

        estimated_total_cost = self.cost_model.calculate_cost(
            trade_value=trade_value,
            symbol=order.symbol,
            market_volatility=market_volatility,
        )

        base_slippage = self.fill_config.base_slippage_bps / 10000.0
        vol_adjustment = self.fill_config.volatility_slippage_factor * market_volatility
        estimated_slippage_pct = base_slippage + vol_adjustment
        estimated_slippage = float(current_price) * estimated_slippage_pct

        return {
            "trade_value": current_price * order.quantity,
            "estimated_transaction_cost": Decimal(str(estimated_total_cost)).quantize(
                Decimal("0.01")
            ),
            "estimated_slippage": Decimal(str(estimated_slippage)).quantize(
                Decimal("0.01")
            ),
            "total_estimated_cost": Decimal(
                str(estimated_total_cost + estimated_slippage)
            ).quantize(Decimal("0.01")),
        }


class OrderValidator:
    """Validates orders before execution."""

    @staticmethod
    def validate_order(order: PaperOrder) -> list[str]:
        """Validate order fields.

        Args:
            order: Order to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        if order.quantity <= Decimal("0"):
            errors.append("Quantity must be positive")

        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                errors.append("Limit price required for limit orders")
            elif order.limit_price <= Decimal("0"):
                errors.append("Limit price must be positive")

        if not order.symbol:
            errors.append("Symbol is required")

        return errors
