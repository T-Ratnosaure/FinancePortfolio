"""Pre-trade risk validation for PEA Portfolio.

This module provides pre-trade validation to ensure orders comply with
risk limits before execution. It checks position limits, leveraged exposure,
cash buffers, order sizes, market hours, and price deviations.

The PreTradeValidator class should be called before any trade execution
to prevent risk limit breaches.
"""

from datetime import datetime, time
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from src.data.models import (
    MAX_LEVERAGED_EXPOSURE,
    MAX_SINGLE_POSITION,
    MIN_CASH_BUFFER,
    ETFSymbol,
)


class OrderType(str, Enum):
    """Order type for trade execution."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderAction(str, Enum):
    """Order action type."""

    BUY = "BUY"
    SELL = "SELL"


class Order(BaseModel):
    """Trade order to be validated before execution.

    Attributes:
        symbol: ETF symbol to trade
        action: BUY or SELL
        quantity: Number of shares to trade
        price: Limit price or current market price
        order_type: Type of order (MARKET, LIMIT, etc.)
        timestamp: Order creation timestamp (defaults to now)
    """

    symbol: str = Field(description="ETF symbol to trade")
    action: OrderAction
    quantity: Decimal = Field(gt=Decimal("0"), description="Number of shares to trade")
    price: Decimal = Field(gt=Decimal("0"), description="Order price per share")
    order_type: OrderType = Field(default=OrderType.MARKET)
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def total_value(self) -> Decimal:
        """Calculate total order value."""
        return self.quantity * self.price


class Portfolio(BaseModel):
    """Portfolio state for pre-trade validation.

    Attributes:
        holdings: Dictionary mapping symbol to number of shares held
        cash_balance: Available cash in EUR
        prices: Current prices per symbol
    """

    holdings: dict[str, Decimal] = Field(
        default_factory=dict,
        description="Shares held by symbol",
    )
    cash_balance: Decimal = Field(
        ge=Decimal("0"),
        description="Available cash in EUR",
    )
    prices: dict[str, Decimal] = Field(
        default_factory=dict,
        description="Current prices per symbol",
    )

    @property
    def total_value(self) -> Decimal:
        """Calculate total portfolio value including cash."""
        holdings_value = sum(
            self.holdings.get(sym, Decimal("0")) * self.prices.get(sym, Decimal("0"))
            for sym in self.holdings
        )
        return holdings_value + self.cash_balance

    def get_position_value(self, symbol: str) -> Decimal:
        """Get current value of a position."""
        shares = self.holdings.get(symbol, Decimal("0"))
        price = self.prices.get(symbol, Decimal("0"))
        return shares * price

    def get_position_weight(self, symbol: str) -> float:
        """Get current weight of a position in portfolio."""
        total = self.total_value
        if total <= 0:
            return 0.0
        return float(self.get_position_value(symbol) / total)


class ValidationResult(BaseModel):
    """Result of pre-trade validation.

    Attributes:
        is_valid: Whether the order passes all validation checks
        warnings: Non-blocking warnings (order can proceed but with caution)
        errors: Blocking errors (order should not proceed)
        suggested_adjustments: Optional adjustments to make order valid
    """

    is_valid: bool = Field(description="Whether order passes validation")
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking warnings",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Blocking validation errors",
    )
    suggested_adjustments: dict[str, float] | None = Field(
        default=None,
        description="Suggested order adjustments to pass validation",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "ValidationResult":
        """Ensure is_valid is consistent with errors."""
        if self.errors and self.is_valid:
            raise ValueError("is_valid cannot be True when there are errors")
        return self


class PreTradeValidatorConfig(BaseModel):
    """Configuration for pre-trade validation limits.

    Attributes:
        max_single_position: Maximum single position weight (default 25%)
        max_leveraged_exposure: Maximum combined leveraged ETF weight (default 30%)
        min_cash_buffer: Minimum cash buffer after trade (default 10%)
        min_order_value: Minimum order value in EUR
        max_order_value: Maximum single order value in EUR
        price_deviation_threshold: Alert threshold for price movement (default 2%)
        market_open: Market opening time (Paris time)
        market_close: Market closing time (Paris time)
    """

    max_single_position: float = Field(
        default=MAX_SINGLE_POSITION,
        ge=0.0,
        le=1.0,
        description="Maximum single position weight",
    )
    max_leveraged_exposure: float = Field(
        default=MAX_LEVERAGED_EXPOSURE,
        ge=0.0,
        le=1.0,
        description="Maximum combined leveraged ETF weight",
    )
    min_cash_buffer: float = Field(
        default=MIN_CASH_BUFFER,
        ge=0.0,
        le=1.0,
        description="Minimum cash buffer after trade",
    )
    min_order_value: Decimal = Field(
        default=Decimal("50.0"),
        ge=Decimal("0"),
        description="Minimum order value in EUR",
    )
    max_order_value: Decimal = Field(
        default=Decimal("100000.0"),
        gt=Decimal("0"),
        description="Maximum single order value in EUR",
    )
    price_deviation_threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Price deviation alert threshold",
    )
    market_open: time = Field(
        default=time(9, 0),
        description="Market opening time (Paris)",
    )
    market_close: time = Field(
        default=time(17, 30),
        description="Market closing time (Paris)",
    )


# Leveraged ETF symbols for exposure calculation
LEVERAGED_SYMBOLS: set[str] = {ETFSymbol.LQQ.value, ETFSymbol.CL2.value}


class PreTradeValidator:
    """Pre-trade risk validator for order validation.

    Validates orders against risk limits before execution to prevent
    limit breaches and ensure portfolio compliance.

    Attributes:
        config: Validation configuration with risk limits
        reference_prices: Optional reference prices for deviation checks
    """

    def __init__(
        self,
        config: PreTradeValidatorConfig | None = None,
        reference_prices: dict[str, Decimal] | None = None,
    ) -> None:
        """Initialize the pre-trade validator.

        Args:
            config: Optional custom configuration. Uses defaults if not provided.
            reference_prices: Optional reference prices for deviation checks
                (e.g., previous close prices for intraday deviation alerts).
        """
        self.config = config or PreTradeValidatorConfig()
        self.reference_prices = reference_prices or {}

    def _aggregate_result(
        self,
        result: ValidationResult,
        warnings: list[str],
        errors: list[str],
        suggested_adjustments: dict[str, float] | None,
    ) -> dict[str, float] | None:
        """Aggregate a validation result into cumulative lists.

        Args:
            result: The validation result to aggregate
            warnings: List to extend with result warnings
            errors: List to extend with result errors
            suggested_adjustments: Current suggested adjustments (or None)

        Returns:
            Updated suggested_adjustments (first non-None value wins)
        """
        errors.extend(result.errors)
        warnings.extend(result.warnings)
        if result.suggested_adjustments and suggested_adjustments is None:
            return result.suggested_adjustments
        return suggested_adjustments

    def validate_order(
        self,
        order: Order,
        portfolio: Portfolio,
    ) -> ValidationResult:
        """Validate an order against all risk limits before execution.

        Runs all validation checks and aggregates results into a single
        ValidationResult. The order is valid only if no blocking errors occur.

        Args:
            order: The order to validate
            portfolio: Current portfolio state

        Returns:
            ValidationResult with validation outcome, warnings, and errors
        """
        warnings: list[str] = []
        errors: list[str] = []
        suggested_adjustments: dict[str, float] | None = None

        # Run all validation checks and aggregate results
        checks = [
            self.check_position_limit(order, portfolio),
            self.check_leveraged_exposure(order, portfolio),
            self.check_cash_buffer(order, portfolio),
            self.check_order_size(order),
            self.check_market_hours(order),
            self.check_price_deviation(order),
        ]

        for result in checks:
            suggested_adjustments = self._aggregate_result(
                result, warnings, errors, suggested_adjustments
            )

        # type: ignore comments needed for pyrefly compatibility with Pydantic LaxStr
        return ValidationResult(
            is_valid=len(errors) == 0,
            warnings=warnings,  # type: ignore[arg-type]
            errors=errors,  # type: ignore[arg-type]
            suggested_adjustments=suggested_adjustments,
        )

    def check_position_limit(
        self,
        order: Order,
        portfolio: Portfolio,
    ) -> ValidationResult:
        """Verify single position does not exceed maximum weight.

        Checks that the order would not cause any single position to exceed
        the maximum position limit (default 25% of portfolio).

        Args:
            order: The order to validate
            portfolio: Current portfolio state

        Returns:
            ValidationResult for position limit check
        """
        total_value = portfolio.total_value
        if total_value <= 0:
            return ValidationResult(
                is_valid=False,
                errors=["Portfolio has no value - cannot validate position limits"],
            )

        # Calculate post-trade position value
        current_position_value = portfolio.get_position_value(order.symbol)

        if order.action == OrderAction.BUY:
            new_position_value = current_position_value + order.total_value
            # Portfolio value increases by order value for buys
            new_total_value = total_value + order.total_value
        else:
            new_position_value = current_position_value - order.total_value
            # Portfolio value stays same for sells (cash replaces position)
            new_total_value = total_value

        if new_position_value < 0:
            return ValidationResult(
                is_valid=False,
                errors=[
                    f"Cannot sell more than current position: "
                    f"trying to sell {order.total_value:.2f} EUR "
                    f"but only hold {current_position_value:.2f} EUR"
                ],
            )

        new_weight = float(new_position_value / new_total_value)

        if new_weight > self.config.max_single_position:
            # Calculate maximum allowed quantity
            max_position_value = (
                Decimal(str(self.config.max_single_position)) * new_total_value
            )
            max_additional_value = max_position_value - current_position_value
            max_quantity = (
                max_additional_value / order.price if order.price > 0 else Decimal("0")
            )

            return ValidationResult(
                is_valid=False,
                errors=[
                    f"Position limit exceeded: {order.symbol} would be "
                    f"{new_weight:.1%} of portfolio "
                    f"(limit: {self.config.max_single_position:.0%})"
                ],
                suggested_adjustments={
                    "max_quantity": float(max(Decimal("0"), max_quantity)),
                    "max_value": float(max(Decimal("0"), max_additional_value)),
                },
            )

        # Warning if approaching limit (within 5% of max)
        warning_threshold = self.config.max_single_position * 0.8
        if new_weight > warning_threshold:
            return ValidationResult(
                is_valid=True,
                warnings=[
                    f"Position {order.symbol} approaching limit: "
                    f"{new_weight:.1%} (limit: {self.config.max_single_position:.0%})"
                ],
            )

        return ValidationResult(is_valid=True)

    def check_leveraged_exposure(
        self,
        order: Order,
        portfolio: Portfolio,
    ) -> ValidationResult:
        """Verify combined leveraged ETF exposure does not exceed limit.

        Checks that LQQ + CL2 combined weight does not exceed the maximum
        leveraged exposure limit (default 30% of portfolio).

        Args:
            order: The order to validate
            portfolio: Current portfolio state

        Returns:
            ValidationResult for leveraged exposure check
        """
        # Only check if order is for a leveraged ETF
        if order.symbol not in LEVERAGED_SYMBOLS:
            return ValidationResult(is_valid=True)

        # Sells of leveraged ETFs are always allowed (they reduce exposure)
        if order.action == OrderAction.SELL:
            return ValidationResult(is_valid=True)

        total_value = portfolio.total_value
        if total_value <= 0:
            return ValidationResult(
                is_valid=False,
                errors=["Portfolio has no value - cannot validate leveraged exposure"],
            )

        # Calculate current leveraged exposure
        current_leveraged_value = Decimal("0")
        for sym in LEVERAGED_SYMBOLS:
            current_leveraged_value += portfolio.get_position_value(sym)

        # Calculate post-trade leveraged exposure (buy only at this point)
        new_leveraged_value = current_leveraged_value + order.total_value
        new_total_value = total_value + order.total_value

        new_leveraged_weight = float(new_leveraged_value / new_total_value)

        if new_leveraged_weight > self.config.max_leveraged_exposure:
            # Calculate maximum allowed additional leveraged exposure
            max_leveraged_value = (
                Decimal(str(self.config.max_leveraged_exposure)) * new_total_value
            )
            max_additional_value = max_leveraged_value - current_leveraged_value
            max_quantity = (
                max_additional_value / order.price if order.price > 0 else Decimal("0")
            )

            return ValidationResult(
                is_valid=False,
                errors=[
                    f"Leveraged exposure limit exceeded: combined leveraged ETFs "
                    f"would be {new_leveraged_weight:.1%} "
                    f"(limit: {self.config.max_leveraged_exposure:.0%})"
                ],
                suggested_adjustments={
                    "max_quantity": float(max(Decimal("0"), max_quantity)),
                    "max_value": float(max(Decimal("0"), max_additional_value)),
                },
            )

        # Warning if approaching limit (within 5% of max)
        warning_threshold = self.config.max_leveraged_exposure * 0.8
        if new_leveraged_weight > warning_threshold:
            return ValidationResult(
                is_valid=True,
                warnings=[
                    f"Leveraged exposure approaching limit: "
                    f"{new_leveraged_weight:.1%} "
                    f"(limit: {self.config.max_leveraged_exposure:.0%})"
                ],
            )

        return ValidationResult(is_valid=True)

    def check_cash_buffer(
        self,
        order: Order,
        portfolio: Portfolio,
    ) -> ValidationResult:
        """Ensure cash buffer remains above minimum after trade.

        For BUY orders, verifies that sufficient cash remains after the
        purchase to maintain the minimum cash buffer (default 10%).

        Args:
            order: The order to validate
            portfolio: Current portfolio state

        Returns:
            ValidationResult for cash buffer check
        """
        # Only relevant for BUY orders
        if order.action != OrderAction.BUY:
            return ValidationResult(is_valid=True)

        total_value = portfolio.total_value
        if total_value <= 0:
            return ValidationResult(
                is_valid=False,
                errors=["Portfolio has no value - cannot validate cash buffer"],
            )

        # Calculate post-trade cash
        new_cash = portfolio.cash_balance - order.total_value
        new_total_value = total_value + order.total_value
        new_cash_weight = float(new_cash / new_total_value)

        if new_cash < 0:
            return ValidationResult(
                is_valid=False,
                errors=[
                    f"Insufficient cash: order requires {order.total_value:.2f} EUR "
                    f"but only {portfolio.cash_balance:.2f} EUR available"
                ],
                suggested_adjustments={
                    "max_value": float(portfolio.cash_balance),
                    "max_quantity": float(portfolio.cash_balance / order.price)
                    if order.price > 0
                    else 0.0,
                },
            )

        if new_cash_weight < self.config.min_cash_buffer:
            # Calculate maximum spend to maintain cash buffer
            min_cash_required = (
                Decimal(str(self.config.min_cash_buffer)) * new_total_value
            )
            max_spend = portfolio.cash_balance - min_cash_required
            max_quantity = max_spend / order.price if order.price > 0 else Decimal("0")

            return ValidationResult(
                is_valid=False,
                errors=[
                    f"Cash buffer violated: cash would be {new_cash_weight:.1%} "
                    f"of portfolio (minimum: {self.config.min_cash_buffer:.0%})"
                ],
                suggested_adjustments={
                    "max_quantity": float(max(Decimal("0"), max_quantity)),
                    "max_value": float(max(Decimal("0"), max_spend)),
                },
            )

        # Warning if approaching minimum (within 3% of limit)
        warning_threshold = self.config.min_cash_buffer * 1.3
        if new_cash_weight < warning_threshold:
            return ValidationResult(
                is_valid=True,
                warnings=[
                    f"Cash buffer approaching minimum: {new_cash_weight:.1%} "
                    f"(minimum: {self.config.min_cash_buffer:.0%})"
                ],
            )

        return ValidationResult(is_valid=True)

    def check_order_size(
        self,
        order: Order,
    ) -> ValidationResult:
        """Validate order size against minimum and maximum limits.

        Checks that the order value falls within the acceptable range
        to avoid tiny inefficient trades or oversized risky trades.

        Args:
            order: The order to validate

        Returns:
            ValidationResult for order size check
        """
        order_value = order.total_value

        if order_value < self.config.min_order_value:
            return ValidationResult(
                is_valid=False,
                errors=[
                    f"Order value {order_value:.2f} EUR is below minimum "
                    f"{self.config.min_order_value:.2f} EUR"
                ],
            )

        if order_value > self.config.max_order_value:
            return ValidationResult(
                is_valid=False,
                errors=[
                    f"Order value {order_value:.2f} EUR exceeds maximum "
                    f"{self.config.max_order_value:.2f} EUR"
                ],
                suggested_adjustments={
                    "max_quantity": float(self.config.max_order_value / order.price)
                    if order.price > 0
                    else 0.0,
                    "max_value": float(self.config.max_order_value),
                },
            )

        return ValidationResult(is_valid=True)

    def check_market_hours(
        self,
        order: Order,
    ) -> ValidationResult:
        """Warn if order is placed outside market hours.

        This is a warning only, as limit orders can be placed outside
        market hours. Market orders outside hours may face execution risks.

        Args:
            order: The order to validate

        Returns:
            ValidationResult with warning if outside market hours
        """
        order_time = order.timestamp.time()

        # Check if weekend (Saturday=5, Sunday=6)
        weekday = order.timestamp.weekday()
        if weekday >= 5:
            return ValidationResult(
                is_valid=True,
                warnings=[
                    f"Order placed on weekend ({order.timestamp.strftime('%A')}). "
                    "Order will be executed when market opens."
                ],
            )

        # Check if outside market hours
        if order_time < self.config.market_open:
            return ValidationResult(
                is_valid=True,
                warnings=[
                    f"Order placed before market open "
                    f"({order_time.strftime('%H:%M')} < "
                    f"{self.config.market_open.strftime('%H:%M')}). "
                    "Order may not execute until market opens."
                ],
            )

        if order_time > self.config.market_close:
            return ValidationResult(
                is_valid=True,
                warnings=[
                    f"Order placed after market close "
                    f"({order_time.strftime('%H:%M')} > "
                    f"{self.config.market_close.strftime('%H:%M')}). "
                    "Order may not execute until next trading day."
                ],
            )

        return ValidationResult(is_valid=True)

    def check_price_deviation(
        self,
        order: Order,
    ) -> ValidationResult:
        """Alert if order price deviates significantly from reference.

        Compares the order price against reference prices (e.g., previous
        close) to detect potential issues with stale prices or unusual
        market movements.

        Args:
            order: The order to validate

        Returns:
            ValidationResult with warning if significant price deviation
        """
        if not self.reference_prices:
            return ValidationResult(is_valid=True)

        reference_price = self.reference_prices.get(order.symbol)
        if reference_price is None or reference_price <= 0:
            return ValidationResult(is_valid=True)

        deviation = abs(float(order.price - reference_price) / float(reference_price))

        if deviation > self.config.price_deviation_threshold:
            direction = "higher" if order.price > reference_price else "lower"
            return ValidationResult(
                is_valid=True,
                warnings=[
                    f"Order price {order.price:.2f} is {deviation:.1%} {direction} "
                    f"than reference price {reference_price:.2f}. "
                    "Please verify current market price."
                ],
            )

        return ValidationResult(is_valid=True)

    def set_reference_prices(self, prices: dict[str, Decimal]) -> None:
        """Update reference prices for deviation checks.

        Args:
            prices: Dictionary mapping symbol to reference price
        """
        self.reference_prices = prices

    def validate_batch(
        self,
        orders: list[Order],
        portfolio: Portfolio,
    ) -> dict[int, ValidationResult]:
        """Validate multiple orders in sequence.

        Validates each order considering the cumulative effect of
        previous orders in the batch on portfolio state.

        Args:
            orders: List of orders to validate
            portfolio: Initial portfolio state

        Returns:
            Dictionary mapping order index to ValidationResult
        """
        results: dict[int, ValidationResult] = {}

        # Create mutable copy of portfolio for cumulative validation
        current_holdings = dict(portfolio.holdings)
        current_cash = portfolio.cash_balance
        current_prices = dict(portfolio.prices)

        for idx, order in enumerate(orders):
            # Create portfolio state for this order
            # type: ignore comments needed for pyrefly compatibility with Pydantic LaxStr
            temp_portfolio = Portfolio(
                holdings=current_holdings,  # type: ignore[arg-type]
                cash_balance=current_cash,
                prices=current_prices,  # type: ignore[arg-type]
            )

            # Validate order
            result = self.validate_order(order, temp_portfolio)
            results[idx] = result

            # If valid, update portfolio state for next order
            if result.is_valid:
                if order.action == OrderAction.BUY:
                    current_holdings[order.symbol] = (
                        current_holdings.get(order.symbol, Decimal("0"))
                        + order.quantity
                    )
                    current_cash -= order.total_value
                else:
                    current_holdings[order.symbol] = (
                        current_holdings.get(order.symbol, Decimal("0"))
                        - order.quantity
                    )
                    current_cash += order.total_value

        return results
