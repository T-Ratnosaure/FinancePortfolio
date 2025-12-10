"""Portfolio rebalancer for PEA Portfolio Optimization System.

This module provides the Rebalancer class for calculating and optimizing
trade recommendations to realign portfolio allocations with target weights.
It integrates with AllocationOptimizer for drift detection and allocation
validation, adding trade-level optimizations for practical execution.

Features:
- Trade recommendation generation with FIFO tax-lot optimization for PEA
- Minimum trade size filtering to avoid costly small trades
- Trade prioritization (sells before buys for funding)
- Transaction cost estimation
- Comprehensive rebalance reporting
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from src.data.models import (
    ETFSymbol,
    Position,
    TradeAction,
)
from src.signals.allocation import (
    AllocationError,
    AllocationOptimizer,
    RiskLimits,
)


class TradePriority(int, Enum):
    """Priority level for trade execution ordering.

    Lower values indicate higher priority (execute first).
    """

    SELL_LEVERAGED = 1  # Sell leveraged positions first (risk reduction)
    SELL_REGULAR = 2  # Sell non-leveraged positions
    BUY_REGULAR = 3  # Buy non-leveraged positions
    BUY_LEVERAGED = 4  # Buy leveraged positions last


class TradeRecommendation(BaseModel):
    """Trade recommendation with execution details.

    Attributes:
        symbol: ETF symbol to trade
        action: BUY or SELL
        shares: Number of shares to trade
        estimated_value: Estimated trade value in EUR
        reason: Explanation for the trade
        priority: Execution priority (lower = execute first)
        current_weight: Current position weight in portfolio
        target_weight: Target position weight after trade
        drift: Absolute drift from target that triggered this trade
    """

    symbol: str = Field(description="ETF symbol to trade")
    action: TradeAction
    shares: Decimal = Field(gt=Decimal("0"), description="Number of shares to trade")
    estimated_value: Decimal = Field(
        gt=Decimal("0"), description="Estimated trade value in EUR"
    )
    reason: str = Field(description="Explanation for the trade")
    priority: TradePriority = Field(
        default=TradePriority.BUY_REGULAR,
        description="Execution priority (lower = execute first)",
    )
    current_weight: float = Field(ge=0.0, le=1.0, description="Current position weight")
    target_weight: float = Field(ge=0.0, le=1.0, description="Target position weight")
    drift: float = Field(ge=0.0, description="Absolute drift from target")

    @model_validator(mode="after")
    def validate_consistency(self) -> "TradeRecommendation":
        """Validate trade recommendation consistency."""
        # Verify action aligns with weight change
        weight_diff = self.target_weight - self.current_weight
        if self.action == TradeAction.BUY and weight_diff < 0:
            raise ValueError("BUY action requires target_weight > current_weight")
        if self.action == TradeAction.SELL and weight_diff > 0:
            raise ValueError("SELL action requires target_weight < current_weight")
        return self


class RebalanceReport(BaseModel):
    """Summary report for a rebalancing operation.

    Attributes:
        generated_at: Timestamp when report was generated
        as_of_date: Date for which rebalance was calculated
        portfolio_value: Total portfolio value in EUR
        needs_rebalancing: Whether rebalancing is recommended
        max_drift: Maximum position drift from target
        max_drift_symbol: Symbol with maximum drift
        total_trades: Number of trades recommended
        total_sell_value: Total value of sell trades
        total_buy_value: Total value of buy trades
        estimated_costs: Estimated transaction costs
        net_cash_change: Net cash change from trades (positive = cash inflow)
        trades: List of trade recommendations
        current_allocation: Current portfolio weights
        target_allocation: Target portfolio weights
        risk_violations: List of any risk limit violations
        notes: Additional notes or warnings
    """

    generated_at: datetime = Field(default_factory=datetime.now)
    as_of_date: date
    portfolio_value: Decimal = Field(gt=Decimal("0"))
    needs_rebalancing: bool
    max_drift: float = Field(ge=0.0)
    max_drift_symbol: str | None = None
    total_trades: int = Field(ge=0)
    total_sell_value: Decimal = Field(ge=Decimal("0"))
    total_buy_value: Decimal = Field(ge=Decimal("0"))
    estimated_costs: Decimal = Field(ge=Decimal("0"))
    net_cash_change: Decimal = Field(description="Positive = cash inflow from sells")
    trades: list[TradeRecommendation] = Field(default_factory=list)
    current_allocation: dict[str, float]
    target_allocation: dict[str, float]
    risk_violations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RebalancerConfig(BaseModel):
    """Configuration for the Rebalancer.

    Attributes:
        min_trade_value: Minimum trade value in EUR (avoid tiny trades)
        default_commission_rate: Default commission as fraction of trade value
        fixed_commission: Fixed commission per trade in EUR
        spread_cost: Estimated bid-ask spread cost as fraction
        tax_lot_method: Tax lot selection method (FIFO for PEA)
    """

    min_trade_value: Decimal = Field(
        default=Decimal("50.0"),
        ge=Decimal("0"),
        description="Minimum trade value in EUR",
    )
    default_commission_rate: Decimal = Field(
        default=Decimal("0.001"),  # 0.1% typical for PEA brokers
        ge=Decimal("0"),
        le=Decimal("0.05"),
        description="Commission as fraction of trade value",
    )
    fixed_commission: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        description="Fixed commission per trade in EUR",
    )
    spread_cost: Decimal = Field(
        default=Decimal("0.001"),  # 0.1% typical spread
        ge=Decimal("0"),
        le=Decimal("0.05"),
        description="Estimated bid-ask spread cost",
    )
    tax_lot_method: str = Field(
        default="FIFO",
        description="Tax lot selection method (FIFO required for PEA)",
    )


# Symbol mapping for allocation keys to ETFSymbol
ALLOCATION_TO_ETF: dict[str, ETFSymbol] = {
    "LQQ": ETFSymbol.LQQ,
    "CL2": ETFSymbol.CL2,
    "WPEA": ETFSymbol.WPEA,
}

# Leveraged ETF symbols (for priority ordering)
LEVERAGED_SYMBOLS: set[str] = {"LQQ", "CL2"}


class Rebalancer:
    """Portfolio rebalancer for calculating optimal trade recommendations.

    Integrates with AllocationOptimizer for drift detection and validation,
    adding trade-level optimizations including:
    - Minimum trade size filtering
    - Trade prioritization (sells before buys)
    - Transaction cost estimation
    - FIFO tax-lot optimization for PEA accounts

    Attributes:
        optimizer: AllocationOptimizer for allocation calculations
        config: Rebalancer configuration
    """

    def __init__(
        self,
        optimizer: AllocationOptimizer | None = None,
        config: RebalancerConfig | None = None,
        risk_limits: RiskLimits | None = None,
    ) -> None:
        """Initialize the Rebalancer.

        Args:
            optimizer: Optional AllocationOptimizer instance.
                      Creates default if not provided.
            config: Optional RebalancerConfig. Uses defaults if not provided.
            risk_limits: Optional RiskLimits for the optimizer.
                        Ignored if optimizer is provided.
        """
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = AllocationOptimizer(risk_limits=risk_limits)
        self.config = config or RebalancerConfig()

    def calculate_trades(
        self,
        current: dict[str, float],
        target: dict[str, float],
        portfolio_value: Decimal,
        prices: dict[str, Decimal] | None = None,
    ) -> list[TradeRecommendation]:
        """Calculate trades needed to rebalance portfolio.

        Only triggers trades for positions with drift > rebalance threshold (5%).
        Filters out trades below minimum trade value.

        Args:
            current: Current position weights by symbol (e.g., {"LQQ": 0.20})
            target: Target position weights by symbol
            portfolio_value: Total portfolio value in EUR
            prices: Current prices per symbol for share calculation.
                   If None, returns trades with value-based quantities.

        Returns:
            List of TradeRecommendation ordered by priority

        Raises:
            AllocationError: If target allocation violates risk limits
            ValueError: If portfolio_value is not positive
        """
        if portfolio_value <= 0:
            raise ValueError(f"Portfolio value must be positive, got {portfolio_value}")

        # Validate target allocation against risk limits
        is_valid, violations = self.optimizer.validate_allocation(target)
        if not is_valid:
            raise AllocationError(
                f"Target allocation violates risk limits: {', '.join(violations)}",
                violations=violations,
            )

        trades: list[TradeRecommendation] = []

        # Get all symbols involved
        all_symbols = set(current.keys()) | set(target.keys())
        # Exclude CASH from trades (cash is result of trades, not traded itself)
        tradeable_symbols = all_symbols - {"CASH"}

        for symbol in tradeable_symbols:
            current_weight = current.get(symbol, 0.0)
            target_weight = target.get(symbol, 0.0)
            drift = abs(target_weight - current_weight)

            # Only create trade if drift exceeds threshold
            if drift <= self.optimizer.risk_limits.rebalance_threshold:
                continue

            # Calculate trade value
            weight_diff = target_weight - current_weight
            trade_value = abs(Decimal(str(weight_diff)) * portfolio_value)

            # Filter out small trades
            if trade_value < self.config.min_trade_value:
                continue

            # Determine action
            action = TradeAction.BUY if weight_diff > 0 else TradeAction.SELL

            # Calculate shares if prices provided
            if prices and symbol in prices:
                price = prices[symbol]
                shares = (trade_value / price).quantize(Decimal("0.0001"))
            else:
                # Use trade value as proxy for shares when price unknown
                shares = trade_value

            # Determine priority
            priority = self._get_trade_priority(symbol, action)

            # Generate reason
            reason = self._generate_trade_reason(
                symbol, action, current_weight, target_weight, drift
            )

            trade = TradeRecommendation(
                symbol=symbol,
                action=action,
                shares=shares,
                estimated_value=trade_value,
                reason=reason,
                priority=priority,
                current_weight=current_weight,
                target_weight=target_weight,
                drift=drift,
            )
            trades.append(trade)

        # Sort by priority (lower = execute first)
        return sorted(trades, key=lambda t: t.priority.value)

    def optimize_trade_order(
        self,
        trades: list[TradeRecommendation],
    ) -> list[TradeRecommendation]:
        """Optimize trade execution order for best results.

        Ordering logic:
        1. Sells before buys (generate cash for purchases)
        2. Within sells: leveraged first (risk reduction)
        3. Within buys: non-leveraged first, leveraged last

        This ordering minimizes risk exposure during rebalancing and
        ensures sufficient cash for buy orders.

        Args:
            trades: List of trade recommendations to order

        Returns:
            Trades reordered for optimal execution
        """
        if not trades:
            return []

        # Separate sells and buys
        sells = [t for t in trades if t.action == TradeAction.SELL]
        buys = [t for t in trades if t.action == TradeAction.BUY]

        # Sort sells: leveraged first (higher priority value = later)
        sells_sorted = sorted(
            sells,
            key=lambda t: (
                0 if t.symbol in LEVERAGED_SYMBOLS else 1,
                -float(t.estimated_value),  # Larger trades first within category
            ),
        )

        # Sort buys: non-leveraged first, then leveraged
        buys_sorted = sorted(
            buys,
            key=lambda t: (
                1 if t.symbol in LEVERAGED_SYMBOLS else 0,
                -float(t.estimated_value),  # Larger trades first within category
            ),
        )

        # Combine: all sells, then all buys
        result = sells_sorted + buys_sorted

        # Update priorities to reflect final order
        updated_trades: list[TradeRecommendation] = []
        for trade in result:
            # Recalculate priority based on position
            if trade.action == TradeAction.SELL:
                if trade.symbol in LEVERAGED_SYMBOLS:
                    new_priority = TradePriority.SELL_LEVERAGED
                else:
                    new_priority = TradePriority.SELL_REGULAR
            else:
                if trade.symbol in LEVERAGED_SYMBOLS:
                    new_priority = TradePriority.BUY_LEVERAGED
                else:
                    new_priority = TradePriority.BUY_REGULAR

            updated_trade = trade.model_copy(update={"priority": new_priority})
            updated_trades.append(updated_trade)

        return updated_trades

    def estimate_transaction_costs(
        self,
        trades: list[TradeRecommendation],
    ) -> Decimal:
        """Estimate total transaction costs for trades.

        Costs include:
        - Commission (percentage of trade value + fixed fee)
        - Estimated bid-ask spread cost

        Args:
            trades: List of trade recommendations

        Returns:
            Total estimated transaction costs in EUR
        """
        if not trades:
            return Decimal("0.0")

        total_cost = Decimal("0.0")

        for trade in trades:
            # Commission cost
            commission = (
                trade.estimated_value * self.config.default_commission_rate
                + self.config.fixed_commission
            )

            # Spread cost (applied to both buys and sells)
            spread = trade.estimated_value * self.config.spread_cost

            total_cost += commission + spread

        return total_cost.quantize(Decimal("0.01"))

    def _calculate_max_drift(
        self,
        current: dict[str, float],
        target: dict[str, float],
    ) -> tuple[float, str | None]:
        """Calculate maximum drift and the symbol with max drift.

        Args:
            current: Current position weights
            target: Target position weights

        Returns:
            Tuple of (max_drift, symbol_with_max_drift)
        """
        max_drift = 0.0
        max_drift_symbol: str | None = None
        all_symbols = set(current.keys()) | set(target.keys())

        for symbol in all_symbols:
            drift = abs(current.get(symbol, 0.0) - target.get(symbol, 0.0))
            if drift > max_drift:
                max_drift = drift
                max_drift_symbol = symbol

        return max_drift, max_drift_symbol

    def _generate_trades_for_report(
        self,
        current: dict[str, float],
        target: dict[str, float],
        portfolio_value: Decimal,
        prices: dict[str, Decimal] | None,
        is_valid: bool,
        needs_rebalancing: bool,
        violations: list[str],
    ) -> tuple[list[TradeRecommendation], list[str]]:
        """Generate trades and notes for rebalance report.

        Args:
            current: Current position weights
            target: Target position weights
            portfolio_value: Total portfolio value
            prices: Current prices per symbol
            is_valid: Whether target allocation is valid
            needs_rebalancing: Whether rebalancing is needed
            violations: List of risk violations

        Returns:
            Tuple of (trades, notes)
        """
        trades: list[TradeRecommendation] = []
        notes: list[str] = []

        if not is_valid:
            notes.append(
                f"Target allocation has risk violations: {', '.join(violations)}"
            )
        elif needs_rebalancing:
            try:
                trades = self.calculate_trades(current, target, portfolio_value, prices)
                trades = self.optimize_trade_order(trades)
            except AllocationError as e:
                notes.append(f"Could not calculate trades: {e}")
            except ValueError as e:
                notes.append(f"Invalid input: {e}")

        return trades, notes

    def generate_rebalance_report(
        self,
        current: dict[str, float],
        target: dict[str, float],
        portfolio_value: Decimal,
        prices: dict[str, Decimal] | None = None,
        as_of_date: date | None = None,
    ) -> RebalanceReport:
        """Generate comprehensive rebalance report.

        Analyzes current vs target allocation and produces a detailed
        report including trade recommendations, costs, and risk analysis.

        Args:
            current: Current position weights by symbol
            target: Target position weights by symbol
            portfolio_value: Total portfolio value in EUR
            prices: Current prices per symbol (optional)
            as_of_date: Date for the report. Defaults to today.

        Returns:
            RebalanceReport with complete analysis
        """
        if as_of_date is None:
            as_of_date = date.today()

        # Check for risk violations and rebalancing need
        is_valid, violations = self.optimizer.validate_allocation(target)
        needs_rebalancing = self.optimizer.needs_rebalancing(current, target)

        # Calculate max drift
        max_drift, max_drift_symbol = self._calculate_max_drift(current, target)

        # Generate trades
        trades, notes = self._generate_trades_for_report(
            current,
            target,
            portfolio_value,
            prices,
            is_valid,
            needs_rebalancing,
            violations,
        )

        # Calculate trade totals
        total_sell = sum(
            (t.estimated_value for t in trades if t.action == TradeAction.SELL),
            Decimal("0.0"),
        )
        total_buy = sum(
            (t.estimated_value for t in trades if t.action == TradeAction.BUY),
            Decimal("0.0"),
        )

        # Estimate costs and net cash
        estimated_costs = self.estimate_transaction_costs(trades)
        net_cash = total_sell - total_buy - estimated_costs

        # Add contextual notes
        self._add_report_notes(
            notes, needs_rebalancing, max_drift, trades, current, target
        )

        # type: ignore comments needed for pyrefly compatibility with Pydantic LaxStr/LaxFloat
        return RebalanceReport(
            as_of_date=as_of_date,
            portfolio_value=portfolio_value,
            needs_rebalancing=needs_rebalancing,
            max_drift=max_drift,
            max_drift_symbol=max_drift_symbol,
            total_trades=len(trades),
            total_sell_value=total_sell,
            total_buy_value=total_buy,
            estimated_costs=estimated_costs,
            net_cash_change=net_cash,
            trades=trades,
            current_allocation=current,  # type: ignore[arg-type]
            target_allocation=target,  # type: ignore[arg-type]
            risk_violations=violations,  # type: ignore[arg-type]
            notes=notes,  # type: ignore[arg-type]
        )

    def _add_report_notes(
        self,
        notes: list[str],
        needs_rebalancing: bool,
        max_drift: float,
        trades: list[TradeRecommendation],
        current: dict[str, float],
        target: dict[str, float],
    ) -> None:
        """Add contextual notes to rebalance report.

        Args:
            notes: List to append notes to (mutated in place)
            needs_rebalancing: Whether rebalancing is needed
            max_drift: Maximum position drift
            trades: Generated trades
            current: Current allocation
            target: Target allocation
        """
        if not needs_rebalancing:
            notes.append(
                f"Portfolio within tolerance (max drift: {max_drift:.2%}, "
                f"threshold: {self.optimizer.risk_limits.rebalance_threshold:.2%})"
            )
            return

        if not trades:
            return

        all_symbols = set(current.keys()) | set(target.keys())
        filtered_count = len(
            [
                s
                for s in all_symbols - {"CASH"}
                if abs(current.get(s, 0.0) - target.get(s, 0.0))
                > self.optimizer.risk_limits.rebalance_threshold
            ]
        ) - len(trades)

        if filtered_count > 0:
            notes.append(
                f"{filtered_count} trade(s) filtered due to minimum "
                f"trade value (EUR {self.config.min_trade_value})"
            )

    def check_sufficient_shares(
        self,
        positions: dict[str, Position],
        trades: list[TradeRecommendation],
    ) -> tuple[bool, list[str]]:
        """Check if there are sufficient shares for sell trades.

        Args:
            positions: Current positions by symbol
            trades: Trade recommendations to validate

        Returns:
            Tuple of (all_sufficient, list of insufficient trade messages)
        """
        issues: list[str] = []

        for trade in trades:
            if trade.action != TradeAction.SELL:
                continue

            position = positions.get(trade.symbol)
            if position is None:
                issues.append(f"Cannot sell {trade.symbol}: no position held")
                continue

            if Decimal(str(position.shares)) < trade.shares:
                issues.append(
                    f"Insufficient {trade.symbol} shares: "
                    f"need {trade.shares}, have {position.shares}"
                )

        return len(issues) == 0, issues

    def adjust_for_available_cash(
        self,
        trades: list[TradeRecommendation],
        available_cash: Decimal,
    ) -> list[TradeRecommendation]:
        """Adjust buy trades if insufficient cash available.

        Uses FIFO priority order - later buys (leveraged) are reduced first.

        Args:
            trades: Original trade recommendations
            available_cash: Available cash for purchases

        Returns:
            Adjusted trades with reduced buy amounts if necessary
        """
        if not trades:
            return []

        # Calculate cash from sells
        cash_from_sells = sum(
            (t.estimated_value for t in trades if t.action == TradeAction.SELL),
            Decimal("0.0"),
        )

        # Estimate transaction costs
        costs = self.estimate_transaction_costs(trades)

        # Total available for buys
        total_available = available_cash + cash_from_sells - costs

        # Calculate total buy amount
        total_buys = sum(
            (t.estimated_value for t in trades if t.action == TradeAction.BUY),
            Decimal("0.0"),
        )

        # If sufficient cash, return unchanged
        if total_available >= total_buys:
            return trades

        # Need to reduce buys - start from lowest priority (highest value)
        adjusted: list[TradeRecommendation] = []
        remaining_budget = total_available

        # Process sells unchanged
        for trade in trades:
            if trade.action == TradeAction.SELL:
                adjusted.append(trade)

        # Process buys in reverse priority order (reduce lowest priority first)
        buys = sorted(
            [t for t in trades if t.action == TradeAction.BUY],
            key=lambda t: -t.priority.value,  # Highest priority value first
        )

        # First pass: identify what can be fully funded
        fully_funded: list[TradeRecommendation] = []
        to_reduce: list[TradeRecommendation] = []

        for trade in reversed(buys):  # Start with highest priority
            if remaining_budget >= trade.estimated_value:
                remaining_budget -= trade.estimated_value
                fully_funded.append(trade)
            else:
                to_reduce.append(trade)

        # Add fully funded trades
        adjusted.extend(fully_funded)

        # For remaining budget, partially fund highest priority unfunded trade
        if remaining_budget > self.config.min_trade_value and to_reduce:
            # Take the highest priority trade from to_reduce
            trade = to_reduce[0]  # Already in reverse priority order

            # Calculate reduced shares/value
            scale = remaining_budget / trade.estimated_value
            reduced_value = remaining_budget
            reduced_shares = (trade.shares * Decimal(str(scale))).quantize(
                Decimal("0.0001")
            )

            if reduced_value >= self.config.min_trade_value:
                reduced_trade = trade.model_copy(
                    update={
                        "shares": reduced_shares,
                        "estimated_value": reduced_value,
                        "reason": f"{trade.reason} (reduced due to cash constraint)",
                    }
                )
                adjusted.append(reduced_trade)

        # Re-sort by priority
        return sorted(adjusted, key=lambda t: t.priority.value)

    def _get_trade_priority(
        self,
        symbol: str,
        action: TradeAction,
    ) -> TradePriority:
        """Determine trade priority based on symbol and action.

        Args:
            symbol: ETF symbol
            action: Trade action (BUY/SELL)

        Returns:
            TradePriority for ordering
        """
        is_leveraged = symbol in LEVERAGED_SYMBOLS

        if action == TradeAction.SELL:
            return (
                TradePriority.SELL_LEVERAGED
                if is_leveraged
                else TradePriority.SELL_REGULAR
            )
        else:
            return (
                TradePriority.BUY_LEVERAGED
                if is_leveraged
                else TradePriority.BUY_REGULAR
            )

    def _generate_trade_reason(
        self,
        symbol: str,
        action: TradeAction,
        current_weight: float,
        target_weight: float,
        drift: float,
    ) -> str:
        """Generate human-readable reason for trade.

        Args:
            symbol: ETF symbol
            action: Trade action
            current_weight: Current position weight
            target_weight: Target position weight
            drift: Absolute drift from target

        Returns:
            Explanation string for the trade
        """
        action_word = "Increase" if action == TradeAction.BUY else "Reduce"
        drift_pct = drift * 100

        return (
            f"{action_word} {symbol} from {current_weight:.1%} to {target_weight:.1%} "
            f"(drift: {drift_pct:.1f}%)"
        )
