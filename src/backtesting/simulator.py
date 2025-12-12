"""Trade execution simulator for backtesting.

This module simulates realistic trade execution with transaction costs,
including rebalancing logic and cost calculations.
"""

from datetime import datetime

from src.backtesting.costs import TransactionCostModel
from src.backtesting.models import MarketConditions, RebalanceResult, Trade


class TradeSimulator:
    """Simulate realistic trade execution with transaction costs.

    The simulator handles portfolio rebalancing by calculating trades needed
    to move from current positions to target weights, then applies realistic
    transaction costs based on market conditions.

    Attributes:
        cost_model: TransactionCostModel for calculating transaction costs
    """

    def __init__(
        self,
        cost_model: TransactionCostModel | None = None,
    ) -> None:
        """Initialize the trade simulator.

        Args:
            cost_model: Optional custom cost model. Uses default if not provided.
        """
        self.cost_model = cost_model or TransactionCostModel()

    def execute_rebalance(
        self,
        current_positions: dict[str, float],
        target_weights: dict[str, float],
        current_prices: dict[str, float],
        portfolio_value: float,
        market_conditions: MarketConditions | None = None,
        timestamp: datetime | None = None,
    ) -> RebalanceResult:
        """Simulate portfolio rebalancing with transaction costs.

        Calculates trades needed to move from current positions to target
        weights, executes them with realistic costs, and returns the result.

        Args:
            current_positions: Current holdings in EUR by symbol
            target_weights: Target allocation weights (must sum to ~1.0)
            current_prices: Current prices by symbol (use 1.0 for CASH)
            portfolio_value: Total portfolio value in EUR
            market_conditions: Optional market conditions for cost adjustment
            timestamp: Optional timestamp for trades (defaults to now)

        Returns:
            RebalanceResult with trades executed and new positions

        Raises:
            ValueError: If inputs are invalid or inconsistent
        """
        if portfolio_value <= 0:
            raise ValueError(f"Portfolio value must be positive, got {portfolio_value}")

        # Validate target weights sum to ~1.0
        total_weight = sum(target_weights.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Target weights must sum to 1.0, got {total_weight}")

        # Use current time if not provided
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate current weights
        current_weights = self._calculate_current_weights(
            current_positions, portfolio_value
        )

        # Calculate trades needed
        trades = self._calculate_trades(
            current_weights,
            target_weights,
            portfolio_value,
            current_prices,
            market_conditions,
            timestamp,
        )

        # Calculate total transaction costs
        total_cost = sum(trade.cost for trade in trades)

        # Calculate new positions after rebalancing
        new_positions = self._calculate_new_positions(
            target_weights,
            portfolio_value,
            total_cost,
        )

        return RebalanceResult(
            trades=trades,
            total_cost=total_cost,
            new_positions=new_positions,  # type: ignore[arg-type]
        )

    def calculate_transaction_cost(
        self,
        symbol: str,
        action: str,
        amount: float,
        market_conditions: MarketConditions | None = None,
    ) -> float:
        """Calculate transaction cost for a single trade.

        Args:
            symbol: Asset symbol
            action: 'BUY' or 'SELL'
            amount: Trade amount in EUR
            market_conditions: Optional market conditions

        Returns:
            Transaction cost in EUR

        Raises:
            ValueError: If action is not BUY or SELL
        """
        if action not in ("BUY", "SELL"):
            raise ValueError(f"Action must be 'BUY' or 'SELL', got {action}")

        # Get volatility from market conditions or use default
        volatility = (
            market_conditions.volatility if market_conditions else 0.15
        )  # 15% default

        # Cash trades have no transaction costs
        if symbol == "CASH":
            return 0.0

        return self.cost_model.calculate_cost(amount, symbol, volatility)

    def _calculate_current_weights(
        self,
        current_positions: dict[str, float],
        portfolio_value: float,
    ) -> dict[str, float]:
        """Calculate current portfolio weights.

        Args:
            current_positions: Current holdings in EUR
            portfolio_value: Total portfolio value

        Returns:
            Dictionary of current weights by symbol
        """
        if portfolio_value <= 0:
            return {}

        weights: dict[str, float] = {}
        for symbol, position_value in current_positions.items():
            weights[symbol] = position_value / portfolio_value

        return weights

    def _calculate_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
        current_prices: dict[str, float],
        market_conditions: MarketConditions | None,
        timestamp: datetime,
    ) -> list[Trade]:
        """Calculate trades needed for rebalancing.

        Args:
            current_weights: Current position weights
            target_weights: Target position weights
            portfolio_value: Total portfolio value
            current_prices: Current prices by symbol
            market_conditions: Market conditions for cost calculation
            timestamp: Timestamp for trades

        Returns:
            List of Trade objects
        """
        trades: list[Trade] = []
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in sorted(all_symbols):  # Sort for deterministic order
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            diff = target_weight - current_weight

            # Skip negligible differences
            if abs(diff) < 1e-6:
                continue

            # Calculate trade amount
            amount = abs(diff) * portfolio_value
            action = "BUY" if diff > 0 else "SELL"

            # Get price (1.0 for CASH)
            price = current_prices.get(symbol, 1.0)

            # Calculate transaction cost
            cost = self.calculate_transaction_cost(
                symbol,
                action,
                amount,
                market_conditions,
            )

            # Create trade record
            trade = Trade(
                symbol=symbol,
                action=action,
                amount=amount,
                price=price,
                cost=cost,
                timestamp=timestamp,
            )
            trades.append(trade)

        return trades

    def _calculate_new_positions(
        self,
        target_weights: dict[str, float],
        portfolio_value: float,
        total_cost: float,
    ) -> dict[str, float]:
        """Calculate new position weights after rebalancing.

        Transaction costs reduce portfolio value, so we need to adjust
        the actual weights to account for this.

        Args:
            target_weights: Target weights before costs
            portfolio_value: Portfolio value before costs
            total_cost: Total transaction costs incurred

        Returns:
            New position weights after accounting for costs
        """
        # New portfolio value after costs
        new_portfolio_value = portfolio_value - total_cost

        if new_portfolio_value <= 0:
            raise ValueError(
                f"Transaction costs ({total_cost}) exceed portfolio value "
                f"({portfolio_value})"
            )

        # Target weights remain the same in terms of proportions,
        # but the total portfolio value is reduced
        # For simplicity, we return the target weights as-is since
        # the proportions don't change, only the absolute values
        return target_weights.copy()
