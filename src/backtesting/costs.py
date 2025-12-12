"""Transaction cost modeling for backtesting.

This module provides realistic transaction cost calculations for PEA ETF
trading, including bid-ask spreads, slippage, and market impact.
"""


class TransactionCostModel:
    """Model for realistic transaction costs in PEA ETF trading.

    Cost components are modeled as basis points (bps) where 1 bp = 0.01%.
    Base assumptions reflect liquid French ETFs on Euronext Paris.

    Attributes:
        SPREAD_BPS: Bid-ask spread cost (5 bps = 0.05%)
        SLIPPAGE_BPS: Market order slippage (3 bps = 0.03%)
        IMPACT_BPS: Market impact for retail orders (2 bps = 0.02%)
        TOTAL_BPS: Total cost per trade (10 bps = 0.10%)
    """

    # Cost components (basis points)
    SPREAD_BPS: float = 5.0  # 0.05%
    SLIPPAGE_BPS: float = 3.0  # 0.03%
    IMPACT_BPS: float = 2.0  # 0.02%
    TOTAL_BPS: float = 10.0  # 0.10% per side

    def calculate_cost(
        self,
        trade_value: float,
        symbol: str,
        market_volatility: float,
    ) -> float:
        """Calculate total transaction cost for a trade.

        The base cost is adjusted upward during high volatility periods
        to reflect wider bid-ask spreads and increased slippage.

        Args:
            trade_value: Absolute value of the trade in EUR
            symbol: ETF symbol being traded
            market_volatility: Current annualized market volatility (e.g., 0.20)

        Returns:
            Total transaction cost in EUR

        Raises:
            ValueError: If trade_value is negative or market_volatility invalid
        """
        if trade_value < 0:
            raise ValueError(f"trade_value must be non-negative, got {trade_value}")
        if market_volatility < 0:
            raise ValueError(
                f"market_volatility must be non-negative, got {market_volatility}"
            )

        # Base cost: 0.10% (10 bps)
        base_cost = trade_value * (self.TOTAL_BPS / 10000)

        # Volatility adjustment
        # Increase cost by 50% when volatility exceeds 30% annualized
        if market_volatility > 0.30:
            volatility_adjustment = 1.5
        else:
            volatility_adjustment = 1.0

        total_cost = base_cost * volatility_adjustment
        return total_cost
