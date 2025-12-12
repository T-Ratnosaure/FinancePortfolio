"""Pydantic models for backtesting framework.

This module defines all data models used in backtesting, including trades,
rebalance results, window results, and comprehensive backtest results.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field

from src.data.models import Regime


class Trade(BaseModel):
    """Individual trade execution record.

    Attributes:
        symbol: Asset symbol (e.g., 'LQQ.PA', 'CL2.PA', 'WPEA.PA', 'CASH')
        action: Trade direction ('BUY' or 'SELL')
        amount: Trade amount in EUR (absolute value)
        price: Execution price (for ETFs; 1.0 for cash)
        cost: Transaction cost incurred in EUR
        timestamp: Execution datetime
    """

    symbol: str
    action: str = Field(pattern=r"^(BUY|SELL)$")
    amount: float = Field(gt=0.0, description="Trade amount in EUR")
    price: float = Field(gt=0.0, description="Execution price")
    cost: float = Field(ge=0.0, description="Transaction cost in EUR")
    timestamp: datetime


class RebalanceResult(BaseModel):
    """Result of a portfolio rebalancing operation.

    Attributes:
        trades: List of trades executed during rebalancing
        total_cost: Total transaction costs for all trades in EUR
        new_positions: Updated position weights after rebalancing
    """

    trades: list[Trade]
    total_cost: float = Field(ge=0.0, description="Total transaction costs in EUR")
    new_positions: dict[str, float] = Field(
        description="Position weights after rebalancing"
    )


class MarketConditions(BaseModel):
    """Market conditions snapshot for cost modeling.

    Attributes:
        volatility: Current annualized market volatility (e.g., 0.20 for 20%)
        regime: Current market regime
        vix_level: VIX index level (if available)
        spread_multiplier: Spread multiplier relative to normal (default 1.0)
    """

    volatility: float = Field(ge=0.0, description="Annualized volatility")
    regime: Regime
    vix_level: float | None = Field(default=None, ge=0.0)
    spread_multiplier: float = Field(
        default=1.0, ge=0.0, description="Spread multiplier"
    )


class WindowResult(BaseModel):
    """Results for a single walk-forward window.

    Attributes:
        train_start: Training period start date
        train_end: Training period end date
        test_start: Testing period start date
        test_end: Testing period end date
        metrics: Performance metrics for this window
    """

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    metrics: dict[str, float] = Field(
        description="Performance metrics (return, sharpe, etc.)"
    )


class BacktestResult(BaseModel):
    """Comprehensive backtesting results.

    Contains all performance metrics, risk metrics, trade statistics,
    and time series data from a complete backtest run.

    Attributes:
        start_date: Backtest period start date
        end_date: Backtest period end date
        total_days: Number of days in backtest period
        total_return: Total return as decimal (e.g., 0.50 for 50%)
        annualized_return: Annualized return as decimal
        volatility: Annualized volatility (standard deviation)
        sharpe_ratio: Annualized Sharpe ratio
        sortino_ratio: Annualized Sortino ratio
        max_drawdown: Maximum drawdown as negative decimal (e.g., -0.20)
        max_drawdown_duration: Maximum drawdown duration in days
        total_trades: Total number of trades executed
        avg_trades_per_month: Average trades per month
        win_rate: Percentage of profitable periods (0 to 1)
        profit_factor: Gross profit / Gross loss
        total_transaction_costs: Total transaction costs in EUR
        costs_as_pct_aum: Annual costs as % of average AUM
        cost_drag_on_returns: Return impact from costs (bps)
        gross_return: Return before transaction costs
        var_95: 1-day Value at Risk at 95% confidence
        expected_shortfall: Expected Shortfall (CVaR)
        beta_to_benchmark: Beta relative to benchmark (optional)
        tracking_error: Tracking error vs benchmark (optional)
        regime_distribution: Percentage of time in each regime
        regime_performance: Performance breakdown by regime
        equity_curve: Not stored in model (computed separately)
        drawdown_series: Not stored in model (computed separately)
        regime_history: Not stored in model (computed separately)
        trade_log: Complete list of all trades
        window_results: Results for each walk-forward window
        look_ahead_bias_check: Whether look-ahead bias check passed
        statistical_significance: Statistical test results (p-values, etc.)
        parameter_stability: Parameter sensitivity test results
    """

    # Period
    start_date: date
    end_date: date
    total_days: int = Field(ge=0)

    # Performance
    total_return: float
    annualized_return: float
    volatility: float = Field(ge=0.0)
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float = Field(le=0.0)
    max_drawdown_duration: int = Field(ge=0, description="Days")

    # Trade Statistics
    total_trades: int = Field(ge=0)
    avg_trades_per_month: float = Field(ge=0.0)
    win_rate: float = Field(ge=0.0, le=1.0)
    profit_factor: float = Field(ge=0.0)

    # Transaction Costs
    total_transaction_costs: float = Field(ge=0.0)
    costs_as_pct_aum: float = Field(ge=0.0)
    cost_drag_on_returns: float = Field(
        description="Return impact in bps"
    )  # Can be negative
    gross_return: float  # Before costs

    # Risk Metrics
    var_95: float = Field(ge=0.0)
    expected_shortfall: float = Field(ge=0.0)
    beta_to_benchmark: float | None = None
    tracking_error: float | None = Field(default=None, ge=0.0)

    # Regime Analysis
    regime_distribution: dict[str, float] = Field(description="% time in each regime")
    regime_performance: dict[str, dict[str, float]] = Field(
        description="Performance by regime"
    )

    # Trade Log
    trade_log: list[Trade]

    # Walk-Forward Windows
    window_results: list[WindowResult]

    # Quality Checks
    look_ahead_bias_check: bool
    statistical_significance: dict[str, float]
    parameter_stability: dict[str, float]

    model_config = {
        "arbitrary_types_allowed": True,
    }


class RegimePerformance(BaseModel):
    """Performance metrics for a specific market regime.

    Attributes:
        regime: Market regime
        num_periods: Number of periods in this regime
        avg_return: Average return during this regime
        volatility: Volatility during this regime
        sharpe_ratio: Sharpe ratio during this regime
        max_drawdown: Maximum drawdown during this regime
    """

    regime: Regime
    num_periods: int = Field(ge=0)
    avg_return: float
    volatility: float = Field(ge=0.0)
    sharpe_ratio: float
    max_drawdown: float = Field(le=0.0)
