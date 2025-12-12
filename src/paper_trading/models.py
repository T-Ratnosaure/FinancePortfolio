"""Pydantic models for the Paper Trading Engine.

This module defines all data models used throughout the paper trading system,
including orders, positions, snapshots, and configuration.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from src.data.models import Regime


class SessionStatus(str, Enum):
    """Status of a paper trading session."""

    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"


class OrderStatus(str, Enum):
    """Status of an order."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class OrderAction(str, Enum):
    """Order action type."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class SessionConfig(BaseModel):
    """Configuration for a paper trading session."""

    session_name: str = Field(description="Human-readable session name")
    initial_capital: Decimal = Field(
        default=Decimal("10000.00"),
        ge=Decimal("100.00"),
        description="Initial capital in EUR",
    )
    currency: str = Field(default="EUR", description="Base currency")
    auto_rebalance: bool = Field(
        default=True, description="Enable automatic rebalancing on regime changes"
    )
    rebalance_mode: Literal["threshold", "weekly", "monthly"] = Field(
        default="threshold", description="Rebalancing trigger mode"
    )
    rebalance_threshold: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Weight drift threshold for rebalancing (5% default)",
    )
    data_refresh_interval_seconds: int = Field(
        default=60, ge=10, le=3600, description="Price data refresh interval"
    )
    use_live_prices: bool = Field(
        default=True, description="Use real market prices vs simulated"
    )
    execution_delay_seconds: float = Field(
        default=0.5, ge=0.0, le=5.0, description="Simulated execution delay"
    )


class FillSimulationConfig(BaseModel):
    """Configuration for simulating order fills."""

    base_slippage_bps: float = Field(
        default=3.0, ge=0.0, le=50.0, description="Base slippage in basis points"
    )
    volatility_slippage_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Additional slippage per volatility unit",
    )
    min_execution_delay_ms: int = Field(
        default=100, ge=0, le=5000, description="Minimum execution delay in ms"
    )
    max_execution_delay_ms: int = Field(
        default=500, ge=100, le=10000, description="Maximum execution delay in ms"
    )
    enable_partial_fills: bool = Field(
        default=False,
        description="Enable partial fills (typically False for retail PEA orders)",
    )
    max_price_deviation_pct: float = Field(
        default=0.02,
        ge=0.001,
        le=0.10,
        description="Max price deviation before rejection",
    )


class PaperOrder(BaseModel):
    """Order to be executed in paper trading."""

    order_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = Field(description="Associated session ID")
    symbol: str = Field(description="ETF symbol (e.g., 'LQQ.PA')")
    action: OrderAction = Field(description="Buy or sell")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Order type")
    quantity: Decimal = Field(gt=Decimal("0"), description="Number of shares")
    limit_price: Decimal | None = Field(
        default=None, description="Limit price for limit orders"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    time_in_force: Literal["DAY", "GTC"] = Field(
        default="DAY", description="Time in force"
    )

    def model_post_init(self, __context: object) -> None:
        """Validate limit price is set for limit orders."""
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit price required for LIMIT orders")


class OrderResult(BaseModel):
    """Result of order execution."""

    order_id: str = Field(description="Original order ID")
    status: OrderStatus = Field(description="Execution status")
    fill_price: Decimal | None = Field(default=None, description="Executed price")
    fill_quantity: Decimal | None = Field(default=None, description="Executed quantity")
    fill_timestamp: datetime | None = Field(
        default=None, description="Execution timestamp"
    )
    transaction_cost: Decimal = Field(
        default=Decimal("0"), description="Transaction cost incurred"
    )
    slippage_cost: Decimal = Field(
        default=Decimal("0"), description="Slippage cost incurred"
    )
    rejection_reason: str | None = Field(
        default=None, description="Reason for rejection"
    )


class VirtualPosition(BaseModel):
    """Virtual position in paper trading."""

    symbol: str = Field(description="ETF symbol")
    shares: Decimal = Field(ge=Decimal("0"), description="Number of shares held")
    average_cost: Decimal = Field(ge=Decimal("0"), description="Average cost per share")
    current_price: Decimal = Field(ge=Decimal("0"), description="Current market price")
    market_value: Decimal = Field(description="Current market value")
    unrealized_pnl: Decimal = Field(description="Unrealized P&L")
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last price update time"
    )

    @classmethod
    def create(
        cls,
        symbol: str,
        shares: Decimal,
        average_cost: Decimal,
        current_price: Decimal,
    ) -> "VirtualPosition":
        """Create a position with calculated market value and P&L."""
        market_value = shares * current_price
        unrealized_pnl = (current_price - average_cost) * shares
        return cls(
            symbol=symbol,
            shares=shares,
            average_cost=average_cost,
            current_price=current_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
        )


class VirtualCashPosition(BaseModel):
    """Virtual cash position."""

    amount: Decimal = Field(description="Cash amount")
    currency: str = Field(default="EUR", description="Currency")
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last update time"
    )


class PortfolioSnapshot(BaseModel):
    """Point-in-time snapshot of portfolio state."""

    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = Field(description="Associated session ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    positions: list[VirtualPosition] = Field(
        default_factory=list, description="Position snapshots"
    )
    cash: VirtualCashPosition = Field(description="Cash position")
    total_value: Decimal = Field(description="Total portfolio value")
    weights: dict[str, float] = Field(
        default_factory=dict, description="Position weights including cash"
    )


class PriceTick(BaseModel):
    """Single price update."""

    symbol: str = Field(description="ETF symbol")
    price: Decimal = Field(gt=Decimal("0"), description="Current price")
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str = Field(default="Yahoo Finance", description="Data source")


class DailyPnLSnapshot(BaseModel):
    """Daily PnL record for paper trading."""

    pnl_date: date = Field(description="Date of the snapshot")
    session_id: str = Field(description="Associated session ID")
    starting_value: Decimal = Field(description="Portfolio value at start of day")
    ending_value: Decimal = Field(description="Portfolio value at end of day")
    realized_pnl: Decimal = Field(description="Realized P&L for the day")
    unrealized_pnl: Decimal = Field(description="Unrealized P&L change")
    total_pnl: Decimal = Field(description="Total P&L for the day")
    transaction_costs: Decimal = Field(description="Transaction costs for the day")
    net_pnl: Decimal = Field(description="Net P&L after costs")
    trades_executed: int = Field(ge=0, description="Number of trades executed")


class PerformanceSummary(BaseModel):
    """Summary of paper trading performance."""

    session_id: str = Field(description="Session ID")
    session_name: str = Field(description="Session name")
    start_date: date = Field(description="Session start date")
    end_date: date | None = Field(default=None, description="Session end date")
    initial_capital: Decimal = Field(description="Initial capital")
    current_value: Decimal = Field(description="Current portfolio value")
    total_return: float = Field(description="Total return percentage")
    annualized_return: float | None = Field(
        default=None, description="Annualized return"
    )
    total_pnl: Decimal = Field(description="Total P&L")
    realized_pnl: Decimal = Field(description="Realized P&L")
    unrealized_pnl: Decimal = Field(description="Unrealized P&L")
    total_transaction_costs: Decimal = Field(description="Total transaction costs")
    total_trades: int = Field(ge=0, description="Total number of trades")
    winning_trades: int = Field(ge=0, description="Number of winning trades")
    losing_trades: int = Field(ge=0, description="Number of losing trades")
    win_rate: float | None = Field(default=None, description="Win rate percentage")
    max_drawdown: float | None = Field(default=None, description="Maximum drawdown")
    sharpe_ratio: float | None = Field(default=None, description="Sharpe ratio")
    sortino_ratio: float | None = Field(default=None, description="Sortino ratio")
    volatility: float | None = Field(default=None, description="Portfolio volatility")


class SignalEvent(BaseModel):
    """Signal event from regime detection."""

    timestamp: datetime = Field(default_factory=datetime.now)
    regime: Regime = Field(description="Detected market regime")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level")
    target_weights: dict[str, float] = Field(description="Target allocation weights")
    requires_rebalance: bool = Field(
        default=False, description="Whether rebalancing is needed"
    )


class TradeRecord(BaseModel):
    """Record of an executed trade for logging."""

    trade_id: str = Field(default_factory=lambda: str(uuid4()))
    order_id: str = Field(description="Original order ID")
    session_id: str = Field(description="Session ID")
    symbol: str = Field(description="ETF symbol")
    action: OrderAction = Field(description="Buy or sell")
    quantity: Decimal = Field(description="Executed quantity")
    price: Decimal = Field(description="Execution price")
    total_value: Decimal = Field(description="Total trade value")
    transaction_cost: Decimal = Field(description="Transaction cost")
    slippage_cost: Decimal = Field(description="Slippage cost")
    realized_pnl: Decimal = Field(
        default=Decimal("0"), description="Realized P&L (for sells)"
    )
    executed_at: datetime = Field(default_factory=datetime.now)
