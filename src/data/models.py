"""Pydantic models for PEA Portfolio financial data.

This module defines all data models used throughout the portfolio system,
including ETF data, macro indicators, regime detection, positions, and trades.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ETFSymbol(str, Enum):
    """PEA-eligible ETF symbols traded on Euronext Paris."""

    LQQ = "LQQ.PA"  # Amundi Nasdaq-100 Daily (2x) Leveraged
    CL2 = "CL2.PA"  # Amundi MSCI USA Daily (2x) Leveraged
    WPEA = "WPEA.PA"  # Amundi MSCI World PEA


class ETFInfo(BaseModel):
    """ETF information including regulatory details.

    Attributes:
        symbol: ETF trading symbol
        isin: International Securities Identification Number
        name: Full ETF name
        leverage: Leverage factor (1 for unleveraged, 2 for 2x)
        ter: Total Expense Ratio (annual fee)
        pea_eligible: Whether eligible for French PEA account
        accumulating: Whether dividends are reinvested
    """

    symbol: ETFSymbol
    isin: str = Field(pattern=r"^[A-Z]{2}[A-Z0-9]{10}$")
    name: str
    leverage: int = Field(ge=1, le=3)
    ter: float = Field(ge=0.0, le=0.05, description="TER as decimal (0.006 = 0.6%)")
    pea_eligible: bool = True
    accumulating: bool = True


class DailyPrice(BaseModel):
    """Daily OHLCV price data for an ETF.

    Attributes:
        symbol: ETF symbol
        date: Trading date
        open: Opening price
        high: Highest price during the day
        low: Lowest price during the day
        close: Closing price
        volume: Number of shares traded
        adjusted_close: Price adjusted for splits and dividends
    """

    symbol: ETFSymbol
    date: date
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: int = Field(ge=0)
    adjusted_close: Decimal | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_price_consistency(self) -> "DailyPrice":
        """Validate that high >= low and prices are consistent."""
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low must be <= open and close")
        return self


class MacroIndicator(BaseModel):
    """Macroeconomic indicator from FRED or other sources.

    Attributes:
        indicator_name: Identifier for the indicator (e.g., 'VIX', 'DGS10')
        date: Observation date
        value: Indicator value
        source: Data source (default: 'FRED')
    """

    indicator_name: str
    date: date
    value: float
    source: str = "FRED"


class Regime(str, Enum):
    """Market regime classification for allocation decisions."""

    RISK_ON = "risk_on"  # Low VIX, positive trend, tight spreads
    NEUTRAL = "neutral"  # Mixed signals
    RISK_OFF = "risk_off"  # High VIX, negative trend, wide spreads


class AllocationRecommendation(BaseModel):
    """Portfolio allocation recommendation based on regime detection.

    Attributes:
        date: Recommendation date
        regime: Detected market regime
        lqq_weight: Target weight for LQQ (max 30% for leveraged)
        cl2_weight: Target weight for CL2 (max 30% for leveraged)
        wpea_weight: Target weight for WPEA
        cash_weight: Target cash weight (min 10%)
        confidence: Model confidence score
        reasoning: Explanation of the recommendation
    """

    date: date
    regime: Regime
    lqq_weight: float = Field(ge=0.0, le=0.30)
    cl2_weight: float = Field(ge=0.0, le=0.30)
    wpea_weight: float = Field(ge=0.0, le=1.0)
    cash_weight: float = Field(ge=0.10, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None

    @model_validator(mode="after")
    def validate_weights(self) -> "AllocationRecommendation":
        """Validate allocation weights sum to 1 and respect limits."""
        total = self.lqq_weight + self.cl2_weight + self.wpea_weight + self.cash_weight
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        leveraged_total = self.lqq_weight + self.cl2_weight
        if leveraged_total > 0.30:
            raise ValueError(
                f"Combined leveraged ETF weight ({leveraged_total}) exceeds 30% limit"
            )

        return self


class Position(BaseModel):
    """Current portfolio position.

    Attributes:
        symbol: ETF symbol
        shares: Number of shares held
        average_cost: Average cost per share
        current_price: Current market price per share
        market_value: Current market value (shares * current_price)
        unrealized_pnl: Unrealized profit/loss
        weight: Position weight in portfolio (0-1)
    """

    symbol: ETFSymbol
    shares: float = Field(ge=0.0)
    average_cost: float = Field(gt=0.0)
    current_price: float = Field(gt=0.0)
    market_value: float = Field(ge=0.0)
    unrealized_pnl: float
    weight: float = Field(ge=0.0, le=1.0)


class TradeAction(str, Enum):
    """Trade action type."""

    BUY = "BUY"
    SELL = "SELL"


class Trade(BaseModel):
    """Trade execution record for manual recording.

    Attributes:
        symbol: ETF symbol traded
        date: Execution datetime
        action: BUY or SELL
        shares: Number of shares
        price: Execution price per share
        total_value: Total trade value
        commission: Trading commission/fees
    """

    symbol: ETFSymbol
    date: datetime
    action: TradeAction
    shares: float = Field(gt=0.0)
    price: float = Field(gt=0.0)
    total_value: float
    commission: float = Field(ge=0.0, default=0.0)

    @model_validator(mode="after")
    def validate_total_value(self) -> "Trade":
        """Validate total_value matches shares * price."""
        expected = self.shares * self.price
        if abs(self.total_value - expected) > 0.01:
            raise ValueError(
                f"total_value ({self.total_value}) should equal "
                f"shares * price ({expected})"
            )
        return self


# Risk limits as constants (non-negotiable)
MAX_LEVERAGED_EXPOSURE = 0.30  # LQQ + CL2 <= 30%
MAX_SINGLE_POSITION = 0.25
MIN_CASH_BUFFER = 0.10
REBALANCE_THRESHOLD = 0.05
DRAWDOWN_ALERT = -0.20


# PEA-eligible ETF registry
PEA_ETFS = {
    ETFSymbol.LQQ: ETFInfo(
        symbol=ETFSymbol.LQQ,
        isin="FR0010342592",
        name="Amundi Nasdaq-100 Daily (2x) Leveraged UCITS ETF",
        leverage=2,
        ter=0.006,
        pea_eligible=True,
        accumulating=True,
    ),
    ETFSymbol.CL2: ETFInfo(
        symbol=ETFSymbol.CL2,
        isin="FR0010755611",
        name="Amundi MSCI USA Daily (2x) Leveraged UCITS ETF",
        leverage=2,
        ter=0.005,
        pea_eligible=True,
        accumulating=True,
    ),
    ETFSymbol.WPEA: ETFInfo(
        symbol=ETFSymbol.WPEA,
        isin="FR0011869353",
        name="Amundi MSCI World UCITS ETF",
        leverage=1,
        ter=0.0038,
        pea_eligible=True,
        accumulating=True,
    ),
}
