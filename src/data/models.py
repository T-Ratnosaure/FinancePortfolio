"""Pydantic models for PEA Portfolio financial data.

This module defines all data models used throughout the portfolio system,
including ETF data, macro indicators, regime detection, positions, and trades.
"""

from datetime import date, datetime, timedelta
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
    open: Decimal = Field(gt=Decimal("0"))
    high: Decimal = Field(gt=Decimal("0"))
    low: Decimal = Field(gt=Decimal("0"))
    close: Decimal = Field(gt=Decimal("0"))
    volume: int = Field(ge=0)
    adjusted_close: Decimal | None = Field(default=None, gt=Decimal("0"))

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


class CashPosition(BaseModel):
    """Cash position in the portfolio.

    Attributes:
        amount: Cash balance in euros
        currency: Currency code (default: EUR)
        updated_at: Last update timestamp
    """

    amount: Decimal = Field(ge=Decimal("0"))
    currency: str = "EUR"
    updated_at: datetime | None = None


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics over a time period.

    Attributes:
        start_date: Period start date
        end_date: Period end date
        start_value: Portfolio value at start
        end_value: Portfolio value at end
        total_return: Total return as decimal (0.10 = 10%)
        annualized_return: Annualized return as decimal
        volatility: Annualized volatility (standard deviation of returns)
        sharpe_ratio: Sharpe ratio (assuming risk-free rate of 0)
        max_drawdown: Maximum drawdown as decimal (negative value)
        num_trades: Number of trades in period
    """

    start_date: date
    end_date: date
    start_value: Decimal = Field(ge=Decimal("0"))
    end_value: Decimal = Field(ge=Decimal("0"))
    total_return: float
    annualized_return: float | None = None
    volatility: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    num_trades: int = Field(ge=0, default=0)


class DiscrepancyType(str, Enum):
    """Type of discrepancy between database and broker."""

    MISSING_IN_DB = "missing_in_db"  # Position exists at broker but not in DB
    MISSING_AT_BROKER = "missing_at_broker"  # Position in DB but not at broker
    SHARES_MISMATCH = "shares_mismatch"  # Share count differs
    PRICE_MISMATCH = "price_mismatch"  # Price differs significantly


class Discrepancy(BaseModel):
    """Discrepancy between database and broker positions.

    Attributes:
        symbol: ETF symbol with discrepancy
        discrepancy_type: Type of discrepancy found
        db_value: Value in database (shares or price)
        broker_value: Value at broker (shares or price)
        difference: Absolute difference
        description: Human-readable description
    """

    symbol: str
    discrepancy_type: DiscrepancyType
    db_value: float | None = None
    broker_value: float | None = None
    difference: float | None = None
    description: str


# Risk limits as constants (non-negotiable)
MAX_LEVERAGED_EXPOSURE = 0.30  # LQQ + CL2 <= 30%
MAX_SINGLE_POSITION = 0.25
MIN_CASH_BUFFER = 0.10
REBALANCE_THRESHOLD = 0.05
DRAWDOWN_ALERT = -0.20


class DataCategory(str, Enum):
    """Category of data for freshness tracking."""

    PRICE_DATA = "price_data"  # ETF prices
    MACRO_DATA = "macro_data"  # Macroeconomic indicators
    PORTFOLIO_DATA = "portfolio_data"  # Portfolio positions
    TRADE_DATA = "trade_data"  # Trade records


class FreshnessStatus(str, Enum):
    """Data freshness status."""

    FRESH = "fresh"  # Within acceptable staleness threshold
    STALE = "stale"  # Beyond staleness threshold but usable with warning
    CRITICAL = "critical"  # Too old to use safely


# Staleness thresholds by data category
STALENESS_THRESHOLDS = {
    DataCategory.PRICE_DATA: timedelta(days=1),  # Daily prices should be < 1 day old
    DataCategory.MACRO_DATA: timedelta(days=7),  # Macro data can be < 1 week old
    DataCategory.PORTFOLIO_DATA: timedelta(hours=1),  # Positions should be < 1 hour
    DataCategory.TRADE_DATA: timedelta(days=365 * 100),  # Historical, no threshold
}

# Critical thresholds (data is too old to use)
CRITICAL_THRESHOLDS = {
    DataCategory.PRICE_DATA: timedelta(days=7),  # Price data > 1 week is critical
    DataCategory.MACRO_DATA: timedelta(days=30),  # Macro data > 1 month is critical
    DataCategory.PORTFOLIO_DATA: timedelta(hours=24),  # Positions > 1 day critical
    DataCategory.TRADE_DATA: timedelta(days=365 * 100),  # No critical threshold
}


class DataFreshness(BaseModel):
    """Metadata about data freshness for staleness detection.

    Attributes:
        data_category: Category of data (price, macro, portfolio, trade)
        symbol: Optional symbol for symbol-specific data
        indicator_name: Optional indicator name for macro data
        last_updated: Timestamp when data was last fetched/updated
        record_count: Number of records in this dataset
        source: Data source (e.g., 'Yahoo Finance', 'FRED')
    """

    data_category: DataCategory
    symbol: str | None = None
    indicator_name: str | None = None
    last_updated: datetime
    record_count: int = Field(ge=0)
    source: str

    def get_age(self) -> timedelta:
        """Calculate age of the data.

        Returns:
            Time elapsed since last update
        """
        return datetime.now() - self.last_updated

    def get_status(self) -> FreshnessStatus:
        """Determine freshness status of the data.

        Returns:
            FreshnessStatus enum value
        """
        age = self.get_age()
        staleness_threshold = STALENESS_THRESHOLDS[self.data_category]
        critical_threshold = CRITICAL_THRESHOLDS[self.data_category]

        if age > critical_threshold:
            return FreshnessStatus.CRITICAL
        if age > staleness_threshold:
            return FreshnessStatus.STALE
        return FreshnessStatus.FRESH

    def is_stale(self) -> bool:
        """Check if data is stale (beyond normal threshold).

        Returns:
            True if data is stale or critical
        """
        return self.get_status() in (FreshnessStatus.STALE, FreshnessStatus.CRITICAL)

    def is_critical(self) -> bool:
        """Check if data staleness is critical (too old to use safely).

        Returns:
            True if data is critically stale
        """
        return self.get_status() == FreshnessStatus.CRITICAL

    def get_warning_message(self) -> str | None:
        """Generate warning message if data is stale.

        Returns:
            Warning message if stale, None if fresh
        """
        status = self.get_status()
        if status == FreshnessStatus.FRESH:
            return None

        age = self.get_age()
        age_str = self._format_age(age)

        identifier = ""
        if self.symbol:
            identifier = f" for {self.symbol}"
        elif self.indicator_name:
            identifier = f" for {self.indicator_name}"

        if status == FreshnessStatus.CRITICAL:
            return (
                f"CRITICAL: {self.data_category.value}{identifier} "
                f"is {age_str} old (last updated: {self.last_updated}). "
                f"Data may be too stale for reliable decisions."
            )
        return (
            f"WARNING: {self.data_category.value}{identifier} is {age_str} old "
            f"(last updated: {self.last_updated}). "
            f"Consider refreshing data."
        )

    def _format_age(self, age: timedelta) -> str:
        """Format age as human-readable string.

        Args:
            age: Time delta to format

        Returns:
            Human-readable age string
        """
        total_seconds = int(age.total_seconds())
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60

        if days > 0:
            return f"{days} day{'s' if days != 1 else ''}"
        if hours > 0:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        return f"{minutes} minute{'s' if minutes != 1 else ''}"


class StaleDataError(Exception):
    """Raised when attempting to use critically stale data."""

    def __init__(self, freshness: DataFreshness) -> None:
        """Initialize stale data error.

        Args:
            freshness: DataFreshness object with stale data info
        """
        self.freshness = freshness
        message = freshness.get_warning_message() or "Data is too stale for safe usage"
        super().__init__(message)


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
