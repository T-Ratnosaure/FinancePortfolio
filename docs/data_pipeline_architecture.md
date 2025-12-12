# Data Pipeline Architecture for French PEA Portfolio System

## Executive Summary

This document provides a comprehensive data pipeline architecture for tracking a French PEA portfolio containing LQQ, CL2, and WPEA ETFs. The design prioritizes cost-effectiveness, reliability, and data quality for a retail investor while adhering to the FinancePortfolio project's technical standards.

## 1. ETF Price Data Pipeline

### 1.1 Data Source Selection

**Recommended Primary Source: Yahoo Finance (yfinance)**

**Rationale:**
- **Cost:** FREE - critical for retail investors
- **Reliability:** 99%+ uptime, backed by Yahoo/Verizon infrastructure
- **Coverage:** Excellent for European ETFs traded on Euronext
- **Python SDK:** `yfinance` library - mature, well-maintained, actively developed
- **Rate Limits:** Generous for retail use (no hard limits for reasonable usage)
- **Data Quality:** Accurate OHLCV data with 15-20 minute delay
- **Licensing:** Free for personal use

**ETF Ticker Mappings:**
- LQQ (Leverage Shares 2x Long NASDAQ 100): `LQQ.PA` (Euronext Paris)
- CL2 (Amundi Nasdaq-100 2x Leveraged Daily): `CL2.PA` (Euronext Paris)
- WPEA (Amundi MSCI World UCITS ETF): `WPEA.PA` (Euronext Paris)

**Backup Source: Alpha Vantage**

**Rationale:**
- **Cost:** FREE tier (25 API calls/day)
- **Use Case:** Fallback when yfinance fails
- **Coverage:** Global markets including European exchanges
- **Limitations:** Low rate limit requires careful scheduling

**Data Not Recommended:**
- **Polygon.io:** $29/month minimum (too expensive for retail)
- **IEX Cloud:** Limited European coverage
- **Quandl/Nasdaq Data Link:** Reduced free tier, premium for real-time

### 1.2 Data Refresh Frequency

**Recommended Schedule:**

```
Market Hours (09:00-17:30 CET):
  - Primary: Every 30 minutes (reduces API load while maintaining timeliness)
  - Use case: Intraday monitoring during active trading

End of Day (18:00 CET):
  - Primary: Full daily snapshot with adjusted close
  - Use case: Official daily portfolio valuation

Pre-Market (08:30 CET):
  - Check: Retrieve previous day's close if missed
  - Use case: Ensure data consistency before market open
```

**Rationale:**
- PEA accounts are long-term investment vehicles (manual execution via Boursobank)
- Real-time data not necessary for retail strategy
- 30-minute intervals balance timeliness with API courtesy
- End-of-day data is authoritative for portfolio tracking

### 1.3 Data Format and Schema

**Pydantic Data Models:**

```python
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class ETFTicker(BaseModel):
    """ETF identifier and metadata."""

    symbol: Literal["LQQ.PA", "CL2.PA", "WPEA.PA"]
    name: str
    isin: str
    currency: Literal["EUR"]
    exchange: Literal["XPAR"]  # Euronext Paris MIC code

    class Config:
        frozen = True  # Immutable


class OHLCVData(BaseModel):
    """Open, High, Low, Close, Volume data point."""

    ticker: str = Field(..., pattern=r"^[A-Z0-9]+\.PA$")
    timestamp: datetime
    open: Decimal = Field(..., gt=0, decimal_places=4)
    high: Decimal = Field(..., gt=0, decimal_places=4)
    low: Decimal = Field(..., gt=0, decimal_places=4)
    close: Decimal = Field(..., gt=0, decimal_places=4)
    volume: int = Field(..., ge=0)
    source: Literal["yahoo_finance", "alpha_vantage"]
    fetch_timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: Decimal, info) -> Decimal:
        """Validate that high >= low."""
        if "low" in info.data and v < info.data["low"]:
            raise ValueError("High must be >= Low")
        return v

    @field_validator("close")
    @classmethod
    def close_in_range(cls, v: Decimal, info) -> Decimal:
        """Validate that close is within [low, high]."""
        if "low" in info.data and "high" in info.data:
            if not (info.data["low"] <= v <= info.data["high"]):
                raise ValueError("Close must be within [Low, High] range")
        return v

    class Config:
        frozen = True


class DailyPriceSnapshot(BaseModel):
    """Daily aggregated price data for portfolio valuation."""

    ticker: str
    date: datetime
    adjusted_close: Decimal = Field(..., gt=0, decimal_places=4)
    volume: int
    is_complete: bool  # True if full trading day
    source: str

    class Config:
        frozen = True
```

## 2. Market Indicator Pipelines

### 2.1 VIX (Volatility Index)

**Data Source:** Yahoo Finance (`^VIX`)

**Refresh Frequency:**
- Market hours: Every 1 hour
- End of day: Daily snapshot

**Use Case:**
- Portfolio risk assessment
- Market sentiment indicator
- Trigger for defensive positioning

**Pydantic Model:**

```python
class VIXData(BaseModel):
    """VIX volatility index data."""

    timestamp: datetime
    value: Decimal = Field(..., ge=0, le=100, decimal_places=2)
    source: str = "yahoo_finance"

    @property
    def volatility_regime(self) -> str:
        """Classify market volatility regime."""
        if self.value < Decimal("12"):
            return "very_low"
        elif self.value < Decimal("20"):
            return "low"
        elif self.value < Decimal("30"):
            return "elevated"
        else:
            return "high"
```

### 2.2 Market Indices

**Indices to Track:**

| Index | Ticker | Rationale | Frequency |
|-------|--------|-----------|-----------|
| NASDAQ-100 | `^NDX` | Direct correlation to LQQ/CL2 | 30 min |
| MSCI World | `URTH` (iShares proxy) | Correlation to WPEA | Daily |
| CAC 40 | `^FCHI` | French market benchmark | Daily |
| S&P 500 | `^GSPC` | Global market sentiment | Daily |

**Pydantic Model:**

```python
class IndexData(BaseModel):
    """Market index data point."""

    index_symbol: str
    timestamp: datetime
    value: Decimal = Field(..., gt=0, decimal_places=2)
    change_percent: Decimal | None = Field(None, decimal_places=4)
    source: str
```

### 2.3 Interest Rates

**Rates to Track:**

| Rate | Source | Rationale | Frequency |
|------|--------|-----------|-----------|
| ECB Deposit Rate | ECB Statistical Data Warehouse | Cost of capital (EUR) | Weekly |
| EURIBOR 3M | Yahoo Finance (`EURIBOR3M.FX`) | Money market benchmark | Daily |
| US 10Y Treasury | Yahoo Finance (`^TNX`) | Global risk-free rate | Daily |

**Pydantic Model:**

```python
class InterestRateData(BaseModel):
    """Interest rate data point."""

    rate_name: str
    date: datetime
    rate: Decimal = Field(..., ge=-2, le=20, decimal_places=4)  # % per annum
    currency: Literal["EUR", "USD"]
    source: str

    class Config:
        frozen = True
```

## 3. Data Validation and Quality Checks

### 3.1 Validation Layers

**Layer 1: Schema Validation (Pydantic)**
- Automatic via Pydantic models
- Type checking, range validation, business rules
- Fails fast on malformed data

**Layer 2: Business Logic Validation**

```python
from typing import Protocol


class PriceValidator(Protocol):
    """Protocol for price validation strategies."""

    def validate(self, current: OHLCVData, previous: OHLCVData | None) -> bool:
        """Validate price data point."""
        ...


class CircuitBreakerValidator:
    """Detect circuit breaker events (extreme price moves)."""

    def __init__(self, threshold_percent: Decimal = Decimal("15.0")):
        self.threshold = threshold_percent

    def validate(self, current: OHLCVData, previous: OHLCVData | None) -> bool:
        """Check for circuit breaker conditions."""
        if previous is None:
            return True

        change_pct = abs(
            (current.close - previous.close) / previous.close * 100
        )

        if change_pct > self.threshold:
            # Log warning but don't reject (could be real event)
            return False  # Requires manual review

        return True


class StalenessValidator:
    """Detect stale data (trading halts, API issues)."""

    def __init__(self, max_age_hours: int = 24):
        self.max_age = max_age_hours

    def validate(self, current: OHLCVData, previous: OHLCVData | None) -> bool:
        """Check data freshness."""
        age_hours = (
            datetime.utcnow() - current.fetch_timestamp
        ).total_seconds() / 3600

        return age_hours <= self.max_age


class VolumeAnomalyValidator:
    """Detect abnormal trading volume."""

    def __init__(self, z_score_threshold: float = 3.0):
        self.threshold = z_score_threshold

    def validate(self, current: OHLCVData, historical_mean: float,
                 historical_std: float) -> bool:
        """Check for volume anomalies using z-score."""
        if historical_std == 0:
            return True

        z_score = abs((current.volume - historical_mean) / historical_std)
        return z_score <= self.threshold
```

### 3.2 Data Quality Metrics

**Tracking Metrics:**

```python
from enum import Enum


class DataQualityMetric(BaseModel):
    """Data quality metric for monitoring."""

    ticker: str
    date: datetime
    metric_type: Literal[
        "completeness",
        "freshness",
        "accuracy",
        "consistency"
    ]
    score: Decimal = Field(..., ge=0, le=100, decimal_places=2)
    issues: list[str] = Field(default_factory=list)


class DataQualityStatus(str, Enum):
    """Overall data quality status."""

    EXCELLENT = "excellent"  # 95-100% score
    GOOD = "good"            # 80-95% score
    DEGRADED = "degraded"    # 60-80% score
    CRITICAL = "critical"    # <60% score
```

### 3.3 Handling Missing Data

**Strategy:**

1. **Real-time gaps (< 1 day):**
   - Retry with exponential backoff (3 attempts)
   - Switch to backup source (Alpha Vantage)
   - Forward-fill from last valid point (with staleness warning)

2. **Historical gaps:**
   - Backfill from backup source
   - Mark as imputed in metadata
   - Log gap for audit trail

3. **Extended outages (> 1 day):**
   - Alert user (email/notification)
   - Use last known good value with warning flag
   - Do NOT auto-trade on stale data

**Implementation:**

```python
from typing import Callable
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class DataFetchError(Exception):
    """Custom exception for data fetch failures."""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(DataFetchError),
)
def fetch_with_retry(
    fetcher: Callable[[], OHLCVData],
    ticker: str,
) -> OHLCVData:
    """Fetch data with exponential backoff retry."""
    try:
        return fetcher()
    except Exception as e:
        raise DataFetchError(f"Failed to fetch {ticker}: {e}") from e
```

## 4. Storage Architecture

### 4.1 Recommended Storage: SQLite + Parquet Hybrid

**Rationale for Retail Investor:**
- **Cost:** FREE (no cloud costs)
- **Complexity:** Minimal (no server management)
- **Performance:** Excellent for < 10M rows
- **Portability:** Single file, easy backup
- **Query Performance:** Sufficient for portfolio analytics

**Architecture:**

```
C:\Users\larai\FinancePortfolio\data\
├── portfolio.db                 # SQLite database (hot data, last 90 days)
├── historical/                  # Parquet files (cold data, > 90 days)
│   ├── etf_prices_2024.parquet
│   ├── etf_prices_2025.parquet
│   ├── market_indicators_2024.parquet
│   └── market_indicators_2025.parquet
└── cache/                       # Temporary cache for API responses
    └── daily_fetch_cache.json
```

**SQLite Schema:**

```sql
-- ETF price data (hot storage)
CREATE TABLE etf_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    source TEXT NOT NULL,
    fetch_timestamp DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, timestamp)
);

CREATE INDEX idx_etf_prices_ticker_timestamp
ON etf_prices(ticker, timestamp DESC);

-- Market indicators
CREATE TABLE market_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_type TEXT NOT NULL,  -- 'vix', 'index', 'rate'
    indicator_symbol TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    value REAL NOT NULL,
    metadata JSON,
    source TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(indicator_type, indicator_symbol, timestamp)
);

-- Data quality log
CREATE TABLE data_quality_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    check_timestamp DATETIME NOT NULL,
    check_type TEXT NOT NULL,
    status TEXT NOT NULL,  -- 'pass', 'fail', 'warning'
    details JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline run metadata
CREATE TABLE pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    pipeline_name TEXT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    status TEXT NOT NULL,  -- 'running', 'success', 'failed'
    records_processed INTEGER,
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Parquet Storage (Cold Data):**

```python
import polars as pl
from pathlib import Path


def archive_to_parquet(
    cutoff_date: datetime,
    db_path: Path,
    archive_dir: Path,
) -> None:
    """Move old data from SQLite to Parquet for compression."""
    # Read old data
    old_data = pl.read_database(
        query=f"""
        SELECT * FROM etf_prices
        WHERE timestamp < '{cutoff_date.isoformat()}'
        """,
        connection=f"sqlite:///{db_path}",
    )

    # Write to Parquet with compression
    year = cutoff_date.year
    parquet_path = archive_dir / f"etf_prices_{year}.parquet"

    old_data.write_parquet(
        parquet_path,
        compression="zstd",
        compression_level=9,
    )

    # Delete from SQLite after successful archive
    # (with transaction safety)
```

### 4.2 Alternative: Cloud Storage (Optional Upgrade)

**If scaling beyond local:**

- **PostgreSQL on Supabase:** FREE tier (500MB, sufficient for years of data)
- **TimescaleDB:** Time-series optimized, FREE self-hosted
- **AWS S3 + Athena:** Query Parquet files directly, pay-per-query

**Not Recommended:**
- Cloud databases with minimum costs (RDS, Azure SQL)
- Real-time databases (Firebase, DynamoDB) - unnecessary latency premium

## 5. Refresh Scheduling

### 5.1 Scheduler Architecture

**Recommended: APScheduler (Advanced Python Scheduler)**

**Rationale:**
- Lightweight, no external dependencies
- Persistent job store (SQLite-backed)
- Timezone-aware (critical for CET market hours)
- Graceful error handling

**Installation:**

```bash
uv add apscheduler
```

**Implementation:**

```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from pytz import timezone


# Paris timezone for Euronext hours
PARIS_TZ = timezone("Europe/Paris")

# Configure scheduler
jobstores = {
    "default": SQLAlchemyJobStore(
        url="sqlite:///data/scheduler.db"
    )
}

scheduler = BackgroundScheduler(
    jobstores=jobstores,
    timezone=PARIS_TZ,
)


# Intraday ETF price fetch (every 30 min during market hours)
scheduler.add_job(
    func=fetch_etf_prices_intraday,
    trigger=CronTrigger(
        day_of_week="mon-fri",
        hour="9-17",
        minute="*/30",
        timezone=PARIS_TZ,
    ),
    id="etf_prices_intraday",
    replace_existing=True,
    max_instances=1,
)

# End-of-day snapshot (authoritative daily close)
scheduler.add_job(
    func=fetch_etf_prices_eod,
    trigger=CronTrigger(
        day_of_week="mon-fri",
        hour=18,
        minute=0,
        timezone=PARIS_TZ,
    ),
    id="etf_prices_eod",
    replace_existing=True,
    max_instances=1,
)

# VIX hourly (market hours)
scheduler.add_job(
    func=fetch_vix_data,
    trigger=CronTrigger(
        day_of_week="mon-fri",
        hour="9-17",
        minute=0,
        timezone=PARIS_TZ,
    ),
    id="vix_hourly",
    replace_existing=True,
)

# Market indices daily (after US close)
scheduler.add_job(
    func=fetch_market_indices,
    trigger=CronTrigger(
        day_of_week="mon-fri",
        hour=22,  # 22:00 CET = after US market close
        minute=30,
        timezone=PARIS_TZ,
    ),
    id="indices_daily",
    replace_existing=True,
)

# Interest rates weekly (ECB publishes weekly)
scheduler.add_job(
    func=fetch_interest_rates,
    trigger=CronTrigger(
        day_of_week="wed",  # ECB publishes Wednesdays
        hour=15,
        minute=0,
        timezone=PARIS_TZ,
    ),
    id="rates_weekly",
    replace_existing=True,
)

# Data quality check daily
scheduler.add_job(
    func=run_data_quality_checks,
    trigger=CronTrigger(
        day_of_week="mon-fri",
        hour=19,
        minute=0,
        timezone=PARIS_TZ,
    ),
    id="quality_checks_daily",
    replace_existing=True,
)

# Archive old data monthly
scheduler.add_job(
    func=archive_old_data_to_parquet,
    trigger=CronTrigger(
        day=1,  # First day of month
        hour=3,
        minute=0,
        timezone=PARIS_TZ,
    ),
    id="archive_monthly",
    replace_existing=True,
)

scheduler.start()
```

### 5.2 Holiday Calendar

**Critical:** Euronext Paris follows French market holidays.

```python
from datetime import date
import pandas_market_calendars as mcal


def is_trading_day(check_date: date) -> bool:
    """Check if Euronext Paris is open."""
    xpar = mcal.get_calendar("XPAR")
    schedule = xpar.schedule(
        start_date=check_date,
        end_date=check_date,
    )
    return len(schedule) > 0


# Skip scheduled jobs on non-trading days
def fetch_etf_prices_intraday() -> None:
    """Fetch ETF prices with trading day check."""
    if not is_trading_day(date.today()):
        return  # Skip on holidays

    # ... fetch logic
```

**Installation:**

```bash
uv add pandas-market-calendars
```

## 6. Data Contracts for Each Dataset

### 6.1 Contract Overview

**Purpose:**
- Define explicit schemas for all datasets
- Enable version control of data structures
- Support schema evolution
- Enable contract testing

**Implementation: Pydantic Models as Contracts**

All data contracts defined in `C:\Users\larai\FinancePortfolio\src\data\contracts\`

```python
# src/data/contracts/__init__.py
"""Data contracts for FinancePortfolio."""

from .etf_contracts import ETFTicker, OHLCVData, DailyPriceSnapshot
from .indicator_contracts import VIXData, IndexData, InterestRateData
from .quality_contracts import DataQualityMetric, DataQualityStatus

__all__ = [
    "ETFTicker",
    "OHLCVData",
    "DailyPriceSnapshot",
    "VIXData",
    "IndexData",
    "InterestRateData",
    "DataQualityMetric",
    "DataQualityStatus",
]
```

### 6.2 Contract Versioning

```python
# src/data/contracts/etf_contracts.py
"""ETF data contracts - Version 1.0"""

CONTRACT_VERSION = "1.0.0"


class OHLCVData(BaseModel):
    """OHLCV data contract v1.0."""

    _contract_version: str = CONTRACT_VERSION

    # ... fields
```

### 6.3 Contract Testing

```python
# tests/contracts/test_etf_contracts.py
"""Contract tests for ETF data models."""

import pytest
from decimal import Decimal
from datetime import datetime
from src.data.contracts import OHLCVData


def test_ohlcv_valid_data():
    """Test OHLCV contract accepts valid data."""
    data = OHLCVData(
        ticker="LQQ.PA",
        timestamp=datetime(2025, 12, 10, 15, 30),
        open=Decimal("150.25"),
        high=Decimal("152.00"),
        low=Decimal("149.50"),
        close=Decimal("151.75"),
        volume=100000,
        source="yahoo_finance",
    )
    assert data.ticker == "LQQ.PA"


def test_ohlcv_rejects_high_less_than_low():
    """Test OHLCV contract rejects high < low."""
    with pytest.raises(ValueError):
        OHLCVData(
            ticker="LQQ.PA",
            timestamp=datetime.now(),
            open=Decimal("150.00"),
            high=Decimal("148.00"),  # Invalid: high < low
            low=Decimal("149.00"),
            close=Decimal("149.50"),
            volume=100000,
            source="yahoo_finance",
        )
```

## 7. Error Handling and Alerting

### 7.1 Error Taxonomy

**Error Categories:**

```python
from enum import Enum


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    DEBUG = "debug"          # Informational, no action needed
    INFO = "info"            # Expected events (e.g., market closed)
    WARNING = "warning"      # Degraded but operational
    ERROR = "error"          # Failed operation, retry possible
    CRITICAL = "critical"    # System failure, immediate action required


class DataError(Exception):
    """Base class for data pipeline errors."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity,
        context: dict | None = None,
    ):
        self.message = message
        self.severity = severity
        self.context = context or {}
        super().__init__(message)


class DataFetchError(DataError):
    """Error fetching data from external source."""
    pass


class DataValidationError(DataError):
    """Error validating data quality."""
    pass


class DataStorageError(DataError):
    """Error storing data."""
    pass
```

### 7.2 Circuit Breaker Pattern

**Prevent cascading failures:**

```python
from datetime import datetime, timedelta
from typing import Callable, TypeVar


T = TypeVar("T")


class CircuitBreaker:
    """Circuit breaker for API calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state: Literal["closed", "open", "half_open"] = "closed"

    def call(self, func: Callable[[], T]) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func()
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return False
        return (
            datetime.now() - self.last_failure_time
        ).total_seconds() >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# Usage
yahoo_finance_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=300,  # 5 minutes
    expected_exception=DataFetchError,
)
```

### 7.3 Alerting Strategy

**For Retail Investor (Cost-Effective):**

**Recommended Alerting Channels:**

1. **Email (Primary):** FREE via Gmail SMTP
2. **Local Logging:** Structured logs for debugging
3. **Windows Notifications:** Local desktop alerts (optional)

**Alert Types:**

| Severity | Alert Method | Frequency | Example |
|----------|-------------|-----------|---------|
| CRITICAL | Email + Log | Immediate | Data fetch failed for 24+ hours |
| ERROR | Email | Batched (hourly) | Single fetch retry exhausted |
| WARNING | Log only | Real-time | Stale data warning (< 24h) |
| INFO | Log only | Real-time | Scheduled job completed |

**Email Alerting Implementation:**

```python
import smtplib
from email.message import EmailMessage
from typing import Protocol


class AlertChannel(Protocol):
    """Protocol for alert channels."""

    def send_alert(
        self,
        severity: ErrorSeverity,
        message: str,
        context: dict,
    ) -> None:
        """Send alert via this channel."""
        ...


class EmailAlerter:
    """Send alerts via email."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender: str,
        password: str,
        recipient: str,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender = sender
        self.password = password
        self.recipient = recipient

    def send_alert(
        self,
        severity: ErrorSeverity,
        message: str,
        context: dict,
    ) -> None:
        """Send email alert."""
        if severity not in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
            return  # Only email for ERROR and above

        msg = EmailMessage()
        msg["Subject"] = f"[{severity.value.upper()}] FinancePortfolio Alert"
        msg["From"] = self.sender
        msg["To"] = self.recipient

        body = f"""
        Severity: {severity.value}
        Message: {message}

        Context:
        {self._format_context(context)}

        Timestamp: {datetime.now().isoformat()}
        """
        msg.set_content(body)

        with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port) as smtp:
            smtp.login(self.sender, self.password)
            smtp.send_message(msg)

    def _format_context(self, context: dict) -> str:
        """Format context dictionary for email."""
        return "\n".join(f"{k}: {v}" for k, v in context.items())


# Configuration (store in environment variables)
email_alerter = EmailAlerter(
    smtp_host="smtp.gmail.com",
    smtp_port=465,
    sender="your-email@gmail.com",
    password="your-app-password",  # Use app-specific password
    recipient="your-email@gmail.com",
)
```

**Structured Logging:**

```python
import logging
from pythonjsonlogger import jsonlogger


# Configure JSON logging
logger = logging.getLogger("financeportfolio")
handler = logging.FileHandler("data/logs/pipeline.jsonl")
formatter = jsonlogger.JsonFormatter(
    "%(timestamp)s %(name)s %(levelname)s %(message)s %(context)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_data_event(
    event_type: str,
    severity: ErrorSeverity,
    message: str,
    context: dict,
) -> None:
    """Log structured data pipeline event."""
    extra = {
        "event_type": event_type,
        "context": context,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if severity == ErrorSeverity.CRITICAL:
        logger.critical(message, extra=extra)
    elif severity == ErrorSeverity.ERROR:
        logger.error(message, extra=extra)
    elif severity == ErrorSeverity.WARNING:
        logger.warning(message, extra=extra)
    else:
        logger.info(message, extra=extra)
```

**Installation:**

```bash
uv add python-json-logger
```

## 8. Cost-Effective Solutions Summary

### 8.1 Total Cost Breakdown

| Component | Solution | Monthly Cost |
|-----------|----------|--------------|
| ETF Price Data | Yahoo Finance (yfinance) | FREE |
| Market Indicators | Yahoo Finance | FREE |
| Backup Data Source | Alpha Vantage (free tier) | FREE |
| Storage | SQLite + Parquet (local) | FREE |
| Scheduling | APScheduler | FREE |
| Alerting | Gmail SMTP | FREE |
| Hosting | Local machine (Windows) | FREE |
| **TOTAL** | | **€0/month** |

### 8.2 Scaling Costs (Optional Upgrades)

**If you need more:**

| Upgrade | Cost | When Needed |
|---------|------|-------------|
| Alpha Vantage Premium | $49/month | Real-time data requirement |
| Supabase PostgreSQL | FREE → $25/month | Multi-device sync |
| Cloud VM (AWS t3.micro) | ~$8/month | 24/7 uptime requirement |
| Polygon.io Basic | $29/month | Options data, more granular |

**Recommendation:** Start with FREE tier, upgrade only when specific limitations hit.

### 8.3 Data Volume Estimates

**Year 1 Projections:**

```
ETF Prices (3 ETFs):
  - Intraday (30 min intervals): 3 ETFs × 8 hours × 2 samples/hour × 250 days = 12,000 rows
  - Daily snapshots: 3 ETFs × 250 days = 750 rows
  - Total: ~12,750 rows/year

Market Indicators:
  - VIX: 8 hours × 250 days = 2,000 rows/year
  - Indices (4 indices × 250 days): 1,000 rows/year
  - Rates (3 rates × 52 weeks): 156 rows/year
  - Total: ~3,156 rows/year

Total data: ~15,906 rows/year
SQLite capacity: 281 TB (theoretical), 10M+ rows (practical)

Storage size: ~5 MB/year (uncompressed), ~1 MB/year (Parquet compressed)
10-year projection: ~10 MB total
```

**Conclusion:** Local SQLite + Parquet is more than sufficient for decades of data.

## 9. Implementation Roadmap

### Phase 1: Core Pipeline (Week 1-2)
- [ ] Install dependencies (yfinance, apscheduler, polars)
- [ ] Implement Pydantic data contracts
- [ ] Build Yahoo Finance fetcher with retry logic
- [ ] Create SQLite schema and database
- [ ] Implement basic ETF price pipeline
- [ ] Add structured logging

### Phase 2: Validation & Quality (Week 3)
- [ ] Implement validation layers
- [ ] Add circuit breaker pattern
- [ ] Create data quality checks
- [ ] Build Alpha Vantage backup fetcher
- [ ] Add email alerting

### Phase 3: Market Indicators (Week 4)
- [ ] Add VIX data pipeline
- [ ] Add market indices pipeline
- [ ] Add interest rates pipeline
- [ ] Implement indicator-specific validation

### Phase 4: Scheduling & Automation (Week 5)
- [ ] Configure APScheduler jobs
- [ ] Add holiday calendar integration
- [ ] Implement Parquet archiving
- [ ] Create data quality dashboard (optional)

### Phase 5: Testing & Documentation (Week 6)
- [ ] Write contract tests
- [ ] Write integration tests
- [ ] Add pipeline monitoring
- [ ] Document operational procedures

## 10. Python/UV Tooling Specifics

### 10.1 Required Dependencies

```bash
# Core data fetching
uv add yfinance
uv add httpx  # Already in dependencies

# Data validation and processing
# pydantic already in dependencies
uv add polars  # Fast DataFrame library for Parquet

# Scheduling
uv add apscheduler
uv add pandas-market-calendars

# Alerting and logging
uv add python-json-logger

# Backup data source
uv add alpha-vantage

# Async support (already in dev dependencies)
# pytest-asyncio (use anyio instead per project standards)

# Optional: CLI for manual operations
uv add typer
```

### 10.2 Project Structure

```
C:\Users\larai\FinancePortfolio\
├── src\
│   ├── data\
│   │   ├── contracts\
│   │   │   ├── __init__.py
│   │   │   ├── etf_contracts.py
│   │   │   ├── indicator_contracts.py
│   │   │   └── quality_contracts.py
│   │   ├── fetchers\
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── yahoo_finance.py
│   │   │   └── alpha_vantage.py
│   │   ├── validators\
│   │   │   ├── __init__.py
│   │   │   ├── price_validators.py
│   │   │   └── quality_checks.py
│   │   ├── storage\
│   │   │   ├── __init__.py
│   │   │   ├── sqlite_store.py
│   │   │   └── parquet_archive.py
│   │   ├── pipeline\
│   │   │   ├── __init__.py
│   │   │   ├── etf_pipeline.py
│   │   │   ├── indicator_pipeline.py
│   │   │   └── scheduler.py
│   │   └── alerts\
│   │       ├── __init__.py
│   │       ├── email_alerter.py
│   │       └── logger.py
├── tests\
│   ├── contracts\
│   │   └── test_etf_contracts.py
│   ├── fetchers\
│   │   └── test_yahoo_finance.py
│   └── validators\
│       └── test_price_validators.py
├── data\
│   ├── portfolio.db
│   ├── scheduler.db
│   ├── logs\
│   │   └── pipeline.jsonl
│   ├── historical\
│   │   └── *.parquet
│   └── cache\
├── config\
│   ├── config.yaml
│   └── .env.example
├── docs\
│   ├── data_pipeline_architecture.md  # This document
│   └── operational_procedures.md
├── main.py
├── pyproject.toml
└── README.md
```

### 10.3 Type Checking with Pyrefly

```bash
# Initialize Pyrefly
uv run pyrefly init

# Run type checks after each change
uv run pyrefly check
```

### 10.4 Code Quality Commands

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check . --fix

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=html
```

### 10.5 Running the Pipeline

```bash
# One-time backfill
uv run python -m src.data.pipeline.backfill --start-date 2024-01-01

# Start scheduled pipeline
uv run python -m src.data.pipeline.scheduler

# Manual fetch (CLI)
uv run python -m src.data.cli fetch-etf LQQ.PA --date 2025-12-10

# Data quality report
uv run python -m src.data.cli quality-report --days 7
```

## 11. Next Steps

1. **Review this architecture** with project stakeholders
2. **Validate ETF ticker symbols** on Yahoo Finance (verify LQQ.PA, CL2.PA, WPEA.PA exist)
3. **Set up development environment** (install dependencies)
4. **Implement Phase 1** (core pipeline)
5. **Test with live data** for 1 week to validate reliability
6. **Iterate based on** real-world data quality observations

---

**Document Version:** 1.0
**Last Updated:** 2025-12-10
**Author:** Sophie, Data Engineer
**Status:** DRAFT - Pending Review
