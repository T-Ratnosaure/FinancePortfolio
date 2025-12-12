# Data Team Analysis: PEA Portfolio Data Architecture

**Date:** December 10, 2025
**Team Lead:** Florian (Data Manager)
**Contributors:** Sophie (Pipeline Design), Dulcy (ML Engineering), Pierre-Jean (ML Advisor), Renaud (Production), Lucas (Cost Optimization)

---

## Executive Summary

The Data team has designed a comprehensive data architecture for the PEA portfolio management system. Key decisions:

1. **Zero-cost MVP possible** using free-tier data sources (Yahoo Finance, FRED, ECB)
2. **DuckDB for storage** - embedded OLAP database, perfect for analytics
3. **Dagster for orchestration** - better than Airflow for data pipelines
4. **Manual trade execution workflow** designed around Boursobank limitations
5. **Estimated operational cost: EUR 0-8/month**

---

## 1. Data Source Strategy

### ETF Price Data (LQQ, CL2, WPEA)

**Requirements:**
- Daily OHLCV (Open, High, Low, Close, Volume)
- Optional intraday (1-min or 5-min bars)
- Bid-ask spreads
- NAV vs market price tracking
- Corporate actions

**Provider Recommendations (Ranked):**

| Provider | Cost | Coverage | Quality | Use Case |
|----------|------|----------|---------|----------|
| **Yahoo Finance** | EUR 0 | Excellent | Good | Primary source |
| **Alpha Vantage** | EUR 0-50 | Good | High | Backup source |
| **Twelve Data** | EUR 8 | Good | High | Reliability backup |
| **EOD Historical** | EUR 75 | Comprehensive | Institutional | Premium option |

**Primary Implementation:**
```python
import yfinance as yf

# Symbols for Euronext Paris
symbols = {
    'LQQ': 'LQQ.PA',   # Lyxor Nasdaq-100 2x
    'CL2': 'CL2.PA',   # Amundi MSCI USA 2x
    'WPEA': 'WPEA.PA'  # World PEA
}

# Download with fallback
def fetch_etf_data(symbol: str, period: str = "15y") -> pd.DataFrame:
    ticker = yf.Ticker(symbols[symbol])
    return ticker.history(period=period)
```

### Market Signal Data

**Required Indices:**
- S&P 500 (SPX) - US market
- NASDAQ-100 (NDX) - Tech sector
- STOXX Europe 600 - European benchmark
- VIX - Volatility index
- EUR/USD - Currency

**Macro Indicators:**
- ECB interest rates
- US Federal Reserve rates
- Yield curve (2Y/10Y spread)
- Commodity prices

**Provider Recommendations:**

| Data Type | Source | Cost | API |
|-----------|--------|------|-----|
| Index prices | Yahoo Finance | Free | `yfinance` |
| VIX | Yahoo Finance | Free | `yfinance` |
| Macro indicators | FRED | Free | `fredapi` |
| ECB rates | ECB API | Free | REST API |
| Yield curves | FRED | Free | `fredapi` |

---

## 2. Data Pipeline Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                            │
├─────────────────┬──────────────┬────────────────────────────┤
│ Yahoo Finance   │ Alpha Vantage│ FRED / ECB APIs            │
│ (Primary ETF)   │ (Backup ETF) │ (Macro Data)               │
└────────┬────────┴──────┬───────┴──────────┬─────────────────┘
         │               │                  │
         └───────────────┼──────────────────┘
                         │
                    ┌────▼────┐
                    │ Ingestion│
                    │  Layer   │
                    │ (Dagster)│
                    └────┬─────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
       ┌────▼───┐   ┌───▼────┐  ┌───▼────┐
       │Raw Data│   │Validate│  │Transform│
       │ Store  │   │ & Clean│  │& Enrich │
       │(DuckDB)│   └───┬────┘  └───┬─────┘
       └────────┘       │           │
                    ┌───▼───────────▼───┐
                    │   Analytics DB    │
                    │    (DuckDB)       │
                    └───────┬───────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
       ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
       │ML Models│    │Dashboard │    │Alerts   │
       │(Training)│   │(Streamlit)│   │(Manual) │
       └─────────┘    └──────────┘    └─────────┘
```

### Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Orchestration** | Dagster | Type-safe, Python 3.12 compatible, built-in data quality |
| **Storage** | DuckDB | Embedded OLAP, columnar, 100x faster for analytics |
| **Validation** | Great Expectations | Automated data quality checks |
| **Dashboard** | Streamlit | Local deployment, Python-native |
| **Alerts** | Email (SMTP) | Simple, free via Gmail |

### Why DuckDB over PostgreSQL

1. **Embedded** - No server management
2. **Columnar** - Optimized for analytical queries
3. **Python-native** - Excellent pandas integration
4. **Single file** - Easy backup and portability
5. **Free** - Zero operational cost

---

## 3. Database Schema Design

### Schema Structure

```
finance_portfolio.duckdb
│
├── raw/
│   ├── etf_prices_raw          (append-only, never delete)
│   ├── macro_indicators_raw
│   └── data_ingestion_log
│
├── cleaned/
│   ├── etf_prices_daily        (validated, no duplicates)
│   ├── etf_prices_intraday     (if implemented)
│   ├── macro_indicators
│   └── derived_features        (volatility, returns, etc.)
│
└── analytics/
    ├── portfolio_positions     (manual input from Boursobank)
    ├── trade_signals           (model outputs)
    └── backtest_results
```

### Table Definitions

**ETF Daily Prices:**
```sql
CREATE TABLE cleaned.etf_prices_daily (
    symbol VARCHAR NOT NULL,           -- 'LQQ.PA', 'CL2.PA', 'WPEA.PA'
    date DATE NOT NULL,
    open DECIMAL(10,4) NOT NULL,
    high DECIMAL(10,4) NOT NULL,
    low DECIMAL(10,4) NOT NULL,
    close DECIMAL(10,4) NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close DECIMAL(10,4),
    source VARCHAR NOT NULL,           -- 'yahoo', 'alpha_vantage'
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
);
```

**Portfolio Positions:**
```sql
CREATE TABLE analytics.portfolio_positions (
    position_id INTEGER PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    entry_date DATE NOT NULL,
    entry_price DECIMAL(10,4) NOT NULL,
    quantity INTEGER NOT NULL,
    exit_date DATE,
    exit_price DECIMAL(10,4),
    trade_type VARCHAR,                -- 'manual', 'signal_based'
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Partitioning Strategy

- Partition `etf_prices_daily` by year
- DuckDB handles partitioning via Parquet export
- Raw data: Keep indefinitely
- Intraday data: Keep 90 days, then aggregate to daily

---

## 4. Data Quality Controls

### Great Expectations Validation Rules

**Price Data Validations:**

1. **Completeness:**
   - No missing trading days (vs Euronext calendar)
   - All OHLCV fields present
   - Zero-volume days flagged

2. **Accuracy:**
   - High >= Low (no inverted candles)
   - Close within [Low, High]
   - Price changes >20% flagged (corporate actions)
   - Volume spikes >10x average flagged

3. **Timeliness:**
   - Data within 30 min of scheduled time
   - SLA breach alerts

4. **Consistency:**
   - Cross-validate Yahoo vs Alpha Vantage (±0.5%)
   - NAV divergence tracked

**Example Validation:**
```python
from great_expectations.core import ExpectationConfiguration

expectation = ExpectationConfiguration(
    expectation_type="expect_column_values_to_be_between",
    kwargs={
        "column": "close",
        "min_value": {"$PARAMETER": "row.low"},
        "max_value": {"$PARAMETER": "row.high"},
        "mostly": 1.0,
    },
)
```

### Monitoring and Alerting

| Alert Type | Channel | Condition |
|------------|---------|-----------|
| **Critical** | Email | Ingestion fails after 3 retries |
| **Critical** | Email | Data quality check failures |
| **Warning** | Dashboard | Price discrepancies >0.5% |
| **Warning** | Dashboard | Unusual volume patterns |
| **Info** | Dashboard | Data delays <30 min |

---

## 5. Pipeline Schedule

### Daily ETF Updates
- **Schedule:** 18:30 CET (after Euronext close at 17:30)
- **Retry policy:** 3 attempts with exponential backoff
- **SLA:** Data available by 19:00 CET

### Macro Data Updates
- **Schedule:** Weekly (Sundays at 06:00 CET)
- **Rationale:** Macro indicators change infrequently

### Historical Backfill
- **One-time:** 15 years of daily data (2010-present)
- **Storage:** ~5MB per ETF
- **Rate limit:** 1 request/second to avoid API bans

---

## 6. Manual Trade Execution Workflow

### Process Flow (Boursobank has no API)

```
┌──────────────┐
│ Model Signal │  (e.g., "BUY LQQ at market")
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Alert Dashboard│ (Streamlit notification)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Human Review  │ (Verify signal, check conditions)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Manual Execute│ (Login to Boursobank, place order)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Manual Record │ (Input actual execution price/time)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Update Position│ (Add to portfolio_positions table)
└──────────────┘
```

### Signal Dashboard Requirements

**Display Elements:**
- Current portfolio value (manual input)
- Recommended trades (symbol, action, quantity, confidence)
- Risk metrics (VaR, portfolio volatility)
- Time since signal generated

**Manual Input Form:**
- Executed trade details
- Rationale for deviation from signal

### Position Reconciliation

**Weekly Process:**
1. CSV export from Boursobank (manual download)
2. Python script to compare with database
3. Flag discrepancies
4. Audit log of all adjustments

---

## 7. Cost Analysis

### Monthly Cost Projection

| Component | Cost (EUR/mo) | Notes |
|-----------|---------------|-------|
| Yahoo Finance | 0 | Free tier |
| Alpha Vantage | 0 | Free tier (25 calls/day) |
| Twelve Data | 8 | Optional reliability backup |
| FRED API | 0 | Free |
| ECB API | 0 | Free |
| DuckDB | 0 | Self-hosted |
| Dagster | 0 | Self-hosted |
| Email alerts | 0 | Gmail free tier |
| **Total (Minimum)** | **EUR 0** | |
| **Total (Recommended)** | **EUR 8** | With Twelve Data backup |

### Cost Optimization Strategies

**Short-term (0-3 months):**
- Free-tier only
- Daily updates (no intraday)
- Self-hosted infrastructure

**Medium-term (3-12 months):**
- Add Twelve Data (EUR 8/mo) for reliability
- Implement caching
- Monitor quota usage

**Long-term (12+ months):**
- Evaluate premium data only if strategy shows alpha
- Consider co-location with other investors to share costs

---

## 8. Implementation Roadmap

### Phase 1: MVP Data Pipeline (Week 1-2)
**Owner:** Sophie
- [ ] Set up DuckDB schema
- [ ] Yahoo Finance daily ETF ingestion
- [ ] FRED macro data ingestion
- [ ] Basic Great Expectations validation
- [ ] Manual CSV upload for positions

**Acceptance:** 15 years ETF data + 10 years macro loaded

### Phase 2: Orchestration (Week 3)
**Owner:** Renaud
- [ ] Dagster pipeline setup
- [ ] Scheduled daily updates
- [ ] Email alerts
- [ ] Basic Streamlit dashboard

**Acceptance:** 7 consecutive days of successful updates

### Phase 3: Reliability (Week 4)
**Owner:** Sophie
- [ ] Alpha Vantage backup integration
- [ ] Cross-validation between sources
- [ ] Retry logic with backoff
- [ ] Historical backfill job

**Acceptance:** Automatic failover working

### Phase 4: Manual Trade Workflow (Week 5)
**Owner:** Renaud
- [ ] Signal dashboard
- [ ] Trade execution form
- [ ] Position reconciliation script

**Acceptance:** End-to-end test passed

### Phase 5: Production Hardening (Week 6)
**Owner:** Lucas
- [ ] Cost monitoring dashboard
- [ ] DuckDB backup automation
- [ ] Performance optimization
- [ ] Documentation complete

**Acceptance:** 14 days without manual intervention

---

## 9. Python Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # Existing
    "langchain>=1.1.3",
    "langchain-anthropic>=1.2.0",
    "langgraph>=1.0.4",
    "pydantic>=2.12.5",
    # Data ingestion
    "yfinance>=0.2.37",
    "fredapi>=0.5.2",
    "alpha-vantage>=2.3.1",
    # Database
    "duckdb>=0.10.0",
    "sqlalchemy>=2.0.0",
    # Orchestration
    "dagster>=1.6.0",
    "dagster-webserver>=1.6.0",
    # Data quality
    "great-expectations>=0.18.0",
    # Dashboard
    "streamlit>=1.31.0",
    "plotly>=5.18.0",
    # Utilities
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "requests>=2.31.0",
]
```

---

## 10. Open Questions

### Critical (Requires Clarification)

1. **Intraday Data:** Is daily sufficient or does model need intraday?
   - Impact: 33x increase in API calls
   - Recommendation: Start daily, add intraday if backtests prove value

2. **Historical Depth:** How many years needed for training?
   - Assumption: 15 years (2010-2025)
   - Includes multiple market cycles

3. **Signal Latency:** Max acceptable delay from signal to execution?
   - Assumption: 24 hours acceptable
   - Signals generated overnight for next day

4. **Rebalancing Frequency:** Daily, weekly, monthly?
   - Assumption: Weekly maximum (manual execution limit)

### Non-Critical (Defer)

- Alternative data sources (sentiment, news)
- Cloud hosting vs local
- Real-time alerts vs email

---

## 11. Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Yahoo rate limiting | High | Medium | Multi-source fallback |
| Bad price data | High | Low | Cross-validation |
| Missing trading days | Medium | Low | Holiday calendar |
| API outages | Medium | Medium | Cached data, retry |
| Manual execution errors | High | Medium | Double-entry verification |
| Pipeline failures | Medium | Low | Email alerts, retries |
| Database corruption | High | Very Low | Daily backups |

---

**Document Version:** 1.0
**Prepared By:** Florian (Data Team Manager)
**Review Required From:** Sophie, Dulcy, Pierre-Jean, Renaud, Lucas
