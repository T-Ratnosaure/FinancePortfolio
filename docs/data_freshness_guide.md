# Data Freshness and Staleness Detection

## Overview

The Data Freshness system automatically tracks when data was last updated and warns when data becomes stale. This prevents making investment decisions based on outdated information.

## Architecture

### Components

1. **DataFreshness Model** (`src/data/models.py`)
   - Pydantic model for tracking data age
   - Automatic status calculation (FRESH, STALE, CRITICAL)
   - Human-readable warning messages

2. **DuckDB Integration** (`src/data/storage/duckdb.py`)
   - `raw.data_freshness` table for metadata storage
   - Automatic tracking on insert operations
   - Query methods for freshness checks

3. **Utility Functions** (`src/data/freshness.py`)
   - High-level functions for common checks
   - Comprehensive reporting
   - Logging integration

## Staleness Thresholds

### Default Thresholds

| Data Category | Stale After | Critical After |
|--------------|-------------|----------------|
| Price Data | 1 day | 7 days |
| Macro Data | 7 days | 30 days |
| Portfolio Positions | 1 hour | 24 hours |
| Trade Records | Never | Never |

These thresholds are configured in `src/data/models.py`:

```python
STALENESS_THRESHOLDS = {
    DataCategory.PRICE_DATA: timedelta(days=1),
    DataCategory.MACRO_DATA: timedelta(days=7),
    DataCategory.PORTFOLIO_DATA: timedelta(hours=1),
}
```

### Freshness Status

- **FRESH**: Data is within acceptable staleness threshold
- **STALE**: Data is beyond threshold but usable with warning
- **CRITICAL**: Data is too old to use safely

## Usage

### 1. Automatic Tracking

Freshness is tracked automatically when inserting data:

```python
from src.data.storage.duckdb import DuckDBStorage
from src.data.models import DailyPrice, ETFSymbol

storage = DuckDBStorage("data/portfolio.duckdb")

# Insert price data - freshness is tracked automatically
prices = [
    DailyPrice(
        symbol=ETFSymbol.LQQ,
        date=datetime.now().date(),
        open=100.0,
        high=105.0,
        low=99.0,
        close=103.0,
        volume=1000000,
        adjusted_close=103.0,
    )
]
storage.insert_prices(prices)
```

### 2. Checking Freshness

Check if specific data is stale:

```python
from src.data.freshness import check_price_data_freshness

# Check price data freshness
freshness = check_price_data_freshness(
    storage,
    "LQQ.PA",
    raise_on_critical=True  # Raise error if critically stale
)

if freshness.is_stale():
    print(freshness.get_warning_message())
```

### 3. Generating Reports

Generate comprehensive freshness reports:

```python
from src.data.freshness import generate_freshness_report

report = generate_freshness_report(storage)

# Print human-readable report
print(report)

# Get dictionary for programmatic access
report_dict = report.to_dict()
print(f"Stale datasets: {report_dict['summary']['stale']}")

# Check for issues
if report.has_critical_issues():
    print("CRITICAL: Some data is too stale!")
```

### 4. Handling Stale Data

When data is stale, you have options:

```python
from src.data.models import StaleDataError

try:
    # This will raise if data is critically stale
    storage.check_freshness(
        DataCategory.PRICE_DATA,
        symbol="LQQ.PA",
        raise_on_critical=True
    )
except StaleDataError as e:
    # Option 1: Re-fetch the data
    print("Data is critically stale, refreshing...")
    # ... call fetcher to update data ...

    # Option 2: Proceed with warning (not recommended)
    print(f"Warning: {e}")
    # ... use stale data anyway ...
```

## Integration with Data Pipeline

### In Fetchers

Fetchers should check freshness before using cached data:

```python
def get_prices(self, symbol: str) -> list[DailyPrice]:
    # Check if cached data is fresh
    freshness = storage.get_freshness(
        DataCategory.PRICE_DATA,
        symbol=symbol
    )

    if freshness and not freshness.is_stale():
        # Use cached data
        return storage.get_prices(symbol, start_date, end_date)

    # Data is stale or missing, fetch fresh data
    fresh_data = self._fetch_from_api(symbol)
    storage.insert_prices(fresh_data)
    return fresh_data
```

### In Analysis Pipelines

Always check freshness before running analysis:

```python
from src.data.freshness import log_freshness_warnings

def run_analysis(storage: DuckDBStorage):
    # Log any freshness issues
    log_freshness_warnings(storage)

    # Check critical data
    storage.check_freshness(
        DataCategory.PRICE_DATA,
        raise_on_critical=True
    )

    # Proceed with analysis...
```

### In Portfolio Management

Check position data freshness:

```python
from src.data.freshness import check_portfolio_freshness

freshness = check_portfolio_freshness(
    storage,
    raise_on_critical=False
)

if freshness and freshness.is_stale():
    print("Warning: Portfolio positions may be outdated")
    print(f"Last updated: {freshness.last_updated}")
```

## Best Practices

### 1. Check Before Important Decisions

Always check data freshness before:
- Generating trading signals
- Rebalancing portfolio
- Calculating risk metrics
- Running backtests

### 2. Set Appropriate Thresholds

Adjust thresholds based on your needs:
- Intraday trading: Use shorter thresholds
- Long-term investing: Longer thresholds acceptable
- Real-time signals: Very short thresholds

### 3. Automated Monitoring

Set up regular freshness checks:

```python
import schedule

def check_data_health():
    report = generate_freshness_report(storage)
    if report.has_critical_issues():
        send_alert("Critical data staleness detected!")

schedule.every().hour.do(check_data_health)
```

### 4. Logging

The system logs warnings automatically:

```python
import logging

logging.basicConfig(level=logging.WARNING)

# This will log warnings for stale data
storage.check_freshness(DataCategory.PRICE_DATA, symbol="LQQ.PA")
```

## Database Schema

The `raw.data_freshness` table tracks metadata:

```sql
CREATE TABLE raw.data_freshness (
    id INTEGER PRIMARY KEY,
    data_category VARCHAR NOT NULL,     -- 'price_data', 'macro_data', etc.
    symbol VARCHAR,                      -- Optional: Symbol for price data
    indicator_name VARCHAR,              -- Optional: Indicator for macro data
    last_updated TIMESTAMP NOT NULL,     -- When data was last fetched
    record_count INTEGER NOT NULL,       -- Number of records updated
    source VARCHAR NOT NULL,             -- Data source name
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(data_category, symbol, indicator_name, source)
);
```

## API Reference

### DataFreshness Model

```python
class DataFreshness(BaseModel):
    data_category: DataCategory
    symbol: str | None = None
    indicator_name: str | None = None
    last_updated: datetime
    record_count: int
    source: str

    def get_age() -> timedelta
    def get_status() -> FreshnessStatus
    def is_stale() -> bool
    def is_critical() -> bool
    def get_warning_message() -> str | None
```

### DuckDBStorage Methods

```python
# Update freshness (called automatically by insert methods)
storage._update_freshness(
    data_category: DataCategory,
    record_count: int,
    source: str,
    symbol: str | None = None,
    indicator_name: str | None = None
)

# Get freshness metadata
storage.get_freshness(
    data_category: DataCategory,
    symbol: str | None = None,
    indicator_name: str | None = None
) -> DataFreshness | None

# Check freshness with warnings/errors
storage.check_freshness(
    data_category: DataCategory,
    symbol: str | None = None,
    indicator_name: str | None = None,
    raise_on_critical: bool = True
) -> DataFreshness | None

# Get all freshness status
storage.get_all_freshness_status() -> list[DataFreshness]
```

### Utility Functions

```python
# Generate comprehensive report
generate_freshness_report(storage: DuckDBStorage) -> FreshnessReport

# Check specific data types
check_price_data_freshness(storage, symbol, raise_on_critical=True)
check_macro_data_freshness(storage, indicator_name, raise_on_critical=True)
check_portfolio_freshness(storage, raise_on_critical=True)

# Log all warnings
log_freshness_warnings(storage)
```

## Examples

See `examples/data_freshness_example.py` for a complete working example.

## Testing

Comprehensive tests are in `tests/test_data/test_freshness.py`:

```bash
# Run freshness detection tests
uv run pytest tests/test_data/test_freshness.py -v
```

## Troubleshooting

### Data Always Shows as Fresh

- Check that `insert_prices()` or `insert_macro()` completed successfully
- Verify freshness table is being updated: `SELECT * FROM raw.data_freshness`
- Ensure system time is correct

### False Stale Warnings

- Review threshold configuration in `src/data/models.py`
- Consider your data update frequency
- Adjust thresholds if needed

### Performance Concerns

- Freshness queries use indexed columns
- Metadata table is lightweight (one row per dataset)
- Minimal overhead on insert operations

## Future Enhancements

Potential improvements:
- Configurable thresholds per symbol/indicator
- Automatic data refresh triggers
- Integration with alerting systems
- Dashboard visualization
- Historical freshness tracking
