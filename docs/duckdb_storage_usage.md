# DuckDB Storage Layer Usage Guide

## Overview

The DuckDB storage layer provides a robust, three-tiered database architecture for the PEA Portfolio system:

- **Raw Layer**: Unprocessed data directly from external sources
- **Cleaned Layer**: Validated, deduplicated, and processed data
- **Analytics Layer**: Derived metrics, portfolio positions, and analysis results

## Quick Start

```python
from src.data.storage.duckdb import DuckDBStorage
from src.data.models import DailyPrice, ETFSymbol
from datetime import date

# Initialize storage
storage = DuckDBStorage("data/portfolio.duckdb")

# Or use as context manager (recommended)
with DuckDBStorage("data/portfolio.duckdb") as storage:
    # Insert price data
    prices = [
        DailyPrice(
            symbol=ETFSymbol.LQQ,
            date=date(2024, 1, 1),
            open=450.0,
            high=455.0,
            low=449.0,
            close=454.0,
            volume=100000,
            adjusted_close=454.0,
        )
    ]
    storage.insert_prices(prices)

    # Query price data
    results = storage.get_prices(
        "LQQ.PA",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31)
    )
```

## Key Features

### 1. Price Data Management

**Insert prices** (bulk insert with validation):
```python
count = storage.insert_prices(prices)
print(f"Inserted {count} price records")
```

**Query price history**:
```python
prices = storage.get_prices(
    symbol="LQQ.PA",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31)
)
```

**Get latest prices**:
```python
latest = storage.get_latest_prices()
lqq_price = latest["LQQ.PA"]
print(f"LQQ.PA latest close: {lqq_price.close}")
```

### 2. Macroeconomic Indicators

```python
from src.data.models import MacroIndicator

indicators = [
    MacroIndicator(
        indicator_name="US_10Y_YIELD",
        date=date(2024, 1, 1),
        value=4.25,
        source="FRED",
    )
]
storage.insert_macro(indicators)
```

### 3. Portfolio Positions

```python
from src.data.models import Position

position = Position(
    symbol=ETFSymbol.LQQ,
    shares=10.0,
    average_cost=450.0,
    current_price=455.0,
    market_value=4550.0,
    unrealized_pnl=50.0,
    weight=0.5,
)
storage.insert_position(position)

# Get all current positions
positions = storage.get_positions()
```

### 4. Trade Records

```python
from src.data.models import Trade, TradeAction
from datetime import datetime

trade = Trade(
    symbol=ETFSymbol.WPEA,
    date=datetime.now(),
    action=TradeAction.BUY,
    shares=10.0,
    price=450.0,
    total_value=4500.0,
    commission=5.0,
)
storage.insert_trade(trade)

# Query trade history
trades = storage.get_trades(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)
```

## Database Schema

### Raw Layer
- `raw.etf_prices_raw` - Raw price data with ingestion metadata
- `raw.macro_indicators_raw` - Raw macro indicator data
- `raw.data_ingestion_log` - Data pipeline execution logs

### Cleaned Layer
- `cleaned.etf_prices_daily` - Validated daily prices (primary data source)
- `cleaned.macro_indicators` - Validated macro indicators
- `cleaned.derived_features` - Computed features (e.g., moving averages)

### Analytics Layer
- `analytics.portfolio_positions` - Current and historical positions
- `analytics.trades` - Trade execution records
- `analytics.trade_signals` - Generated trading signals
- `analytics.backtest_results` - Backtesting results

## Data Flow

1. **Ingestion**: Raw data is inserted into `raw.*` tables with timestamps
2. **Validation**: Data is validated using Pydantic models
3. **Cleaning**: Validated data is inserted into `cleaned.*` tables
4. **Analytics**: Derived metrics and analysis results go into `analytics.*` tables

## Error Handling

The storage layer includes comprehensive error handling:

- **Validation Errors**: Pydantic validation catches data quality issues
- **Duplicate Handling**: Uses `INSERT OR REPLACE` for idempotent operations
- **Logging**: All operations are logged for debugging
- **Graceful Degradation**: Invalid records are logged but don't stop processing

## Performance Considerations

- **Indexes**: Automatically created on frequently queried columns
- **Bulk Inserts**: Use `executemany()` for efficient batch operations
- **Query Optimization**: Leverages DuckDB's columnar storage for fast analytics
- **Connection Management**: Use context manager for proper cleanup

## Testing

Comprehensive test suite included in `tests/test_data/test_duckdb_storage.py`:

```bash
# Run tests
uv run pytest tests/test_data/test_duckdb_storage.py -v

# Run with coverage
uv run pytest tests/test_data/test_duckdb_storage.py --cov=src/data/storage
```

## Best Practices

1. **Always use context manager** for automatic connection cleanup
2. **Validate data** with Pydantic models before insertion
3. **Use bulk operations** for better performance
4. **Monitor logs** for data quality issues
5. **Regular backups** of the DuckDB file
6. **Test with realistic data** before production use

## Dependencies

- `duckdb`: Database engine
- `pydantic`: Data validation
- `typing`: Type hints

All dependencies are managed via UV and specified in `pyproject.toml`.
