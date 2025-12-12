# Yahoo Finance Fetcher Guide

## Overview

The `YahooFinanceFetcher` provides a robust interface for retrieving financial market data from Yahoo Finance. It's specifically designed to support the PEA Portfolio system with features including:

- ETF price data fetching (including PEA-compatible French ETFs)
- Index price retrieval
- VIX (volatility index) data
- Automatic retry logic with exponential backoff
- Rate limiting to respect API constraints
- Type-safe data models using Pydantic

## Installation

The required dependencies are already included in the project:

```bash
uv sync
```

Dependencies:
- `yfinance`: Yahoo Finance API wrapper
- `pandas`: Data manipulation
- `tenacity`: Retry logic
- `pydantic`: Data validation (already in project)

## Basic Usage

### 1. Initialize the Fetcher

```python
from src.data.fetchers.yahoo import YahooFinanceFetcher

# Create fetcher with default settings
fetcher = YahooFinanceFetcher()

# Or customize rate limiting and retries
fetcher = YahooFinanceFetcher(
    delay_between_requests=0.5,  # 500ms between requests
    max_retries=3                 # Retry up to 3 times on failure
)
```

### 2. Validate Connection

```python
if fetcher.validate_connection():
    print("Connected to Yahoo Finance")
else:
    print("Connection failed")
```

### 3. Fetch ETF Prices

```python
from datetime import date, timedelta
from src.data.models import ETFSymbol

end_date = date.today()
start_date = end_date - timedelta(days=30)

# Using ETFSymbol enum (for PEA-eligible symbols defined in models.py)
prices = fetcher.fetch_etf_prices(
    [ETFSymbol.LQQ, ETFSymbol.CL2, ETFSymbol.WPEA],
    start_date,
    end_date
)

# Using string symbols (for any ticker, including additional PEA ETFs)
additional_pea_prices = fetcher.fetch_etf_prices(
    ["PAEEM.PA", "PANX.PA"],  # Additional PEA-eligible ETFs
    start_date,
    end_date
)

# Process results
for price in prices:
    print(f"{price.symbol} on {price.date}: ${price.close:.2f}")
```

### 4. Fetch VIX Data

```python
vix_df = fetcher.fetch_vix(start_date, end_date)

print(f"Current VIX: {vix_df['vix'].iloc[-1]:.2f}")
print(f"Average VIX: {vix_df['vix'].mean():.2f}")
```

### 5. Fetch Index Prices

```python
# Fetch major US indices
indices_df = fetcher.fetch_index_prices(
    ["^GSPC", "^DJI", "^IXIC"],  # S&P 500, Dow, NASDAQ
    start_date,
    end_date
)

# Access specific index data
sp500_close = indices_df[("^GSPC", "Close")]
```

### 6. Batch Downloads

For efficiency when fetching multiple symbols:

```python
batch_df = fetcher.fetch_multiple_symbols(
    ["LQQ.PA", "CL2.PA", "WPEA.PA"],
    start_date,
    end_date
)
```

## PEA-Compatible ETFs

The system supports French PEA-compatible ETFs:

| Symbol | Name | Description |
|--------|------|-------------|
| `LQQ.PA` | Lyxor Nasdaq-100 2x | 2x leveraged NASDAQ-100 exposure |
| `CL2.PA` | Lyxor MSCI USA 2x | 2x leveraged US equity exposure |
| `WPEA.PA` | World PEA | Global equity exposure |

### Important Note on PEA Symbols

These PEA-eligible symbols are defined in the `ETFSymbol` enum in `src/data/models.py`:

```python
class ETFSymbol(str, Enum):
    """PEA-eligible ETF symbols traded on Euronext Paris."""

    LQQ = "LQQ.PA"  # Amundi Nasdaq-100 Daily (2x) Leveraged
    CL2 = "CL2.PA"  # Amundi MSCI USA Daily (2x) Leveraged
    WPEA = "WPEA.PA"  # Amundi MSCI World PEA
```

Additional PEA-eligible ETFs can be used directly as string symbols without adding them to the enum.

## Error Handling

The fetcher provides specific exception types:

```python
from src.data.fetchers.base import (
    FetchError,
    RateLimitError,
    DataNotAvailableError
)
from src.data.fetchers.yahoo import YahooFinanceFetcherError

try:
    prices = fetcher.fetch_etf_prices(symbols, start_date, end_date)
except RateLimitError:
    print("Rate limit exceeded, try again later")
except DataNotAvailableError as e:
    print(f"Data not available: {e}")
except YahooFinanceFetcherError as e:
    print(f"Fetcher error: {e}")
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

## Data Models

### DailyPrice

Returned by `fetch_etf_prices()`:

```python
class DailyPrice(BaseModel):
    symbol: ETFSymbol        # ETF symbol
    date: date              # Price date
    open: float             # Opening price
    high: float             # High price
    low: float              # Low price
    close: float            # Closing price
    volume: int             # Trading volume
    adjusted_close: float   # Adjusted close (for splits/dividends)
```

## Rate Limiting

Yahoo Finance has undocumented rate limits. The fetcher handles this through:

1. **Configurable Delays**: Set `delay_between_requests` to control request frequency
2. **Automatic Retries**: Uses exponential backoff (2s, 4s, 8s) for rate limit errors
3. **Batch Requests**: Use `fetch_multiple_symbols()` for efficient multi-symbol downloads

Recommended settings:
- Development/testing: `delay_between_requests=0.1` (100ms)
- Production: `delay_between_requests=0.5` (500ms)
- Heavy usage: `delay_between_requests=1.0` (1s)

## Best Practices

### 1. Date Range Validation

Always ensure start_date < end_date:

```python
from datetime import date

start = date(2024, 1, 1)
end = date(2024, 1, 31)

# This will raise ValueError
# bad_prices = fetcher.fetch_etf_prices(symbols, end, start)

# Correct usage
prices = fetcher.fetch_etf_prices(symbols, start, end)
```

### 2. Handle Missing Data

Some symbols may not have data for all requested dates:

```python
try:
    prices = fetcher.fetch_etf_prices(symbols, start_date, end_date)
except DataNotAvailableError as e:
    print(f"No data found: {e}")
    prices = []
```

### 3. Use Batch Downloads

For multiple symbols, batch downloads are more efficient:

```python
# Less efficient - multiple requests
for symbol in ["LQQ.PA", "CL2.PA", "WPEA.PA"]:
    df = fetcher._fetch_ticker_data(symbol, start, end)

# More efficient - single request
batch_df = fetcher.fetch_multiple_symbols(
    ["LQQ.PA", "CL2.PA", "WPEA.PA"], start, end
)
```

### 4. Implement Caching

For frequently accessed data, implement caching:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_prices(symbol: str, start: date, end: date):
    return fetcher.fetch_etf_prices([symbol], start, end)
```

## Testing

Run the test suite:

```bash
uv run pytest tests/test_yahoo_fetcher.py -v
```

Test coverage includes:
- Connection validation
- Single and multiple symbol fetching
- Date range validation
- VIX data retrieval
- Index price fetching
- Rate limiting enforcement
- Error handling

## Troubleshooting

### Issue: "No data available"

**Cause**: Symbol may be invalid or data not available for date range

**Solution**:
- Verify symbol exists on Yahoo Finance
- Check date range is reasonable
- Try a shorter date range

### Issue: "Rate limit exceeded"

**Cause**: Too many requests in short time

**Solution**:
- Increase `delay_between_requests`
- Use batch downloads
- Wait before retrying

### Issue: "Symbol not found in ETFSymbol enum"

**Cause**: Using string symbol but `fetch_etf_prices()` expects enum

**Solution**:
- Add symbol to `ETFSymbol` enum in `models.py`, OR
- Modify yahoo.py to handle string symbols gracefully (already implemented)

## API Reference

### YahooFinanceFetcher

#### Methods

- `__init__(delay_between_requests=0.5, max_retries=3)`: Initialize fetcher
- `validate_connection() -> bool`: Test Yahoo Finance connectivity
- `fetch_etf_prices(symbols, start_date, end_date) -> list[DailyPrice]`: Fetch ETF prices
- `fetch_index_prices(symbols, start_date, end_date) -> pd.DataFrame`: Fetch index data
- `fetch_vix(start_date, end_date) -> pd.DataFrame`: Fetch VIX data
- `fetch_multiple_symbols(symbols, start_date, end_date) -> pd.DataFrame`: Batch download

## Examples

See `C:/Users/larai/FinancePortfolio/examples/yahoo_fetcher_usage.py` for complete working examples.

## Architecture Notes

### Design Decisions

1. **Retry Logic**: Uses `tenacity` library for declarative retry behavior
2. **Rate Limiting**: Time-based throttling between requests
3. **Type Safety**: Pydantic models ensure data validation
4. **Error Hierarchy**: Specific exception types for different failure modes
5. **Flexibility**: Supports both enum and string symbols

### Integration Points

- **Base Fetcher**: Inherits from `BaseFetcher` abstract class
- **Models**: Uses `DailyPrice` and `ETFSymbol` from `src.data.models`
- **Storage**: Output models are ready for database storage
- **Signals**: Price data can feed into regime detection system

## Future Enhancements

Potential improvements:

1. **Caching Layer**: Add Redis/file-based cache for historical data
2. **Async Support**: Implement async methods for concurrent fetching
3. **Data Quality Checks**: Add validation for suspicious price data
4. **Extended Metrics**: Calculate additional metrics (returns, volatility)
5. **WebSocket Support**: Real-time data streaming for intraday trading

## Support

For issues or questions:
1. Check test suite for usage patterns
2. Review examples in `examples/yahoo_fetcher_usage.py`
3. Consult Yahoo Finance API documentation
4. Review error messages and stack traces
