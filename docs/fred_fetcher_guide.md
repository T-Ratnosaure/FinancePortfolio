# FRED Data Fetcher Guide

## Overview

The FRED (Federal Reserve Economic Data) fetcher provides access to macroeconomic indicators from the Federal Reserve Bank of St. Louis FRED database. This guide explains how to use the fetcher to retrieve market data for the PEA Portfolio system.

## Setup

### 1. Get a FRED API Key

1. Visit [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Create a free account if you don't have one
3. Request an API key (instant and free)

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
FRED_API_KEY=your_api_key_here
```

Or set the environment variable directly:

```bash
# Linux/Mac
export FRED_API_KEY=your_api_key_here

# Windows
set FRED_API_KEY=your_api_key_here
```

## Data Sources

The FRED fetcher retrieves the following indicators:

| Indicator | FRED Series | Description |
|-----------|------------|-------------|
| VIX | VIXCLS | CBOE Volatility Index - market volatility expectations |
| 2Y Treasury | DGS2 | 2-Year Treasury Constant Maturity Rate |
| 10Y Treasury | DGS10 | 10-Year Treasury Constant Maturity Rate |
| 2s10s Spread | T10Y2Y | 10-Year minus 2-Year Treasury spread (yield curve) |
| HY OAS Spread | BAMLH0A0HYM2 | ICE BofA High Yield Option-Adjusted Spread |

## Usage

### Basic Initialization

```python
from data.fetchers.fred import FREDFetcher

# Initialize with API key from environment
fetcher = FREDFetcher()

# Or pass API key explicitly
fetcher = FREDFetcher(api_key="your_api_key")

# Validate connection
if fetcher.validate_connection():
    print("Connected to FRED!")
```

### Fetch VIX Data

```python
from datetime import date, timedelta

end_date = date.today()
start_date = end_date - timedelta(days=30)

# Returns pandas DataFrame with 'vix' column
vix_df = fetcher.fetch_vix(start_date, end_date)
print(vix_df.head())
```

### Fetch Treasury Yields

```python
# Returns DataFrame with 'treasury_2y' and 'treasury_10y' columns
treasury_df = fetcher.fetch_treasury_yields(start_date, end_date)
print(treasury_df.head())

# Calculate yield curve slope
treasury_df['curve_slope'] = (
    treasury_df['treasury_10y'] - treasury_df['treasury_2y']
)
```

### Fetch Credit Spreads

```python
# Returns DataFrame with 'spread_2s10s' and 'hy_oas_spread' columns
spreads_df = fetcher.fetch_credit_spreads(start_date, end_date)
print(spreads_df.head())
```

### Fetch All Indicators

```python
# Returns list of MacroIndicator Pydantic models
indicators = fetcher.fetch_macro_indicators(start_date, end_date)

# Each indicator is a validated Pydantic model
for indicator in indicators[:5]:
    print(f"{indicator.indicator_name}: {indicator.value} on {indicator.date}")

# Group by indicator name
from collections import defaultdict
by_name = defaultdict(list)
for ind in indicators:
    by_name[ind.indicator_name].append(ind)

print(f"VIX observations: {len(by_name['VIX'])}")
```

## Error Handling

The fetcher includes robust error handling:

```python
from data.fetchers.fred import FREDFetcherError

try:
    fetcher = FREDFetcher()
    indicators = fetcher.fetch_macro_indicators(start_date, end_date)
except FREDFetcherError as e:
    print(f"FRED API error: {e}")
except ValueError as e:
    print(f"Invalid date range: {e}")
```

Common errors:
- **FREDFetcherError**: API key missing, invalid, or API request failed
- **ValueError**: Invalid date range (start_date > end_date)

## Data Quality Considerations

### Missing Data

FRED data may have gaps due to:
- **Weekends and holidays**: No trading data on non-business days
- **Publication schedules**: Some indicators published weekly/monthly
- **Data revisions**: Historical data may be updated

The fetcher handles missing data gracefully:
- Returns `NaN` values for missing observations
- Uses `pd.notna()` checks when converting to Pydantic models
- Only includes valid data points in the results

### Data Frequency

Different indicators have different update frequencies:
- **VIX**: Daily (business days)
- **Treasury Yields**: Daily (business days)
- **Credit Spreads**: Daily (business days)

### Data Latency

FRED data typically has minimal latency:
- Most daily indicators available by end of trading day
- Some indicators may have 1-day delay

## Integration with PEA Portfolio

The FRED fetcher integrates with the regime detection system:

```python
from data.fetchers.fred import FREDFetcher
from datetime import date, timedelta

# Fetch recent macro data
fetcher = FREDFetcher()
end_date = date.today()
start_date = end_date - timedelta(days=90)

indicators = fetcher.fetch_macro_indicators(start_date, end_date)

# Use indicators for regime classification
# High VIX + widening spreads = RISK_OFF regime
# Low VIX + tightening spreads = RISK_ON regime
```

## Testing

Run the test suite:

```bash
# Run all FRED fetcher tests
uv run pytest tests/data/fetchers/test_fred.py

# Run with coverage
uv run pytest tests/data/fetchers/test_fred.py --cov=data.fetchers.fred

# Run specific test
uv run pytest tests/data/fetchers/test_fred.py::TestFREDFetcher::test_fetch_vix
```

## Example Script

See `examples/fred_fetcher_example.py` for a complete working example:

```bash
uv run python examples/fred_fetcher_example.py
```

## API Rate Limits

FRED API has the following limits (as of 2024):
- **Requests per day**: 120,000
- **Requests per second**: No explicit limit, but throttling recommended

The fetcher does not implement rate limiting - add if needed for high-frequency usage.

## Troubleshooting

### Issue: "FRED API key not found"

**Solution**: Set the `FRED_API_KEY` environment variable or pass the key explicitly to the constructor.

### Issue: "No data returned for series"

**Possible causes**:
- Date range too recent (data not yet published)
- Date range too old (data not available)
- FRED series temporarily unavailable

**Solution**: Adjust date range or check FRED website for data availability.

### Issue: Connection validation fails

**Possible causes**:
- Network connectivity issues
- Invalid API key
- FRED API temporarily down

**Solution**: Check API key, network connection, and FRED status page.

## Architecture

### Class Structure

```
BaseFetcher (ABC)
    ├── __init__() [abstract]
    ├── validate_connection() [abstract]
    └── _validate_date_range() [concrete helper]

FREDFetcher(BaseFetcher)
    ├── __init__(api_key)
    ├── validate_connection()
    ├── fetch_vix(start_date, end_date) -> DataFrame
    ├── fetch_treasury_yields(start_date, end_date) -> DataFrame
    ├── fetch_credit_spreads(start_date, end_date) -> DataFrame
    ├── fetch_macro_indicators(start_date, end_date) -> list[MacroIndicator]
    └── _fetch_series(series_id, start_date, end_date) -> DataFrame [private]
```

### Data Flow

1. **Fetch**: Retrieve raw data from FRED API via fredapi client
2. **Transform**: Convert to pandas DataFrame with proper column names
3. **Validate**: Check for missing data, handle errors
4. **Model**: Convert to Pydantic MacroIndicator models
5. **Return**: Provide clean, validated data structures

## References

- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/fred/)
- [fredapi Python Library](https://github.com/mortada/fredapi)
- [FRED Data Series Browser](https://fred.stlouisfed.org/)
