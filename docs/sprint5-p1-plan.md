# Sprint 5 P1 Plan: Data Quality & Monitoring

**Sprint:** Sprint 5 - P1 (Week 2)
**Duration:** 2 weeks
**Focus:** Data quality monitoring, circuit breakers, logging infrastructure
**Owner:** Sophie (Data Engineer)

---

## Overview

Sprint 5 P0 successfully delivered staleness detection and FRED retry logic. P1 focuses on completing the data quality infrastructure to make the system fully production-ready.

**P0 Completion Score:** 8/10
**P1 Target Score:** 9/10 (Production-ready)

---

## P1 Priorities

### Priority 1: Circuit Breaker Pattern (HIGH)

**Issue:** No circuit breaker means failed APIs are repeatedly called, wasting resources and time.

**Deliverable:** Implement circuit breaker for all external API calls

**Files to Create/Modify:**
```
src/data/fetchers/base.py         # Add CircuitBreakerMixin
src/data/fetchers/fred.py          # Apply circuit breaker
src/data/fetchers/yahoo.py         # Apply circuit breaker
tests/data/fetchers/test_circuit.py # Circuit breaker tests
```

**Implementation:**
```python
# src/data/fetchers/base.py
from pybreaker import CircuitBreaker, CircuitBreakerError

class CircuitBreakerMixin:
    """Mixin to add circuit breaker to fetchers."""

    def __init__(self, fail_max: int = 5, timeout_duration: int = 300):
        """Initialize circuit breaker.

        Args:
            fail_max: Number of failures before opening circuit
            timeout_duration: Seconds to keep circuit open
        """
        self.breaker = CircuitBreaker(
            fail_max=fail_max,
            timeout_duration=timeout_duration,
            name=self.__class__.__name__
        )

    def _call_with_breaker(self, func, *args, **kwargs):
        """Call function with circuit breaker protection."""
        try:
            return self.breaker.call(func, *args, **kwargs)
        except CircuitBreakerError as e:
            raise DataNotAvailableError(
                f"Circuit breaker open for {self.__class__.__name__}. "
                f"Service temporarily unavailable.",
                source=self.__class__.__name__
            ) from e
```

**Effort:** 4 hours
**Tests Required:**
- Circuit opens after N failures
- Circuit auto-closes after timeout
- Calls succeed when circuit closed
- Calls fail fast when circuit open

**Acceptance Criteria:**
- [ ] All fetchers inherit CircuitBreakerMixin
- [ ] Circuit state is logged on state changes
- [ ] Circuit breaker behavior is tested
- [ ] Documentation updated with circuit breaker config

---

### Priority 2: Data Quality Monitoring Framework (HIGH)

**Issue:** No automated data quality checks beyond Pydantic validation.

**Deliverable:** Comprehensive data quality checking framework

**Files to Create:**
```
src/data/quality.py                 # Data quality checker
tests/test_data/test_quality.py     # Quality tests
```

**Implementation:**
```python
# src/data/quality.py
from datetime import date, timedelta
from typing import Protocol

class DataQualityChecker:
    """Check data quality for various data types."""

    def check_price_completeness(
        self,
        storage: DuckDBStorage,
        symbol: str,
        start_date: date,
        end_date: date,
        min_completeness: float = 0.95
    ) -> tuple[float, list[date]]:
        """Check if price data is complete for date range.

        Args:
            storage: DuckDB storage instance
            symbol: Symbol to check
            start_date: Start date of range
            end_date: End date of range
            min_completeness: Minimum completeness ratio (0-1)

        Returns:
            Tuple of (completeness_ratio, missing_dates)

        Raises:
            DataQualityError: If completeness below threshold
        """
        expected_days = self._count_trading_days(start_date, end_date)
        prices = storage.get_prices(symbol, start_date, end_date)
        actual_days = len(prices)

        completeness = actual_days / expected_days if expected_days > 0 else 0.0

        if completeness < min_completeness:
            missing_dates = self._find_missing_trading_days(
                prices, start_date, end_date
            )
            raise DataQualityError(
                f"Data completeness {completeness:.1%} below threshold "
                f"{min_completeness:.1%} for {symbol}. "
                f"Missing {len(missing_dates)} trading days."
            )

        return completeness, []

    def check_price_consistency(self, prices: list[DailyPrice]) -> list[str]:
        """Check OHLC consistency and other data quality rules.

        Rules:
        - high >= low (already validated by Pydantic)
        - high >= open, close
        - low <= open, close
        - No price changes > 20% in one day (outlier)
        - Volume not 0 (except holidays)

        Returns:
            List of warning messages
        """
        warnings = []

        for i, price in enumerate(prices):
            # Check for extreme price movements (>20% change)
            if i > 0:
                prev_close = float(prices[i-1].close)
                curr_close = float(price.close)
                pct_change = abs((curr_close - prev_close) / prev_close)

                if pct_change > 0.20:
                    warnings.append(
                        f"{price.symbol} on {price.date}: Extreme price movement "
                        f"{pct_change:.1%}. Possible data error or halt/resume."
                    )

            # Check for zero volume (suspicious)
            if price.volume == 0:
                warnings.append(
                    f"{price.symbol} on {price.date}: Zero volume. "
                    f"Possible holiday or data gap."
                )

        return warnings

    def detect_outliers(
        self, prices: list[DailyPrice], z_threshold: float = 3.0
    ) -> list[tuple[DailyPrice, str]]:
        """Detect outliers using Z-score method.

        Args:
            prices: List of prices to analyze
            z_threshold: Z-score threshold for outlier detection

        Returns:
            List of (price, reason) tuples for outliers
        """
        # Implementation with pandas for Z-score calculation
        ...

    def _count_trading_days(self, start_date: date, end_date: date) -> int:
        """Count expected trading days (Mon-Fri, excluding holidays)."""
        # Use pandas.tseries.holiday for market holidays
        from pandas.tseries.holiday import USFederalHolidayCalendar
        from pandas.tseries.offsets import CustomBusinessDay

        cal = USFederalHolidayCalendar()
        bday = CustomBusinessDay(calendar=cal)

        # Count business days
        return len(pd.date_range(start_date, end_date, freq=bday))


class DataQualityError(Exception):
    """Raised when data quality checks fail."""
    pass
```

**Effort:** 8 hours
**Tests Required:**
- Completeness detection
- Consistency validation
- Outlier detection
- Trading day calculation

**Acceptance Criteria:**
- [ ] Completeness checker implemented and tested
- [ ] Consistency checker validates OHLC rules
- [ ] Outlier detection with configurable thresholds
- [ ] Quality checks integrated into data pipeline
- [ ] Quality metrics logged to DuckDB

---

### Priority 3: Complete Logging Infrastructure (MEDIUM)

**Issue:** Many modules still use print() instead of proper logging.

**Deliverable:** Replace all print() with structured logging

**Files to Modify:**
```
src/data/fetchers/yahoo.py         # Replace print with logging
examples/yahoo_fetcher_example.py  # Replace print with logging
examples/fred_fetcher_example.py   # Replace print with logging
main.py                            # Replace print with logging
```

**Implementation Strategy:**

1. **Add logging to Yahoo fetcher:**
   ```python
   # Before:
   print(f"Fetching data for {symbol}...")

   # After:
   logger.info(f"Fetching data for {symbol}", extra={
       "symbol": symbol,
       "start_date": start_date.isoformat(),
       "end_date": end_date.isoformat()
   })
   ```

2. **Use structured logging for production:**
   ```python
   # src/data/logging_config.py
   import logging.config

   LOGGING_CONFIG = {
       "version": 1,
       "formatters": {
           "json": {
               "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
               "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
           },
           "standard": {
               "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
           }
       },
       "handlers": {
           "console": {
               "class": "logging.StreamHandler",
               "formatter": "standard",
               "stream": "ext://sys.stdout"
           },
           "file": {
               "class": "logging.handlers.RotatingFileHandler",
               "formatter": "json",
               "filename": "logs/data_pipeline.log",
               "maxBytes": 10485760,  # 10MB
               "backupCount": 5
           }
       },
       "root": {
           "level": "INFO",
           "handlers": ["console", "file"]
       }
   }
   ```

**Effort:** 4 hours

**Acceptance Criteria:**
- [ ] No print() statements in src/ directory
- [ ] All fetchers use logger.info/warning/error
- [ ] Structured logging configuration available
- [ ] Log levels properly used (DEBUG/INFO/WARNING/ERROR)

---

### Priority 4: Retry Logic Base Class (MEDIUM)

**Issue:** Retry logic duplicated between Yahoo and FRED fetchers.

**Deliverable:** Extract retry logic to reusable base class

**Files to Modify:**
```
src/data/fetchers/base.py          # Add RetryableFetcherMixin
src/data/fetchers/fred.py          # Use mixin
src/data/fetchers/yahoo.py         # Use mixin
```

**Implementation:**
```python
# src/data/fetchers/base.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class RetryableFetcherMixin:
    """Mixin to add configurable retry logic to fetchers."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_min_wait: int = 2,
        retry_max_wait: int = 10,
        retry_multiplier: int = 1
    ):
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            retry_min_wait: Minimum wait time in seconds
            retry_max_wait: Maximum wait time in seconds
            retry_multiplier: Exponential backoff multiplier
        """
        self._max_retries = max_retries
        self._retry_min_wait = retry_min_wait
        self._retry_max_wait = retry_max_wait
        self._retry_multiplier = retry_multiplier

    def _create_retry_decorator(self):
        """Create retry decorator with configured parameters."""
        return retry(
            retry=retry_if_exception_type((RateLimitError, NetworkError)),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(
                multiplier=self._retry_multiplier,
                min=self._retry_min_wait,
                max=self._retry_max_wait
            )
        )

    def _retry_call(self, func, *args, **kwargs):
        """Execute function with retry logic."""
        decorator = self._create_retry_decorator()
        return decorator(func)(*args, **kwargs)
```

**Effort:** 4 hours

**Acceptance Criteria:**
- [ ] Base retry logic extracted to mixin
- [ ] Both fetchers use common retry logic
- [ ] Retry parameters configurable per fetcher
- [ ] Tests verify retry behavior
- [ ] Documentation updated

---

## Estimated Timeline

| Task | Priority | Effort | Week |
|------|----------|--------|------|
| Circuit Breaker Pattern | P1 | 4h | Week 1 |
| Data Quality Framework | P1 | 8h | Week 1 |
| Logging Infrastructure | P2 | 4h | Week 1 |
| Retry Base Class | P2 | 4h | Week 2 |
| **Total** | - | **20h** | **2 weeks** |

---

## Success Metrics

### Code Quality
- [ ] No print() statements in src/ directory
- [ ] All fetchers have circuit breakers
- [ ] Data quality checks cover 3+ categories
- [ ] Test coverage remains >90%

### Production Readiness
- [ ] Circuit breakers prevent cascading failures
- [ ] Data quality issues are detected automatically
- [ ] Logs are structured and machine-readable
- [ ] All retry logic is consistent

### Performance
- [ ] Data quality checks run in <1s per symbol
- [ ] Circuit breaker adds <10ms overhead
- [ ] Logging doesn't impact throughput

---

## Dependencies

**Required Python Packages:**
```bash
uv add pybreaker              # Circuit breaker pattern
uv add python-json-logger     # Structured JSON logging
uv add pandas                 # For trading day calculations
```

**Configuration Files:**
```
src/data/logging_config.py    # Logging configuration
src/data/quality_config.py    # Data quality thresholds
```

---

## Testing Requirements

### New Test Files
```
tests/data/fetchers/test_circuit.py      # Circuit breaker tests
tests/test_data/test_quality.py          # Data quality tests
tests/test_data/test_logging.py          # Logging tests
```

### Minimum Coverage
- Circuit breaker: 95%
- Data quality: 90%
- Retry logic: 95%

---

## Documentation Updates

1. **Update FRED fetcher guide:**
   - Add circuit breaker configuration
   - Document retry behavior
   - Add troubleshooting section

2. **Update data pipeline architecture:**
   - Add data quality layer diagram
   - Document quality thresholds
   - Circuit breaker patterns

3. **Create data quality guide:**
   - How to configure quality checks
   - How to interpret quality reports
   - How to handle quality failures

---

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Circuit breaker too sensitive | High | Medium | Make thresholds configurable |
| Quality checks too slow | Medium | Low | Optimize with vectorized pandas ops |
| Breaking existing code | High | Low | Comprehensive regression tests |
| JSON logging breaks existing tools | Medium | Medium | Keep both formatters available |

---

## Post-P1 Review Criteria

Sprint 5 P1 is complete when:

1. ✅ All P1 tasks delivered and tested
2. ✅ Production readiness score ≥9/10
3. ✅ No blocking issues in data pipeline
4. ✅ All tests passing with >90% coverage
5. ✅ Documentation updated
6. ✅ Review by IT-Core (Lamine) completed
7. ✅ Review by Quality Control completed

---

**Prepared by:** Sophie (Data Engineer)
**Review Date:** December 12, 2025
**Next Review:** Post-Sprint 5 P1 completion
