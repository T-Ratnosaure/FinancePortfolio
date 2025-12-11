# Sprint 5 Roadmap - Comprehensive Development Plan

**Date:** December 11, 2025
**Version:** 1.0
**Status:** APPROVED
**Prepared by:** Multi-Team Review (10 Specialized Agents)

---

## Executive Summary

Following the completion of Sprint 4 (Security Remediation), a comprehensive multi-team review was conducted involving 10 specialized agents. This document consolidates all findings into a prioritized roadmap for Sprint 5.

### Current Project Status

| Metric | Value |
|--------|-------|
| Tests Passing | 222 (10 skipped - network dependent) |
| Overall Grade | B- (Conditional Pass) |
| Security Grade | B (up from C-) |
| Quality Grade | 7.2/10 |
| Production Ready | NO |

### Sprint 5 Goals

1. **Close Critical Gaps** - README, backtesting, integration tests
2. **Validate Strategy** - Backtesting framework with 3+ years of data
3. **Harden Infrastructure** - Pre-trade checks, staleness detection, CI improvements
4. **Complete Documentation** - Risk rationale, allocation methodology

---

## Priority Matrix

### P0 - CRITICAL (Blocking Production)

| ID | Task | Owner | Effort | Dependencies |
|----|------|-------|--------|--------------|
| P0-01 | Create comprehensive README.md | IT-Core | 2h | None |
| P0-02 | Harmonize risk limit constants (30% vs 50%) | Risk | 2h | None |
| P0-03 | Add pyrefly type checking to CI | CI/CD | 30m | None |
| P0-04 | Fix HMM minimum sample size (9 → 1,730+) | Research | 4h | None |
| P0-05 | Implement data staleness detection | Data | 8h | None |
| P0-06 | Add FRED fetcher retry logic | Data | 2h | None |
| P0-07 | Implement backtesting framework | Research/Execution | 24h | P0-04 |
| P0-08 | Add integration tests (3 workflows) | Quality | 16h | P0-07 |
| P0-09 | Implement functional CLI (main.py) | IT-Core | 8h | None |

**Total P0 Effort: ~67 hours**

### P1 - HIGH (Sprint 5 Must-Have)

| ID | Task | Owner | Effort | Dependencies |
|----|------|-------|--------|--------------|
| P1-01 | Create RISK_LIMITS_RATIONALE.md | Legal | 4h | None |
| P1-02 | Create ALLOCATION_METHODOLOGY.md | Legal | 4h | None |
| P1-03 | Implement pre-trade risk validation | Risk | 8h | P0-02 |
| P1-04 | Implement Expected Shortfall (CVaR) | Research | 4h | None |
| P1-05 | Multi-feature regime mapping | Research | 8h | P0-04 |
| P1-06 | Data quality monitoring framework | Data | 16h | P0-05 |
| P1-07 | Increase FRED test coverage (31% → 80%) | Data | 12h | P0-06 |
| P1-08 | Add pre-commit hooks | Quality | 4h | None |
| P1-09 | Complete portfolio performance metrics | Quality | 12h | None |
| P1-10 | Add dependency vulnerability scanning | Security | 2h | None |
| P1-11 | File integrity verification for models | Security | 4h | None |
| P1-12 | Create .env.example template | CI/CD | 30m | None |

**Total P1 Effort: ~79 hours**

### P2 - MEDIUM (Sprint 5 Should-Have)

| ID | Task | Owner | Effort | Dependencies |
|----|------|-------|--------|--------------|
| P2-01 | HMM cross-validation | Research | 8h | P0-04 |
| P2-02 | Tiered alert system | Risk | 4h | P1-03 |
| P2-03 | Risk monitoring service | Risk | 8h | P1-03 |
| P2-04 | Circuit breaker pattern | Data | 12h | P0-06 |
| P2-05 | Paper trading engine | Execution | 16h | P0-07 |
| P2-06 | Stress testing framework | Risk | 8h | P0-07 |
| P2-07 | Coverage enforcement (80%) | Quality | 2h | P0-08 |
| P2-08 | Configuration management | Quality | 8h | None |
| P2-09 | DATA_SOURCE_DISCLAIMER.md | Legal | 2h | None |
| P2-10 | Refactor DRY violations in risk.py | Quality | 4h | None |
| P2-11 | Sharpe autocorrelation correction | Research | 4h | None |
| P2-12 | Magic number constants extraction | Quality | 2h | None |

**Total P2 Effort: ~78 hours**

---

## Detailed Task Specifications

### P0-01: Create Comprehensive README.md

**Status:** CRITICAL - Currently empty
**Owner:** IT-Core (Clovis)
**Effort:** 2 hours

**Acceptance Criteria:**
- [ ] Project overview and description
- [ ] Quick start guide (installation, configuration)
- [ ] Usage examples for main commands
- [ ] Tech stack documentation
- [ ] Development setup instructions
- [ ] Links to detailed documentation

---

### P0-02: Harmonize Risk Limit Constants

**Status:** CRITICAL - Conflicting values
**Owner:** Risk (Nicolas)
**Effort:** 2 hours

**Problem:**
```python
# src/data/models.py (line 275)
MAX_LEVERAGED_EXPOSURE = 0.30  # 30%

# risk_assessment.py (line ~100)
leveraged_etf_max_total = 0.50  # 50%
```

**Solution:**
1. Consolidate all risk limits into `src/data/models.py`
2. Remove duplicate definitions from `risk_assessment.py`
3. Create single `RiskLimits` model as source of truth
4. Update all references to use centralized constants

**Acceptance Criteria:**
- [ ] Single source of truth for all risk limits
- [ ] No duplicate limit definitions
- [ ] All tests pass with harmonized limits

---

### P0-03: Add Pyrefly Type Checking to CI

**Status:** CRITICAL - Type errors undetected
**Owner:** CI/CD (Lamine)
**Effort:** 30 minutes

**Implementation:**
Add to `.github/workflows/ci.yml` after line 45:
```yaml
- name: Type check with Pyrefly
  run: uv run pyrefly check src/
```

**Acceptance Criteria:**
- [ ] CI fails on type violations
- [ ] Current codebase passes type check

---

### P0-04: Fix HMM Minimum Sample Size

**Status:** CRITICAL - Model unreliable
**Owner:** Research (Alexios)
**Effort:** 4 hours

**Problem:**
```python
# src/signals/regime.py (line 224)
min_samples = self.n_states * 3  # Only 9 samples!
```

**Required Minimum:**
- 3-state HMM with 9 features and full covariance
- Parameters: 27 (means) + 135 (covariances) + 6 (transitions) + 2 (initial) = 170
- Using 10:1 rule: **1,700 samples minimum** (~6.75 years daily data)

**Implementation:**
```python
def _calculate_minimum_samples(self) -> int:
    """Calculate minimum samples based on parameter count."""
    n_features = self._n_features or 9
    params_per_state = n_features + (n_features * (n_features + 1)) // 2
    total_params = (
        self.n_states * params_per_state
        + self.n_states * (self.n_states - 1)
        + (self.n_states - 1)
    )
    return max(total_params * 10, 1000)
```

**Acceptance Criteria:**
- [ ] Minimum sample requirement >= 1,700
- [ ] Clear error message when insufficient data
- [ ] Documentation of sample requirements

---

### P0-05: Implement Data Staleness Detection

**Status:** CRITICAL - Risk of stale decisions
**Owner:** Data (Sophie)
**Effort:** 8 hours

**Implementation:**
```python
# src/data/freshness.py
class DataFreshnessChecker:
    """Monitor and validate data freshness."""

    THRESHOLDS = {
        "price": {"warning": timedelta(hours=6), "stale": timedelta(days=1)},
        "macro": {"warning": timedelta(days=2), "stale": timedelta(days=7)},
    }

    def check_freshness(self, symbol: str, data_type: str) -> FreshnessLevel:
        """Check if data is fresh, warning, or stale."""
        pass

    def get_stale_symbols(self) -> list[str]:
        """Get all symbols with stale data."""
        pass
```

**Acceptance Criteria:**
- [ ] Freshness levels: FRESH, WARNING, STALE, CRITICAL
- [ ] Configurable thresholds per data type
- [ ] Integration with portfolio tracker
- [ ] Alerts for stale data

---

### P0-06: Add FRED Fetcher Retry Logic

**Status:** CRITICAL - Single failures permanent
**Owner:** Data (Sophie)
**Effort:** 2 hours

**Implementation:**
```python
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
)
def _fetch_series(self, series_id: str, start_date: date, end_date: date):
    # Existing implementation
    pass
```

**Acceptance Criteria:**
- [ ] 3 retry attempts with exponential backoff
- [ ] Proper logging of retry attempts
- [ ] Tests for retry behavior

---

### P0-07: Implement Backtesting Framework

**Status:** CRITICAL - Cannot validate strategy
**Owner:** Research (Jean-Yves) / Execution (Helena)
**Effort:** 24 hours

**Structure:**
```
src/backtesting/
    __init__.py
    engine.py          # Walk-forward validation engine
    simulator.py       # Trade execution simulator
    metrics.py         # Performance attribution
    costs.py           # Transaction cost modeling
```

**Key Components:**

1. **BacktestEngine**
```python
class BacktestEngine:
    def run(
        self,
        start_date: date,
        end_date: date,
        initial_capital: Decimal,
        rebalance_frequency: str,
    ) -> BacktestResult:
        """Execute walk-forward backtest."""
        pass
```

2. **BacktestResult**
```python
class BacktestResult(BaseModel):
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    total_costs: Decimal
    equity_curve: list[EquityPoint]
    trades: list[Trade]
```

**Acceptance Criteria:**
- [ ] Walk-forward validation with configurable windows
- [ ] Realistic transaction costs
- [ ] Performance metrics matching existing risk.py
- [ ] Equity curve generation
- [ ] Trade log with details

---

### P0-08: Add Integration Tests

**Status:** CRITICAL - Zero integration tests
**Owner:** Quality (Olivier)
**Effort:** 16 hours

**Required Tests:**

1. **Regime Detection Pipeline**
```python
# tests/integration/test_regime_pipeline.py
def test_full_regime_detection_workflow():
    """Test: Data fetch → DuckDB → Features → Regime → Allocation"""
    pass
```

2. **Portfolio Rebalancing Workflow**
```python
# tests/integration/test_rebalancing_pipeline.py
def test_complete_rebalancing_cycle():
    """Test: Positions → Drift → Trades → Execution"""
    pass
```

3. **Risk Reporting Pipeline**
```python
# tests/integration/test_risk_pipeline.py
def test_risk_report_generation():
    """Test: Portfolio → Market data → Risk metrics → Report"""
    pass
```

**Acceptance Criteria:**
- [ ] 3 critical workflow tests
- [ ] pytest markers configured
- [ ] CI runs integration tests separately
- [ ] All workflows validated end-to-end

---

### P0-09: Implement Functional CLI

**Status:** CRITICAL - main.py is placeholder
**Owner:** IT-Core (Clovis)
**Effort:** 8 hours

**Commands:**
```bash
python main.py fetch --source [yahoo|fred|all] --days 30
python main.py detect --train
python main.py portfolio --summary
python main.py rebalance --dry-run
python main.py risk --report
```

**Implementation:**
```python
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PEA Portfolio Optimization System"
    )
    subparsers = parser.add_subparsers(dest='command')

    # fetch command
    fetch_parser = subparsers.add_parser('fetch')
    fetch_parser.add_argument('--source', choices=['yahoo', 'fred', 'all'])
    fetch_parser.add_argument('--days', type=int, default=30)

    # detect command
    detect_parser = subparsers.add_parser('detect')
    detect_parser.add_argument('--train', action='store_true')

    # ... additional commands

    args = parser.parse_args()
    # Route to appropriate handler
```

**Acceptance Criteria:**
- [ ] All commands functional
- [ ] Help text for each command
- [ ] Proper error handling
- [ ] Exit codes for scripting

---

## Weekly Schedule

### Week 1: Critical Foundations (Dec 11-17)

| Day | Tasks | Hours |
|-----|-------|-------|
| Mon | P0-01 (README), P0-02 (Risk limits), P0-03 (CI pyrefly) | 4.5h |
| Tue | P0-04 (HMM samples), P0-06 (FRED retry) | 6h |
| Wed | P0-05 (Staleness detection) | 8h |
| Thu | P0-09 (CLI) - Start | 8h |
| Fri | P0-09 (CLI) - Complete, P1-12 (.env.example) | 4.5h |

**Week 1 Deliverables:**
- README.md populated
- Risk limits harmonized
- Pyrefly in CI
- HMM sample fix
- Staleness detection
- FRED retry logic
- Functional CLI

### Week 2: Core Infrastructure (Dec 18-24)

| Day | Tasks | Hours |
|-----|-------|-------|
| Mon | P0-07 (Backtesting) - Day 1 | 8h |
| Tue | P0-07 (Backtesting) - Day 2 | 8h |
| Wed | P0-07 (Backtesting) - Day 3 | 8h |
| Thu | P0-08 (Integration tests) - Day 1 | 8h |
| Fri | P0-08 (Integration tests) - Day 2 | 8h |

**Week 2 Deliverables:**
- Backtesting framework complete
- Integration tests for 3 workflows
- Strategy validated over historical data

### Week 3: Validation & Documentation (Dec 25-31)

| Day | Tasks | Hours |
|-----|-------|-------|
| Mon | P1-01, P1-02 (Legal docs) | 8h |
| Tue | P1-03 (Pre-trade validation) | 8h |
| Wed | P1-04 (CVaR), P1-05 (Multi-feature regime) | 8h |
| Thu | P1-08 (Pre-commit), P1-10 (Dep scanning) | 6h |
| Fri | P1-11 (Model integrity), P1-09 (Perf metrics) - Start | 8h |

**Week 3 Deliverables:**
- Risk documentation complete
- Pre-trade validation
- CVaR implementation
- Pre-commit hooks

### Week 4: Hardening (Jan 1-7)

| Day | Tasks | Hours |
|-----|-------|-------|
| Mon | P1-09 (Perf metrics) - Complete | 8h |
| Tue | P1-06 (Data quality) - Day 1 | 8h |
| Wed | P1-06 (Data quality) - Day 2 | 8h |
| Thu | P1-07 (FRED tests) | 8h |
| Fri | P2 items as time permits | 8h |

**Week 4 Deliverables:**
- Portfolio performance metrics
- Data quality framework
- FRED test coverage 80%+

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Backtesting reveals poor strategy | Medium | High | Iterate on allocation model |
| Integration tests uncover bugs | High | Medium | Fix as discovered |
| HMM requires more historical data | Medium | Medium | Use 7+ years of data |
| CI changes break builds | Low | Low | Test locally first |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Backtesting takes longer | Medium | High | Start early, parallelize |
| Holiday disruptions | High | Medium | Front-load critical work |
| Dependencies block progress | Low | Medium | Clear dependency order |

---

## Success Criteria

### Sprint 5 Complete When:

- [ ] All P0 tasks complete and merged
- [ ] All P1 tasks complete and merged
- [ ] 80%+ of P2 tasks complete
- [ ] Backtesting validates strategy over 3+ years
- [ ] Integration tests pass for all workflows
- [ ] Security grade maintained at B or higher
- [ ] Quality grade improved to 8.0+/10
- [ ] README provides complete onboarding

### Production Ready When:

- [ ] All P0 and P1 tasks complete
- [ ] Backtesting shows positive risk-adjusted returns
- [ ] 30 days paper trading successful
- [ ] All critical documentation complete
- [ ] Security audit re-run and passed
- [ ] Quality audit re-run and passed

---

## Appendix: Team Review Summaries

### IT-Core (Clovis) - Grade: 7/10
- Fixed: Examples, CI pipeline
- Remaining: Empty README, placeholder main.py, complexity violations

### Security (Maxime) - Grade: B
- Fixed: Pickle vulnerability, .gitignore, path validation
- Remaining: Dependency scanning, model integrity verification

### Research (Jean-Yves) - Grade: B+
- Fixed: Sortino ratio
- Remaining: Backtesting, HMM samples, multi-feature mapping

### Data (Sophie) - Grade: 6.5/10
- Fixed: Logging infrastructure
- Remaining: Staleness detection, FRED retry, test coverage

### Legal (Marc) - Grade: MEDIUM-LOW Risk
- Compliant for personal use
- Remaining: Risk limits documentation, allocation methodology

### Risk (Nicolas) - **Critical Findings**
- Fixed: Risk calculations correct
- Remaining: Limit harmonization, pre-trade validation, monitoring

### Execution (Helena) - Strong Foundation
- Fixed: Rebalancer complete
- Remaining: Backtesting, paper trading

### Quality (Olivier) - Grade: 7.2/10
- Fixed: Examples, logging
- Remaining: Integration tests, README, CLI, performance metrics

### CI/CD (Lamine) - Grade: C+
- Fixed: Basic CI working
- Remaining: Type checking, env validation, deployment workflow

### Cost (Lucas) - Grade: A+ (97/100)
- Excellent: Already optimized at 0 EUR/month
- No urgent actions needed

---

**Document Version:** 1.0
**Last Updated:** December 11, 2025
**Next Review:** End of Sprint 5
