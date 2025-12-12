# Backtesting Framework - Executive Summary

**Date:** December 12, 2025
**Owner:** Helena (Backtester Agent)
**Status:** Design Complete, Ready for Implementation

---

## Overview

A comprehensive backtesting framework has been designed to validate the PEA portfolio optimization strategy through rigorous historical simulation. This framework will test the complete pipeline: HMM regime detection → allocation optimization → rebalancing logic with realistic transaction costs.

---

## Key Design Decisions

### 1. What to Backtest

**Complete Strategy Pipeline:**
- Feature calculation (VIX, spreads, trends)
- HMM regime detection (RISK_ON/NEUTRAL/RISK_OFF)
- Allocation optimization (LQQ, CL2, WPEA, CASH)
- Drift detection and rebalancing
- Trade execution with transaction costs
- Portfolio performance tracking

**Walk-Forward Validation:**
- Training window: 5 years (1,260+ days for HMM stability)
- Testing window: 1 year
- Step size: 6 months
- Total coverage: 2015-2025 (10 years)

### 2. Metrics to Track

**Performance Metrics:**
- Total/Annualized Return (Target: > 6%)
- Volatility (Target: 10-15%)
- Sharpe Ratio (Target: > 1.0)
- Sortino Ratio (Target: > 1.2)
- Max Drawdown (Target: < -15%)
- Win Rate (Target: > 55%)

**Risk Metrics:**
- Value at Risk (95%)
- Expected Shortfall (CVaR)
- Leveraged exposure compliance (< 30% HARD LIMIT)
- Cash buffer compliance (> 10% HARD LIMIT)
- Portfolio turnover

**Regime-Specific Analysis:**
- Performance by regime (RISK_ON, NEUTRAL, RISK_OFF)
- Regime persistence (average duration)
- Regime transition accuracy

### 3. Historical Period Coverage

**Minimum Requirements:**
- Training: 7 years (for HMM sample requirements)
- Testing: 3 years (to capture regime changes)
- Total: 10 years (2015-2025)

**Market Regime Diversity:**
- 2015-2016: Volatility spike (China, oil crash)
- 2017-2019: Bull market (low VIX)
- 2020 Q1: COVID crash (-35% drop)
- 2020 Q2-2021: V-shaped recovery
- 2022: Bear market (rates, inflation)
- 2023-2024: Tech rally recovery

**Out-of-Sample Validation:**
- In-sample: 2015-2023 (for development)
- Out-of-sample: 2024 (held out for final validation)

### 4. Transaction Cost Modeling

**Cost Components (per trade):**
- Bid-ask spread: 0.05% (5 bps)
- Slippage: 0.03% (3 bps)
- Market impact: 0.02% (2 bps)
- **Total: 0.10% per trade (10 bps)**

**Execution Assumptions:**
- Signal at close day t
- Execute at open/close day t+1
- Commission: 0 EUR (PEA broker)
- Volatility adjustment: 1.5x cost during high VIX

**Expected Annual Costs:**
- Threshold-based rebalancing (5% drift): 0.15-0.25% annually
- Monthly rebalancing: 0.50-1.00% annually

### 5. Walk-Forward Validation

**Methodology:**
```
Step 1: Train 2015-2020 → Test 2020-2021
Step 2: Train 2015.5-2020.5 → Test 2020.5-2021.5
Step 3: Train 2016-2021 → Test 2021-2022
... continue every 6 months ...
Final: Train 2019-2024 → Test 2024 (OUT-OF-SAMPLE)
```

**Bias Prevention:**
- No future data in features
- HMM trained only on historical data
- No parameter tuning on test periods
- Realistic execution timing (t+1)

---

## Quality Assurance

### Red Flags (Investigate if ANY occur)

- Sharpe ratio > 2.5 (likely overfit)
- Max drawdown < -40% (strategy failed)
- Win rate > 75% (unrealistic)
- Returns > 30% annually (too good to be true)
- Out-of-sample Sharpe < 50% of in-sample
- Parameter sensitivity > 50% performance swing

### Acceptance Criteria

The framework is COMPLETE when:
- [ ] Walk-forward validation runs without errors
- [ ] Minimum 7 years training, 3 years testing
- [ ] No look-ahead bias detected
- [ ] Risk limits enforced 100% of time periods
- [ ] Transaction costs realistic (0.10% per trade)
- [ ] Sharpe < 2.0 (no overfitting)
- [ ] Out-of-sample performance consistent

---

## Implementation Plan

### Phase 1: Core Infrastructure (8 hours)
- `BacktestEngine` class
- `TradeSimulator` class
- `TransactionCostModel` class
- `BacktestResult` model
- Unit tests

### Phase 2: Walk-Forward Logic (8 hours)
- Walk-forward window generation
- HMM training/testing pipeline
- Regime-based allocation integration
- Rebalancing logic
- Sample data validation

### Phase 3: Metrics & Validation (6 hours)
- Performance metric calculations
- Statistical validation (t-tests, stability)
- Equity curve and drawdown series
- Regime-specific performance analysis

### Phase 4: Full Backtest & Report (2 hours)
- Run 10-year backtest
- Generate markdown report
- Analyze results
- Document recommendations

**Total Effort:** 24 hours

---

## Expected Outcomes

### Success Scenario (Deploy Strategy)

```
Total Return:              80-100% (10 years)
Annualized Return:         6-7%
Sharpe Ratio:              1.0-1.5
Max Drawdown:              -15% to -25%
Transaction Costs:         < 0.25% annually
Out-of-Sample Consistency: Sharpe within ± 20%
```

**Interpretation:** Strategy demonstrates positive risk-adjusted returns with effective risk management. Proceed to paper trading.

### Neutral Scenario (Needs Iteration)

```
Total Return:              40-60% (10 years)
Annualized Return:         3-5%
Sharpe Ratio:              0.6-1.0
Max Drawdown:              -25% to -35%
Transaction Costs:         0.25-0.50% annually
Out-of-Sample Consistency: Sharpe drops by 30-40%
```

**Interpretation:** Strategy shows promise but requires refinement. Iterate on allocation logic or regime detection before deployment.

### Failure Scenario (Don't Deploy)

```
Total Return:              < 30% (10 years)
Annualized Return:         < 3%
Sharpe Ratio:              < 0.5
Max Drawdown:              < -40%
Transaction Costs:         > 0.50% annually
Out-of-Sample Consistency: Sharpe drops > 50%
```

**Interpretation:** Strategy does not provide sufficient risk-adjusted returns. Fundamental redesign required.

---

## Risk Assessment

### Technical Risks

| Risk | Mitigation |
|------|------------|
| Insufficient historical data | Use alternative sources, reduce training window carefully |
| HMM training instability | Monitor convergence, use robust initialization |
| Transaction cost uncertainty | Test multiple scenarios (0.05%, 0.10%, 0.15%) |

### Methodology Risks

| Risk | Mitigation |
|------|------------|
| Look-ahead bias | Rigorous code review, explicit checks |
| Overfitting to recent data | Out-of-sample validation, parameter stability tests |
| Regime detection failure | Manual regime labeling validation |
| Unrealistic execution | Conservative cost assumptions, stress testing |

### Interpretation Risks

| Risk | Mitigation |
|------|------------|
| Over-confidence in results | Clear caveats in documentation, skeptical reporting |
| Ignoring regime weakness | Detailed regime breakdown analysis |
| Dismissing transaction costs | Emphasize cost impact prominently in report |

---

## Deliverables

### Code
1. `src/backtesting/engine.py` - Main orchestration
2. `src/backtesting/simulator.py` - Trade execution
3. `src/backtesting/metrics.py` - Performance metrics
4. `src/backtesting/costs.py` - Transaction costs
5. `src/backtesting/walk_forward.py` - Validation logic
6. Unit tests (> 80% coverage)

### Documentation
1. **Backtest Report (Markdown):**
   - Performance summary table
   - Equity curve chart
   - Drawdown analysis
   - Regime distribution
   - Transaction cost breakdown
   - Statistical validation results
   - Deployment recommendation

2. **Statistical Validation:**
   - t-test on returns (significance)
   - Parameter stability matrix
   - Out-of-sample comparison

---

## Next Steps

### Immediate Actions (Week 2 of Sprint 5)
1. Validate 10-year historical data availability
2. Create `src/backtesting/` directory structure
3. Implement `TransactionCostModel` (simplest first)
4. Write unit tests for cost model
5. Begin `BacktestEngine` scaffolding

### Data Preparation
1. Fetch data: 2015-01-01 to 2024-12-31
2. Verify quality (no gaps, splits handled, dividends adjusted)
3. Prepare macro indicators (VIX, DGS2, DGS10, BAA10Y)
4. Store in DuckDB with indexing

### Integration
- Use existing `RegimeDetector` (src/signals/regime.py)
- Use existing `AllocationOptimizer` (src/signals/allocation.py)
- Use existing `RiskCalculator` (src/portfolio/risk.py)
- Extend for backtesting context as needed

---

## Escalation Criteria

Escalate to Jean-Yves (Research Lead) if:
- Historical data unavailable for 7+ years
- HMM training fails convergence tests
- Backtest results show Sharpe > 2.5 (overfitting suspected)
- Out-of-sample performance degrades > 50%

Escalate to Helena (Execution Manager) if:
- Transaction cost assumptions unclear
- Execution timing logic needs validation
- Resource constraints impact timeline

---

## Conclusion

The backtesting framework design is comprehensive, rigorous, and designed to catch overfitting and methodological errors before capital deployment. The walk-forward validation, realistic transaction costs, and out-of-sample testing will provide a reliable assessment of strategy viability.

**Status:** Ready for implementation (24-hour effort estimated)

**Critical Success Factor:** Maintain skepticism throughout. A backtest is only as good as its methodology. Our job is to ensure we don't deploy strategies that look great on paper but fail in reality.

---

**Helena (Backtester Agent)**
Reporting to Execution Manager
December 12, 2025
