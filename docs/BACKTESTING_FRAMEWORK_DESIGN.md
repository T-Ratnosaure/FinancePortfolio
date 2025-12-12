# Backtesting Framework Design - P0-07

**Document Version:** 1.0
**Date:** December 12, 2025
**Owner:** Helena (Execution Manager) / Jean-Yves (Research Lead)
**Status:** DESIGN PHASE
**Estimated Effort:** 24 hours

---

## Executive Summary

This document provides a comprehensive design for the backtesting framework required to validate the PEA portfolio optimization strategy. The framework will test the complete pipeline: regime detection, allocation optimization, and rebalancing logic against historical market data with realistic transaction costs and execution assumptions.

### Critical Validation Goals

1. Validate the HMM regime detection produces sensible allocations over time
2. Measure risk-adjusted returns (Sharpe, Sortino, max drawdown)
3. Assess transaction cost impact on portfolio performance
4. Verify the strategy does not overfit to recent market conditions
5. Ensure risk limits are consistently enforced

---

## 1. What Should Be Backtested?

### 1.1 Complete Strategy Pipeline

The backtest must simulate the ENTIRE decision-making process:

```
Historical Market Data
    |
    v
Feature Calculation (VIX, trend, spreads, etc.)
    |
    v
HMM Regime Detection (RISK_ON/NEUTRAL/RISK_OFF)
    |
    v
Allocation Optimization (LQQ, CL2, WPEA, CASH weights)
    |
    v
Drift Detection & Rebalancing Logic
    |
    v
Trade Execution (with transaction costs)
    |
    v
Portfolio Valuation & Performance Tracking
```

### 1.2 Core Components to Test

| Component | What to Test | Why Critical |
|-----------|--------------|--------------|
| **HMM Training** | Walk-forward training windows | Prevents look-ahead bias |
| **Regime Detection** | Classification stability over time | Detects regime-switching logic flaws |
| **Allocation Optimizer** | Risk limit enforcement | Ensures limits are never violated |
| **Rebalancing Logic** | Trade generation, drift thresholds | Validates execution timing |
| **Transaction Costs** | Slippage, commissions, bid-ask | Realistic performance expectations |
| **Risk Management** | VaR, drawdowns, position limits | Portfolio safety validation |

### 1.3 Walk-Forward Validation Approach

CRITICAL: Must use walk-forward validation to prevent look-ahead bias.

```
Training Window: 5 years (1,260 trading days minimum for HMM)
Testing Window: 1 year (252 trading days)
Step Size: 6 months (126 trading days)

Timeline Example:
- Train: 2015-01 to 2020-01 → Test: 2020-01 to 2021-01
- Train: 2015-07 to 2020-07 → Test: 2020-07 to 2021-07
- Train: 2016-01 to 2021-01 → Test: 2021-01 to 2022-01
... continue through 2024-12
```

This ensures the model is NEVER trained on data from the future relative to the test period.

---

## 2. What Metrics to Track?

### 2.1 Performance Metrics

| Metric | Formula | Target | Red Flag |
|--------|---------|--------|----------|
| **Total Return** | (End Value - Start Value) / Start Value | > 5% annualized | < 0% |
| **Annualized Return** | (1 + Total Return)^(252/Days) - 1 | > 6% | < 3% |
| **Volatility** | Std(Daily Returns) * sqrt(252) | 10-15% | > 20% |
| **Sharpe Ratio** | (Return - RFR) / Volatility | > 1.0 | < 0.5 |
| **Sortino Ratio** | (Return - RFR) / Downside Deviation | > 1.2 | < 0.6 |
| **Max Drawdown** | Min(Portfolio / Peak - 1) | < -15% | < -30% |
| **Win Rate** | Positive Months / Total Months | > 55% | < 45% |
| **Profit Factor** | Gross Profit / Gross Loss | > 1.5 | < 1.1 |

### 2.2 Risk Metrics

| Metric | Calculation | Alert Threshold |
|--------|-------------|-----------------|
| **Value at Risk (95%)** | 5th percentile daily loss | > 3% |
| **Expected Shortfall (CVaR)** | Mean loss beyond VaR | > 4% |
| **Leveraged Exposure** | LQQ + CL2 weights | > 30% (HARD LIMIT) |
| **Single Position** | Max individual weight | > 25% (for LQQ/CL2) |
| **Cash Buffer** | CASH weight | < 10% (HARD LIMIT) |
| **Turnover** | Sum(abs(trades)) / Portfolio Value | > 200% annually |
| **Drawdown Duration** | Days from peak to recovery | > 365 days |

### 2.3 Regime-Specific Metrics

Track performance by regime to validate allocation logic:

| Regime | Expected Behavior | Validation |
|--------|-------------------|------------|
| **RISK_ON** | Higher returns, moderate volatility | Sharpe > 1.5 |
| **NEUTRAL** | Moderate returns, moderate volatility | Drawdown < -10% |
| **RISK_OFF** | Capital preservation, low volatility | Max DD < -5% |

### 2.4 Transaction Cost Analysis

| Component | Measurement |
|-----------|-------------|
| **Total Costs** | Sum of all transaction costs over backtest |
| **Cost per Trade** | Average transaction cost per rebalance |
| **Cost as % AUM** | Annual costs / Average portfolio value |
| **Cost Impact on Sharpe** | Sharpe(Gross Returns) - Sharpe(Net Returns) |

### 2.5 Statistical Validation Metrics

| Test | Purpose | Pass Criteria |
|------|---------|---------------|
| **t-test on Returns** | Verify returns > 0 are statistically significant | p-value < 0.05 |
| **Jarque-Bera Test** | Check return distribution normality | Document result |
| **Runs Test** | Check for autocorrelation in returns | p-value > 0.05 |
| **Regime Persistence** | Validate regime stability | Average duration > 20 days |

---

## 3. What Historical Periods to Test?

### 3.1 Minimum Data Requirements

Based on P0-04 HMM sample size requirements:
- **Minimum Training Period:** 7 years (1,700+ daily observations)
- **Minimum Testing Period:** 3 years (to capture regime changes)
- **Total Data Required:** 10+ years (2015-2025 or earlier if available)

### 3.2 Market Regime Coverage

The backtest MUST include diverse market conditions:

| Period | Market Regime | Key Events | Why Critical |
|--------|---------------|------------|--------------|
| **2015-2016** | Volatility spike | China devaluation, oil crash | Test RISK_OFF detection |
| **2017-2019** | Bull market | Low VIX, steady gains | Test RISK_ON allocation |
| **2020 Q1** | COVID crash | -35% S&P 500 drop | Test defensive positioning |
| **2020 Q2-2021** | Recovery rally | V-shaped recovery | Test regime switching |
| **2022** | Bear market | Rate hikes, inflation | Test RISK_OFF durability |
| **2023-2024** | Recovery | Tech rally, AI boom | Test RISK_ON re-entry |

### 3.3 Out-of-Sample Periods

Reserve the most recent period for final validation:
- **In-Sample:** 2015-01 to 2023-12 (9 years)
- **Out-of-Sample:** 2024-01 to 2024-12 (1 year, held out)

The out-of-sample period is NEVER used for parameter tuning. It serves as a final reality check.

### 3.4 Stress Test Scenarios

Additional scenario testing:
1. **2008 Financial Crisis** (if data available)
2. **2011 Euro Crisis** (if data available)
3. **COVID Crash (Feb-Mar 2020)** - isolated stress test
4. **2022 Drawdown** - isolated stress test

---

## 4. How to Handle Transaction Costs?

### 4.1 Transaction Cost Components

Realistic cost modeling is CRITICAL to avoid overstating performance.

| Cost Component | Assumption | Calculation |
|----------------|------------|-------------|
| **Commission** | 0 EUR (PEA, no broker commission) | 0 |
| **Bid-Ask Spread** | 0.05% for liquid ETFs | Trade Value * 0.0005 |
| **Slippage** | 0.03% (market orders) | Trade Value * 0.0003 |
| **Market Impact** | 0.02% (small retail orders) | Trade Value * 0.0002 |
| **Total Per Trade** | ~0.10% per side | Trade Value * 0.001 |

### 4.2 Execution Model

```python
def execute_trade(symbol: str, action: str, amount: float) -> float:
    """
    Simulate realistic trade execution with costs.

    Args:
        symbol: ETF symbol
        action: 'BUY' or 'SELL'
        amount: Trade amount in EUR

    Returns:
        Total transaction cost in EUR
    """
    # Base transaction cost: 0.10% (10 bps)
    base_cost_rate = 0.001

    # Slippage increases with trade size (negligible for retail)
    # For simplicity, use flat rate for small trades
    slippage_rate = 0.0003

    # Bid-ask spread cost
    spread_rate = 0.0005

    total_cost_rate = base_cost_rate + slippage_rate + spread_rate
    total_cost = amount * total_cost_rate

    return total_cost
```

### 4.3 Cost Model Assumptions

| Scenario | Cost Assumption | Rationale |
|----------|-----------------|-----------|
| **Normal Market** | 0.10% per trade | Typical for liquid French ETFs |
| **High Volatility** | 0.15% per trade | Wider spreads during stress |
| **Large Position** | 0.10% (no scaling) | Retail orders don't move market |
| **Cash Rebalancing** | 0% | No cost to hold or deploy cash |

### 4.4 Rebalancing Frequency Impact

Test multiple rebalancing frequencies:
- **Daily:** Highest costs, most responsive (NOT RECOMMENDED)
- **Weekly:** Moderate costs, balance responsiveness
- **Monthly:** Lower costs, lag in regime changes
- **Threshold-Based (5% drift):** Adaptive, minimize unnecessary trades

Expected turnover by frequency:
- Daily: 300-500% annually (excessive costs)
- Weekly: 100-200% annually (high costs)
- Monthly: 50-100% annually (moderate costs)
- **Threshold-Based: 30-80% annually (RECOMMENDED)**

### 4.5 Cost Monitoring in Backtest

Track costs in BacktestResult:
```python
class BacktestResult:
    total_trades: int
    total_transaction_costs: float  # EUR
    avg_cost_per_trade: float  # EUR
    costs_as_pct_aum: float  # Annual costs / AUM
    cost_drag_on_returns: float  # Return impact
    gross_return: float  # Before costs
    net_return: float  # After costs
```

---

## 5. Walk-Forward Validation Approach

### 5.1 Walk-Forward Methodology

```
Full Historical Dataset: 2015-01-01 to 2024-12-31 (10 years)

Step 1: Train on 2015-01 to 2020-01 (5 years)
        Test on 2020-01 to 2021-01 (1 year)

Step 2: Train on 2015-07 to 2020-07 (5 years)
        Test on 2020-07 to 2021-07 (1 year)

Step 3: Train on 2016-01 to 2021-01 (5 years)
        Test on 2021-01 to 2022-01 (1 year)

... continue with 6-month steps ...

Final: Train on 2019-01 to 2024-01 (5 years)
       Test on 2024-01 to 2024-12 (1 year, OUT-OF-SAMPLE)
```

### 5.2 Walk-Forward Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Training Window** | 5 years (1,260 days) | Minimum for HMM stability (per P0-04) |
| **Testing Window** | 1 year (252 days) | Sufficient for regime changes |
| **Step Size** | 6 months (126 days) | Balance between granularity and compute |
| **Minimum HMM Samples** | 1,700+ observations | Per P0-04 requirements |

### 5.3 Validation Process

For each walk-forward window:

1. **Train HMM** on training window
   - Calculate features for training period
   - Fit 3-state Gaussian HMM
   - Validate minimum sample requirements
   - Store model state-to-regime mapping

2. **Test on Forward Period**
   - Load trained HMM (no retraining)
   - Calculate features day-by-day
   - Predict regime each day
   - Generate allocations
   - Simulate rebalancing trades
   - Apply transaction costs
   - Track portfolio value

3. **Aggregate Results**
   - Combine all test windows
   - Calculate overall metrics
   - Check for parameter stability
   - Identify regime-dependent performance

### 5.4 Preventing Look-Ahead Bias

CRITICAL SAFEGUARDS:

1. **Feature Calculation:** Use only data available at time t for prediction at t
   - Moving averages: Use past data only
   - Volatility: Rolling window ending at t
   - Spreads: Use previous close, not intraday

2. **HMM Training:** Never train on test period data
   - Save trained model after training window
   - Load model for testing (frozen parameters)
   - No parameter updates during test period

3. **Allocation Decisions:** Use only known regime at time t
   - No future regime knowledge
   - No post-hoc optimization

4. **Trade Execution:** Simulate realistic timing
   - Signal at close day t
   - Execute at open day t+1 (or close t+1 conservatively)
   - Apply transaction costs

### 5.5 Parameter Stability Testing

After walk-forward backtest, analyze parameter sensitivity:

| Parameter | Test Range | Acceptable Variation |
|-----------|------------|----------------------|
| **HMM States** | 2, 3, 4 states | Performance change < 20% |
| **Rebalance Threshold** | 3%, 5%, 7% | Sharpe change < 0.2 |
| **Leveraged Exposure Limit** | 20%, 30%, 40% | Drawdown increase < 5% |
| **Cash Buffer** | 5%, 10%, 15% | Return change < 15% |

If results are highly sensitive to parameter choices, the strategy is likely overfit.

---

## 6. Implementation Architecture

### 6.1 Module Structure

```
src/backtesting/
    __init__.py
    engine.py              # Main backtest orchestration
    simulator.py           # Trade execution simulation
    metrics.py             # Performance metric calculations
    costs.py               # Transaction cost modeling
    walk_forward.py        # Walk-forward validation logic
    visualizations.py      # Equity curves, drawdown charts
```

### 6.2 Core Classes

#### 6.2.1 BacktestEngine

```python
class BacktestEngine:
    """Main backtesting engine with walk-forward validation."""

    def __init__(
        self,
        risk_calculator: RiskCalculator,
        allocation_optimizer: AllocationOptimizer,
        transaction_cost_model: TransactionCostModel,
    ):
        """Initialize backtest engine with components."""
        self.risk_calculator = risk_calculator
        self.allocation_optimizer = allocation_optimizer
        self.cost_model = transaction_cost_model

    def run_backtest(
        self,
        start_date: date,
        end_date: date,
        initial_capital: float,
        prices_history: dict[str, pd.Series],
        macro_data: pd.DataFrame,
        rebalance_frequency: str = "threshold",
    ) -> BacktestResult:
        """Execute full backtest with walk-forward validation."""
        pass

    def _run_walk_forward_window(
        self,
        train_start: date,
        train_end: date,
        test_start: date,
        test_end: date,
    ) -> WindowResult:
        """Run single walk-forward window."""
        pass
```

#### 6.2.2 TradeSimulator

```python
class TradeSimulator:
    """Simulate realistic trade execution with costs."""

    def execute_rebalance(
        self,
        current_positions: dict[str, float],  # Current holdings in EUR
        target_weights: dict[str, float],     # Target allocation weights
        current_prices: dict[str, float],     # Current prices
        portfolio_value: float,               # Total value
    ) -> RebalanceResult:
        """
        Simulate portfolio rebalancing with transaction costs.

        Returns:
            RebalanceResult with trades executed and costs incurred
        """
        pass

    def calculate_transaction_cost(
        self,
        symbol: str,
        action: str,
        amount: float,
        market_conditions: MarketConditions,
    ) -> float:
        """Calculate realistic transaction cost for a trade."""
        pass
```

#### 6.2.3 BacktestResult

```python
class BacktestResult(BaseModel):
    """Comprehensive backtest results."""

    # Period
    start_date: date
    end_date: date
    total_days: int

    # Performance
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # Days

    # Trade Statistics
    total_trades: int
    avg_trades_per_month: float
    win_rate: float
    profit_factor: float

    # Transaction Costs
    total_transaction_costs: float
    costs_as_pct_aum: float
    cost_drag_on_returns: float
    gross_return: float  # Before costs

    # Risk Metrics
    var_95: float
    expected_shortfall: float
    beta_to_benchmark: float | None
    tracking_error: float | None

    # Regime Analysis
    regime_distribution: dict[Regime, float]  # % time in each regime
    regime_performance: dict[Regime, RegimePerformance]

    # Time Series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    regime_history: pd.Series
    trade_log: list[Trade]

    # Walk-Forward Windows
    window_results: list[WindowResult]

    # Quality Checks
    look_ahead_bias_check: bool
    statistical_significance: dict[str, float]
    parameter_stability: dict[str, float]
```

#### 6.2.4 TransactionCostModel

```python
class TransactionCostModel:
    """Model for realistic transaction costs."""

    # Cost components (basis points)
    SPREAD_BPS = 5      # 0.05%
    SLIPPAGE_BPS = 3    # 0.03%
    IMPACT_BPS = 2      # 0.02%
    TOTAL_BPS = 10      # 0.10% per side

    def calculate_cost(
        self,
        trade_value: float,
        symbol: str,
        market_volatility: float,
    ) -> float:
        """Calculate total transaction cost for a trade."""
        base_cost = trade_value * (self.TOTAL_BPS / 10000)

        # Increase cost during high volatility
        if market_volatility > 0.30:  # 30% annualized vol
            volatility_adjustment = 1.5
        else:
            volatility_adjustment = 1.0

        total_cost = base_cost * volatility_adjustment
        return total_cost
```

### 6.3 Data Flow

```
1. Load Historical Data
   - ETF prices (LQQ, CL2, WPEA)
   - Macro indicators (VIX, DGS2, DGS10, etc.)
   - Benchmark (WPEA or custom)

2. Define Walk-Forward Windows
   - Training: 5 years
   - Testing: 1 year
   - Step: 6 months

3. For Each Window:
   a. Train HMM on training data
   b. Predict regimes on test data (day by day)
   c. Generate allocations based on regime
   d. Detect drift (5% threshold)
   e. Execute rebalancing trades (with costs)
   f. Update portfolio value
   g. Record metrics

4. Aggregate All Windows
   - Combine equity curves
   - Calculate overall metrics
   - Analyze regime-specific performance
   - Generate visualizations

5. Statistical Validation
   - t-test on returns
   - Parameter sensitivity analysis
   - Out-of-sample validation
```

---

## 7. Quality Assurance Checklist

### 7.1 Data Quality

- [ ] All price data aligned by date
- [ ] No look-ahead bias in features
- [ ] Corporate actions properly handled (splits, dividends)
- [ ] Survivorship bias avoided (N/A for ETFs, but document)
- [ ] Missing data handled consistently

### 7.2 Methodology Quality

- [ ] Walk-forward windows properly implemented
- [ ] HMM trained only on historical data
- [ ] No parameter tuning on out-of-sample period
- [ ] Realistic execution timing (t+1 or close-to-close)
- [ ] Transaction costs applied to every trade

### 7.3 Result Quality

- [ ] Sharpe ratio < 2.0 (if higher, suspect overfitting)
- [ ] Returns concentrated in multiple periods, not single event
- [ ] Drawdowns recoverable within 12 months
- [ ] Win rate between 45-65% (not 80%+)
- [ ] Results stable across walk-forward windows
- [ ] Out-of-sample performance consistent with in-sample

### 7.4 Red Flags

If ANY of these occur, investigate thoroughly:
- [ ] Sharpe ratio > 2.5 (likely overfit or biased)
- [ ] Max drawdown < -40% (strategy failed stress test)
- [ ] Win rate > 75% (unrealistic)
- [ ] Returns > 30% annually (too good to be true)
- [ ] All profits from single month/event
- [ ] Out-of-sample Sharpe < 50% of in-sample Sharpe
- [ ] Transaction costs ignored or < 0.05% per trade
- [ ] Parameter changes of 10% cause 50%+ performance swing

---

## 8. Expected Deliverables

### 8.1 Code Deliverables

1. `src/backtesting/engine.py` - Main backtest orchestration
2. `src/backtesting/simulator.py` - Trade execution simulator
3. `src/backtesting/metrics.py` - Performance metrics
4. `src/backtesting/costs.py` - Transaction cost modeling
5. `src/backtesting/walk_forward.py` - Walk-forward validation
6. `tests/test_backtesting/` - Comprehensive unit tests

### 8.2 Documentation Deliverables

1. **Backtest Report (Markdown)**
   - Executive summary
   - Methodology description
   - Performance metrics table
   - Equity curve chart
   - Drawdown chart
   - Regime distribution
   - Transaction cost breakdown
   - Statistical validation results
   - Conclusions and recommendations

2. **Jupyter Notebook (Optional)**
   - Interactive backtest exploration
   - Parameter sensitivity analysis
   - Regime-specific performance breakdown

### 8.3 Validation Deliverables

1. **Statistical Tests**
   - t-test on returns (p-value)
   - Jarque-Bera normality test
   - Runs test for autocorrelation
   - Parameter stability matrix

2. **Out-of-Sample Report**
   - 2024 performance vs. in-sample
   - Regime detection accuracy
   - Risk metric consistency

---

## 9. Acceptance Criteria

The backtesting framework is COMPLETE when:

1. **Functionality**
   - [ ] Walk-forward validation runs without errors
   - [ ] All performance metrics calculated correctly
   - [ ] Transaction costs applied to all trades
   - [ ] Equity curve and drawdown series generated
   - [ ] Regime history tracked throughout backtest

2. **Data Requirements**
   - [ ] Minimum 7 years of training data used
   - [ ] Minimum 3 years of testing data covered
   - [ ] Out-of-sample period (2024) reserved and tested

3. **Quality Checks**
   - [ ] No look-ahead bias detected
   - [ ] HMM sample size validation passed (1,700+ samples)
   - [ ] Risk limits enforced in 100% of test periods
   - [ ] Transaction costs realistic (0.10% per trade)

4. **Performance Validation**
   - [ ] Sharpe ratio < 2.0 (no overfitting)
   - [ ] Max drawdown > -40% (strategy survives stress)
   - [ ] Win rate between 45-70% (realistic)
   - [ ] Out-of-sample Sharpe > 70% of in-sample Sharpe

5. **Documentation**
   - [ ] Backtest report generated (Markdown)
   - [ ] Methodology documented
   - [ ] Results interpreted with caveats
   - [ ] Recommendations provided (Deploy / Don't Deploy / Iterate)

6. **Testing**
   - [ ] Unit tests for all backtesting components
   - [ ] Integration test for full backtest pipeline
   - [ ] Test coverage > 80% for backtesting module

---

## 10. Implementation Timeline

### Phase 1: Core Infrastructure (8 hours)
- [ ] Create `BacktestEngine` class
- [ ] Implement `TradeSimulator`
- [ ] Build `TransactionCostModel`
- [ ] Create `BacktestResult` model
- [ ] Write unit tests for components

### Phase 2: Walk-Forward Logic (8 hours)
- [ ] Implement walk-forward window generation
- [ ] Build HMM training/testing pipeline
- [ ] Integrate regime detection with allocation
- [ ] Implement rebalancing logic
- [ ] Test walk-forward on sample data

### Phase 3: Metrics & Validation (6 hours)
- [ ] Implement performance metric calculations
- [ ] Build statistical validation tests
- [ ] Create equity curve and drawdown series
- [ ] Generate regime-specific performance
- [ ] Test metrics against known benchmarks

### Phase 4: Full Backtest & Report (2 hours)
- [ ] Run full 10-year backtest
- [ ] Generate backtest report
- [ ] Analyze results
- [ ] Document findings and recommendations
- [ ] Integration test for complete pipeline

**Total Estimated Effort:** 24 hours

---

## 11. Risk Assessment

### 11.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient historical data | Medium | High | Use alternative data sources, reduce training window cautiously |
| HMM training instability | Medium | High | Add convergence monitoring, use robust initialization |
| Transaction cost uncertainty | Low | Medium | Test multiple cost scenarios (0.05%, 0.10%, 0.15%) |
| Computational time | Low | Low | Parallelize walk-forward windows |

### 11.2 Methodology Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Look-ahead bias | Low | Critical | Rigorous code review, explicit checks |
| Overfitting to recent data | Medium | High | Out-of-sample validation, parameter stability tests |
| Regime detection failure | Medium | High | Manual regime labeling validation |
| Unrealistic execution assumptions | Medium | Medium | Conservative cost assumptions, stress test |

### 11.3 Interpretation Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Over-confidence in results | High | High | Clear caveats in documentation, skeptical reporting |
| Ignoring regime-specific weakness | Medium | Medium | Detailed regime breakdown analysis |
| Dismissing transaction costs | Low | High | Emphasize cost impact in report |

---

## 12. Next Steps

### 12.1 Immediate Actions
1. Validate historical data availability (10 years of LQQ, CL2, WPEA, VIX, etc.)
2. Create `src/backtesting/` directory structure
3. Implement `TransactionCostModel` (simplest component first)
4. Write unit tests for cost model
5. Begin `BacktestEngine` implementation

### 12.2 Data Preparation
1. Fetch historical data from 2015-01-01 to 2024-12-31
2. Verify data quality (no gaps, splits handled, dividends adjusted)
3. Prepare macro indicators (VIX, DGS2, DGS10, BAA10Y, etc.)
4. Store in DuckDB with proper indexing

### 12.3 Integration with Existing Code
- Use existing `RegimeDetector` class (src/signals/regime.py)
- Use existing `AllocationOptimizer` class (src/signals/allocation.py)
- Use existing `RiskCalculator` class (src/portfolio/risk.py)
- Extend as needed for backtesting context

---

## 13. Success Definition

The backtesting framework is SUCCESSFUL if:

1. **Validation Quality:** No look-ahead bias, realistic costs, proper walk-forward
2. **Performance Realism:** Sharpe < 2.0, drawdown > -40%, win rate 45-70%
3. **Statistical Significance:** Returns significantly > 0 (p < 0.05)
4. **Parameter Stability:** Results stable across reasonable parameter variations
5. **Out-of-Sample Consistency:** 2024 performance consistent with historical
6. **Risk Management:** All risk limits enforced 100% of the time
7. **Actionable Insights:** Clear recommendation to deploy, iterate, or abandon

---

## Appendix A: Sample BacktestResult Output

```
==================================================
BACKTEST REPORT - PEA Portfolio Optimization
==================================================

Period: 2015-01-01 to 2024-12-31 (10 years)
Initial Capital: 10,000 EUR
Rebalancing: Threshold-based (5% drift)

--------------------------------------------------
PERFORMANCE SUMMARY
--------------------------------------------------
Total Return:              87.3%
Annualized Return:         6.5%
Annualized Volatility:     12.1%
Sharpe Ratio:              1.24
Sortino Ratio:             1.58
Max Drawdown:              -18.7% (2020-03-16 to 2020-05-12)
Drawdown Duration:         57 days

--------------------------------------------------
RISK METRICS
--------------------------------------------------
Value at Risk (95%):       2.1%
Expected Shortfall:        3.4%
Beta to WPEA:              0.87
Win Rate:                  58.3%
Profit Factor:             1.67

--------------------------------------------------
TRANSACTION COSTS
--------------------------------------------------
Total Trades:              142
Avg Trades per Month:      1.2
Total Transaction Costs:   187.50 EUR
Costs as % AUM:            0.19%
Cost Drag on Returns:      -0.15% annually
Gross Return:              6.65% annually
Net Return:                6.50% annually

--------------------------------------------------
REGIME ANALYSIS
--------------------------------------------------
RISK_ON:    42% of days, Return: +9.2% annualized
NEUTRAL:    35% of days, Return: +5.1% annualized
RISK_OFF:   23% of days, Return: +2.3% annualized

--------------------------------------------------
QUALITY CHECKS
--------------------------------------------------
Look-ahead Bias:           PASS
HMM Sample Size:           PASS (1,730+ samples)
Risk Limit Enforcement:    100% compliance
Statistical Significance:  p = 0.003 (returns > 0)
Parameter Stability:       PASS (Sharpe stable ± 0.15)

--------------------------------------------------
OUT-OF-SAMPLE (2024)
--------------------------------------------------
Return:                    7.1% (vs 6.5% in-sample)
Sharpe:                    1.18 (vs 1.24 in-sample)
Max Drawdown:              -9.2% (vs -18.7% in-sample)

--------------------------------------------------
RECOMMENDATION
--------------------------------------------------
CONDITIONAL DEPLOY

The strategy demonstrates:
+ Positive risk-adjusted returns (Sharpe > 1.0)
+ Effective risk management (max DD < -20%)
+ Realistic transaction costs (< 0.20% annually)
+ Out-of-sample consistency
+ No evidence of overfitting

Concerns:
- Performance during prolonged bear markets untested
- Leveraged ETF decay not fully captured (need 2008 data)
- Regime detection lag during rapid transitions

Next Steps:
1. Paper trade for 3 months to validate live execution
2. Stress test with 2008 data if available
3. Implement pre-trade risk checks
4. Monitor regime detection accuracy in real-time

==================================================
```

---

**Document End**

This design document should be reviewed by:
- Jean-Yves (Research Lead) - Methodology validation
- Helena (Execution Manager) - Execution realism
- Nicolas (Risk Manager) - Risk metric coverage
- Sophie (Data Lead) - Data requirements feasibility

Approval required before implementation begins.
