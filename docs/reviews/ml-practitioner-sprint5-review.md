# ML Practitioner Review - Sprint 5 P0 Implementation

**Reviewer:** Pierre-Jean (Data Team - ML Practitioner)
**Date:** December 12, 2025
**Commit:** 74ca951 (Sprint 5 P0 - HMM fixes)
**Status:** SOLID FOUNDATION - Ready for P1/P2 with guidance

---

## Executive Summary

The P0 HMM implementation is **statistically sound and production-ready from an ML perspective**. The parameter counting, sample size validation, and covariance type handling are rigorous and follow established ML best practices. This is exactly the kind of careful work that prevents the "looks good but doesn't work" problem common in applied ML.

**Grade: A- (Excellent foundation, needs hyperparameter tuning)**

### Key Strengths
1. Rigorous parameter counting for all covariance types
2. Enforced 10:1 sample-to-parameter ratio (minimum 1,700 samples)
3. Feature standardization for numerical stability
4. Proper model serialization with joblib + JSON
5. Clear separation of concerns (features → HMM → allocation)

### Critical Next Steps (P1/P2)
1. Hyperparameter tuning for covariance type and state count
2. Walk-forward cross-validation for realistic performance
3. Feature selection and engineering validation
4. Training window size optimization
5. Backtesting with proper out-of-sample testing

---

## 1. Sample Size Validation - APPROACH IS SOUND

### What You Did Right

The parameter counting is **mathematically correct** and comprehensive:

```python
def calculate_hmm_parameters(n_states, n_features, covariance_type):
    n_init_params = n_states - 1                    # Initial distribution
    n_trans_params = n_states * (n_states - 1)      # Transition matrix
    n_mean_params = n_states * n_features           # Gaussian means

    # Covariance (depends on type)
    if covariance_type == "full":
        n_cov_params = n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        n_cov_params = n_states * n_features
    # ... etc
```

For your default config (3 states, 9 features, full covariance):
- Initial: 2 parameters
- Transitions: 6 parameters (3×2)
- Means: 27 parameters (3×9)
- Covariances: 135 parameters (3 × 9×10/2)
- **Total: 170 parameters**

With 10:1 ratio → 1,700 samples minimum (6.7 years daily data).

### Why This Matters

In my experience, HMMs are **particularly sensitive to small sample sizes** because:
1. EM algorithm can get stuck in local optima with limited data
2. Transition probabilities become noisy (regime persistence uncertain)
3. Covariance estimation is unstable with high-dimensional features
4. Overfitting to noise instead of capturing true market dynamics

Your 10:1 ratio is appropriate. I've seen production systems use 20:1 for critical applications.

### Practical Validation

**The 1,700 sample minimum is justified, but test it empirically:**

```python
# Suggested validation experiment (P1-05 "Multi-feature regime mapping")
# Test model stability with different training windows

training_windows = [2, 3, 5, 7, 10]  # years
for window_years in training_windows:
    n_samples = window_years * 252

    # Train multiple models with different random seeds
    regimes_by_seed = []
    for seed in range(10):
        detector = RegimeDetector(random_state=seed)
        detector.fit(features[-n_samples:])
        regimes = detector.predict_regime(test_features)
        regimes_by_seed.append(regimes)

    # Measure stability: do different seeds give similar regimes?
    stability = calculate_agreement(regimes_by_seed)
    print(f"{window_years}y: stability = {stability:.2%}")
```

**What to look for:**
- Stability should increase with more data
- If stability plateaus at 5 years → you can relax to 5y minimum
- If still unstable at 7 years → you need 10+ years or simpler model

---

## 2. Hyperparameter Tuning - CRITICAL FOR P1

### Current State: Untested Defaults

You're using sensible defaults, but they're **unvalidated**:

```python
n_states=3                # Why 3? Could be 2 or 4
covariance_type="full"    # Most flexible, but highest parameter count
n_iterations=100          # May converge earlier or need more
random_state=42           # Good for reproducibility
```

### What Needs Tuning (P1-05)

**Priority 1: Covariance Type (biggest impact)**

The covariance type massively affects parameter count and model capacity:

| Type | Parameters (3 states, 9 features) | Min Samples (10:1) | Training Years |
|------|-----------------------------------|---------------------|----------------|
| spherical | 35 | 350 | 1.4y |
| diag | 89 | 890 | 3.5y |
| full | 170 | 1,700 | 6.7y |
| tied | 53 | 530 | 2.1y |

**My recommendation: Start with "diag"**

Reasons:
1. **Diagonal covariance assumes feature independence** - reasonable for your features (VIX, trend, spreads are somewhat orthogonal)
2. **3.5 years of data is achievable** - easier to gather and test
3. **Reduces overfitting risk** - fewer parameters = more robust
4. **Faster training** - important for backtesting iterations

How to test:
```python
# P1-05: Compare covariance types with cross-validation
for cov_type in ["spherical", "diag", "full", "tied"]:
    config = RegimeDetectorConfig(covariance_type=cov_type)

    # Walk-forward validation (see backtesting section)
    log_likelihoods = []
    regime_transitions = []

    for train_window, test_window in walk_forward_splits(features):
        detector = RegimeDetector(config=config)
        detector.fit(train_window)

        # Out-of-sample log-likelihood
        ll = detector._model.score(test_window)
        log_likelihoods.append(ll)

        # Regime stability (penalize excessive switching)
        regimes = detector.predict_regime(test_window)
        transitions = count_transitions(regimes)
        regime_transitions.append(transitions)

    print(f"{cov_type}: avg_ll={np.mean(log_likelihoods):.2f}, "
          f"avg_transitions={np.mean(regime_transitions):.1f}")
```

**What you're looking for:**
- Higher log-likelihood = better fit to out-of-sample data
- Moderate transition count (5-10 per year is reasonable for macro regimes)
- Too many transitions = model is noisy
- Too few transitions = model is stuck

**Priority 2: Number of States**

You assume 3 states (RISK_ON, NEUTRAL, RISK_OFF), but this is a modeling choice:

```python
# Test 2-5 states with BIC for model selection
for n_states in range(2, 6):
    detector = RegimeDetector(n_states=n_states, covariance_type="diag")
    detector.fit(training_features)

    # BIC penalizes model complexity
    bic = calculate_bic(detector._model, training_features)

    # Analyze state characteristics
    if n_states >= 3:
        state_chars = detector.get_state_characteristics()
        print(f"{n_states} states: BIC={bic:.0f}")
        for regime, chars in state_chars.items():
            print(f"  {regime}: VIX={chars['feature_0']:.1f}")
```

**My experience says 3 states is right for macro regimes**, but you should validate:
- 2 states: Too coarse (bull/bear only)
- 3 states: Standard regime framework (RISK_ON/NEUTRAL/RISK_OFF)
- 4+ states: May discover sub-regimes (e.g., "crash" as distinct from "risk_off")

**Priority 3: Training Window Length**

Even with sufficient samples (1,700+), you need to decide **how much history to use**:

```python
# Test expanding vs rolling windows
# Expanding: Use all available history (1970-present)
# Rolling: Use last N years only (e.g., last 10 years)

# Hypothesis: Recent data may be more relevant (market structure changes)
window_lengths = [5, 7, 10, 15, "all"]

for window in window_lengths:
    if window == "all":
        train_features = all_features
    else:
        train_features = all_features[-(window * 252):]

    # Test on recent out-of-sample period
    detector = RegimeDetector(covariance_type="diag")
    detector.fit(train_features)

    # Evaluate on last 2 years (held out)
    test_regimes = detector.predict_regime(test_features)
    test_allocations = [get_target_allocation(r) for r in test_regimes]
    test_returns = backtest_allocations(test_allocations, test_prices)

    print(f"{window}y window: Sharpe={calculate_sharpe(test_returns):.2f}")
```

**My recommendation: Start with 10 years rolling**
- Captures multiple market cycles (2008 crisis, 2020 COVID, 2022 inflation)
- Not so long that it includes irrelevant regimes (e.g., 1970s stagflation)
- Enough data for stable estimation with diagonal covariance

---

## 3. Backtesting Structure for ML Validation (P0-07)

### The Challenge: Temporal Dependence

Standard cross-validation **DOES NOT WORK** for time series. Why?
- Random splits leak future information into training
- HMM learns temporal dynamics (transition probabilities)
- Financial regimes are persistent (autocorrelation)

### What You Need: Walk-Forward Validation

**This is the gold standard for validating trading strategies:**

```python
# Backtesting framework structure (P0-07)

class WalkForwardValidator:
    """Walk-forward cross-validation for HMM regime detector."""

    def __init__(
        self,
        initial_training_years: int = 7,
        test_period_months: int = 3,
        retrain_frequency_months: int = 3,
    ):
        """
        Args:
            initial_training_years: Years of data for first training
            test_period_months: Length of each out-of-sample test period
            retrain_frequency_months: How often to retrain model
        """
        self.initial_training_years = initial_training_years
        self.test_period_months = test_period_months
        self.retrain_frequency_months = retrain_frequency_months

    def run(
        self,
        features: np.ndarray,
        prices: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> BacktestResult:
        """Execute walk-forward validation.

        Timeline:
        [-------Training (7y)-------][Test 3m][Train+3m][Test 3m]...

        Each test period:
        1. Train HMM on all data up to test start
        2. Predict regimes for test period (no peeking!)
        3. Generate allocations based on predicted regimes
        4. Simulate trades and calculate returns
        5. Roll forward and repeat
        """
        results = []

        # Initial training period
        train_start_idx = 0
        train_end_idx = self.initial_training_years * 252

        current_idx = train_end_idx

        while current_idx < len(features):
            # Define test window
            test_start_idx = current_idx
            test_end_idx = min(
                current_idx + self.test_period_months * 21,  # ~21 trading days/month
                len(features)
            )

            # Train on all data up to test period
            train_features = features[train_start_idx:test_start_idx]
            test_features = features[test_start_idx:test_end_idx]

            # Fit HMM
            detector = RegimeDetector(covariance_type="diag")
            detector.fit(train_features)

            # Predict test regimes (day by day, as if real-time)
            test_regimes = []
            for i in range(len(test_features)):
                # Use all features up to day i for prediction
                regime = detector.predict_regime(test_features[:i+1])
                test_regimes.append(regime)

            # Generate allocations and simulate trades
            period_result = self._simulate_period(
                regimes=test_regimes,
                prices=prices.iloc[test_start_idx:test_end_idx],
            )
            results.append(period_result)

            # Move to next period
            current_idx = test_end_idx

            # Optionally retrain more frequently
            # (keep expanding training window)

        return self._aggregate_results(results)
```

### Key Implementation Details

**1. Simulation Realism (Critical!)**

```python
def _simulate_period(self, regimes, prices):
    """Simulate trading for a test period."""

    portfolio_value = [100_000.0]  # Start with 100k EUR
    positions = {"LQQ": 0, "CL2": 0, "WPEA": 0, "CASH": 100_000.0}
    trades = []

    for day, regime in enumerate(regimes):
        # Get target allocation for detected regime
        target_allocation = get_target_allocation(regime)

        # Calculate required trades
        current_value = calculate_portfolio_value(positions, prices.iloc[day])
        target_positions = calculate_target_positions(
            target_allocation, current_value, prices.iloc[day]
        )

        # Apply transaction costs (ESSENTIAL)
        trades_today = []
        for symbol in ["LQQ", "CL2", "WPEA"]:
            if abs(target_positions[symbol] - positions[symbol]) > MIN_TRADE_SIZE:
                trade = Trade(
                    symbol=symbol,
                    quantity=target_positions[symbol] - positions[symbol],
                    price=prices.iloc[day][symbol],
                    commission=0.1,  # 10 cents per trade (Boursorama)
                    spread=0.05,     # 5 bps spread estimate
                )
                trades_today.append(trade)

                # Deduct costs from cash
                cost = trade.quantity * trade.price * (1 + trade.spread/100)
                cost += trade.commission
                positions["CASH"] -= cost
                positions[symbol] += trade.quantity

        trades.extend(trades_today)

        # Mark to market
        portfolio_value.append(
            calculate_portfolio_value(positions, prices.iloc[day])
        )

    return PeriodResult(
        portfolio_value=portfolio_value,
        trades=trades,
        regimes=regimes,
    )
```

**What makes this realistic:**
- Transaction costs (commissions + spreads)
- Minimum trade size (don't rebalance for tiny drifts)
- Execution at next-day open (no look-ahead bias)
- Cash buffer constraints
- Regime detection lag (uses features up to day t to predict day t+1)

**2. Performance Metrics (aligned with risk.py)**

```python
class BacktestResult(BaseModel):
    """Complete backtest results with performance metrics."""

    # Returns
    total_return: float
    annualized_return: float

    # Risk metrics (use your existing risk.py calculations)
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trading metrics
    total_trades: int
    total_transaction_costs: Decimal
    turnover_annual: float  # Portfolio turnover per year
    avg_holding_period_days: float

    # Regime metrics
    regime_distribution: dict[Regime, float]  # % time in each regime
    regime_transition_count: int  # Total regime changes
    avg_days_per_regime: float

    # Comparison to benchmarks
    benchmark_return: float  # 100% WPEA
    information_ratio: float  # (return - benchmark) / tracking_error

    # Time series
    equity_curve: list[EquityPoint]
    regime_sequence: list[Regime]
    trades: list[Trade]
```

**3. Out-of-Sample Testing (CRITICAL)**

```python
# Reserve last 2 years (2023-2024) as final validation set
# NEVER train on this data during development!

FINAL_TEST_START = date(2023, 1, 1)
FINAL_TEST_END = date(2024, 12, 31)

# Use 2015-2022 for development
# - Walk-forward validation
# - Hyperparameter tuning
# - Feature selection

# Only after ALL decisions are made, run on 2023-2024
# This gives you an unbiased estimate of future performance
```

### Expected Outcomes (Reality Check)

Based on my experience with similar regime-switching strategies:

**Realistic targets for HMM-based allocation:**
- Annualized return: 8-12% (vs 10% for buy-and-hold WPEA)
- Sharpe ratio: 0.7-1.0 (vs 0.5-0.6 for buy-and-hold)
- Max drawdown: -15% to -25% (vs -30%+ for buy-and-hold)
- Win rate: 50-60% (regime detection is noisy)

**Red flags:**
- Sharpe > 1.5 → Overfitting or unrealistic assumptions
- Max drawdown < -10% → Not enough historical coverage (need 2008, 2020)
- Turnover > 50x/year → Excessive regime switching (transaction costs will kill you)
- Returns > 15% annualized → Leverage, overfitting, or luck

**If your backtest shows Sharpe > 2.0 or returns > 20%:**
1. Check for look-ahead bias (using future data in features)
2. Verify transaction costs are included
3. Test on longer out-of-sample period
4. Reduce leverage (your 30% leveraged ETF cap is good)

---

## 4. Feature Engineering Validation (P1-05)

### Current Features (9 total)

You have a well-thought-out feature set:

**Volatility (3 features):**
- vix_level: Implied volatility (forward-looking)
- vix_percentile_20d: Relative VIX level (regime context)
- realized_vol_20d: Historical volatility (backward-looking)

**Trend (3 features):**
- price_vs_ma200: Long-term trend (price / 200-day MA)
- ma50_vs_ma200: Momentum (golden/death cross)
- momentum_3m: Recent return (3-month)

**Macro/Credit (3 features):**
- yield_curve_slope: 10Y-2Y spread (recession indicator)
- hy_spread: High-yield credit spread (risk appetite)
- hy_spread_change_1m: Credit trend (improving/deteriorating)

### What's Good

1. **Diverse signal types**: Vol, trend, macro (not all correlated)
2. **Mix of levels and changes**: VIX level + VIX percentile captures both absolute and relative
3. **Forward and backward looking**: VIX (implied) vs realized vol
4. **Standard indicators**: These are well-studied in literature

### What to Test (P1-05)

**1. Feature Importance via Ablation Study**

```python
# Which features actually matter for regime detection?

all_features = [
    "vix_level", "vix_percentile_20d", "realized_vol_20d",
    "price_vs_ma200", "ma50_vs_ma200", "momentum_3m",
    "yield_curve_slope", "hy_spread", "hy_spread_change_1m"
]

baseline_sharpe = run_backtest(features=all_features)

# Drop each feature one at a time
for dropped in all_features:
    features_subset = [f for f in all_features if f != dropped]
    sharpe = run_backtest(features=features_subset)
    importance = baseline_sharpe - sharpe  # Positive = feature was helpful
    print(f"Drop {dropped}: Δ Sharpe = {importance:+.3f}")
```

**My hypothesis (to be tested):**
- **VIX level** will be most important (volatility regime is primary)
- **yield_curve_slope** will be second (macro regime)
- **momentum_3m** may be least important (noisy, overlaps with trend)

**If a feature has near-zero importance → drop it** (reduces overfitting)

**2. Feature Correlation Analysis**

```python
# Check for redundant features
feature_array = compute_feature_array(all_dates)
correlation_matrix = np.corrcoef(feature_array.T)

# Look for high correlations (|r| > 0.7)
# Example: price_vs_ma200 and ma50_vs_ma200 might be redundant
# Example: vix_level and realized_vol_20d might be correlated

# If correlated, keep the one that's easier to calculate or more robust
```

**3. Feature Engineering Ideas (if performance is weak)**

Additional features to consider:
- **VIX term structure**: VIX1M - VIX3M (contango/backwardation)
- **Credit momentum**: 3-month change in HY spread (better than 1-month?)
- **Put/call ratio**: Investor sentiment (options market)
- **Commodity volatility**: Oil or gold vol (inflation/crisis signal)

But **test one at a time** and validate improvement in out-of-sample backtest.

---

## 5. Training Process Refinements (P1)

### Current Fit Method

Your fit method is solid but could use production enhancements:

```python
def fit(self, features, *, skip_sample_validation=False):
    # Validation ✓
    # Standardization ✓
    # HMM fitting ✓
    # State mapping ✓

    # Missing: Convergence diagnostics
    # Missing: Stability checks
```

### Recommended Enhancements

**1. Convergence Monitoring**

```python
def fit(self, features, *, skip_sample_validation=False):
    # ... existing code ...

    self._model.fit(features_standardized)

    # Check convergence
    if not self._model.monitor_.converged:
        logger.warning(
            f"HMM did not converge after {self.config.n_iterations} iterations. "
            f"Final log-likelihood: {self._model.monitor_.history[-1]:.2f}. "
            f"Consider increasing n_iterations or simplifying model."
        )

    # Log convergence info
    logger.info(
        f"HMM converged in {len(self._model.monitor_.history)} iterations. "
        f"Log-likelihood: {self._model.score(features_standardized):.2f}"
    )

    # Check for degenerate states (all prob in one state)
    stationary_dist = self.get_stationary_distribution()
    for regime, prob in stationary_dist.items():
        if prob < 0.05:
            logger.warning(
                f"Regime {regime} has very low stationary probability ({prob:.1%}). "
                "Model may have degenerate states."
            )
```

**2. Feature Scaling Diagnostics**

```python
# After standardization, check for issues
if np.any(np.abs(features_standardized) > 10):
    logger.warning(
        "Standardized features have extreme values (|z| > 10). "
        "Possible outliers or non-stationary data."
    )

# Log feature statistics for debugging
for i, name in enumerate(FeatureSet.feature_names()):
    logger.debug(
        f"{name}: mean={self._feature_means[i]:.3f}, "
        f"std={self._feature_stds[i]:.3f}"
    )
```

**3. State Interpretation**

```python
def fit(self, features, skip_sample_validation=False):
    # ... existing code ...

    # After mapping states to regimes, log characteristics
    state_chars = self.get_state_characteristics()
    for regime, chars in state_chars.items():
        logger.info(f"{regime} characteristics:")
        logger.info(f"  VIX level: {chars['feature_0']:.1f}")
        logger.info(f"  Yield curve: {chars['feature_6']:.2f}%")
        logger.info(f"  HY spread: {chars['feature_7']:.2f}%")
```

This helps you **understand what the model learned** and catch problems:
- If RISK_ON has high VIX → state mapping is wrong
- If RISK_OFF has positive yield curve → may need more data

---

## 6. Production Monitoring (P2)

Once deployed, you need to monitor model health:

### 1. Regime Persistence Check

```python
# Regimes should persist for weeks/months, not flip daily

def check_regime_stability(recent_regimes: list[Regime], window_days: int = 20):
    """Alert if excessive regime switching."""
    switches = sum(1 for i in range(1, len(recent_regimes))
                   if recent_regimes[i] != recent_regimes[i-1])

    if switches > window_days * 0.3:  # More than 30% of days are switches
        logger.warning(
            f"Excessive regime switching: {switches} switches in {window_days} days. "
            "Model may be unstable or features are noisy."
        )
```

### 2. Prediction Confidence

```python
# Low confidence → use more conservative allocation

regime_probs = detector.predict_regime_probabilities(latest_features)
max_prob = max(regime_probs.values())

if max_prob < 0.6:
    logger.warning(
        f"Low regime confidence: {max_prob:.1%}. "
        "Consider using NEUTRAL allocation or increasing cash buffer."
    )
```

### 3. Feature Drift Detection

```python
# Check if recent features are within historical bounds

recent_features = features[-252:]  # Last year
historical_features = features[:-252]  # All prior history

for i, name in enumerate(FeatureSet.feature_names()):
    recent_mean = np.mean(recent_features[:, i])
    hist_mean = np.mean(historical_features[:, i])
    hist_std = np.std(historical_features[:, i])

    z_score = (recent_mean - hist_mean) / hist_std

    if abs(z_score) > 2:
        logger.warning(
            f"Feature drift detected: {name} recent mean is {z_score:.1f} "
            "std devs from historical mean. Model may need retraining."
        )
```

---

## 7. Recommended P1/P2 Implementation Order

Based on ML best practices and your Sprint 5 roadmap:

### Week 2: Backtesting Framework (P0-07) - 24 hours
**Priority: CRITICAL - Can't validate anything without this**

1. Implement WalkForwardValidator class (8h)
2. Add transaction cost modeling (4h)
3. Implement performance metrics (use existing risk.py) (4h)
4. Create equity curve and trade log (4h)
5. Test on 2015-2022 data (4h)

**Deliverable**: Backtest showing Sharpe ratio, max drawdown, turnover on out-of-sample data

### Week 3: Hyperparameter Tuning (P1-05) - 12 hours
**Priority: HIGH - Significantly impacts performance**

1. Test covariance types (spherical, diag, full, tied) (4h)
2. Test n_states (2-5) with BIC (3h)
3. Test training window lengths (5y, 7y, 10y, all) (3h)
4. Select best config based on out-of-sample Sharpe (2h)

**Deliverable**: Tuned HMM config with documented rationale

### Week 3: Feature Selection (P1-05) - 8 hours
**Priority: MEDIUM - May improve or simplify**

1. Run ablation study (drop each feature) (4h)
2. Analyze feature correlations (2h)
3. Test 1-2 new features if performance is weak (2h)

**Deliverable**: Reduced feature set (if beneficial) or validation of current set

### Week 4: Cross-Validation (P2-01) - 8 hours
**Priority: MEDIUM - Validates robustness**

1. Implement k-fold time-series cross-validation (4h)
2. Test model on 5 different out-of-sample periods (2h)
3. Calculate confidence intervals for Sharpe ratio (2h)

**Deliverable**: Mean Sharpe ± std dev across folds

### Week 4: Monitoring (P2-03) - 8 hours
**Priority: LOW - Important for production, but not for validation**

1. Add convergence logging to fit() (2h)
2. Implement regime stability checks (2h)
3. Add feature drift detection (2h)
4. Create model health dashboard (2h)

**Deliverable**: Automated alerts for model degradation

---

## 8. Common Pitfalls to Avoid

### Pitfall 1: Look-Ahead Bias

**DON'T:**
```python
# WRONG: Using same-day features to predict same-day regime
features_today = calculate_features(today)
regime_today = detector.predict_regime(features_today)
allocation_today = get_allocation(regime_today)
execute_trades(allocation_today, today)  # Executes at today's close
```

**DO:**
```python
# CORRECT: Use yesterday's features to predict today's regime
features_yesterday = calculate_features(yesterday)
regime_today = detector.predict_regime(features_yesterday)
allocation_today = get_allocation(regime_today)
execute_trades(allocation_today, today_open)  # Executes at today's open
```

### Pitfall 2: Overfitting to Specific Events

**Problem**: If you tune hyperparameters to maximize Sharpe on 2008-2020 data, you might overfit to 2008 crisis recovery.

**Solution**: Use multiple non-overlapping test periods:
- 2008-2010 (Financial crisis)
- 2011-2013 (Euro crisis)
- 2014-2016 (Oil crash)
- 2017-2019 (Low vol bull market)
- 2020-2022 (COVID + inflation)

Model should be **reasonably stable across all periods**, not perfect on one.

### Pitfall 3: Ignoring Transaction Costs

**Reality check**: Leveraged ETFs (LQQ, CL2) have wider spreads than vanilla ETFs.

**Estimate conservative costs:**
- Commission: 0.10 EUR per trade (Boursorama)
- Spread: 5-10 bps for leveraged ETFs, 2-5 bps for WPEA
- Slippage: 1-2 bps (small personal account, negligible)

**If your strategy rebalances weekly**:
- 3 ETFs × 2 trades/week × 52 weeks = 312 trades/year
- At 0.10 EUR/trade = 31.20 EUR/year
- On 100k portfolio = 0.03% cost (negligible)

**But if you have 20% annual turnover on 100k**:
- 20k traded × 5 bps spread = 100 EUR cost
- 0.10% drag on returns (significant!)

### Pitfall 4: Overly Frequent Retraining

**Question**: How often should you retrain the HMM?

**My recommendation**: Quarterly retraining is sufficient
- Regimes change slowly (months/years)
- Retraining too frequently introduces instability
- EM algorithm can give different results with small data changes

**Test this empirically**:
```python
retrain_frequencies = ["monthly", "quarterly", "annually", "never"]
for freq in retrain_frequencies:
    backtest_sharpe = run_backtest(retrain_frequency=freq)
    print(f"{freq}: Sharpe = {backtest_sharpe:.2f}")
```

---

## 9. Success Criteria for Sprint 5 ML Work

### Minimum Viable Validation (P0/P1)

**You're ready for production if:**
1. ✅ Backtest covers 5+ years including 2008 and 2020 crises
2. ✅ Out-of-sample Sharpe ratio > 0.5 (better than random)
3. ✅ Max drawdown < -30% (comparable to buy-and-hold)
4. ✅ Regime transitions 5-15 per year (not too noisy, not stuck)
5. ✅ Model converges in <50 iterations (stable)
6. ✅ Hyperparameters chosen via systematic cross-validation

### Target Performance (Realistic)

**Good strategy:**
- Sharpe ratio: 0.7-1.0
- Max drawdown: -20% to -25%
- Annual turnover: 30-50%
- Win rate: 55-60%

**Excellent strategy:**
- Sharpe ratio: 1.0-1.3
- Max drawdown: -15% to -20%
- Annual turnover: 20-30%
- Win rate: 60-65%

**Too good to be true:**
- Sharpe ratio > 1.5 → Check for overfitting
- Max drawdown < -15% → Need more historical coverage
- Win rate > 70% → Probably overfitting or look-ahead bias

---

## 10. Final ML Practitioner Advice

### What You've Done Right

1. **Rigorous parameter counting** - Rare to see this done correctly
2. **Proper model serialization** - joblib + JSON instead of pickle
3. **Feature standardization** - Prevents numerical instability
4. **Clear separation of concerns** - features → HMM → allocation

### What to Focus On (P1/P2)

1. **Backtesting is everything** - You can't trust the model without it
2. **Start simple, add complexity only if needed** - Try diagonal covariance first
3. **Test on multiple regimes** - 2008, 2020, and low-vol periods
4. **Be skeptical of great results** - If Sharpe > 1.5, something is probably wrong

### Red Flags to Watch For

1. **Model doesn't converge** → Reduce complexity or increase iterations
2. **All probability in one state** → Degenerate model, need more data or different init
3. **Excessive regime switching** → Features are too noisy or model is overfit
4. **Great in-sample, poor out-of-sample** → Classic overfitting

### The ML Practitioner Mindset

In production ML, **simplicity and robustness beat complexity and overfitting**:
- Diagonal covariance that's stable > Full covariance that overfits
- 3-state model that generalizes > 5-state model that memorizes
- 7 important features > 15 features with redundancy

**Your goal**: Build a model that works reasonably well in **all market conditions**, not perfectly in historical backtests.

---

## Conclusion

The P0 HMM implementation is **excellent foundational work**. The parameter counting, sample size validation, and code structure are production-quality.

**Next steps (in order):**
1. Build backtesting framework (P0-07) - Can't proceed without this
2. Run hyperparameter tuning with walk-forward validation (P1-05)
3. Test feature importance and drop weak features (P1-05)
4. Validate on held-out 2023-2024 data (final reality check)

**Timeline:** 2-3 weeks for complete validation
**Outcome:** Realistic assessment of strategy performance with confidence intervals

You're on the right track. Let's get that backtesting framework up and running so we can see if this HMM actually makes money out-of-sample.

---

**Contact:** Pierre-Jean (Data Team)
**Next Review:** After P0-07 (Backtesting) is complete
