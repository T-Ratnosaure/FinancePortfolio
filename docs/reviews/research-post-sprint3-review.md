# Research Team Post-Sprint 3 Review

**Date:** December 10, 2025
**Document Type:** Post-Sprint 3 Review
**Author:** Jean-Yves, Portfolio Manager and Research Team Director
**Contributors:** Remy (Equity Quant), Iacopo (Macro Analyst), Alexios (ML Designer)

---

## Executive Summary

This document presents a comprehensive quantitative review of the FinancePortfolio project following Sprint 3. The review evaluates the mathematical correctness of financial calculations, model validity, statistical assumptions, and potential risks inherent in the current implementation.

**Overall Assessment: APPROVED WITH RESERVATIONS**

| Component | Rating | Status |
|-----------|--------|--------|
| HMM Regime Detection | B+ | Adequate with improvements needed |
| Risk Calculations | A- | Well-implemented |
| Allocation Optimization | B | Conservative and sound |
| Feature Engineering | B+ | Good foundation |
| Backtesting Framework | C+ | Incomplete, needs attention |
| Risk Limits | A | Appropriately conservative |

---

## Table of Contents

1. [HMM Regime Detection Model Analysis](#1-hmm-regime-detection-model-analysis)
2. [Risk Calculation Accuracy](#2-risk-calculation-accuracy)
3. [Allocation Optimization Methodology](#3-allocation-optimization-methodology)
4. [Feature Engineering Quality](#4-feature-engineering-quality)
5. [Backtesting Framework Assessment](#5-backtesting-framework-assessment)
6. [Statistical Assumptions and Validity](#6-statistical-assumptions-and-validity)
7. [Overfitting Risk Analysis](#7-overfitting-risk-analysis)
8. [Risk Limit Appropriateness](#8-risk-limit-appropriateness)
9. [Recommendations](#9-recommendations)
10. [Appendix: Mathematical Derivations](#appendix-mathematical-derivations)

---

## 1. HMM Regime Detection Model Analysis

### 1.1 Model Specification

**File:** `C:\Users\larai\FinancePortfolio\src\signals\regime.py`

The implementation uses a Gaussian Hidden Markov Model from the `hmmlearn` library with the following specification:

```python
GaussianHMM(
    n_components=3,           # Three regimes: RISK_ON, NEUTRAL, RISK_OFF
    covariance_type="full",   # Full covariance matrix
    n_iter=100,               # Maximum EM iterations
    random_state=42           # Reproducibility
)
```

### 1.2 Mathematical Foundation

The HMM assumes observations follow a multivariate Gaussian distribution conditioned on the hidden state:

$$P(x_t | s_t = k) = \mathcal{N}(x_t; \mu_k, \Sigma_k)$$

Where:
- $x_t$ is the observation vector at time $t$
- $s_t$ is the hidden state (regime)
- $\mu_k$ is the mean vector for state $k$
- $\Sigma_k$ is the covariance matrix for state $k$

**Assessment:** The choice of full covariance matrices is mathematically appropriate as it captures feature correlations within each regime.

### 1.3 State-to-Regime Mapping Logic

**Current Implementation (Lines 256-308):**

```python
def _map_states_to_regimes(self) -> None:
    first_feature_means = state_means[:, 0]
    sorted_state_indices = np.argsort(first_feature_means)

    # RISK_ON: lowest VIX-like feature mean
    self._state_to_regime[sorted_state_indices[0]] = Regime.RISK_ON
    # RISK_OFF: highest VIX-like feature mean
    self._state_to_regime[sorted_state_indices[-1]] = Regime.RISK_OFF
```

**Critique from Remy (Equity Quant):**

The mapping logic relies solely on the first feature (VIX level) to determine regime assignment. This is a reasonable heuristic but introduces fragility:

1. **Feature Ordering Dependency:** The implementation assumes VIX-like features are always in the first position. This assumption should be enforced programmatically.

2. **No Multi-Feature Regime Identification:** A more robust approach would use a weighted combination of features:

   $$\text{regime\_score}_k = w_1 \cdot \mu_{k,\text{VIX}} + w_2 \cdot \mu_{k,\text{trend}} + w_3 \cdot \mu_{k,\text{spread}}$$

3. **Missing Regime Coherence Check:** No validation that identified regimes exhibit expected characteristics across all features.

**Severity:** MEDIUM - Works for current implementation but limits extensibility.

### 1.4 Standardization Approach

**Current Implementation (Lines 227-234):**

```python
self._feature_means = np.mean(features, axis=0)
self._feature_stds = np.std(features, axis=0)
self._feature_stds = np.where(
    self._feature_stds < 1e-8, 1.0, self._feature_stds
)
features_standardized = (features - self._feature_means) / self._feature_stds
```

**Assessment:** Correct z-score standardization with appropriate handling of zero variance. The threshold of 1e-8 is appropriate for numerical stability.

### 1.5 Stationary Distribution Calculation

**Current Implementation (Lines 441-459):**

```python
eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
stationary = np.real(eigenvectors[:, idx])
stationary = stationary / stationary.sum()
```

**Mathematical Correctness:** VERIFIED

The stationary distribution $\pi$ satisfies $\pi = \pi A$ where $A$ is the transition matrix. This is equivalent to finding the left eigenvector corresponding to eigenvalue 1.

**Potential Issue:** For near-singular transition matrices, numerical instability may occur. Recommend adding condition number check.

### 1.6 Minimum Sample Requirements

**Current Implementation (Lines 217-223):**

```python
min_samples = self.n_states * 3  # 9 samples for 3 states
```

**Critique from Alexios (ML Designer):**

This minimum is dangerously low. For a 3-state HMM with full covariance and 9 features:

- **Parameters to estimate per state:**
  - Mean vector: 9 parameters
  - Covariance matrix (full): 9*(9+1)/2 = 45 parameters

- **Total parameters:** 3 * (9 + 45) + transition matrix = 162 + 9 = **171 parameters**

**Recommendation:** Minimum samples should be at least 10x the number of parameters, suggesting:

$$n_{\min} = 10 \times 171 = 1,710 \text{ samples}$$

**Severity:** HIGH - Current minimum allows severely underfit models.

---

## 2. Risk Calculation Accuracy

### 2.1 Value at Risk (VaR)

**File:** `C:\Users\larai\FinancePortfolio\src\portfolio\risk.py`

#### 2.1.1 Historical VaR Implementation (Lines 180-183)

```python
if method == "historical":
    var = -float(np.percentile(clean_returns, (1 - confidence) * 100))
```

**Mathematical Formula:**

$$\text{VaR}_\alpha = -\text{Quantile}_{1-\alpha}(R)$$

**Assessment:** CORRECT

The implementation correctly uses the negative of the $(1-\alpha)$ quantile to express VaR as a positive loss.

#### 2.1.2 Parametric VaR Implementation (Lines 184-193)

```python
mean_return = float(clean_returns.mean())
std_return = float(clean_returns.std(ddof=1))
z_score = float(scipy.stats.norm.ppf(1 - confidence))
var = -(mean_return + z_score * std_return)
```

**Mathematical Formula:**

$$\text{VaR}_\alpha = -(\mu + z_{1-\alpha} \cdot \sigma)$$

Where $z_{1-\alpha}$ is the standard normal quantile.

**Assessment:** CORRECT with caveat

The use of sample standard deviation with `ddof=1` (Bessel's correction) is appropriate. However, the parametric approach assumes normally distributed returns, which is known to underestimate tail risk for financial returns.

**Recommendation from Iacopo (Macro Analyst):** Consider implementing Expected Shortfall (CVaR) as a complement:

$$\text{ES}_\alpha = -\frac{1}{\alpha} \int_0^\alpha \text{Quantile}_p(R) \, dp$$

### 2.2 Portfolio Volatility Calculation

**Implementation (Lines 198-273):**

```python
# Calculate covariance matrix
cov_matrix = asset_returns.cov().values

# Portfolio variance: w' * Cov * w
portfolio_variance = float(weight_vector @ cov_matrix @ weight_vector)

# Annualization
daily_volatility = np.sqrt(portfolio_variance)
annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
```

**Mathematical Formula:**

$$\sigma_p = \sqrt{w^T \Sigma w}$$

$$\sigma_{p,\text{annual}} = \sigma_{p,\text{daily}} \times \sqrt{252}$$

**Assessment:** CORRECT

The matrix formulation is the standard portfolio variance calculation. The annualization using $\sqrt{252}$ assumes returns are i.i.d., which is a standard approximation.

**Note:** The implementation correctly normalizes weights for the non-cash portion (Lines 258-261), which is essential for portfolios with cash.

### 2.3 Sharpe Ratio Calculation

**Implementation (Lines 408-460):**

```python
# Convert annual risk-free rate to daily
daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1

# Calculate excess returns
excess_returns = clean_returns - daily_rf

mean_excess = float(excess_returns.mean())
std_returns = float(clean_returns.std(ddof=1))

# Daily Sharpe to annualized
daily_sharpe = mean_excess / std_returns
annualized_sharpe = daily_sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)
```

**Mathematical Formula:**

$$\text{SR}_{\text{daily}} = \frac{\bar{r}_{\text{excess}}}{\sigma_r}$$

$$\text{SR}_{\text{annual}} = \text{SR}_{\text{daily}} \times \sqrt{252}$$

**Assessment:** CORRECT with note

The annualization of Sharpe ratio using $\sqrt{T}$ is standard but assumes i.i.d. returns. For autocorrelated returns, the Lo (2002) correction should be considered:

$$\text{SR}_{\text{adjusted}} = \text{SR} \times \sqrt{\frac{q}{1 + 2\sum_{k=1}^{q-1}(1-k/q)\rho_k}}$$

Where $\rho_k$ is the $k$-th autocorrelation.

### 2.4 Sortino Ratio Calculation

**Implementation (Lines 462-521):**

```python
# Downside deviation: std of returns below target
negative_returns = clean_returns[clean_returns < daily_rf]
downside_deviation = float(negative_returns.std(ddof=1))

daily_sortino = mean_excess / downside_deviation
annualized_sortino = daily_sortino * np.sqrt(TRADING_DAYS_PER_YEAR)
```

**Mathematical Formula:**

$$\text{Sortino} = \frac{\bar{r} - r_f}{\sigma_{\text{downside}}}$$

Where:
$$\sigma_{\text{downside}} = \sqrt{\frac{1}{n}\sum_{r_i < r_f}(r_i - r_f)^2}$$

**Critique from Remy:**

The implementation uses `std(ddof=1)` on negative returns only, which is not the standard downside deviation formula. The correct formula should use all observations but only count deviations below target:

```python
downside_squared = np.where(returns < target, (returns - target)**2, 0)
downside_deviation = np.sqrt(downside_squared.mean())
```

**Severity:** MEDIUM - Current implementation underestimates downside deviation.

### 2.5 Leveraged ETF Decay Calculation

**Implementation (Lines 335-406):**

```python
# Actual cumulative returns
etf_cumret = float(etf_prod) - 1
idx_cumret = float(idx_prod) - 1

# Theoretical leveraged return
theoretical_leveraged_return = leverage * idx_cumret

# Total decay
total_decay = theoretical_leveraged_return - etf_cumret

# Volatility-based decay estimate
volatility_based_decay = (
    (leverage**2 - leverage) * (index_volatility**2) / 2
) * TRADING_DAYS_PER_YEAR
```

**Theoretical Foundation:**

The volatility drag for leveraged ETFs is derived from Ito's lemma:

$$\text{Decay} = \frac{L(L-1)\sigma^2}{2}$$

For 2x leverage with 20% volatility:
$$\text{Decay} = \frac{2(2-1)(0.20)^2}{2} = 0.04 = 4\%$$

**Assessment:** CORRECT

The implementation correctly uses both empirical and theoretical decay estimates, averaging them for robustness.

### 2.6 Maximum Drawdown Calculation

**Implementation (Lines 275-333):**

```python
# Calculate running maximum
running_max = clean_prices.cummax()

# Calculate drawdown at each point
drawdowns = (clean_prices - running_max) / running_max

# Find maximum drawdown
max_dd = float(drawdowns.min())
```

**Mathematical Formula:**

$$\text{DD}_t = \frac{P_t - \max_{s \leq t} P_s}{\max_{s \leq t} P_s}$$

$$\text{MaxDD} = \min_t \text{DD}_t$$

**Assessment:** CORRECT

The implementation correctly identifies the peak before the trough, which is essential for accurate drawdown attribution.

---

## 3. Allocation Optimization Methodology

### 3.1 Regime-Based Target Allocations

**File:** `C:\Users\larai\FinancePortfolio\src\signals\allocation.py`

**Current Allocations (Lines 86-90):**

```python
REGIME_ALLOCATIONS: dict[Regime, dict[str, float]] = {
    Regime.RISK_ON: {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10},
    Regime.NEUTRAL: {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20},
    Regime.RISK_OFF: {"LQQ": 0.05, "CL2": 0.05, "WPEA": 0.60, "CASH": 0.30},
}
```

### 3.2 Assessment from Jean-Yves (Portfolio Manager)

#### 3.2.1 Allocation Rationale

The allocations are **appropriately conservative** for a PEA portfolio:

| Regime | Leveraged Exposure | Risk Assessment |
|--------|-------------------|-----------------|
| RISK_ON | 30% | Maximum allowed |
| NEUTRAL | 20% | Moderate |
| RISK_OFF | 10% | Minimal |

**Kelly Criterion Validation:**

For 2x leveraged ETFs with typical parameters:
- Win rate: 55%
- Win/loss ratio: 1.2

$$f^* = \frac{0.55 \times 1.2 - 0.45}{1.2} = \frac{0.66 - 0.45}{1.2} = 0.175 = 17.5\%$$

Using half-Kelly (recommended for risk management): **8.75% per leveraged ETF**

The RISK_ON allocation of 15% per leveraged ETF is approximately 1.7x Kelly, which is aggressive but defensible given the regime confidence requirement.

#### 3.2.2 Confidence Blending

**Implementation (Lines 330-356):**

```python
def _blend_allocations(
    self,
    target: dict[str, float],
    neutral: dict[str, float],
    confidence: float,
) -> dict[str, float]:
    blended[symbol] = (
        confidence * target_weight + (1 - confidence) * neutral_weight
    )
```

**Mathematical Formula:**

$$w_{\text{blended}} = \alpha \cdot w_{\text{regime}} + (1-\alpha) \cdot w_{\text{neutral}}$$

Where $\alpha$ is the confidence level.

**Assessment:** SOUND

This approach provides smooth transitions between regimes and naturally reduces risk during uncertain periods.

### 3.3 Risk Limit Enforcement

**Hard-Coded Limits (from `models.py`, Lines 276-280):**

```python
MAX_LEVERAGED_EXPOSURE = 0.30  # LQQ + CL2 <= 30%
MAX_SINGLE_POSITION = 0.25
MIN_CASH_BUFFER = 0.10
REBALANCE_THRESHOLD = 0.05
DRAWDOWN_ALERT = -0.20
```

**Assessment:** These limits are appropriate for a personal PEA portfolio with leveraged ETF exposure.

### 3.4 Missing Optimization Components

**Critique from Alexios:**

The current implementation is a **rule-based system**, not an optimization framework. Missing components include:

1. **Mean-Variance Optimization:** No implementation of Markowitz optimization
2. **Risk Parity:** No equal risk contribution calculation
3. **Black-Litterman:** No views integration capability
4. **Transaction Cost Modeling:** Rebalancer includes basic costs but no optimization around them

**Recommendation:** For Sprint 4, consider implementing constrained mean-variance optimization:

$$\max_w \left( w^T \mu - \frac{\lambda}{2} w^T \Sigma w \right)$$

Subject to:
- $\sum_i w_i = 1$
- $w_{\text{LQQ}} + w_{\text{CL2}} \leq 0.30$
- $w_i \geq 0$ for all $i$
- $w_{\text{CASH}} \geq 0.10$

---

## 4. Feature Engineering Quality

### 4.1 Feature Set Composition

**File:** `C:\Users\larai\FinancePortfolio\src\signals\features.py`

**Features Implemented (9 total):**

| Category | Feature | Description | Quality |
|----------|---------|-------------|---------|
| Volatility | vix_level | Current VIX value | Excellent |
| Volatility | vix_percentile_20d | VIX rank over 20 days | Good |
| Volatility | realized_vol_20d | 20-day realized volatility | Excellent |
| Trend | price_vs_ma200 | Price relative to MA200 | Good |
| Trend | ma50_vs_ma200 | Moving average crossover | Good |
| Trend | momentum_3m | 3-month return | Good |
| Macro | yield_curve_slope | 10Y - 2Y spread | Excellent |
| Macro | hy_spread | High yield spread | Excellent |
| Macro | hy_spread_change_1m | HY spread momentum | Good |

### 4.2 Feature Calculation Assessment

#### 4.2.1 Realized Volatility (Lines 253-263)

```python
returns_20d = returns.iloc[-self.MIN_DAYS_VOLATILITY :]
realized_vol = float(returns_20d.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR))
```

**Formula:**
$$\sigma_{\text{realized}} = \sqrt{252} \cdot \text{std}(r_1, ..., r_{20})$$

**Assessment:** CORRECT

Standard close-to-close realized volatility. For future enhancement, consider Parkinson or Garman-Klass estimators that use OHLC data.

#### 4.2.2 VIX Percentile (Lines 249-251)

```python
vix_20d = vix_clean.iloc[-self.MIN_DAYS_VOLATILITY :]
vix_percentile = float((vix_20d < vix_level).sum() / len(vix_20d))
```

**Assessment:** CORRECT

This calculates the percentile rank of current VIX relative to the last 20 days. Note this is different from a longer-term percentile which might be more informative for regime detection.

**Recommendation from Iacopo:** Consider adding multiple lookback periods (60-day, 252-day) for more robust percentile signals.

#### 4.2.3 Trend Features (Lines 271-313)

```python
# 200-day moving average
ma200 = float(price_clean.iloc[-200:].mean())

# 50-day moving average
ma50 = float(price_clean.iloc[-50:].mean())
```

**Assessment:** CORRECT

Simple moving averages are calculated correctly. The price_vs_ma200 ratio is a well-established trend-following indicator.

### 4.3 Missing Features

**Identified by Alexios:**

1. **VIX Term Structure:** VIX/VIX3M ratio (contango/backwardation)
2. **Correlation Regime:** Stock-bond correlation (20-day rolling)
3. **Put/Call Ratio:** Options market sentiment
4. **Breadth Indicators:** Advance/decline ratios
5. **Credit Market Indicators:** Investment grade spreads

### 4.4 Data Validation

**Implementation (Lines 83-110):**

```python
@model_validator(mode="after")
def validate_feature_consistency(self) -> "FeatureSet":
    # VIX sanity check
    if self.vix_level > 100:
        raise ValueError(...)

    # Realized vol sanity check
    if self.realized_vol_20d > 2.0:
        raise ValueError(...)

    # Price ratio sanity check
    if self.price_vs_ma200 > 2.0 or self.price_vs_ma200 < 0.3:
        raise ValueError(...)
```

**Assessment:** GOOD

Reasonable bounds are enforced to catch data quality issues. The VIX upper bound of 100 is conservative (historical max ~90) but appropriate.

---

## 5. Backtesting Framework Assessment

### 5.1 Current State

**CRITICAL FINDING:** No formal backtesting framework has been implemented in Sprint 3.

The project documentation (`RESEARCH_TEAM_ANALYSIS.md`) outlines a walk-forward validation protocol:

```
Training: 5 years
Validation: 1 year
Test: 1 year (walk-forward)
```

However, no code implementation exists in the repository.

### 5.2 Required Components (Not Yet Implemented)

1. **Walk-Forward Engine:** Rolling window train/test framework
2. **Performance Attribution:** Decomposition of returns by regime
3. **Transaction Cost Simulation:** Realistic cost modeling
4. **Regime Accuracy Metrics:** Confusion matrix for regime predictions
5. **Out-of-Sample Reporting:** Holdout period performance

### 5.3 Risk Assessment

**Severity:** HIGH

Without proper backtesting:
- Model performance claims are unverified
- Overfitting cannot be detected
- Transaction costs are not accounted for
- Regime transition accuracy is unknown

**Recommendation:** Sprint 4 should prioritize backtesting infrastructure before any live deployment.

---

## 6. Statistical Assumptions and Validity

### 6.1 HMM Assumptions

| Assumption | Validity | Impact |
|------------|----------|--------|
| Markov property (no memory beyond current state) | Moderate | May miss multi-day patterns |
| Gaussian emissions | Low | Financial returns have fat tails |
| Time-homogeneous transitions | Moderate | Transition probabilities may vary with macro conditions |
| Complete state space | High | Three regimes cover major market conditions |

### 6.2 Return Distribution Assumptions

**Parametric VaR assumes normality:**

Financial returns exhibit:
- **Negative skewness:** -0.3 to -0.5 typical for equity indices
- **Excess kurtosis:** 3-10 typical (fat tails)

**Impact:** Parametric VaR underestimates tail risk by 20-40% for typical equity returns.

**Mitigation in Current Implementation:** Historical VaR is available as an alternative, which does not assume normality.

### 6.3 Annualization Assumptions

All annualization uses $\sqrt{252}$ which assumes:
- I.I.D. returns (no autocorrelation)
- 252 trading days per year

**Assessment:** Standard industry practice. Autocorrelation effects are typically small for daily returns.

---

## 7. Overfitting Risk Analysis

### 7.1 Model Complexity vs. Data

**HMM Parameter Count Analysis:**

For 9 features and 3 states with full covariance:

| Component | Parameters |
|-----------|-----------|
| Mean vectors | 3 x 9 = 27 |
| Covariance matrices | 3 x 45 = 135 |
| Transition matrix | 9 (3x3 minus constraints) |
| Initial distribution | 2 |
| **Total** | **173** |

**Minimum Data Requirement:** Following the 10:1 rule of thumb:
$$n_{\min} = 10 \times 173 = 1,730 \text{ observations}$$

At daily frequency, this requires approximately **7 years** of data.

### 7.2 Red Flags from Research Team Analysis

The team has appropriately identified overfitting indicators:

| Metric | Suspicious Threshold | Status |
|--------|---------------------|--------|
| Sharpe Ratio | > 2.0 | Not yet measurable |
| Win Rate | > 65% | Not yet measurable |
| Max Drawdown | < 5% | Not yet measurable |
| Turnover | > 200%/year | Not yet measurable |

### 7.3 Mitigation Measures in Place

1. **Random seed fixing** (`random_state=42`) - Ensures reproducibility
2. **Feature standardization** - Prevents scale-based overfitting
3. **Conservative allocations** - Rule-based limits prevent extreme positions

### 7.4 Missing Mitigation Measures

1. **Cross-validation not implemented**
2. **No regularization on HMM parameters**
3. **No ensemble methods**
4. **No out-of-sample testing framework**

---

## 8. Risk Limit Appropriateness

### 8.1 Leveraged Exposure Limit (30%)

**Current Limit:** `MAX_LEVERAGED_EXPOSURE = 0.30`

**Assessment from Jean-Yves:**

For 2x leveraged ETFs tracking the Nasdaq-100 and S&P 500:

| Scenario | Index Drop | 2x ETF Drop | Portfolio Impact (30% exposure) |
|----------|-----------|-------------|--------------------------------|
| Normal correction | -10% | -20% | -6% |
| Severe correction | -20% | -40% | -12% |
| Bear market | -35% | -70% | -21% |
| Crash (1987-style) | -22.6% | -45.2% | -13.6% |

The 30% limit ensures portfolio drawdown does not exceed 21% even in a bear market, which is within the -20% drawdown alert threshold.

**Verdict:** APPROPRIATE

### 8.2 Cash Buffer (10%)

**Current Limit:** `MIN_CASH_BUFFER = 0.10`

**Purpose:**
1. Rebalancing liquidity
2. Drawdown buffer
3. Opportunistic deployment

**Assessment:**

With 10% cash and 30% leveraged exposure:
- Maximum drawdown from leveraged portion: 21%
- Cash buffer covers approximately 50% of worst-case loss

**Recommendation:** Consider increasing to 15% in NEUTRAL regime (already implemented at 20%) and 25% in RISK_OFF (implemented at 30%).

**Verdict:** APPROPRIATE

### 8.3 Single Position Limit (25%)

**Current Limit:** `MAX_SINGLE_POSITION = 0.25`

**Assessment:**

This limit applies only to leveraged ETFs (Lines 310-318 in allocation.py):

```python
# WPEA is the core safe holding - not subject to max single position limit
for symbol, weight in weights.items():
    if (
        symbol in LEVERAGED_SYMBOLS
        and weight > self.risk_limits.max_single_position + 1e-6
    ):
```

This is appropriate as WPEA serves as the portfolio's core holding and limiting it would force unnecessary leveraged exposure.

**Verdict:** APPROPRIATE

### 8.4 Rebalancing Threshold (5%)

**Current Limit:** `REBALANCE_THRESHOLD = 0.05`

**Assessment:**

Transaction costs for PEA accounts (from `RebalancerConfig`):
- Commission: 0.1%
- Spread: 0.1%
- Total round-trip: ~0.4%

For a 5% drift rebalance:
- Trade size: 5% of portfolio
- Cost: 0.4% of 5% = 0.02% of portfolio

Break-even analysis:
$$\frac{\text{Cost}}{\text{Drift}^2/2} = \frac{0.0002}{0.05^2/2} = 0.16$$

This suggests the 5% threshold is reasonable from a cost-benefit perspective.

**Verdict:** APPROPRIATE

### 8.5 Drawdown Alert (-20%)

**Current Limit:** `DRAWDOWN_ALERT = -0.20`

**Assessment:**

Historical context:
- Average bear market drawdown: -35%
- 2008-2009: -57%
- 2020 COVID crash: -34%
- 2022: -25%

A -20% alert is conservative and allows time for defensive action before typical bear market bottoms.

**Verdict:** APPROPRIATE but should trigger automatic shift to RISK_OFF regime

---

## 9. Recommendations

### 9.1 Critical (Must Fix Before Production)

| ID | Issue | Recommendation | Priority |
|----|-------|----------------|----------|
| C1 | No backtesting framework | Implement walk-forward validation engine | P0 |
| C2 | Minimum sample size too low | Increase to 1,730+ observations | P0 |
| C3 | Sortino ratio calculation | Fix downside deviation formula | P1 |

### 9.2 High Priority (Sprint 4)

| ID | Issue | Recommendation | Priority |
|----|-------|----------------|----------|
| H1 | HMM regime mapping | Add multi-feature regime identification | P1 |
| H2 | No Expected Shortfall | Implement CVaR alongside VaR | P1 |
| H3 | Missing features | Add VIX term structure, correlation regime | P2 |
| H4 | No cross-validation | Implement k-fold CV for HMM training | P2 |

### 9.3 Medium Priority (Future Sprints)

| ID | Issue | Recommendation | Priority |
|----|-------|----------------|----------|
| M1 | Rule-based allocation | Implement mean-variance optimization | P3 |
| M2 | Static transition matrix | Consider regime-switching extensions | P3 |
| M3 | Normal distribution assumption | Add Student-t HMM variant | P3 |
| M4 | Sharpe ratio autocorrelation | Implement Lo (2002) correction | P3 |

### 9.4 Enhancements (Nice to Have)

| ID | Enhancement | Benefit |
|----|-------------|---------|
| E1 | Garman-Klass volatility | Better intraday vol estimate |
| E2 | Regime probability smoothing | Reduce whipsaw transitions |
| E3 | Transaction tax optimization | PEA-specific tax efficiency |
| E4 | LangChain regime verification | LLM sanity check on regime calls |

---

## Appendix: Mathematical Derivations

### A.1 Leveraged ETF Decay Derivation

Starting from geometric Brownian motion for the underlying index:

$$dS/S = \mu \, dt + \sigma \, dW$$

For a leveraged ETF with leverage factor $L$:

$$dV/V = L(\mu \, dt + \sigma \, dW) + (1-L) r \, dt$$

Applying Ito's lemma to $\log V$:

$$d(\log V) = L\mu \, dt - \frac{L^2 \sigma^2}{2} \, dt + L\sigma \, dW + (1-L)r \, dt$$

Expected log return:
$$E[\log(V_T/V_0)] = L\mu T - \frac{L^2 \sigma^2}{2} T + (1-L)rT$$

The term $-\frac{L^2 \sigma^2}{2}$ represents the volatility drag.

For $L=2$:
$$\text{Drag} = -\frac{4\sigma^2}{2} = -2\sigma^2$$

### A.2 Kelly Criterion Derivation

For a bet with probability $p$ of winning $b$ times the stake:

$$f^* = \arg\max_f E[\log(W)]$$

Where $W = (1 + fb)^{\text{win}} \times (1 - f)^{\text{loss}}$

Taking the derivative and setting to zero:

$$\frac{pb}{1+fb} - \frac{1-p}{1-f} = 0$$

Solving for $f$:

$$f^* = \frac{p(b+1) - 1}{b} = \frac{pb - q}{b}$$

Where $q = 1-p$.

### A.3 Stationary Distribution Derivation

For a transition matrix $A$ with stationary distribution $\pi$:

$$\pi A = \pi$$

Rearranging:
$$\pi (A - I) = 0$$

This means $\pi$ is in the null space of $(A - I)$, or equivalently, $\pi^T$ is the left eigenvector of $A$ corresponding to eigenvalue 1.

Existence is guaranteed for any stochastic matrix by Perron-Frobenius theorem.

---

## Document Approval

| Role | Name | Status | Date |
|------|------|--------|------|
| Portfolio Manager | Jean-Yves | APPROVED WITH RESERVATIONS | Dec 10, 2025 |
| Equity Quant | Remy | REVIEWED | Dec 10, 2025 |
| Macro Analyst | Iacopo | REVIEWED | Dec 10, 2025 |
| ML Designer | Alexios | REVIEWED | Dec 10, 2025 |

---

**Document Version:** 1.0
**Classification:** Internal
**Next Review:** Post-Sprint 4
