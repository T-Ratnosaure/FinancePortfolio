# Sprint 5 P0 - Macro/Futures Analyst Review

**Review Date:** December 12, 2025
**Reviewer:** Iacopo (Macro/Futures Analyst)
**Sprint:** Sprint 5 - P0 Completion
**Scope:** Macro data staleness thresholds, regime detection, economic indicator coverage
**Status:** POST-IMPLEMENTATION REVIEW

---

## Executive Summary

Sprint 5 P0 delivered meaningful improvements to the macro data infrastructure, particularly around FRED data resilience and staleness detection. From a macro analyst perspective, the implementation is **functionally sound but strategically limited**. The 7-day staleness threshold for macro data is appropriate for most indicators but fails to account for publication frequency heterogeneity. The HMM regime detector captures broad risk-on/risk-off dynamics but lacks the macro nuance required for sophisticated regime identification.

**Overall Macro Infrastructure Score: 6.5/10**

---

## 1. Macro Data Staleness Thresholds Assessment

### 1.1 Current Implementation

```python
# From models.py lines 300-314
STALENESS_THRESHOLDS = {
    DataCategory.MACRO_DATA: timedelta(days=7),   # Weekly tolerance
}
CRITICAL_THRESHOLDS = {
    DataCategory.MACRO_DATA: timedelta(days=30),  # 1 month = critical
}
```

### 1.2 Assessment by Indicator Category

| Indicator Category | Publication Frequency | Current Threshold (7 days) | Appropriate? | Recommended |
|-------------------|----------------------|---------------------------|--------------|-------------|
| **Daily Indicators** | | | | |
| VIX (VIXCLS) | Daily | 7 days | Too lenient | 1 day |
| Treasury Yields (DGS2, DGS10) | Daily | 7 days | Too lenient | 1 day |
| HY OAS Spread (BAMLH0A0HYM2) | Daily | 7 days | Too lenient | 1 day |
| **Weekly Indicators** | | | | |
| Initial Jobless Claims | Weekly (Thurs) | 7 days | Appropriate | 7 days |
| Consumer Credit | Weekly | 7 days | Appropriate | 7 days |
| **Monthly Indicators** | | | | |
| CPI | Monthly | 7 days | Too strict | 35 days |
| PPI | Monthly | 7 days | Too strict | 35 days |
| Nonfarm Payrolls | Monthly (1st Fri) | 7 days | Too strict | 35 days |
| ISM PMI | Monthly | 7 days | Too strict | 35 days |
| **Quarterly Indicators** | | | | |
| GDP (GDPC1) | Quarterly | 7 days | Far too strict | 100 days |
| Flow of Funds | Quarterly | 7 days | Far too strict | 100 days |

### 1.3 Critical Finding: One-Size-Fits-All Problem

**The current implementation treats all macro data identically, which is fundamentally inappropriate for macro analysis.**

**Example Scenario:**
- GDP data published January 30th is still the "most recent" data on April 15th
- Current system would flag it as "CRITICAL" after 30 days
- In reality, this is expected behavior for quarterly data

**Impact:**
- False positive staleness warnings for lower-frequency indicators
- Risk of ignoring genuinely stale daily indicators (VIX could be 5 days old and marked "FRESH")

### 1.4 Recommended Solution

Implement indicator-specific staleness thresholds:

```python
MACRO_INDICATOR_THRESHOLDS = {
    # Daily indicators - should be < 1 day stale
    "VIX": {"stale": timedelta(days=1), "critical": timedelta(days=3)},
    "TREASURY_2Y": {"stale": timedelta(days=1), "critical": timedelta(days=3)},
    "TREASURY_10Y": {"stale": timedelta(days=1), "critical": timedelta(days=3)},
    "HY_OAS_SPREAD": {"stale": timedelta(days=1), "critical": timedelta(days=5)},
    "SPREAD_2S10S": {"stale": timedelta(days=1), "critical": timedelta(days=3)},

    # Weekly indicators
    "INITIAL_CLAIMS": {"stale": timedelta(days=7), "critical": timedelta(days=14)},

    # Monthly indicators
    "CPI": {"stale": timedelta(days=35), "critical": timedelta(days=70)},
    "UNRATE": {"stale": timedelta(days=35), "critical": timedelta(days=70)},
    "ISM_PMI": {"stale": timedelta(days=35), "critical": timedelta(days=70)},

    # Quarterly indicators
    "GDP": {"stale": timedelta(days=100), "critical": timedelta(days=180)},
}
```

**Priority:** P1 for Sprint 5 P1
**Effort:** 4 hours

---

## 2. Regime Detection Macro Analysis

### 2.1 Current HMM Feature Set

From `features.py`, the regime detector uses 9 features:

| Feature | Category | Macro Relevance | Assessment |
|---------|----------|-----------------|------------|
| `vix_level` | Volatility | High - Forward-looking fear gauge | Good |
| `vix_percentile_20d` | Volatility | Medium - Relative VIX context | Good |
| `realized_vol_20d` | Volatility | Medium - Backward-looking risk | Good |
| `price_vs_ma200` | Trend | Low - Price-based only | Weak |
| `ma50_vs_ma200` | Trend | Low - Technical indicator | Weak |
| `momentum_3m` | Trend | Low - Price momentum | Weak |
| `yield_curve_slope` | Macro | High - Recession predictor | Good |
| `hy_spread` | Macro | High - Credit stress indicator | Good |
| `hy_spread_change_1m` | Macro | High - Credit acceleration | Good |

### 2.2 Feature Set Strengths

**Volatility Features (Score: 8/10)**
- VIX is the gold standard for equity risk-off detection
- VIX percentile provides relative context
- Realized vol offers backward confirmation

**Credit Features (Score: 7/10)**
- HY OAS spread captures credit stress effectively
- Spread change captures acceleration/deceleration dynamics
- Yield curve slope is a proven recession indicator

### 2.3 Feature Set Weaknesses

**Trend Features (Score: 4/10)**
- Entirely price-based, no fundamental macro input
- MA crossovers are lagging indicators
- 3-month momentum conflates cyclical and structural trends

**Missing Macro Dimensions:**

1. **Labor Market Health** (Critical Gap)
   - No employment data (NFP, initial claims, unemployment rate)
   - Labor market deterioration often precedes equity drawdowns
   - Recommended: Add ICSA (Initial Claims) z-score

2. **Inflation Regime** (Critical Gap)
   - No CPI or inflation expectations data
   - Current environment: Inflation regime matters enormously
   - Recommended: Add T5YIFR (5Y Breakeven Inflation) or CPI YoY

3. **Financial Conditions** (Moderate Gap)
   - No financial conditions index
   - Recommended: Add NFCI (Chicago Fed National Financial Conditions Index)

4. **Growth Expectations** (Moderate Gap)
   - No GDP nowcast or leading indicators
   - Recommended: Add LEI (Leading Economic Indicators) or ISM PMI

5. **Liquidity Conditions** (Moderate Gap)
   - No Fed balance sheet or reserve data
   - No TED spread or LIBOR-OIS
   - Relevant for understanding liquidity-driven risk regimes

### 2.4 State Mapping Methodology

The current state-to-regime mapping uses VIX as the primary discriminator:

```python
# From regime.py lines 480-484
first_feature_means = state_means[:, 0]  # VIX is feature 0
sorted_state_indices = np.argsort(first_feature_means)
# Lowest VIX mean -> RISK_ON
# Highest VIX mean -> RISK_OFF
```

**Assessment:**
- Simple and interpretable
- Works well for VIX-driven regimes
- May misclassify macro-driven regimes where VIX lags fundamentals

**Alternative Approach (Recommended):**
Consider composite scoring using multiple features:

```python
def _compute_regime_score(self, state_means: np.ndarray) -> np.ndarray:
    """Compute risk score for each state using multiple features."""
    # Feature weights based on macro relevance
    weights = np.array([
        0.25,  # vix_level (primary)
        0.10,  # vix_percentile
        0.05,  # realized_vol
        0.05,  # price_vs_ma200
        0.05,  # ma50_vs_ma200
        0.05,  # momentum_3m
        0.15,  # yield_curve_slope (inverted = higher risk)
        0.20,  # hy_spread (higher = higher risk)
        0.10,  # hy_spread_change
    ])

    # Standardize means and compute weighted score
    standardized = (state_means - state_means.mean(axis=0)) / state_means.std(axis=0)
    risk_scores = standardized @ weights
    return risk_scores
```

### 2.5 Sample Size Requirements

The HMM implementation correctly enforces minimum sample sizes:

```
3 states x 9 features x full covariance = 170 parameters
Minimum samples: 1,700 (approximately 7 years of daily data)
```

**Assessment:**
- This is statistically sound
- 7 years captures multiple business cycles
- However, this means the model cannot adapt quickly to structural regime changes

**Macro Concern:**
The model trained on 2017-2024 data will reflect:
- Post-GFC low rate environment
- COVID shock dynamics
- 2022-2023 inflation regime

If we're entering a structurally different macro environment (higher rates, deglobalization), the learned transition probabilities may be stale.

**Recommendation:** Consider rolling window retraining or regime-conditional training data weighting.

---

## 3. Economic Indicator Integration Priorities

### 3.1 Priority Matrix

| Indicator | FRED Code | Priority | Macro Rationale | Effort |
|-----------|-----------|----------|-----------------|--------|
| Initial Jobless Claims | ICSA | P0 | Leading recession indicator | 2h |
| ISM Manufacturing PMI | MANEMP | P1 | Business cycle timing | 4h |
| CPI YoY | CPIAUCSL | P1 | Inflation regime context | 2h |
| 5Y Breakeven Inflation | T5YIFR | P1 | Market inflation expectations | 2h |
| NFCI | NFCI | P2 | Financial conditions | 2h |
| TED Spread | TEDRATE | P2 | Interbank stress | 2h |
| LEI | USSLIND | P2 | Leading indicators composite | 2h |

### 3.2 Recommended Feature Additions (Sprint 5 P1)

**Tier 1 - Add Immediately:**

1. **Initial Claims Z-Score**
   ```python
   # Rolling 52-week z-score of initial claims
   claims_zscore = (claims - claims.rolling(252).mean()) / claims.rolling(252).std()
   ```
   - Why: Best real-time labor market indicator
   - Publication: Weekly (Thursday), minimal lag
   - Threshold: Z-score > 1.5 signals labor market stress

2. **CPI YoY Rate of Change**
   ```python
   # Year-over-year CPI change
   cpi_yoy = cpi.pct_change(periods=12) * 100
   ```
   - Why: Distinguishes inflation vs disinflation regimes
   - Publication: Monthly (mid-month), ~2 week lag
   - Threshold: YoY > 3% = inflationary regime

**Tier 2 - Add in Sprint 6:**

3. **NFCI (National Financial Conditions Index)**
   - Composite index capturing overall financial conditions
   - Positive = tighter than average, negative = looser
   - Weekly publication

4. **5Y Breakeven Inflation**
   - Market-implied inflation expectations
   - Real-time, derived from TIPS
   - Captures inflation regime shifts

### 3.3 FRED Fetcher Expansion

The current FRED fetcher supports 5 series:
- VIXCLS (VIX)
- DGS10 (10Y Treasury)
- DGS2 (2Y Treasury)
- T10Y2Y (2s10s spread)
- BAMLH0A0HYM2 (HY OAS)

**Recommended Additions:**

```python
# Additional FRED series for macro indicators
SERIES_INITIAL_CLAIMS = "ICSA"       # Initial Jobless Claims
SERIES_CPI = "CPIAUCSL"              # CPI All Items
SERIES_BREAKEVEN_5Y = "T5YIFR"       # 5Y Breakeven Inflation
SERIES_NFCI = "NFCI"                 # Chicago Fed NFCI
SERIES_ISM_PMI = "MANEMP"            # ISM Manufacturing Employment
SERIES_TED_SPREAD = "TEDRATE"        # TED Spread (when available)
```

---

## 4. Interest Rate Sensitivity Analysis

### 4.1 Current Rate Exposure

The portfolio holds PEA-eligible ETFs:
- **LQQ (2x Nasdaq-100)**: Extreme duration sensitivity via tech multiples
- **CL2 (2x MSCI USA)**: High duration sensitivity
- **WPEA (MSCI World)**: Moderate duration sensitivity

### 4.2 Missing Rate Risk Features

**Current Implementation:**
- 2s10s spread captures yield curve shape
- No Fed Funds rate path information
- No real rate (TIPS) information
- No term premium estimation

**Recommended Additions:**

1. **Fed Funds Effective Rate**
   - FRED Code: FEDFUNDS or DFF
   - Captures current policy stance
   - Important for cash allocation decisions

2. **Fed Funds Futures Implied Path**
   - Derive from CME FF futures
   - Captures market expectations for policy
   - More dynamic than spot rate

3. **10Y Real Rate (TIPS)**
   - FRED Code: DFII10
   - Higher real rates = pressure on equity multiples
   - Critical for growth stock valuation regimes

4. **Term Premium Estimate**
   - Use ACM model term premium (NY Fed)
   - FRED Code: THREEFYTP10
   - Separates rate expectations from risk compensation

### 4.3 Rate Sensitivity Feature Proposal

```python
# New rate sensitivity features
rate_features = {
    "fed_funds_level": float,           # Current FF rate
    "ff_12m_expectations": float,        # Market-implied FF 12 months ahead
    "real_rate_10y": float,             # 10Y TIPS yield
    "rate_vol_1m": float,               # 1-month Treasury volatility (MOVE)
    "term_premium_10y": float,          # ACM term premium
}
```

### 4.4 Portfolio Duration Risk Flag

Implement a rate sensitivity warning when:
- Real 10Y rate is rising AND > 1.5%
- Fed Funds futures imply hikes
- Rate volatility (MOVE index) elevated

```python
def should_reduce_rate_exposure(
    real_rate_10y: float,
    move_index: float,
    ff_expectations_delta: float
) -> bool:
    """Determine if rate conditions warrant reduced equity duration."""
    return (
        real_rate_10y > 1.5 and  # High real rates
        move_index > 120 and     # Elevated rate volatility
        ff_expectations_delta > 0.25  # Market pricing more hikes
    )
```

---

## 5. Regime Detection Improvement Roadmap

### 5.1 Short-Term (Sprint 5 P1)

1. **Add labor market feature**
   - Initial Claims z-score
   - Integrate into FeatureSet
   - Retrain HMM with 10 features

2. **Implement indicator-specific staleness**
   - Daily vs weekly vs monthly thresholds
   - Update DataFreshness model

3. **Add inflation regime context**
   - CPI YoY as feature
   - Distinguish inflation vs growth regimes

### 5.2 Medium-Term (Sprint 6)

1. **Expand macro feature set**
   - NFCI, breakeven inflation, ISM PMI
   - Total features: 12-15

2. **Implement composite regime scoring**
   - Move beyond VIX-only state mapping
   - Weight features by macro relevance

3. **Add rate sensitivity analysis**
   - Real rate tracking
   - Fed policy path integration

### 5.3 Long-Term (Future Sprints)

1. **Consider regime-switching models**
   - Markov-Switching VAR for macro
   - TVTP (Time-Varying Transition Probabilities)

2. **Add nowcasting layer**
   - GDP nowcast integration
   - Bridge high-frequency to low-frequency data

3. **Sentiment integration**
   - AAII sentiment, put/call ratios
   - News sentiment (if NLP available)

---

## 6. FRED Fetcher Retry Logic Assessment

### 6.1 Current Implementation (Score: 9/10)

The FRED fetcher retry logic is well-implemented:

```python
@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
```

**Strengths:**
- Exponential backoff appropriate for API rate limits
- Error classification distinguishes retriable vs non-retriable
- Rate limiting between requests (0.5s) respects FRED limits

**Weaknesses:**
- No circuit breaker (addressed in data review)
- `max_retries` parameter not actually used in decorator

### 6.2 Macro Data Considerations

**Publication Timing Awareness:**
FRED data has specific publication schedules:
- Treasury yields: Available ~4pm ET
- VIX: Available at market close
- CPI: 8:30am ET on release day

**Recommendation:** Add publication awareness to avoid spurious "data not available" errors:

```python
FRED_PUBLICATION_HOURS = {
    "VIXCLS": 16,      # 4pm ET
    "DGS10": 16,       # 4pm ET
    "CPIAUCSL": 8,     # 8:30am ET on release day
    "ICSA": 8,         # 8:30am ET Thursdays
}
```

---

## 7. Risk Assessment

### 7.1 Macro Model Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Regime misclassification due to stale data | Medium | High | Implement indicator-specific thresholds |
| HMM trained on non-representative period | Medium | High | Rolling window retraining |
| Missing labor market signal | High | Medium | Add initial claims feature |
| Inflation regime not captured | High | Medium | Add CPI feature |
| Rate shock undetected | Medium | High | Add real rate features |

### 7.2 Data Quality Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FRED API downtime | Low | High | Circuit breaker, fallback data |
| Data revisions changing historical | Medium | Low | Version historical data |
| Publication delays | Medium | Medium | Publication timing awareness |

---

## 8. Recommendations Summary

### 8.1 Immediate Actions (Sprint 5 P1)

1. **Implement indicator-specific staleness thresholds**
   - Priority: P0
   - Effort: 4 hours
   - Owner: Data team

2. **Add initial claims to feature set**
   - Priority: P1
   - Effort: 4 hours
   - Owner: Research team

3. **Add CPI YoY to feature set**
   - Priority: P1
   - Effort: 2 hours
   - Owner: Research team

### 8.2 Near-Term Actions (Sprint 6)

4. **Expand FRED fetcher for new indicators**
   - NFCI, T5YIFR, ICSA, DFII10
   - Effort: 8 hours

5. **Implement composite regime scoring**
   - Move beyond VIX-only classification
   - Effort: 8 hours

6. **Add rate sensitivity analysis module**
   - Real rates, term premium, Fed expectations
   - Effort: 16 hours

### 8.3 Documentation Updates

7. **Document macro indicator publication schedules**
8. **Document feature engineering rationale**
9. **Add macro regime interpretation guide**

---

## 9. Conclusion

Sprint 5 P0 delivers solid foundational improvements for macro data handling. The staleness detection and FRED retry logic are production-quality. However, the macro content of the regime detection system requires enhancement to capture the full spectrum of macro regimes.

**Key Takeaways:**

1. **Staleness thresholds need indicator-specific calibration** - A 7-day uniform threshold is inappropriate for the heterogeneity of macro data publication frequencies.

2. **The regime detector is VIX-centric** - This works for volatility-driven regimes but misses labor market and inflation regime shifts.

3. **Rate sensitivity is underweighted** - Given the leveraged equity exposure in the portfolio, real rate and duration risk features should be added.

4. **The HMM sample requirements are statistically sound** - But this creates model rigidity that may lag structural macro changes.

**Overall Assessment:**
- Macro Data Infrastructure: 7/10 (up from 5/10 pre-Sprint 5)
- Regime Detection Macro Content: 5/10 (needs expansion)
- Rate Sensitivity Coverage: 3/10 (significant gap)

The foundation is solid. Sprint 5 P1 should focus on expanding the macro feature set and implementing indicator-specific staleness thresholds.

---

**Reviewed by:** Iacopo (Macro/Futures Analyst)
**Next Review:** Post-Sprint 5 P1 completion
**Approved for:** Sprint 5 P1 continuation with recommendations implemented
