# Research Team Analysis: PEA Portfolio Optimization

**Date:** December 10, 2025
**Team Lead:** Jean-Yves (Portfolio Manager)
**Contributors:** Remy (Equity Quant), Iacopo (Macro Analyst), Alexios (ML Designer), Antoine (NLP Expert)

---

## Executive Summary

The Research team has conducted comprehensive analysis of the PEA portfolio optimization challenge. Key findings:

1. **Current portfolio is dangerously correlated** (0.85-0.95 between all assets)
2. **Leveraged ETFs require strict regime-based allocation** (volatility decay is significant)
3. **Recommended max allocation to leveraged ETFs: 25-30%** with 20-30% cash buffer
4. **ML models should expect 50-70% of backtest returns** in production
5. **Sentiment signals are weak for short-term** but valuable at weekly/monthly horizons

---

## 1. Portfolio Manager Assessment (Jean-Yves)

### Current Portfolio Issues

**Critical Finding: False Diversification**
- LQQ, CL2, WPEA correlation matrix shows 0.85-0.95 correlations
- All three are essentially US equity exposure
- This is NOT a diversified portfolio

### Mathematical Framework

**Kelly Criterion for Leveraged ETFs:**
```
f* = (p*b - q) / b
Where:
  p = probability of gain
  b = win/loss ratio
  q = 1-p

For 2x leverage with 60% win rate, 1:1 payout:
f* = (0.6*1 - 0.4) / 1 = 0.2 = 20% max allocation
```

**Volatility Decay Formula:**
```
E[log(V_T/V_0)] ≈ L*mu - (L^2 * sigma^2)/2
Where:
  L = leverage factor
  mu = underlying drift
  sigma = underlying volatility
```

### Research Agenda Assigned

| Analyst | Task | Deliverable |
|---------|------|-------------|
| Remy | Volatility decay quantification | ETF-specific decay rates |
| Iacopo | Regime detection framework | HMM model specification |
| Alexios | ML architecture design | Risk-adjusted allocator |
| Antoine | Sentiment integration | Transaction cost analysis |

### Recommended Allocation Ranges

| Regime | LQQ | CL2 | WPEA | Cash |
|--------|-----|-----|------|------|
| Risk-On | 15-20% | 15-20% | 50-60% | 10-15% |
| Neutral | 10-15% | 10-15% | 55-65% | 15-20% |
| Risk-Off | 5-10% | 5-10% | 55-60% | 25-30% |

---

## 2. Equity Quantitative Analysis (Remy)

### Stochastic Calculus Foundation

**Leveraged ETF Dynamics:**
```
dS_L/S_L = L * (dS/S) + (L-1) * r * dt

For daily rebalancing with 2x leverage:
Daily return: R_L = 2 * R_underlying - fees
```

### Volatility and Decay Analysis

| ETF | Annual Volatility | Annual Decay | Break-even Return Needed |
|-----|-------------------|--------------|--------------------------|
| **LQQ** (2x Nasdaq) | ~45% | ~10.1%/year | >10.4% underlying |
| **CL2** (2x USA) | ~35% | ~6.1%/year | >6.5% underlying |
| **WPEA** (1x World) | ~15% | 0% | N/A |

**Decay Formula Applied:**
```
LQQ: Decay = L^2 * sigma^2 / 2 = 4 * 0.45^2 / 2 = 0.405 / 2 = 10.1% annual drag

For LQQ to outperform 1x over long term:
Required: mu_nasdaq > 10.4% annually
Historical Nasdaq-100 CAGR: ~14% (1999-2024, with significant variance)
```

### Risk Parity Analysis

**Equal Risk Contribution Weights:**
```
w_i proportional to 1/sigma_i

Volatilities: LQQ=45%, CL2=35%, WPEA=15%
Risk parity weights:
  LQQ: (1/45) / (1/45 + 1/35 + 1/15) = 14%
  CL2: (1/35) / (1/45 + 1/35 + 1/15) = 18%
  WPEA: (1/15) / (1/45 + 1/35 + 1/15) = 68%
```

### Recommended Moderate Allocation

| Asset | Weight | Rationale |
|-------|--------|-----------|
| LQQ | 10% | High decay, limit exposure |
| CL2 | 10% | Moderate decay, broader exposure |
| WPEA | 65% | Core stable holding |
| Cash | 15% | Rebalancing buffer |

### Rebalancing Strategy

**Threshold-Based Rebalancing:**
- Trigger when any asset drifts >5% from target
- LQQ tends to drift fastest due to 2x leverage
- Monthly review minimum, weekly during high volatility

---

## 3. Macro & Futures Analysis (Iacopo)

### Current Regime Assessment (December 2024)

**Status: "Soft Landing in Progress" - Late-Cycle Expansion**

**Key Indicators:**
- Fed began easing (first cut Sep 2024, 50bps)
- Inflation declining but above target (2.5-3%)
- Labor market softening but resilient
- Credit conditions tightening moderately
- VIX: Elevated (18-22 range)

**Assessment: NEUTRAL regime** - neither full Risk-On nor Risk-Off

### Three-Regime Framework

#### Regime 1: Risk-On (Soft Landing Confirmed)
**Triggers:**
- NFP consistently > 150k
- Inflation < 2.5%
- VIX < 15
- Credit spreads < 350bps
- ISM > 52

**Allocation:**
| Asset | Min | Max |
|-------|-----|-----|
| LQQ | 35% | 40% |
| CL2 | 35% | 40% |
| WPEA | 20% | 30% |

#### Regime 2: Neutral (Current State)
**Triggers:**
- NFP 100k-200k
- Inflation 2.5-3.5%
- VIX 15-25
- Credit spreads 350-500bps
- ISM 48-52

**Allocation:**
| Asset | Min | Max |
|-------|-----|-----|
| LQQ | 25% | 30% |
| CL2 | 25% | 30% |
| WPEA | 40% | 50% |

#### Regime 3: Risk-Off (Recession)
**Triggers:**
- NFP < 50k or negative
- VIX > 25 sustained
- Credit spreads > 500bps
- ISM < 45
- Yield curve un-inversion with rising unemployment

**Allocation:**
| Asset | Min | Max |
|-------|-----|-----|
| LQQ | 10% | 15% |
| CL2 | 10% | 15% |
| WPEA | 70% | 80% |

### Key Macro Triggers to Monitor

| Indicator | Frequency | Risk-Off Threshold | Source |
|-----------|-----------|-------------------|--------|
| NFP | Monthly | < 50k | BLS |
| VIX | Daily | > 25 (3-day avg) | CBOE |
| HY Spreads | Daily | > 500bps | FRED |
| ISM Manufacturing | Monthly | < 45 | ISM |
| 2s10s Spread | Daily | Un-invert + rising UE | FRED |

### Currency Considerations

**EUR/USD Impact:**
- Portfolio is USD-denominated assets held in EUR account
- USD strength = positive for EUR investor (currency gain)
- USD weakness = negative (currency drag)
- Recommendation: Do NOT hedge FX (adds cost, PEA naturally long-term)

---

## 4. ML Model Design (Alexios)

### Architecture: Two-Stage Hierarchical Model

```
Stage 1: Regime Detection (HMM)
         ↓
Stage 2: Regime-Conditional Allocation Optimizer
```

### Stage 1: Hidden Markov Model for Regime Detection

**States:** 3 (Risk-On, Neutral, Risk-Off)
**Features:**
- VIX level and percentile (20-day, 60-day)
- VIX term structure (VIX/VIX3M ratio)
- Price vs 200-day MA (trend)
- Credit spread level and change
- Yield curve slope

**Transition Matrix (Prior):**
```
          Risk-On  Neutral  Risk-Off
Risk-On    0.85     0.12     0.03
Neutral    0.15     0.70     0.15
Risk-Off   0.05     0.20     0.75
```

### Stage 2: Regime-Conditional Allocation

**For each regime, optimize:**
```
max E[r_p] - lambda * Var(r_p)

Subject to:
  - Sum(weights) = 1
  - 0 <= w_i <= max_weight[regime]
  - Leveraged ETF total <= 30%
```

### Anti-Overfitting Framework

**Critical Rule: Haircut Principle**
- Expect 50-70% of backtest returns in production
- If backtest Sharpe > 2.0, something is wrong
- If backtest win rate > 65%, likely curve-fitted

**Red Flags (Reject Model If):**
| Metric | Suspicious Threshold |
|--------|---------------------|
| Sharpe Ratio | > 2.0 |
| Win Rate | > 65% |
| Max Drawdown | < 5% |
| Turnover | > 200%/year |
| Parameters | > 20 |

**Realistic Expectations:**
| Metric | Target Range |
|--------|-------------|
| Sharpe Ratio | 0.3 - 0.8 |
| Annual Return | 5% - 12% |
| Max Drawdown | 15% - 25% |
| Win Rate | 50% - 58% |

### Feature Taxonomy

**Category 1: Volatility (Most Important)**
- VIX level (percentile rank)
- VIX term structure slope
- Realized volatility (20-day)
- VVIX (volatility of volatility)

**Category 2: Trend**
- Price vs MA200
- MA50 vs MA200 (golden/death cross)
- Momentum (3-month return)

**Category 3: Correlation**
- Stock-bond correlation (20-day rolling)
- Cross-asset correlation breakdown indicator

**Category 4: Macro**
- Yield curve slope (2s10s)
- Credit spreads (HY-IG)
- Dollar index trend

**Category 5: Calendar**
- Month-of-year (seasonality)
- Days to earnings (for LQQ tech exposure)

### Walk-Forward Validation Protocol

```
Training: 5 years
Validation: 1 year
Test: 1 year (walk-forward)

Rolling windows:
  Window 1: Train 2010-2014, Val 2015, Test 2016
  Window 2: Train 2011-2015, Val 2016, Test 2017
  ... continue to present

Final model: Average performance across all test periods
```

---

## 5. NLP & Sentiment Analysis (Antoine)

### Priority Data Sources

**Tier 1: Highest Signal (70-80% directional accuracy)**
- Fed/ECB communications (FOMC statements, minutes, speeches)
- Major tech earnings (for LQQ exposure)

**Tier 2: Moderate Signal**
- Financial news headlines (Bloomberg, Reuters)
- Analyst upgrades/downgrades

**Tier 3: Noisy but Useful**
- Reddit (r/investing, r/wallstreetbets) - contrarian indicator
- Twitter/X finance accounts

### Composite Fear Index (CFI)

**Formula:**
```
CFI = 0.30 * news_sentiment
    + 0.25 * fed_hawkishness
    + 0.20 * earnings_sentiment
    + 0.15 * social_sentiment
    + 0.10 * policy_uncertainty

Range: 0 (extreme greed) to 100 (extreme fear)
```

**Thresholds:**
| CFI Range | Interpretation | Action |
|-----------|---------------|--------|
| 0-20 | Extreme Greed | Reduce risk |
| 20-40 | Greed | Neutral |
| 40-60 | Neutral | Hold |
| 60-80 | Fear | Add risk |
| 80-100 | Extreme Fear | Max risk |

### Central Bank Hawkishness Scoring

**Implementation using LangChain + Claude:**

```python
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

class HawkishnessScore(BaseModel):
    score: float  # -1 (dovish) to +1 (hawkish)
    confidence: float
    key_phrases: list[str]

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
structured_llm = llm.with_structured_output(HawkishnessScore)

prompt = """Analyze this Fed statement for monetary policy stance.
Score from -1 (very dovish) to +1 (very hawkish).
Statement: {text}"""
```

### Realistic Expectations for Sentiment Signals

**Important Caveat:**
- Sentiment signals have weak predictive power for short-term (daily)
- Correlation with next-day returns: 0.1 - 0.3
- Better at weekly/monthly horizons (0.2 - 0.4 correlation)

**Usage Recommendation:**
- Use sentiment as CONFIRMATION, not primary signal
- Weight: 10-20% in final allocation decision
- Most useful for regime transitions (not daily trading)

### Free/Low-Cost Data Sources

| Source | Cost | API | Quality |
|--------|------|-----|---------|
| Fed/ECB RSS | Free | Yes | High |
| Alpha Vantage | Free tier | Yes | Medium |
| Finnhub | Free tier | Yes | Medium |
| Reddit API | Free | Yes | Low (noisy) |
| GDELT | Free | Yes | Medium |

---

## 6. Consolidated Recommendations

### Immediate Actions

1. **Reduce correlation risk** - Current portfolio is essentially 100% US equity
2. **Cap leveraged ETF exposure at 30%** combined (LQQ + CL2)
3. **Maintain 15-25% cash buffer** for rebalancing and drawdown protection
4. **Implement regime detection** before deploying capital

### Allocation Framework

**Conservative Start (Recommended):**
| Asset | Weight |
|-------|--------|
| LQQ | 10% |
| CL2 | 10% |
| WPEA | 60% |
| Cash | 20% |

**After Regime Model Validated:**
| Regime | LQQ | CL2 | WPEA | Cash |
|--------|-----|-----|------|------|
| Risk-On | 20% | 20% | 50% | 10% |
| Neutral | 15% | 15% | 55% | 15% |
| Risk-Off | 5% | 5% | 60% | 30% |

### Risk Limits

| Metric | Limit |
|--------|-------|
| Total leveraged ETF exposure | 30% max |
| Single position max | 25% |
| Cash minimum | 10% |
| Max drawdown trigger | -20% (reduce risk) |
| Rebalancing threshold | 5% drift |

### ML Model Requirements

- Walk-forward validation mandatory
- Backtest Sharpe must be < 2.0
- Apply 50% haircut to expected returns
- Feature count < 20
- Monthly retraining maximum

---

## Appendix: Key Formulas

### Leveraged ETF Decay
```
Annual Decay = L^2 * sigma^2 / 2
LQQ (2x, 45% vol): 4 * 0.2025 / 2 = 10.1%
CL2 (2x, 35% vol): 4 * 0.1225 / 2 = 6.1%
```

### Kelly Criterion
```
f* = (p*b - q) / b
Fractional Kelly (recommended): f*/2 or f*/4
```

### Risk Parity Weights
```
w_i = (1/sigma_i) / sum(1/sigma_j)
```

### VaR (95%, 1-day)
```
VaR_95 = Portfolio_Value * sigma_daily * 1.645
```

---

**Document Version:** 1.0
**Next Review:** After implementation phase begins
