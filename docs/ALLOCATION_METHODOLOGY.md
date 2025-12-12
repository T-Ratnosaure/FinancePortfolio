# Allocation Methodology

This document provides comprehensive documentation for the regime-based allocation strategy implemented in the PEA Portfolio Optimization System. The methodology combines Hidden Markov Model (HMM) regime detection with risk-constrained allocation rules.

---

## Table of Contents

1. [Overview](#overview)
2. [Regime Detection with Hidden Markov Models](#regime-detection-with-hidden-markov-models)
3. [Regime-Based Allocation Rules](#regime-based-allocation-rules)
4. [Rebalancing Methodology](#rebalancing-methodology)
5. [Walk-Forward Validation](#walk-forward-validation)
6. [Expected Performance Characteristics](#expected-performance-characteristics)
7. [Implementation Details](#implementation-details)

---

## Overview

The allocation methodology follows a three-stage process:

```
Feature Calculation -> Regime Detection -> Allocation Generation
         |                    |                     |
   9 market features    3-state HMM         Risk-constrained weights
```

### Core Principles

1. **Regime Awareness:** Market conditions vary; allocation should adapt
2. **Risk Primacy:** Hard limits are non-negotiable regardless of regime
3. **Statistical Rigor:** Decisions are driven by probabilistic models, not intuition
4. **Temporal Discipline:** Strict separation of training and testing periods

---

## Regime Detection with Hidden Markov Models

### Why Hidden Markov Models?

Hidden Markov Models are particularly suited for financial regime detection because:

1. **Latent State Modeling:** Markets have underlying "regimes" that are not directly observable
2. **Temporal Persistence:** Regimes tend to persist (bull markets last years, not days)
3. **Probabilistic Output:** HMMs provide probability distributions, not just point estimates
4. **Transition Dynamics:** The model learns how regimes transition between states

### Mathematical Foundation

A Gaussian HMM is defined by:

**Hidden States:** S = {s_1, s_2, ..., s_K} where K = 3 in our implementation

**Observable Features:** O_t in R^d where d = 9 (our feature count)

**Model Parameters (theta):**

1. **Initial Distribution (pi):**
   ```
   pi_i = P(S_1 = s_i)
   ```

2. **Transition Matrix (A):**
   ```
   a_ij = P(S_{t+1} = s_j | S_t = s_i)
   ```

3. **Emission Distribution (B):**
   For Gaussian emissions:
   ```
   b_i(o) = N(o | mu_i, Sigma_i)
   ```
   Where mu_i is the mean vector and Sigma_i is the covariance matrix for state i.

### Three-State Configuration

The system uses K = 3 hidden states, mapped to market regimes post-fitting:

| State | Regime | Characteristics | Mapping Logic |
|-------|--------|-----------------|---------------|
| 0 | RISK_ON | Low VIX, positive trends, tight spreads | Lowest mean on first feature (VIX proxy) |
| 1 | NEUTRAL | Mixed signals, transitional | Middle state(s) |
| 2 | RISK_OFF | High VIX, negative trends, wide spreads | Highest mean on first feature |

### Feature Vector

The HMM observes a 9-dimensional feature vector:

```python
Features = [
    vix_level,           # VIX closing value
    vix_percentile,      # VIX relative to 252-day range
    realized_vol_20d,    # 20-day realized volatility
    ma_trend_50_200,     # 50-day MA vs 200-day MA signal
    yield_curve_2s10s,   # 2-year vs 10-year Treasury spread
    credit_spread_hy,    # High-yield credit spread
    momentum_1m,         # 1-month price momentum
    momentum_3m,         # 3-month price momentum
    rsi_14,              # 14-day Relative Strength Index
]
```

### Diagram: HMM State Transitions

```
                    +------------------+
                    |                  |
           a_00     v    a_01          |
        +--------> [RISK_ON] --------+ |
        |              ^             | |
        |              | a_10        | |
        |              |             v |
   +--------+     +--------+    +----------+
   |RISK_ON |<----|NEUTRAL |<---|RISK_OFF  |
   +--------+  a_21+--------+ a_12+----------+
        ^              |             |
        |              | a_11        |
        |              v             |
        +---------- [NEUTRAL] <------+
                    a_22     a_02

Note: Self-transitions (a_ii) typically dominate,
indicating regime persistence.
```

### Parameter Estimation

Parameters are estimated using the **Expectation-Maximization (EM)** algorithm:

**E-Step:** Compute expected state occupancies using forward-backward algorithm
**M-Step:** Update parameters to maximize expected log-likelihood

```python
# From src/signals/regime.py
self._model = GaussianHMM(
    n_components=self.n_states,           # K = 3
    covariance_type="full",               # Full covariance matrices
    n_iter=self.config.n_iterations,      # EM iterations (default: 100)
    random_state=self.config.random_state # Reproducibility
)
self._model.fit(features_standardized)
```

### Minimum Sample Requirements

For reliable HMM estimation, we require sufficient samples per parameter. The number of free parameters is:

```
n_params = (K - 1)                    # Initial distribution
         + K * (K - 1)                # Transition matrix
         + K * d                      # Means
         + K * d * (d + 1) / 2        # Full covariance (symmetric)
```

For K = 3 states and d = 9 features:

```
n_params = 2 + 6 + 27 + 135 = 170 parameters
```

With the rule of 10 samples per parameter:

```
min_samples = 170 * 10 = 1,700 samples
```

This corresponds to approximately 7 years of daily trading data.

### State-to-Regime Mapping

After fitting, states are mapped to regimes based on emission means:

```python
def _map_states_to_regimes(self) -> None:
    state_means = self._model.means_
    first_feature_means = state_means[:, 0]  # VIX-like feature
    sorted_state_indices = np.argsort(first_feature_means)

    # RISK_ON: lowest VIX -> sorted_state_indices[0]
    # RISK_OFF: highest VIX -> sorted_state_indices[-1]
    # NEUTRAL: middle states
```

---

## Regime-Based Allocation Rules

### Target Allocations by Regime

Each regime maps to a predefined target allocation:

```python
# From src/signals/allocation.py
REGIME_ALLOCATIONS = {
    Regime.RISK_ON:  {"LQQ": 0.15, "CL2": 0.15, "WPEA": 0.60, "CASH": 0.10},
    Regime.NEUTRAL:  {"LQQ": 0.10, "CL2": 0.10, "WPEA": 0.60, "CASH": 0.20},
    Regime.RISK_OFF: {"LQQ": 0.05, "CL2": 0.05, "WPEA": 0.60, "CASH": 0.30},
}
```

### Allocation Logic

| Regime | Leveraged (LQQ + CL2) | Core (WPEA) | Cash | Rationale |
|--------|----------------------|-------------|------|-----------|
| RISK_ON | 30% | 60% | 10% | Maximize leveraged exposure within limits |
| NEUTRAL | 20% | 60% | 20% | Balanced positioning, increased buffer |
| RISK_OFF | 10% | 60% | 30% | Defensive stance, maximize cash buffer |

### Confidence-Based Blending

When regime confidence is low, allocation is blended toward NEUTRAL:

```python
def _blend_allocations(
    self,
    target: dict[str, float],
    neutral: dict[str, float],
    confidence: float,
) -> dict[str, float]:
    """
    Formula: blended = confidence * target + (1 - confidence) * neutral
    """
    blended = {}
    for symbol in VALID_SYMBOLS:
        blended[symbol] = (
            confidence * target[symbol] + (1 - confidence) * neutral[symbol]
        )
    return blended
```

**Example:** RISK_ON with 70% confidence:

```
LQQ  = 0.70 * 0.15 + 0.30 * 0.10 = 0.135 (13.5%)
CL2  = 0.70 * 0.15 + 0.30 * 0.10 = 0.135 (13.5%)
WPEA = 0.70 * 0.60 + 0.30 * 0.60 = 0.60  (60%)
CASH = 0.70 * 0.10 + 0.30 * 0.20 = 0.13  (13%)
```

### Diagram: Allocation Decision Flow

```
                 +------------------+
                 | Feature Vector   |
                 | (9 dimensions)   |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 | HMM Prediction   |
                 | predict_regime() |
                 +--------+---------+
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
    +---------+     +---------+     +-----------+
    | RISK_ON |     | NEUTRAL |     | RISK_OFF  |
    | Prob    |     | Prob    |     | Prob      |
    +---------+     +---------+     +-----------+
          |               |               |
          +---------------+---------------+
                          |
                          v
                 +------------------+
                 | Regime Selection |
                 | (argmax prob)    |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 | Confidence       |
                 | = max(probs)     |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 | Blend toward     |
                 | NEUTRAL if       |
                 | confidence < 1.0 |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 | Risk Limit       |
                 | Validation       |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 | Final Allocation |
                 | Recommendation   |
                 +------------------+
```

---

## Rebalancing Methodology

### Drift-Based Triggering

Rebalancing is triggered when any position drifts more than 5% from its target:

```python
REBALANCE_THRESHOLD = 0.05

def needs_rebalancing(
    self,
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> bool:
    for symbol in all_symbols:
        drift = abs(current_weights.get(symbol, 0.0) - target_weights.get(symbol, 0.0))
        if drift > self.risk_limits.rebalance_threshold:
            return True
    return False
```

### Trade Generation

When rebalancing is needed, trades are generated:

```python
def calculate_trades(
    self,
    current: dict[str, float],
    target: dict[str, float],
    portfolio_value: Decimal,
) -> list[TradeRecommendation]:
    trades = []
    for symbol in tradeable_symbols:
        drift = abs(target_weight - current_weight)
        if drift > REBALANCE_THRESHOLD:
            trade_value = abs(weight_diff) * portfolio_value
            if trade_value >= MIN_TRADE_VALUE:
                trades.append(TradeRecommendation(...))
    return trades
```

### Trade Priority Ordering

Trades are ordered for optimal execution:

| Priority | Action | Symbol Type | Rationale |
|----------|--------|-------------|-----------|
| 1 | SELL | Leveraged | Reduce risk first |
| 2 | SELL | Regular | Free up cash |
| 3 | BUY | Regular | Build core position |
| 4 | BUY | Leveraged | Add risk last |

```python
class TradePriority(int, Enum):
    SELL_LEVERAGED = 1
    SELL_REGULAR = 2
    BUY_REGULAR = 3
    BUY_LEVERAGED = 4
```

### Transaction Cost Estimation

```python
def estimate_transaction_costs(self, trades: list[TradeRecommendation]) -> Decimal:
    total_cost = Decimal("0.0")
    for trade in trades:
        commission = trade.estimated_value * 0.001  # 0.1%
        spread = trade.estimated_value * 0.001      # 0.1%
        total_cost += commission + spread
    return total_cost
```

---

## Walk-Forward Validation

### Methodology

Walk-forward validation prevents look-ahead bias by ensuring:

1. Training data precedes testing data
2. No overlap between training and testing periods
3. Model parameters are frozen during testing

### Timeline Structure

```
Window 0: Train [2015-01-01 to 2020-01-01] -> Test [2020-01-01 to 2021-01-01]
Window 1: Train [2015-07-01 to 2020-07-01] -> Test [2020-07-01 to 2021-07-01]
Window 2: Train [2016-01-01 to 2021-01-01] -> Test [2021-01-01 to 2022-01-01]
...
```

### Diagram: Walk-Forward Timeline

```
2015        2016        2017        2018        2019        2020        2021        2022        2023        2024
  |           |           |           |           |           |           |           |           |           |
  [=================== TRAIN 0 ===================]
                                                  [=== TEST 0 ===]
        [=================== TRAIN 1 ===================]
                                                        [=== TEST 1 ===]
              [=================== TRAIN 2 ===================]
                                                              [=== TEST 2 ===]
```

### Configuration

```python
# From src/backtesting/walk_forward.py
class WalkForwardConfig(BaseModel):
    train_years: int = 7           # Training window length
    test_years: int = 1            # Testing window length
    step_months: int = 6           # Step size between windows
    min_training_samples: int = 1700  # Minimum samples for HMM
    execution_delay_days: int = 1  # Signal -> execution delay
```

### Look-Ahead Bias Prevention

```python
def validate_no_lookahead(
    self,
    window: WalkForwardWindow,
    features_date: date,
    prediction_date: date,
) -> bool:
    # Features date must be on or before prediction date
    if features_date > prediction_date:
        raise LookaheadBiasError(...)

    # Prediction date must be within test period
    if prediction_date < window.test_start or prediction_date > window.test_end:
        raise LookaheadBiasError(...)

    return True
```

### Execution Timing Rule

```
Signal generated: Day t (at market close)
Trade executed:   Day t+1 (at market open)
```

This prevents using information that would not have been available in real-time trading.

---

## Expected Performance Characteristics

### Risk-Adjusted Returns

Based on the methodology design, expected characteristics:

| Metric | Expectation | Rationale |
|--------|-------------|-----------|
| Sharpe Ratio | 0.5 - 1.0 | Regime awareness should improve risk-adjusted returns |
| Max Drawdown | -25% to -35% | Leveraged exposure limits bound worst-case |
| Volatility | 15% - 25% | Higher than index due to leverage, lower than pure leveraged ETF |

### Regime Detection Accuracy

Typical HMM performance on financial data:

- **In-sample accuracy:** 70-85% (state persistence is high)
- **Out-of-sample accuracy:** 55-65% (forward prediction is harder)
- **Value added:** Comes from avoiding large drawdowns, not prediction accuracy

### Transaction Costs Impact

With 5% rebalance threshold and quarterly expected rebalancing:

```
Expected annual trades: 4-8
Average trade size: 10-15% of portfolio
Transaction costs: 0.1-0.2% of portfolio per year
```

### Volatility Decay Drag

From leveraged ETF exposure (see Risk Limits documentation):

```
Expected decay at 30% leveraged allocation: ~1.5% per year
Expected decay at 10% leveraged allocation: ~0.5% per year
```

### Model Degradation

HMM parameters may become stale over time. Recommended retraining frequency:

```
Full retrain: Annually or when significant regime shift detected
Rolling window: New data added continuously
Validation: Run walk-forward validation before deploying new parameters
```

---

## Implementation Details

### Key Files

| File | Purpose |
|------|---------|
| `src/signals/regime.py` | HMM regime detector implementation |
| `src/signals/allocation.py` | Allocation optimizer with risk limits |
| `src/signals/features.py` | Feature engineering |
| `src/portfolio/rebalancer.py` | Trade generation and optimization |
| `src/backtesting/walk_forward.py` | Walk-forward validation framework |

### Core Classes

**RegimeDetector** (`src/signals/regime.py`):
- `fit(features)`: Train HMM on historical data
- `predict_regime(features)`: Get current regime
- `predict_regime_probabilities(features)`: Get probability distribution
- `get_transition_matrix()`: Access learned transition probabilities

**AllocationOptimizer** (`src/signals/allocation.py`):
- `get_target_allocation(regime, confidence)`: Generate allocation recommendation
- `validate_allocation(weights)`: Check risk limits
- `calculate_rebalance_trades(current, target, value)`: Generate trades
- `needs_rebalancing(current, target)`: Check drift threshold

**Rebalancer** (`src/portfolio/rebalancer.py`):
- `calculate_trades(current, target, value, prices)`: Full trade calculation
- `optimize_trade_order(trades)`: Order for execution
- `estimate_transaction_costs(trades)`: Cost estimation
- `generate_rebalance_report(...)`: Comprehensive reporting

### Error Handling

The system defines specific exceptions:

```python
class NotFittedError(RegimeDetectorError):
    """Raised when prediction attempted before fitting."""

class InsufficientSamplesError(Exception):
    """Raised when training data has insufficient samples."""

class LookaheadBiasError(Exception):
    """Raised when look-ahead bias is detected."""

class AllocationError(Exception):
    """Raised for allocation validation failures."""
```

---

## Conclusion

The allocation methodology combines:

1. **Statistical Foundation:** HMM provides a principled framework for regime detection
2. **Risk Discipline:** Hard-coded limits ensure safety regardless of model output
3. **Practical Implementation:** Transaction costs and execution timing are considered
4. **Validation Rigor:** Walk-forward testing prevents overfitting and look-ahead bias

The approach aims to add value not through prediction accuracy, but through:
- Reducing exposure during high-risk periods
- Maintaining discipline through programmatic enforcement
- Capturing regime persistence without overtrading

---

**Document Version:** 1.0
**Last Updated:** 2025-12-12
**Author:** Portfolio Management System
**Review Cycle:** Annual or upon material methodology changes
