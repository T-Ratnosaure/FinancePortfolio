# Integration Tests for FinancePortfolio

This directory contains comprehensive integration tests that verify end-to-end workflows across multiple components of the FinancePortfolio system.

## Test Coverage

### 1. Data-to-Signal Pipeline (`test_data_to_signal_pipeline.py`)
**6 tests - Verifies the complete flow from market data to allocation recommendations**

Tests cover:
- End-to-end pipeline: data → features → regime detection → allocation
- Regime-specific behavior (RISK_ON, NEUTRAL, RISK_OFF)
- Missing data handling with forward-fill
- Feature validation and consistency checks
- Confidence blending effects on allocation

**Key Scenarios:**
- `test_end_to_end_pipeline_with_mock_data`: Primary integration test verifying all components work together
- `test_pipeline_handles_risk_on_regime`: Validates aggressive allocation in low VIX environments
- `test_pipeline_handles_risk_off_regime`: Validates defensive allocation in high VIX environments
- `test_pipeline_handles_missing_data_gracefully`: Ensures robustness to data gaps
- `test_pipeline_feature_consistency`: Validates feature ranges and data quality
- `test_pipeline_regime_probability_confidence`: Verifies confidence blending logic

---

### 2. Signal-to-Portfolio Pipeline (`test_signal_to_portfolio_pipeline.py`)
**10 tests - Verifies signal processing and portfolio rebalancing**

Tests cover:
- Complete rebalancing workflow from regime signal to trade orders
- Regime change detection and response
- Risk limit enforcement (leveraged exposure, position size, cash buffer)
- Rebalancing threshold logic to prevent excessive trading
- Confidence effects on portfolio positioning
- Custom risk limits configuration
- Trade execution validation

**Key Scenarios:**
- `test_signal_to_rebalance_full_pipeline`: End-to-end rebalancing workflow
- `test_regime_change_triggers_rebalance`: Market shift response (RISK_ON → RISK_OFF)
- `test_risk_limits_enforcement`: Hard constraint validation (30% leveraged cap)
- `test_max_leveraged_exposure_constant_respected`: Constant verification across all regimes
- `test_rebalance_threshold_prevents_excessive_trading`: 5% drift threshold validation
- `test_confidence_blending_affects_portfolio`: Low confidence → neutral positioning
- `test_cash_buffer_enforcement`: Minimum 10% cash buffer validation
- `test_custom_risk_limits`: Institutional tighter limits support
- `test_rebalance_trades_are_executable`: Mathematical correctness of trade calculations
- `test_no_trades_when_already_balanced`: Avoid unnecessary trading costs

---

### 3. Backtesting Integration (`test_backtesting_integration.py`)
**8 tests - Verifies walk-forward validation and backtesting workflows**

Tests cover:
- Walk-forward window generation and validation
- Look-ahead bias prevention
- Execution timing constraints (T+1 execution)
- Training data validation
- Multi-window backtesting consistency
- Performance metrics calculation
- Regime persistence and transition dynamics

**Key Scenarios:**
- `test_walk_forward_window_generation`: Proper window creation with no overlap
- `test_backtest_with_simple_data`: Basic backtest execution workflow
- `test_walk_forward_prevents_lookahead_bias`: Critical bias prevention validation
- `test_execution_timing_validation`: T+1 execution enforcement (no same-day trades)
- `test_training_data_validation`: Catches common data leakage errors
- `test_multiple_window_backtest`: Consistency across expanding windows
- `test_performance_metrics_calculation`: Metrics computation from backtest results
- `test_regime_persistence_across_windows`: Transition matrix validation

---

### 4. Risk Management Integration (`test_risk_management_integration.py`)
**13 tests - Verifies risk controls and monitoring**

Tests cover:
- Position limit enforcement in allocation system
- VaR calculation with realistic return distributions
- Leveraged exposure limits (MAX_LEVERAGED_EXPOSURE = 30%)
- Drawdown alert generation (threshold: -20%)
- Comprehensive risk report generation
- Cash buffer enforcement (MIN_CASH_BUFFER = 10%)
- Portfolio volatility with covariance matrix
- Sharpe and Sortino ratio calculations
- Leveraged ETF decay estimation
- Risk alert generation for limit breaches
- Insufficient data error handling
- Integration between risk and allocation systems

**Key Scenarios:**
- `test_position_limits_enforced_in_allocation`: MAX_SINGLE_POSITION verification
- `test_var_calculation_with_realistic_data`: Historical and parametric VaR
- `test_exposure_limits_enforcement`: Hard 30% leveraged cap
- `test_drawdown_alert_generation`: -20% threshold alert verification
- `test_risk_report_generation_with_positions`: Comprehensive report validation
- `test_cash_buffer_enforcement`: 10% minimum cash validation
- `test_portfolio_volatility_calculation`: Covariance-based risk calculation
- `test_sharpe_ratio_calculation`: Risk-adjusted return metrics
- `test_sortino_ratio_calculation`: Downside deviation metrics
- `test_leveraged_decay_estimation`: 2x ETF volatility decay
- `test_risk_alerts_for_excessive_leverage`: Alert generation for breaches
- `test_insufficient_data_handling`: Graceful error handling
- `test_risk_integration_with_allocation`: System-level consistency check

---

## Test Data and Fixtures

### Shared Fixtures (`conftest.py`)

All integration tests use realistic mock data generated with proper statistical properties:

1. **`mock_price_data`**: 10 years (2,500 trading days) of daily OHLCV data for LQQ.PA, CL2.PA, WPEA.PA
   - Realistic drift (7.5% annual)
   - Higher volatility for leveraged ETFs (3% daily) vs unleveraged (1.5% daily)
   - Random seed for reproducibility

2. **`mock_vix_data`**: 10 years of VIX data with regime patterns
   - Low regime: ~12 VIX (RISK_ON)
   - Neutral regime: ~18 VIX
   - High regime: ~30 VIX (RISK_OFF)
   - Switches regimes every 500 days

3. **`mock_treasury_data`**: 10 years of 2Y and 10Y Treasury yields
   - Normal yield curve (10Y > 2Y)
   - Realistic spread dynamics

4. **`mock_hy_spread_data`**: 10 years of high-yield credit spreads
   - Typical range: 2-6%
   - Mean reversion dynamics

5. **`mock_daily_prices`**: Pydantic DailyPrice models (300 days)

6. **`mock_macro_indicators`**: Pydantic MacroIndicator models (300 days)

---

## Running the Tests

### Run all integration tests:
```bash
uv run pytest tests/test_integration/ -v
```

### Run specific test file:
```bash
uv run pytest tests/test_integration/test_data_to_signal_pipeline.py -v
```

### Run specific test:
```bash
uv run pytest tests/test_integration/test_risk_management_integration.py::TestRiskManagementIntegration::test_var_calculation_with_realistic_data -v
```

### Run with coverage:
```bash
uv run pytest tests/test_integration/ --cov=src --cov-report=html
```

---

## Test Execution Time

- **Total**: ~87 seconds for 37 tests
- **Average**: ~2.4 seconds per test
- **Slowest**: Data-to-signal pipeline tests (~10-15s each due to HMM training)

---

## Key Constants Verified

All integration tests verify these critical constants from `src/data/models.py`:

- `MAX_LEVERAGED_EXPOSURE = 0.30` (30% combined LQQ + CL2)
- `MAX_SINGLE_POSITION = 0.25` (25% per leveraged ETF)
- `MIN_CASH_BUFFER = 0.10` (10% minimum cash)
- `REBALANCE_THRESHOLD = 0.05` (5% drift triggers rebalance)
- `DRAWDOWN_ALERT = -0.20` (-20% drawdown threshold)

---

## PEA ETFs Used in Tests

- **LQQ.PA**: Leveraged NASDAQ 100 (2x daily leverage)
- **CL2.PA**: Leveraged S&P 500 (2x daily leverage)
- **WPEA.PA**: World equity unleveraged (core holding)

---

## Integration Test Design Principles

1. **End-to-End Validation**: Each test file covers complete workflows, not just individual functions
2. **Realistic Data**: Mock data has proper statistical properties (drift, volatility, correlations)
3. **Edge Cases**: Tests include missing data, extreme volatility, regime shifts
4. **Risk Enforcement**: All tests verify hard-coded risk limits are respected
5. **No Look-Ahead Bias**: Backtesting tests strictly enforce temporal ordering
6. **Deterministic**: Fixed random seeds ensure reproducible results
7. **Fast Execution**: Tests optimized for CI/CD (< 90 seconds total)

---

## What Integration Tests DON'T Cover

These tests focus on integration workflows. Unit tests in other directories cover:
- Individual function correctness (`tests/test_signals/`, `tests/test_portfolio/`, etc.)
- Mathematical formula verification
- Boundary condition handling
- Error message content

---

## Future Enhancements

Potential additions for Sprint 6+:
- Integration with real broker APIs (paper trading)
- Database persistence layer integration
- Multi-currency support integration tests
- Tax optimization integration (PEA limits, capital gains)
- Notification system integration (email/SMS alerts)
