"""Tests for backtesting metrics module.

Tests use known values to validate mathematical correctness of all metrics.
Special attention is given to the Sortino ratio formula which was corrected
in Sprint 4 to use the proper downside deviation calculation.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.metrics import (
    TRADING_DAYS_PER_YEAR,
    BacktestMetrics,
    Regime,
    RegimePerformance,
)


@pytest.fixture
def metrics() -> BacktestMetrics:
    """Create a BacktestMetrics instance."""
    return BacktestMetrics()


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    """Generate sample equity curve for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    # Start at 100, drift up with volatility
    returns = 0.0004 + 0.01 * np.random.randn(252)
    equity = 100 * np.cumprod(1 + returns)
    return pd.Series(equity, index=dates)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Generate sample daily returns."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    # Returns with mean ~0.04% daily, vol ~1% daily
    returns = 0.0004 + 0.01 * np.random.randn(252)
    return pd.Series(returns, index=dates)


@pytest.fixture
def sample_benchmark_returns() -> pd.Series:
    """Generate sample benchmark returns."""
    np.random.seed(123)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    returns = 0.0003 + 0.008 * np.random.randn(252)
    return pd.Series(returns, index=dates)


# =============================================================================
# Performance Metrics Tests
# =============================================================================


class TestTotalReturn:
    """Tests for total_return calculation."""

    def test_total_return_known_value(self, metrics: BacktestMetrics) -> None:
        """Test total return with known expected value."""
        # 100 -> 110 = 10% return
        equity = pd.Series([100.0, 105.0, 110.0])
        result = metrics.total_return(equity)
        assert abs(result - 0.10) < 0.001

    def test_total_return_negative(self, metrics: BacktestMetrics) -> None:
        """Test negative total return."""
        # 100 -> 90 = -10% return
        equity = pd.Series([100.0, 95.0, 90.0])
        result = metrics.total_return(equity)
        assert abs(result - (-0.10)) < 0.001

    def test_total_return_empty_raises(self, metrics: BacktestMetrics) -> None:
        """Test that empty equity curve raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            metrics.total_return(pd.Series(dtype=float))

    def test_total_return_negative_values_raises(
        self, metrics: BacktestMetrics
    ) -> None:
        """Test that negative equity values raise error."""
        equity = pd.Series([100.0, -50.0, 80.0])
        with pytest.raises(ValueError, match="positive values"):
            metrics.total_return(equity)


class TestAnnualizedReturn:
    """Tests for annualized_return calculation."""

    def test_annualized_return_one_year(self, metrics: BacktestMetrics) -> None:
        """Test annualized return for exactly one year."""
        # 10% over 252 days = 10% annualized
        result = metrics.annualized_return(0.10, 252)
        assert abs(result - 0.10) < 0.001

    def test_annualized_return_half_year(self, metrics: BacktestMetrics) -> None:
        """Test annualized return for half year."""
        # 5% over 126 days = ~10.25% annualized (compound)
        result = metrics.annualized_return(0.05, 126)
        expected = (1.05) ** 2 - 1  # ~10.25%
        assert abs(result - expected) < 0.001

    def test_annualized_return_zero_days_raises(self, metrics: BacktestMetrics) -> None:
        """Test that zero days raises error."""
        with pytest.raises(ValueError, match="positive"):
            metrics.annualized_return(0.10, 0)

    def test_annualized_return_total_loss_raises(
        self, metrics: BacktestMetrics
    ) -> None:
        """Test that -100% or worse raises error."""
        with pytest.raises(ValueError, match="> -100%"):
            metrics.annualized_return(-1.0, 252)


class TestVolatility:
    """Tests for volatility calculation."""

    def test_volatility_known_value(self, metrics: BacktestMetrics) -> None:
        """Test volatility with known daily std."""
        # Create returns with known daily std of 1%
        np.random.seed(42)
        returns = pd.Series(0.01 * np.random.randn(252))
        result = metrics.volatility(returns)

        # Should be close to 1% * sqrt(252) = ~15.87%
        expected_approx = 0.01 * np.sqrt(TRADING_DAYS_PER_YEAR)
        assert abs(result - expected_approx) < 0.02

    def test_volatility_zero_returns(self, metrics: BacktestMetrics) -> None:
        """Test volatility with zero variance returns."""
        returns = pd.Series([0.01] * 50)
        result = metrics.volatility(returns)
        assert result == 0.0

    def test_volatility_insufficient_data(self, metrics: BacktestMetrics) -> None:
        """Test volatility with insufficient data raises error."""
        returns = pd.Series([0.01])
        with pytest.raises(ValueError, match="at least 2"):
            metrics.volatility(returns)


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_ratio_positive(
        self, metrics: BacktestMetrics, sample_returns: pd.Series
    ) -> None:
        """Test Sharpe ratio calculation produces reasonable value."""
        result = metrics.sharpe_ratio(sample_returns, risk_free_rate=0.02)
        # Should be in reasonable range
        assert -3.0 < result < 5.0

    def test_sharpe_ratio_higher_rf_lower_sharpe(
        self, metrics: BacktestMetrics, sample_returns: pd.Series
    ) -> None:
        """Test that higher risk-free rate lowers Sharpe."""
        sharpe_low_rf = metrics.sharpe_ratio(sample_returns, risk_free_rate=0.0)
        sharpe_high_rf = metrics.sharpe_ratio(sample_returns, risk_free_rate=0.05)
        assert sharpe_high_rf < sharpe_low_rf

    def test_sharpe_ratio_insufficient_data(self, metrics: BacktestMetrics) -> None:
        """Test Sharpe with insufficient data raises error."""
        returns = pd.Series([0.01])
        with pytest.raises(ValueError, match="at least 2"):
            metrics.sharpe_ratio(returns)


class TestSortinoRatio:
    """Tests for Sortino ratio with CORRECT formula.

    The correct Sortino formula uses downside deviation:
    DD = sqrt(mean((r_i - target)^2 for all r_i < target))

    This divides by total n observations, NOT just downside count.
    """

    def test_sortino_ratio_correct_formula(self, metrics: BacktestMetrics) -> None:
        """Test Sortino ratio uses correct downside deviation formula.

        This validates the fix for Sprint 4 issue where wrong formula was:
        std(negative_returns_only)

        Correct formula:
        sqrt((1/n) * sum((r_i - target)^2 for r_i < target))
        """
        # Create known returns: 5 values, 2 negative
        # Returns: [0.02, -0.01, 0.03, -0.02, 0.01]
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        target = 0.0

        # Manual calculation:
        # Mean return = 0.006
        # Downside squared: [0, 0.0001, 0, 0.0004, 0]
        # Sum = 0.0005
        # Downside variance = 0.0005 / 5 = 0.0001
        # Downside deviation = sqrt(0.0001) = 0.01
        # Daily Sortino = 0.006 / 0.01 = 0.6
        # Annualized = 0.6 * sqrt(252) = ~9.52

        expected_dd = np.sqrt(0.0001)  # 0.01
        expected_daily_sortino = 0.006 / expected_dd  # 0.6
        expected_annualized = expected_daily_sortino * np.sqrt(252)

        result = metrics.sortino_ratio(returns, target=target)
        assert abs(result - expected_annualized) < 0.1

    def test_sortino_vs_wrong_formula(self, metrics: BacktestMetrics) -> None:
        """Test that Sortino differs from the wrong formula.

        The wrong formula was: std(negative_returns_only)
        Our correct formula should give different results.
        """
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.015])
        target = 0.0

        # Calculate with our correct formula
        correct_sortino = metrics.sortino_ratio(returns, target=target)

        # Calculate with WRONG formula for comparison
        negative_returns = returns[returns < target]
        wrong_dd = float(negative_returns.std(ddof=1))
        mean_excess = float(returns.mean() - target)
        wrong_sortino = (mean_excess / wrong_dd) * np.sqrt(252)

        # They should be different (wrong formula divides by count of negatives)
        # This test documents that we use the correct formula
        assert correct_sortino != pytest.approx(wrong_sortino, rel=0.1)

    def test_sortino_no_downside_returns(self, metrics: BacktestMetrics) -> None:
        """Test Sortino when all returns are positive."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.015])
        result = metrics.sortino_ratio(returns, target=0.0)
        # No downside returns - should return 0
        assert result == 0.0

    def test_sortino_all_negative_returns(self, metrics: BacktestMetrics) -> None:
        """Test Sortino when all returns are negative."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.03, -0.015])
        result = metrics.sortino_ratio(returns, target=0.0)
        # Should be negative since mean is negative
        assert result < 0


class TestMaxDrawdown:
    """Tests for max_drawdown calculation."""

    def test_max_drawdown_known_value(self, metrics: BacktestMetrics) -> None:
        """Test max drawdown with known expected value."""
        # Prices: 100 -> 120 -> 90 -> 100
        # Peak = 120, Trough = 90, DD = (90-120)/120 = -25%
        equity = pd.Series([100.0, 120.0, 90.0, 100.0])
        max_dd, peak_idx, trough_idx = metrics.max_drawdown(equity)

        assert abs(max_dd - (-0.25)) < 0.001
        assert peak_idx == 1  # Index of 120
        assert trough_idx == 2  # Index of 90

    def test_max_drawdown_monotonic_increase(self, metrics: BacktestMetrics) -> None:
        """Test drawdown is zero for monotonically increasing equity."""
        equity = pd.Series([100.0, 110.0, 120.0, 130.0])
        max_dd, peak_idx, trough_idx = metrics.max_drawdown(equity)

        assert max_dd == 0.0
        assert peak_idx == 0
        assert trough_idx == 0

    def test_max_drawdown_empty_raises(self, metrics: BacktestMetrics) -> None:
        """Test error for empty equity curve."""
        with pytest.raises(ValueError, match="cannot be empty"):
            metrics.max_drawdown(pd.Series(dtype=float))


class TestMaxDrawdownDuration:
    """Tests for max_drawdown_duration calculation."""

    def test_drawdown_duration_known_value(self, metrics: BacktestMetrics) -> None:
        """Test drawdown duration with known period."""
        # Drawdown for 3 days, then recovery
        equity = pd.Series([100.0, 110.0, 100.0, 95.0, 90.0, 110.0, 120.0])
        result = metrics.max_drawdown_duration(equity)
        # Days in drawdown: index 2, 3, 4 (3 days)
        assert result == 3

    def test_drawdown_duration_no_recovery(self, metrics: BacktestMetrics) -> None:
        """Test drawdown that never recovers."""
        equity = pd.Series([100.0, 110.0, 100.0, 90.0, 85.0])
        result = metrics.max_drawdown_duration(equity)
        # Drawdown from index 2 onwards (3 days)
        assert result == 3


class TestWinRate:
    """Tests for win_rate calculation."""

    def test_win_rate_known_value(self, metrics: BacktestMetrics) -> None:
        """Test win rate with known value."""
        # 3 positive, 2 negative = 60% win rate
        monthly_returns = pd.Series([0.05, -0.02, 0.03, 0.01, -0.01])
        result = metrics.win_rate(monthly_returns)
        assert abs(result - 0.60) < 0.001

    def test_win_rate_all_positive(self, metrics: BacktestMetrics) -> None:
        """Test win rate when all months positive."""
        monthly_returns = pd.Series([0.01, 0.02, 0.03])
        result = metrics.win_rate(monthly_returns)
        assert result == 1.0

    def test_win_rate_empty_raises(self, metrics: BacktestMetrics) -> None:
        """Test error for empty returns."""
        with pytest.raises(ValueError, match="cannot be empty"):
            metrics.win_rate(pd.Series(dtype=float))


class TestProfitFactor:
    """Tests for profit_factor calculation."""

    def test_profit_factor_known_value(self, metrics: BacktestMetrics) -> None:
        """Test profit factor with known value."""
        # Gross profit = 0.05 + 0.03 = 0.08
        # Gross loss = 0.02 + 0.01 = 0.03
        # PF = 0.08 / 0.03 = 2.67
        returns = pd.Series([0.05, -0.02, 0.03, -0.01])
        result = metrics.profit_factor(returns)
        assert abs(result - 2.6667) < 0.01

    def test_profit_factor_no_losses(self, metrics: BacktestMetrics) -> None:
        """Test profit factor with no losses."""
        returns = pd.Series([0.01, 0.02, 0.03])
        result = metrics.profit_factor(returns)
        assert result == float("inf")


# =============================================================================
# Risk Metrics Tests
# =============================================================================


class TestValueAtRisk:
    """Tests for value_at_risk calculation."""

    def test_var_95_reasonable_range(
        self, metrics: BacktestMetrics, sample_returns: pd.Series
    ) -> None:
        """Test VaR 95% is in reasonable range."""
        result = metrics.value_at_risk(sample_returns, confidence=0.95)
        # For ~1% daily vol, 95% VaR should be roughly 1.5-2.5%
        assert 0.005 < result < 0.05

    def test_var_99_greater_than_95(
        self, metrics: BacktestMetrics, sample_returns: pd.Series
    ) -> None:
        """Test 99% VaR is larger than 95% VaR."""
        var_95 = metrics.value_at_risk(sample_returns, confidence=0.95)
        var_99 = metrics.value_at_risk(sample_returns, confidence=0.99)
        assert var_99 > var_95

    def test_var_positive(
        self, metrics: BacktestMetrics, sample_returns: pd.Series
    ) -> None:
        """Test VaR is always positive (represents loss)."""
        result = metrics.value_at_risk(sample_returns, confidence=0.95)
        assert result >= 0

    def test_var_invalid_confidence(
        self, metrics: BacktestMetrics, sample_returns: pd.Series
    ) -> None:
        """Test invalid confidence raises error."""
        with pytest.raises(ValueError, match="Confidence must be"):
            metrics.value_at_risk(sample_returns, confidence=1.5)

    def test_var_insufficient_data(self, metrics: BacktestMetrics) -> None:
        """Test insufficient data raises error."""
        returns = pd.Series([0.01] * 10)
        with pytest.raises(ValueError, match="at least 30"):
            metrics.value_at_risk(returns)


class TestExpectedShortfall:
    """Tests for expected_shortfall (CVaR) calculation."""

    def test_es_greater_than_var(
        self, metrics: BacktestMetrics, sample_returns: pd.Series
    ) -> None:
        """Test Expected Shortfall >= VaR (by definition)."""
        var = metrics.value_at_risk(sample_returns, confidence=0.95)
        es = metrics.expected_shortfall(sample_returns, confidence=0.95)
        assert es >= var

    def test_es_reasonable_range(
        self, metrics: BacktestMetrics, sample_returns: pd.Series
    ) -> None:
        """Test ES is in reasonable range."""
        result = metrics.expected_shortfall(sample_returns, confidence=0.95)
        assert 0.005 < result < 0.10


class TestBetaToBenchmark:
    """Tests for beta_to_benchmark calculation."""

    def test_beta_leveraged_relationship(self, metrics: BacktestMetrics) -> None:
        """Test beta for known leveraged relationship."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")

        benchmark = pd.Series(0.0005 + 0.01 * np.random.randn(100), index=dates)
        # Portfolio with 2x leverage
        portfolio = benchmark * 2 + 0.001 * np.random.randn(100)

        result = metrics.beta_to_benchmark(portfolio, benchmark)
        # Should be close to 2.0
        assert 1.5 < result < 2.5

    def test_beta_insufficient_data(self, metrics: BacktestMetrics) -> None:
        """Test beta with insufficient data raises error."""
        port = pd.Series([0.01] * 10)
        bench = pd.Series([0.005] * 10)
        with pytest.raises(ValueError, match="at least 20"):
            metrics.beta_to_benchmark(port, bench)


class TestTrackingError:
    """Tests for tracking_error calculation."""

    def test_tracking_error_known_value(self, metrics: BacktestMetrics) -> None:
        """Test tracking error with known difference."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")

        benchmark = pd.Series(0.0005 + 0.01 * np.random.randn(100), index=dates)
        # Portfolio with small tracking difference
        active_noise = 0.002 * np.random.randn(100)
        portfolio = benchmark + active_noise

        result = metrics.tracking_error(pd.Series(portfolio, index=dates), benchmark)

        # Annualized tracking error should be ~0.002 * sqrt(252) = ~3.2%
        expected_approx = 0.002 * np.sqrt(252)
        assert abs(result - expected_approx) < 0.02

    def test_tracking_error_identical_returns(self, metrics: BacktestMetrics) -> None:
        """Test tracking error is zero for identical returns."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
        returns = pd.Series([0.01] * 100, index=dates)

        result = metrics.tracking_error(returns, returns)
        assert result == 0.0


class TestDrawdownSeries:
    """Tests for calculate_drawdown_series."""

    def test_drawdown_series_values(self, metrics: BacktestMetrics) -> None:
        """Test drawdown series has correct values."""
        equity = pd.Series([100.0, 120.0, 90.0, 100.0])
        result = metrics.calculate_drawdown_series(equity)

        assert result.iloc[0] == 0.0  # At first value, no drawdown
        assert result.iloc[1] == 0.0  # At new high
        assert abs(result.iloc[2] - (-0.25)) < 0.001  # -25% drawdown
        assert abs(result.iloc[3] - (-0.167)) < 0.01  # Partial recovery


# =============================================================================
# Regime-Specific Metrics Tests
# =============================================================================


class TestRegimePerformance:
    """Tests for RegimePerformance model."""

    def test_regime_performance_creation(self) -> None:
        """Test RegimePerformance model creation."""
        rp = RegimePerformance(
            regime=Regime.RISK_ON,
            annualized_return=0.12,
            volatility=0.15,
            sharpe_ratio=0.8,
            max_drawdown=-0.10,
            days_in_regime=100,
        )
        assert rp.regime == Regime.RISK_ON
        assert rp.annualized_return == 0.12
        assert rp.volatility == 0.15
        assert rp.days_in_regime == 100


class TestCalculateRegimePerformance:
    """Tests for calculate_regime_performance."""

    def test_regime_performance_multiple_regimes(
        self, metrics: BacktestMetrics
    ) -> None:
        """Test regime performance calculation with multiple regimes."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=150, freq="B")

        # Create returns
        returns = pd.Series(0.0005 + 0.01 * np.random.randn(150), index=dates)

        # Create regime history
        regimes = ["RISK_ON"] * 50 + ["NEUTRAL"] * 50 + ["RISK_OFF"] * 50
        regime_history = pd.Series(regimes, index=dates)

        result = metrics.calculate_regime_performance(returns, regime_history)

        # Should have entries for all three regimes
        assert "RISK_ON" in result
        assert "NEUTRAL" in result
        assert "RISK_OFF" in result

        # Check structure
        assert isinstance(result["RISK_ON"], RegimePerformance)
        assert result["RISK_ON"].days_in_regime == 50

    def test_regime_performance_single_regime(self, metrics: BacktestMetrics) -> None:
        """Test regime performance with only one regime."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
        returns = pd.Series([0.001] * 100, index=dates)
        regimes = pd.Series(["RISK_ON"] * 100, index=dates)

        result = metrics.calculate_regime_performance(returns, regimes)

        assert result["RISK_ON"].days_in_regime == 100
        # NEUTRAL and RISK_OFF should have 0 days
        assert result["NEUTRAL"].days_in_regime == 0
        assert result["RISK_OFF"].days_in_regime == 0

    def test_regime_performance_empty_raises(self, metrics: BacktestMetrics) -> None:
        """Test empty series raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            metrics.calculate_regime_performance(
                pd.Series(dtype=float),
                pd.Series(dtype=str),
            )


# =============================================================================
# Statistical Validation Tests
# =============================================================================


class TestTTestReturns:
    """Tests for t_test_returns."""

    def test_t_test_positive_returns(self, metrics: BacktestMetrics) -> None:
        """Test t-test with significantly positive returns."""
        np.random.seed(42)
        # Returns with clear positive mean
        returns = pd.Series(0.002 + 0.005 * np.random.randn(100))

        t_stat, p_value = metrics.t_test_returns(returns)

        # Should have positive t-stat and low p-value
        assert t_stat > 0
        assert p_value < 0.05

    def test_t_test_zero_mean_returns(self, metrics: BacktestMetrics) -> None:
        """Test t-test with zero-mean returns."""
        np.random.seed(42)
        returns = pd.Series(0.01 * np.random.randn(100))

        t_stat, p_value = metrics.t_test_returns(returns)

        # p-value should be higher (not significant)
        # t_stat can be positive or negative (random)
        assert isinstance(t_stat, float)
        assert 0 <= p_value <= 1

    def test_t_test_insufficient_data(self, metrics: BacktestMetrics) -> None:
        """Test t-test with insufficient data raises error."""
        returns = pd.Series([0.01] * 20)
        with pytest.raises(ValueError, match="at least 30"):
            metrics.t_test_returns(returns)


class TestJarqueBeraTest:
    """Tests for jarque_bera_test."""

    def test_jarque_bera_normal_returns(self, metrics: BacktestMetrics) -> None:
        """Test Jarque-Bera with normal-like returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500))

        jb_stat, p_value = metrics.jarque_bera_test(returns)

        # Normal returns should have higher p-value (fail to reject)
        assert isinstance(jb_stat, float)
        assert 0 <= p_value <= 1

    def test_jarque_bera_fat_tails(self, metrics: BacktestMetrics) -> None:
        """Test Jarque-Bera with fat-tailed returns."""
        np.random.seed(42)
        # t-distribution has fat tails
        returns = pd.Series(np.random.standard_t(3, 500) * 0.01)

        jb_stat, p_value = metrics.jarque_bera_test(returns)

        # Fat-tailed returns should have low p-value (reject normality)
        assert jb_stat > 0
        assert p_value < 0.05


class TestRunsTest:
    """Tests for runs_test."""

    def test_runs_test_random_returns(self, metrics: BacktestMetrics) -> None:
        """Test runs test with random returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100))

        z_stat, p_value = metrics.runs_test(returns)

        # Random returns should have high p-value (fail to reject)
        assert isinstance(z_stat, float)
        assert 0 <= p_value <= 1

    def test_runs_test_trending_returns(self, metrics: BacktestMetrics) -> None:
        """Test runs test with trending (autocorrelated) returns."""
        # Create returns with positive autocorrelation (momentum)
        # This should have fewer runs than random
        returns = pd.Series([0.01] * 20 + [-0.01] * 20 + [0.01] * 20 + [-0.01] * 20)

        z_stat, p_value = metrics.runs_test(returns)

        # Should detect non-randomness (low p-value)
        assert z_stat < 0  # Too few runs (negative z)
        assert p_value < 0.05

    def test_runs_test_all_same_sign(self, metrics: BacktestMetrics) -> None:
        """Test runs test when all returns same sign."""
        returns = pd.Series([0.01] * 50)

        z_stat, p_value = metrics.runs_test(returns)

        # Cannot compute meaningful test
        assert z_stat == 0.0
        assert p_value == 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetricsIntegration:
    """Integration tests combining multiple metrics."""

    def test_full_metrics_calculation(
        self,
        metrics: BacktestMetrics,
        sample_equity_curve: pd.Series,
        sample_returns: pd.Series,
    ) -> None:
        """Test calculating all metrics on sample data."""
        # Performance metrics
        total_ret = metrics.total_return(sample_equity_curve)
        days = len(sample_equity_curve)
        ann_ret = metrics.annualized_return(total_ret, days)
        vol = metrics.volatility(sample_returns)
        sharpe = metrics.sharpe_ratio(sample_returns)
        sortino = metrics.sortino_ratio(sample_returns)
        max_dd, _, _ = metrics.max_drawdown(sample_equity_curve)

        # All should return valid values
        assert isinstance(total_ret, float)
        assert isinstance(ann_ret, float)
        assert isinstance(vol, float)
        assert isinstance(sharpe, float)
        assert isinstance(sortino, float)
        assert isinstance(max_dd, float)

        # Sanity checks
        assert vol > 0
        assert max_dd <= 0

    def test_risk_metrics_integration(
        self,
        metrics: BacktestMetrics,
        sample_returns: pd.Series,
        sample_benchmark_returns: pd.Series,
    ) -> None:
        """Test risk metrics integration."""
        var = metrics.value_at_risk(sample_returns)
        es = metrics.expected_shortfall(sample_returns)
        beta = metrics.beta_to_benchmark(sample_returns, sample_benchmark_returns)
        te = metrics.tracking_error(sample_returns, sample_benchmark_returns)

        # All should return valid values
        assert var >= 0
        assert es >= var
        assert isinstance(beta, float)
        assert te >= 0

    def test_statistical_validation_integration(
        self,
        metrics: BacktestMetrics,
        sample_returns: pd.Series,
    ) -> None:
        """Test statistical validation integration."""
        t_stat, t_pval = metrics.t_test_returns(sample_returns)
        jb_stat, jb_pval = metrics.jarque_bera_test(sample_returns)
        z_stat, runs_pval = metrics.runs_test(sample_returns)

        # All should return valid values
        assert isinstance(t_stat, float)
        assert 0 <= t_pval <= 1
        assert isinstance(jb_stat, float)
        assert 0 <= jb_pval <= 1
        assert isinstance(z_stat, float)
        assert 0 <= runs_pval <= 1
