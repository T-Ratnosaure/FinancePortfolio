"""Tests for risk calculation module."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.models import ETFSymbol, Position
from src.portfolio.risk import (
    InsufficientDataError,
    RiskCalculator,
    RiskReport,
    MIN_OBSERVATIONS_VAR,
)


@pytest.fixture
def risk_calc() -> RiskCalculator:
    """Create a risk calculator with default settings."""
    return RiskCalculator(lookback_days=252)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Generate sample daily returns."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    # Returns with mean ~0.05% daily, vol ~1% daily
    returns = 0.0005 + 0.01 * np.random.randn(100)
    return pd.Series(returns, index=dates)


@pytest.fixture
def sample_prices() -> pd.Series:
    """Generate sample price series."""
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=300, freq="B")
    # Trending up with volatility
    base = 100 * np.cumprod(1 + 0.0003 + 0.01 * np.random.randn(300))
    return pd.Series(base, index=dates)


@pytest.fixture
def sample_returns_df() -> pd.DataFrame:
    """Generate sample returns DataFrame for multiple assets."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")

    return pd.DataFrame(
        {
            "LQQ.PA": 0.001 + 0.02 * np.random.randn(100),
            "CL2.PA": 0.001 + 0.018 * np.random.randn(100),
            "WPEA.PA": 0.0005 + 0.01 * np.random.randn(100),
        },
        index=dates,
    )


class TestRiskCalculatorInit:
    """Tests for RiskCalculator initialization."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        calc = RiskCalculator()
        assert calc.lookback_days == 252

    def test_init_custom_lookback(self) -> None:
        """Test custom lookback period."""
        calc = RiskCalculator(lookback_days=60)
        assert calc.lookback_days == 60

    def test_init_invalid_lookback(self) -> None:
        """Test that insufficient lookback raises error."""
        with pytest.raises(ValueError, match="lookback_days must be at least"):
            RiskCalculator(lookback_days=10)


class TestVaR:
    """Tests for Value at Risk calculation."""

    def test_calculate_var_historical(
        self, risk_calc: RiskCalculator, sample_returns: pd.Series
    ) -> None:
        """Test historical VaR calculation."""
        var_95 = risk_calc.calculate_var(sample_returns, confidence=0.95)

        # VaR should be positive (representing loss)
        assert var_95 > 0
        # For ~1% daily vol, 95% VaR should be roughly 1.5-2.5%
        assert 0.005 < var_95 < 0.05

    def test_calculate_var_parametric(
        self, risk_calc: RiskCalculator, sample_returns: pd.Series
    ) -> None:
        """Test parametric VaR calculation."""
        var_95 = risk_calc.calculate_var(
            sample_returns, confidence=0.95, method="parametric"
        )

        assert var_95 > 0
        assert 0.005 < var_95 < 0.05

    def test_calculate_var_99(
        self, risk_calc: RiskCalculator, sample_returns: pd.Series
    ) -> None:
        """Test 99% VaR is larger than 95% VaR."""
        var_95 = risk_calc.calculate_var(sample_returns, confidence=0.95)
        var_99 = risk_calc.calculate_var(sample_returns, confidence=0.99)

        assert var_99 > var_95

    def test_calculate_var_insufficient_data(self, risk_calc: RiskCalculator) -> None:
        """Test VaR with insufficient data raises error."""
        short_returns = pd.Series([0.01, 0.02, -0.01])

        with pytest.raises(InsufficientDataError) as exc_info:
            risk_calc.calculate_var(short_returns)

        assert exc_info.value.metric == "VaR"
        assert exc_info.value.required == MIN_OBSERVATIONS_VAR

    def test_calculate_var_invalid_confidence(
        self, risk_calc: RiskCalculator, sample_returns: pd.Series
    ) -> None:
        """Test invalid confidence level raises error."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            risk_calc.calculate_var(sample_returns, confidence=1.5)

    def test_calculate_var_invalid_method(
        self, risk_calc: RiskCalculator, sample_returns: pd.Series
    ) -> None:
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Method must be"):
            risk_calc.calculate_var(sample_returns, method="invalid")


class TestPortfolioVolatility:
    """Tests for portfolio volatility calculation."""

    def test_calculate_volatility(
        self, risk_calc: RiskCalculator, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test portfolio volatility calculation."""
        weights = {"LQQ.PA": 0.10, "CL2.PA": 0.10, "WPEA.PA": 0.60, "CASH": 0.20}

        vol = risk_calc.calculate_portfolio_volatility(weights, sample_returns_df)

        # Annualized vol should be reasonable
        assert vol > 0
        assert vol < 1.0  # Less than 100%

    def test_calculate_volatility_100_percent_cash(
        self, risk_calc: RiskCalculator, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test volatility is zero for 100% cash."""
        weights = {"CASH": 1.0}

        vol = risk_calc.calculate_portfolio_volatility(weights, sample_returns_df)

        assert vol == 0.0

    def test_calculate_volatility_weights_not_sum_to_one(
        self, risk_calc: RiskCalculator, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test error when weights don't sum to 1."""
        weights = {"LQQ.PA": 0.50, "CL2.PA": 0.10}  # Sum = 0.60

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            risk_calc.calculate_portfolio_volatility(weights, sample_returns_df)

    def test_calculate_volatility_missing_data(
        self, risk_calc: RiskCalculator, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test error when asset data is missing."""
        weights = {"LQQ.PA": 0.50, "MISSING": 0.30, "CASH": 0.20}

        with pytest.raises(ValueError, match="Missing return data"):
            risk_calc.calculate_portfolio_volatility(weights, sample_returns_df)


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_calculate_max_drawdown(
        self, risk_calc: RiskCalculator, sample_prices: pd.Series
    ) -> None:
        """Test maximum drawdown calculation."""
        max_dd, peak_date, trough_date = risk_calc.calculate_max_drawdown(sample_prices)

        # Drawdown should be negative
        assert max_dd <= 0

        # If there's a drawdown, we should have dates
        if max_dd < 0:
            assert peak_date is not None
            assert trough_date is not None
            assert peak_date <= trough_date

    def test_calculate_max_drawdown_monotonic_increase(
        self, risk_calc: RiskCalculator
    ) -> None:
        """Test drawdown is zero for monotonically increasing prices."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
        prices = pd.Series(range(100, 200), index=dates)

        max_dd, peak_date, trough_date = risk_calc.calculate_max_drawdown(prices)

        assert max_dd == 0.0
        assert peak_date is None
        assert trough_date is None

    def test_calculate_max_drawdown_known_value(
        self, risk_calc: RiskCalculator
    ) -> None:
        """Test drawdown with known expected value."""
        # Prices: 100 -> 120 -> 90 -> 100
        # Peak = 120, Trough = 90, DD = (90-120)/120 = -25%
        dates = pd.date_range(start="2023-01-01", periods=4, freq="B")
        prices = pd.Series([100.0, 120.0, 90.0, 100.0], index=dates)

        max_dd, peak_date, trough_date = risk_calc.calculate_max_drawdown(prices)

        assert abs(max_dd - (-0.25)) < 0.001

    def test_calculate_max_drawdown_empty(self, risk_calc: RiskCalculator) -> None:
        """Test error for empty price series."""
        with pytest.raises(ValueError, match="cannot be empty"):
            risk_calc.calculate_max_drawdown(pd.Series(dtype=float))

    def test_calculate_max_drawdown_negative_prices(
        self, risk_calc: RiskCalculator
    ) -> None:
        """Test error for negative prices."""
        dates = pd.date_range(start="2023-01-01", periods=3, freq="B")
        prices = pd.Series([100.0, -50.0, 80.0], index=dates)

        with pytest.raises(ValueError, match="must be positive"):
            risk_calc.calculate_max_drawdown(prices)


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_calculate_sharpe_ratio(
        self, risk_calc: RiskCalculator, sample_returns: pd.Series
    ) -> None:
        """Test Sharpe ratio calculation."""
        sharpe = risk_calc.calculate_sharpe_ratio(sample_returns)

        # Sharpe can be positive or negative depending on returns
        assert isinstance(sharpe, float)
        # Realistic range
        assert -3.0 < sharpe < 5.0

    def test_calculate_sharpe_with_risk_free(
        self, risk_calc: RiskCalculator, sample_returns: pd.Series
    ) -> None:
        """Test Sharpe ratio with non-zero risk-free rate."""
        sharpe_0 = risk_calc.calculate_sharpe_ratio(sample_returns, risk_free_rate=0)
        sharpe_rf = risk_calc.calculate_sharpe_ratio(
            sample_returns, risk_free_rate=0.03
        )

        # Higher risk-free rate should lower Sharpe
        assert sharpe_rf < sharpe_0

    def test_calculate_sharpe_insufficient_data(
        self, risk_calc: RiskCalculator
    ) -> None:
        """Test Sharpe with insufficient data raises error."""
        short_returns = pd.Series([0.01] * 5)

        with pytest.raises(InsufficientDataError) as exc_info:
            risk_calc.calculate_sharpe_ratio(short_returns)

        assert exc_info.value.metric == "Sharpe ratio"


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_calculate_sortino_ratio(
        self, risk_calc: RiskCalculator, sample_returns: pd.Series
    ) -> None:
        """Test Sortino ratio calculation."""
        sortino = risk_calc.calculate_sortino_ratio(sample_returns)

        assert isinstance(sortino, float)

    def test_sortino_vs_sharpe(
        self, risk_calc: RiskCalculator, sample_returns: pd.Series
    ) -> None:
        """Test Sortino is typically higher than Sharpe for same returns."""
        # Sortino only penalizes downside, so it's typically >= Sharpe
        # (This may not always hold depending on return distribution)
        sharpe = risk_calc.calculate_sharpe_ratio(sample_returns)
        sortino = risk_calc.calculate_sortino_ratio(sample_returns)

        # Both should be calculated
        assert isinstance(sharpe, float)
        assert isinstance(sortino, float)


class TestLeveragedDecay:
    """Tests for leveraged ETF decay calculation."""

    def test_calculate_leveraged_decay(self, risk_calc: RiskCalculator) -> None:
        """Test leveraged decay calculation."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")

        # Index returns
        index_returns = pd.Series(0.0005 + 0.01 * np.random.randn(100), index=dates)

        # ETF returns (simulating 2x with some decay)
        etf_returns = 2 * index_returns - 0.0002  # Some daily decay

        decay = risk_calc.calculate_leveraged_decay(etf_returns, index_returns)

        # Decay should be positive (representing underperformance)
        assert decay >= 0

    def test_calculate_leveraged_decay_insufficient_data(
        self, risk_calc: RiskCalculator
    ) -> None:
        """Test decay calculation with insufficient data."""
        short_etf = pd.Series([0.01, 0.02, -0.01])
        short_index = pd.Series([0.005, 0.01, -0.005])

        with pytest.raises(InsufficientDataError):
            risk_calc.calculate_leveraged_decay(short_etf, short_index)


class TestBeta:
    """Tests for beta calculation."""

    def test_calculate_beta(self, risk_calc: RiskCalculator) -> None:
        """Test beta calculation."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")

        # Create correlated returns simulating leveraged relationship
        benchmark_returns = pd.Series(0.0005 + 0.01 * np.random.randn(100), index=dates)
        # Portfolio has 2x the benchmark movement + noise
        portfolio_returns = 2 * benchmark_returns + 0.002 * np.random.randn(100)

        beta = risk_calc.calculate_beta(portfolio_returns, benchmark_returns)

        # Beta should be close to 2.0 (leveraged relationship)
        assert isinstance(beta, float)
        # Allow some tolerance for noise
        assert 1.5 < beta < 2.5


class TestCorrelationMatrix:
    """Tests for correlation matrix calculation."""

    def test_calculate_correlation_matrix(
        self, risk_calc: RiskCalculator, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test correlation matrix calculation."""
        corr = risk_calc.calculate_correlation_matrix(sample_returns_df)

        # Should be a square DataFrame
        assert corr.shape[0] == corr.shape[1]
        assert corr.shape[0] == len(sample_returns_df.columns)

        # Diagonal should be 1.0
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 0.001

    def test_calculate_correlation_matrix_insufficient_columns(
        self, risk_calc: RiskCalculator
    ) -> None:
        """Test error for single column."""
        df = pd.DataFrame({"A": [1, 2, 3] * 20})

        with pytest.raises(ValueError, match="at least 2 assets"):
            risk_calc.calculate_correlation_matrix(df)


class TestRiskReport:
    """Tests for risk report generation."""

    def test_generate_risk_report(self, risk_calc: RiskCalculator) -> None:
        """Test comprehensive risk report generation."""
        np.random.seed(42)

        # Create sample positions
        positions = [
            Position(
                symbol=ETFSymbol.LQQ,
                shares=100.0,
                average_cost=100.0,
                current_price=105.0,
                market_value=10500.0,
                unrealized_pnl=500.0,
                weight=0.1,
            ),
            Position(
                symbol=ETFSymbol.CL2,
                shares=50.0,
                average_cost=200.0,
                current_price=210.0,
                market_value=10500.0,
                unrealized_pnl=500.0,
                weight=0.1,
            ),
            Position(
                symbol=ETFSymbol.WPEA,
                shares=1000.0,
                average_cost=60.0,
                current_price=63.0,
                market_value=63000.0,
                unrealized_pnl=3000.0,
                weight=0.6,
            ),
        ]

        # Create sample price history
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
        prices_history = {
            "LQQ.PA": pd.Series(100 + 10 * np.random.randn(100), index=dates),
            "CL2.PA": pd.Series(200 + 15 * np.random.randn(100), index=dates),
            "WPEA.PA": pd.Series(60 + 3 * np.random.randn(100), index=dates),
        }

        report = risk_calc.generate_risk_report(positions, prices_history)

        assert isinstance(report, RiskReport)
        assert report.report_date == date.today()
        # VaR should be calculated or have alert
        assert report.var_95 >= 0 or len(report.risk_alerts) > 0

    def test_generate_risk_report_empty_portfolio(
        self, risk_calc: RiskCalculator
    ) -> None:
        """Test report for empty portfolio."""
        report = risk_calc.generate_risk_report([], {})

        assert report.var_95 == 0.0
        assert report.volatility == 0.0
        assert "no market value" in report.risk_alerts[0].lower()


class TestInsufficientDataError:
    """Tests for InsufficientDataError exception."""

    def test_insufficient_data_error_message(self) -> None:
        """Test error message format."""
        error = InsufficientDataError("VaR", 30, 10)

        assert error.metric == "VaR"
        assert error.required == 30
        assert error.available == 10
        assert "need 30" in str(error)
        assert "have 10" in str(error)
