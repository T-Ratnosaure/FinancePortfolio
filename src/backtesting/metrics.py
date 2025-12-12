"""Performance metrics calculation for backtesting.

This module provides comprehensive metrics for evaluating backtesting results:
- Performance metrics (returns, volatility, Sharpe, Sortino, max drawdown)
- Risk metrics (VaR, CVaR/Expected Shortfall, beta, tracking error)
- Regime-specific performance analysis
- Statistical validation tests (t-test, Jarque-Bera, runs test)

The metrics follow industry-standard formulas with special attention to:
- Correct Sortino ratio formula using downside deviation
- Annualization using 252 trading days
- Proper handling of edge cases and missing data
"""

from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

# Trading days per year for annualization
TRADING_DAYS_PER_YEAR = 252


class Regime(str, Enum):
    """Market regime classification."""

    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"


class RegimePerformance(BaseModel):
    """Performance metrics for a specific market regime.

    Captures key statistics for portfolio performance during each
    regime to validate allocation logic effectiveness.

    Attributes:
        regime: The market regime being analyzed
        annualized_return: Annualized return during this regime
        volatility: Annualized volatility during this regime
        sharpe_ratio: Risk-adjusted return (Sharpe) during this regime
        max_drawdown: Maximum drawdown during this regime (negative value)
        days_in_regime: Number of trading days in this regime
    """

    regime: Regime
    annualized_return: float = Field(description="Annualized return during regime")
    volatility: float = Field(
        ge=0.0,
        description="Annualized volatility during regime",
    )
    sharpe_ratio: float | None = Field(
        default=None,
        description="Sharpe ratio during regime (None if insufficient data)",
    )
    max_drawdown: float = Field(
        le=0.0,
        description="Maximum drawdown during regime (negative value)",
    )
    days_in_regime: int = Field(
        ge=0,
        description="Number of trading days in this regime",
    )


class BacktestMetrics:
    """Calculator for backtesting performance and risk metrics.

    Provides a comprehensive suite of metrics following industry standards
    and best practices for portfolio performance evaluation.

    All return values are expressed as decimals (e.g., 0.10 for 10%).
    Volatility and risk metrics are annualized using 252 trading days.

    Example:
        >>> metrics = BacktestMetrics()
        >>> equity = pd.Series([100, 105, 102, 108, 110])
        >>> total_ret = metrics.total_return(equity)
        >>> print(f"Total return: {total_ret:.2%}")
        Total return: 10.00%
    """

    # =========================================================================
    # Performance Metrics
    # =========================================================================

    def total_return(self, equity_curve: pd.Series) -> float:
        """Calculate total return from an equity curve.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Total return as a decimal (e.g., 0.10 for 10%)

        Raises:
            ValueError: If equity curve is empty or contains invalid values
        """
        if equity_curve.empty:
            raise ValueError("Equity curve cannot be empty")

        clean_equity = equity_curve.dropna()
        if clean_equity.empty:
            raise ValueError("Equity curve contains only NaN values")

        if (clean_equity <= 0).any():
            raise ValueError("Equity curve must contain only positive values")

        start_value = float(clean_equity.iloc[0])
        end_value = float(clean_equity.iloc[-1])

        return (end_value - start_value) / start_value

    def annualized_return(self, total_return: float, days: int) -> float:
        """Convert total return to annualized return.

        Uses the formula: (1 + total_return)^(252/days) - 1

        Args:
            total_return: Total return as a decimal
            days: Number of trading days in the period

        Returns:
            Annualized return as a decimal

        Raises:
            ValueError: If days is not positive or total_return < -1
        """
        if days <= 0:
            raise ValueError(f"Days must be positive, got {days}")

        if total_return <= -1.0:
            raise ValueError(f"Total return must be > -100%, got {total_return:.2%}")

        # Handle very short periods
        if days < 1:
            return 0.0

        annualization_factor = TRADING_DAYS_PER_YEAR / days
        return float((1 + total_return) ** annualization_factor - 1)

    def volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility from daily returns.

        Volatility is computed as the standard deviation of daily returns,
        annualized by multiplying by sqrt(252).

        Args:
            returns: Series of daily returns (as decimals)

        Returns:
            Annualized volatility as a decimal

        Raises:
            ValueError: If returns series is empty or has insufficient data
        """
        clean_returns = returns.dropna()

        if len(clean_returns) < 2:
            raise ValueError("Need at least 2 return observations")

        daily_vol = float(clean_returns.std(ddof=1))
        annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

        return annualized_vol

    def sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
    ) -> float:
        """Calculate annualized Sharpe ratio.

        The Sharpe ratio measures risk-adjusted return:
        Sharpe = (R_p - R_f) / sigma_p

        where R_p is portfolio return, R_f is risk-free rate,
        and sigma_p is portfolio volatility.

        Args:
            returns: Series of daily returns (as decimals)
            risk_free_rate: Annualized risk-free rate (default 2%)

        Returns:
            Annualized Sharpe ratio

        Raises:
            ValueError: If returns series has insufficient data
        """
        clean_returns = returns.dropna()

        if len(clean_returns) < 2:
            raise ValueError("Need at least 2 return observations")

        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1

        # Calculate excess returns
        excess_returns = clean_returns - daily_rf

        mean_excess = float(excess_returns.mean())
        std_returns = float(clean_returns.std(ddof=1))

        # Handle zero volatility edge case
        if std_returns < 1e-10:
            return 0.0

        # Daily Sharpe to annualized
        daily_sharpe = mean_excess / std_returns
        annualized_sharpe = daily_sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)

        return float(annualized_sharpe)

    def sortino_ratio(
        self,
        returns: pd.Series,
        target: float = 0.0,
    ) -> float:
        """Calculate annualized Sortino ratio with CORRECT downside deviation.

        The Sortino ratio uses downside deviation instead of total volatility:
        Sortino = (R_p - target) / DD

        CORRECT FORMULA for Downside Deviation:
        DD = sqrt(mean((r_i - target)^2 for all r_i < target))

        This formula:
        1. Only considers returns BELOW the target
        2. Uses ALL observations in the denominator (not just downside count)
        3. Calculates squared deviations from target, not from zero

        Note: This corrects the formula discovered as wrong in Sprint 4.
        The WRONG formula was: std(negative_returns_only)

        Args:
            returns: Series of daily returns (as decimals)
            target: Daily target return (default 0.0 for MAR = 0%)

        Returns:
            Annualized Sortino ratio

        Raises:
            ValueError: If returns series has insufficient data
        """
        clean_returns = returns.dropna()

        if len(clean_returns) < 2:
            raise ValueError("Need at least 2 return observations")

        # Calculate mean excess return over target
        mean_return = float(clean_returns.mean())
        mean_excess = mean_return - target

        # CORRECT Sortino Downside Deviation Formula:
        # DD = sqrt((1/n) * sum((r_i - target)^2) for all r_i < target)
        # This uses ALL n observations in the denominator,
        # but only sums squared deviations where returns < target
        returns_array = clean_returns.to_numpy()
        n_total = len(returns_array)

        # Calculate squared deviations below target
        downside_squared = np.where(
            returns_array < target,
            (returns_array - target) ** 2,
            0.0,
        )

        # Mean over ALL observations (not just downside)
        downside_variance = float(np.sum(downside_squared) / n_total)
        downside_deviation = np.sqrt(downside_variance)

        # Check if there are any returns below target
        n_downside = np.sum(returns_array < target)
        if n_downside == 0:
            # No downside returns - cannot compute meaningful Sortino
            return 0.0

        # Handle zero downside deviation edge case
        if downside_deviation < 1e-10:
            return 0.0

        # Daily Sortino to annualized
        daily_sortino = mean_excess / downside_deviation
        annualized_sortino = daily_sortino * np.sqrt(TRADING_DAYS_PER_YEAR)

        return float(annualized_sortino)

    def max_drawdown(
        self,
        equity_curve: pd.Series,
    ) -> tuple[float, int, int]:
        """Calculate maximum drawdown with peak and trough indices.

        Maximum drawdown measures the largest peak-to-trough decline.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Tuple of (max_drawdown, peak_idx, trough_idx) where:
                - max_drawdown: Maximum drawdown as negative decimal
                - peak_idx: Integer index of the peak
                - trough_idx: Integer index of the trough
            Returns (0.0, 0, 0) if no drawdown (monotonically increasing)

        Raises:
            ValueError: If equity curve is empty or contains invalid values
        """
        if equity_curve.empty:
            raise ValueError("Equity curve cannot be empty")

        clean_equity = equity_curve.dropna()
        if clean_equity.empty:
            raise ValueError("Equity curve contains only NaN values")

        if (clean_equity <= 0).any():
            raise ValueError("Equity curve must contain only positive values")

        # Calculate running maximum
        running_max = clean_equity.cummax()

        # Calculate drawdown at each point
        drawdowns = (clean_equity - running_max) / running_max

        # Find maximum drawdown
        max_dd = float(drawdowns.min())

        if max_dd >= 0:
            # No drawdown - prices are monotonically increasing
            return 0.0, 0, 0

        # Find trough (point of maximum drawdown) - get integer position
        trough_idx = int(drawdowns.argmin())

        # Find peak - the position where running_max equals the peak value at trough
        # Look at the equity values up to the trough
        equity_to_trough = clean_equity.iloc[: trough_idx + 1]
        peak_idx = int(equity_to_trough.argmax())

        return max_dd, peak_idx, trough_idx

    def max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """Calculate maximum drawdown duration (days from peak to recovery).

        Finds the longest period the portfolio was below its previous peak.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Number of days from peak to recovery (or end if not recovered)

        Raises:
            ValueError: If equity curve is empty or invalid
        """
        if equity_curve.empty:
            raise ValueError("Equity curve cannot be empty")

        clean_equity = equity_curve.dropna()
        if clean_equity.empty:
            raise ValueError("Equity curve contains only NaN values")

        if (clean_equity <= 0).any():
            raise ValueError("Equity curve must contain only positive values")

        # Calculate running maximum
        running_max = clean_equity.cummax()

        # Track drawdown periods
        in_drawdown = clean_equity < running_max

        # Find the longest consecutive drawdown period
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def win_rate(self, monthly_returns: pd.Series) -> float:
        """Calculate win rate from monthly returns.

        Win rate is the percentage of months with positive returns.

        Args:
            monthly_returns: Series of monthly returns (as decimals)

        Returns:
            Win rate as a decimal (e.g., 0.60 for 60% winning months)

        Raises:
            ValueError: If monthly returns series is empty
        """
        clean_returns = monthly_returns.dropna()

        if clean_returns.empty:
            raise ValueError("Monthly returns cannot be empty")

        n_positive = int((clean_returns > 0).sum())
        n_total = len(clean_returns)

        return n_positive / n_total

    def profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss).

        Profit factor measures the ratio of total gains to total losses.
        Values > 1 indicate profitable strategy, > 1.5 is typically good.

        Args:
            returns: Series of returns (daily or any frequency)

        Returns:
            Profit factor (gross profit / gross loss)
            Returns infinity if no losing periods.

        Raises:
            ValueError: If returns series is empty
        """
        clean_returns = returns.dropna()

        if clean_returns.empty:
            raise ValueError("Returns cannot be empty")

        gross_profit = float(clean_returns[clean_returns > 0].sum())
        gross_loss = float(abs(clean_returns[clean_returns < 0].sum()))

        if gross_loss < 1e-10:
            # No losses - return a large number instead of infinity
            return float("inf") if gross_profit > 0 else 1.0

        return gross_profit / gross_loss

    # =========================================================================
    # Risk Metrics
    # =========================================================================

    def value_at_risk(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Value at Risk (VaR) using historical simulation.

        VaR represents the maximum expected loss at a given confidence level.
        Returns a positive value representing potential loss.

        Args:
            returns: Series of daily returns (as decimals)
            confidence: Confidence level (default 0.95 for 95% VaR)

        Returns:
            VaR as a positive decimal (e.g., 0.02 for 2% potential loss)

        Raises:
            ValueError: If confidence is not in (0, 1) or insufficient data
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

        clean_returns = returns.dropna()

        if len(clean_returns) < 30:
            raise ValueError("Need at least 30 observations for VaR")

        # Historical VaR: Use empirical quantile
        # The (1 - confidence) quantile gives us the loss threshold
        var = -float(np.percentile(clean_returns, (1 - confidence) * 100))

        # VaR should be positive (representing a loss)
        return max(0.0, var)

    def expected_shortfall(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Expected Shortfall (CVaR / Conditional VaR).

        Expected Shortfall is the average loss beyond the VaR threshold.
        It captures tail risk better than VaR alone.

        Args:
            returns: Series of daily returns (as decimals)
            confidence: Confidence level (default 0.95)

        Returns:
            Expected shortfall as a positive decimal

        Raises:
            ValueError: If confidence is not in (0, 1) or insufficient data
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

        clean_returns = returns.dropna()

        if len(clean_returns) < 30:
            raise ValueError("Need at least 30 observations for ES")

        # Find the VaR threshold
        var_threshold = float(np.percentile(clean_returns, (1 - confidence) * 100))

        # Expected Shortfall: Average of returns below VaR threshold
        tail_returns = clean_returns[clean_returns <= var_threshold]

        if len(tail_returns) == 0:
            # Fallback: use VaR if no returns below threshold
            return max(0.0, -var_threshold)

        es = -float(tail_returns.mean())

        return max(0.0, es)

    def beta_to_benchmark(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Calculate portfolio beta relative to a benchmark.

        Beta measures systematic risk:
        Beta = Cov(R_p, R_b) / Var(R_b)

        Args:
            portfolio_returns: Series of daily portfolio returns
            benchmark_returns: Series of daily benchmark returns

        Returns:
            Portfolio beta coefficient

        Raises:
            ValueError: If insufficient aligned observations
        """
        # Align the two series
        aligned = pd.DataFrame(
            {
                "portfolio": portfolio_returns,
                "benchmark": benchmark_returns,
            }
        ).dropna()

        if len(aligned) < 20:
            raise ValueError("Need at least 20 aligned observations for beta")

        port_rets = pd.Series(aligned["portfolio"])
        bench_rets = pd.Series(aligned["benchmark"])

        # Calculate covariance and variance
        covariance = float(port_rets.cov(bench_rets))
        bench_var = bench_rets.var(ddof=1)
        benchmark_variance = float(bench_var) if bench_var is not None else 0.0

        # Handle zero variance edge case
        if benchmark_variance < 1e-10:
            return 0.0

        return covariance / benchmark_variance

    def tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Calculate annualized tracking error.

        Tracking error is the standard deviation of the difference between
        portfolio and benchmark returns, annualized.

        Args:
            portfolio_returns: Series of daily portfolio returns
            benchmark_returns: Series of daily benchmark returns

        Returns:
            Annualized tracking error as a decimal

        Raises:
            ValueError: If insufficient aligned observations
        """
        # Align the two series
        aligned = pd.DataFrame(
            {
                "portfolio": portfolio_returns,
                "benchmark": benchmark_returns,
            }
        ).dropna()

        if len(aligned) < 20:
            raise ValueError("Need at least 20 aligned observations for tracking error")

        # Calculate active returns (difference)
        active_returns = aligned["portfolio"] - aligned["benchmark"]

        # Annualize the standard deviation
        daily_te = float(active_returns.std(ddof=1))
        annualized_te = daily_te * np.sqrt(TRADING_DAYS_PER_YEAR)

        return annualized_te

    def calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate the drawdown series from an equity curve.

        Returns a series showing the drawdown at each point in time.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Series of drawdowns (negative values, 0 at new highs)

        Raises:
            ValueError: If equity curve is empty or invalid
        """
        if equity_curve.empty:
            raise ValueError("Equity curve cannot be empty")

        clean_equity = equity_curve.dropna()
        if clean_equity.empty:
            raise ValueError("Equity curve contains only NaN values")

        if (clean_equity <= 0).any():
            raise ValueError("Equity curve must contain only positive values")

        # Calculate running maximum
        running_max = clean_equity.cummax()

        # Calculate drawdown at each point
        drawdowns = (clean_equity - running_max) / running_max

        return drawdowns

    # =========================================================================
    # Regime-Specific Metrics
    # =========================================================================

    def calculate_regime_performance(
        self,
        returns: pd.Series,
        regime_history: pd.Series,
    ) -> dict[str, RegimePerformance]:
        """Calculate performance metrics for each market regime.

        Args:
            returns: Series of daily returns with datetime index
            regime_history: Series of Regime values aligned with returns

        Returns:
            Dictionary mapping regime names to RegimePerformance objects

        Raises:
            ValueError: If series are misaligned or empty
        """
        if returns.empty or regime_history.empty:
            raise ValueError("Returns and regime history cannot be empty")

        # Align the series
        aligned = pd.DataFrame(
            {
                "returns": returns,
                "regime": regime_history,
            }
        ).dropna()

        if aligned.empty:
            raise ValueError("No aligned data between returns and regimes")

        results: dict[str, RegimePerformance] = {}

        # Calculate metrics for each regime
        for regime in Regime:
            regime_mask = aligned["regime"] == regime.value
            regime_returns = aligned.loc[regime_mask, "returns"]

            if len(regime_returns) < 2:
                # Not enough data for this regime
                results[regime.value] = RegimePerformance(
                    regime=regime,
                    annualized_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=None,
                    max_drawdown=0.0,
                    days_in_regime=len(regime_returns),
                )
                continue

            # Calculate regime-specific metrics
            try:
                regime_vol = self.volatility(regime_returns)
            except ValueError:
                regime_vol = 0.0

            # Annualized return
            total_ret = float((1 + regime_returns).prod() - 1)
            days = len(regime_returns)
            try:
                ann_ret = self.annualized_return(total_ret, days)
            except ValueError:
                ann_ret = 0.0

            # Sharpe ratio
            try:
                sharpe = self.sharpe_ratio(regime_returns, risk_free_rate=0.02)
            except ValueError:
                sharpe = None

            # Max drawdown - construct equity curve for regime
            regime_equity = (1 + regime_returns).cumprod() * 100
            try:
                max_dd, _, _ = self.max_drawdown(regime_equity)
            except ValueError:
                max_dd = 0.0

            results[regime.value] = RegimePerformance(
                regime=regime,
                annualized_return=ann_ret,
                volatility=regime_vol,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                days_in_regime=days,
            )

        return results

    # =========================================================================
    # Statistical Validation
    # =========================================================================

    def t_test_returns(self, returns: pd.Series) -> tuple[float, float]:
        """Perform one-sample t-test to check if returns are significantly > 0.

        Tests the null hypothesis that the mean return equals zero.
        A low p-value (< 0.05) suggests returns are statistically significant.

        Args:
            returns: Series of returns

        Returns:
            Tuple of (t_statistic, p_value)

        Raises:
            ValueError: If insufficient data for test
        """
        clean_returns = returns.dropna()

        if len(clean_returns) < 30:
            raise ValueError("Need at least 30 observations for t-test")

        # One-sample t-test against zero
        t_stat, p_value = stats.ttest_1samp(clean_returns, 0.0)

        # For one-tailed test (returns > 0), divide p-value by 2
        # and check sign of t-statistic
        if t_stat > 0:
            one_tailed_p = p_value / 2
        else:
            one_tailed_p = 1 - (p_value / 2)

        return float(t_stat), float(one_tailed_p)

    def jarque_bera_test(self, returns: pd.Series) -> tuple[float, float]:
        """Perform Jarque-Bera test for normality of returns.

        Tests whether returns follow a normal distribution.
        A low p-value (< 0.05) indicates non-normal distribution.

        Note: Financial returns are typically NOT normal (fat tails),
        so this test is expected to reject normality.

        Args:
            returns: Series of returns

        Returns:
            Tuple of (JB_statistic, p_value)

        Raises:
            ValueError: If insufficient data for test
        """
        clean_returns = returns.dropna()

        if len(clean_returns) < 30:
            raise ValueError("Need at least 30 observations for Jarque-Bera test")

        jb_stat, p_value = stats.jarque_bera(clean_returns)

        return float(jb_stat), float(p_value)

    def runs_test(self, returns: pd.Series) -> tuple[float, float]:
        """Perform runs test for randomness / autocorrelation.

        Tests whether the sequence of returns is random.
        A low p-value (< 0.05) indicates non-random patterns.

        The test counts "runs" of consecutive positive or negative returns.
        Too few or too many runs suggests autocorrelation.

        Args:
            returns: Series of returns

        Returns:
            Tuple of (z_statistic, p_value)

        Raises:
            ValueError: If insufficient data for test
        """
        clean_returns = returns.dropna()

        if len(clean_returns) < 30:
            raise ValueError("Need at least 30 observations for runs test")

        # Convert returns to binary (positive = 1, negative/zero = 0)
        binary_returns = (clean_returns > 0).astype(int).to_numpy()

        n = len(binary_returns)
        n_positive = int(np.sum(binary_returns))
        n_negative = n - n_positive

        # Handle edge cases
        if n_positive == 0 or n_negative == 0:
            # All same sign - cannot compute runs test
            return 0.0, 1.0

        # Count runs
        runs = 1
        for i in range(1, n):
            if binary_returns[i] != binary_returns[i - 1]:
                runs += 1

        # Expected runs and variance under null hypothesis
        expected_runs = (2 * n_positive * n_negative / n) + 1
        variance_runs = (
            2 * n_positive * n_negative * (2 * n_positive * n_negative - n)
        ) / (n**2 * (n - 1))

        # Handle zero variance
        if variance_runs < 1e-10:
            return 0.0, 1.0

        # Z-statistic
        z_stat = (runs - expected_runs) / np.sqrt(variance_runs)

        # Two-tailed p-value using scipy.stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # type: ignore[attr-defined]

        return float(z_stat), float(p_value)
