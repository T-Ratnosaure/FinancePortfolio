"""Risk calculation and monitoring for PEA Portfolio.

This module provides comprehensive risk metrics calculation including:
- Value at Risk (VaR) using historical and parametric methods
- Portfolio volatility with proper annualization
- Maximum drawdown tracking with peak/trough identification
- Leveraged ETF decay estimation
- Risk-adjusted performance metrics (Sharpe, Sortino)
- Correlation analysis and beta calculation

The RiskCalculator follows conservative assumptions appropriate for
a retail PEA portfolio with leveraged ETF exposure.
"""

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.data.models import (
    DRAWDOWN_ALERT,
    MAX_LEVERAGED_EXPOSURE,
    PEA_ETFS,
    ETFSymbol,
    Position,
)

# Trading days per year for annualization
TRADING_DAYS_PER_YEAR = 252

# Minimum data requirements
MIN_OBSERVATIONS_VAR = 30
MIN_OBSERVATIONS_VOLATILITY = 20


class RiskReport(BaseModel):
    """Comprehensive risk report for the portfolio.

    Aggregates key risk metrics and generates alerts when limits are exceeded.

    Attributes:
        report_date: Date of the risk report
        var_95: 1-day Value at Risk at 95% confidence level (as positive loss)
        volatility: Annualized portfolio volatility (standard deviation)
        max_drawdown: Maximum drawdown from peak (as negative value)
        sharpe_ratio: Annualized Sharpe ratio
        sortino_ratio: Annualized Sortino ratio (optional if insufficient data)
        leveraged_decay_estimate: Estimated annual decay per leveraged ETF
        risk_alerts: List of warning messages when limits are breached
    """

    report_date: date
    var_95: float = Field(
        ge=0.0,
        description="1-day 95% VaR as positive loss percentage",
    )
    volatility: float = Field(
        ge=0.0,
        description="Annualized portfolio volatility",
    )
    max_drawdown: float = Field(
        le=0.0,
        description="Maximum drawdown as negative percentage",
    )
    sharpe_ratio: float | None = Field(
        default=None,
        description="Annualized Sharpe ratio",
    )
    sortino_ratio: float | None = Field(
        default=None,
        description="Annualized Sortino ratio",
    )
    leveraged_decay_estimate: dict[str, float] = Field(
        default_factory=dict,
        description="Estimated annual decay per leveraged ETF",
    )
    risk_alerts: list[str] = Field(
        default_factory=list,
        description="Active risk warnings",
    )


class RiskCalculatorError(Exception):
    """Base exception for risk calculation errors."""

    pass


class InsufficientDataError(RiskCalculatorError):
    """Raised when insufficient data is available for calculation."""

    def __init__(self, metric: str, required: int, available: int) -> None:
        """Initialize insufficient data error.

        Args:
            metric: Name of the metric being calculated
            required: Minimum observations required
            available: Observations available
        """
        self.metric = metric
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient data for {metric}: need {required}, have {available}"
        )


class RiskCalculator:
    """Calculator for portfolio risk metrics.

    Provides methods for calculating various risk metrics with proper
    handling of edge cases and data quality issues. All metrics use
    conservative assumptions appropriate for leveraged ETF exposure.

    Attributes:
        lookback_days: Number of trading days for historical calculations
    """

    def __init__(self, lookback_days: int = 252) -> None:
        """Initialize the risk calculator.

        Args:
            lookback_days: Number of trading days for lookback period.
                Default 252 (one trading year).
        """
        if lookback_days < MIN_OBSERVATIONS_VAR:
            raise ValueError(
                f"lookback_days must be at least {MIN_OBSERVATIONS_VAR}, "
                f"got {lookback_days}"
            )
        self.lookback_days = lookback_days

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: Literal["historical", "parametric"] = "historical",
    ) -> float:
        """Calculate Value at Risk.

        VaR represents the maximum expected loss over a 1-day period at
        the specified confidence level. Returns a positive value representing
        the potential loss.

        Args:
            returns: Series of daily returns (as decimals, e.g., 0.01 for 1%)
            confidence: Confidence level (default 0.95 for 95% VaR)
            method: Calculation method - 'historical' or 'parametric'.
                Historical uses empirical quantiles (recommended for fat tails).
                Parametric assumes normal distribution.

        Returns:
            VaR as a positive decimal (e.g., 0.02 for 2% potential loss)

        Raises:
            InsufficientDataError: If returns has fewer than MIN_OBSERVATIONS_VAR
            ValueError: If confidence is not in (0, 1) or method is invalid
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

        if method not in ("historical", "parametric"):
            raise ValueError(
                f"Method must be 'historical' or 'parametric', got {method}"
            )

        # Clean and validate data
        clean_returns = returns.dropna()
        n_obs = len(clean_returns)

        if n_obs < MIN_OBSERVATIONS_VAR:
            raise InsufficientDataError("VaR", MIN_OBSERVATIONS_VAR, n_obs)

        # Apply lookback limit
        if n_obs > self.lookback_days:
            clean_returns = clean_returns.iloc[-self.lookback_days :]

        if method == "historical":
            # Historical VaR: Use empirical quantile
            # The (1 - confidence) quantile gives us the loss threshold
            var = -float(np.percentile(clean_returns, (1 - confidence) * 100))
        else:
            # Parametric VaR: Assume normal distribution
            import scipy.stats

            mean_return = float(clean_returns.mean())
            std_return = float(clean_returns.std(ddof=1))

            # z-score for the confidence level
            z_score = float(scipy.stats.norm.ppf(1 - confidence))  # type: ignore[attr-defined]
            var = -(mean_return + z_score * std_return)

        # VaR should be positive (representing a loss)
        return max(0.0, var)

    def calculate_portfolio_volatility(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
    ) -> float:
        """Calculate annualized portfolio volatility.

        Uses the covariance matrix of asset returns weighted by portfolio
        allocations to compute overall portfolio risk.

        Args:
            weights: Dictionary mapping asset symbols to portfolio weights
                (must sum to approximately 1.0)
            returns: DataFrame with columns for each asset's daily returns

        Returns:
            Annualized portfolio volatility as a decimal

        Raises:
            InsufficientDataError: If insufficient return observations
            ValueError: If weights don't match returns columns or weights invalid
        """
        # Validate weights
        if not weights:
            raise ValueError("Weights dictionary cannot be empty")

        total_weight = sum(weights.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        # Get the assets we have weights for (excluding cash)
        weighted_assets = [k for k, v in weights.items() if v > 0 and k != "CASH"]

        if not weighted_assets:
            # Portfolio is 100% cash - zero volatility
            return 0.0

        # Check that all weighted assets are in returns DataFrame
        missing_assets = set(weighted_assets) - set(returns.columns)
        if missing_assets:
            raise ValueError(f"Missing return data for assets: {missing_assets}")

        # Filter returns to only weighted assets
        asset_returns = returns[weighted_assets].dropna()
        n_obs = len(asset_returns)

        if n_obs < MIN_OBSERVATIONS_VOLATILITY:
            raise InsufficientDataError(
                "portfolio volatility", MIN_OBSERVATIONS_VOLATILITY, n_obs
            )

        # Apply lookback limit
        if n_obs > self.lookback_days:
            asset_returns = asset_returns.iloc[-self.lookback_days :]

        # Build weight vector (same order as columns)
        weight_vector = np.array(
            [weights.get(col, 0.0) for col in asset_returns.columns]
        )

        # Normalize weights for non-cash portion
        non_cash_weight = sum(weights.get(col, 0.0) for col in asset_returns.columns)
        if non_cash_weight > 0:
            weight_vector = weight_vector / non_cash_weight

        # Calculate covariance matrix
        cov_df: pd.DataFrame = asset_returns.cov()  # type: ignore[assignment]
        cov_matrix = cov_df.to_numpy()

        # Portfolio variance: w' * Cov * w
        portfolio_variance = float(weight_vector @ cov_matrix @ weight_vector)

        # Daily volatility to annualized
        daily_volatility = np.sqrt(portfolio_variance)
        annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

        return float(annualized_volatility)

    def calculate_max_drawdown(
        self,
        prices: pd.Series,
    ) -> tuple[float, date | None, date | None]:
        """Calculate maximum drawdown with peak and trough dates.

        Maximum drawdown measures the largest peak-to-trough decline in
        portfolio value. This is a key risk metric for understanding
        potential losses during market downturns.

        Args:
            prices: Series of prices or portfolio values with datetime index

        Returns:
            Tuple of (max_drawdown, peak_date, trough_date) where:
                - max_drawdown: Maximum drawdown as a negative decimal
                - peak_date: Date of the peak before the drawdown
                - trough_date: Date of the trough
            Returns (0.0, None, None) if prices are monotonically increasing

        Raises:
            ValueError: If prices series is empty or contains invalid values
        """
        if prices.empty:
            raise ValueError("Prices series cannot be empty")

        clean_prices = prices.dropna()
        if clean_prices.empty:
            raise ValueError("Prices series contains only NaN values")

        if (clean_prices <= 0).any():
            raise ValueError("Prices must be positive")

        # Calculate running maximum
        running_max = clean_prices.cummax()

        # Calculate drawdown at each point
        drawdowns = (clean_prices - running_max) / running_max

        # Find maximum drawdown
        max_dd = float(drawdowns.min())

        if max_dd >= 0:
            # No drawdown - prices are monotonically increasing
            return 0.0, None, None

        # Find trough (point of maximum drawdown)
        trough_idx = drawdowns.idxmin()

        # Find peak (running max at trough location)
        # The peak is the last date before trough where running_max increased
        prices_to_trough = clean_prices.loc[:trough_idx]
        peak_idx = prices_to_trough.idxmax()

        # Convert to date objects
        peak_date = self._to_date(peak_idx)
        trough_date = self._to_date(trough_idx)

        return max_dd, peak_date, trough_date

    def calculate_leveraged_decay(
        self,
        etf_returns: pd.Series,
        index_returns: pd.Series,
        leverage: int = 2,
    ) -> float:
        """Estimate annualized decay from leveraged ETF vs underlying index.

        Leveraged ETFs experience volatility decay due to daily rebalancing.
        This method estimates the annual decay by comparing actual leveraged
        ETF performance to the theoretical leveraged return of the index.

        The decay formula is based on: Decay = sigma^2 * (L^2 - L) / 2
        where sigma is volatility and L is leverage.

        Args:
            etf_returns: Daily returns of the leveraged ETF
            index_returns: Daily returns of the underlying index
            leverage: Leverage factor of the ETF (default 2x)

        Returns:
            Estimated annual decay as a positive decimal (e.g., 0.05 for 5%)

        Raises:
            InsufficientDataError: If insufficient aligned observations
            ValueError: If series lengths don't match after alignment
        """
        if leverage < 1:
            raise ValueError(f"Leverage must be >= 1, got {leverage}")

        # Align the two series
        aligned = pd.DataFrame({"etf": etf_returns, "index": index_returns}).dropna()
        n_obs = len(aligned)

        if n_obs < MIN_OBSERVATIONS_VAR:
            raise InsufficientDataError("leveraged decay", MIN_OBSERVATIONS_VAR, n_obs)

        # Apply lookback limit
        if n_obs > self.lookback_days:
            aligned = aligned.iloc[-self.lookback_days :]

        etf_rets = pd.Series(aligned["etf"])
        idx_rets = pd.Series(aligned["index"])

        # Calculate actual cumulative returns
        etf_prod = (1 + etf_rets).prod()
        idx_prod = (1 + idx_rets).prod()
        etf_cumret = float(etf_prod) - 1  # type: ignore[arg-type]
        idx_cumret = float(idx_prod) - 1  # type: ignore[arg-type]

        # Theoretical leveraged return (simple multiply, no decay)
        theoretical_leveraged_return = leverage * idx_cumret

        # Daily decay (actual vs theoretical)
        # Positive decay means ETF underperformed the theoretical return
        total_decay = theoretical_leveraged_return - etf_cumret

        # Annualize the decay
        days_in_period = len(aligned)
        annualized_decay = total_decay * (TRADING_DAYS_PER_YEAR / days_in_period)

        # Alternative: Use volatility-based decay estimate
        # This is more robust when we have limited data
        index_volatility = float(idx_rets.std())
        volatility_based_decay = (
            (leverage**2 - leverage) * (index_volatility**2) / 2
        ) * TRADING_DAYS_PER_YEAR

        # Return the average of both methods for robustness
        estimated_decay = (annualized_decay + volatility_based_decay) / 2

        return max(0.0, estimated_decay)

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Calculate annualized Sharpe ratio.

        The Sharpe ratio measures risk-adjusted return:
        Sharpe = (R_p - R_f) / sigma_p

        where R_p is portfolio return, R_f is risk-free rate, and sigma_p
        is portfolio volatility.

        Args:
            returns: Series of daily returns
            risk_free_rate: Annualized risk-free rate (default 0)

        Returns:
            Annualized Sharpe ratio

        Raises:
            InsufficientDataError: If insufficient return observations
        """
        clean_returns = returns.dropna()
        n_obs = len(clean_returns)

        if n_obs < MIN_OBSERVATIONS_VOLATILITY:
            raise InsufficientDataError(
                "Sharpe ratio", MIN_OBSERVATIONS_VOLATILITY, n_obs
            )

        # Apply lookback limit
        if n_obs > self.lookback_days:
            clean_returns = clean_returns.iloc[-self.lookback_days :]

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

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Calculate annualized Sortino ratio.

        The Sortino ratio is similar to Sharpe but uses downside deviation
        instead of total volatility, penalizing only negative returns:
        Sortino = (R_p - R_f) / sigma_down

        The downside deviation is calculated as:
        sigma_down = sqrt((1/n) * sum((r_i - r_f)^2) for all r_i < r_f)

        This formula uses ALL observations in the denominator (n), but only
        counts squared deviations where returns fall below the target.

        Args:
            returns: Series of daily returns
            risk_free_rate: Annualized risk-free rate (default 0)

        Returns:
            Annualized Sortino ratio

        Raises:
            InsufficientDataError: If insufficient return observations
        """
        clean_returns = returns.dropna()
        n_obs = len(clean_returns)

        if n_obs < MIN_OBSERVATIONS_VOLATILITY:
            raise InsufficientDataError(
                "Sortino ratio", MIN_OBSERVATIONS_VOLATILITY, n_obs
            )

        # Apply lookback limit
        if n_obs > self.lookback_days:
            clean_returns = clean_returns.iloc[-self.lookback_days :]

        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1

        # Calculate excess returns
        excess_returns = clean_returns - daily_rf

        mean_excess = float(excess_returns.mean())

        # Downside deviation: sqrt of mean squared deviations below target
        # Formula: sigma_down = sqrt((1/n) * sum((r_i - r_f)^2) for r_i < r_f)
        # This uses ALL observations in denominator, but only counts deviations
        # where return is below the target (risk-free rate)
        returns_array = clean_returns.to_numpy()
        downside_squared = np.where(
            returns_array < daily_rf,
            (returns_array - daily_rf) ** 2,
            0.0,
        )
        downside_deviation = float(np.sqrt(downside_squared.mean()))

        # Check if there are any negative returns at all
        if (returns_array < daily_rf).sum() == 0:
            # No returns below target - cannot compute meaningful Sortino
            return 0.0

        # Handle zero downside deviation edge case
        if downside_deviation < 1e-10:
            return 0.0

        # Daily Sortino to annualized
        daily_sortino = mean_excess / downside_deviation
        annualized_sortino = daily_sortino * np.sqrt(TRADING_DAYS_PER_YEAR)

        return float(annualized_sortino)

    def calculate_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Calculate portfolio beta relative to a benchmark.

        Beta measures the sensitivity of portfolio returns to benchmark
        movements:
        Beta = Cov(R_p, R_b) / Var(R_b)

        Args:
            portfolio_returns: Series of daily portfolio returns
            benchmark_returns: Series of daily benchmark returns

        Returns:
            Portfolio beta coefficient

        Raises:
            InsufficientDataError: If insufficient aligned observations
        """
        # Align the two series
        aligned = pd.DataFrame(
            {
                "portfolio": portfolio_returns,
                "benchmark": benchmark_returns,
            }
        ).dropna()
        n_obs = len(aligned)

        if n_obs < MIN_OBSERVATIONS_VOLATILITY:
            raise InsufficientDataError("beta", MIN_OBSERVATIONS_VOLATILITY, n_obs)

        # Apply lookback limit
        if n_obs > self.lookback_days:
            aligned = aligned.iloc[-self.lookback_days :]

        port_rets = pd.Series(aligned["portfolio"])
        bench_rets = pd.Series(aligned["benchmark"])

        # Calculate covariance and variance
        covariance = float(port_rets.cov(bench_rets))
        bench_var = bench_rets.var(ddof=1)
        benchmark_variance: float = (
            float(bench_var) if bench_var is not None else 0.0  # type: ignore[arg-type]
        )

        # Handle zero variance edge case
        if benchmark_variance < 1e-10:
            return 0.0

        beta = covariance / benchmark_variance
        return float(beta)

    def calculate_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate correlation matrix for asset returns.

        Args:
            returns_df: DataFrame with columns for each asset's daily returns

        Returns:
            Correlation matrix as a DataFrame

        Raises:
            ValueError: If returns_df is empty or has fewer than 2 columns
        """
        if returns_df.empty:
            raise ValueError("Returns DataFrame cannot be empty")

        if len(returns_df.columns) < 2:
            raise ValueError("Need at least 2 assets for correlation matrix")

        # Drop rows with any NaN values
        clean_returns = returns_df.dropna()

        if len(clean_returns) < MIN_OBSERVATIONS_VOLATILITY:
            raise InsufficientDataError(
                "correlation matrix",
                MIN_OBSERVATIONS_VOLATILITY,
                len(clean_returns),
            )

        # Apply lookback limit
        if len(clean_returns) > self.lookback_days:
            clean_returns = clean_returns.iloc[-self.lookback_days :]

        return clean_returns.corr()

    def generate_risk_report(
        self,
        positions: list[Position],
        prices_history: dict[str, pd.Series],
        benchmark_returns: pd.Series | None = None,
        risk_free_rate: float = 0.0,
    ) -> RiskReport:
        """Generate a comprehensive risk report for the portfolio.

        Calculates all key risk metrics and generates alerts for any
        limit breaches.

        Args:
            positions: List of current Position objects
            prices_history: Dictionary mapping symbol strings to price Series
            benchmark_returns: Optional benchmark returns for beta calculation
            risk_free_rate: Annualized risk-free rate for Sharpe/Sortino

        Returns:
            RiskReport with all calculated metrics and any risk alerts
        """
        report_date = date.today()

        # Build portfolio weights
        total_value = sum(p.market_value for p in positions)
        if total_value <= 0:
            return self._empty_risk_report(report_date)

        weights = self._build_weights_from_positions(positions, total_value)

        # Calculate returns data
        returns_df = self._build_returns_dataframe(prices_history)
        portfolio_returns = self._calculate_portfolio_returns(weights, returns_df)
        portfolio_values = self._calculate_portfolio_values(weights, prices_history)

        # Calculate all metrics and collect alerts
        alerts = self._check_leveraged_exposure(weights)
        var_95, var_alerts = self._safe_calculate_var(portfolio_returns)
        volatility, vol_alerts = self._safe_calculate_volatility(weights, returns_df)
        max_drawdown, dd_alerts = self._safe_calculate_drawdown(portfolio_values)
        sharpe_ratio = self._safe_calculate_sharpe(portfolio_returns, risk_free_rate)
        sortino_ratio = self._safe_calculate_sortino(portfolio_returns, risk_free_rate)
        leveraged_decay = self._calculate_decay_estimates(weights, returns_df)

        alerts.extend(var_alerts + vol_alerts + dd_alerts)

        return RiskReport(
            report_date=report_date,
            var_95=var_95,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            leveraged_decay_estimate=leveraged_decay,  # type: ignore[arg-type]
            risk_alerts=alerts,  # type: ignore[arg-type]
        )

    def _empty_risk_report(self, report_date: date) -> RiskReport:
        """Create an empty risk report for portfolios with no value."""
        return RiskReport(
            report_date=report_date,
            var_95=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            sharpe_ratio=None,
            sortino_ratio=None,
            leveraged_decay_estimate={},
            risk_alerts=["Portfolio has no market value"],
        )

    def _build_weights_from_positions(
        self,
        positions: list[Position],
        total_value: float,
    ) -> dict[str, float]:
        """Build weights dictionary from position list."""
        weights: dict[str, float] = {}
        for pos in positions:
            symbol_key = (
                pos.symbol.value
                if isinstance(pos.symbol, ETFSymbol)
                else str(pos.symbol)
            )
            weights[symbol_key] = pos.market_value / total_value
        return weights

    def _check_leveraged_exposure(self, weights: dict[str, float]) -> list[str]:
        """Check if leveraged exposure exceeds limits."""
        alerts: list[str] = []
        leveraged_exposure = sum(
            w
            for sym, w in weights.items()
            if sym in [ETFSymbol.LQQ.value, ETFSymbol.CL2.value]
        )
        if leveraged_exposure > MAX_LEVERAGED_EXPOSURE:
            alerts.append(
                f"Leveraged exposure {leveraged_exposure:.1%} exceeds "
                f"limit {MAX_LEVERAGED_EXPOSURE:.1%}"
            )
        return alerts

    def _safe_calculate_var(
        self, portfolio_returns: pd.Series
    ) -> tuple[float, list[str]]:
        """Calculate VaR with error handling."""
        try:
            var_95 = self.calculate_var(portfolio_returns, confidence=0.95)
            return var_95, []
        except (InsufficientDataError, ValueError) as e:
            return 0.0, [f"VaR calculation failed: {e}"]

    def _safe_calculate_volatility(
        self,
        weights: dict[str, float],
        returns_df: pd.DataFrame,
    ) -> tuple[float, list[str]]:
        """Calculate volatility with error handling."""
        try:
            weights_for_vol = {
                k: v
                for k, v in weights.items()
                if k in returns_df.columns or k == "CASH"
            }
            if weights_for_vol:
                volatility = self.calculate_portfolio_volatility(
                    weights_for_vol, returns_df
                )
                return volatility, []
            return 0.0, []
        except (InsufficientDataError, ValueError) as e:
            return 0.0, [f"Volatility calculation failed: {e}"]

    def _safe_calculate_drawdown(
        self, portfolio_values: pd.Series
    ) -> tuple[float, list[str]]:
        """Calculate max drawdown with error handling and alert generation."""
        alerts: list[str] = []
        try:
            max_drawdown, _, _ = self.calculate_max_drawdown(portfolio_values)
            if max_drawdown < DRAWDOWN_ALERT:
                alerts.append(
                    f"Maximum drawdown {max_drawdown:.1%} exceeds "
                    f"alert threshold {DRAWDOWN_ALERT:.1%}"
                )
            return max_drawdown, alerts
        except ValueError as e:
            return 0.0, [f"Drawdown calculation failed: {e}"]

    def _safe_calculate_sharpe(
        self,
        portfolio_returns: pd.Series,
        risk_free_rate: float,
    ) -> float | None:
        """Calculate Sharpe ratio with error handling."""
        try:
            return self.calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
        except InsufficientDataError:
            return None

    def _safe_calculate_sortino(
        self,
        portfolio_returns: pd.Series,
        risk_free_rate: float,
    ) -> float | None:
        """Calculate Sortino ratio with error handling."""
        try:
            return self.calculate_sortino_ratio(portfolio_returns, risk_free_rate)
        except InsufficientDataError:
            return None

    def _calculate_decay_estimates(
        self,
        weights: dict[str, float],
        returns_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate leveraged ETF decay estimates."""
        leveraged_decay: dict[str, float] = {}
        wpea_symbol = ETFSymbol.WPEA.value

        if wpea_symbol not in returns_df.columns:
            return leveraged_decay

        index_rets = pd.Series(returns_df[wpea_symbol])

        for etf_symbol in [ETFSymbol.LQQ, ETFSymbol.CL2]:
            symbol_str = etf_symbol.value
            if symbol_str not in weights or weights[symbol_str] <= 0:
                continue
            if symbol_str not in returns_df.columns:
                continue

            try:
                etf_info = PEA_ETFS.get(etf_symbol)
                leverage = etf_info.leverage if etf_info else 2
                etf_rets = pd.Series(returns_df[symbol_str])
                decay = self.calculate_leveraged_decay(etf_rets, index_rets, leverage)
                leveraged_decay[symbol_str] = decay
            except (InsufficientDataError, ValueError):
                pass

        return leveraged_decay

    def _build_returns_dataframe(
        self,
        prices_history: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Build a DataFrame of returns from price history.

        Args:
            prices_history: Dictionary mapping symbols to price series

        Returns:
            DataFrame with daily returns for each asset
        """
        returns_dict: dict[str, pd.Series] = {}

        for symbol, prices in prices_history.items():
            if not prices.empty:
                returns = prices.pct_change().dropna()
                returns_dict[symbol] = returns

        if not returns_dict:
            return pd.DataFrame()

        return pd.DataFrame(returns_dict)

    def _calculate_portfolio_returns(
        self,
        weights: dict[str, float],
        returns_df: pd.DataFrame,
    ) -> pd.Series:
        """Calculate weighted portfolio returns.

        Args:
            weights: Asset weights
            returns_df: DataFrame of asset returns

        Returns:
            Series of portfolio returns
        """
        if returns_df.empty:
            return pd.Series(dtype=float)

        # Get assets present in both weights and returns
        common_assets = set(weights.keys()) & set(returns_df.columns)

        if not common_assets:
            return pd.Series(dtype=float)

        total_weight = sum(weights.get(a, 0.0) for a in common_assets)
        if total_weight <= 0:
            return pd.Series(dtype=float)

        # Calculate weighted returns using numpy for type safety
        result: np.ndarray = np.zeros(len(returns_df.index))
        for asset in common_assets:
            weight = weights.get(asset, 0.0) / total_weight
            asset_values: np.ndarray = returns_df[asset].to_numpy()
            result = result + weight * asset_values

        portfolio_returns = pd.Series(result, index=returns_df.index)
        return portfolio_returns.dropna()

    def _calculate_portfolio_values(
        self,
        weights: dict[str, float],
        prices_history: dict[str, pd.Series],
    ) -> pd.Series:
        """Calculate portfolio value series from weights and prices.

        Assumes weights are rebalanced daily (simplified approximation).

        Args:
            weights: Asset weights
            prices_history: Price history by symbol

        Returns:
            Series of portfolio values (normalized to start at 100)
        """
        if not prices_history:
            return pd.Series(dtype=float)

        # Build normalized price DataFrame (each asset starts at 100)
        normalized_prices: dict[str, pd.Series] = {}
        for symbol, prices in prices_history.items():
            if not prices.empty and symbol in weights:
                normalized = 100 * prices / prices.iloc[0]
                normalized_prices[symbol] = normalized

        if not normalized_prices:
            return pd.Series(dtype=float)

        prices_df = pd.DataFrame(normalized_prices)

        # Calculate weighted portfolio value
        common_assets = set(weights.keys()) & set(prices_df.columns)
        total_weight = sum(weights.get(a, 0.0) for a in common_assets)

        if total_weight <= 0:
            return pd.Series(dtype=float)

        # Calculate weighted values using numpy for type safety
        result: np.ndarray = np.zeros(len(prices_df.index))
        for asset in common_assets:
            weight = weights.get(asset, 0.0) / total_weight
            asset_values: np.ndarray = prices_df[asset].to_numpy()
            result = result + weight * asset_values

        portfolio_value = pd.Series(result, index=prices_df.index)
        return portfolio_value.dropna()

    @staticmethod
    def _to_date(idx: pd.Timestamp | date | str | int | None) -> date | None:
        """Convert pandas index value to date object.

        Args:
            idx: Index value to convert (can be Timestamp, date, str, or int)

        Returns:
            date object or None
        """
        if idx is None:
            return None
        if isinstance(idx, date) and not isinstance(idx, pd.Timestamp):
            return idx
        if isinstance(idx, pd.Timestamp):
            return idx.date()
        if isinstance(idx, (str, int)):
            try:
                return pd.to_datetime(idx).date()
            except (ValueError, TypeError):
                return None
        return None
