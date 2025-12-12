"""Profit and Loss (PnL) tracking, attribution, and reconciliation.

This module provides comprehensive PnL management functionality including:
- Daily PnL calculation (realized and unrealized)
- Period PnL aggregation
- Attribution analysis (by symbol, regime, factor)
- Reconciliation and validation
"""

import logging
from datetime import date

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from src.data.models import Position, Regime, Trade, TradeAction

logger = logging.getLogger(__name__)


class DailyPnL(BaseModel):
    """Daily profit and loss breakdown.

    Attributes:
        date: Date of the PnL calculation
        total_pnl: Total PnL for the day (realized + unrealized)
        realized_pnl: PnL from closed positions
        unrealized_pnl: Mark-to-market changes in open positions
        transaction_costs: Costs incurred (commissions, fees)
        net_pnl: Net PnL after transaction costs
    """

    date: date
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    transaction_costs: float = Field(ge=0.0)
    net_pnl: float

    @model_validator(mode="after")
    def validate_pnl_consistency(self) -> "DailyPnL":
        """Validate that PnL components are consistent."""
        expected_total = self.realized_pnl + self.unrealized_pnl
        if abs(self.total_pnl - expected_total) > 0.01:
            raise ValueError(
                f"total_pnl ({self.total_pnl}) should equal "
                f"realized_pnl + unrealized_pnl ({expected_total})"
            )

        expected_net = self.total_pnl - self.transaction_costs
        if abs(self.net_pnl - expected_net) > 0.01:
            raise ValueError(
                f"net_pnl ({self.net_pnl}) should equal "
                f"total_pnl - transaction_costs ({expected_net})"
            )

        return self

    def round_values(self) -> "DailyPnL":
        """Round all values to 2 decimal places for currency display.

        Returns:
            New DailyPnL with rounded values
        """
        return DailyPnL(
            date=self.date,
            total_pnl=round(self.total_pnl, 2),
            realized_pnl=round(self.realized_pnl, 2),
            unrealized_pnl=round(self.unrealized_pnl, 2),
            transaction_costs=round(self.transaction_costs, 2),
            net_pnl=round(self.net_pnl, 2),
        )


class PeriodPnL(BaseModel):
    """Aggregated PnL for a time period.

    Attributes:
        start_date: Period start date
        end_date: Period end date
        total_pnl: Total PnL for the period
        realized_pnl: Total realized PnL
        unrealized_pnl: Total unrealized PnL
        transaction_costs: Total transaction costs
        net_pnl: Net PnL after transaction costs
        num_trading_days: Number of trading days in period
        daily_pnls: List of daily PnL records
    """

    start_date: date
    end_date: date
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    transaction_costs: float = Field(ge=0.0)
    net_pnl: float
    num_trading_days: int = Field(ge=0)
    daily_pnls: list[DailyPnL] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_period(self) -> "PeriodPnL":
        """Validate period dates and PnL consistency."""
        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")

        expected_total = self.realized_pnl + self.unrealized_pnl
        if abs(self.total_pnl - expected_total) > 0.01:
            raise ValueError(
                f"total_pnl ({self.total_pnl}) should equal "
                f"realized_pnl + unrealized_pnl ({expected_total})"
            )

        expected_net = self.total_pnl - self.transaction_costs
        if abs(self.net_pnl - expected_net) > 0.01:
            raise ValueError(
                f"net_pnl ({self.net_pnl}) should equal "
                f"total_pnl - transaction_costs ({expected_net})"
            )

        return self

    def round_values(self) -> "PeriodPnL":
        """Round all values to 2 decimal places for currency display.

        Returns:
            New PeriodPnL with rounded values
        """
        return PeriodPnL(
            start_date=self.start_date,
            end_date=self.end_date,
            total_pnl=round(self.total_pnl, 2),
            realized_pnl=round(self.realized_pnl, 2),
            unrealized_pnl=round(self.unrealized_pnl, 2),
            transaction_costs=round(self.transaction_costs, 2),
            net_pnl=round(self.net_pnl, 2),
            num_trading_days=self.num_trading_days,
            daily_pnls=[pnl.round_values() for pnl in self.daily_pnls],
        )


class PnLAttribution(BaseModel):
    """PnL attribution breakdown by various dimensions.

    Attributes:
        by_symbol: PnL attributed to each symbol
        by_regime: PnL attributed to each market regime
        by_factor: Factor attribution (beta, alpha, etc.)
    """

    by_symbol: dict[str, float] = Field(default_factory=dict)
    by_regime: dict[str, float] = Field(default_factory=dict)
    by_factor: dict[str, float] = Field(default_factory=dict)

    def round_values(self) -> "PnLAttribution":
        """Round all attribution values to 2 decimal places.

        Returns:
            New PnLAttribution with rounded values
        """
        return PnLAttribution(
            by_symbol={k: round(v, 2) for k, v in self.by_symbol.items()},
            by_regime={k: round(v, 2) for k, v in self.by_regime.items()},
            by_factor={k: round(v, 2) for k, v in self.by_factor.items()},
        )


class ReconciliationResult(BaseModel):
    """Result of PnL reconciliation check.

    Attributes:
        matches: Whether calculated PnL matches expected within tolerance
        calculated_pnl: PnL calculated by the system
        expected_pnl: Expected PnL (from broker or other source)
        difference: Absolute difference
        tolerance: Tolerance threshold used
        message: Human-readable reconciliation message
    """

    matches: bool
    calculated_pnl: float
    expected_pnl: float
    difference: float
    tolerance: float = Field(gt=0.0)
    message: str

    @model_validator(mode="after")
    def validate_difference(self) -> "ReconciliationResult":
        """Validate difference is calculated correctly."""
        expected_diff = abs(self.calculated_pnl - self.expected_pnl)
        if abs(self.difference - expected_diff) > 0.01:
            raise ValueError(
                f"difference ({self.difference}) should equal "
                f"abs(calculated - expected) ({expected_diff})"
            )
        return self


class PnLCalculator:
    """Calculator for daily and period PnL.

    This class provides methods to calculate PnL from positions, prices,
    and trade data, handling both realized and unrealized components.
    """

    def calculate_daily_pnl(
        self,
        positions: dict[str, Position],
        prices_today: dict[str, float],
        prices_yesterday: dict[str, float],
        trades_today: list[Trade],
        pnl_date: date,
    ) -> DailyPnL:
        """Calculate PnL for a single day.

        Args:
            positions: Current positions (after today's trades)
            prices_today: Today's closing prices by symbol
            prices_yesterday: Yesterday's closing prices by symbol
            trades_today: Trades executed today
            pnl_date: Date for this PnL calculation

        Returns:
            DailyPnL object with calculated values

        Example:
            >>> calculator = PnLCalculator()
            >>> positions = {"LQQ.PA": Position(...)}
            >>> prices_today = {"LQQ.PA": 105.50}
            >>> prices_yesterday = {"LQQ.PA": 105.00}
            >>> trades = []
            >>> pnl = calculator.calculate_daily_pnl(
            ...     positions, prices_today, prices_yesterday, trades, date.today()
            ... )
        """
        realized_pnl = 0.0
        unrealized_pnl = 0.0
        transaction_costs = 0.0

        # Calculate realized PnL from trades
        for trade in trades_today:
            transaction_costs += trade.commission

            if trade.action == TradeAction.SELL:
                # Realized PnL = (sell price - average cost) * shares sold
                symbol_str = trade.symbol.value
                if symbol_str in positions:
                    avg_cost = positions[symbol_str].average_cost
                    realized_pnl += (trade.price - avg_cost) * trade.shares

        # Calculate unrealized PnL from mark-to-market changes
        for symbol, position in positions.items():
            if symbol in prices_today and symbol in prices_yesterday:
                price_change = prices_today[symbol] - prices_yesterday[symbol]
                position_unrealized = price_change * position.shares
                unrealized_pnl += position_unrealized

        # Round components
        realized_pnl = round(realized_pnl, 2)
        unrealized_pnl = round(unrealized_pnl, 2)
        transaction_costs = round(transaction_costs, 2)
        total_pnl = realized_pnl + unrealized_pnl
        net_pnl = total_pnl - transaction_costs

        return DailyPnL(
            date=pnl_date,
            total_pnl=total_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            transaction_costs=transaction_costs,
            net_pnl=net_pnl,
        )

    def _get_prices_for_date(
        self,
        positions: dict[str, Position],
        price_history: dict[str, pd.Series],
        current_date: pd.Timestamp,
        prev_date: pd.Timestamp | None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Extract prices for today and yesterday for all positions.

        Args:
            positions: Current positions
            price_history: Price series by symbol
            current_date: Current date
            prev_date: Previous date (None for first day)

        Returns:
            Tuple of (prices_today, prices_yesterday)
        """
        prices_today = {}
        prices_yesterday = {}

        for symbol in positions.keys():
            if symbol not in price_history:
                continue

            series = price_history[symbol]

            # Get price at or before this date
            valid_prices = series[series.index <= current_date]
            if len(valid_prices) > 0:  # type: ignore[arg-type]
                prices_today[symbol] = float(valid_prices.iloc[-1])  # type: ignore[union-attr]

            # Get previous day's price
            if prev_date is not None:
                prev_valid = series[series.index <= prev_date]
                if len(prev_valid) > 0:  # type: ignore[arg-type]
                    prices_yesterday[symbol] = float(prev_valid.iloc[-1])  # type: ignore[union-attr]
            else:
                # First day - use same price (no change)
                prices_yesterday[symbol] = prices_today.get(symbol, 0.0)

        return prices_today, prices_yesterday

    def _get_trades_for_date(
        self,
        trade_log: list[Trade],
        target_date: date,
    ) -> list[Trade]:
        """Filter trades for a specific date.

        Args:
            trade_log: All trades
            target_date: Date to filter for

        Returns:
            List of trades for the target date
        """
        return [
            t
            for t in trade_log
            if (t.date.date() if hasattr(t.date, "date") else t.date) == target_date
        ]

    def calculate_period_pnl(
        self,
        start_date: date,
        end_date: date,
        position_history: list[dict[str, Position]],
        price_history: dict[str, pd.Series],
        trade_log: list[Trade],
    ) -> PeriodPnL:
        """Calculate PnL for a period.

        Args:
            start_date: Period start date
            end_date: Period end date
            position_history: List of position snapshots (daily)
            price_history: Price series by symbol
            trade_log: All trades in the period

        Returns:
            PeriodPnL object with aggregated values

        Example:
            >>> calculator = PnLCalculator()
            >>> period_pnl = calculator.calculate_period_pnl(
            ...     date(2024, 1, 1),
            ...     date(2024, 1, 31),
            ...     position_snapshots,
            ...     price_data,
            ...     trades
            ... )
        """
        if end_date < start_date:
            raise ValueError("end_date must be >= start_date")

        daily_pnls: list[DailyPnL] = []
        total_realized = 0.0
        total_unrealized = 0.0
        total_costs = 0.0

        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        for i, current_date in enumerate(date_range):
            current_date_obj = current_date.date()

            # Get positions for this day
            if i < len(position_history):
                positions = position_history[i]
            else:
                positions = position_history[-1] if position_history else {}

            # Get prices for today and yesterday
            prev_date = pd.Timestamp(date_range[i - 1]) if i > 0 else None  # type: ignore[assignment]
            prices_today, prices_yesterday = self._get_prices_for_date(
                positions,
                price_history,
                current_date,
                prev_date,  # type: ignore[arg-type]
            )

            # Get trades for this day
            trades_today = self._get_trades_for_date(trade_log, current_date_obj)

            # Calculate daily PnL
            if positions and (prices_today or trades_today):
                daily_pnl = self.calculate_daily_pnl(
                    positions,
                    prices_today,
                    prices_yesterday,
                    trades_today,
                    current_date_obj,
                )
                daily_pnls.append(daily_pnl)

                total_realized += daily_pnl.realized_pnl
                total_unrealized += daily_pnl.unrealized_pnl
                total_costs += daily_pnl.transaction_costs

        # Round totals
        total_realized = round(total_realized, 2)
        total_unrealized = round(total_unrealized, 2)
        total_costs = round(total_costs, 2)
        total_pnl = total_realized + total_unrealized
        net_pnl = total_pnl - total_costs

        return PeriodPnL(
            start_date=start_date,
            end_date=end_date,
            total_pnl=total_pnl,
            realized_pnl=total_realized,
            unrealized_pnl=total_unrealized,
            transaction_costs=total_costs,
            net_pnl=net_pnl,
            num_trading_days=len(daily_pnls),
            daily_pnls=daily_pnls,
        )

    def attribute_by_symbol(
        self,
        daily_pnls: list[DailyPnL],
        positions: dict[str, Position],
        price_history: dict[str, pd.Series],
    ) -> dict[str, float]:
        """Attribute PnL by symbol/position.

        Args:
            daily_pnls: List of daily PnL records
            positions: Position holdings
            price_history: Price series by symbol

        Returns:
            Dictionary mapping symbol to attributed PnL

        Example:
            >>> attribution = calculator.attribute_by_symbol(
            ...     daily_pnls, positions, prices
            ... )
            >>> print(f"LQQ.PA PnL: {attribution['LQQ.PA']:.2f}")
        """
        symbol_pnl: dict[str, float] = {}

        for symbol, position in positions.items():
            if symbol not in price_history:
                continue

            series = price_history[symbol]
            if series.empty or len(series) < 2:
                continue

            # Calculate price change over period
            start_price = float(series.iloc[0])
            end_price = float(series.iloc[-1])
            price_change = end_price - start_price

            # Attribute PnL = price change * shares
            pnl = price_change * position.shares
            symbol_pnl[symbol] = round(pnl, 2)

        return symbol_pnl

    def attribute_by_regime(
        self,
        daily_pnls: list[DailyPnL],
        regime_history: dict[date, Regime],
    ) -> dict[str, float]:
        """Attribute PnL by market regime.

        Args:
            daily_pnls: List of daily PnL records
            regime_history: Mapping of date to regime

        Returns:
            Dictionary mapping regime to attributed PnL

        Example:
            >>> attribution = calculator.attribute_by_regime(
            ...     daily_pnls, regimes
            ... )
            >>> print(f"Risk-on PnL: {attribution['risk_on']:.2f}")
        """
        regime_pnl: dict[str, float] = {
            Regime.RISK_ON.value: 0.0,
            Regime.NEUTRAL.value: 0.0,
            Regime.RISK_OFF.value: 0.0,
        }

        for daily_pnl in daily_pnls:
            if daily_pnl.date in regime_history:
                regime = regime_history[daily_pnl.date]
                regime_pnl[regime.value] += daily_pnl.net_pnl

        # Round values
        return {k: round(v, 2) for k, v in regime_pnl.items()}

    def calculate_alpha_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> dict[str, float]:
        """Calculate alpha and beta factor attribution.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series

        Returns:
            Dictionary with 'beta', 'alpha', and 'beta_pnl' values

        Example:
            >>> factors = calculator.calculate_alpha_beta(
            ...     portfolio_rets, benchmark_rets
            ... )
            >>> print(f"Alpha: {factors['alpha']:.4f}")
        """
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Portfolio and benchmark returns must have same length")

        if len(portfolio_returns) < 2:
            return {"beta": 0.0, "alpha": 0.0, "beta_pnl": 0.0}

        # Calculate beta using covariance
        covariance = portfolio_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()

        if benchmark_variance == 0:
            beta = 0.0
        else:
            beta = covariance / benchmark_variance

        # Calculate alpha (intercept)
        portfolio_mean = portfolio_returns.mean()
        benchmark_mean = benchmark_returns.mean()
        alpha = portfolio_mean - (beta * benchmark_mean)

        # Calculate beta contribution to PnL
        # Beta PnL = beta * benchmark_return * portfolio_value
        # For simplicity, use total return
        beta_pnl = beta * benchmark_returns.sum()

        return {
            "beta": round(beta, 4),
            "alpha": round(alpha, 4),
            "beta_pnl": round(beta_pnl, 2),
        }


class PnLReconciler:
    """Reconciler for PnL validation and verification.

    This class provides methods to reconcile calculated PnL against
    expected values from broker statements or other sources.
    """

    def reconcile(
        self,
        calculated_pnl: float,
        expected_pnl: float,
        tolerance: float = 0.01,
    ) -> ReconciliationResult:
        """Reconcile calculated PnL with expected PnL.

        Args:
            calculated_pnl: PnL calculated by the system
            expected_pnl: Expected PnL from broker or other source
            tolerance: Acceptable difference threshold (default: 0.01 = $0.01)

        Returns:
            ReconciliationResult with match status and details

        Example:
            >>> reconciler = PnLReconciler()
            >>> result = reconciler.reconcile(1000.50, 1000.45, tolerance=0.10)
            >>> if result.matches:
            ...     print("PnL reconciled successfully")
        """
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")

        difference = abs(calculated_pnl - expected_pnl)
        matches = difference <= tolerance

        if matches:
            message = (
                f"PnL reconciled successfully: calculated={calculated_pnl:.2f}, "
                f"expected={expected_pnl:.2f}, "
                f"difference={difference:.2f} (within tolerance {tolerance:.2f})"
            )
        else:
            message = (
                f"PnL reconciliation FAILED: calculated={calculated_pnl:.2f}, "
                f"expected={expected_pnl:.2f}, "
                f"difference={difference:.2f} (exceeds tolerance {tolerance:.2f})"
            )

        logger.info(message)

        return ReconciliationResult(
            matches=matches,
            calculated_pnl=round(calculated_pnl, 2),
            expected_pnl=round(expected_pnl, 2),
            difference=round(difference, 2),
            tolerance=tolerance,
            message=message,
        )

    def validate_pnl_components(
        self,
        daily_pnl: DailyPnL,
    ) -> list[str]:
        """Validate PnL component consistency.

        Args:
            daily_pnl: DailyPnL to validate

        Returns:
            List of validation error messages (empty if valid)

        Example:
            >>> reconciler = PnLReconciler()
            >>> errors = reconciler.validate_pnl_components(daily_pnl)
            >>> if errors:
            ...     print(f"Validation errors: {errors}")
        """
        errors: list[str] = []

        # Check total PnL = realized + unrealized
        expected_total = daily_pnl.realized_pnl + daily_pnl.unrealized_pnl
        if abs(daily_pnl.total_pnl - expected_total) > 0.01:
            errors.append(
                f"Total PnL mismatch: {daily_pnl.total_pnl} != "
                f"{expected_total} (realized + unrealized)"
            )

        # Check net PnL = total - costs
        expected_net = daily_pnl.total_pnl - daily_pnl.transaction_costs
        if abs(daily_pnl.net_pnl - expected_net) > 0.01:
            errors.append(
                f"Net PnL mismatch: {daily_pnl.net_pnl} != "
                f"{expected_net} (total - costs)"
            )

        # Check costs are non-negative
        if daily_pnl.transaction_costs < 0:
            errors.append(
                f"Transaction costs cannot be negative: {daily_pnl.transaction_costs}"
            )

        return errors
