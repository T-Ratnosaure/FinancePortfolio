"""Performance Tracker for Paper Trading Engine.

This module tracks and calculates performance metrics for paper trading sessions,
integrating with existing PnL and risk calculators.
"""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.paper_trading.models import (
    DailyPnLSnapshot,
    OrderAction,
    PerformanceSummary,
    TradeRecord,
)

if TYPE_CHECKING:
    from src.paper_trading.storage import PaperTradingStorage


class PerformanceTracker:
    """Tracks performance metrics for paper trading session.

    Calculates comprehensive metrics including returns, risk-adjusted
    performance, and trade statistics.
    """

    RISK_FREE_RATE = 0.03
    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        session_id: str,
        session_name: str,
        initial_capital: Decimal,
        start_date: date,
        storage: "PaperTradingStorage",
    ) -> None:
        """Initialize performance tracker.

        Args:
            session_id: Session identifier.
            session_name: Human-readable session name.
            initial_capital: Initial capital.
            start_date: Session start date.
            storage: Storage backend.
        """
        self.session_id = session_id
        self.session_name = session_name
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.storage = storage

        self._trades: list[TradeRecord] = []
        self._daily_values: dict[date, Decimal] = {start_date: initial_capital}
        self._daily_trades: dict[date, int] = {}
        self._realized_pnl: Decimal = Decimal("0")
        self._total_transaction_costs: Decimal = Decimal("0")

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a trade execution.

        Args:
            trade: Trade record to track.
        """
        self._trades.append(trade)

        self._realized_pnl += trade.realized_pnl
        self._total_transaction_costs += trade.transaction_cost + trade.slippage_cost

        trade_date = trade.executed_at.date()
        self._daily_trades[trade_date] = self._daily_trades.get(trade_date, 0) + 1

        self.storage.save_trade(trade)

    def record_portfolio_value(
        self, value_date: date, portfolio_value: Decimal
    ) -> None:
        """Record end-of-day portfolio value.

        Args:
            value_date: Date of the value.
            portfolio_value: Total portfolio value.
        """
        self._daily_values[value_date] = portfolio_value

    def record_daily_pnl(
        self,
        pnl_date: date,
        starting_value: Decimal,
        ending_value: Decimal,
        realized_pnl_today: Decimal,
        transaction_costs_today: Decimal,
        trades_today: int,
    ) -> DailyPnLSnapshot:
        """Record daily PnL snapshot.

        Args:
            pnl_date: Date of the PnL.
            starting_value: Portfolio value at start of day.
            ending_value: Portfolio value at end of day.
            realized_pnl_today: Realized P&L for the day.
            transaction_costs_today: Transaction costs for the day.
            trades_today: Number of trades executed.

        Returns:
            Daily PnL snapshot.
        """
        total_pnl = ending_value - starting_value
        unrealized_pnl = total_pnl - realized_pnl_today
        net_pnl = total_pnl - transaction_costs_today

        snapshot = DailyPnLSnapshot(
            pnl_date=pnl_date,
            session_id=self.session_id,
            starting_value=starting_value,
            ending_value=ending_value,
            realized_pnl=realized_pnl_today,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            transaction_costs=transaction_costs_today,
            net_pnl=net_pnl,
            trades_executed=trades_today,
        )

        self.storage.save_daily_pnl(snapshot)
        self._daily_values[pnl_date] = ending_value

        return snapshot

    def get_performance_summary(
        self, current_value: Decimal, end_date: date | None = None
    ) -> PerformanceSummary:
        """Calculate comprehensive performance summary.

        Args:
            current_value: Current portfolio value.
            end_date: Optional end date (defaults to today).

        Returns:
            Performance summary.
        """
        if end_date is None:
            end_date = date.today()

        unrealized_pnl = current_value - self.initial_capital - self._realized_pnl

        total_return = float(
            (current_value - self.initial_capital) / self.initial_capital
        )

        days_elapsed = (end_date - self.start_date).days
        annualized_return = None
        if days_elapsed > 30:
            years = days_elapsed / 365.0
            annualized_return = (1 + total_return) ** (1 / years) - 1

        winning_trades = sum(
            1
            for t in self._trades
            if t.action == OrderAction.SELL and t.realized_pnl > 0
        )
        losing_trades = sum(
            1
            for t in self._trades
            if t.action == OrderAction.SELL and t.realized_pnl < 0
        )
        total_closed_trades = winning_trades + losing_trades
        win_rate = (
            winning_trades / total_closed_trades if total_closed_trades > 0 else None
        )

        daily_returns = self._calculate_daily_returns()
        volatility = self._calculate_volatility(daily_returns)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns, volatility)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        max_drawdown = self._calculate_max_drawdown()

        return PerformanceSummary(
            session_id=self.session_id,
            session_name=self.session_name,
            start_date=self.start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            current_value=current_value,
            total_return=total_return,
            annualized_return=annualized_return,
            total_pnl=current_value - self.initial_capital,
            realized_pnl=self._realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_transaction_costs=self._total_transaction_costs,
            total_trades=len(self._trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            volatility=volatility,
        )

    def _calculate_daily_returns(self) -> list[float]:
        """Calculate daily returns from portfolio values.

        Returns:
            List of daily returns.
        """
        if len(self._daily_values) < 2:
            return []

        sorted_dates = sorted(self._daily_values.keys())
        returns = []

        for i in range(1, len(sorted_dates)):
            prev_value = float(self._daily_values[sorted_dates[i - 1]])
            curr_value = float(self._daily_values[sorted_dates[i]])

            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)

        return returns

    def _calculate_volatility(self, daily_returns: list[float]) -> float | None:
        """Calculate annualized volatility.

        Args:
            daily_returns: List of daily returns.

        Returns:
            Annualized volatility or None if insufficient data.
        """
        if len(daily_returns) < 5:
            return None

        daily_vol = float(np.std(daily_returns, ddof=1))  # type: ignore[call-overload]
        return daily_vol * np.sqrt(self.TRADING_DAYS_PER_YEAR)

    def _calculate_sharpe_ratio(
        self, daily_returns: list[float], volatility: float | None
    ) -> float | None:
        """Calculate annualized Sharpe ratio.

        Args:
            daily_returns: List of daily returns.
            volatility: Annualized volatility.

        Returns:
            Sharpe ratio or None if insufficient data.
        """
        if len(daily_returns) < 20 or volatility is None or volatility == 0:
            return None

        mean_daily_return = float(np.mean(daily_returns))  # type: ignore[call-overload]
        annualized_return = mean_daily_return * self.TRADING_DAYS_PER_YEAR
        excess_return = annualized_return - self.RISK_FREE_RATE

        return excess_return / volatility

    def _calculate_sortino_ratio(self, daily_returns: list[float]) -> float | None:
        """Calculate Sortino ratio using downside deviation.

        Args:
            daily_returns: List of daily returns.

        Returns:
            Sortino ratio or None if insufficient data.
        """
        if len(daily_returns) < 20:
            return None

        target_return = self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR

        downside_returns = [
            r - target_return for r in daily_returns if r < target_return
        ]
        if not downside_returns:
            return None

        downside_variance = sum(r**2 for r in downside_returns) / len(daily_returns)
        downside_deviation = np.sqrt(downside_variance) * np.sqrt(
            self.TRADING_DAYS_PER_YEAR
        )

        if downside_deviation == 0:
            return None

        mean_daily_return = float(np.mean(daily_returns))  # type: ignore[call-overload]
        annualized_return = mean_daily_return * self.TRADING_DAYS_PER_YEAR
        excess_return = annualized_return - self.RISK_FREE_RATE

        return excess_return / downside_deviation

    def _calculate_max_drawdown(self) -> float | None:
        """Calculate maximum drawdown.

        Returns:
            Maximum drawdown as a decimal (e.g., 0.15 for 15%) or None.
        """
        if len(self._daily_values) < 2:
            return None

        sorted_dates = sorted(self._daily_values.keys())
        values = [float(self._daily_values[d]) for d in sorted_dates]

        peak = values[0]
        max_dd = 0.0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)

        return max_dd if max_dd > 0 else None

    def get_trade_log(self) -> list[dict]:
        """Get complete trade log with details.

        Returns:
            List of trade dictionaries.
        """
        return [
            {
                "trade_id": t.trade_id,
                "order_id": t.order_id,
                "timestamp": t.executed_at.isoformat(),
                "symbol": t.symbol,
                "action": t.action.value,
                "quantity": float(t.quantity),
                "price": float(t.price),
                "total_value": float(t.total_value),
                "transaction_cost": float(t.transaction_cost),
                "slippage_cost": float(t.slippage_cost),
                "realized_pnl": float(t.realized_pnl),
            }
            for t in self._trades
        ]

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export performance data for analysis.

        Returns:
            DataFrame with daily performance data.
        """
        sorted_dates = sorted(self._daily_values.keys())

        data = []
        prev_value: Decimal | None = None

        for d in sorted_dates:
            value = self._daily_values[d]
            daily_return = 0.0
            if prev_value is not None:
                daily_return = float((value - prev_value) / prev_value)

            data.append(
                {
                    "date": d,
                    "portfolio_value": float(value),
                    "daily_return": daily_return,
                    "cumulative_return": float(
                        (value - self.initial_capital) / self.initial_capital
                    ),
                    "trades": self._daily_trades.get(d, 0),
                }
            )
            prev_value = value

        return pd.DataFrame(data)

    def get_daily_pnl_history(self) -> list[DailyPnLSnapshot]:
        """Get daily PnL history from storage.

        Returns:
            List of daily PnL snapshots.
        """
        return self.storage.get_daily_pnl_history(self.session_id)

    def load_trades_from_storage(self) -> None:
        """Load trades from storage (for resuming sessions)."""
        self._trades = self.storage.get_trades(self.session_id)

        self._realized_pnl = Decimal("0")
        self._total_transaction_costs = Decimal("0")
        self._daily_trades = {}

        for trade in self._trades:
            self._realized_pnl += trade.realized_pnl
            self._total_transaction_costs += (
                trade.transaction_cost + trade.slippage_cost
            )
            trade_date = trade.executed_at.date()
            self._daily_trades[trade_date] = self._daily_trades.get(trade_date, 0) + 1
