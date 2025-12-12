"""Example demonstrating PnL tracking and attribution.

This example shows how to:
1. Calculate daily PnL from positions and price changes
2. Reconcile PnL with expected values
3. Attribute PnL by symbol and regime
4. Calculate alpha and beta contributions
"""

from datetime import date, datetime

import pandas as pd

from src.data.models import ETFSymbol, Position, Regime, Trade, TradeAction
from src.portfolio.pnl import (
    PnLCalculator,
    PnLReconciler,
)


def main() -> None:
    """Demonstrate PnL tracking functionality."""
    print("=" * 70)
    print("PnL Tracking and Attribution Example")
    print("=" * 70)

    # Create calculator and reconciler
    calculator = PnLCalculator()
    reconciler = PnLReconciler()

    # Example 1: Daily PnL Calculation
    print("\n1. Daily PnL Calculation")
    print("-" * 70)

    # Current positions
    positions = {
        "LQQ.PA": Position(
            symbol=ETFSymbol.LQQ,
            shares=10.0,
            average_cost=100.0,
            current_price=105.0,
            market_value=1050.0,
            unrealized_pnl=50.0,
            weight=0.5,
        ),
        "CL2.PA": Position(
            symbol=ETFSymbol.CL2,
            shares=5.0,
            average_cost=200.0,
            current_price=210.0,
            market_value=1050.0,
            unrealized_pnl=50.0,
            weight=0.5,
        ),
    }

    # Prices today and yesterday
    prices_today = {"LQQ.PA": 105.0, "CL2.PA": 210.0}
    prices_yesterday = {"LQQ.PA": 100.0, "CL2.PA": 205.0}

    # Calculate daily PnL
    daily_pnl = calculator.calculate_daily_pnl(
        positions=positions,
        prices_today=prices_today,
        prices_yesterday=prices_yesterday,
        trades_today=[],
        pnl_date=date(2024, 12, 12),
    )

    print(f"Date: {daily_pnl.date}")
    print(f"Total PnL: EUR {daily_pnl.total_pnl:.2f}")
    print(f"  Realized PnL: EUR {daily_pnl.realized_pnl:.2f}")
    print(f"  Unrealized PnL: EUR {daily_pnl.unrealized_pnl:.2f}")
    print(f"  Transaction Costs: EUR {daily_pnl.transaction_costs:.2f}")
    print(f"Net PnL: EUR {daily_pnl.net_pnl:.2f}")

    # Example 2: PnL with Trading
    print("\n2. Daily PnL with Sell Trade")
    print("-" * 70)

    sell_trade = Trade(
        symbol=ETFSymbol.LQQ,
        date=datetime(2024, 12, 12, 10, 0),
        action=TradeAction.SELL,
        shares=2.0,
        price=105.0,
        total_value=210.0,
        commission=1.99,
    )

    daily_pnl_with_trade = calculator.calculate_daily_pnl(
        positions=positions,
        prices_today=prices_today,
        prices_yesterday=prices_yesterday,
        trades_today=[sell_trade],
        pnl_date=date(2024, 12, 12),
    )

    print(f"Date: {daily_pnl_with_trade.date}")
    print(f"Total PnL: EUR {daily_pnl_with_trade.total_pnl:.2f}")
    print(f"  Realized PnL: EUR {daily_pnl_with_trade.realized_pnl:.2f}")
    print(f"  Unrealized PnL: EUR {daily_pnl_with_trade.unrealized_pnl:.2f}")
    print(f"  Transaction Costs: EUR {daily_pnl_with_trade.transaction_costs:.2f}")
    print(f"Net PnL: EUR {daily_pnl_with_trade.net_pnl:.2f}")

    # Example 3: PnL Reconciliation
    print("\n3. PnL Reconciliation")
    print("-" * 70)

    # Broker reports EUR 75.01 (close to our calculated 75.00)
    broker_pnl = 75.01
    reconciliation_result = reconciler.reconcile(
        calculated_pnl=daily_pnl_with_trade.net_pnl,
        expected_pnl=broker_pnl,
        tolerance=0.10,
    )

    print(f"Calculated PnL: EUR {reconciliation_result.calculated_pnl:.2f}")
    print(f"Broker PnL: EUR {reconciliation_result.expected_pnl:.2f}")
    print(f"Difference: EUR {reconciliation_result.difference:.2f}")
    print(f"Status: {'PASS' if reconciliation_result.matches else 'FAIL'}")

    # Example 4: Attribution by Symbol
    print("\n4. PnL Attribution by Symbol")
    print("-" * 70)

    # Create price history
    dates = pd.date_range(start="2024-12-10", periods=3, freq="D")
    price_history = {
        "LQQ.PA": pd.Series([100.0, 102.0, 105.0], index=dates),
        "CL2.PA": pd.Series([205.0, 207.0, 210.0], index=dates),
    }

    attribution = calculator.attribute_by_symbol(
        daily_pnls=[],
        positions=positions,
        price_history=price_history,
    )

    print("Symbol Attribution:")
    for symbol, pnl in attribution.items():
        print(f"  {symbol}: EUR {pnl:.2f}")

    # Example 5: Attribution by Regime
    print("\n5. PnL Attribution by Regime")
    print("-" * 70)

    # Create sample daily PnLs
    daily_pnls_sample = [
        calculator.calculate_daily_pnl(
            positions,
            {"LQQ.PA": 102.0, "CL2.PA": 207.0},
            prices_yesterday,
            [],
            pd.Timestamp(dates[1]).date(),  # type: ignore[arg-type]
        ),
        calculator.calculate_daily_pnl(
            positions,
            {"LQQ.PA": 105.0, "CL2.PA": 210.0},
            {"LQQ.PA": 102.0, "CL2.PA": 207.0},
            [],
            pd.Timestamp(dates[2]).date(),  # type: ignore[arg-type]
        ),
    ]

    regime_history = {
        pd.Timestamp(dates[1]).date(): Regime.RISK_ON,  # type: ignore[index]
        pd.Timestamp(dates[2]).date(): Regime.RISK_ON,  # type: ignore[index]
    }

    regime_attribution = calculator.attribute_by_regime(
        daily_pnls_sample,
        regime_history,  # type: ignore[arg-type]
    )

    print("Regime Attribution:")
    for regime, pnl in regime_attribution.items():
        print(f"  {regime}: EUR {pnl:.2f}")

    # Example 6: Alpha/Beta Calculation
    print("\n6. Alpha and Beta Attribution")
    print("-" * 70)

    # Sample returns
    portfolio_returns = pd.Series([0.02, 0.01, 0.03, -0.01, 0.02])
    benchmark_returns = pd.Series([0.015, 0.008, 0.025, -0.008, 0.015])

    factors = calculator.calculate_alpha_beta(portfolio_returns, benchmark_returns)

    print(f"Beta: {factors['beta']:.4f}")
    print(f"Alpha: {factors['alpha']:.4f}")
    print(f"Beta PnL: EUR {factors['beta_pnl']:.2f}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
