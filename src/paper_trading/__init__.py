"""Paper Trading Engine for PEA Portfolio Simulation.

This module provides a complete paper trading simulation engine that allows
testing strategies with live-like data feeds without real money.

Key components:
- PaperTradingSession: Main orchestrator for paper trading lifecycle
- VirtualOrderExecutor: Simulates order execution with realistic fills
- PortfolioStateManager: Manages virtual portfolio state
- PerformanceTracker: Tracks and calculates performance metrics

Example usage:
    from src.paper_trading import PaperTradingSession, SessionConfig

    config = SessionConfig(
        session_name="My Test Session",
        initial_capital=Decimal("10000.00"),
        auto_rebalance=True,
    )

    session = PaperTradingSession.create_new(config)

    # Execute a trade
    result, trade = session.execute_order(
        symbol="LQQ.PA",
        action=OrderAction.BUY,
        quantity=Decimal("10"),
        current_price=Decimal("100.00"),
    )

    # Get performance summary
    summary = session.get_performance_summary()
    session.stop()
"""

from src.paper_trading.models import (
    DailyPnLSnapshot,
    FillSimulationConfig,
    OrderAction,
    OrderResult,
    OrderStatus,
    OrderType,
    PaperOrder,
    PerformanceSummary,
    PortfolioSnapshot,
    PriceTick,
    SessionConfig,
    SessionStatus,
    SignalEvent,
    TradeRecord,
    VirtualCashPosition,
    VirtualPosition,
)

__all__ = [
    "DailyPnLSnapshot",
    "FillSimulationConfig",
    "OrderAction",
    "OrderResult",
    "OrderStatus",
    "OrderType",
    "PaperOrder",
    "PerformanceSummary",
    "PortfolioSnapshot",
    "PriceTick",
    "SessionConfig",
    "SessionStatus",
    "SignalEvent",
    "TradeRecord",
    "VirtualCashPosition",
    "VirtualPosition",
]
