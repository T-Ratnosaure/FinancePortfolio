"""Backtesting framework for PEA Portfolio strategy validation.

This module provides infrastructure for backtesting trading strategies with
realistic transaction costs, walk-forward validation, and comprehensive
performance metrics.

Implemented:
- Transaction cost modeling (costs.py)
- Trade simulation (simulator.py)
- Basic models for trades and rebalancing (models.py)
- Walk-forward validation (walk_forward.py)
- Performance metrics calculation (metrics.py)
- Complete backtesting orchestration (engine.py)
"""

from src.backtesting.costs import TransactionCostModel
from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import BacktestMetrics
from src.backtesting.models import (
    BacktestResult,
    MarketConditions,
    RebalanceResult,
    RegimePerformance,
    Trade,
    WindowResult,
)
from src.backtesting.simulator import TradeSimulator
from src.backtesting.walk_forward import (
    DEFAULT_STEP_MONTHS,
    DEFAULT_TEST_YEARS,
    DEFAULT_TRAIN_YEARS,
    MIN_HMM_TRAINING_SAMPLES,
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_PER_YEAR,
    LookaheadBiasError,
    WalkForwardConfig,
    WalkForwardValidator,
    WalkForwardWindow,
)

__all__ = [
    # Core classes
    "BacktestEngine",
    "TransactionCostModel",
    "TradeSimulator",
    "BacktestMetrics",
    # Models
    "Trade",
    "RebalanceResult",
    "MarketConditions",
    "WindowResult",
    "BacktestResult",
    "RegimePerformance",
    # Walk-forward validation
    "WalkForwardWindow",
    "WalkForwardConfig",
    "WalkForwardValidator",
    "LookaheadBiasError",
    # Constants
    "TRADING_DAYS_PER_YEAR",
    "TRADING_DAYS_PER_MONTH",
    "DEFAULT_TRAIN_YEARS",
    "DEFAULT_TEST_YEARS",
    "DEFAULT_STEP_MONTHS",
    "MIN_HMM_TRAINING_SAMPLES",
]
