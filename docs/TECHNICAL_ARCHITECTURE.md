# PEA Portfolio Management System - Technical Architecture

**Document Version:** 1.0
**Author:** Jean-David (IT Core Team Manager)
**Contributors:** Clovis (Code Quality), Lamine (CI/CD), Olivier (QC), Maxime (Security)
**Date:** 2025-12-10

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Module Structure and Responsibilities](#3-module-structure-and-responsibilities)
4. [Data Flow Architecture](#4-data-flow-architecture)
5. [Signal Generation Pipeline](#5-signal-generation-pipeline)
6. [Portfolio Tracking Components](#6-portfolio-tracking-components)
7. [Alerting and Notification System](#7-alerting-and-notification-system)
8. [Backtesting Framework](#8-backtesting-framework)
9. [Deployment Considerations](#9-deployment-considerations)
10. [Testing Strategy](#10-testing-strategy)
11. [CI/CD Requirements](#11-cicd-requirements)
12. [Security Considerations](#12-security-considerations)
13. [Implementation Phases](#13-implementation-phases)

---

## 1. Executive Summary

### 1.1 Purpose

This document defines the technical architecture for a PEA (Plan d'Epargne en Actions) portfolio management system designed to track and manage leveraged ETF positions. The system focuses on:

- **LQQ** (Lyxor Nasdaq-100 x2 Leveraged)
- **CL2** (Amundi ETF Leveraged MSCI USA Daily x2)
- **WPEA** (Amundi MSCI World PEA)

### 1.2 Key Design Principles

| Principle | Description |
|-----------|-------------|
| **Manual Execution** | No broker API integration; system generates signals for human execution |
| **Data-Driven Decisions** | All signals based on quantitative analysis and configurable strategies |
| **Modularity** | Loosely coupled components for easy testing and maintenance |
| **Observability** | Comprehensive logging, metrics, and alerting |
| **Security First** | Encrypted sensitive data, no credential exposure |

### 1.3 Technology Stack

```
Runtime:        Python 3.12
Package Mgmt:   UV (Astral)
Validation:     Pydantic v2
LLM Framework:  LangChain / LangGraph
Data:           pandas, numpy
Visualization:  plotly, matplotlib
Testing:        pytest, anyio
Type Checking:  pyrefly
Linting:        ruff, bandit, isort
```

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                              PEA Portfolio Management System                       |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +------------------+    +------------------+    +------------------+             |
|  |   Data Layer     |    |  Business Logic  |    |  Presentation    |             |
|  |                  |    |                  |    |                  |             |
|  | - Market Data    |--->| - Signal Engine  |--->| - CLI Interface  |             |
|  | - Portfolio DB   |    | - Risk Manager   |    | - Reports        |             |
|  | - Config Store   |    | - Backtester     |    | - Notifications  |             |
|  +------------------+    +------------------+    +------------------+             |
|           ^                      ^                       |                        |
|           |                      |                       v                        |
|  +------------------+    +------------------+    +------------------+             |
|  |  External APIs   |    |   LLM Services   |    |  Alert Channels  |             |
|  |                  |    |                  |    |                  |             |
|  | - Yahoo Finance  |    | - Anthropic API  |    | - Email (SMTP)   |             |
|  | - ECB Rates      |    | - Local Prompts  |    | - Desktop Notif  |             |
|  +------------------+    +------------------+    +------------------+             |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### 2.2 Architectural Style

The system follows a **layered architecture** with clear separation of concerns:

1. **Data Layer**: Handles all data persistence, retrieval, and external API integration
2. **Business Logic Layer**: Contains core algorithms, signal generation, and risk management
3. **Presentation Layer**: CLI interface, reporting, and notifications

### 2.3 Component Interaction Pattern

```
                    +------------------+
                    |   Orchestrator   |
                    | (LangGraph Agent)|
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
         v                   v                   v
+----------------+  +----------------+  +----------------+
| Data Fetcher   |  | Signal Engine  |  | Notification   |
| (Tool)         |  | (Tool)         |  | Manager (Tool) |
+----------------+  +----------------+  +----------------+
```

---

## 3. Module Structure and Responsibilities

### 3.1 Project Directory Structure

```
financeportfolio/
|-- __init__.py
|-- main.py                      # Application entry point
|-- config/
|   |-- __init__.py
|   |-- settings.py              # Pydantic Settings configuration
|   |-- constants.py             # System-wide constants
|   |-- etf_definitions.py       # ETF metadata (LQQ, CL2, WPEA)
|
|-- core/
|   |-- __init__.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- portfolio.py         # Portfolio Pydantic models
|   |   |-- position.py          # Position models
|   |   |-- transaction.py       # Transaction models
|   |   |-- signal.py            # Trading signal models
|   |   |-- market_data.py       # Price/OHLCV models
|   |
|   |-- exceptions.py            # Custom exceptions
|   |-- enums.py                 # Enumerations (SignalType, etc.)
|
|-- data/
|   |-- __init__.py
|   |-- fetchers/
|   |   |-- __init__.py
|   |   |-- base.py              # Abstract base fetcher
|   |   |-- yahoo_fetcher.py     # Yahoo Finance integration
|   |   |-- ecb_fetcher.py       # ECB rate fetcher
|   |
|   |-- storage/
|   |   |-- __init__.py
|   |   |-- portfolio_store.py   # Portfolio persistence (JSON/SQLite)
|   |   |-- cache.py             # Data caching layer
|   |
|   |-- validators.py            # Data validation utilities
|
|-- signals/
|   |-- __init__.py
|   |-- engine.py                # Signal generation orchestrator
|   |-- strategies/
|   |   |-- __init__.py
|   |   |-- base.py              # Abstract strategy
|   |   |-- momentum.py          # Momentum-based signals
|   |   |-- mean_reversion.py    # Mean reversion signals
|   |   |-- volatility.py        # Volatility-based signals
|   |   |-- rebalancing.py       # Threshold rebalancing
|   |
|   |-- indicators/
|   |   |-- __init__.py
|   |   |-- moving_averages.py   # SMA, EMA, etc.
|   |   |-- rsi.py               # Relative Strength Index
|   |   |-- volatility.py        # ATR, Bollinger Bands
|   |   |-- drawdown.py          # Drawdown calculations
|
|-- portfolio/
|   |-- __init__.py
|   |-- tracker.py               # Portfolio state tracking
|   |-- analyzer.py              # Performance analysis
|   |-- rebalancer.py            # Rebalancing logic
|   |-- risk_manager.py          # Risk metrics and limits
|
|-- notifications/
|   |-- __init__.py
|   |-- manager.py               # Notification orchestrator
|   |-- channels/
|   |   |-- __init__.py
|   |   |-- base.py              # Abstract channel
|   |   |-- email.py             # Email notifications
|   |   |-- desktop.py           # Desktop notifications
|   |   |-- console.py           # Console output
|
|-- backtesting/
|   |-- __init__.py
|   |-- engine.py                # Backtest execution engine
|   |-- simulator.py             # Trade simulation
|   |-- metrics.py               # Performance metrics
|   |-- report.py                # Backtest reporting
|
|-- llm/
|   |-- __init__.py
|   |-- agent.py                 # LangGraph agent definition
|   |-- tools/
|   |   |-- __init__.py
|   |   |-- data_tools.py        # Data fetching tools
|   |   |-- signal_tools.py      # Signal generation tools
|   |   |-- portfolio_tools.py   # Portfolio management tools
|   |   |-- notification_tools.py
|   |
|   |-- prompts/
|   |   |-- system.py            # System prompts
|   |   |-- analysis.py          # Analysis prompts
|
|-- cli/
|   |-- __init__.py
|   |-- app.py                   # CLI application
|   |-- commands/
|   |   |-- __init__.py
|   |   |-- portfolio.py         # Portfolio commands
|   |   |-- signals.py           # Signal commands
|   |   |-- backtest.py          # Backtest commands
|
|-- utils/
|   |-- __init__.py
|   |-- dates.py                 # Date utilities
|   |-- formatting.py            # Output formatting
|   |-- logging.py               # Logging configuration
|
tests/
|-- __init__.py
|-- conftest.py                  # Pytest fixtures
|-- unit/
|   |-- test_models.py
|   |-- test_fetchers.py
|   |-- test_signals.py
|   |-- test_portfolio.py
|-- integration/
|   |-- test_data_flow.py
|   |-- test_signal_pipeline.py
|-- e2e/
|   |-- test_full_workflow.py
```

### 3.2 Module Responsibilities Matrix

| Module | Responsibility | Dependencies |
|--------|---------------|--------------|
| `config` | Application settings, ETF definitions | pydantic-settings |
| `core/models` | Data structures, validation | pydantic |
| `data/fetchers` | External data acquisition | httpx, yfinance |
| `data/storage` | Persistence layer | json, sqlite3 |
| `signals` | Trading signal generation | numpy, pandas |
| `portfolio` | Position and performance tracking | core/models |
| `notifications` | Alert delivery | smtplib, plyer |
| `backtesting` | Historical strategy testing | pandas, numpy |
| `llm` | AI-powered analysis | langchain, langgraph |
| `cli` | User interface | typer |

---

## 4. Data Flow Architecture

### 4.1 Data Flow Diagram

```
+-------------+     +---------------+     +----------------+     +-------------+
| Yahoo API   |---->| Data Fetcher  |---->| Data Validator |---->| Cache Layer |
+-------------+     +---------------+     +----------------+     +------+------+
                                                                        |
                                                                        v
+-------------+     +---------------+     +----------------+     +-------------+
| Signals Out |<----| Signal Engine |<----| Indicator Calc |<----| Clean Data  |
+-------------+     +---------------+     +----------------+     +-------------+
      |
      v
+-------------+     +---------------+     +----------------+
| Notification|---->| Alert Manager |---->| User Action    |
+-------------+     +---------------+     +----------------+
```

### 4.2 Data Models (Pydantic)

```python
# core/models/market_data.py
from datetime import date, datetime
from decimal import Decimal
from pydantic import BaseModel, Field

class OHLCVBar(BaseModel):
    """Single OHLCV price bar."""

    date: date
    open: Decimal = Field(ge=0)
    high: Decimal = Field(ge=0)
    low: Decimal = Field(ge=0)
    close: Decimal = Field(ge=0)
    volume: int = Field(ge=0)
    adjusted_close: Decimal | None = None


class ETFPriceHistory(BaseModel):
    """Price history for an ETF."""

    ticker: str
    name: str
    currency: str = "EUR"
    bars: list[OHLCVBar]
    last_updated: datetime


# core/models/position.py
class Position(BaseModel):
    """Individual ETF position."""

    ticker: str
    shares: Decimal = Field(ge=0)
    average_cost: Decimal = Field(ge=0)
    current_price: Decimal | None = None

    @property
    def market_value(self) -> Decimal | None:
        if self.current_price is None:
            return None
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> Decimal | None:
        if self.current_price is None:
            return None
        return (self.current_price - self.average_cost) * self.shares


# core/models/portfolio.py
class Portfolio(BaseModel):
    """Complete portfolio state."""

    id: str
    name: str
    positions: dict[str, Position]  # ticker -> Position
    cash_balance: Decimal = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    target_allocations: dict[str, Decimal] | None = None  # ticker -> weight


# core/models/signal.py
from enum import Enum

class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REBALANCE = "rebalance"


class TradingSignal(BaseModel):
    """Trading signal with reasoning."""

    ticker: str
    signal_type: SignalType
    strength: float = Field(ge=0, le=1)
    target_shares: Decimal | None = None
    target_value: Decimal | None = None
    reasoning: str
    generated_at: datetime
    strategy_name: str
    confidence: float = Field(ge=0, le=1)
```

### 4.3 Data Caching Strategy

```python
# data/storage/cache.py
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel

class CacheConfig(BaseModel):
    """Cache configuration."""

    cache_dir: Path = Path(".cache")
    price_ttl: timedelta = timedelta(hours=1)
    portfolio_ttl: timedelta = timedelta(minutes=5)

    def is_stale(self, cached_at: datetime, data_type: str) -> bool:
        ttl = self.price_ttl if data_type == "price" else self.portfolio_ttl
        return datetime.now() - cached_at > ttl
```

---

## 5. Signal Generation Pipeline

### 5.1 Pipeline Architecture

```
+------------------+
| Raw Market Data  |
+--------+---------+
         |
         v
+------------------+     +------------------+
| Indicator Calc   |---->| Indicator Cache  |
+--------+---------+     +------------------+
         |
         v
+------------------+
| Strategy Layer   |
| - Momentum       |
| - Mean Reversion |
| - Volatility     |
| - Rebalancing    |
+--------+---------+
         |
         v
+------------------+
| Signal Aggregator|
+--------+---------+
         |
         v
+------------------+
| Risk Filter      |
+--------+---------+
         |
         v
+------------------+
| Final Signals    |
+------------------+
```

### 5.2 Strategy Interface

```python
# signals/strategies/base.py
from abc import ABC, abstractmethod
from typing import Protocol

from core.models.market_data import ETFPriceHistory
from core.models.signal import TradingSignal


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for identification."""
        pass

    @abstractmethod
    def generate_signal(
        self,
        ticker: str,
        price_history: ETFPriceHistory,
        current_position: Position | None = None,
    ) -> TradingSignal | None:
        """Generate a trading signal based on price history."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict[str, any]:
        """Return current strategy parameters."""
        pass


# signals/strategies/momentum.py
class MomentumStrategy(Strategy):
    """Momentum-based trading strategy using moving averages."""

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 50,
        threshold: float = 0.02,
    ) -> None:
        self._short_window = short_window
        self._long_window = long_window
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"Momentum({self._short_window}/{self._long_window})"

    def generate_signal(
        self,
        ticker: str,
        price_history: ETFPriceHistory,
        current_position: Position | None = None,
    ) -> TradingSignal | None:
        # Implementation details...
        pass


# signals/strategies/rebalancing.py
class ThresholdRebalancingStrategy(Strategy):
    """Rebalance when allocation drifts beyond threshold."""

    def __init__(
        self,
        target_allocations: dict[str, Decimal],
        threshold: float = 0.05,  # 5% drift threshold
    ) -> None:
        self._target_allocations = target_allocations
        self._threshold = threshold

    def check_rebalance_needed(
        self,
        portfolio: Portfolio,
    ) -> list[TradingSignal]:
        """Check if rebalancing is needed for any position."""
        pass
```

### 5.3 Signal Aggregation

```python
# signals/engine.py
from dataclasses import dataclass

@dataclass
class SignalEngineConfig:
    """Configuration for signal generation engine."""

    strategies: list[Strategy]
    aggregation_method: str = "weighted_average"
    min_confidence: float = 0.5
    risk_filters_enabled: bool = True


class SignalEngine:
    """Orchestrates signal generation across strategies."""

    def __init__(self, config: SignalEngineConfig) -> None:
        self._config = config
        self._strategies = config.strategies

    def generate_signals(
        self,
        portfolio: Portfolio,
        price_data: dict[str, ETFPriceHistory],
    ) -> list[TradingSignal]:
        """Generate aggregated signals for all positions."""
        raw_signals: list[TradingSignal] = []

        for ticker, history in price_data.items():
            position = portfolio.positions.get(ticker)

            for strategy in self._strategies:
                signal = strategy.generate_signal(ticker, history, position)
                if signal and signal.confidence >= self._config.min_confidence:
                    raw_signals.append(signal)

        aggregated = self._aggregate_signals(raw_signals)

        if self._config.risk_filters_enabled:
            return self._apply_risk_filters(aggregated, portfolio)

        return aggregated
```

---

## 6. Portfolio Tracking Components

### 6.1 Portfolio Tracker

```python
# portfolio/tracker.py
from datetime import datetime
from pathlib import Path

from core.models.portfolio import Portfolio
from core.models.transaction import Transaction
from data.storage.portfolio_store import PortfolioStore


class PortfolioTracker:
    """Manages portfolio state and transaction history."""

    def __init__(self, store: PortfolioStore) -> None:
        self._store = store
        self._portfolio: Portfolio | None = None

    def load_portfolio(self, portfolio_id: str) -> Portfolio:
        """Load portfolio from storage."""
        self._portfolio = self._store.load(portfolio_id)
        return self._portfolio

    def record_transaction(self, transaction: Transaction) -> None:
        """Record a transaction and update portfolio state."""
        if self._portfolio is None:
            raise ValueError("No portfolio loaded")

        self._apply_transaction(transaction)
        self._portfolio.updated_at = datetime.now()
        self._store.save(self._portfolio)
        self._store.append_transaction(self._portfolio.id, transaction)

    def get_current_allocations(self) -> dict[str, Decimal]:
        """Calculate current portfolio allocations."""
        if self._portfolio is None:
            raise ValueError("No portfolio loaded")

        total_value = self.get_total_value()
        if total_value == 0:
            return {}

        return {
            ticker: position.market_value / total_value
            for ticker, position in self._portfolio.positions.items()
            if position.market_value is not None
        }

    def get_drift_from_target(self) -> dict[str, Decimal]:
        """Calculate drift from target allocations."""
        if self._portfolio.target_allocations is None:
            return {}

        current = self.get_current_allocations()
        return {
            ticker: current.get(ticker, Decimal(0)) - target
            for ticker, target in self._portfolio.target_allocations.items()
        }
```

### 6.2 Performance Analyzer

```python
# portfolio/analyzer.py
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""

    total_return: Decimal
    annualized_return: Decimal
    volatility: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    win_rate: Decimal | None
    avg_win: Decimal | None
    avg_loss: Decimal | None


class PortfolioAnalyzer:
    """Calculates portfolio performance metrics."""

    def __init__(self, risk_free_rate: Decimal = Decimal("0.03")) -> None:
        self._risk_free_rate = risk_free_rate

    def calculate_metrics(
        self,
        portfolio: Portfolio,
        price_history: dict[str, ETFPriceHistory],
        transactions: list[Transaction],
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        pass

    def calculate_correlation_matrix(
        self,
        price_histories: dict[str, ETFPriceHistory],
    ) -> pd.DataFrame:
        """Calculate correlation between ETF returns."""
        pass
```

### 6.3 Risk Manager

```python
# portfolio/risk_manager.py
from dataclasses import dataclass

@dataclass
class RiskLimits:
    """Portfolio risk limits."""

    max_position_size: Decimal = Decimal("0.40")  # 40% max single position
    max_leverage_exposure: Decimal = Decimal("0.70")  # 70% max in leveraged ETFs
    max_drawdown_alert: Decimal = Decimal("0.15")  # 15% drawdown alert
    max_volatility: Decimal = Decimal("0.30")  # 30% annualized vol limit


class RiskManager:
    """Monitors and enforces risk limits."""

    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits

    def check_limits(self, portfolio: Portfolio) -> list[RiskAlert]:
        """Check all risk limits and return any violations."""
        alerts = []

        alerts.extend(self._check_position_limits(portfolio))
        alerts.extend(self._check_leverage_exposure(portfolio))
        alerts.extend(self._check_drawdown(portfolio))

        return alerts

    def validate_signal(
        self,
        signal: TradingSignal,
        portfolio: Portfolio,
    ) -> tuple[bool, str | None]:
        """Validate if executing signal would violate risk limits."""
        pass
```

---

## 7. Alerting and Notification System

### 7.1 Notification Architecture

```
+------------------+
| Signal Generated |
+--------+---------+
         |
         v
+------------------+
| Alert Classifier |
| - Priority       |
| - Channel Select |
+--------+---------+
         |
    +----+----+
    |         |
    v         v
+-------+ +--------+
| Email | |Desktop |
+-------+ +--------+
```

### 7.2 Notification Manager

```python
# notifications/manager.py
from enum import Enum
from typing import Protocol

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(Protocol):
    """Protocol for notification channels."""

    def send(self, message: str, priority: AlertPriority) -> bool:
        """Send notification through this channel."""
        ...


class NotificationManager:
    """Orchestrates notifications across channels."""

    def __init__(self, channels: list[NotificationChannel]) -> None:
        self._channels = channels
        self._priority_routing: dict[AlertPriority, list[str]] = {
            AlertPriority.LOW: ["console"],
            AlertPriority.MEDIUM: ["console", "desktop"],
            AlertPriority.HIGH: ["console", "desktop", "email"],
            AlertPriority.CRITICAL: ["console", "desktop", "email"],
        }

    def notify_signal(self, signal: TradingSignal) -> None:
        """Send notification for a trading signal."""
        priority = self._classify_priority(signal)
        message = self._format_signal_message(signal)

        for channel in self._get_channels_for_priority(priority):
            channel.send(message, priority)

    def notify_rebalance(
        self,
        portfolio: Portfolio,
        signals: list[TradingSignal],
    ) -> None:
        """Send rebalancing notification with all required trades."""
        pass
```

### 7.3 Channel Implementations

```python
# notifications/channels/email.py
import smtplib
from email.mime.text import MIMEText

from pydantic import BaseModel, SecretStr


class EmailConfig(BaseModel):
    """Email configuration."""

    smtp_server: str
    smtp_port: int = 587
    username: str
    password: SecretStr
    from_address: str
    to_addresses: list[str]
    use_tls: bool = True


class EmailChannel:
    """Email notification channel."""

    def __init__(self, config: EmailConfig) -> None:
        self._config = config

    def send(self, message: str, priority: AlertPriority) -> bool:
        """Send email notification."""
        subject = self._get_subject(priority)

        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = self._config.from_address
        msg["To"] = ", ".join(self._config.to_addresses)

        try:
            with smtplib.SMTP(
                self._config.smtp_server,
                self._config.smtp_port,
            ) as server:
                if self._config.use_tls:
                    server.starttls()
                server.login(
                    self._config.username,
                    self._config.password.get_secret_value(),
                )
                server.send_message(msg)
            return True
        except Exception:
            return False
```

---

## 8. Backtesting Framework

### 8.1 Backtest Architecture

```
+------------------+     +------------------+
| Historical Data  |---->| Data Preprocessor|
+------------------+     +--------+---------+
                                  |
                                  v
                         +------------------+
                         | Strategy Config  |
                         +--------+---------+
                                  |
                                  v
+------------------+     +------------------+
| Trade Simulator  |<----| Backtest Engine  |
+--------+---------+     +------------------+
         |
         v
+------------------+     +------------------+
| Performance Calc |---->| Report Generator |
+------------------+     +------------------+
```

### 8.2 Backtest Engine

```python
# backtesting/engine.py
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

@dataclass
class BacktestConfig:
    """Backtest configuration."""

    start_date: date
    end_date: date
    initial_capital: Decimal
    transaction_cost_bps: Decimal = Decimal("10")  # 0.10%
    slippage_bps: Decimal = Decimal("5")  # 0.05%
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly


@dataclass
class BacktestResult:
    """Complete backtest results."""

    config: BacktestConfig
    final_value: Decimal
    total_return: Decimal
    annualized_return: Decimal
    volatility: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    trade_count: int
    win_rate: Decimal
    equity_curve: list[tuple[date, Decimal]]
    trades: list[SimulatedTrade]


class BacktestEngine:
    """Executes strategy backtests."""

    def __init__(
        self,
        config: BacktestConfig,
        strategies: list[Strategy],
    ) -> None:
        self._config = config
        self._strategies = strategies
        self._simulator = TradeSimulator(
            transaction_cost_bps=config.transaction_cost_bps,
            slippage_bps=config.slippage_bps,
        )

    def run(
        self,
        price_data: dict[str, ETFPriceHistory],
        target_allocations: dict[str, Decimal] | None = None,
    ) -> BacktestResult:
        """Execute backtest and return results."""
        portfolio = self._initialize_portfolio()
        equity_curve = []
        trades = []

        for current_date in self._get_trading_dates():
            # Get prices for current date
            current_prices = self._get_prices_for_date(
                price_data, current_date
            )

            # Generate signals
            signals = self._generate_signals(portfolio, price_data, current_date)

            # Execute trades
            for signal in signals:
                trade = self._simulator.execute(signal, portfolio, current_prices)
                if trade:
                    trades.append(trade)

            # Record equity
            equity_curve.append((current_date, self._calculate_equity(portfolio)))

        return self._calculate_results(equity_curve, trades)
```

### 8.3 Performance Metrics

```python
# backtesting/metrics.py
import numpy as np
import pandas as pd

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    return float(
        np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    )


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from equity curve."""
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


def calculate_calmar_ratio(
    annualized_return: float,
    max_drawdown: float,
) -> float:
    """Calculate Calmar ratio (return / max drawdown)."""
    if max_drawdown == 0:
        return 0.0
    return annualized_return / abs(max_drawdown)
```

---

## 9. Deployment Considerations

### 9.1 Deployment Options Comparison

| Aspect | Local Deployment | Cloud Deployment |
|--------|-----------------|------------------|
| **Cost** | Free (uses existing hardware) | Monthly fee (~$5-20/mo) |
| **Availability** | When PC is running | 24/7 availability |
| **Alerts** | Desktop notifications | Email, SMS, webhooks |
| **Data Storage** | Local SQLite/JSON | Cloud database |
| **Scheduling** | Windows Task Scheduler | Cloud scheduler (cron) |
| **Complexity** | Lower | Higher |
| **Maintenance** | Manual updates | CI/CD automated |

### 9.2 Recommended Deployment: Hybrid Approach

```
+-------------------+
| Local Development |
| - CLI Interface   |
| - Backtesting     |
| - Manual Analysis |
+--------+----------+
         |
         | git push
         v
+-------------------+     +-------------------+
| GitHub Actions    |---->| Scheduled Checks  |
| - Run signals     |     | - Daily at 18:00  |
| - Send alerts     |     | - Market close    |
+-------------------+     +-------------------+
         |
         v
+-------------------+
| Email/Webhook     |
| Notifications     |
+-------------------+
```

### 9.3 Local Deployment Configuration

```python
# config/settings.py
from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PEA_",
    )

    # Data paths
    data_dir: Path = Path("./data")
    cache_dir: Path = Path("./.cache")

    # ETF configuration
    tracked_etfs: list[str] = ["LQQ.PA", "CL2.PA", "WPEA.PA"]

    # Notification settings
    email_enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: SecretStr = SecretStr("")

    # LLM settings
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-4-20250514"

    # Risk settings
    max_position_pct: float = 0.40
    rebalance_threshold: float = 0.05

    # Scheduling
    check_frequency_hours: int = 24
```

### 9.4 Docker Configuration (Optional)

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install UV
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

# Run application
CMD ["uv", "run", "python", "-m", "financeportfolio.cli"]
```

---

## 10. Testing Strategy

### 10.1 Testing Pyramid

```
        /\
       /  \
      / E2E\        <- Few, critical path tests
     /------\
    /  Integ \      <- API contracts, data flow
   /----------\
  /    Unit    \    <- Many, fast, isolated
 /--------------\
```

### 10.2 Test Categories

#### Unit Tests

```python
# tests/unit/test_signals.py
import pytest
from decimal import Decimal
from datetime import date

from core.models.market_data import OHLCVBar, ETFPriceHistory
from signals.strategies.momentum import MomentumStrategy


class TestMomentumStrategy:
    """Unit tests for momentum strategy."""

    @pytest.fixture
    def strategy(self) -> MomentumStrategy:
        return MomentumStrategy(short_window=5, long_window=20)

    @pytest.fixture
    def bullish_history(self) -> ETFPriceHistory:
        """Price history with clear uptrend."""
        bars = [
            OHLCVBar(
                date=date(2024, 1, i),
                open=Decimal(100 + i),
                high=Decimal(101 + i),
                low=Decimal(99 + i),
                close=Decimal(100 + i),
                volume=1000000,
            )
            for i in range(1, 31)
        ]
        return ETFPriceHistory(
            ticker="LQQ.PA",
            name="Lyxor Nasdaq-100 x2",
            bars=bars,
        )

    def test_generates_buy_signal_in_uptrend(
        self,
        strategy: MomentumStrategy,
        bullish_history: ETFPriceHistory,
    ) -> None:
        signal = strategy.generate_signal("LQQ.PA", bullish_history)

        assert signal is not None
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence > 0.5

    def test_returns_none_for_insufficient_data(
        self,
        strategy: MomentumStrategy,
    ) -> None:
        short_history = ETFPriceHistory(
            ticker="LQQ.PA",
            name="Lyxor Nasdaq-100 x2",
            bars=[],  # No data
        )

        signal = strategy.generate_signal("LQQ.PA", short_history)

        assert signal is None
```

#### Integration Tests

```python
# tests/integration/test_signal_pipeline.py
import pytest
from anyio import create_task_group

from data.fetchers.yahoo_fetcher import YahooFetcher
from signals.engine import SignalEngine, SignalEngineConfig
from signals.strategies.momentum import MomentumStrategy


class TestSignalPipeline:
    """Integration tests for signal generation pipeline."""

    @pytest.fixture
    def fetcher(self) -> YahooFetcher:
        return YahooFetcher()

    @pytest.fixture
    def signal_engine(self) -> SignalEngine:
        config = SignalEngineConfig(
            strategies=[MomentumStrategy()],
            min_confidence=0.3,
        )
        return SignalEngine(config)

    @pytest.mark.anyio
    async def test_full_signal_generation_pipeline(
        self,
        fetcher: YahooFetcher,
        signal_engine: SignalEngine,
    ) -> None:
        # Fetch real data
        price_data = await fetcher.fetch_history(
            tickers=["LQQ.PA"],
            period="3mo",
        )

        # Create test portfolio
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            positions={},
            cash_balance=Decimal("10000"),
        )

        # Generate signals
        signals = signal_engine.generate_signals(portfolio, price_data)

        # Verify structure
        for signal in signals:
            assert signal.ticker == "LQQ.PA"
            assert signal.signal_type in SignalType
            assert 0 <= signal.confidence <= 1
```

### 10.3 Test Configuration

```python
# tests/conftest.py
import pytest
from pathlib import Path
from decimal import Decimal

from core.models.portfolio import Portfolio


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Standard test portfolio with typical allocations."""
    return Portfolio(
        id="test-001",
        name="Test PEA Portfolio",
        positions={
            "LQQ.PA": Position(
                ticker="LQQ.PA",
                shares=Decimal("10"),
                average_cost=Decimal("650.00"),
            ),
            "CL2.PA": Position(
                ticker="CL2.PA",
                shares=Decimal("15"),
                average_cost=Decimal("45.00"),
            ),
            "WPEA.PA": Position(
                ticker="WPEA.PA",
                shares=Decimal("100"),
                average_cost=Decimal("28.00"),
            ),
        },
        cash_balance=Decimal("500.00"),
        target_allocations={
            "LQQ.PA": Decimal("0.35"),
            "CL2.PA": Decimal("0.35"),
            "WPEA.PA": Decimal("0.30"),
        },
    )


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
```

### 10.4 Coverage Requirements

| Module | Minimum Coverage |
|--------|-----------------|
| `core/models` | 95% |
| `signals/strategies` | 90% |
| `portfolio` | 85% |
| `data/fetchers` | 80% |
| `notifications` | 75% |
| **Overall** | **85%** |

---

## 11. CI/CD Requirements

### 11.1 Enhanced CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [master, main]
  pull_request:
    branches: [master, main]

jobs:
  quality-checks:
    name: Code Quality & Tests
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version-file: '.python-version'

      - name: Install UV
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"
          enable-cache: true

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Check import sorting with isort
        run: uv run isort --check-only --diff .

      - name: Check code formatting with Ruff
        run: uv run ruff format --check .

      - name: Lint with Ruff
        run: uv run ruff check .

      - name: Type check with pyrefly
        run: uv run pyrefly check

      - name: Security scan with Bandit
        run: uv run bandit -c pyproject.toml -r . --severity-level medium

      - name: Complexity check with Xenon
        run: uv run xenon --max-absolute B --max-modules A --max-average A . --exclude ".venv,venv"

      - name: Run tests with pytest
        run: uv run pytest -v --cov --cov-report=xml --cov-fail-under=85

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false

  # Scheduled signal generation job
  scheduled-signals:
    name: Daily Signal Check
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version-file: '.python-version'

      - name: Install UV
        uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: uv sync

      - name: Generate signals
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: uv run python -m financeportfolio.cli signals generate --notify
```

### 11.2 Scheduled Signal Generation

```yaml
# .github/workflows/scheduled-signals.yml
name: Scheduled Signal Generation

on:
  schedule:
    # Run at 18:00 UTC on trading days (Mon-Fri)
    - cron: '0 18 * * 1-5'
  workflow_dispatch:  # Allow manual trigger

jobs:
  generate-signals:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install UV
        uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: uv sync

      - name: Check for trading signals
        env:
          PEA_SMTP_SERVER: ${{ secrets.SMTP_SERVER }}
          PEA_SMTP_USERNAME: ${{ secrets.SMTP_USERNAME }}
          PEA_SMTP_PASSWORD: ${{ secrets.SMTP_PASSWORD }}
          PEA_EMAIL_ENABLED: "true"
        run: |
          uv run python -m financeportfolio.cli signals check \
            --portfolio default \
            --notify-email
```

---

## 12. Security Considerations

*Contributed by Maxime (Security)*

### 12.1 Threat Model

| Threat | Risk Level | Mitigation |
|--------|-----------|------------|
| API key exposure | High | Use secrets manager, never commit |
| Portfolio data leak | Medium | Encrypt at rest, access controls |
| Malicious dependencies | Medium | Pin versions, audit regularly |
| Email credential theft | Medium | App-specific passwords, TLS |

### 12.2 Security Implementation

```python
# config/security.py
from pydantic import SecretStr
from cryptography.fernet import Fernet


class SecureStorage:
    """Secure storage for sensitive data."""

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key or Fernet.generate_key()
        self._fernet = Fernet(self._key)

    def encrypt(self, data: str) -> bytes:
        """Encrypt sensitive data."""
        return self._fernet.encrypt(data.encode())

    def decrypt(self, encrypted: bytes) -> str:
        """Decrypt sensitive data."""
        return self._fernet.decrypt(encrypted).decode()


# Environment variable validation
def validate_env_security() -> list[str]:
    """Check for security issues in environment."""
    warnings = []

    if os.getenv("PEA_SMTP_PASSWORD") and not os.getenv("PEA_SMTP_TLS"):
        warnings.append("SMTP password set without TLS enabled")

    if os.path.exists(".env") and not os.path.exists(".gitignore"):
        warnings.append(".env file exists without .gitignore")

    return warnings
```

### 12.3 Secrets Management

```
.env (NEVER commit)
-------------------
PEA_SMTP_PASSWORD=secret_value
PEA_ANTHROPIC_API_KEY=sk-xxx
PEA_ENCRYPTION_KEY=base64_key

.env.example (commit this)
--------------------------
PEA_SMTP_PASSWORD=
PEA_ANTHROPIC_API_KEY=
PEA_ENCRYPTION_KEY=
```

---

## 13. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

- [ ] Project structure setup
- [ ] Core Pydantic models
- [ ] Yahoo Finance data fetcher
- [ ] Basic portfolio storage (JSON)
- [ ] Unit tests for models

### Phase 2: Signal Engine (Weeks 3-4)

- [ ] Indicator calculations (SMA, RSI, ATR)
- [ ] Momentum strategy implementation
- [ ] Rebalancing strategy
- [ ] Signal aggregation engine
- [ ] Signal tests

### Phase 3: Portfolio Management (Weeks 5-6)

- [ ] Portfolio tracker implementation
- [ ] Performance analyzer
- [ ] Risk manager
- [ ] Transaction recording
- [ ] Integration tests

### Phase 4: Notifications (Week 7)

- [ ] Notification manager
- [ ] Email channel
- [ ] Desktop notifications
- [ ] Alert formatting

### Phase 5: CLI & Integration (Week 8)

- [ ] CLI application with Typer
- [ ] Command implementations
- [ ] End-to-end testing
- [ ] Documentation

### Phase 6: Backtesting (Weeks 9-10)

- [ ] Backtest engine
- [ ] Trade simulator
- [ ] Performance metrics
- [ ] Report generation
- [ ] Historical validation

### Phase 7: LLM Integration (Weeks 11-12)

- [ ] LangGraph agent setup
- [ ] Tool definitions
- [ ] Prompt engineering
- [ ] Analysis capabilities

---

## Appendix A: ETF Definitions

```python
# config/etf_definitions.py
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ETFDefinition:
    """ETF metadata and characteristics."""

    ticker: str
    yahoo_ticker: str
    name: str
    leverage: Decimal
    underlying_index: str
    currency: str
    ter: Decimal  # Total Expense Ratio
    is_leveraged: bool


ETF_REGISTRY: dict[str, ETFDefinition] = {
    "LQQ": ETFDefinition(
        ticker="LQQ",
        yahoo_ticker="LQQ.PA",
        name="Lyxor Nasdaq-100 Daily (2x) Leveraged UCITS ETF",
        leverage=Decimal("2.0"),
        underlying_index="Nasdaq-100",
        currency="EUR",
        ter=Decimal("0.0060"),  # 0.60%
        is_leveraged=True,
    ),
    "CL2": ETFDefinition(
        ticker="CL2",
        yahoo_ticker="CL2.PA",
        name="Amundi ETF Leveraged MSCI USA Daily UCITS ETF",
        leverage=Decimal("2.0"),
        underlying_index="MSCI USA",
        currency="EUR",
        ter=Decimal("0.0050"),  # 0.50%
        is_leveraged=True,
    ),
    "WPEA": ETFDefinition(
        ticker="WPEA",
        yahoo_ticker="WPEA.PA",
        name="Amundi MSCI World UCITS ETF - EUR (C)",
        leverage=Decimal("1.0"),
        underlying_index="MSCI World",
        currency="EUR",
        ter=Decimal("0.0038"),  # 0.38%
        is_leveraged=False,
    ),
}
```

---

## Appendix B: Recommended Dependencies

```toml
# pyproject.toml additions

[project]
dependencies = [
    # Core
    "pydantic>=2.12.5",
    "pydantic-settings>=2.7.1",

    # LLM
    "langchain>=1.1.3",
    "langchain-anthropic>=1.2.0",
    "langgraph>=1.0.4",

    # Data
    "pandas>=2.2.0",
    "numpy>=2.0.0",
    "yfinance>=0.2.50",
    "httpx>=0.28.0",

    # CLI
    "typer>=0.15.0",
    "rich>=13.9.0",

    # Notifications
    "plyer>=2.1.0",  # Desktop notifications

    # Visualization
    "plotly>=5.24.0",

    # Security
    "cryptography>=44.0.0",
]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-cov>=7.0.0",
    "pytest-anyio>=0.0.0",
    "ruff>=0.14.8",
    "bandit>=1.9.2",
    "isort>=7.0.0",
    "xenon>=0.9.3",
    "pyrefly>=0.1.0",
]
```

---

**Document prepared by the IT Core Team**

- **Jean-David** - IT Core Team Manager (Architecture Lead)
- **Clovis** - Code Quality & Git Workflow
- **Lamine** - CI/CD & Deployment
- **Olivier** - Quality Control
- **Maxime** - Security

*This document should be reviewed and updated as the implementation progresses.*
