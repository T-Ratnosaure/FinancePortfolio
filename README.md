cl# FinancePortfolio - PEA Portfolio Optimization System

An AI-driven portfolio management system for French PEA (Plan d'Epargne en Actions) accounts, leveraging Hidden Markov Model regime detection and risk-optimized allocation strategies.

## Overview

FinancePortfolio is a personal portfolio optimization tool designed specifically for French PEA accounts. It uses machine learning to detect market regimes (Risk-On, Neutral, Risk-Off) and automatically generates allocation recommendations for a portfolio of PEA-eligible ETFs.

### Key Features

- **Regime Detection**: 3-state Gaussian HMM for market regime classification
- **Risk-Optimized Allocation**: Conservative allocation strategies with hard-coded risk limits
- **Portfolio Tracking**: DuckDB-backed position tracking with broker reconciliation
- **Risk Management**: VaR, CVaR, Sharpe, Sortino, max drawdown analysis
- **Rebalancing Engine**: Drift-based rebalancing with transaction cost optimization
- **Data Pipeline**: Automated fetching from Yahoo Finance and FRED APIs

### Supported ETFs

| Symbol | Name | Type | PEA Eligible |
|--------|------|------|--------------|
| LQQ | Lyxor Nasdaq-100 x2 | Leveraged Equity | Yes |
| CL2 | Amundi Euro Stoxx 50 x2 | Leveraged Equity | Yes |
| WPEA | Amundi MSCI World | World Equity | Yes |

### Risk Limits (Hard-Coded)

| Limit | Value | Description |
|-------|-------|-------------|
| Max Leveraged Exposure | 30% | LQQ + CL2 combined |
| Max Single Position | 25% | Any single ETF |
| Min Cash Buffer | 10% | Always maintain cash |
| Rebalance Threshold | 5% | Drift trigger |
| Drawdown Alert | -20% | Risk warning threshold |

## Quick Start

### Prerequisites

- Python 3.12+
- [UV package manager](https://docs.astral.sh/uv/)
- FRED API key (free from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html))

### Installation

```bash
# Clone repository
git clone https://github.com/T-Ratnosaure/FinancePortfolio.git
cd FinancePortfolio

# Install dependencies with UV
uv sync
```

### Configuration

Create a `.env` file in the project root:

```bash
# Required for FRED macro data
FRED_API_KEY=your_api_key_here

# Optional logging configuration
LOG_LEVEL=INFO
LOG_FORMAT=console  # or 'json' for structured logging
```

### Basic Usage

```bash
# Fetch market data (last 30 days)
python main.py fetch --source all --days 30

# Detect current market regime
python main.py detect

# View portfolio summary
python main.py portfolio --summary

# Generate rebalancing recommendations
python main.py rebalance --dry-run

# Generate risk report
python main.py risk --report
```

## Architecture

### Project Structure

```
FinancePortfolio/
|-- src/
|   |-- data/
|   |   |-- fetchers/        # Yahoo Finance & FRED API clients
|   |   |-- storage/         # DuckDB storage layer
|   |   +-- models.py        # Pydantic data models
|   |-- signals/
|   |   |-- features.py      # Feature engineering
|   |   |-- regime.py        # HMM regime detector
|   |   +-- allocation.py    # Allocation optimizer
|   |-- portfolio/
|   |   |-- tracker.py       # Position tracking
|   |   |-- rebalancer.py    # Trade generation
|   |   +-- risk.py          # Risk calculations
|   +-- config/
|       +-- logging.py       # Centralized logging
|-- tests/                   # Comprehensive test suite (232 tests)
|-- examples/                # Usage examples
|-- compliance/              # Risk disclosures & legal docs
+-- docs/                    # Technical documentation
```

### Data Flow

```
Yahoo Finance / FRED APIs
         |
         v
    Data Fetchers (with retry & rate limiting)
         |
         v
    DuckDB Storage (3-layer: raw/cleaned/analytics)
         |
         v
    Feature Calculator (9 features)
         |
         v
    HMM Regime Detector (3 states)
         |
         v
    Allocation Optimizer (risk-constrained)
         |
         v
    Rebalancer (trade recommendations)
         |
         v
    Manual Execution (no broker integration)
```

### Regime Detection

The system uses a 3-state Gaussian Hidden Markov Model:

| Regime | Description | Allocation Style |
|--------|-------------|------------------|
| RISK_ON | Bull market, low volatility | Higher leveraged exposure |
| NEUTRAL | Uncertain conditions | Balanced allocation |
| RISK_OFF | Bear market, high volatility | Defensive, higher cash |

**Features Used:**
- VIX level and percentile
- 20-day realized volatility
- 50/200 MA trend indicator
- Treasury yield curve (2s10s spread)
- High-yield credit spread
- Momentum indicators

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov --cov-report=term-missing

# Run specific test module
uv run pytest tests/test_signals/test_regime.py -v
```

### Code Quality

```bash
# Import sorting (run first)
uv run isort .

# Code formatting
uv run ruff format .

# Linting
uv run ruff check .

# Type checking
pyrefly check src/

# Security scanning
uv run bandit -c pyproject.toml -r .
```

### Pre-Commit Hooks

```bash
# Install pre-commit
uv add --dev pre-commit
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Package Manager | UV |
| Data Validation | Pydantic v2 |
| Database | DuckDB |
| ML Models | hmmlearn (Gaussian HMM) |
| Data Sources | yfinance, fredapi |
| Testing | pytest |
| Type Checking | pyrefly |
| Formatting | ruff, isort |
| Security | bandit |

## Documentation

- [Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md)
- [Data Pipeline Guide](docs/data_pipeline_architecture.md)
- [Yahoo Fetcher Guide](docs/yahoo_fetcher_guide.md)
- [FRED Fetcher Guide](docs/fred_fetcher_guide.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Sprint 5 Roadmap](docs/SPRINT5_ROADMAP.md)

## Compliance & Risk

### Important Disclaimers

- **Personal Use Only**: This system is for personal portfolio management, not commercial use
- **No Investment Advice**: Recommendations are algorithmic, not professional financial advice
- **PEA Rules**: Users must understand French PEA regulations (5-year holding period, EUR 150,000 contribution ceiling)
- **Leveraged ETF Risks**: LQQ and CL2 are 2x leveraged products with volatility decay risk

### Risk Disclosures

See [compliance/risk_disclosures.md](compliance/risk_disclosures.md) for comprehensive risk documentation including:
- Leveraged ETF-specific risks
- Market regime detection limitations
- Algorithmic trading risks
- PEA regulatory considerations

## Project Status

### Completed (Sprints 1-4)

- [x] Data layer with Yahoo Finance and FRED integration
- [x] DuckDB storage with 3-layer architecture
- [x] Pydantic models with validation
- [x] HMM regime detection
- [x] Feature engineering (9 features)
- [x] Allocation optimization
- [x] Portfolio tracking
- [x] Risk calculations (VaR, Sharpe, Sortino, etc.)
- [x] Rebalancing engine
- [x] Security remediation (pickle vulnerability fixed)
- [x] Centralized logging infrastructure

### In Progress (Sprint 5)

- [ ] Backtesting framework
- [ ] Integration tests
- [ ] Pre-trade risk validation
- [ ] Data staleness detection
- [ ] Paper trading capabilities

### Planned (Future)

- [ ] Broker API integration
- [ ] Dashboard UI
- [ ] Email alerts
- [ ] Multi-portfolio support

## Contributing

This is a personal project, but suggestions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Commit with conventional commits (`feat(scope): description`)
4. Push and create a Pull Request

### Development Guidelines

- Follow [CLAUDE.md](CLAUDE.md) for coding standards
- All code requires type hints
- New features require tests
- Run formatters before committing

## License

This project is for personal use. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Market data provided by [Yahoo Finance](https://finance.yahoo.com/) and [FRED](https://fred.stlouisfed.org/)
- Built with guidance from Claude Code (Anthropic)

---

**Current Version:** Post-Sprint 4
**Last Updated:** December 11, 2025
**Tests:** 222 passing, 10 skipped (network-dependent)
