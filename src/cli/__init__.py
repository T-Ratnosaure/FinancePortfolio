"""CLI module for PEA Portfolio management.

This module provides a command-line interface for managing and analyzing
PEA (Plan d'Epargne en Actions) portfolios, including regime detection,
backtesting, and portfolio status monitoring.

Commands:
    analyze: Run market regime analysis
    backtest: Run strategy backtest with parameters
    status: Show current portfolio status
    fetch: Fetch latest market data
"""

from src.cli.main import app

__all__ = ["app"]
