"""CLI commands for PEA Portfolio management."""

from src.cli.commands.analyze import analyze
from src.cli.commands.backtest import backtest
from src.cli.commands.fetch import fetch
from src.cli.commands.status import status

__all__ = ["analyze", "backtest", "fetch", "status"]
