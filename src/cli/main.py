"""Main CLI entry point for PEA Portfolio application.

This module defines the main Typer application and registers all subcommands.
It provides logging configuration options and error handling.
"""

import logging
from typing import Optional

import typer
from rich.console import Console

from src.config.logging import setup_logging

# Create the main Typer application
app = typer.Typer(
    name="pea-portfolio",
    help="PEA Portfolio Optimization System - Manage and analyze your PEA portfolio.",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Rich console for beautiful output
console = Console()


# Callback for global options
@app.callback()
def main_callback(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output (DEBUG level logging).",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-essential output (WARNING level logging).",
    ),
    log_file: Optional[str] = typer.Option(
        None,
        "--log-file",
        help="Path to log file for persistent logging.",
    ),
) -> None:
    """PEA Portfolio Optimization System.

    A command-line tool for managing PEA (Plan d'Epargne en Actions) portfolios
    using regime-based allocation strategies and quantitative analysis.
    """
    # Determine log level
    if verbose and quiet:
        console.print(
            "[yellow]Warning:[/yellow] Both --verbose and --quiet specified. "
            "Using --verbose."
        )
        log_level = "DEBUG"
    elif verbose:
        log_level = "DEBUG"
    elif quiet:
        log_level = "WARNING"
    else:
        log_level = "INFO"

    # Setup logging with appropriate level
    from pathlib import Path

    log_path = Path(log_file) if log_file else None
    setup_logging(level=log_level, log_file=log_path)


# Import and register subcommands
# These imports are used at module level to register commands
from src.cli.commands.analyze import analyze  # noqa: E402
from src.cli.commands.backtest import backtest  # noqa: E402
from src.cli.commands.fetch import fetch  # noqa: E402
from src.cli.commands.status import status  # noqa: E402

app.command(name="analyze")(analyze)
app.command(name="backtest")(backtest)
app.command(name="status")(status)
app.command(name="fetch")(fetch)


def cli_main() -> None:
    """Entry point for the CLI application."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        raise typer.Exit(code=130) from None
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception("Unexpected error occurred")
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    cli_main()
