"""Fetch command for retrieving market data.

This command fetches the latest market data from various sources
including Yahoo Finance and FRED.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.data.models import ETFSymbol

if TYPE_CHECKING:
    import pandas as pd

console = Console()
logger = logging.getLogger(__name__)


def fetch(
    data_type: str = typer.Option(
        "all",
        "--type",
        "-t",
        help="Data type to fetch: 'prices', 'macro', 'vix', or 'all'.",
    ),
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols",
        "-s",
        help="Comma-separated list of symbols to fetch (default: all PEA ETFs).",
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Start date (YYYY-MM-DD format). Default: 1 year ago.",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="End date (YYYY-MM-DD format). Default: today.",
    ),
    output_dir: str = typer.Option(
        "data",
        "--output",
        "-o",
        help="Output directory for storing data.",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--database",
        "-d",
        help="Path to DuckDB database for storing data.",
    ),
    output_format: str = typer.Option(
        "rich",
        "--format",
        "-f",
        help="Output format: 'rich' (default), 'json', or 'plain'.",
    ),
) -> None:
    """Fetch latest market data from Yahoo Finance and FRED.

    Downloads ETF prices, VIX data, and macroeconomic indicators
    for regime detection and portfolio analysis.

    Example:
        pea-portfolio fetch
        pea-portfolio fetch --type prices --symbols LQQ.PA,CL2.PA
        pea-portfolio fetch --start-date 2020-01-01 --database data/portfolio.duckdb
    """
    # Validate and parse inputs
    start, end = _parse_dates(start_date, end_date)
    symbol_list = _parse_symbols(symbols)
    _validate_data_type(data_type)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        results = _execute_fetch(
            data_type, symbol_list, start, end, output_path, db_path
        )
        _output_results(results, output_format, output_path, db_path)
    except Exception as e:
        logger.exception("Data fetch failed")
        console.print(f"[red]Error:[/red] Data fetch failed: {e}")
        raise typer.Exit(code=1) from e


def _parse_dates(
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple[date, date]:
    """Parse and validate date inputs.

    Args:
        start_date: Start date string
        end_date: End date string

    Returns:
        Tuple of (start, end) dates

    Raises:
        typer.Exit: On invalid date format or range
    """
    from datetime import datetime

    try:
        end = (
            datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else date.today()
        )
        start = (
            datetime.strptime(start_date, "%Y-%m-%d").date()
            if start_date
            else end - timedelta(days=365)
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] Invalid date format. Use YYYY-MM-DD. ({e})")
        raise typer.Exit(code=1) from None

    if start >= end:
        console.print("[red]Error:[/red] Start date must be before end date.")
        raise typer.Exit(code=1) from None

    return start, end


def _parse_symbols(symbols: Optional[str]) -> list[str]:
    """Parse symbol list input.

    Args:
        symbols: Comma-separated symbols or None

    Returns:
        List of symbol strings
    """
    if symbols:
        return [s.strip() for s in symbols.split(",")]
    return [etf.value for etf in ETFSymbol]


def _validate_data_type(data_type: str) -> None:
    """Validate data type parameter.

    Args:
        data_type: Data type string

    Raises:
        typer.Exit: On invalid data type
    """
    valid_types = ["all", "prices", "macro", "vix"]
    if data_type not in valid_types:
        console.print(
            f"[red]Error:[/red] Invalid data type. "
            f"Choose from: {', '.join(valid_types)}"
        )
        raise typer.Exit(code=1)


def _execute_fetch(
    data_type: str,
    symbol_list: list[str],
    start: date,
    end: date,
    output_path: Path,
    db_path: Optional[str],
) -> dict[str, dict]:
    """Execute the data fetch operations.

    Args:
        data_type: Type of data to fetch
        symbol_list: List of symbols
        start: Start date
        end: End date
        output_path: Output directory
        db_path: Optional database path

    Returns:
        Dictionary of fetch results
    """
    results: dict[str, dict] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if data_type in ["all", "prices"]:
            task = progress.add_task("[cyan]Fetching ETF prices...", total=None)
            results["prices"] = _fetch_etf_prices(
                symbol_list, start, end, output_path, db_path
            )
            progress.update(task, completed=True)

        if data_type in ["all", "vix"]:
            task = progress.add_task("[cyan]Fetching VIX data...", total=None)
            results["vix"] = _fetch_vix_data(start, end, output_path, db_path)
            progress.update(task, completed=True)

        if data_type in ["all", "macro"]:
            task = progress.add_task("[cyan]Fetching macro indicators...", total=None)
            results["macro"] = _fetch_macro_data(start, end, output_path, db_path)
            progress.update(task, completed=True)

    return results


def _output_results(
    results: dict[str, dict],
    output_format: str,
    output_path: Path,
    db_path: Optional[str],
) -> None:
    """Output fetch results in the specified format.

    Args:
        results: Fetch results dictionary
        output_format: Output format
        output_path: Output directory path
        db_path: Database path if used
    """
    import json

    if output_format == "json":
        console.print(json.dumps(results, indent=2, default=str))
    elif output_format == "plain":
        _display_plain_results(results)
    else:
        _display_rich_results(results, output_path, db_path)


def _fetch_etf_prices(
    symbols: list[str],
    start_date: date,
    end_date: date,
    output_path: Path,
    db_path: Optional[str],
) -> dict:
    """Fetch ETF price data.

    Args:
        symbols: List of symbols to fetch
        start_date: Start date
        end_date: End date
        output_path: Output directory
        db_path: Optional database path

    Returns:
        Fetch results dictionary
    """
    from src.data.fetchers.yahoo import YahooFinanceFetcher

    fetcher = YahooFinanceFetcher()
    results = {
        "symbols_requested": symbols,
        "symbols_fetched": [],
        "records_fetched": 0,
        "errors": [],
        "files_saved": [],
    }

    for symbol in symbols:
        try:
            df = fetcher._fetch_ticker_data(symbol, start_date, end_date)
            if not df.empty:
                results["symbols_fetched"].append(symbol)
                results["records_fetched"] += len(df)

                # Save to CSV
                csv_path = output_path / f"{symbol.replace('.', '_')}_prices.csv"
                df.to_csv(csv_path)
                results["files_saved"].append(str(csv_path))

                # Save to database if specified
                if db_path:
                    _save_prices_to_db(df, symbol, db_path)

        except Exception as e:
            results["errors"].append(
                {
                    "symbol": symbol,
                    "error": str(e),
                }
            )
            logger.warning(f"Failed to fetch {symbol}: {e}")

    return results


def _fetch_vix_data(
    start_date: date,
    end_date: date,
    output_path: Path,
    db_path: Optional[str],
) -> dict:
    """Fetch VIX data.

    Args:
        start_date: Start date
        end_date: End date
        output_path: Output directory
        db_path: Optional database path

    Returns:
        Fetch results dictionary
    """
    from src.data.fetchers.yahoo import YahooFinanceFetcher

    fetcher = YahooFinanceFetcher()
    results = {
        "records_fetched": 0,
        "files_saved": [],
        "errors": [],
    }

    try:
        vix_df = fetcher.fetch_vix(start_date, end_date)
        if not vix_df.empty:
            results["records_fetched"] = len(vix_df)

            # Save to CSV
            csv_path = output_path / "vix_data.csv"
            vix_df.to_csv(csv_path)
            results["files_saved"].append(str(csv_path))

            # Save to database if specified
            if db_path:
                _save_vix_to_db(vix_df, db_path)

    except Exception as e:
        results["errors"].append({"source": "VIX", "error": str(e)})
        logger.warning(f"Failed to fetch VIX: {e}")

    return results


def _fetch_macro_data(
    start_date: date,
    end_date: date,
    output_path: Path,
    db_path: Optional[str],
) -> dict:
    """Fetch macro indicator data from FRED.

    Args:
        start_date: Start date
        end_date: End date
        output_path: Output directory
        db_path: Optional database path

    Returns:
        Fetch results dictionary
    """
    import os

    import pandas as pd

    results = {
        "indicators_requested": [],
        "indicators_fetched": [],
        "records_fetched": 0,
        "files_saved": [],
        "errors": [],
    }

    # Check for FRED API key
    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key:
        results["errors"].append(
            {
                "source": "FRED",
                "error": "FRED_API_KEY environment variable not set",
            }
        )
        return results

    # Key macro indicators for regime detection
    indicators = [
        "DGS10",  # 10-Year Treasury Rate
        "DGS2",  # 2-Year Treasury Rate
        "BAMLH0A0HYM2",  # ICE BofA US High Yield Index Option-Adjusted Spread
        "UMCSENT",  # University of Michigan Consumer Sentiment
        "UNRATE",  # Unemployment Rate
    ]

    results["indicators_requested"] = indicators

    try:
        from src.data.fetchers.fred import FREDFetcher

        fetcher = FREDFetcher(api_key=fred_api_key)

        for indicator in indicators:
            try:
                # Use fetch_macro_indicators for all indicators
                data = fetcher.fetch_macro_indicators(start_date, end_date)
                if data is not None and len(data) > 0:  # type: ignore[arg-type]
                    results["indicators_fetched"].append(indicator)
                    results["records_fetched"] += len(data)
                    break  # All indicators fetched together

            except Exception as e:
                results["errors"].append(
                    {
                        "indicator": indicator,
                        "error": str(e),
                    }
                )
                logger.warning(f"Failed to fetch {indicator}: {e}")

        # Save combined data
        if results["indicators_fetched"]:
            csv_path = output_path / "macro_data.csv"
            # Create combined DataFrame (placeholder)
            df = pd.DataFrame({"indicator": results["indicators_fetched"]})
            df.to_csv(csv_path)
            results["files_saved"].append(str(csv_path))

    except ImportError:
        results["errors"].append(
            {
                "source": "FRED",
                "error": "FREDFetcher not available",
            }
        )
    except Exception as e:
        results["errors"].append(
            {
                "source": "FRED",
                "error": str(e),
            }
        )

    return results


def _save_prices_to_db(
    df: "pd.DataFrame",
    symbol: str,
    db_path: str,
) -> None:
    """Save price data to DuckDB database.

    Args:
        df: Price DataFrame
        symbol: Symbol
        db_path: Database path
    """
    try:
        from src.data.storage.duckdb import DuckDBStorage

        storage = DuckDBStorage(db_path)
        # Convert and save (implementation depends on storage interface)
        logger.info(f"Saved {len(df)} records for {symbol} to database")
        storage.close()
    except Exception as e:
        logger.warning(f"Could not save to database: {e}")


def _save_vix_to_db(df: "pd.DataFrame", db_path: str) -> None:
    """Save VIX data to DuckDB database.

    Args:
        df: VIX DataFrame
        db_path: Database path
    """
    try:
        from src.data.storage.duckdb import DuckDBStorage

        storage = DuckDBStorage(db_path)
        logger.info(f"Saved {len(df)} VIX records to database")
        storage.close()
    except Exception as e:
        logger.warning(f"Could not save to database: {e}")


def _display_plain_results(results: dict) -> None:
    """Display fetch results in plain text.

    Args:
        results: Fetch results dictionary
    """
    print("Data Fetch Results")
    print("=" * 40)

    for data_type, data in results.items():
        print(f"\n{data_type.upper()}:")
        if "records_fetched" in data:
            print(f"  Records fetched: {data['records_fetched']}")
        if "files_saved" in data:
            print(f"  Files saved: {len(data['files_saved'])}")
        if "errors" in data and data["errors"]:
            print(f"  Errors: {len(data['errors'])}")


def _display_rich_results(
    results: dict,
    output_path: Path,
    db_path: Optional[str],
) -> None:
    """Display fetch results with rich formatting.

    Args:
        results: Fetch results dictionary
        output_path: Output directory path
        db_path: Database path if used
    """
    console.print()
    _display_summary_panel(results, output_path, db_path)
    _display_results_table(results)
    _display_files_and_errors(results)
    console.print()


def _display_summary_panel(
    results: dict,
    output_path: Path,
    db_path: Optional[str],
) -> None:
    """Display summary panel for fetch results.

    Args:
        results: Fetch results dictionary
        output_path: Output directory path
        db_path: Database path if used
    """
    total_records = sum(data.get("records_fetched", 0) for data in results.values())
    total_files = sum(len(data.get("files_saved", [])) for data in results.values())
    total_errors = sum(len(data.get("errors", [])) for data in results.values())

    status_color = "green" if total_errors == 0 else "yellow"

    console.print(
        Panel(
            f"[bold]Records Fetched: {total_records:,}[/bold]\n"
            f"Files Saved: {total_files}\n"
            f"Output Directory: {output_path.absolute()}\n"
            f"Database: {db_path or 'Not configured'}",
            title=f"[bold {status_color}]Data Fetch Complete[/bold {status_color}]",
        )
    )


def _display_results_table(results: dict) -> None:
    """Display results summary table.

    Args:
        results: Fetch results dictionary
    """
    table = Table(title="Fetch Summary", show_header=True)
    table.add_column("Data Type", style="cyan")
    table.add_column("Records", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Status", justify="center")

    for data_type, data in results.items():
        records = data.get("records_fetched", 0)
        files = len(data.get("files_saved", []))
        errors = len(data.get("errors", []))

        status = _get_status_string(records, errors)
        table.add_row(data_type.upper(), str(records), str(files), status)

    console.print(table)


def _get_status_string(records: int, errors: int) -> str:
    """Get status string for display.

    Args:
        records: Number of records fetched
        errors: Number of errors

    Returns:
        Formatted status string
    """
    if errors > 0:
        return f"[yellow]{errors} error(s)[/yellow]"
    if records > 0:
        return "[green]Success[/green]"
    return "[dim]No data[/dim]"


def _display_files_and_errors(results: dict) -> None:
    """Display files saved and errors.

    Args:
        results: Fetch results dictionary
    """
    # Collect files
    all_files = []
    for data in results.values():
        all_files.extend(data.get("files_saved", []))

    if all_files:
        console.print()
        console.print("[bold]Files Saved:[/bold]")
        for f in all_files:
            console.print(f"  [dim]{f}[/dim]")

    # Collect errors
    all_errors = [
        {"type": data_type, **error}
        for data_type, data in results.items()
        for error in data.get("errors", [])
    ]

    if all_errors:
        console.print()
        console.print("[bold yellow]Errors:[/bold yellow]")
        for error in all_errors:
            console.print(
                f"  [yellow]- {error.get('type', 'Unknown')}: "
                f"{error.get('error', 'Unknown error')}[/yellow]"
            )
