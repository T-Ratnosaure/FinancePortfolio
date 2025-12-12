"""Status command for portfolio monitoring.

This command displays the current portfolio status including positions,
allocation vs target, drift analysis, and rebalancing recommendations.
"""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data.models import REBALANCE_THRESHOLD, Regime

console = Console()
logger = logging.getLogger(__name__)


def status(
    db_path: str = typer.Option(
        "data/portfolio.duckdb",
        "--database",
        "-d",
        help="Path to portfolio database.",
    ),
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to portfolio configuration file (JSON/YAML).",
    ),
    target_regime: Optional[str] = typer.Option(
        None,
        "--target-regime",
        help="Override target regime for allocation comparison.",
    ),
    output_format: str = typer.Option(
        "rich",
        "--format",
        "-f",
        help="Output format: 'rich' (default), 'json', or 'plain'.",
    ),
    show_trades: bool = typer.Option(
        False,
        "--show-trades",
        "-t",
        help="Show recent trade history.",
    ),
) -> None:
    """Display current portfolio status and allocation analysis.

    Shows current holdings, compares actual vs target allocation,
    calculates drift, and suggests rebalancing trades if needed.

    Example:
        pea-portfolio status
        pea-portfolio status --database my_portfolio.duckdb
        pea-portfolio status --target-regime risk_off --show-trades
    """
    import json

    try:
        # Check if database exists
        db_file = Path(db_path)

        # Parse target regime if provided
        regime = None
        if target_regime:
            try:
                regime = Regime(target_regime.lower())
            except ValueError:
                console.print(
                    f"[red]Error:[/red] Invalid regime. "
                    f"Choose from: {', '.join(r.value for r in Regime)}"
                )
                raise typer.Exit(code=1) from None

        with console.status("[bold blue]Loading portfolio data...[/bold blue]"):
            # Load portfolio data
            portfolio_data = _load_portfolio_data(db_file, config_file)

            # Get target allocation
            target_allocation = _get_target_allocation(regime)

            # Calculate drift
            drift_analysis = _calculate_drift(
                portfolio_data["weights"], target_allocation
            )

            # Check if rebalancing needed
            needs_rebalance = any(
                d["drift_pct"] > REBALANCE_THRESHOLD * 100 for d in drift_analysis
            )

        # Display results
        if output_format == "json":
            result = {
                "portfolio_value": portfolio_data["total_value"],
                "positions": portfolio_data["positions"],
                "weights": portfolio_data["weights"],
                "target_allocation": target_allocation,
                "drift_analysis": drift_analysis,
                "needs_rebalancing": needs_rebalance,
                "rebalance_threshold": REBALANCE_THRESHOLD,
            }
            console.print(json.dumps(result, indent=2, default=str))

        elif output_format == "plain":
            _display_plain_status(portfolio_data, target_allocation, drift_analysis)

        else:  # rich format
            _display_rich_status(
                portfolio_data,
                target_allocation,
                drift_analysis,
                needs_rebalance,
                show_trades,
            )

    except Exception as e:
        logger.exception("Failed to load portfolio status")
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e


def _load_portfolio_data(
    db_path: Path,
    config_file: Optional[str],
) -> dict:
    """Load portfolio data from database or config file.

    Args:
        db_path: Path to DuckDB database
        config_file: Optional config file path

    Returns:
        Portfolio data dictionary
    """
    import json

    # Try to load from database
    if db_path.exists():
        try:
            from src.portfolio.tracker import PortfolioTracker

            with PortfolioTracker(str(db_path)) as tracker:
                positions = tracker.get_current_positions()
                weights = tracker.get_portfolio_weights()
                total_value = float(tracker.get_portfolio_value())
                cash = tracker.get_cash_position()

                return {
                    "positions": [
                        {
                            "symbol": pos.symbol.value,
                            "shares": pos.shares,
                            "average_cost": pos.average_cost,
                            "current_price": pos.current_price,
                            "market_value": pos.market_value,
                            "unrealized_pnl": pos.unrealized_pnl,
                        }
                        for pos in positions.values()
                    ],
                    "weights": weights,
                    "total_value": total_value,
                    "cash": {
                        "amount": float(cash.amount),
                        "currency": cash.currency,
                    },
                }
        except Exception as e:
            logger.warning(f"Could not load from database: {e}")

    # Try to load from config file
    if config_file and Path(config_file).exists():
        with open(config_file) as f:
            return json.load(f)

    # Return demo portfolio
    console.print(
        "[yellow]Warning: No portfolio data found. Showing demo portfolio.[/yellow]"
    )
    return _get_demo_portfolio()


def _get_demo_portfolio() -> dict:
    """Get demo portfolio for demonstration purposes.

    Returns:
        Demo portfolio data
    """
    return {
        "positions": [
            {
                "symbol": "LQQ.PA",
                "shares": 10.0,
                "average_cost": 95.00,
                "current_price": 102.50,
                "market_value": 1025.00,
                "unrealized_pnl": 75.00,
            },
            {
                "symbol": "CL2.PA",
                "shares": 5.0,
                "average_cost": 180.00,
                "current_price": 195.00,
                "market_value": 975.00,
                "unrealized_pnl": 75.00,
            },
            {
                "symbol": "WPEA.PA",
                "shares": 25.0,
                "average_cost": 220.00,
                "current_price": 235.00,
                "market_value": 5875.00,
                "unrealized_pnl": 375.00,
            },
        ],
        "weights": {
            "LQQ.PA": 0.10,
            "CL2.PA": 0.10,
            "WPEA.PA": 0.59,
            "CASH": 0.21,
        },
        "total_value": 10000.00,
        "cash": {
            "amount": 2125.00,
            "currency": "EUR",
        },
    }


def _get_target_allocation(regime: Optional[Regime] = None) -> dict[str, float]:
    """Get target allocation for given regime.

    Args:
        regime: Market regime (uses NEUTRAL if not specified)

    Returns:
        Target allocation weights
    """
    from src.signals.allocation import REGIME_ALLOCATIONS

    if regime is None:
        regime = Regime.NEUTRAL

    base_allocation = REGIME_ALLOCATIONS[regime]

    # Convert to full symbol format
    return {
        "LQQ.PA": base_allocation["LQQ"],
        "CL2.PA": base_allocation["CL2"],
        "WPEA.PA": base_allocation["WPEA"],
        "CASH": base_allocation["CASH"],
    }


def _calculate_drift(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> list[dict]:
    """Calculate drift between current and target allocation.

    Args:
        current_weights: Current portfolio weights
        target_weights: Target allocation weights

    Returns:
        List of drift analysis per asset
    """
    all_assets = set(current_weights.keys()) | set(target_weights.keys())
    drift_analysis = []

    for asset in sorted(all_assets):
        current = current_weights.get(asset, 0.0)
        target = target_weights.get(asset, 0.0)
        drift = current - target

        drift_analysis.append(
            {
                "asset": asset,
                "current_weight": current,
                "target_weight": target,
                "drift": drift,
                "drift_pct": abs(drift) * 100,
                "action_needed": (
                    "REDUCE"
                    if drift > REBALANCE_THRESHOLD
                    else "INCREASE"
                    if drift < -REBALANCE_THRESHOLD
                    else "OK"
                ),
            }
        )

    return drift_analysis


def _display_plain_status(
    portfolio_data: dict,
    target_allocation: dict[str, float],
    drift_analysis: list[dict],
) -> None:
    """Display status in plain text format.

    Args:
        portfolio_data: Portfolio data
        target_allocation: Target allocation weights
        drift_analysis: Drift analysis results
    """
    print(f"Portfolio Value: {portfolio_data['total_value']:,.2f} EUR")
    print(f"Cash: {portfolio_data['cash']['amount']:,.2f} EUR")
    print()
    print("Positions:")
    for pos in portfolio_data["positions"]:
        print(f"  {pos['symbol']}: {pos['shares']} shares @ {pos['current_price']:.2f}")
    print()
    print("Drift Analysis:")
    for d in drift_analysis:
        print(
            f"  {d['asset']}: {d['current_weight']:.1%} vs {d['target_weight']:.1%} "
            f"(drift: {d['drift']:.1%}) - {d['action_needed']}"
        )


def _display_rich_status(
    portfolio_data: dict,
    target_allocation: dict[str, float],
    drift_analysis: list[dict],
    needs_rebalance: bool,
    show_trades: bool,
) -> None:
    """Display status with rich formatting.

    Args:
        portfolio_data: Portfolio data
        target_allocation: Target allocation weights
        drift_analysis: Drift analysis results
        needs_rebalance: Whether rebalancing is needed
        show_trades: Whether to show trade history
    """
    # Header panel
    console.print()
    total_value = portfolio_data["total_value"]
    cash_amount = portfolio_data["cash"]["amount"]

    rebalance_status = (
        "[yellow]REBALANCING RECOMMENDED[/yellow]"
        if needs_rebalance
        else "[green]BALANCED[/green]"
    )

    console.print(
        Panel(
            f"[bold]Total Portfolio Value: {total_value:,.2f} EUR[/bold]\n"
            f"Cash: {cash_amount:,.2f} EUR\n"
            f"Status: {rebalance_status}",
            title="[bold cyan]PEA Portfolio Status[/bold cyan]",
        )
    )

    # Positions table
    pos_table = Table(title="Current Positions", show_header=True)
    pos_table.add_column("Symbol", style="cyan")
    pos_table.add_column("Shares", justify="right")
    pos_table.add_column("Price", justify="right")
    pos_table.add_column("Value", justify="right")
    pos_table.add_column("P&L", justify="right")

    for pos in portfolio_data["positions"]:
        pnl = pos["unrealized_pnl"]
        pnl_color = "green" if pnl >= 0 else "red"
        pnl_sign = "+" if pnl >= 0 else ""

        pos_table.add_row(
            pos["symbol"],
            f"{pos['shares']:.2f}",
            f"{pos['current_price']:.2f}",
            f"{pos['market_value']:,.2f}",
            f"[{pnl_color}]{pnl_sign}{pnl:,.2f}[/{pnl_color}]",
        )

    console.print(pos_table)

    # Allocation comparison table
    alloc_table = Table(title="Allocation vs Target", show_header=True)
    alloc_table.add_column("Asset", style="cyan")
    alloc_table.add_column("Current", justify="right")
    alloc_table.add_column("Target", justify="right")
    alloc_table.add_column("Drift", justify="right")
    alloc_table.add_column("Action", justify="center")

    for d in drift_analysis:
        drift_color = "red" if d["action_needed"] != "OK" else "green"
        action_style = (
            "bold red"
            if d["action_needed"] == "REDUCE"
            else "bold yellow"
            if d["action_needed"] == "INCREASE"
            else "green"
        )

        alloc_table.add_row(
            d["asset"],
            f"{d['current_weight']:.1%}",
            f"{d['target_weight']:.1%}",
            f"[{drift_color}]{d['drift']:+.1%}[/{drift_color}]",
            f"[{action_style}]{d['action_needed']}[/{action_style}]",
        )

    console.print(alloc_table)

    # Rebalancing recommendation
    if needs_rebalance:
        console.print()
        console.print(
            Panel(
                "[yellow]Rebalancing is recommended.[/yellow]\n\n"
                "Run [cyan]pea-portfolio analyze[/cyan] to get specific "
                "trade recommendations based on current market regime.",
                title="[bold yellow]Rebalancing Alert[/bold yellow]",
                border_style="yellow",
            )
        )

    # Visual allocation bar
    console.print()
    console.print("[bold]Current Allocation:[/bold]")
    _display_allocation_bar(portfolio_data["weights"])

    console.print()
    console.print("[bold]Target Allocation:[/bold]")
    _display_allocation_bar(target_allocation)

    console.print()


def _display_allocation_bar(weights: dict[str, float]) -> None:
    """Display a visual allocation bar.

    Args:
        weights: Portfolio weights by asset
    """
    bar_width = 50
    colors = {
        "LQQ.PA": "red",
        "CL2.PA": "magenta",
        "WPEA.PA": "blue",
        "CASH": "green",
    }

    bar_parts: list[str] = []
    for asset, weight in weights.items():
        width = int(weight * bar_width)
        color = colors.get(asset, "white")
        if width > 0:
            bar_parts.append(f"[{color}]" + "|" * width + f"[/{color}]")

    console.print("  " + "".join(bar_parts))

    # Legend
    legend_parts: list[str] = []
    for asset, weight in weights.items():
        if weight > 0:
            color = colors.get(asset, "white")
            name = asset.replace(".PA", "")
            legend_parts.append(f"[{color}]{name}: {weight:.0%}[/{color}]")

    console.print("  " + "  ".join(legend_parts))
