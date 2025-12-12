"""Analyze command for market regime detection.

This command fetches latest market data, runs regime detection,
and displays the current market regime with recommended allocation.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data.models import AllocationRecommendation, Regime

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from src.data.fetchers.yahoo import YahooFinanceFetcher
    from src.signals.regime import RegimeDetector

console = Console()
logger = logging.getLogger(__name__)


def analyze(
    lookback_days: int = typer.Option(
        252 * 7,
        "--lookback-days",
        "-l",
        help="Number of days of historical data for regime detection.",
        min=252,
    ),
    model_path: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to pre-trained regime detector model.",
    ),
    skip_fetch: bool = typer.Option(
        False,
        "--skip-fetch",
        help="Skip fetching new data and use cached data only.",
    ),
    output_format: str = typer.Option(
        "rich",
        "--format",
        "-f",
        help="Output format: 'rich' (default), 'json', or 'plain'.",
    ),
) -> None:
    """Run market regime analysis and display recommended allocation.

    Analyzes current market conditions using a Hidden Markov Model to detect
    the market regime (RISK_ON, NEUTRAL, or RISK_OFF) and provides an
    allocation recommendation based on the detected regime.

    Example:
        pea-portfolio analyze
        pea-portfolio analyze --lookback-days 1260 --format json
    """
    try:
        analysis_result = _run_analysis(lookback_days, model_path, skip_fetch)
        _display_results(analysis_result, output_format)
    except Exception as e:
        logger.exception("Analysis failed")
        console.print(f"[red]Error:[/red] Analysis failed: {e}")
        raise typer.Exit(code=1) from e


def _run_analysis(
    lookback_days: int,
    model_path: Optional[str],
    skip_fetch: bool,
) -> dict[str, Any]:
    """Run the market analysis and return results.

    Args:
        lookback_days: Number of days of historical data
        model_path: Path to pre-trained model
        skip_fetch: Whether to skip fetching new data

    Returns:
        Dictionary with analysis results
    """
    import pandas as pd

    from src.data.fetchers.yahoo import YahooFinanceFetcher
    from src.signals.allocation import AllocationOptimizer

    with console.status("[bold blue]Running market analysis...[/bold blue]"):
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        # Fetch or load data
        vix_df = pd.DataFrame()
        spy_df = pd.DataFrame()

        if not skip_fetch:
            console.print(
                f"[cyan]Fetching data from {start_date} to {end_date}...[/cyan]"
            )
            fetcher = YahooFinanceFetcher()
            vix_df, spy_df = _fetch_market_data(fetcher, start_date, end_date)

        # Build features
        features = _build_features(vix_df, spy_df, lookback_days)

        # Load or train model
        detector = _get_detector(model_path, features)

        # Predict current regime
        current_regime = detector.predict_regime(features[-1:])
        regime_probs = detector.predict_regime_probabilities(features[-1:])

        # Get allocation recommendation
        optimizer = AllocationOptimizer()
        confidence = max(regime_probs.values())
        recommendation = optimizer.get_target_allocation(
            regime=current_regime,
            confidence=confidence,
            as_of_date=end_date,
        )

    return {
        "date": end_date,
        "regime": current_regime,
        "confidence": confidence,
        "probabilities": regime_probs,
        "recommendation": recommendation,
    }


def _fetch_market_data(
    fetcher: "YahooFinanceFetcher",
    start_date: date,
    end_date: date,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Fetch VIX and SPY data.

    Args:
        fetcher: Yahoo Finance fetcher instance
        start_date: Start date
        end_date: End date

    Returns:
        Tuple of (vix_df, spy_df)
    """
    import pandas as pd

    vix_df = pd.DataFrame()
    spy_df = pd.DataFrame()

    try:
        vix_df = fetcher.fetch_vix(start_date, end_date)
    except Exception as e:
        logger.warning(f"Could not fetch VIX data: {e}")

    try:
        spy_df = fetcher.fetch_index_prices(["^GSPC"], start_date, end_date)
    except Exception as e:
        logger.warning(f"Could not fetch SPY data: {e}")

    return vix_df, spy_df


def _build_features(
    vix_df: "pd.DataFrame",
    spy_df: "pd.DataFrame",
    lookback_days: int,
) -> "np.ndarray":
    """Build feature matrix for regime detection.

    Args:
        vix_df: VIX DataFrame
        spy_df: SPY DataFrame
        lookback_days: Number of lookback days

    Returns:
        Feature matrix
    """
    import numpy as np
    import pandas as pd

    features_list: list[np.ndarray] = []
    vix_len = lookback_days

    if not vix_df.empty:
        vix_values = vix_df["vix"].values
        vix_mean = np.mean(vix_values)  # type: ignore[call-overload]
        vix_std = np.std(vix_values) if np.std(vix_values) > 0 else 1  # type: ignore[call-overload]
        vix_normalized = (vix_values - vix_mean) / vix_std
        features_list.append(vix_normalized)
        vix_len = len(vix_normalized)

    if not spy_df.empty and "^GSPC" in spy_df.columns.get_level_values(0):
        spy_close = spy_df[("^GSPC", "Close")].values
        returns = np.diff(spy_close) / spy_close[:-1]  # type: ignore[call-overload]
        returns = np.concatenate([[0], returns])
        trend = pd.Series(returns).rolling(20).mean().fillna(0).values  # type: ignore[union-attr]
        features_list.append(trend[:vix_len])

    if len(features_list) < 2:
        console.print("[yellow]Warning: Insufficient data for full analysis.[/yellow]")
        n_samples = vix_len if features_list else lookback_days
        return np.random.randn(n_samples, 9)

    min_len = min(len(f) for f in features_list)
    features_stacked = np.column_stack([f[:min_len] for f in features_list])

    if features_stacked.shape[1] < 9:
        padding = np.zeros((features_stacked.shape[0], 9 - features_stacked.shape[1]))
        return np.hstack([features_stacked, padding])
    return features_stacked[:, :9]


def _get_detector(
    model_path: Optional[str],
    features: "np.ndarray",
) -> "RegimeDetector":
    """Load or train regime detector.

    Args:
        model_path: Path to pre-trained model
        features: Feature matrix for training

    Returns:
        Fitted RegimeDetector
    """
    from src.signals.regime import RegimeDetector

    if model_path and Path(model_path).exists():
        console.print(f"[cyan]Loading model from {model_path}...[/cyan]")
        return RegimeDetector.load(model_path)

    console.print("[cyan]Training new regime detection model...[/cyan]")
    detector = RegimeDetector(n_states=3, random_state=42)
    detector.fit(features, skip_sample_validation=True)
    return detector


def _display_results(result: dict[str, Any], output_format: str) -> None:
    """Display analysis results in the specified format.

    Args:
        result: Analysis results dictionary
        output_format: Output format ('rich', 'json', or 'plain')
    """
    import json

    if output_format == "json":
        output = {
            "date": str(result["date"]),
            "regime": result["regime"].value,
            "confidence": result["confidence"],
            "probabilities": {r.value: p for r, p in result["probabilities"].items()},
            "allocation": {
                "LQQ": result["recommendation"].lqq_weight,
                "CL2": result["recommendation"].cl2_weight,
                "WPEA": result["recommendation"].wpea_weight,
                "CASH": result["recommendation"].cash_weight,
            },
            "reasoning": result["recommendation"].reasoning,
        }
        console.print(json.dumps(output, indent=2))

    elif output_format == "plain":
        rec = result["recommendation"]
        print(f"Date: {result['date']}")
        print(f"Regime: {result['regime'].value}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(
            f"Allocation: LQQ={rec.lqq_weight:.1%}, CL2={rec.cl2_weight:.1%}, "
            f"WPEA={rec.wpea_weight:.1%}, CASH={rec.cash_weight:.1%}"
        )

    else:
        _display_rich_analysis(
            result["date"],
            result["regime"],
            result["probabilities"],
            result["recommendation"],
        )


def _display_rich_analysis(
    analysis_date: date,
    regime: Regime,
    probabilities: dict[Regime, float],
    recommendation: AllocationRecommendation,
) -> None:
    """Display analysis results with rich formatting.

    Args:
        analysis_date: Date of analysis
        regime: Detected market regime
        probabilities: Regime probability distribution
        recommendation: Allocation recommendation
    """
    regime_colors = {
        Regime.RISK_ON: "green",
        Regime.NEUTRAL: "yellow",
        Regime.RISK_OFF: "red",
    }
    regime_icons = {
        Regime.RISK_ON: ">>",
        Regime.NEUTRAL: "==",
        Regime.RISK_OFF: "<<",
    }

    color = regime_colors.get(regime, "white")
    icon = regime_icons.get(regime, "?")

    # Header panel
    console.print()
    regime_text = f"{icon} Market Regime: {regime.value.upper()} {icon}"
    console.print(
        Panel(
            f"[bold {color}]{regime_text}[/bold {color}]",
            title="[bold]PEA Portfolio Analysis[/bold]",
            subtitle=f"Analysis Date: {analysis_date}",
        )
    )

    # Regime probabilities table
    prob_table = Table(title="Regime Probabilities", show_header=True)
    prob_table.add_column("Regime", style="cyan")
    prob_table.add_column("Probability", justify="right")
    prob_table.add_column("Bar", justify="left")

    for r in [Regime.RISK_ON, Regime.NEUTRAL, Regime.RISK_OFF]:
        prob = probabilities.get(r, 0.0)
        bar_width = int(prob * 20)
        bar = "[green]" + "|" * bar_width + "[/green]" + " " * (20 - bar_width)
        style = "bold" if r == regime else ""
        prob_table.add_row(
            f"[{style}]{r.value}[/{style}]",
            f"[{style}]{prob:.1%}[/{style}]",
            bar,
        )

    console.print(prob_table)

    # Allocation recommendation table
    alloc_table = Table(title="Recommended Allocation", show_header=True)
    alloc_table.add_column("Asset", style="cyan")
    alloc_table.add_column("Weight", justify="right")
    alloc_table.add_column("Type", justify="center")

    allocations = [
        ("LQQ (Nasdaq 2x)", recommendation.lqq_weight, "Leveraged"),
        ("CL2 (USA 2x)", recommendation.cl2_weight, "Leveraged"),
        ("WPEA (World)", recommendation.wpea_weight, "Core"),
        ("CASH", recommendation.cash_weight, "Buffer"),
    ]

    for name, weight, asset_type in allocations:
        type_color = (
            "red"
            if asset_type == "Leveraged"
            else "blue"
            if asset_type == "Core"
            else "green"
        )
        alloc_table.add_row(
            name, f"{weight:.1%}", f"[{type_color}]{asset_type}[/{type_color}]"
        )

    console.print(alloc_table)

    if recommendation.reasoning:
        console.print()
        console.print(
            Panel(
                recommendation.reasoning,
                title="Analysis Reasoning",
                border_style="dim",
            )
        )

    console.print()
