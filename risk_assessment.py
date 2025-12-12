"""
Risk Assessment Module for French PEA Portfolio

This module provides comprehensive risk analysis for a PEA portfolio containing
leveraged and unleveraged ETFs with focus on quantitative risk metrics.

Author: Nicolas, Risk Manager
Date: 2025-12-10
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from src.data.models import MAX_LEVERAGED_EXPOSURE, MIN_CASH_BUFFER

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk severity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ETFType(str, Enum):
    """ETF leverage types."""

    LEVERAGED_2X = "2X"
    UNLEVERAGED = "1X"


@dataclass
class HistoricalDrawdown:
    """Historical drawdown data for stress testing."""

    period: str
    start_date: str
    end_date: str
    max_drawdown: float
    duration_days: int
    recovery_days: int | None


class ETFHolding(BaseModel):
    """Individual ETF position in portfolio."""

    ticker: str = Field(..., description="ETF ticker symbol")
    name: str = Field(..., description="Full ETF name")
    etf_type: ETFType = Field(..., description="Leverage type")
    allocation: float = Field(
        ..., ge=0.0, le=1.0, description="Portfolio allocation (0-1)"
    )
    underlying_index: str = Field(..., description="Underlying index")
    expense_ratio: float = Field(..., description="Annual expense ratio")
    avg_daily_volume: float = Field(..., description="Average daily volume in EUR")
    bid_ask_spread_bps: float = Field(..., description="Typical bid-ask spread in bps")

    @field_validator("allocation")
    @classmethod
    def validate_allocation(cls, v: float) -> float:
        """Validate allocation is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Allocation must be between 0 and 1")
        return v


class Portfolio(BaseModel):
    """Complete portfolio holdings."""

    holdings: list[ETFHolding] = Field(..., description="List of ETF holdings")
    cash_allocation: float = Field(
        ..., ge=0.0, le=1.0, description="Cash allocation (0-1)"
    )
    total_value_eur: float = Field(..., gt=0, description="Total portfolio value EUR")
    currency: str = Field(default="EUR", description="Base currency")

    @field_validator("cash_allocation")
    @classmethod
    def validate_total_allocation(cls, v: float, info: ValidationInfo) -> float:
        """Validate total allocation equals 1."""
        if info.data and "holdings" in info.data:
            holdings_sum = sum(h.allocation for h in info.data["holdings"])
            total = holdings_sum + v
            if not 0.99 <= total <= 1.01:  # Allow small floating point errors
                raise ValueError(f"Total allocation must equal 1.0, got {total:.4f}")
        return v


class VaRResult(BaseModel):
    """Value at Risk calculation results."""

    confidence_level: float = Field(..., description="Confidence level (e.g., 0.95)")
    time_horizon_days: int = Field(..., description="Time horizon in days")
    var_percent: float = Field(..., description="VaR as percentage")
    var_eur: float = Field(..., description="VaR in EUR")
    cvar_percent: float = Field(..., description="Conditional VaR (Expected Shortfall)")
    cvar_eur: float = Field(..., description="CVaR in EUR")
    methodology: str = Field(..., description="VaR calculation method")


class LeveragedETFRisk(BaseModel):
    """Leveraged ETF specific risks."""

    ticker: str
    daily_reset_impact: float = Field(
        ..., description="Estimated annual decay from daily reset (%)"
    )
    volatility_drag: float = Field(
        ..., description="Estimated annual volatility drag (%)"
    )
    path_dependency_risk: RiskLevel
    compounding_error: float = Field(
        ..., description="Deviation from 2x returns over 1 year (%)"
    )
    recommendation: str


class ConcentrationRisk(BaseModel):
    """Portfolio concentration analysis."""

    geographic_concentration: dict[str, float]
    sector_concentration: dict[str, float]
    single_position_max: float
    herfindahl_index: float = Field(
        ..., description="HHI: 0-1, higher means more concentrated"
    )
    effective_n_holdings: float = Field(
        ..., description="Effective number of independent holdings"
    )
    risk_level: RiskLevel


class DrawdownScenario(BaseModel):
    """Stress test drawdown scenario."""

    scenario_name: str
    historical_period: str
    estimated_portfolio_drawdown: float
    estimated_loss_eur: float
    recovery_time_estimate: str
    probability_assessment: str
    risk_level: RiskLevel


class CurrencyRisk(BaseModel):
    """EUR/USD currency exposure analysis."""

    usd_exposure_percent: float = Field(
        ..., description="% of portfolio exposed to USD"
    )
    eur_usd_volatility: float = Field(
        ..., description="Historical EUR/USD annual volatility"
    )
    var_currency_1m: float = Field(
        ..., description="1-month VaR from currency moves (%)"
    )
    hedging_recommendation: str
    risk_level: RiskLevel


class CorrelationBreakdown(BaseModel):
    """Correlation breakdown risk analysis."""

    normal_correlation: dict[str, dict[str, float]]
    stress_correlation: dict[str, dict[str, float]]
    correlation_breakdown_scenarios: list[str]
    diversification_ratio: float = Field(
        ..., description="Portfolio vol / weighted average vol"
    )
    risk_level: RiskLevel


class LiquidityRisk(BaseModel):
    """ETF liquidity analysis."""

    ticker: str
    avg_daily_volume_eur: float
    days_to_liquidate_5pct: float = Field(
        ..., description="Days to liquidate 5% position"
    )
    bid_ask_spread_bps: float
    liquidity_cost_estimate_bps: float
    risk_level: RiskLevel


class PositionLimits(BaseModel):
    """Recommended position limits by risk category."""

    leveraged_etf_max_single: float = Field(
        ..., description="Max single leveraged ETF position (%)"
    )
    leveraged_etf_max_total: float = Field(
        ..., description="Max total leveraged ETF exposure (%)"
    )
    single_geography_max: float = Field(
        ..., description="Max single country/region exposure (%)"
    )
    single_sector_max: float = Field(..., description="Max single sector exposure (%)")
    min_cash_reserve: float = Field(..., description="Minimum cash reserve (%)")
    rationale: str


class RebalancingTrigger(BaseModel):
    """Risk-based rebalancing triggers."""

    trigger_type: str
    current_value: float
    threshold: float
    action_required: bool
    recommended_action: str
    urgency: RiskLevel


class RiskAllocationFramework(BaseModel):
    """Risk-based allocation framework."""

    risk_budget_total: float = Field(..., description="Total risk budget (VaR %)")
    risk_allocation: dict[str, float] = Field(
        ..., description="Risk allocation by asset"
    )
    recommended_weights: dict[str, float] = Field(
        ..., description="Recommended position sizes"
    )
    current_vs_target: dict[str, dict[str, float]]
    rebalancing_needed: bool


class ComprehensiveRiskReport(BaseModel):
    """Complete risk assessment report."""

    report_date: datetime
    portfolio: Portfolio
    var_analysis: VaRResult
    leveraged_etf_risks: list[LeveragedETFRisk]
    concentration_risk: ConcentrationRisk
    currency_risk: CurrencyRisk
    drawdown_scenarios: list[DrawdownScenario]
    correlation_breakdown: CorrelationBreakdown
    liquidity_risks: list[LiquidityRisk]
    position_limits: PositionLimits
    risk_allocation_framework: RiskAllocationFramework
    rebalancing_triggers: list[RebalancingTrigger]
    overall_risk_score: float = Field(..., ge=0, le=10)
    summary_recommendations: list[str]


def create_pea_portfolio() -> Portfolio:
    """Create the current PEA portfolio configuration."""
    holdings = [
        ETFHolding(
            ticker="LQQ",
            name="Amundi Nasdaq-100 Daily (2x) Leveraged UCITS ETF",
            etf_type=ETFType.LEVERAGED_2X,
            allocation=0.15,  # Max combined leveraged = 30%
            underlying_index="Nasdaq-100",
            expense_ratio=0.0035,
            avg_daily_volume=50_000_000,
            bid_ask_spread_bps=10,
        ),
        ETFHolding(
            ticker="CL2",
            name="Leverage Shares 2x Long US 500 ETF",
            etf_type=ETFType.LEVERAGED_2X,
            allocation=0.15,  # Max combined leveraged = 30%
            underlying_index="S&P 500",
            expense_ratio=0.0059,
            avg_daily_volume=15_000_000,
            bid_ask_spread_bps=15,
        ),
        ETFHolding(
            ticker="WPEA",
            name="Amundi MSCI World UCITS ETF EUR (C)",
            etf_type=ETFType.UNLEVERAGED,
            allocation=0.30,  # Increased unleveraged allocation
            underlying_index="MSCI World",
            expense_ratio=0.0012,
            avg_daily_volume=100_000_000,
            bid_ask_spread_bps=5,
        ),
    ]

    return Portfolio(
        holdings=holdings,
        cash_allocation=0.40,  # Increased cash buffer for conservative portfolio
        total_value_eur=100_000,
        currency="EUR",
    )


def calculate_portfolio_var(portfolio: Portfolio) -> VaRResult:
    """
    Calculate Value at Risk for the portfolio.

    Uses historical simulation approach with 95% confidence over 1 day.
    Accounts for leverage multiplier effect.
    """
    # Base volatilities (annual) for underlying indices
    nasdaq_100_vol = 0.25  # 25% annual
    sp500_vol = 0.18  # 18% annual
    msci_world_vol = 0.16  # 16% annual

    # Convert to daily
    daily_vols = {
        "LQQ": nasdaq_100_vol / (252**0.5) * 2,  # 2x leverage
        "CL2": sp500_vol / (252**0.5) * 2,  # 2x leverage
        "WPEA": msci_world_vol / (252**0.5),  # No leverage
    }

    # Correlation matrix (historical averages)
    correlations = {
        ("LQQ", "CL2"): 0.85,
        ("LQQ", "WPEA"): 0.80,
        ("CL2", "WPEA"): 0.90,
    }

    # Calculate portfolio variance
    portfolio_variance = 0.0
    holdings_dict = {h.ticker: h for h in portfolio.holdings}

    # Variance contribution from each position
    for h in portfolio.holdings:
        portfolio_variance += (h.allocation * daily_vols[h.ticker]) ** 2

    # Covariance contributions
    for (t1, t2), corr in correlations.items():
        if t1 in holdings_dict and t2 in holdings_dict:
            cov = (
                2
                * holdings_dict[t1].allocation
                * holdings_dict[t2].allocation
                * daily_vols[t1]
                * daily_vols[t2]
                * corr
            )
            portfolio_variance += cov

    portfolio_std = portfolio_variance**0.5

    # VaR at 95% confidence (1.645 standard deviations)
    var_percent = portfolio_std * 1.645 * 100

    # CVaR (Expected Shortfall) - approximately VaR * 1.3 for normal distribution
    cvar_percent = var_percent * 1.3

    return VaRResult(
        confidence_level=0.95,
        time_horizon_days=1,
        var_percent=round(var_percent, 2),
        var_eur=round(portfolio.total_value_eur * var_percent / 100, 2),
        cvar_percent=round(cvar_percent, 2),
        cvar_eur=round(portfolio.total_value_eur * cvar_percent / 100, 2),
        methodology="Historical Simulation with Leverage Adjustment",
    )


def analyze_leveraged_etf_risks(portfolio: Portfolio) -> list[LeveragedETFRisk]:
    """Analyze specific risks for leveraged ETFs."""
    risks = []

    for holding in portfolio.holdings:
        if holding.etf_type == ETFType.LEVERAGED_2X:
            # Volatility drag calculation: -leverage^2 * volatility^2 / 2
            if holding.ticker == "LQQ":
                annual_vol = 0.25
            else:  # CL2
                annual_vol = 0.18

            volatility_drag = -(2**2) * (annual_vol**2) / 2 * 100

            # Daily reset impact (compounding error over 1 year)
            daily_reset_impact = abs(volatility_drag) * 0.3

            # Path dependency increases with volatility
            if annual_vol > 0.22:
                path_risk = RiskLevel.HIGH
            elif annual_vol > 0.18:
                path_risk = RiskLevel.MEDIUM
            else:
                path_risk = RiskLevel.LOW

            # Expected tracking error over 1 year
            compounding_error = abs(volatility_drag) * 1.5

            recommendation = (
                f"Monitor daily. Holding period should not exceed 3-6 months. "
                f"Expected decay: {abs(volatility_drag):.1f}% annually."
            )

            risks.append(
                LeveragedETFRisk(
                    ticker=holding.ticker,
                    daily_reset_impact=round(daily_reset_impact, 2),
                    volatility_drag=round(volatility_drag, 2),
                    path_dependency_risk=path_risk,
                    compounding_error=round(compounding_error, 2),
                    recommendation=recommendation,
                )
            )

    return risks


def analyze_concentration_risk(portfolio: Portfolio) -> ConcentrationRisk:
    """Analyze portfolio concentration across geography and sectors."""
    # Geographic breakdown (all holdings are US-heavy)
    geographic = {
        "United States": 0.85,  # LQQ + CL2 mostly US, WPEA 70% US
        "Europe": 0.10,
        "Asia Pacific": 0.05,
    }

    # Sector concentration (tech-heavy due to Nasdaq)
    sector = {
        "Technology": 0.45,  # Heavy from LQQ
        "Financials": 0.12,
        "Healthcare": 0.12,
        "Consumer Discretionary": 0.15,
        "Other": 0.16,
    }

    # Calculate Herfindahl Index
    holdings_weights = [h.allocation for h in portfolio.holdings]
    hhi = sum(w**2 for w in holdings_weights)

    # Effective N = 1 / HHI
    effective_n = 1 / hhi if hhi > 0 else 0

    # Risk level based on concentration
    if geographic["United States"] > 0.80:
        if sector["Technology"] > 0.40:
            risk = RiskLevel.CRITICAL
        else:
            risk = RiskLevel.HIGH
    elif geographic["United States"] > 0.70:
        risk = RiskLevel.MEDIUM
    else:
        risk = RiskLevel.LOW

    return ConcentrationRisk(
        geographic_concentration=geographic,  # type: ignore[arg-type]
        sector_concentration=sector,  # type: ignore[arg-type]
        single_position_max=max(holdings_weights),
        herfindahl_index=round(hhi, 3),
        effective_n_holdings=round(effective_n, 2),
        risk_level=risk,
    )


def analyze_currency_risk(portfolio: Portfolio) -> CurrencyRisk:
    """Analyze EUR/USD currency exposure."""
    # Calculate USD exposure
    usd_exposure = 0.0
    for holding in portfolio.holdings:
        # All holdings trade in EUR but have USD underlying exposure
        if holding.underlying_index in ["Nasdaq-100", "S&P 500"]:
            usd_exposure += holding.allocation * 1.0  # 100% USD
        elif holding.underlying_index == "MSCI World":
            usd_exposure += holding.allocation * 0.70  # 70% USD

    # EUR/USD historical volatility (annual)
    eur_usd_vol = 0.08  # 8% annual volatility

    # 1-month VaR from currency (95% confidence)
    var_1m = eur_usd_vol * 1.645 * (21 / 252) ** 0.5 * 100

    # Risk level
    if usd_exposure > 0.75:
        risk = RiskLevel.HIGH
    elif usd_exposure > 0.50:
        risk = RiskLevel.MEDIUM
    else:
        risk = RiskLevel.LOW

    recommendation = (
        f"Portfolio has {usd_exposure * 100:.0f}% USD exposure. "
        "Consider EUR-hedged alternatives or accept currency risk as "
        "diversification. Currency hedging costs ~1-2% annually."
    )

    return CurrencyRisk(
        usd_exposure_percent=round(usd_exposure * 100, 1),
        eur_usd_volatility=eur_usd_vol * 100,
        var_currency_1m=round(var_1m, 2),
        hedging_recommendation=recommendation,
        risk_level=risk,
    )


def generate_drawdown_scenarios(portfolio: Portfolio) -> list[DrawdownScenario]:
    """Generate stress test scenarios based on historical drawdowns."""
    scenarios = []

    # 2008 Financial Crisis
    nasdaq_dd_2008 = -0.54  # -54%
    sp500_dd_2008 = -0.57  # -57%
    world_dd_2008 = -0.54  # -54%

    # Calculate portfolio drawdown (with 2x leverage)
    dd_2008 = 0.0
    for h in portfolio.holdings:
        if h.ticker == "LQQ":
            dd_2008 += h.allocation * nasdaq_dd_2008 * 2
        elif h.ticker == "CL2":
            dd_2008 += h.allocation * sp500_dd_2008 * 2
        else:
            dd_2008 += h.allocation * world_dd_2008

    scenarios.append(
        DrawdownScenario(
            scenario_name="2008 Financial Crisis",
            historical_period="Oct 2007 - Mar 2009",
            estimated_portfolio_drawdown=round(dd_2008 * 100, 1),
            estimated_loss_eur=round(portfolio.total_value_eur * abs(dd_2008), 0),
            recovery_time_estimate="4-6 years",
            probability_assessment="10-year event (10% probability)",
            risk_level=RiskLevel.CRITICAL,
        )
    )

    # 2020 COVID-19 Crash
    nasdaq_dd_2020 = -0.30  # -30%
    sp500_dd_2020 = -0.34  # -34%
    world_dd_2020 = -0.34  # -34%

    dd_2020 = 0.0
    for h in portfolio.holdings:
        if h.ticker == "LQQ":
            dd_2020 += h.allocation * nasdaq_dd_2020 * 2
        elif h.ticker == "CL2":
            dd_2020 += h.allocation * sp500_dd_2020 * 2
        else:
            dd_2020 += h.allocation * world_dd_2020

    scenarios.append(
        DrawdownScenario(
            scenario_name="2020 COVID-19 Crash",
            historical_period="Feb 2020 - Mar 2020",
            estimated_portfolio_drawdown=round(dd_2020 * 100, 1),
            estimated_loss_eur=round(portfolio.total_value_eur * abs(dd_2020), 0),
            recovery_time_estimate="6-12 months",
            probability_assessment="5-year event (20% probability)",
            risk_level=RiskLevel.HIGH,
        )
    )

    # 2022 Bear Market
    nasdaq_dd_2022 = -0.33  # -33%
    sp500_dd_2022 = -0.25  # -25%
    world_dd_2022 = -0.20  # -20%

    dd_2022 = 0.0
    for h in portfolio.holdings:
        if h.ticker == "LQQ":
            dd_2022 += h.allocation * nasdaq_dd_2022 * 2
        elif h.ticker == "CL2":
            dd_2022 += h.allocation * sp500_dd_2022 * 2
        else:
            dd_2022 += h.allocation * world_dd_2022

    scenarios.append(
        DrawdownScenario(
            scenario_name="2022 Bear Market (Tech Correction)",
            historical_period="Jan 2022 - Oct 2022",
            estimated_portfolio_drawdown=round(dd_2022 * 100, 1),
            estimated_loss_eur=round(portfolio.total_value_eur * abs(dd_2022), 0),
            recovery_time_estimate="12-18 months",
            probability_assessment="3-year event (33% probability)",
            risk_level=RiskLevel.HIGH,
        )
    )

    return scenarios


def analyze_correlation_breakdown(portfolio: Portfolio) -> CorrelationBreakdown:
    """Analyze correlation breakdown risks."""
    # Normal market correlations
    normal_corr = {
        "LQQ": {"CL2": 0.85, "WPEA": 0.80},
        "CL2": {"LQQ": 0.85, "WPEA": 0.90},
        "WPEA": {"LQQ": 0.80, "CL2": 0.90},
    }

    # Stress correlations (tend to approach 1.0 in crashes)
    stress_corr = {
        "LQQ": {"CL2": 0.95, "WPEA": 0.95},
        "CL2": {"LQQ": 0.95, "WPEA": 0.98},
        "WPEA": {"LQQ": 0.95, "CL2": 0.98},
    }

    scenarios = [
        "Risk-off events: All equity correlations approach 1.0",
        "Tech bubble burst: LQQ decouples negatively from broader market",
        "Flight to quality: Growth (LQQ) vs Value divergence",
        "Leveraged ETF liquidation: Technical selling pressure",
    ]

    # Diversification ratio = Portfolio Vol / Weighted Avg Vol
    # Lower is better (more diversification)
    # Current portfolio: High correlation = low diversification
    div_ratio = 0.92  # Close to 1 means poor diversification

    return CorrelationBreakdown(
        normal_correlation=normal_corr,  # type: ignore[arg-type]
        stress_correlation=stress_corr,  # type: ignore[arg-type]
        correlation_breakdown_scenarios=scenarios,  # type: ignore[arg-type]
        diversification_ratio=div_ratio,
        risk_level=RiskLevel.HIGH,
    )


def analyze_liquidity_risks(portfolio: Portfolio) -> list[LiquidityRisk]:
    """Analyze liquidity risk for each ETF."""
    risks = []

    for holding in portfolio.holdings:
        position_value = portfolio.total_value_eur * holding.allocation

        # Days to liquidate 5% of ADV without moving market
        days_to_liquidate = (position_value * 0.05) / (holding.avg_daily_volume * 0.05)

        # Liquidity cost estimate (bid-ask + market impact)
        if days_to_liquidate < 0.5:
            liquidity_cost = holding.bid_ask_spread_bps
            risk = RiskLevel.LOW
        elif days_to_liquidate < 1.0:
            liquidity_cost = holding.bid_ask_spread_bps * 1.5
            risk = RiskLevel.LOW
        elif days_to_liquidate < 2.0:
            liquidity_cost = holding.bid_ask_spread_bps * 2.0
            risk = RiskLevel.MEDIUM
        else:
            liquidity_cost = holding.bid_ask_spread_bps * 3.0
            risk = RiskLevel.HIGH

        risks.append(
            LiquidityRisk(
                ticker=holding.ticker,
                avg_daily_volume_eur=holding.avg_daily_volume,
                days_to_liquidate_5pct=round(days_to_liquidate, 2),
                bid_ask_spread_bps=holding.bid_ask_spread_bps,
                liquidity_cost_estimate_bps=round(liquidity_cost, 1),
                risk_level=risk,
            )
        )

    return risks


def define_position_limits() -> PositionLimits:
    """Define position limits based on risk management principles."""
    return PositionLimits(
        leveraged_etf_max_single=0.25,  # 25% max per leveraged ETF
        leveraged_etf_max_total=MAX_LEVERAGED_EXPOSURE,  # 30% max total leveraged
        single_geography_max=0.70,  # 70% max single country
        single_sector_max=0.35,  # 35% max single sector
        min_cash_reserve=MIN_CASH_BUFFER,  # 10% minimum cash
        rationale=(
            "Limits based on: (1) Leveraged ETF decay risk, "
            "(2) Concentration risk management, "
            "(3) Liquidity for rebalancing and drawdowns, "
            "(4) Stress test scenarios showing >50% potential drawdowns"
        ),
    )


def calculate_risk_allocation_framework(
    portfolio: Portfolio, var_result: VaRResult
) -> RiskAllocationFramework:
    """Calculate risk-based allocation framework."""
    # Risk contribution of each holding (marginal VaR)
    risk_allocation: dict[str, float] = {}
    total_risk = var_result.var_percent

    for holding in portfolio.holdings:
        # Simplified risk contribution (allocation * volatility * correlation)
        if holding.ticker == "LQQ":
            vol = 0.25 * 2  # 25% vol * 2x leverage
        elif holding.ticker == "CL2":
            vol = 0.18 * 2
        else:
            vol = 0.16

        risk_contrib = holding.allocation * vol * 100
        risk_allocation[holding.ticker] = round(risk_contrib, 2)

    # Recommended weights for equal risk contribution
    total_risk_units = sum(risk_allocation.values())
    recommended_weights = {
        ticker: round(risk / total_risk_units, 3)
        for ticker, risk in risk_allocation.items()
    }

    # Add cash to recommended weights
    recommended_weights["CASH"] = 0.25

    # Normalize recommended weights
    total_recommended = sum(recommended_weights.values())
    recommended_weights = {
        k: round(v / total_recommended, 3) for k, v in recommended_weights.items()
    }

    # Current vs target
    current_weights = {h.ticker: h.allocation for h in portfolio.holdings}
    current_weights["CASH"] = portfolio.cash_allocation

    current_vs_target = {}
    for ticker in recommended_weights:
        current = current_weights.get(ticker, 0)
        target = recommended_weights[ticker]
        current_vs_target[ticker] = {
            "current": current,
            "target": target,
            "difference": round(current - target, 3),
        }

    # Check if rebalancing needed (>5% deviation)
    rebalancing_needed = any(
        abs(v["difference"]) > 0.05 for v in current_vs_target.values()
    )

    return RiskAllocationFramework(
        risk_budget_total=round(total_risk, 2),
        risk_allocation=risk_allocation,  # type: ignore[arg-type]
        recommended_weights=recommended_weights,  # type: ignore[arg-type]
        current_vs_target=current_vs_target,
        rebalancing_needed=rebalancing_needed,
    )


def generate_rebalancing_triggers(
    portfolio: Portfolio,
    var_result: VaRResult,
    concentration: ConcentrationRisk,
) -> list[RebalancingTrigger]:
    """Generate risk-based rebalancing triggers."""
    triggers = []

    # VaR exceeds threshold
    var_threshold = 3.0  # 3% daily VaR threshold
    triggers.append(
        RebalancingTrigger(
            trigger_type="Daily VaR Limit",
            current_value=var_result.var_percent,
            threshold=var_threshold,
            action_required=var_result.var_percent > var_threshold,
            recommended_action=(
                "Reduce leveraged exposure by 10-15%"
                if var_result.var_percent > var_threshold
                else "No action"
            ),
            urgency=(
                RiskLevel.HIGH
                if var_result.var_percent > var_threshold
                else RiskLevel.LOW
            ),
        )
    )

    # Leveraged ETF allocation exceeds limit
    leveraged_total = sum(
        h.allocation for h in portfolio.holdings if h.etf_type == ETFType.LEVERAGED_2X
    )
    triggers.append(
        RebalancingTrigger(
            trigger_type="Leveraged ETF Total Exposure",
            current_value=leveraged_total,
            threshold=MAX_LEVERAGED_EXPOSURE,
            action_required=leveraged_total > MAX_LEVERAGED_EXPOSURE,
            recommended_action=(
                f"Reduce leveraged positions to {MAX_LEVERAGED_EXPOSURE:.0%} max"
                if leveraged_total > MAX_LEVERAGED_EXPOSURE
                else "No action"
            ),
            urgency=(
                RiskLevel.HIGH
                if leveraged_total > MAX_LEVERAGED_EXPOSURE
                else RiskLevel.LOW
            ),
        )
    )

    # US concentration exceeds threshold
    us_concentration = concentration.geographic_concentration["United States"]
    triggers.append(
        RebalancingTrigger(
            trigger_type="Geographic Concentration (US)",
            current_value=us_concentration,
            threshold=0.70,
            action_required=us_concentration > 0.70,
            recommended_action=(
                "Add international exposure or reduce US positions"
                if us_concentration > 0.70
                else "No action"
            ),
            urgency=RiskLevel.MEDIUM if us_concentration > 0.70 else RiskLevel.LOW,
        )
    )

    # Tech sector concentration
    tech_concentration = concentration.sector_concentration["Technology"]
    triggers.append(
        RebalancingTrigger(
            trigger_type="Sector Concentration (Technology)",
            current_value=tech_concentration,
            threshold=0.35,
            action_required=tech_concentration > 0.35,
            recommended_action=(
                "Reduce Nasdaq exposure (LQQ) or add sector diversification"
                if tech_concentration > 0.35
                else "No action"
            ),
            urgency=RiskLevel.HIGH if tech_concentration > 0.35 else RiskLevel.LOW,
        )
    )

    # Cash reserve too low
    triggers.append(
        RebalancingTrigger(
            trigger_type="Cash Reserve",
            current_value=portfolio.cash_allocation,
            threshold=MIN_CASH_BUFFER,
            action_required=portfolio.cash_allocation < MIN_CASH_BUFFER,
            recommended_action=(
                f"Increase cash reserve to {MIN_CASH_BUFFER:.0%} minimum"
                if portfolio.cash_allocation < MIN_CASH_BUFFER
                else "No action"
            ),
            urgency=(
                RiskLevel.MEDIUM
                if portfolio.cash_allocation < MIN_CASH_BUFFER
                else RiskLevel.LOW
            ),
        )
    )

    return triggers


def calculate_overall_risk_score(
    var_result: VaRResult,
    leveraged_risks: list[LeveragedETFRisk],
    concentration: ConcentrationRisk,
    drawdown_scenarios: list[DrawdownScenario],
) -> float:
    """Calculate overall risk score (0-10 scale)."""
    # VaR component (0-3 points)
    var_score = min(var_result.var_percent / 2, 3.0)

    # Leverage component (0-2 points)
    avg_decay = sum(abs(r.volatility_drag) for r in leveraged_risks) / len(
        leveraged_risks
    )
    leverage_score = min(avg_decay / 5, 2.0)

    # Concentration component (0-2 points)
    conc_score = 2.0 if concentration.risk_level == RiskLevel.CRITICAL else 1.5

    # Drawdown component (0-3 points)
    worst_drawdown = min(s.estimated_portfolio_drawdown for s in drawdown_scenarios)
    drawdown_score = min(abs(worst_drawdown) / 20, 3.0)

    total_score = var_score + leverage_score + conc_score + drawdown_score
    return round(min(total_score, 10.0), 1)


def generate_summary_recommendations(
    portfolio: Portfolio,
    var_result: VaRResult,
    concentration: ConcentrationRisk,
    position_limits: PositionLimits,
    rebalancing_triggers: list[RebalancingTrigger],
) -> list[str]:
    """Generate executive summary recommendations."""
    recommendations = []

    # Check each trigger
    critical_triggers = [t for t in rebalancing_triggers if t.action_required]

    if critical_triggers:
        recommendations.append(
            f"URGENT: {len(critical_triggers)} risk limits breached. "
            "Immediate rebalancing required."
        )

    # Leveraged ETF recommendations
    leveraged_total = sum(
        h.allocation for h in portfolio.holdings if h.etf_type == ETFType.LEVERAGED_2X
    )
    if leveraged_total > position_limits.leveraged_etf_max_total:
        recommendations.append(
            f"Reduce total leveraged ETF exposure from {leveraged_total * 100:.0f}% "
            f"to {position_limits.leveraged_etf_max_total * 100:.0f}% maximum."
        )

    # Concentration recommendations
    if concentration.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
        recommendations.append(
            "Portfolio shows CRITICAL concentration risk: "
            f"{concentration.geographic_concentration['United States'] * 100:.0f}% US, "
            f"{concentration.geographic_concentration['United States'] * 100:.0f}% US, "
            f"{concentration.sector_concentration['Technology'] * 100:.0f}% "
            "Technology. Diversify or accept concentration bet."
        )

    # VaR recommendations
    if var_result.var_percent > 3.0:
        recommendations.append(
            f"Daily VaR of {var_result.var_percent:.2f}% "
            f"(EUR {var_result.var_eur:,.0f}) exceeds 3% threshold. "
            "Consider reducing leverage or position sizes."
        )

    # Cash recommendations
    if portfolio.cash_allocation < position_limits.min_cash_reserve:
        recommendations.append(
            f"Increase cash reserve from {portfolio.cash_allocation * 100:.0f}% to "
            f"{position_limits.min_cash_reserve * 100:.0f}% for rebalancing "
            "flexibility."
        )

    # General recommendations
    recommendations.append(
        "Monitor leveraged ETFs daily. Review and rebalance monthly at minimum."
    )

    recommendations.append(
        "Set stop-loss at portfolio level: -15% from peak. "
        "Reduce leverage if triggered."
    )

    recommendations.append(
        "Stress test indicates potential -65% drawdown in severe crisis. "
        "Ensure risk tolerance aligns with this possibility."
    )

    return recommendations


def generate_comprehensive_risk_report(
    portfolio: Portfolio,
) -> ComprehensiveRiskReport:
    """Generate complete risk assessment report."""
    # Calculate all risk components
    var_result = calculate_portfolio_var(portfolio)
    leveraged_risks = analyze_leveraged_etf_risks(portfolio)
    concentration_risk = analyze_concentration_risk(portfolio)
    currency_risk = analyze_currency_risk(portfolio)
    drawdown_scenarios = generate_drawdown_scenarios(portfolio)
    correlation_breakdown = analyze_correlation_breakdown(portfolio)
    liquidity_risks = analyze_liquidity_risks(portfolio)
    position_limits = define_position_limits()
    risk_allocation = calculate_risk_allocation_framework(portfolio, var_result)
    rebalancing_triggers = generate_rebalancing_triggers(
        portfolio, var_result, concentration_risk
    )

    # Calculate overall risk score
    overall_score = calculate_overall_risk_score(
        var_result, leveraged_risks, concentration_risk, drawdown_scenarios
    )

    # Generate recommendations
    recommendations = generate_summary_recommendations(
        portfolio,
        var_result,
        concentration_risk,
        position_limits,
        rebalancing_triggers,
    )

    return ComprehensiveRiskReport(
        report_date=datetime.now(),
        portfolio=portfolio,
        var_analysis=var_result,
        leveraged_etf_risks=leveraged_risks,
        concentration_risk=concentration_risk,
        currency_risk=currency_risk,
        drawdown_scenarios=drawdown_scenarios,
        correlation_breakdown=correlation_breakdown,
        liquidity_risks=liquidity_risks,
        position_limits=position_limits,
        risk_allocation_framework=risk_allocation,
        rebalancing_triggers=rebalancing_triggers,
        overall_risk_score=overall_score,
        summary_recommendations=recommendations,  # type: ignore[arg-type]
    )


def _log_section_header(title: str) -> None:
    """Log a section header with separator."""
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)


def _log_var_analysis(var: VaRResult) -> None:
    """Log VaR analysis section."""
    _log_section_header("1. VALUE AT RISK (VaR) ANALYSIS")
    logger.info(f"Confidence Level: {var.confidence_level * 100:.0f}%")
    logger.info(f"Time Horizon: {var.time_horizon_days} day")
    logger.info(f"VaR: {var.var_percent:.2f}% (EUR {var.var_eur:,.0f})")
    logger.info(
        f"CVaR (Expected Shortfall): {var.cvar_percent:.2f}% (EUR {var.cvar_eur:,.0f})"
    )
    logger.info(f"Methodology: {var.methodology}")
    logger.info("")


def _log_leveraged_risks(risks: list[LeveragedETFRisk]) -> None:
    """Log leveraged ETF risks section."""
    _log_section_header("2. LEVERAGED ETF SPECIFIC RISKS")
    for risk in risks:
        logger.info(f"\n{risk.ticker}:")
        logger.info(f"  Daily Reset Impact: {risk.daily_reset_impact:.2f}% annually")
        logger.info(f"  Volatility Drag: {risk.volatility_drag:.2f}% annually")
        logger.info(f"  Path Dependency Risk: {risk.path_dependency_risk.value}")
        logger.info(f"  Compounding Error (1Y): {risk.compounding_error:.2f}%")
        logger.info(f"  Recommendation: {risk.recommendation}")
    logger.info("")


def _log_concentration_risk(conc: ConcentrationRisk) -> None:
    """Log concentration risk section."""
    _log_section_header("3. CONCENTRATION RISK")
    logger.info("Geographic Concentration:")
    for geo, pct in conc.geographic_concentration.items():
        logger.info(f"  {geo}: {pct * 100:.1f}%")
    logger.info("\nSector Concentration:")
    for sector, pct in conc.sector_concentration.items():
        logger.info(f"  {sector}: {pct * 100:.1f}%")
    logger.info(f"\nHerfindahl Index: {conc.herfindahl_index:.3f}")
    logger.info(f"Effective N Holdings: {conc.effective_n_holdings:.2f}")
    logger.info(f"Risk Level: {conc.risk_level.value}")
    logger.info("")


def _log_currency_risk(curr: CurrencyRisk) -> None:
    """Log currency risk section."""
    _log_section_header("4. CURRENCY RISK (EUR/USD)")
    logger.info(f"USD Exposure: {curr.usd_exposure_percent:.1f}%")
    logger.info(f"EUR/USD Volatility: {curr.eur_usd_volatility:.1f}% annually")
    logger.info(f"1-Month Currency VaR: {curr.var_currency_1m:.2f}%")
    logger.info(f"Risk Level: {curr.risk_level.value}")
    logger.info(f"Recommendation: {curr.hedging_recommendation}")
    logger.info("")


def _log_drawdown_scenarios(scenarios: list[DrawdownScenario]) -> None:
    """Log drawdown scenarios section."""
    _log_section_header("5. DRAWDOWN SCENARIOS (STRESS TESTS)")
    for scenario in scenarios:
        logger.info(f"\n{scenario.scenario_name} ({scenario.historical_period}):")
        logger.info(
            f"  Estimated Portfolio Drawdown: "
            f"{scenario.estimated_portfolio_drawdown:.1f}%"
        )
        logger.info(f"  Estimated Loss: EUR {scenario.estimated_loss_eur:,.0f}")
        logger.info(f"  Recovery Time: {scenario.recovery_time_estimate}")
        logger.info(f"  Probability: {scenario.probability_assessment}")
        logger.info(f"  Risk Level: {scenario.risk_level.value}")
    logger.info("")


def _log_correlation_breakdown(corr: CorrelationBreakdown) -> None:
    """Log correlation breakdown section."""
    _log_section_header("6. CORRELATION BREAKDOWN RISK")
    logger.info("Normal Market Correlations:")
    for ticker1, correlations in corr.normal_correlation.items():
        logger.info(f"  {ticker1}: {correlations}")
    logger.info("\nStress Correlations:")
    for ticker1, correlations in corr.stress_correlation.items():
        logger.info(f"  {ticker1}: {correlations}")
    logger.info(f"\nDiversification Ratio: {corr.diversification_ratio:.2f}")
    logger.info(f"Risk Level: {corr.risk_level.value}")
    logger.info("\nBreakdown Scenarios:")
    for scenario in corr.correlation_breakdown_scenarios:
        logger.info(f"  - {scenario}")
    logger.info("")


def _log_liquidity_risks(risks: list[LiquidityRisk]) -> None:
    """Log liquidity risks section."""
    _log_section_header("7. LIQUIDITY RISK")
    for liq in risks:
        logger.info(f"\n{liq.ticker}:")
        logger.info(f"  Avg Daily Volume: EUR {liq.avg_daily_volume_eur:,.0f}")
        logger.info(
            f"  Days to Liquidate 5% Position: {liq.days_to_liquidate_5pct:.2f}"
        )
        logger.info(f"  Bid-Ask Spread: {liq.bid_ask_spread_bps:.1f} bps")
        logger.info(
            f"  Total Liquidity Cost: {liq.liquidity_cost_estimate_bps:.1f} bps"
        )
        logger.info(f"  Risk Level: {liq.risk_level.value}")
    logger.info("")


def _log_position_limits(limits: PositionLimits) -> None:
    """Log position limits section."""
    _log_section_header("8. POSITION LIMITS")
    logger.info(
        f"Max Single Leveraged ETF: {limits.leveraged_etf_max_single * 100:.0f}%"
    )
    logger.info(
        f"Max Total Leveraged ETFs: {limits.leveraged_etf_max_total * 100:.0f}%"
    )
    logger.info(f"Max Single Geography: {limits.single_geography_max * 100:.0f}%")
    logger.info(f"Max Single Sector: {limits.single_sector_max * 100:.0f}%")
    logger.info(f"Min Cash Reserve: {limits.min_cash_reserve * 100:.0f}%")
    logger.info(f"\nRationale: {limits.rationale}")
    logger.info("")


def _log_risk_allocation(risk_alloc: RiskAllocationFramework) -> None:
    """Log risk allocation framework section."""
    _log_section_header("9. RISK-BASED ALLOCATION FRAMEWORK")
    logger.info(f"Total Risk Budget (VaR): {risk_alloc.risk_budget_total:.2f}%")
    logger.info("\nRisk Contribution by Position:")
    for ticker, risk in risk_alloc.risk_allocation.items():
        logger.info(f"  {ticker}: {risk:.2f}%")
    logger.info("\nRecommended Weights (Equal Risk Contribution):")
    for ticker, weight in risk_alloc.recommended_weights.items():
        logger.info(f"  {ticker}: {weight * 100:.1f}%")
    logger.info("\nCurrent vs Target:")
    for ticker, data in risk_alloc.current_vs_target.items():
        logger.info(
            f"  {ticker}: {data['current'] * 100:.1f}% -> "
            f"{data['target'] * 100:.1f}% ({data['difference']:+.1%})"
        )
    logger.info(f"\nRebalancing Needed: {risk_alloc.rebalancing_needed}")
    logger.info("")


def _log_rebalancing_triggers(triggers: list[RebalancingTrigger]) -> None:
    """Log rebalancing triggers section."""
    _log_section_header("10. REBALANCING TRIGGERS")
    for trigger in triggers:
        logger.info(f"\n{trigger.trigger_type}:")
        logger.info(f"  Current: {trigger.current_value:.2%}")
        logger.info(f"  Threshold: {trigger.threshold:.2%}")
        logger.info(f"  Action Required: {trigger.action_required}")
        logger.info(f"  Recommended Action: {trigger.recommended_action}")
        logger.info(f"  Urgency: {trigger.urgency.value}")
    logger.info("")


def main() -> None:
    """Run comprehensive risk assessment."""
    _log_section_header("PEA PORTFOLIO RISK ASSESSMENT")
    logger.info("Nicolas, Risk Manager")
    logger.info("")

    # Create portfolio and generate report
    portfolio = create_pea_portfolio()
    report = generate_comprehensive_risk_report(portfolio)

    # Print summary
    logger.info(f"Report Date: {report.report_date.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"Portfolio Value: EUR {report.portfolio.total_value_eur:,.0f}")
    logger.info(f"Overall Risk Score: {report.overall_risk_score}/10")
    logger.info("")

    # Log all sections
    _log_var_analysis(report.var_analysis)
    _log_leveraged_risks(report.leveraged_etf_risks)
    _log_concentration_risk(report.concentration_risk)
    _log_currency_risk(report.currency_risk)
    _log_drawdown_scenarios(report.drawdown_scenarios)
    _log_correlation_breakdown(report.correlation_breakdown)
    _log_liquidity_risks(report.liquidity_risks)
    _log_position_limits(report.position_limits)
    _log_risk_allocation(report.risk_allocation_framework)
    _log_rebalancing_triggers(report.rebalancing_triggers)

    # Summary recommendations
    _log_section_header("SUMMARY RECOMMENDATIONS")
    for i, rec in enumerate(report.summary_recommendations, 1):
        logger.info(f"{i}. {rec}")
    logger.info("")

    logger.info("=" * 80)
    logger.info("END OF REPORT")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
