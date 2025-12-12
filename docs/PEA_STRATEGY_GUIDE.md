# PEA Strategy Guide

This document provides comprehensive documentation for the PEA (Plan d'Epargne en Actions) investment strategy implemented in the PEA Portfolio Optimization System, including ETF selection rationale, tax optimization considerations, and long-term investment guidance.

---

## Table of Contents

1. [PEA Overview](#pea-overview)
2. [Tax Benefits Framework](#tax-benefits-framework)
3. [ETF Selection Rationale](#etf-selection-rationale)
4. [PEA Eligibility Requirements](#pea-eligibility-requirements)
5. [Long-Term Investment Horizon](#long-term-investment-horizon)
6. [Tax Optimization Strategies](#tax-optimization-strategies)
7. [Risk Warnings and Disclaimers](#risk-warnings-and-disclaimers)

---

## PEA Overview

### What is a PEA?

The Plan d'Epargne en Actions (PEA) is a French tax-advantaged investment account designed to encourage long-term equity investment by French tax residents.

### Key Characteristics

| Attribute | Value |
|-----------|-------|
| Contribution Ceiling | 150,000 EUR (principal only, gains excluded) |
| Eligible Instruments | EU-domiciled equities and eligible ETFs |
| Minimum Holding Period | 5 years for tax advantages |
| Account Type | Individual (one PEA per person) |
| Tax Treatment | Capital gains exempt from income tax after 5 years |

### Regulatory Framework

- **Legal Basis:** Articles L221-30 to L221-32 of the Code Monetaire et Financier
- **Oversight:** Autorite des Marches Financiers (AMF)
- **Tax Authority:** Direction Generale des Finances Publiques (DGFiP)

---

## Tax Benefits Framework

### Holding Period Tax Implications

The PEA's primary benefit is conditional on holding period:

#### Before 5 Years (Early Withdrawal)

```
Tax on Gains = Income Tax Rate + Social Charges
             = 12.8% (flat) + 17.2%
             = 30% (Prelevement Forfaitaire Unique - PFU)

OR

             = Marginal Income Tax Rate + 17.2% Social Charges
             (Option on election, beneficial if marginal rate < 12.8%)
```

**Additional Consequences of Early Withdrawal:**
- Immediate and permanent PEA closure
- Loss of tax-advantaged status
- Cannot open new PEA for period specified by regulation

#### After 5 Years (Qualified Withdrawal)

```
Tax on Gains = 0% Income Tax + 17.2% Social Charges
             = 17.2% total on net gains
```

**Benefits:**
- Partial withdrawals allowed without closing account
- Tax-free compounding on future gains
- Can continue contributing up to ceiling

### Comparative Tax Analysis

For a 10,000 EUR gain over different periods:

| Scenario | Holding Period | Tax Liability | Net After Tax |
|----------|---------------|---------------|---------------|
| Early Exit | < 5 years | 3,000 EUR (30%) | 7,000 EUR |
| Qualified | >= 5 years | 1,720 EUR (17.2%) | 8,280 EUR |
| CTO (Standard) | Any | 3,000 EUR (30%) | 7,000 EUR |

**5-Year Tax Savings:** 1,280 EUR per 10,000 EUR gain (12.8% improvement)

### Social Charges Evolution

Historical social charges rates (Prelevements Sociaux):

| Period | Rate |
|--------|------|
| Before 2018 | 15.5% |
| 2018-Present | 17.2% |

**Note:** Historic gains realized during lower-rate periods retain their applicable rate at withdrawal.

---

## ETF Selection Rationale

### Selected ETFs

The system uses three PEA-eligible ETFs:

| Symbol | ISIN | Name | Type |
|--------|------|------|------|
| LQQ.PA | FR0010342592 | Amundi Nasdaq-100 Daily (2x) Leveraged UCITS ETF | Leveraged Equity |
| CL2.PA | FR0010755611 | Amundi MSCI USA Daily (2x) Leveraged UCITS ETF | Leveraged Equity |
| WPEA.PA | FR0011869353 | Amundi MSCI World UCITS ETF | World Equity |

### ETF Details

#### LQQ.PA - Amundi Nasdaq-100 (2x) Leveraged

```python
# From src/data/models.py
ETFInfo(
    symbol=ETFSymbol.LQQ,
    isin="FR0010342592",
    name="Amundi Nasdaq-100 Daily (2x) Leveraged UCITS ETF",
    leverage=2,
    ter=0.006,  # 0.60% annual fee
    pea_eligible=True,
    accumulating=True,
)
```

**Selection Rationale:**
- **Index Exposure:** Nasdaq-100 (technology-heavy US large caps)
- **Leverage:** 2x daily leverage for enhanced upside capture
- **PEA Eligibility:** Synthetic replication via swap with EU counterparty
- **Accumulating:** Dividends reinvested (tax-efficient within PEA)
- **Use Case:** Risk-on regime allocation for growth exposure

**Risk Considerations:**
- Volatility decay in sideways/volatile markets
- Concentrated sector exposure (technology)
- Higher TER than unleveraged alternatives

#### CL2.PA - Amundi MSCI USA (2x) Leveraged

```python
ETFInfo(
    symbol=ETFSymbol.CL2,
    isin="FR0010755611",
    name="Amundi MSCI USA Daily (2x) Leveraged UCITS ETF",
    leverage=2,
    ter=0.005,  # 0.50% annual fee
    pea_eligible=True,
    accumulating=True,
)
```

**Selection Rationale:**
- **Index Exposure:** MSCI USA (broad US market)
- **Leverage:** 2x daily leverage for enhanced beta
- **Diversification:** More diversified than Nasdaq-100
- **Lower TER:** Slightly cheaper than LQQ
- **Use Case:** Complement to LQQ for diversified leveraged exposure

**Risk Considerations:**
- Same volatility decay concerns as LQQ
- High correlation with LQQ (both US equity)
- Currency exposure (USD)

#### WPEA.PA - Amundi MSCI World

```python
ETFInfo(
    symbol=ETFSymbol.WPEA,
    isin="FR0011869353",
    name="Amundi MSCI World UCITS ETF",
    leverage=1,
    ter=0.0038,  # 0.38% annual fee
    pea_eligible=True,
    accumulating=True,
)
```

**Selection Rationale:**
- **Index Exposure:** MSCI World (developed markets globally)
- **No Leverage:** Core stable allocation without decay
- **Diversification:** ~1,500 stocks across 23 developed markets
- **Low TER:** Cost-effective broad exposure
- **Use Case:** Portfolio foundation across all regimes

**Risk Considerations:**
- Still equity risk (no downside protection)
- Geographic concentration in US (~70%)
- Currency exposure (primarily USD, EUR, JPY, GBP)

### Why This Combination?

The three-ETF portfolio is designed for:

1. **Asymmetric Risk Profile:**
   - Leveraged ETFs (LQQ, CL2) provide upside during risk-on
   - WPEA provides stability and diversification
   - Cash provides optionality and drawdown mitigation

2. **PEA Optimization:**
   - All three are accumulating (no dividend tax drag)
   - All three are PEA-eligible (maintaining wrapper benefits)
   - Low TERs minimize cost drag over 5+ year horizon

3. **Regime Adaptability:**
   - Risk-On: Maximize leveraged exposure (30%)
   - Neutral: Balanced approach (20% leveraged)
   - Risk-Off: Minimize leverage (10%), raise cash

### Diagram: Portfolio Composition by Regime

```
RISK_ON Regime
+--------------------------------------------------+
|  LQQ (15%)  |  CL2 (15%)  |      WPEA (60%)      | CASH (10%)
|  Leveraged  |  Leveraged  |       Core           | Buffer
+--------------------------------------------------+

NEUTRAL Regime
+--------------------------------------------------+
| LQQ (10%) | CL2 (10%) |      WPEA (60%)      |   CASH (20%)
| Leveraged | Leveraged |       Core           |   Buffer
+--------------------------------------------------+

RISK_OFF Regime
+--------------------------------------------------+
|LQQ|CL2|         WPEA (60%)         |    CASH (30%)
| 5%| 5%|          Core              |    Buffer
+--------------------------------------------------+
```

---

## PEA Eligibility Requirements

### General Eligibility Rules

For an ETF to be PEA-eligible, it must:

1. **Domicile Requirement:** Be domiciled in the EU or EEA
2. **Composition Requirement:** Invest at least 75% in PEA-eligible securities
3. **Security Types:** Eligible securities include:
   - Shares of EU/EEA companies
   - Certain investment fund units
   - Synthetic exposure via eligible derivatives

### How Synthetic ETFs Achieve Eligibility

LQQ and CL2 use **synthetic replication** via total return swaps:

```
Physical Holdings: EU equities (for PEA eligibility)
Swap Overlay:      Exchange EU equity returns for index returns
Result:            PEA-eligible structure delivering US index exposure
```

**Diagram: Synthetic ETF Structure**

```
                    +------------------+
                    |   ETF Investor   |
                    |   (PEA Account)  |
                    +--------+---------+
                             |
                             | Owns ETF shares
                             v
                    +------------------+
                    |    ETF Fund      |
                    | (FR domiciled)   |
                    +--------+---------+
                             |
          +------------------+------------------+
          |                                     |
          v                                     v
+------------------+                  +------------------+
|  Physical Basket |   <-- Swap -->  | Swap Counterparty|
| (EU Equities)    |                 | (EU Bank)        |
| >= 75% PEA-elig  |                 +------------------+
+------------------+                          |
                                              v
                                    +------------------+
                                    |   Index Return   |
                                    | (Nasdaq-100, etc)|
                                    +------------------+
```

### Eligibility Verification

**Responsibility:** The investor must verify PEA eligibility before purchase.

**Verification Sources:**
1. Fund KIID (Key Investor Information Document)
2. Fund prospectus
3. Broker's PEA-eligible securities list
4. AMF official databases

### Risk of Eligibility Loss

An ETF may lose PEA eligibility if:

1. **Composition Change:** Physical basket falls below 75% eligible
2. **Regulatory Change:** EU rules modify eligibility criteria
3. **Fund Restructuring:** Manager changes fund domicile or strategy

**Consequences:**
- Forced sale of position
- Potential sale at unfavorable prices
- Disruption to investment strategy

**Mitigation:**
- Use established, large funds from major providers
- Monitor fund documentation for changes
- Maintain diversification across multiple eligible ETFs

---

## Long-Term Investment Horizon

### Why 5+ Years?

The PEA strategy is designed for a minimum 5-year holding period for three reasons:

#### 1. Tax Efficiency

```
5-Year Tax Advantage = 12.8% income tax exemption
On 100,000 EUR gain:  12,800 EUR saved vs. CTO
```

#### 2. Leveraged ETF Optimization

Leveraged ETFs are often criticized for long-term decay, but analysis shows:

**Favorable Long-Term Conditions:**
- Strong directional trends (bull markets)
- Low volatility environments
- Compound growth outpaces decay

**Historical Analysis (2x Nasdaq-100):**

| Period | Nasdaq-100 Return | LQQ Return | Decay/Gain |
|--------|-------------------|------------|------------|
| 2010-2019 | +380% | +1,200% | Significant outperformance |
| 2020 | +48% | +76% | Near theoretical |
| 2022 | -33% | -68% | Amplified loss |
| 2023-2024 | +85% | +170% | Near theoretical |

**Key Insight:** Over extended bull markets, 2x ETFs can significantly outperform despite decay. The strategy accepts decay as the "cost" of leveraged exposure.

#### 3. Regime Persistence

Market regimes tend to persist:

```
Average bull market duration:  4-6 years
Average bear market duration:  1-2 years
Neutral/transitional:          1-3 years
```

A 5+ year horizon captures multiple regime cycles, allowing the allocation strategy to demonstrate value.

### Recommended Monitoring

Despite the long-term horizon, regular monitoring is essential:

| Frequency | Activity |
|-----------|----------|
| Daily | Data fetching and storage |
| Weekly | Regime detection update |
| Monthly | Portfolio valuation and drift check |
| Quarterly | Rebalancing review (if threshold triggered) |
| Annually | Strategy review and documentation update |

---

## Tax Optimization Strategies

### Within-PEA Optimization

#### 1. Maximize Tax-Free Compounding

```python
# Accumulating ETFs reinvest dividends without tax event
# Annual dividend yield ~1.5% compounds tax-free

Year 0:  100,000 EUR
Year 5:  107,728 EUR from dividends alone (1.5% compound)
Year 10: 116,054 EUR

Tax saved (17.2% on dividends): ~2,760 EUR over 10 years
```

#### 2. Timing of Contributions

```
Optimal: Contribute early to maximize tax-free compounding
Ceiling: 150,000 EUR lifetime contributions

Strategy: Front-load contributions if capital available
```

#### 3. Avoid Early Withdrawal

```
Before 5 years: 30% tax + PEA closure
After 5 years:  17.2% tax, account remains open

Break-even analysis:
If you need funds in Year 4, borrowing at < 12.8%
may be preferable to early withdrawal.
```

### Exit Strategies

#### After 5 Years: Partial Withdrawals

```
Option A: Withdraw gains only, leave principal working
Option B: Systematic withdrawal (e.g., 4% rule)
Option C: Full liquidation for major purchase
```

#### Tax-Loss Harvesting (Limited)

Within PEA, losses can offset gains:

```
Realized loss:  -10,000 EUR
Realized gain:  +15,000 EUR
Net taxable:    +5,000 EUR
Tax at 17.2%:   860 EUR (vs 2,580 EUR without offset)
```

**Note:** Unlike US accounts, PEA does not allow loss harvesting against income outside the account.

### Coordination with Other Accounts

#### PEA + Assurance-Vie Combination

```
PEA:          Aggressive equity exposure (high growth potential)
Assurance-Vie: Fixed income and defensive assets (stability)

Tax treatment comparison:
- PEA after 5 years:      17.2% on gains
- AV after 8 years:       24.7% on gains up to 150k, then 30%
- AV inheritance:         Favorable succession treatment
```

#### PEA + CTO (Compte-Titres Ordinaire)

```
PEA:  All PEA-eligible holdings (maximize tax advantage)
CTO:  Non-eligible securities (US stocks, bonds, etc.)

Strategy: Fill PEA to ceiling first, then use CTO
```

---

## Risk Warnings and Disclaimers

### Critical Risks

#### 1. Capital Loss Risk

**THIS STRATEGY CAN RESULT IN SIGNIFICANT CAPITAL LOSS.**

Leveraged ETFs can lose 50% or more in a single market correction. The 2022 example:

```
Nasdaq-100:  -33%
LQQ (2x):    -68%

On 100,000 EUR:
WPEA (60%):  60,000 * -25% = -15,000 EUR
LQQ (20%):   20,000 * -68% = -13,600 EUR
CL2 (10%):   10,000 * -65% = -6,500 EUR
Total loss:  -35,100 EUR (35% of portfolio)
```

#### 2. Volatility Decay Risk

In sideways markets, leveraged ETFs lose value even if the index is flat:

```
Day 1: Index +5%, LQQ +10%
Day 2: Index -5%, LQQ -10%
Index net: -0.25%
LQQ net:   -1.00%

Over 50 such cycles: Index -12%, LQQ -39%
```

#### 3. Tax Law Change Risk

French tax law can change. Historical examples:
- Social charges increased from 15.5% to 17.2% in 2018
- PEA ceiling was increased from 132,000 to 150,000 EUR
- Future changes could increase taxes or modify eligibility

#### 4. PEA Eligibility Risk

ETFs may lose PEA eligibility, forcing:
- Forced sale (potentially at unfavorable prices)
- Taxable event
- Strategy disruption

### Disclaimers

```
THIS DOCUMENT IS FOR INFORMATIONAL PURPOSES ONLY AND DOES NOT
CONSTITUTE INVESTMENT ADVICE, TAX ADVICE, OR LEGAL ADVICE.

- Consult a qualified financial advisor before investing
- Consult a tax professional for personalized tax guidance
- Past performance does not predict future results
- All investments carry risk of loss
- The authors and developers accept no liability for losses

FOR PROFESSIONAL ADVICE:
- CIF (Conseiller en Investissements Financiers)
- CGP (Conseiller en Gestion de Patrimoine)
- Avocat fiscaliste (for complex tax situations)
```

### Suitability Considerations

This strategy may NOT be suitable if you:

1. **Cannot afford to lose capital:** Leveraged ETFs can cause significant losses
2. **Need funds within 5 years:** PEA tax benefits require 5-year holding
3. **Have low risk tolerance:** Strategy volatility can be 20-30% annually
4. **Lack investment experience:** Understanding of leverage is essential
5. **Are near retirement:** Time horizon may be insufficient for recovery

---

## Summary

The PEA strategy is designed for:

- **French tax residents** seeking tax-efficient equity exposure
- **Long-term investors** with 5+ year horizons
- **Risk-tolerant individuals** comfortable with leveraged products
- **Disciplined investors** who can follow systematic rules

The combination of LQQ, CL2, and WPEA provides:

- Growth potential through leveraged exposure
- Stability through world equity core
- Tax efficiency through PEA wrapper
- Risk management through regime-based allocation

**Key Success Factors:**
1. Maintain 5+ year holding period
2. Follow regime-based allocation rules
3. Respect hard-coded risk limits
4. Monitor but do not panic during drawdowns
5. Understand and accept leveraged ETF characteristics

---

**Document Version:** 1.0
**Last Updated:** 2025-12-12
**Author:** Portfolio Management System
**Review Cycle:** Annual or upon material regulatory/tax changes

**Regulatory Note:** Tax information is based on 2025 French tax law. Consult current regulations and professional advisors for your specific situation.
