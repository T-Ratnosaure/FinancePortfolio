# Tax Optimization Action Plan - PEA Portfolio

**Author:** David (Legal Team - French Tax Specialist)
**Created:** 2025-12-12
**Status:** DRAFT - Awaiting approval for Sprint 6

---

## EXECUTIVE SUMMARY

The current implementation has excellent PEA-appropriate risk limits but is missing critical tax tracking and reporting features. **Implementing the P0 items below can save 15,000+ EUR in taxes** by properly managing the 5-year holding period rule.

---

## P0 - CRITICAL (Sprint 6 - Week 1)

### 1. Track PEA Opening Date
**Tax Impact:** 15,000+ EUR savings by avoiding early withdrawal
**Effort:** 2-3 hours
**Legal Basis:** Article 163 quinquies D du CGI

**Implementation:**
```python
# Add to analytics schema
CREATE TABLE analytics.pea_metadata (
    id INTEGER PRIMARY KEY,
    pea_opening_date DATE NOT NULL,
    total_contributions DECIMAL(18,2) DEFAULT 0.0,
    contribution_ceiling DECIMAL(18,2) DEFAULT 150000.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

# Add to PortfolioTracker
def get_pea_age(self) -> dict:
    """Get PEA age and tax-free eligibility status."""
    opening_date = self._get_pea_opening_date()
    age_days = (date.today() - opening_date).days
    age_years = age_days / 365.25
    days_until_5_years = max(0, int(5 * 365.25 - age_days))

    return {
        "opening_date": opening_date,
        "age_years": round(age_years, 1),
        "age_days": age_days,
        "is_tax_free_eligible": age_years >= 5.0,
        "days_until_tax_free": days_until_5_years,
        "tax_status": "OPTIMIZED" if age_years >= 5.0 else "SUBOPTIMAL"
    }
```

**CLI Integration:**
```bash
# Show PEA status in portfolio summary
python main.py portfolio --summary

# Output should include:
# PEA Status:
#   Opening Date: 2022-06-15
#   Age: 3.5 years
#   Tax-Free Eligible: No (1.5 years remaining)
#   Warning: Withdrawals before 2027-06-15 will trigger 30% income tax!
```

**Success Criteria:**
- PEA opening date stored in database
- Portfolio summary displays PEA age
- Warning shown if approaching or past 5-year mark
- CLI command to update opening date: `python main.py pea --set-opening-date 2022-06-15`

---

### 2. Integrate Social Charges in Performance Metrics
**Tax Impact:** Shows realistic net returns (17.2% impact on gains)
**Effort:** 3-4 hours
**Legal Basis:** CSG-CRDS prelevement sociaux (17.2%)

**Implementation:**
```python
# Add to RiskCalculator (src/portfolio/risk.py)

SOCIAL_CHARGES_RATE = 0.172  # 17.2% as of 2025

def calculate_total_return_net_of_tax(
    self,
    start_value: Decimal,
    end_value: Decimal,
    social_charge_rate: float = SOCIAL_CHARGES_RATE,
) -> float:
    """Calculate total return net of French social charges (17.2%).

    French PEA gains are subject to 17.2% social charges upon withdrawal,
    even after 5 years. This method calculates the net return an investor
    would actually receive.

    Args:
        start_value: Portfolio value at start
        end_value: Portfolio value at end
        social_charge_rate: Social charges rate (default 0.172 for 2025)

    Returns:
        Net return as decimal (e.g., 0.10 = 10% net return)
    """
    gross_gain = end_value - start_value
    if start_value <= 0:
        return 0.0

    # Only gains are taxed, not the principal
    net_gain = gross_gain * (1 - social_charge_rate)
    net_return = float(net_gain / start_value)
    return net_return

def calculate_sharpe_ratio_net_of_tax(
    self,
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    social_charge_rate: float = SOCIAL_CHARGES_RATE,
) -> float:
    """Calculate Sharpe ratio adjusted for social charges.

    Adjusts returns to reflect net-of-tax performance.
    """
    # Adjust positive returns for tax
    adjusted_returns = returns.apply(
        lambda r: r * (1 - social_charge_rate) if r > 0 else r
    )
    return self.calculate_sharpe_ratio(adjusted_returns, risk_free_rate)
```

**RiskReport Enhancement:**
```python
# Add to RiskReport model (src/portfolio/risk.py)
class RiskReport(BaseModel):
    # ... existing fields ...
    sharpe_ratio_gross: float | None = None  # Renamed from sharpe_ratio
    sharpe_ratio_net: float | None = None    # NEW: Net of social charges
    sortino_ratio_gross: float | None = None
    sortino_ratio_net: float | None = None   # NEW: Net of social charges
```

**Success Criteria:**
- Risk reports show both gross and net Sharpe/Sortino ratios
- Performance calculations include tax-adjusted returns
- Documentation explains 17.2% social charge impact
- Example output:
  ```
  Risk Report:
    Sharpe Ratio (gross): 1.45
    Sharpe Ratio (net of tax): 1.20  <- 17.2% lower returns
    Sortino Ratio (gross): 1.82
    Sortino Ratio (net of tax): 1.51
  ```

---

## P1 - HIGH (Sprint 6 - Week 2)

### 3. Build Withdrawal Tax Calculator
**Tax Impact:** Decision support for withdrawal planning
**Effort:** 4-6 hours

**Implementation:**
```python
# New module: src/portfolio/tax.py

class WithdrawalTaxReport(BaseModel):
    """Tax report for a proposed PEA withdrawal."""
    withdrawal_amount: Decimal
    proportional_gain: Decimal
    income_tax: Decimal  # 0 if > 5 years
    social_charges: Decimal  # Always 17.2%
    total_tax: Decimal
    net_proceeds: Decimal
    effective_tax_rate: float  # As percentage of withdrawal
    pea_will_close: bool  # True if withdrawal before 5 years
    pea_age_years: float
    recommendation: str

def calculate_withdrawal_tax(
    portfolio_value: Decimal,
    total_contributions: Decimal,
    withdrawal_amount: Decimal,
    pea_opening_date: date,
    as_of_date: date = date.today(),
) -> WithdrawalTaxReport:
    """Calculate tax implications of a PEA withdrawal.

    Args:
        portfolio_value: Current total PEA value
        total_contributions: Total contributions made (versements)
        withdrawal_amount: Amount to withdraw
        pea_opening_date: Date PEA was opened
        as_of_date: Withdrawal date (default: today)

    Returns:
        WithdrawalTaxReport with complete tax breakdown
    """
    # Calculate PEA age
    age_years = (as_of_date - pea_opening_date).days / 365.25

    # Calculate total gain and proportional gain
    total_gain = portfolio_value - total_contributions
    if portfolio_value <= 0:
        proportional_gain = Decimal("0")
    else:
        proportional_gain = withdrawal_amount * (total_gain / portfolio_value)

    # Calculate taxes
    if age_years < 5.0:
        # Before 5 years: 12.8% income tax + 17.2% social charges
        income_tax = proportional_gain * Decimal("0.128")
        pea_will_close = True
    else:
        # After 5 years: 0% income tax + 17.2% social charges
        income_tax = Decimal("0")
        pea_will_close = False

    social_charges = proportional_gain * Decimal("0.172")
    total_tax = income_tax + social_charges
    net_proceeds = withdrawal_amount - total_tax
    effective_rate = float(total_tax / withdrawal_amount) if withdrawal_amount > 0 else 0.0

    # Generate recommendation
    if age_years < 5.0:
        days_until_5_years = int((5.0 - age_years) * 365.25)
        potential_savings = float(proportional_gain * Decimal("0.128"))
        recommendation = (
            f"WARNING: Withdrawing now will close your PEA permanently and "
            f"cost {income_tax:.2f} EUR in income tax. "
            f"Consider waiting {days_until_5_years} days to save {potential_savings:.2f} EUR. "
            f"Tax-free withdrawals available from {(pea_opening_date + timedelta(days=5*365.25)).strftime('%Y-%m-%d')}."
        )
    else:
        recommendation = (
            f"PEA is {age_years:.1f} years old (>5 years). "
            f"Withdrawal is tax-optimized: 0% income tax, only 17.2% social charges apply. "
            f"PEA will remain open for future contributions."
        )

    return WithdrawalTaxReport(
        withdrawal_amount=withdrawal_amount,
        proportional_gain=proportional_gain,
        income_tax=income_tax,
        social_charges=social_charges,
        total_tax=total_tax,
        net_proceeds=net_proceeds,
        effective_tax_rate=effective_rate * 100,  # As percentage
        pea_will_close=pea_will_close,
        pea_age_years=age_years,
        recommendation=recommendation,
    )
```

**CLI Integration:**
```bash
# Calculate tax for a proposed withdrawal
python main.py pea --withdrawal-tax 30000

# Output:
# Withdrawal Tax Report
# =====================
# Withdrawal Amount: 30,000.00 EUR
# Proportional Gain: 10,000.00 EUR
# Income Tax (12.8%): 1,280.00 EUR
# Social Charges (17.2%): 1,720.00 EUR
# Total Tax: 3,000.00 EUR
# Net Proceeds: 27,000.00 EUR
# Effective Tax Rate: 10.0%
#
# WARNING: Your PEA is only 3.5 years old.
# Withdrawing now will:
#   1. Close your PEA permanently
#   2. Cost 1,280 EUR in income tax
#   3. Forfeit 1.5 years of tax-free growth
#
# RECOMMENDATION: Wait 548 days until 2027-06-15 to save 1,280 EUR in taxes.
```

**Success Criteria:**
- Accurate tax calculation for any withdrawal amount
- Clear warnings for sub-5-year withdrawals
- CLI command functional
- Tax report export to PDF/CSV

---

### 4. Track Total Contributions (Versements)
**Tax Impact:** Prevents exceeding 150,000 EUR ceiling
**Effort:** 2-3 hours
**Legal Basis:** Article 163 quinquies D du CGI (150k ceiling)

**Implementation:**
```python
# Add to PortfolioTracker (src/portfolio/tracker.py)

def get_contribution_status(self) -> dict:
    """Get PEA contribution status vs ceiling."""
    contributions = self._get_total_contributions()
    ceiling = Decimal("150000.00")
    remaining = ceiling - contributions
    utilization = float(contributions / ceiling)

    return {
        "total_contributions": contributions,
        "contribution_ceiling": ceiling,
        "remaining_capacity": remaining,
        "utilization_percent": utilization * 100,
        "is_at_ceiling": contributions >= ceiling,
    }

def deposit_cash(self, amount: Decimal, is_new_contribution: bool = True) -> None:
    """Add cash to the portfolio.

    Args:
        amount: Amount to deposit
        is_new_contribution: If True, counts toward PEA contribution ceiling.
                            If False, is a redeposit of withdrawn funds.
    """
    # ... existing deposit logic ...

    if is_new_contribution:
        self._record_contribution(amount)
        self._check_contribution_ceiling()

def _record_contribution(self, amount: Decimal) -> None:
    """Record a new contribution to PEA."""
    self.db.conn.execute("""
        UPDATE analytics.pea_metadata
        SET total_contributions = total_contributions + ?,
            updated_at = CURRENT_TIMESTAMP
    """, [float(amount)])

def _check_contribution_ceiling(self) -> None:
    """Check if contribution ceiling has been reached."""
    status = self.get_contribution_status()
    if status["remaining_capacity"] < Decimal("10000"):
        logger.warning(
            f"PEA contribution ceiling warning: "
            f"{status['remaining_capacity']:.2f} EUR remaining "
            f"({status['utilization_percent']:.1f}% utilized)"
        )
    if status["is_at_ceiling"]:
        logger.error("PEA contribution ceiling reached! No further contributions allowed.")
```

**Success Criteria:**
- All deposits tracked as contributions
- Warning when <10,000 EUR remaining before ceiling
- Error if attempting to exceed ceiling
- Contribution history exportable

---

## P2 - MEDIUM (Sprint 7-8)

### 5. FIFO Tax-Lot Tracking
**Tax Impact:** Accurate capital gains for tax reporting
**Effort:** 12-16 hours (significant schema change)

**Why P2 (not P0/P1):**
- Currently using average cost basis (acceptable approximation)
- FIFO matters most for PARTIAL sales and tax reporting
- Can be implemented later without breaking changes
- More important once PEA reaches 5+ years and partial withdrawals begin

**Implementation Outline:**
```python
# New model: src/data/models.py
class TaxLot(BaseModel):
    """Individual tax lot for FIFO tracking."""
    symbol: ETFSymbol
    purchase_date: date
    shares: float
    cost_basis_per_share: float
    total_cost_basis: float
    is_fully_sold: bool = False

# Modify Position to include tax_lots
class Position(BaseModel):
    # ... existing fields ...
    tax_lots: list[TaxLot] = Field(default_factory=list)

# Add FIFO sale logic
def calculate_fifo_gain_loss(
    tax_lots: list[TaxLot],
    shares_to_sell: float,
    sale_price: float,
) -> tuple[Decimal, list[TaxLot]]:
    """Calculate capital gain using FIFO method."""
    # Sell oldest lots first
    # Return: (total_gain, updated_tax_lots)
```

---

### 6. IFU Export Functionality
**Tax Impact:** Simplifies annual tax filing
**Effort:** 4-6 hours

**Implementation:**
```python
# Add to PortfolioTracker
def export_ifu_data(self, tax_year: int) -> pd.DataFrame:
    """Export data for French tax declaration (IFU format).

    For PEA, only exports withdrawal events (internal trades not taxed).
    """
    withdrawals = self.get_withdrawal_history(
        start_date=date(tax_year, 1, 1),
        end_date=date(tax_year, 12, 31)
    )

    data = []
    for withdrawal in withdrawals:
        data.append({
            "Date de retrait": withdrawal.date.strftime("%d/%m/%Y"),
            "Montant brut": withdrawal.amount,
            "Plus-values": withdrawal.gain,
            "Prelevement sociaux (17.2%)": withdrawal.gain * 0.172,
            "Type de retrait": "Partiel" if not withdrawal.closed_pea else "Total",
        })

    return pd.DataFrame(data)
```

---

## P3 - LOW (Sprint 9+)

### 7. Tax-Optimized Withdrawal Order
**Tax Impact:** Marginal (100-500 EUR annually)
**Effort:** HIGH (8-12 hours)

**Why P3:**
- Complex logic with minimal benefit
- Only relevant after 5 years for PARTIAL withdrawals
- Current rebalancer already optimized for risk (sell leveraged first)
- Tax impact is small compared to 5-year rule savings

---

## IMPLEMENTATION SCHEDULE

### Sprint 6 - Week 1 (Dec 16-20, 2025)
- [ ] P0.1: PEA opening date tracking (2-3 hours)
- [ ] P0.2: Social charges in metrics (3-4 hours)
- [ ] Documentation: Add tax considerations to README

### Sprint 6 - Week 2 (Dec 23-27, 2025)
- [ ] P1.3: Withdrawal tax calculator (4-6 hours)
- [ ] P1.4: Contribution tracking (2-3 hours)
- [ ] Testing: Tax calculation scenarios

### Sprint 7 (Jan 2026)
- [ ] P2.5: FIFO tax-lot tracking (12-16 hours)
- [ ] P2.6: IFU export functionality (4-6 hours)

### Sprint 8+ (Feb 2026+)
- [ ] P3.7: Tax-optimized withdrawal order (if needed)
- [ ] Comprehensive tax optimization guide
- [ ] Consultation with avocat fiscaliste for complex scenarios

---

## SUCCESS METRICS

### Quantitative
- **Tax Savings:** Track actual tax saved by waiting for 5-year mark
- **Accuracy:** Tax calculations match official French tax simulator (impots.gouv.fr)
- **Compliance:** Zero tax filing errors or AMF violations

### Qualitative
- **User Confidence:** Clear understanding of PEA tax implications
- **Decision Support:** Informed withdrawal decisions based on tax impact
- **Audit Readiness:** Complete documentation for French tax authorities

---

## RISK MITIGATION

| Risk | Mitigation |
|------|------------|
| **Tax law changes** | Monitor BOFiP updates, make rates configurable |
| **Calculation errors** | Unit tests against official tax examples |
| **User misunderstanding** | Clear warnings, French documentation |
| **Audit issues** | Maintain complete transaction logs, IFU reconciliation |

---

## RESOURCES

### French Tax Law
- **Impots.gouv.fr:** Official PEA tax information
- **BOFiP:** Administrative guidelines
- **AMF:** Financial markets regulator

### Professional Consultation
Consider consulting an **avocat fiscaliste** for:
- Portfolio values > 500,000 EUR
- Complex succession planning
- International tax situations (expatriation)

---

## APPROVAL CHECKLIST

Before implementing, verify:
- [ ] User has confirmed PEA opening date
- [ ] Current social charges rate is 17.2% (verify annually)
- [ ] Contribution ceiling is 150,000 EUR (verify with new loi de finances)
- [ ] No conflicting features in other sprints
- [ ] Database backup completed before schema changes

---

**Status:** READY FOR IMPLEMENTATION
**Owner:** To be assigned
**Estimated Total Effort:** 25-35 hours (P0-P2)
**Expected Tax Savings:** 15,000+ EUR (one-time) + ongoing optimization

---

**End of Action Plan**
