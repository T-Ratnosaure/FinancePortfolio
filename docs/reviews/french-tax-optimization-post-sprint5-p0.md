# French Tax Optimization Review - Post Sprint 5 P0
**Reviewer:** David (Legal Team - French Tax Optimization Specialist)
**Review Date:** 2025-12-12
**Scope:** Risk limits, portfolio allocation, and FIFO tax-lot implementation

---

## 1. SUMMARY / EXECUTIVE OVERVIEW

Sprint 5 P0 has successfully implemented the foundational risk management layer with PEA-appropriate constraints. From a French tax optimization perspective, the implementation shows:

**STRENGTHS:**
- Conservative 30% leveraged ETF limit (excellent for PEA risk profile)
- 10% minimum cash buffer (ensures liquidity for tax-efficient withdrawals)
- FIFO tax-lot method properly referenced in rebalancer
- Risk limits hard-coded as constants (prevents accidental non-compliance)

**AREAS REQUIRING ATTENTION:**
- Missing tax-specific documentation for withdrawal strategies
- No explicit tracking of PEA holding period (5-year rule critical)
- Lack of tax-lot level gain/loss tracking for withdrawal optimization
- No integration of prelevement sociaux (17.2%) in performance calculations

**OVERALL ASSESSMENT:** ACCEPTABLE with recommended enhancements for tax optimization

---

## 2. PEA TAX COMPLIANCE ANALYSIS

### 2.1 Risk Limits - PEA Appropriateness

**FINDING:** Risk limits are EXCELLENT for PEA compliance

**Analysis:**

| Limit | Value | PEA Assessment |
|-------|-------|----------------|
| Max Leveraged Exposure | 30% | EXCELLENT - Conservative given volatility decay risk |
| Max Single Position | 25% | GOOD - Prevents concentration risk |
| Min Cash Buffer | 10% | EXCELLENT - Critical for tax-efficient withdrawals |
| Rebalance Threshold | 5% | GOOD - Reduces transaction costs (important for net returns) |
| Drawdown Alert | -20% | GOOD - Early warning system |

**Legal/Tax Justification:**

1. **30% Leveraged Cap:** Given the volatility decay on 2x leveraged ETFs (LQQ, CL2), a 30% combined limit is prudent. The tax code does not penalize leveraged ETF losses differently, but excessive losses can:
   - Reduce the tax-advantaged capital base in the PEA
   - Force premature withdrawals (triggering 5-year rule violations)
   - Create psychological pressure leading to suboptimal timing

2. **10% Cash Buffer:** This is CRITICAL for PEA tax optimization:
   - **Article 163 quinquies D du CGI:** Withdrawals before 5 years trigger PEA closure
   - Having 10% cash allows for:
     - Emergency withdrawals without forced ETF liquidation at bad prices
     - Opportunistic rebalancing without selling positions
     - Meeting minimum cash needs while preserving tax-advantaged status

3. **25% Single Position Limit:** While not explicitly required by PEA regulations, this prevents:
   - Excessive concentration risk if an ETF loses PEA eligibility (forced sale)
   - Large unrealized gains in one position (complicates withdrawal strategy)

**RECOMMENDATION:** Maintain current limits. Consider documenting the tax rationale in code comments.

### 2.2 FIFO Tax-Lot Implementation

**FINDING:** FIFO correctly referenced but implementation incomplete

**Location:** `src/portfolio/rebalancer.py`, line 163

```python
tax_lot_method: str = Field(
    default="FIFO",
    description="Tax lot selection method (FIFO required for PEA)",
)
```

**PEA Tax Law Reference:**
- **Article 150-0 D du CGI (Code General des Impots):** France uses FIFO (Premier Entre, Premier Sorti) for calculating capital gains on securities.
- This is NOT optional - it's the legal requirement for all French investors.

**Current Status:**
- Configuration correctly defaults to FIFO
- Comment states "FIFO required for PEA" (CORRECT)
- However: No actual tax-lot tracking implementation in `PortfolioTracker`

**MISSING IMPLEMENTATION:**

The `Position` model (src/data/models.py) uses:
```python
average_cost: float  # Weighted average cost
```

But for true FIFO tax-lot optimization, we need:
```python
# Needed for tax optimization:
tax_lots: list[TaxLot]  # Track each purchase separately
    - purchase_date: date
    - shares: float
    - cost_basis_per_share: float
    - holding_period: int  # Days held
```

**TAX IMPLICATIONS:**

Without tax-lot tracking:
1. **Cannot optimize PARTIAL withdrawals:** When selling only part of a position, cannot determine which specific shares are being sold (oldest vs newest)
2. **Cannot calculate ACCURATE capital gains for tax reporting:** Average cost is an approximation, not the legal FIFO calculation
3. **Cannot optimize for 5-year holding period:** Cannot identify which shares have crossed the 5-year threshold

**RECOMMENDATION:**
- **Priority: MEDIUM** - Not blocking for current single-person usage, but required for:
  - Accurate tax reporting to French tax authorities
  - Withdrawal optimization after 5 years
  - Compliance with French accounting standards

### 2.3 PEA 5-Year Holding Period Tracking

**FINDING:** CRITICAL GAP - No tracking of PEA opening date or 5-year milestone

**French Tax Law:**
- **Article 163 quinquies D du CGI:**
  - Withdrawals before 5 years: Gains taxed at 30% PFU (or progressive scale) + 17.2% prelevement sociaux = **up to 47.2% total**
  - Withdrawals after 5 years: **0% income tax** + 17.2% prelevement sociaux = **17.2% total**
  - **Savings: up to 30 percentage points!**

**Current Implementation:**
- No `pea_opening_date` tracked in database
- No warnings when approaching 5-year anniversary
- No visibility into which positions are "safe" to withdraw (>5 years old)

**BUSINESS IMPACT:**

For a 150,000 EUR PEA with 50,000 EUR gains:
- Before 5 years: Tax = 50,000 x 0.30 = 15,000 EUR (income tax) + 50,000 x 0.172 = 8,600 EUR = **23,600 EUR total**
- After 5 years: Tax = 50,000 x 0.172 = **8,600 EUR total**
- **SAVINGS: 15,000 EUR** by waiting for 5-year mark

**RECOMMENDATION:**
- **Priority: HIGH** - Add PEA opening date to database
- Add to `PortfolioTracker.__init__` or separate PEA metadata table
- Create alert system: "PEA will reach 5 years on [DATE]"
- Show in portfolio summary: "PEA age: 4.2 years (0.8 years until tax optimization)"

---

## 3. TAX-OPTIMIZED WITHDRAWAL STRATEGIES

### 3.1 Current Cash Buffer Strategy

**FINDING:** 10% minimum cash buffer is EXCELLENT for tax optimization

**Tax Optimization Use Cases:**

1. **Avoid Forced Sales Before 5 Years:**
   - Emergency expenses can be met from cash buffer
   - Prevents triggering premature PEA closure
   - Preserves tax-advantaged growth potential

2. **Opportunistic Rebalancing:**
   - Can rebalance by deploying cash without selling positions
   - Reduces taxable events (none within PEA, but reduces turnover costs)
   - Maintains target allocation without triggering spreads

3. **Staged Withdrawals After 5 Years:**
   - Can withdraw cash first, preserving ETF positions
   - Allows strategic timing of ETF liquidations for tax efficiency

**RECOMMENDATION:** Maintain 10% minimum. Consider 15% for conservative investors.

### 3.2 Missing: Withdrawal Gain/Loss Calculation

**FINDING:** No functionality to calculate tax implications of a proposed withdrawal

**What's Needed:**

```python
def calculate_withdrawal_tax(
    withdrawal_amount: Decimal,
    pea_opening_date: date,
    as_of_date: date = date.today()
) -> WithdrawalTaxReport:
    """
    Calculate tax implications of a withdrawal from PEA.

    Returns:
        WithdrawalTaxReport with:
        - capital_gain: Taxable gain amount
        - income_tax: Income tax owed (0 if > 5 years)
        - social_charges: 17.2% prelevement sociaux
        - total_tax: Total tax owed
        - net_proceeds: Amount received after tax
        - pea_status: OPEN or CLOSED after withdrawal
    """
```

**Tax Calculation Logic:**

1. Determine total PEA value and total contributions (versements)
2. Calculate overall gain: `gain = current_value - total_contributions`
3. Calculate proportional gain in withdrawal:
   ```
   proportional_gain = withdrawal_amount * (gain / current_value)
   ```
4. Apply tax rates based on holding period:
   - < 5 years: 30% PFU (or progressive scale if more favorable) + 17.2%
   - >= 5 years: 0% income tax + 17.2%

**RECOMMENDATION:**
- **Priority: MEDIUM-HIGH** - Required for informed withdrawal decisions
- Implement before user needs to make first withdrawal

### 3.3 Tax-Efficient Rebalancing

**FINDING:** Rebalancer optimizes trade order but not tax consequences

**Current Order (rebalancer.py, lines 310-376):**
1. Sell leveraged positions first (risk reduction)
2. Sell non-leveraged positions
3. Buy non-leveraged positions
4. Buy leveraged positions last

**Tax Perspective:** This is GOOD because:
- Reduces risk exposure quickly (volatility decay on leveraged ETFs)
- Within PEA: No capital gains tax on internal trades
- Only matters for:
  - Transaction costs (spreads + commissions)
  - Opportunity cost of being out of market during rebalance

**ENHANCEMENT OPPORTUNITY:**

For PARTIAL PEA withdrawals (after 5 years), selling order should consider:
1. **Tax lot age:** Prefer selling oldest lots first (FIFO compliance)
2. **Unrealized gains:** If withdrawing, prefer selling positions with LOWER gains (reduces social charges)
3. **Volatility decay:** For leveraged ETFs, still sell first despite potential gains

**RECOMMENDATION:**
- **Priority: LOW** (only relevant after 5 years and for withdrawals)
- Current rebalancing order is tax-neutral within PEA
- Document tax considerations for future withdrawal optimization

---

## 4. PRELEVEMENT SOCIAUX (17.2%) INTEGRATION

### 4.1 Performance Metrics - Missing Social Charges

**FINDING:** Risk metrics do not account for 17.2% social charges on gains

**Current Metrics (src/portfolio/risk.py):**
- Sharpe Ratio: Calculates risk-adjusted returns
- Sortino Ratio: Calculates downside risk-adjusted returns
- Total Return: Gross return calculation

**Problem:** These are GROSS returns, not NET-OF-TAX returns

**French Tax Reality:**
- ALL PEA gains are subject to 17.2% prelevement sociaux upon withdrawal
- This is NON-NEGOTIABLE even after 5 years
- Applies to dividends, capital gains, and all forms of appreciation

**IMPACT:**

Example with 10% annual gross return:
- Gross 10% return shown in metrics
- Net-of-social-charges return: 10% * (1 - 0.172) = **8.28%**
- **Difference: 1.72 percentage points annually**

Over 10 years with 100,000 EUR initial:
- Gross: 100,000 * (1.10)^10 = 259,374 EUR
- Net-of-tax: 100,000 + (159,374 * 0.828) = 231,964 EUR
- **Tax paid: 27,410 EUR**

**RECOMMENDATION:**
- **Priority: HIGH** - Add tax-adjusted metrics
- Create parallel metrics:
  - `sharpe_ratio_gross` (current)
  - `sharpe_ratio_net_of_social_charges` (NEW)
- Show both in reports with clear labeling

### 4.2 Transaction Cost Estimation - Incomplete

**FINDING:** Rebalancer estimates commissions but not all PEA-specific costs

**Current (rebalancer.py, lines 378-411):**
```python
default_commission_rate: Decimal = Decimal("0.001")  # 0.1%
fixed_commission: Decimal = Decimal("0.0")
spread_cost: Decimal = Decimal("0.001")  # 0.1%
```

**Missing PEA-Specific Costs:**

1. **Courtage (brokerage fees):**
   - Typical French PEA brokers: 0.10% to 0.50%
   - Current 0.1% is optimistic (assumes Boursorama/Fortuneo)
   - BNP/Societe Generale: Often 0.5%

2. **Tax on Transactions (TTF - Taxe sur les Transactions Financieres):**
   - **0.30%** on French stock purchases > 1 billion EUR market cap
   - Does NOT apply to ETFs (ETFs are exempt)
   - Current implementation is CORRECT (no TTF for ETFs)

3. **Currency Conversion Fees:**
   - If using non-EUR broker for EUR-denominated ETFs
   - Not applicable for most French PEA brokers (already in EUR)

**RECOMMENDATION:**
- Current commission rates are reasonable for low-cost French brokers
- Add configuration option for broker-specific fee structures
- Document assumption: Using low-cost PEA broker (Boursorama, Fortuneo, Trade Republic)

---

## 5. TAX DOCUMENTATION REQUIREMENTS

### 5.1 Imprime Fiscal Unique (IFU) Preparation

**FINDING:** No support for generating IFU-compatible data

**French Tax Reporting Requirement:**
- Brokers issue IFU (Imprime Fiscal Unique) by January 31 each year
- Reports: Interest, dividends, capital gains for tax year
- Investor must report on annual tax return (Declaration 2042)

**For PEA:**
- While inside PEA: No annual reporting required (tax-deferred)
- Upon withdrawal: Must report gains on Declaration 2042-C
- After 5 years: Only report social charges (line 2OP)

**Current System:**
- Tracks trades and positions
- Can calculate unrealized gains
- **BUT:** No export functionality for tax declaration

**RECOMMENDATION:**
- **Priority: MEDIUM** - Add tax report export
- Generate CSV with columns:
  - Date de retrait (withdrawal date)
  - Montant brut (gross amount)
  - Plus-values (gains)
  - Prelevements sociaux dus (social charges owed)
- Export compatible with French tax software (e.g., impots.gouv.fr)

### 5.2 Position Valuation - Broker Reconciliation

**FINDING:** Reconciliation logic is GOOD (tracker.py, lines 264-385)

**Tax Importance:**
- Annual PEA statement from broker shows year-end value
- Needed for tracking against 150,000 EUR contribution ceiling
- Reconciliation ensures database matches broker (critical for tax reporting)

**Current Implementation:**
- Checks share count mismatches
- Checks price mismatches (1% tolerance)
- Flags missing positions

**Tax Enhancement Needed:**
- **Track total contributions (versements):** Separate from market value
- **Warn if approaching 150,000 EUR ceiling**
- **Log contribution history** for tax authority queries

**RECOMMENDATION:**
- **Priority: MEDIUM** - Add contribution tracking
- Separate table: `pea_contributions` with date and amount
- Alert: "Total contributions: 145,000 EUR (5,000 EUR remaining before ceiling)"

---

## 6. TAX OPTIMIZATION PRIORITIES

### Priority Matrix

| Priority | Task | Tax Savings Impact | Implementation Effort |
|----------|------|-------------------|----------------------|
| **P0 - CRITICAL** | Track PEA opening date | 15,000+ EUR (5-year rule) | LOW (single field) |
| **P0 - CRITICAL** | Add tax-adjusted performance metrics | Investor awareness | LOW (calculation change) |
| **P1 - HIGH** | Implement withdrawal tax calculator | Decision support | MEDIUM (new module) |
| **P1 - HIGH** | Track total contributions (versements) | Ceiling compliance | LOW (new table) |
| **P2 - MEDIUM** | FIFO tax-lot tracking | Accurate LTCG reporting | HIGH (schema change) |
| **P2 - MEDIUM** | IFU export functionality | Tax filing ease | MEDIUM (export format) |
| **P3 - LOW** | Tax-optimized withdrawal order | Marginal savings | HIGH (complex logic) |

### Recommended Implementation Sequence

**Sprint 5 - Remaining Work (or Sprint 6 P0):**

1. **Add PEA Metadata Tracking** (1-2 hours)
   ```python
   # New table: analytics.pea_metadata
   CREATE TABLE analytics.pea_metadata (
       pea_opening_date DATE NOT NULL,
       total_contributions DECIMAL(18,2) DEFAULT 0.0,
       contribution_ceiling DECIMAL(18,2) DEFAULT 150000.0,
       updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

2. **Add Tax-Adjusted Metrics** (2-3 hours)
   ```python
   # In RiskCalculator
   def calculate_sharpe_ratio_net_of_social_charges(
       self,
       returns: pd.Series,
       social_charge_rate: float = 0.172,
       risk_free_rate: float = 0.0,
   ) -> float:
       # Adjust returns for social charges on gains
       pass
   ```

3. **Create PEA Age Warning System** (1 hour)
   ```python
   # In PortfolioTracker
   def get_pea_status(self) -> dict:
       opening_date = self._get_pea_opening_date()
       age_years = (date.today() - opening_date).days / 365.25
       days_until_5_years = max(0, 365.25 * 5 - (date.today() - opening_date).days)

       return {
           "opening_date": opening_date,
           "age_years": age_years,
           "is_eligible_for_tax_free_withdrawal": age_years >= 5,
           "days_until_tax_optimization": int(days_until_5_years),
       }
   ```

**Sprint 6+:**

4. Implement withdrawal tax calculator
5. Add contribution tracking and ceiling warnings
6. Build FIFO tax-lot tracking (major feature)
7. Create IFU export functionality

---

## 7. COMPLIANCE CHECKLIST

### PEA Regulatory Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| **Article 163 quinquies D - Contribution Ceiling** | PARTIAL | No tracking of total versements |
| **Article 163 quinquies D - 5-Year Rule** | MISSING | No PEA opening date tracked |
| **Article 150-0 D - FIFO Method** | PARTIAL | Referenced but not implemented |
| **Article 163 quinquies D - Social Charges** | PARTIAL | Not integrated in metrics |
| **AMF Disclosure Requirements** | EXCELLENT | See compliance/risk_disclosures.md |
| **PEA Eligibility Monitoring** | MANUAL | User must verify ETF eligibility |

### Tax Reporting Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| **Declaration 2042-C (Withdrawals)** | NOT SUPPORTED | No export functionality |
| **IFU Data Reconciliation** | PARTIAL | Broker reconciliation exists |
| **Capital Gains Calculation** | APPROXIMATE | Uses average cost, not FIFO |
| **Prelevement Sociaux Reporting** | NOT SUPPORTED | No withdrawal tax calculation |

---

## 8. RISK ASSESSMENT - TAX PERSPECTIVE

### Risks from Current Implementation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Withdrawal before 5 years (uninformed)** | MEDIUM | HIGH (15k+ EUR tax) | P0: Add PEA age tracking |
| **Incorrect capital gains reporting** | LOW | MEDIUM (Audit risk) | P2: FIFO tax-lot tracking |
| **Exceeding 150k contribution ceiling** | LOW | HIGH (Penalty + disqualification) | P1: Track versements |
| **Underestimating tax liability** | HIGH | MEDIUM (Cash flow surprise) | P0: Tax-adjusted metrics |
| **Missing IFU reconciliation** | LOW | LOW (Extra work at tax time) | P2: IFU export |

### Opportunities for Tax Alpha

| Opportunity | Annual Savings Potential | Implementation Difficulty |
|-------------|-------------------------|--------------------------|
| **Waiting for 5-year mark** | 15,000 EUR (one-time) | LOW (just track date) |
| **Tax-loss harvesting within PEA** | N/A (no internal taxes) | N/A |
| **Optimized withdrawal timing** | 500-2,000 EUR | MEDIUM (withdrawal optimizer) |
| **Minimizing transaction costs** | 200-500 EUR annually | LOW (already implemented) |
| **Social charges optimization** | Cannot be optimized | N/A (mandatory 17.2%) |

**Key Insight:** The biggest tax optimization opportunity is simply **WAITING FOR 5 YEARS**. This one change can save 30% income tax on all gains.

---

## 9. RECOMMENDATIONS SUMMARY

### Immediate Actions (Sprint 5 or 6 P0)

1. **Add PEA Opening Date Tracking**
   - File: `src/portfolio/tracker.py`
   - Add: `pea_metadata` table with `opening_date`
   - Display PEA age in portfolio summary
   - Alert when approaching 5-year mark

2. **Integrate 17.2% Social Charges in Performance Metrics**
   - File: `src/portfolio/risk.py`
   - Add: `calculate_sharpe_ratio_net_of_social_charges()`
   - Add: `calculate_total_return_net_of_social_charges()`
   - Display both gross and net metrics in reports

3. **Document Tax Assumptions**
   - File: `README.md` or new `docs/TAX_CONSIDERATIONS.md`
   - Explain: 5-year rule, social charges, FIFO method
   - Link to: Official French tax resources (impots.gouv.fr)

### Short-Term Enhancements (Sprint 6-7)

4. **Build Withdrawal Tax Calculator**
   - New module: `src/portfolio/tax.py`
   - Function: `calculate_withdrawal_tax(amount, as_of_date)`
   - Returns: Tax breakdown, net proceeds, PEA status impact

5. **Track Total Contributions**
   - Add: `total_contributions` field to PEA metadata
   - Function: `deposit_cash()` updates contributions counter
   - Alert: Warn when approaching 150,000 EUR ceiling

6. **Create Tax Report Export**
   - Function: `export_tax_report(year, format='csv')`
   - Format: Compatible with French tax declaration
   - Include: Withdrawals, gains, social charges

### Long-Term Improvements (Sprint 8+)

7. **Implement FIFO Tax-Lot Tracking**
   - Redesign: `Position` model to track individual lots
   - Add: `TaxLot` model with purchase_date, shares, cost_basis
   - Function: `calculate_fifo_capital_gain(shares_sold)`

8. **Build Withdrawal Optimizer**
   - Optimize: Which positions to sell for a given withdrawal amount
   - Consider: Tax consequences, rebalancing needs, market conditions
   - Output: Tax-optimal liquidation plan

### Documentation Requirements

9. **Tax Guide for Users** (Priority: HIGH)
   - Document: PEA tax rules in plain French
   - Include: Examples with real numbers
   - Warn: Consequences of early withdrawal
   - Location: `docs/GUIDE_FISCAL_PEA.md`

10. **Code Comment Tax Rationale** (Priority: MEDIUM)
    - Add comments explaining tax considerations
    - Example: Why 10% cash buffer (5-year rule)
    - Example: Why FIFO is mandatory (French tax law)

---

## 10. CONCLUSION

**Overall Assessment:** The Sprint 5 P0 implementation has created a SOLID foundation for tax-efficient PEA portfolio management. The conservative risk limits (30% leveraged, 10% cash buffer) are excellent choices that align with French tax optimization principles.

**Key Strengths:**
- Risk limits protect against forced liquidations before 5-year mark
- Cash buffer enables tax-efficient withdrawals
- FIFO method correctly identified as required
- Conservative allocation reduces volatility decay (preserves tax base)

**Critical Gaps:**
- No tracking of PEA opening date (blocks 5-year rule optimization)
- Social charges (17.2%) not reflected in performance metrics
- No withdrawal tax calculator (makes tax planning difficult)

**Priority Actions:**
1. Add PEA opening date tracking (CRITICAL)
2. Integrate social charges in metrics (CRITICAL)
3. Build withdrawal tax calculator (HIGH)

**Tax Alpha Potential:**
- **15,000+ EUR savings** by properly managing 5-year holding period
- **1.72% annual return improvement** by showing net-of-social-charges metrics
- **500-2,000 EUR annual savings** through optimized withdrawal timing (after 5 years)

**Compliance Status:** ACCEPTABLE for personal use, with recommended enhancements for full tax optimization and reporting compliance.

---

**Next Steps:**
1. Implement P0 recommendations (PEA date tracking, social charges integration)
2. Create comprehensive tax guide for users
3. Plan Sprint 6 tax optimization features (withdrawal calculator, contribution tracking)
4. Consider consultation with French tax advisor (avocat fiscaliste) for complex scenarios

---

**Document Version:** 1.0
**Reviewer:** David (French Tax Optimization Specialist)
**Review Date:** 2025-12-12
**Next Review:** Post-Sprint 6 (tax features implementation)

---

## APPENDIX A: French Tax Code References

### Key Articles

1. **Article 163 quinquies D du CGI** - PEA Framework
   - Defines PEA contribution ceiling (150,000 EUR)
   - Establishes 5-year holding period rule
   - Specifies tax treatment of withdrawals

2. **Article 150-0 D du CGI** - Capital Gains Calculation
   - Mandates FIFO method for securities
   - Applies to both taxable accounts and PEA withdrawals

3. **Article 200 A du CGI** - Flat Tax (PFU)
   - 30% flat tax on investment income (12.8% income tax + 17.2% social charges)
   - Applies to PEA withdrawals before 5 years
   - Option to use progressive income tax scale if more favorable

4. **CSG-CRDS (Prelevement Sociaux)**
   - Current rate: 17.2% (as of 2024)
   - Breakdown: CSG 9.2% + CRDS 0.5% + other charges 7.5%
   - Applies to ALL PEA gains upon withdrawal (even after 5 years)

### Official Resources

- **Impots.gouv.fr:** https://www.impots.gouv.fr/particulier/le-plan-depargne-en-actions-pea
- **AMF (Autorite des Marches Financiers):** https://www.amf-france.org
- **BOFiP (Bulletin Officiel des Finances Publiques):** https://bofip.impots.gouv.fr

---

## APPENDIX B: Tax Calculation Examples

### Example 1: Withdrawal Before 5 Years

**Scenario:**
- PEA opened: 2022-01-01
- Current date: 2025-06-01 (3.4 years)
- Total contributions: 100,000 EUR
- Current value: 150,000 EUR
- Unrealized gain: 50,000 EUR
- Withdrawal: 30,000 EUR (20% of portfolio)

**Tax Calculation:**
```
Proportional gain = 30,000 * (50,000 / 150,000) = 10,000 EUR

Income tax (PFU) = 10,000 * 12.8% = 1,280 EUR
Social charges = 10,000 * 17.2% = 1,720 EUR
Total tax = 3,000 EUR

Net proceeds = 30,000 - 3,000 = 27,000 EUR
Effective tax rate = 10.0% on withdrawal amount
```

**Consequence:** PEA is CLOSED permanently after this withdrawal.

### Example 2: Withdrawal After 5 Years

**Scenario:**
- PEA opened: 2019-01-01
- Current date: 2025-06-01 (6.4 years)
- Total contributions: 100,000 EUR
- Current value: 150,000 EUR
- Unrealized gain: 50,000 EUR
- Withdrawal: 30,000 EUR (20% of portfolio)

**Tax Calculation:**
```
Proportional gain = 30,000 * (50,000 / 150,000) = 10,000 EUR

Income tax = 0 EUR (exempt after 5 years)
Social charges = 10,000 * 17.2% = 1,720 EUR
Total tax = 1,720 EUR

Net proceeds = 30,000 - 1,720 = 28,280 EUR
Effective tax rate = 5.7% on withdrawal amount
```

**Tax Savings vs. Before 5 Years:** 1,280 EUR (income tax avoided)

**PEA Status:** Remains OPEN (can make additional contributions)

### Example 3: Full Liquidation After 5 Years

**Scenario:**
- PEA opened: 2018-01-01
- Current date: 2025-12-12 (7.95 years)
- Total contributions: 150,000 EUR (at ceiling)
- Current value: 300,000 EUR
- Unrealized gain: 150,000 EUR
- Withdrawal: 300,000 EUR (full liquidation)

**Tax Calculation:**
```
Total gain = 150,000 EUR

Income tax = 0 EUR (exempt after 5 years)
Social charges = 150,000 * 17.2% = 25,800 EUR
Total tax = 25,800 EUR

Net proceeds = 300,000 - 25,800 = 274,200 EUR
Effective tax rate = 8.6% on withdrawal amount
```

**Tax Comparison:**
- If withdrawn before 5 years: 150,000 * 30% = 45,000 EUR total tax
- After 5 years: 25,800 EUR total tax
- **SAVINGS: 19,200 EUR**

---

**End of Report**
