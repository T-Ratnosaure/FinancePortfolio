# Compliance Priorities - Sprint 5 Post-P0

**Document Type:** Compliance Action Plan
**Date:** December 12, 2025
**Status:** Sprint 5 P0 COMPLETE - Planning P1/P2

---

## Sprint 5 P0 Compliance Status: ✅ COMPLETE

**All critical compliance issues resolved:**
1. ✅ Risk limits hard-coded (30% leveraged, 10% cash)
2. ✅ Data staleness detection implemented
3. ✅ HMM sample size validation enforced

**Overall Risk Level:** HIGH → **LOW** ✅

---

## Priority 1: Medium Risk (Sprint 5 P1-P2)

### P1-01: Create DATA_SOURCE_DISCLAIMER.md

**Risk Level:** MEDIUM
**Effort:** 1 hour
**Assignee:** Legal team / Documentation
**Target:** Sprint 5 P1

**Rationale:**
Yahoo Finance data via yfinance library may violate Terms of Service. While risk is low for personal use, a disclaimer documents:
- Data source limitations
- Unofficial API status
- Personal use only restriction
- Service availability risks

**Deliverable Location:** `docs/DATA_SOURCE_DISCLAIMER.md`

**Content Requirements:**
1. Yahoo Finance data usage disclaimer
2. yfinance unofficial API warning
3. Data accuracy limitations
4. Personal use restriction
5. Alternative data sources
6. Service availability notice

**Acceptance Criteria:**
- [ ] Disclaimer covers Yahoo Finance ToS risks
- [ ] Personal use restriction clearly stated
- [ ] Alternative data sources documented
- [ ] Reviewed by legal compliance expert

**Regulatory Benefit:**
- Addresses Yahoo Finance ToS uncertainty
- Documents data quality limitations
- Supports "informed decision" defense

---

### P1-02: Document Model Performance Monitoring

**Risk Level:** MEDIUM (model governance)
**Effort:** 2 hours (planning/documentation only)
**Assignee:** Research team
**Target:** Sprint 5 P2

**Rationale:**
SR 11-7 (Model Risk Management) requires ongoing model validation. While Sprint 5 P0 implemented training-time validation, need to document:
- How model performance will be monitored
- Metrics for regime classification accuracy
- Process for retraining when performance degrades

**Deliverable Location:** `docs/MODEL_GOVERNANCE.md`

**Content Requirements:**
1. Model validation requirements (already met)
2. Ongoing monitoring approach
   - Regime classification accuracy
   - Transition probability stability
   - Out-of-sample performance
3. Retraining triggers
   - Performance below threshold
   - Market regime shift
   - Annual review
4. Model limitations documentation
5. Approval process for model changes

**Acceptance Criteria:**
- [ ] Monitoring metrics defined
- [ ] Retraining triggers specified
- [ ] Review cadence established
- [ ] Aligns with SR 11-7 principles

**Regulatory Benefit:**
- Demonstrates ongoing model validation
- Supports model risk management framework
- Provides audit trail for model governance

---

## Priority 2: Low Risk (Sprint 6)

### P2-01: PEA Contribution Tracking

**Risk Level:** LOW
**Effort:** 8 hours
**Assignee:** Data team
**Target:** Sprint 6

**Rationale:**
PEA has €150,000 contribution ceiling (excluding gains). Currently:
- Risk disclosed in documentation
- No system tracking
- User responsibility to monitor

System tracking would:
- Prevent ceiling violations
- Warn when approaching limit
- Support tax reporting

**Implementation:**

1. **Database Schema** (2 hours)
   ```sql
   CREATE TABLE raw.pea_contributions (
       contribution_id INTEGER PRIMARY KEY,
       contribution_date DATE NOT NULL,
       amount_eur DECIMAL(10,2) NOT NULL,
       contribution_type VARCHAR(20), -- 'INITIAL', 'ADDITIONAL', 'TRANSFER'
       notes TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

2. **Pydantic Model** (1 hour)
   ```python
   class PEAContribution(BaseModel):
       contribution_date: date
       amount_eur: Decimal = Field(gt=0, le=150_000)
       contribution_type: Literal["INITIAL", "ADDITIONAL", "TRANSFER"]
       notes: str | None = None
   ```

3. **Tracking Logic** (3 hours)
   - Sum total contributions
   - Calculate remaining ceiling (€150k - total)
   - Generate warnings at thresholds (80%, 90%, 95%)

4. **Integration with Portfolio Tracker** (2 hours)
   - Check contribution limit before adding funds
   - Display contribution status in dashboard

**Acceptance Criteria:**
- [ ] All contributions tracked in database
- [ ] Total contributions calculated correctly
- [ ] Warnings generated at 80%/90%/95% thresholds
- [ ] Cannot exceed €150k without override
- [ ] Integration with portfolio tracker

**Regulatory Benefit:**
- Prevents PEA contribution ceiling violations
- Supports tax reporting (Article 163 quinquies D)
- Demonstrates prudent account management

---

### P2-02: Annual Risk Disclosure Review Process

**Risk Level:** LOW (maintenance)
**Effort:** 2 hours (documentation)
**Assignee:** Legal team
**Target:** Sprint 6

**Rationale:**
Risk disclosures should be reviewed annually to ensure:
- Regulatory changes reflected
- ETF information current
- Tax rules updated
- New risks identified

**Deliverable Location:** `docs/COMPLIANCE_CALENDAR.md`

**Content Requirements:**

1. **Annual Review Checklist**
   - [ ] ETF eligibility still valid
   - [ ] Risk disclosures accurate
   - [ ] Tax rules current
   - [ ] Regulatory changes reviewed
   - [ ] Personal use declaration current

2. **Review Schedule**
   - Annual: December (before tax year end)
   - Ad-hoc: Material regulatory changes

3. **Review Sources**
   - AMF regulatory updates
   - Euronext ETF eligibility lists
   - French tax law changes (Code général des impôts)
   - ETF issuer communications

4. **Documentation Updates**
   - Version control for compliance docs
   - Changelog for each review
   - Sign-off by reviewer

**Acceptance Criteria:**
- [ ] Annual review process documented
- [ ] Checklist created
- [ ] Review sources identified
- [ ] First review scheduled (Dec 2026)

**Regulatory Benefit:**
- Ensures ongoing compliance
- Demonstrates due diligence
- Creates audit trail of reviews

---

## Priority 3: Enhancements (Future)

### P3-01: Automated Compliance Reporting

**Risk Level:** VERY LOW (quality-of-life)
**Effort:** 16+ hours
**Assignee:** Development team
**Target:** Sprint 7+

**Rationale:**
Currently, compliance status is assessed manually through code review. Automated reporting would:
- Generate periodic compliance reports
- Track risk limit breaches
- Monitor data quality metrics
- Support audit preparation

**Potential Features:**
1. Weekly compliance summary
   - Risk limit adherence (leveraged exposure, cash buffer)
   - Data freshness status
   - Model performance metrics
2. Monthly audit report
   - All decisions with supporting data
   - Risk alerts generated
   - Data quality incidents
3. Annual compliance package
   - Risk disclosure status
   - Regulatory change tracking
   - Model governance review

**Deliverable:** Compliance reporting module

**Acceptance Criteria:**
- [ ] Weekly compliance email/report
- [ ] Monthly audit report generated
- [ ] Annual package prepared
- [ ] Configurable thresholds

**Regulatory Benefit:**
- Easier audit preparation
- Proactive risk identification
- Demonstrates ongoing monitoring

---

### P3-02: PEA Age Warning System

**Risk Level:** VERY LOW
**Effort:** 4 hours
**Assignee:** Development team
**Target:** Sprint 7+

**Rationale:**
PEA 5-year rule is critical for tax optimization. System could warn:
- Approaching 5-year anniversary
- Withdrawal tax implications
- Optimal withdrawal timing

**Implementation:**

1. **PEA Opening Date Tracking**
   - Store opening date in configuration
   - Calculate age in days/years

2. **Warning Thresholds**
   - 6 months before 5 years: "Consider waiting for tax exemption"
   - 1 month before 5 years: "PEA matures in 30 days"
   - After 5 years: "Full tax exemption now available"

3. **Dashboard Integration**
   - Display PEA age prominently
   - Show days until tax exemption
   - Calculate withdrawal tax if before 5 years

**Acceptance Criteria:**
- [ ] PEA opening date configurable
- [ ] Age calculated correctly
- [ ] Warnings generated at thresholds
- [ ] Tax calculation for early withdrawal
- [ ] Dashboard display

**Regulatory Benefit:**
- Supports tax optimization (Article 163 quinquies D)
- Prevents costly early withdrawals
- Demonstrates PEA rule awareness

---

## Compliance Maintenance Tasks

### Ongoing (No Sprint Assignment)

#### Monthly Data Source Monitoring

**Task:** Check Yahoo Finance API availability
**Effort:** 15 minutes/month
**Owner:** Data team

**Activities:**
- [ ] Verify yfinance still functional
- [ ] Check for Yahoo ToS changes
- [ ] Monitor data quality issues
- [ ] Document any service interruptions

---

#### Quarterly Model Review

**Task:** Review HMM regime detector performance
**Effort:** 1 hour/quarter
**Owner:** Research team

**Activities:**
- [ ] Check regime classification accuracy
- [ ] Review transition probabilities
- [ ] Validate sample size still adequate
- [ ] Document any performance issues

---

#### Annual Compliance Review

**Task:** Comprehensive compliance assessment
**Effort:** 4 hours/year
**Owner:** Legal team
**Schedule:** Every December

**Activities:**
- [ ] Review risk_disclosures.md
- [ ] Check regulatory changes (AMF, ESMA)
- [ ] Verify ETF eligibility
- [ ] Update personal_use_declaration.md if needed
- [ ] Re-run compliance assessment
- [ ] Document findings and actions

**Next Review:** December 2026

---

## Compliance Risk Heat Map

### Current State (Post Sprint 5 P0)

```
           Impact
           │
   CRITICAL│
           │
      HIGH │
           │
    MEDIUM │  DATA_SOURCE (P1-01)
           │  MODEL_MONITORING (P1-02)
       LOW │  PEA_CONTRIB (P2-01)
           │  ANNUAL_REVIEW (P2-02)
  VERY LOW │  AUTO_REPORTING (P3-01)
           │  PEA_AGE_WARN (P3-02)
           └─────────────────────────
             VERY LOW  LOW  MEDIUM  HIGH
                    Likelihood
```

**Interpretation:**
- All HIGH/CRITICAL risks resolved in Sprint 5 P0 ✅
- Remaining risks are MEDIUM or lower
- Most remaining items are enhancements, not gaps

---

## Compliance Budget Summary

### Sprint 5 P1-P2 (Medium Priority)

| Task | Effort | Priority | Risk Reduction |
|------|--------|----------|----------------|
| DATA_SOURCE_DISCLAIMER.md | 1h | P1 | MEDIUM → LOW |
| Model Performance Docs | 2h | P2 | Governance |
| **Total P1-P2** | **3h** | - | - |

### Sprint 6 (Low Priority)

| Task | Effort | Priority | Risk Reduction |
|------|--------|----------|----------------|
| PEA Contribution Tracking | 8h | P2 | LOW → VERY LOW |
| Annual Review Process | 2h | P2 | Maintenance |
| **Total Sprint 6** | **10h** | - | - |

### Sprint 7+ (Enhancements)

| Task | Effort | Priority | Risk Reduction |
|------|--------|----------|----------------|
| Automated Compliance Reporting | 16h | P3 | Quality-of-life |
| PEA Age Warning System | 4h | P3 | Nice-to-have |
| **Total Sprint 7+** | **20h** | - | - |

**Total Compliance Effort Remaining:** 33 hours

---

## Decision Framework: When to Prioritize Compliance Work

### IMMEDIATE (Stop Other Work):
- Critical regulatory violation discovered
- Service disruption affecting compliance (e.g., Yahoo blocks API)
- AMF/tax authority inquiry

### HIGH PRIORITY (Sprint P1):
- Medium risk items identified
- Regulatory change affecting personal use
- Data source disclaimer creation

### MEDIUM PRIORITY (Sprint P2-6):
- Low risk items
- Enhancement to compliance posture
- Maintenance tasks (annual review setup)

### LOW PRIORITY (Sprint 7+):
- Very low risk items
- Quality-of-life improvements
- Nice-to-have features

---

## Success Criteria

### Sprint 5 P0: ✅ COMPLETE

- ✅ All critical compliance gaps resolved
- ✅ Risk level reduced to LOW
- ✅ Comprehensive audit trail in place

### Sprint 5 P1-P2: Target

- [ ] Data source disclaimer created
- [ ] Model governance documented
- [ ] No MEDIUM or higher compliance risks

### Sprint 6: Target

- [ ] PEA contribution tracking implemented
- [ ] Annual review process documented
- [ ] All compliance maintenance tasks scheduled

### Long-term: Maintain

- Personal use only (no third-party advice)
- Risk disclosures updated annually
- Data source monitoring monthly
- Model performance reviewed quarterly
- Full compliance audit annually

---

## Approvals

**Sprint 5 P0 Compliance Sign-Off:**
- ✅ Jose (Legal Compliance Expert) - December 12, 2025
- Status: APPROVED for personal use

**Next Compliance Review:**
- Scheduled: December 2026 (annual)
- Scope: Full regulatory compliance reassessment
- Trigger: Material system changes or regulatory updates

---

## References

**Detailed Reviews:**
- `docs/reviews/legal-sprint5-p0-compliance-review.md` (Full analysis)
- `docs/reviews/legal-sprint5-p0-executive-summary.md` (Quick reference)
- `docs/reviews/legal-post-sprint3-review.md` (Historical context)

**Compliance Documents:**
- `compliance/risk_disclosures.md` (Risk warnings)
- `compliance/personal_use_declaration.md` (Personal use attestation)
- `compliance/pea_eligible_etfs.json` (ETF registry)

**Regulatory Guidance:**
- MiFID II: Articles 16, 25, 27
- PEA: Code général des impôts Article 163 quinquies D
- SR 11-7: Model risk management guidance
- GDPR: Household exemption (Art. 2(2)(c))

---

**END OF COMPLIANCE PRIORITIES DOCUMENT**

**Version:** 1.0
**Last Updated:** December 12, 2025
**Next Update:** After Sprint 5 P2 completion or regulatory change
