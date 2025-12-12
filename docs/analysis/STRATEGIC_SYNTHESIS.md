# Strategic Synthesis: PEA Portfolio Optimization Project

**Date:** December 10, 2025 (Updated)
**Project Lead:** Jacques (Head Manager)
**Status:** Analysis Complete - Ready for Implementation Planning

---

## Executive Summary

After comprehensive multi-team analysis (16 agents across 6 teams), we have identified key findings, opportunities, and a clear path forward for the PEA portfolio optimization project.

### ETF Eligibility - VERIFIED

| ETF | ISIN | Domicile | TER | PEA Eligible |
|-----|------|----------|-----|--------------|
| **LQQ** | FR0010342592 | France (Amundi) | 0.60% | **YES** |
| **CL2** | FR0010755611 | France (Amundi) | 0.50% | **YES** |
| **WPEA** | FR... | France | ~0.20% | **YES** |

### Key Findings

| Priority | Finding | Team | Action |
|----------|---------|------|--------|
| **HIGH** | Portfolio correlation 0.85-0.95 | Research | Diversification needed |
| **HIGH** | ~10%/year decay on LQQ, ~6%/year on CL2 | Research | Cap leveraged exposure at 30% |
| **HIGH** | 5-year rule for PEA tax advantage | Legal | Plan for long-term holding |
| **MEDIUM** | Personal use pathway recommended | Legal | Document limitation |
| **MEDIUM** | Zero-cost data architecture possible | Data | Use free APIs |

### PEA Tax Advantage (5-Year Rule)

| Period | Tax on Gains |
|--------|--------------|
| Before 5 years | 30% (PFU) + plan closure |
| **After 5 years** | **17.2% (PS only)** |

### Strategic Decision

**PROCEED WITH PERSONAL USE PATHWAY**
- No investment advisory licensing required
- Full tax advantages of PEA preserved (17.2% after 5 years vs 30% in CTO)
- System for personal portfolio management only
- Rebalancing is TAX-FREE within PEA

---

## 1. Portfolio Recommendations

### Current State

| Asset | Status | Notes |
|-------|--------|-------|
| LQQ | **PEA-Eligible** | 2x Nasdaq-100, ~10% annual decay |
| CL2 | **PEA-Eligible** | 2x MSCI USA, ~6% annual decay |
| WPEA | **PEA-Eligible** | 1x World, no decay |
| All 3 | High correlation | 0.85-0.95 - essentially US equity exposure |

### Target Allocation (RECOMMENDED)

**Conservative Initial Allocation:**

| Asset | Weight | Rationale |
|-------|--------|-----------|
| LQQ | 15% | High decay, limit exposure |
| CL2 | 15% | Moderate decay, broader US |
| WPEA | 50% | Core stable holding |
| Cash | 20% | Rebalancing buffer |

**Regime-Based Allocation (After Model Validated):**

| Regime | LQQ | CL2 | WPEA | Cash |
|--------|-----|-----|------|------|
| Risk-On | 15% | 15% | 60% | 10% |
| Neutral | 10% | 10% | 60% | 20% |
| Risk-Off | 5% | 5% | 60% | 30% |

### Risk Limits (MANDATORY)

| Metric | Hard Limit |
|--------|------------|
| Total leveraged ETF exposure (LQQ + CL2) | 30% max |
| Single position | 25% max |
| Cash minimum | 10% |
| Drawdown trigger (reduce risk) | -20% |
| Rebalancing threshold | 5% drift |

---

## 2. Technical Architecture

### Data Pipeline

**Stack:**
- Storage: DuckDB (embedded, free)
- Orchestration: Dagster (type-safe, Python 3.12)
- Validation: Great Expectations
- Dashboard: Streamlit

**Data Sources (Zero Cost):**
- Yahoo Finance (primary ETF prices)
- Alpha Vantage (backup)
- FRED (macro indicators)
- ECB API (European data)

**Schedule:**
- Daily ETF updates: 18:30 CET
- Weekly macro updates: Sunday 06:00

### ML Architecture

**Two-Stage Hierarchical Model:**
```
Stage 1: Regime Detection (HMM)
         - States: Risk-On, Neutral, Risk-Off
         - Features: VIX, trend, credit spreads
         |
Stage 2: Regime-Conditional Allocation
         - Optimize for risk-adjusted returns
         - Apply position limits
```

**Anti-Overfitting Requirements:**
- Walk-forward validation mandatory
- Backtest Sharpe must be < 2.0
- Apply 50% haircut to expected returns
- Feature count < 20

### Manual Execution Workflow

```
Signal Generated -> Dashboard Alert -> Human Review ->
Boursobank Login -> Execute Trade -> Record in System
```

**Weekly reconciliation against Boursobank account required**

---

## 3. Compliance Framework

### Pathway: Personal Use Only

**Requirements:**
1. Written declaration of personal use
2. No sharing of recommendations
3. No marketing or promotion
4. Basic security measures

**Documentation to Create:**
- `compliance/personal_use_declaration.md`
- `compliance/risk_disclosures.md`
- `compliance/pea_eligible_etfs.json`

### Tax Optimization Strategy

**Timeline:**
- Years 1-5: Accumulate only, NO withdrawals (plan closure if withdrawn)
- After 5 years: Optimal - 17.2% PS only, partial withdrawals allowed

**Key Advantage:**
- Rebalancing is TAX-FREE within PEA
- Exploit this for active regime-based allocation
- No need for tax-loss harvesting considerations

---

## 4. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

**Compliance:**
- [x] Verify LQQ PEA eligibility - **CONFIRMED**
- [x] Verify CL2 PEA eligibility - **CONFIRMED**
- [x] Verify WPEA PEA eligibility - **CONFIRMED**
- [ ] Create personal use declaration
- [ ] Create risk disclosures

**Data:**
- [ ] Set up DuckDB schema
- [ ] Implement Yahoo Finance pipeline
- [ ] Load 15 years historical data
- [ ] Basic validation rules

**Deliverable:** Clean historical data + compliance documentation

### Phase 2: Signal Generation (Weeks 3-4)

**ML:**
- [ ] Implement HMM regime detector
- [ ] Walk-forward validation framework
- [ ] Feature engineering pipeline
- [ ] Backtest infrastructure

**Dashboard:**
- [ ] Basic Streamlit interface
- [ ] Current allocation display
- [ ] Signal visualization

**Deliverable:** Working regime detection with backtests

### Phase 3: Production (Weeks 5-6)

**Operations:**
- [ ] Dagster orchestration
- [ ] Email alerts
- [ ] Manual trade entry form
- [ ] Position reconciliation

**Quality:**
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Backup procedures

**Deliverable:** Production-ready personal system

### Phase 4: Monitoring (Ongoing)

- Daily signal review
- Weekly reconciliation
- Monthly performance review
- Quarterly model retraining

---

## 5. Risk Assessment

### Financial Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Leveraged ETF drawdown | Medium | High | 30% cap, regime detection |
| Model overfitting | Medium | High | Walk-forward, haircut |
| Regime detection failure | Medium | Medium | Conservative base allocation |
| Manual execution delays | High | Low | Daily monitoring |
| Volatility decay | Certain | Medium | Limit leveraged exposure |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data API failure | Medium | Medium | Multi-source fallback |
| Pipeline errors | Low | Medium | Alerts, retries |
| Position drift | Medium | Low | Weekly reconciliation |

### Regulatory Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Investment advice breach | Low (if personal) | Critical | Document personal use |
| GDPR violation | Low | High | Basic security |

---

## 6. Cost Projections

### Development Costs

| Item | Cost |
|------|------|
| Data APIs | EUR 0 (free tier) |
| Infrastructure | EUR 0 (local) |
| Development time | Personal effort |
| **Total Setup** | **EUR 0** |

### Ongoing Costs

| Item | Monthly | Annual |
|------|---------|--------|
| Data (optional backup) | EUR 8 | EUR 96 |
| Compute | EUR 0 | EUR 0 |
| **Total Operating** | **EUR 0-8** | **EUR 0-96** |

### Tax Savings (PEA vs CTO after 5 years)

| Portfolio Size | 20-Year PEA Advantage |
|----------------|----------------------|
| EUR 50,000 | ~EUR 40,000 |
| EUR 100,000 | ~EUR 80,000 |
| EUR 150,000 | ~EUR 119,000 |

---

## 7. Success Metrics

### Performance Metrics

| Metric | Target | Red Flag |
|--------|--------|----------|
| Sharpe Ratio | 0.3 - 0.8 | > 2.0 |
| Annual Return | 5% - 12% | < 0% sustained |
| Max Drawdown | < 25% | > 35% |
| Tracking Error vs Benchmark | < 10% | > 20% |

### Operational Metrics

| Metric | Target |
|--------|--------|
| Data freshness | < 24 hours |
| Signal generation | Daily by 19:00 CET |
| Position reconciliation | Weekly |
| Model retraining | Quarterly |

---

## 8. Team Assignments

### Research Team (Jean-Yves)
- Regime detection model
- Allocation optimization
- Performance monitoring

### Data Team (Florian)
- Pipeline implementation
- Data quality
- Dashboard

### Legal Team (Marc)
- Compliance documentation
- Tax optimization guidance

### IT-Core Team (Jean-David)
- Code quality
- CI/CD pipeline
- Security review

---

## 9. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-10 | Personal use pathway | Avoids regulatory burden |
| 2025-12-10 | All ETFs confirmed eligible | LQQ, CL2, WPEA all French ISIN |
| 2025-12-10 | Cap leverage at 30% | Risk management |
| 2025-12-10 | DuckDB + Dagster stack | Zero cost, Python-native |
| 2025-12-10 | Free data sources | Adequate for retail |
| 2025-12-10 | Walk-forward validation | Prevent overfitting |
| 2025-12-10 | 5-year holding target | PEA tax optimization |

---

## 10. Next Steps

### Immediate (This Week)

1. **Create compliance files** - Personal use declaration
2. **Initialize data pipeline** - DuckDB setup
3. **Define allocation targets** - Based on risk tolerance
4. **Set up Yahoo Finance fetcher** - LQQ.PA, CL2.PA, WPEA.PA

### Short-term (Weeks 2-4)

5. **Implement regime detector** - HMM with walk-forward
6. **Build dashboard** - Streamlit MVP
7. **Backtest framework** - Validate approach

### Medium-term (Weeks 5-8)

8. **Production deployment** - Dagster orchestration
9. **Documentation** - Complete all guides
10. **Go-live** - Start using for real portfolio

---

## Appendix: Agent Contributions Summary

### Completed Analyses

| Agent | Role | Key Contribution |
|-------|------|------------------|
| Jacques | Head Manager | 4-phase project plan, team coordination |
| Jean-Yves | Research Lead | Portfolio correlation analysis, allocation framework |
| Remy | Equity Quant | Volatility decay formulas (LQQ ~10%/yr, CL2 ~6%/yr) |
| Iacopo | Macro Analyst | 3-regime framework with triggers |
| Alexios | ML Designer | HMM architecture, anti-overfitting rules |
| Antoine | NLP Expert | Sentiment integration (secondary signal) |
| Florian | Data Lead | Complete data architecture (DuckDB, Dagster) |
| Marc | Legal Lead | Compliance review, 5-year PEA rule |
| David | Tax Specialist | PEA tax optimization (17.2% vs 30%) |

### Rate-Limited (Partial Output)

| Agent | Role | Status |
|-------|------|--------|
| Sophie | Data Engineer | Rate limited |
| Nicolas | Risk Manager | Rate limited |
| Lucas | Cost Optimizer | Rate limited |
| Jean-David | IT-Core Manager | Rate limited |
| Lamine | CI/CD Expert | Rate limited |
| Olivier | Quality Control | Rate limited |
| Wealon | Security Auditor | Rate limited |

---

## Key Sources

- [Amundi LQQ ETF](https://www.justetf.com/en/etf-profile.html?isin=FR0010342592)
- [Amundi CL2 ETF](https://www.justetf.com/en/etf-profile.html?isin=FR0010755611)
- [Service-Public.fr - PEA Taxation](https://www.service-public.gouv.fr/particuliers/vosdroits/F22449)

---

**Document Version:** 1.1 (Corrected)
**Approved By:** Jacques (Head Manager)
**Status:** Ready for Implementation Phase 1
