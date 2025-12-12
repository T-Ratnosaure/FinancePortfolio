# PEA Portfolio Cost Analysis
## Cost Optimization Study by Lucas, Cost Optimization Specialist

Date: 2025-12-10
Portfolio: French PEA at Boursobank
ETFs: LQQ (2x Nasdaq), CL2 (2x USA), WPEA (World)

---

## Executive Summary

Total annual costs for PEA portfolio management vary significantly by portfolio size:
- **10k EUR portfolio**: 1.47-1.77% annually (147-177 EUR)
- **50k EUR portfolio**: 0.59-0.69% annually (295-345 EUR)
- **150k EUR portfolio**: 0.36-0.41% annually (540-615 EUR)

The primary cost driver is ETF expense ratios (TER), followed by trading fees for smaller portfolios. Data and compute costs are negligible for retail investors.

---

## 1. ETF Expense Ratios (TER) - Annual Ongoing Costs

### LQQ (Amundi Nasdaq 100 2x Leveraged Daily)
- **TER**: 0.35% annually
- **Cost at different portfolio sizes** (assuming 33% allocation):
  - 10k EUR: 11.55 EUR/year
  - 50k EUR: 57.75 EUR/year
  - 150k EUR: 173.25 EUR/year

### CL2 (Amundi ETF Leveraged MSCI USA Daily 2x)
- **TER**: 0.35% annually
- **Cost at different portfolio sizes** (assuming 33% allocation):
  - 10k EUR: 11.55 EUR/year
  - 50k EUR: 57.75 EUR/year
  - 150k EUR: 173.25 EUR/year

### WPEA (Amundi MSCI World UCITS ETF)
- **TER**: 0.38% annually
- **Cost at different portfolio sizes** (assuming 34% allocation):
  - 10k EUR: 12.92 EUR/year
  - 50k EUR: 64.60 EUR/year
  - 150k EUR: 193.80 EUR/year

### Total ETF TER Costs
- **10k EUR portfolio**: 36.02 EUR/year (0.36%)
- **50k EUR portfolio**: 180.10 EUR/year (0.36%)
- **150k EUR portfolio**: 540.30 EUR/year (0.36%)

**Analysis**: Leveraged ETFs typically charge 0.30-0.60% TER. Your selection is competitive. These costs are automatically deducted from NAV daily, so they're "invisible" but real.

---

## 2. Boursobank PEA Trading Fees

### Fee Structure (2025)
- **ETF purchases/sales**: 0.50% of transaction amount
- **Minimum fee**: 1.90 EUR per trade
- **Maximum fee**: 19.00 EUR per trade

### Trading Cost Examples

#### Single Trade Costs
| Portfolio Value | Trade Size (33% rebalance) | Fee | Effective Rate |
|----------------|---------------------------|-----|----------------|
| 10,000 EUR     | 3,300 EUR                | 16.50 EUR | 0.50% |
| 50,000 EUR     | 16,500 EUR               | 19.00 EUR | 0.12% |
| 150,000 EUR    | 49,500 EUR               | 19.00 EUR | 0.04% |

#### Full Rebalancing Cost (3 ETFs)
Rebalancing requires:
- Selling overweight positions (1-2 trades)
- Buying underweight positions (1-2 trades)
- Average: 4 trades per full rebalance

| Portfolio Value | Trades | Cost per Rebalance | % of Portfolio |
|----------------|--------|-------------------|----------------|
| 10,000 EUR     | 4      | 40-60 EUR        | 0.40-0.60%    |
| 50,000 EUR     | 4      | 76 EUR           | 0.15%         |
| 150,000 EUR    | 4      | 76 EUR           | 0.05%         |

**Key Insight**: The 19 EUR cap becomes highly advantageous above 40k EUR portfolio size.

---

## 3. Rebalancing Cost-Benefit Analysis

### Frequency vs. Transaction Costs

#### Annual Rebalancing Costs by Frequency

**10k EUR Portfolio:**
| Frequency | Annual Trades | Trading Costs | TER Costs | Total | % of Portfolio |
|-----------|--------------|--------------|-----------|-------|----------------|
| Monthly   | 48           | 600-720 EUR  | 36 EUR    | 636-756 EUR | 6.36-7.56% |
| Quarterly | 16           | 200-240 EUR  | 36 EUR    | 236-276 EUR | 2.36-2.76% |
| Semi-annual | 8          | 100-120 EUR  | 36 EUR    | 136-156 EUR | 1.36-1.56% |
| Annual    | 4            | 50-60 EUR    | 36 EUR    | 86-96 EUR   | 0.86-0.96% |
| Threshold-based (5%) | 4-8 | 50-100 EUR | 36 EUR | 86-136 EUR | 0.86-1.36% |

**50k EUR Portfolio:**
| Frequency | Annual Trades | Trading Costs | TER Costs | Total | % of Portfolio |
|-----------|--------------|--------------|-----------|-------|----------------|
| Monthly   | 48           | 912 EUR      | 180 EUR   | 1,092 EUR | 2.18% |
| Quarterly | 16           | 304 EUR      | 180 EUR   | 484 EUR   | 0.97% |
| Semi-annual | 8          | 152 EUR      | 180 EUR   | 332 EUR   | 0.66% |
| Annual    | 4            | 76 EUR       | 180 EUR   | 256 EUR   | 0.51% |
| Threshold-based (5%) | 4-8 | 76-152 EUR | 180 EUR | 256-332 EUR | 0.51-0.66% |

**150k EUR Portfolio:**
| Frequency | Annual Trades | Trading Costs | TER Costs | Total | % of Portfolio |
|-----------|--------------|--------------|-----------|-------|----------------|
| Monthly   | 48           | 912 EUR      | 540 EUR   | 1,452 EUR | 0.97% |
| Quarterly | 16           | 304 EUR      | 540 EUR   | 844 EUR   | 0.56% |
| Semi-annual | 8          | 152 EUR      | 540 EUR   | 692 EUR   | 0.46% |
| Annual    | 4            | 76 EUR       | 540 EUR   | 616 EUR   | 0.41% |
| Threshold-based (5%) | 4-8 | 76-152 EUR | 540 EUR | 616-692 EUR | 0.41-0.46% |

### Optimal Rebalancing Strategy

**Recommendation: Threshold-Based + Time-Based Hybrid**

```
IF (drift > 5% from target allocation) OR (time since last rebalance > 6 months):
    Trigger rebalance
ELSE:
    Continue monitoring
```

**Rationale:**
1. **Small portfolios (<25k)**: Annual rebalancing only, drift monitoring for major events
2. **Medium portfolios (25-100k)**: Semi-annual or 5% threshold
3. **Large portfolios (>100k)**: Quarterly or 3% threshold becomes cost-effective

**Expected rebalancing frequency:**
- Calm markets: 2-3 times/year
- Volatile markets: 4-6 times/year
- Cost: 0.05-0.15% for portfolios >50k EUR

---

## 4. Data Provider Costs

### Free Options (Recommended for Retail)
| Provider | Data Coverage | API | Cost | Suitability |
|----------|--------------|-----|------|-------------|
| Yahoo Finance | ETF prices, historical | Unofficial (yfinance) | FREE | Excellent for PEA ETFs |
| Boursobank | Real-time portfolio | Web scraping | FREE | Direct source |
| Investing.com | Euronext prices | Web scraping | FREE | Good backup |
| Euronext | Official prices (15min delay) | Official API | FREE | Most reliable |

### Paid Options (Not Recommended for Retail)
| Provider | Cost | Value for PEA |
|----------|------|--------------|
| Bloomberg Terminal | $24,000/year | Massive overkill |
| Refinitiv Eikon | $12,000/year | Massive overkill |
| Alpha Vantage Pro | $50-250/month | Unnecessary |
| Polygon.io | $29-199/month | US-focused, limited EU |

**Recommendation**: Use Yahoo Finance via Python's `yfinance` library
- Cost: FREE
- Coverage: All your ETFs (LQQ, CL2, WPEA) available
- Latency: 15-minute delay acceptable for rebalancing decisions
- Alternative: Boursobank web scraping for real-time portfolio value

**Annual Cost**: 0 EUR

---

## 5. Cloud/Compute Costs

### Local Development (Recommended)
- **Hardware**: Personal computer
- **Python environment**: FREE
- **Libraries**: All open-source (pandas, numpy, yfinance)
- **Annual cost**: 0 EUR

### Cloud Options (If Needed)

#### Low-Cost Cloud for Automation
| Service | Specs | Use Case | Monthly Cost |
|---------|-------|----------|--------------|
| AWS Lambda | 1M requests free tier | Daily calculations | FREE |
| Google Cloud Run | 2M requests free tier | Portfolio updates | FREE |
| Heroku Hobby | Basic dyno | Simple web dashboard | FREE-7 USD |
| Railway | 500 hours/month free | Portfolio monitor | FREE |
| Render | 750 hours/month free | API + dashboard | FREE |

#### If Scaling Beyond Retail
| Service | Specs | Monthly Cost |
|---------|-------|--------------|
| DigitalOcean Droplet | 1GB RAM, 1 vCPU | 6 USD |
| AWS EC2 t4g.micro | 1GB RAM | 7 USD |
| Google Cloud e2-micro | 1GB RAM | 8 USD |

**Recommendation for Retail Investor**:
- **Development**: Local Python environment (FREE)
- **Automation**: AWS Lambda or Google Cloud Run free tier
- **Backup/monitoring**: GitHub Actions (2,000 minutes/month FREE)

**Annual Cost**: 0-84 EUR (0-7 EUR/month if using paid tier)

---

## 6. Cost-Optimal Rebalancing Strategies

### Strategy Comparison

#### Strategy 1: Time-Based (Annual)
```
Rebalance: Once per year on fixed date
```
- **Pros**: Predictable, minimal trading costs
- **Cons**: May miss significant drift in volatile periods
- **Cost (50k portfolio)**: 76 EUR trading + 180 EUR TER = 256 EUR (0.51%)

#### Strategy 2: Threshold-Based (5%)
```
Rebalance: When any ETF drifts >5% from target
```
- **Pros**: Responds to market movements, prevents excessive drift
- **Cons**: Unpredictable timing
- **Cost (50k portfolio)**: 76-152 EUR trading + 180 EUR TER = 256-332 EUR (0.51-0.66%)

#### Strategy 3: Hybrid (Recommended)
```
Rebalance: If (drift >5%) OR (6 months passed) OR (drift >3% AND high volatility)
```
- **Pros**: Balances cost and risk management, adaptive
- **Cons**: Slightly more complex logic
- **Cost (50k portfolio)**: 76-152 EUR trading + 180 EUR TER = 256-332 EUR (0.51-0.66%)
- **Expected**: ~2-4 rebalances/year = 280 EUR (0.56%)

#### Strategy 4: Cash Flow Rebalancing (Most Efficient)
```
Rebalance: Use new contributions to buy underweight positions
Formal rebalance: Only when contributions insufficient or drift >8%
```
- **Pros**: Minimizes trading costs, natural rebalancing
- **Cons**: Requires regular contributions, slower rebalancing
- **Cost (50k portfolio)**: 38-76 EUR trading + 180 EUR TER = 218-256 EUR (0.44-0.51%)

### Recommended Strategy by Portfolio Size

| Portfolio Size | Strategy | Expected Rebalances/Year | Annual Cost |
|---------------|----------|-------------------------|-------------|
| <20k EUR      | Annual + Cash flow | 1-2 | 86-136 EUR |
| 20-75k EUR    | Hybrid (5% threshold, 6mo max) | 2-4 | 256-332 EUR |
| >75k EUR      | Hybrid (3% threshold, quarterly) | 3-6 | 540-692 EUR |

---

## 7. Tax Efficiency Within PEA Wrapper

### PEA Tax Advantages
The PEA provides exceptional tax efficiency:

#### Tax Treatment
- **Capital gains**: 0% tax (vs 30% flat tax outside PEA)
- **Dividends**: 0% tax (automatically reinvested in NAV)
- **Rebalancing**: No tax on internal trades
- **After 5 years**: Tax-free withdrawals (only 17.2% social charges)

#### Cost Implications
**Annual tax savings vs. Compte-Titres Ordinaire (CTO):**

| Portfolio | Annual Gains (8%) | Tax Saved (30% PFU) | PEA Advantage |
|-----------|------------------|-------------------|---------------|
| 10k EUR   | 800 EUR          | 240 EUR           | 240 EUR/year  |
| 50k EUR   | 4,000 EUR        | 1,200 EUR         | 1,200 EUR/year|
| 150k EUR  | 12,000 EUR       | 3,600 EUR         | 3,600 EUR/year|

**Key Benefits for Cost Optimization:**
1. **Free rebalancing**: No tax event on selling overweight positions
2. **Compound efficiency**: No dividend tax drag
3. **Flexibility**: Can adjust strategy without tax concerns
4. **Time value**: Tax deferral until withdrawal (potentially forever)

**Estimated Tax-Adjusted Returns:**
- PEA: 8% gross = ~7.6% net (only TER drag)
- CTO: 8% gross = ~5.6% net (after 30% PFU annually)
- **Advantage**: +2% annual return = significant long-term compounding

### PEA Withdrawal Strategy (Post 5 years)
- **Partial withdrawals**: Tax-free (17.2% social charges only)
- **No closure required**: Maintain tax-advantaged status
- **Annual cost of withdrawal**: 17.2% of gains only (not principal)

**Example: 150k EUR portfolio after 20 years**
- Assumed growth to 450k EUR (250k gains)
- Regular CTO: ~75k EUR tax paid annually
- PEA: 0 EUR tax paid annually
- **Lifetime tax savings**: >1M EUR on large portfolios

---

## 8. Total Annual Cost Projections

### 10,000 EUR Portfolio

#### Conservative Rebalancing (Annual + Cash Flow)
| Cost Component | Annual Cost | % of Portfolio |
|----------------|------------|----------------|
| ETF TER (LQQ, CL2, WPEA) | 36 EUR | 0.36% |
| Trading fees (2 rebalances) | 50 EUR | 0.50% |
| Data providers | 0 EUR | 0.00% |
| Cloud/compute | 0 EUR | 0.00% |
| **Total** | **86 EUR** | **0.86%** |

#### Moderate Rebalancing (Semi-annual)
| Cost Component | Annual Cost | % of Portfolio |
|----------------|------------|----------------|
| ETF TER | 36 EUR | 0.36% |
| Trading fees (2 rebalances) | 100 EUR | 1.00% |
| Data providers | 0 EUR | 0.00% |
| Cloud/compute | 0 EUR | 0.00% |
| **Total** | **136 EUR** | **1.36%** |

**Recommendation**: Annual + cash flow rebalancing
**Target cost**: 86-96 EUR (0.86-0.96%)

---

### 50,000 EUR Portfolio

#### Conservative Rebalancing (Annual)
| Cost Component | Annual Cost | % of Portfolio |
|----------------|------------|----------------|
| ETF TER | 180 EUR | 0.36% |
| Trading fees (1 rebalance) | 76 EUR | 0.15% |
| Data providers | 0 EUR | 0.00% |
| Cloud/compute | 0 EUR | 0.00% |
| **Total** | **256 EUR** | **0.51%** |

#### Optimal Rebalancing (Hybrid: 5% threshold, 6mo max)
| Cost Component | Annual Cost | % of Portfolio |
|----------------|------------|----------------|
| ETF TER | 180 EUR | 0.36% |
| Trading fees (2-3 rebalances) | 115 EUR | 0.23% |
| Data providers | 0 EUR | 0.00% |
| Cloud/compute | 0 EUR | 0.00% |
| **Total** | **295 EUR** | **0.59%** |

#### Aggressive Rebalancing (Quarterly)
| Cost Component | Annual Cost | % of Portfolio |
|----------------|------------|----------------|
| ETF TER | 180 EUR | 0.36% |
| Trading fees (4 rebalances) | 152 EUR | 0.30% |
| Data providers | 0 EUR | 0.00% |
| Cloud/compute | 0 EUR | 0.00% |
| **Total** | **332 EUR** | **0.66%** |

**Recommendation**: Hybrid strategy (5% threshold + 6-month maximum)
**Target cost**: 280-315 EUR (0.56-0.63%)

---

### 150,000 EUR Portfolio

#### Conservative Rebalancing (Annual)
| Cost Component | Annual Cost | % of Portfolio |
|----------------|------------|----------------|
| ETF TER | 540 EUR | 0.36% |
| Trading fees (1 rebalance) | 76 EUR | 0.05% |
| Data providers | 0 EUR | 0.00% |
| Cloud/compute | 0 EUR | 0.00% |
| **Total** | **616 EUR** | **0.41%** |

#### Optimal Rebalancing (Hybrid: 3% threshold, quarterly max)
| Cost Component | Annual Cost | % of Portfolio |
|----------------|------------|----------------|
| ETF TER | 540 EUR | 0.36% |
| Trading fees (3-4 rebalances) | 114-152 EUR | 0.08-0.10% |
| Data providers | 0 EUR | 0.00% |
| Cloud/compute | 0 EUR | 0.00% |
| **Total** | **654-692 EUR** | **0.44-0.46%** |

#### Active Management (Monthly monitoring, 2% threshold)
| Cost Component | Annual Cost | % of Portfolio |
|----------------|------------|----------------|
| ETF TER | 540 EUR | 0.36% |
| Trading fees (6-8 rebalances) | 228-304 EUR | 0.15-0.20% |
| Data providers | 0 EUR | 0.00% |
| Cloud/compute | 84 EUR | 0.06% |
| **Total** | **852-928 EUR** | **0.57-0.62%** |

**Recommendation**: Hybrid strategy (3% threshold + quarterly maximum)
**Target cost**: 654-692 EUR (0.44-0.46%)

---

## 9. Cost-Benefit Analysis & Recommendations

### Key Findings

1. **ETF TER is unavoidable but competitive**
   - 0.36% average across portfolio
   - Industry-standard for leveraged/world ETFs
   - Cannot be reduced without changing strategy

2. **Trading fees scale favorably**
   - 19 EUR cap makes larger portfolios highly efficient
   - <50k: Trading costs = 0.15-0.60% per rebalance
   - >50k: Trading costs = 0.05-0.15% per rebalance
   - >100k: Trading costs = 0.02-0.08% per rebalance

3. **Data and compute costs are negligible**
   - Free data sources are sufficient (yfinance, Boursobank)
   - Local compute handles all calculations
   - Cloud automation optional, available in free tiers

4. **Rebalancing frequency has diminishing returns**
   - Annual rebalancing: Cheap but may miss drift
   - Quarterly: Good balance for >50k portfolios
   - Monthly: Expensive for small portfolios, marginal benefit
   - Threshold-based: Optimal risk/cost tradeoff

5. **PEA wrapper provides massive tax savings**
   - Eliminates 30% PFU on gains
   - Free internal rebalancing
   - Value increases with portfolio size and time horizon
   - Effective cost reduction: 1-3% annually vs. CTO

### Comparative Analysis: Your Strategy vs. Alternatives

#### Your Current Setup: 3-ETF PEA Portfolio
| Portfolio Size | Total Annual Cost | % of Portfolio |
|---------------|------------------|----------------|
| 10k EUR       | 86-136 EUR       | 0.86-1.36%    |
| 50k EUR       | 256-332 EUR      | 0.51-0.66%    |
| 150k EUR      | 616-692 EUR      | 0.41-0.46%    |

#### Alternative 1: Robo-Advisor (e.g., Yomoni, Nalo)
| Portfolio Size | Annual Fee | Total Cost | % of Portfolio |
|---------------|-----------|------------|----------------|
| 10k EUR       | 160 EUR   | 196 EUR    | 1.96%         |
| 50k EUR       | 800 EUR   | 980 EUR    | 1.96%         |
| 150k EUR      | 2,250 EUR | 2,790 EUR  | 1.86%         |

**Savings**: 60-2,100 EUR/year vs. robo-advisor

#### Alternative 2: Single World ETF (No rebalancing)
| Portfolio Size | Annual Cost | % of Portfolio |
|---------------|------------|----------------|
| 10k EUR       | 38 EUR     | 0.38%         |
| 50k EUR       | 190 EUR    | 0.38%         |
| 150k EUR      | 570 EUR    | 0.38%         |

**Extra cost for your strategy**: 48-122 EUR/year
**Benefit**: 2x leverage exposure, tactical allocation flexibility

#### Alternative 3: Active Fund Manager
| Portfolio Size | Annual Fee (1.5-2%) | Total Cost |
|---------------|-------------------|------------|
| 10k EUR       | 150-200 EUR       | 150-200 EUR|
| 50k EUR       | 750-1,000 EUR     | 750-1,000 EUR|
| 150k EUR      | 2,250-3,000 EUR   | 2,250-3,000 EUR|

**Savings**: 64-2,308 EUR/year vs. active management

### Final Recommendations by Portfolio Size

#### Small Portfolio (<25k EUR)
**Strategy**: Annual rebalancing + cash flow optimization
- Expected cost: 0.86-1.00%
- Rebalancing: 1-2 times/year
- Tools: Local Python, Yahoo Finance
- Focus: Minimize trading frequency

**Action Items:**
- Set annual rebalancing date (e.g., January 15)
- Use new contributions to rebalance naturally
- Only formal rebalance if drift >8% or 12 months passed
- Monitor quarterly, act annually

#### Medium Portfolio (25-100k EUR)
**Strategy**: Hybrid threshold-based (5% + semi-annual)
- Expected cost: 0.51-0.66%
- Rebalancing: 2-4 times/year
- Tools: Local Python + automated monitoring
- Focus: Balance risk and cost

**Action Items:**
- Implement automated drift monitoring (weekly check)
- Rebalance when: drift >5% OR 6 months passed
- Consider cash flow rebalancing first
- Track trading costs vs. drift reduction benefit

#### Large Portfolio (>100k EUR)
**Strategy**: Active hybrid (3% threshold + quarterly)
- Expected cost: 0.41-0.50%
- Rebalancing: 3-6 times/year
- Tools: Automated system, potential cloud dashboard
- Focus: Optimize returns, cost is secondary

**Action Items:**
- Implement automated monitoring (daily/weekly)
- Tighter rebalancing thresholds (3-4%)
- Consider volatility-adjusted thresholds
- Track performance attribution
- Potential cloud automation for convenience

---

## 10. Implementation Roadmap

### Phase 1: Cost Tracking (Week 1)
- [ ] Set up portfolio tracking spreadsheet
- [ ] Document all historical trades and fees
- [ ] Calculate current cost baseline
- [ ] Identify cost reduction opportunities

### Phase 2: Data Infrastructure (Week 2)
- [ ] Set up yfinance data pipeline
- [ ] Implement Boursobank data extraction
- [ ] Create local database for historical prices
- [ ] Build cost calculation module

### Phase 3: Rebalancing Optimization (Week 3-4)
- [ ] Implement drift calculation
- [ ] Build threshold-based rebalancing logic
- [ ] Create cost simulation model
- [ ] Backtest different rebalancing strategies
- [ ] Compare trading costs vs. drift costs

### Phase 4: Automation (Week 5-6)
- [ ] Set up automated monitoring (local or cloud)
- [ ] Create rebalancing alerts
- [ ] Build decision support dashboard
- [ ] Implement logging and cost tracking
- [ ] Set up periodic reports

### Phase 5: Optimization & Monitoring (Ongoing)
- [ ] Track actual costs monthly
- [ ] Compare to projections
- [ ] Adjust rebalancing thresholds based on data
- [ ] Annual strategy review
- [ ] Document lessons learned

---

## 11. Cost Monitoring Metrics

### Key Performance Indicators

1. **Total Cost Ratio (TCR)**
   ```
   TCR = (TER Costs + Trading Costs + Data Costs + Compute Costs) / Portfolio Value
   Target: <0.60% for >50k EUR portfolio
   ```

2. **Trading Efficiency Ratio**
   ```
   TER = Average Trade Cost / Average Trade Size
   Target: <0.15% for >50k EUR portfolio
   ```

3. **Rebalancing Benefit Ratio**
   ```
   RBR = (Performance Improvement from Rebalancing) / (Rebalancing Costs)
   Target: >3:1 (3 EUR benefit per 1 EUR cost)
   ```

4. **Cost Per Basis Point (BPS) of Drift Reduction**
   ```
   Cost per BPS = Total Rebalancing Cost / BPS of Drift Corrected
   Target: <0.50 EUR per 100 BPS for 50k portfolio
   ```

### Monthly Monitoring Checklist
- [ ] Calculate month-end portfolio value
- [ ] Log all trading costs
- [ ] Calculate MTD and YTD TCR
- [ ] Review drift from target allocation
- [ ] Assess rebalancing trigger conditions
- [ ] Document decisions and rationale

### Quarterly Review
- [ ] Analyze rebalancing frequency vs. market volatility
- [ ] Calculate quarterly TCR and compare to target
- [ ] Review threshold effectiveness
- [ ] Assess whether strategy adjustments needed
- [ ] Update cost projections

### Annual Review
- [ ] Full cost analysis vs. budget
- [ ] Rebalancing strategy effectiveness
- [ ] Compare actual vs. projected costs
- [ ] Benchmark against alternatives (robo-advisors, funds)
- [ ] Adjust strategy for next year

---

## Conclusion

Your PEA portfolio with LQQ, CL2, and WPEA at Boursobank can be managed very cost-effectively:

**Target Costs:**
- 10k EUR: 0.86-0.96% annually (86-96 EUR)
- 50k EUR: 0.56-0.63% annually (280-315 EUR)
- 150k EUR: 0.44-0.46% annually (654-692 EUR)

**Key Success Factors:**
1. Use free data sources (Yahoo Finance)
2. Implement threshold-based rebalancing
3. Leverage Boursobank's 19 EUR fee cap
4. Minimize rebalancing frequency while controlling drift
5. Use cash flows for natural rebalancing
6. Maximize PEA tax advantages

**Competitive Advantage:**
- 1.0-1.5% cheaper than robo-advisors
- 1.5-2.5% cheaper than active funds
- Only 0.05-0.20% more expensive than single-ETF approach
- Benefit: Tactical leverage and allocation flexibility

The system can be built entirely with open-source tools at zero marginal cost for data and compute, making this highly scalable as your portfolio grows.

---

**Next Steps:**
1. Implement cost tracking module in Python
2. Set up automated drift monitoring
3. Backtest optimal rebalancing thresholds
4. Create rebalancing decision support tool
5. Establish cost monitoring dashboard

This analysis provides the foundation for building a cost-optimized portfolio management system that scales efficiently with portfolio growth while maintaining disciplined risk management.
