# CI/CD Priority Action Plan

**Prepared by:** Lamine, CI/CD Expert
**Date:** December 11, 2025
**Status:** URGENT - Action Required

---

## Executive Summary

The FinancePortfolio CI/CD infrastructure is **FUNCTIONAL BUT INCOMPLETE**. While basic quality gates exist, critical gaps prevent production deployment:

- ❌ **No type checking in CI** - 16 type violations undetected
- ❌ **No deployment pipeline** - Manual only
- ❌ **No environment management** - Secrets at risk
- ❌ **No monitoring** - Production failures invisible

**Overall Grade: C+ (Needs Improvement)**

---

## CRITICAL: Fix This Week (P0)

### 1. Add Type Checking to CI Pipeline ✅ COMPLETED
**File:** `.github/workflows/ci.yml`
**Effort:** 15 minutes
**Impact:** Prevents 16+ type errors from reaching production
**Status:** Implemented - December 12, 2025

**Implementation:**
```yaml
# Added after line 45 (after xenon check):
- name: Type check with pyrefly
  run: uv run pyrefly check
```

**Configuration:** Added `[tool.pyrefly]` section to `pyproject.toml` for future customization.

**Why:** Type violations are now caught by CI before they can reach production.

---

### 2. Create Environment Configuration Template
**File:** `.env.example` (NEW)
**Effort:** 30 minutes
**Impact:** Prevents accidental secret commits

```bash
# Required content:
ANTHROPIC_API_KEY=sk-ant-your-key-here
FRED_API_KEY=your-fred-key-here
DATA_DIR=./data
LOG_LEVEL=INFO
```

**Why:** Addresses SEC-002 (secrets exposure risk).

---

### 3. Update .gitignore for Security
**File:** `.gitignore`
**Effort:** 5 minutes
**Impact:** CRITICAL - Prevents credential leaks

```gitignore
# Add these lines:
.env
.env.*
*.env
.envrc
secrets/
credentials/
*.key
*.pem
```

**Why:** Currently missing security patterns flagged in Executive Summary.

---

### 4. Implement Environment Validation
**File:** `config/env_validator.py` (NEW)
**Effort:** 2 hours
**Impact:** Fail-fast on configuration errors

Create Pydantic-based validator that checks:
- Required API keys present
- Paths valid
- Risk limits in acceptable ranges

**Why:** Prevent runtime failures due to missing/invalid configuration.

---

### 5. Replace print() with Structured Logging
**Files:** All modules using print()
**Effort:** 4 hours
**Impact:** Enable production monitoring

```python
# Bad (current):
print(f"Fetched {len(data)} records")

# Good (required):
logger.info("Fetched %d records", len(data))
```

**Why:** DATA-002 - Cannot monitor production without structured logs.

---

## HIGH PRIORITY: Next 2 Weeks (P1)

### 6. Create Smoke Test Suite
**File:** `tests/smoke/test_deployment.py` (NEW)
**Effort:** 4 hours

Tests to create:
- Environment variables present
- Database connectivity
- Data freshness (< 7 days old)
- API connectivity
- Model files present

---

### 7. Implement Health Check Endpoints
**File:** `src/common/health.py` (NEW)
**Effort:** 2 hours

Checks:
- Database status
- API connectivity
- Data staleness
- Overall system health

---

### 8. Add Integration Tests to CI
**Files:** `tests/integration/*.py`
**Effort:** 6 hours

Test:
- Yahoo Finance API integration
- FRED API integration
- DuckDB CRUD operations
- End-to-end data pipeline

---

### 9. Create Deployment Workflow
**File:** `.github/workflows/deploy.yml` (NEW)
**Effort:** 4 hours

Stages:
1. Pre-deployment validation
2. Build & package
3. Deploy to staging
4. Smoke tests
5. Deploy to production

---

### 10. Setup Monitoring
**Tool:** Healthchecks.io (free tier)
**Effort:** 1 hour

Monitor:
- Daily data updates
- Signal generation jobs
- Portfolio calculations
- System health

---

### 11. Configure GitHub Secrets
**Effort:** 15 minutes

Required secrets:
```bash
ANTHROPIC_API_KEY
FRED_API_KEY
CODECOV_TOKEN (optional)
```

Setup via GitHub CLI:
```bash
"/c/Program Files/GitHub CLI/gh.exe" secret set ANTHROPIC_API_KEY
"/c/Program Files/GitHub CLI/gh.exe" secret set FRED_API_KEY
```

---

### 12. Add Coverage Enforcement
**File:** `.github/workflows/ci.yml`
**Effort:** 5 minutes

```yaml
# Change line 48 from:
run: uv run pytest -v --cov --cov-report=term-missing

# To:
run: uv run pytest -v --cov --cov-report=term-missing --cov-fail-under=80
```

---

## MEDIUM PRIORITY: Next Month (P2)

### 13. Implement Release Workflow
**File:** `.github/workflows/release.yml` (NEW)
**Effort:** 3 hours

Features:
- Automatic changelog generation
- GitHub release creation
- Semantic versioning
- Artifact publishing

---

### 14. Add Dependency Vulnerability Scanning
**Tool:** pip-audit or Dependabot
**Effort:** 1 hour

Scans for:
- Known CVEs in dependencies
- Outdated packages
- License compliance

---

### 15. Create Deployment Documentation
**File:** `docs/DEPLOYMENT_GUIDE.md`
**Effort:** 2 hours

Cover:
- Local deployment setup
- Cloud deployment options
- Environment configuration
- Troubleshooting guide

---

## Implementation Timeline

### Week 1 (Current Sprint)
**Day 1-2:**
- ✅ P0-1: Add pyrefly to CI (15 min) - COMPLETED Dec 12, 2025
- [ ] P0-2: Create .env.example (30 min)
- [ ] P0-3: Update .gitignore (5 min)
- [ ] P1-12: Add coverage enforcement (5 min)

**Day 3-4:**
- ✅ P0-4: Environment validation (2 hours)
- ✅ P0-5: Replace print() with logging (4 hours)

**Day 5:**
- Testing and validation
- Documentation updates

### Week 2
**Day 1-2:**
- ✅ P1-6: Smoke test suite (4 hours)
- ✅ P1-7: Health checks (2 hours)

**Day 3-4:**
- ✅ P1-8: Integration tests (6 hours)
- ✅ P1-10: Setup monitoring (1 hour)

**Day 5:**
- ✅ P1-11: Configure secrets (15 min)
- Documentation

### Week 3-4 (Post-Sprint 4)
- ✅ P1-9: Deployment workflow (4 hours)
- ✅ P2-13: Release workflow (3 hours)
- ✅ P2-14: Dependency scanning (1 hour)
- ✅ P2-15: Deployment docs (2 hours)

---

## Acceptance Criteria

**Sprint 4 CI/CD Goals - Definition of Done:**

- [x] Type checking runs in CI on every PR (COMPLETED Dec 12, 2025)
- [ ] `.env.example` file exists
- [ ] `.gitignore` includes security patterns
- [ ] Environment validation implemented
- [ ] Structured logging replaces all print()
- [ ] Smoke test suite passing
- [ ] Health check endpoints working
- [ ] Integration tests in CI
- [ ] Monitoring configured
- [ ] GitHub secrets set
- [ ] Coverage enforcement active (80%)
- [ ] All CI checks passing

**Success Metrics:**
- CI execution time < 5 minutes
- Zero type checking failures
- 232+ tests passing
- Coverage ≥ 80%
- Zero critical/high security issues
- Zero secrets in repo

---

## Quick Reference: What to Do Now

### Immediate Actions (Today):

1. **Add to `.github/workflows/ci.yml` (after line 45):**
   ```yaml
   - name: Type check with Pyrefly
     run: uv run pyrefly check src/
   ```

2. **Create `.env.example`:**
   ```bash
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   FRED_API_KEY=your-fred-key-here
   DATA_DIR=./data
   LOG_LEVEL=INFO
   ```

3. **Update `.gitignore`:**
   ```gitignore
   .env
   .env.*
   *.env
   secrets/
   credentials/
   *.key
   *.pem
   ```

4. **Add coverage enforcement (line 48 of ci.yml):**
   ```yaml
   run: uv run pytest -v --cov --cov-report=term-missing --cov-fail-under=80
   ```

5. **Commit and create PR:**
   ```bash
   git checkout -b feat/cicd-hardening
   git add .github/workflows/ci.yml .env.example .gitignore
   git commit -m "feat(ci): Add type checking and environment security"
   git push -u origin feat/cicd-hardening
   ```

---

## Risk Assessment

| Risk | Current | After P0 | After P1 |
|------|---------|----------|----------|
| Type errors in production | HIGH | LOW | LOW |
| Secrets committed | HIGH | LOW | LOW |
| Deployment failure | HIGH | HIGH | LOW |
| No monitoring | HIGH | MEDIUM | LOW |
| Config drift | HIGH | MEDIUM | LOW |

---

## Contact

**Questions or issues implementing these changes?**
- Review full analysis: `docs/reviews/cicd-deployment-review.md`
- Deployment guide: `docs/DEPLOYMENT.md`
- IT-Core review: `docs/reviews/it-core-post-sprint3-review.md`

---

**Remember:** If it's not tested, it's not ready for production.

**Next Review:** Post-Sprint 4 completion

---

*Prepared by Lamine - CI/CD & Deployment Expert*
*December 11, 2025*
