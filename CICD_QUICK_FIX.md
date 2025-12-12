# QUICK FIX GUIDE - Sprint 5 P0 Type Errors

**Status:** 🔴 CI WILL FAIL - 9 type errors to fix
**Time:** 2-4 hours
**Files:** 2 files need changes

---

## The Problem

Pyrefly type checking was added to CI but code has type errors.
**Result:** CI will fail when you push.

---

## Quick Fix Checklist

```bash
# 1. Fix type errors (see below)
# 2. Verify locally
uv run pyrefly check  # MUST show "0 errors"
uv run pytest -v      # MUST show "258 passed, 12 skipped"

# 3. Run full CI locally
uv run isort --check-only --diff .
uv run ruff format --check .
uv run ruff check . --ignore UP046,B008
uv run bandit -c pyproject.toml -r . --severity-level medium
uv run xenon --max-absolute C --max-modules B --max-average B src/ --exclude ".venv,venv"
uv run pyrefly check
uv run pytest -v --cov --cov-report=term-missing

# 4. Commit and push
git add .
git commit -m "fix(types): Resolve 9 pyrefly type errors"
git push origin feat/sprint5-p0-critical-fixes
```

---

## File 1: `src/portfolio/risk.py`

### Error 1: Line 264
```python
# BROKEN (missing 'other' argument):
cov_matrix = asset_returns.cov().values

# FIXED:
cov_matrix = asset_returns.cov().to_numpy()
```

### Error 2-3: Lines 376-377
```python
# BROKEN (ambiguous type):
aligned = pd.concat([etf_returns, index_returns], axis=1, keys=["etf", "index"])
etf_rets: pd.Series = aligned["etf"]
idx_rets: pd.Series = aligned["index"]

# FIXED (explicit type):
aligned_df: pd.DataFrame = pd.concat(
    [etf_returns, index_returns],
    axis=1,
    keys=["etf", "index"]
)
etf_rets: pd.Series = aligned_df["etf"]
idx_rets: pd.Series = aligned_df["index"]
```

### Error 4-5: Lines 573-574
```python
# BROKEN (same issue as above):
aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1)
port_rets: pd.Series = aligned["portfolio"]
bench_rets: pd.Series = aligned["benchmark"]

# FIXED (same pattern):
aligned_df: pd.DataFrame = pd.concat(
    [portfolio_returns, benchmark_returns],
    axis=1,
    keys=["portfolio", "benchmark"]
)
port_rets: pd.Series = aligned_df["portfolio"]
bench_rets: pd.Series = aligned_df["benchmark"]
```

### Error 6-7: Line 822
```python
# These errors will be fixed by fixing errors 2-3
# (They're caused by etf_rets and idx_rets having wrong types)
# No additional changes needed once errors 2-3 are fixed
```

---

## File 2: `examples/yahoo_fetcher_usage.py`

### Error 8: Line 45
```python
# BROKEN (list[ETFSymbol] vs list[ETFSymbol | str]):
enum_symbols = [ETFSymbol.LQQ, ETFSymbol.CAC40]
prices = fetcher.fetch_etf_prices(enum_symbols, start_date, end_date)

# FIXED (convert to strings):
enum_symbols = [ETFSymbol.LQQ, ETFSymbol.CAC40]
prices = fetcher.fetch_etf_prices(
    [s.value for s in enum_symbols],
    start_date,
    end_date
)
```

---

## Verification Commands

```bash
# After fixes, run these to verify:

# 1. Type check (MUST show "0 errors")
uv run pyrefly check

# 2. Tests (MUST show "258 passed, 12 skipped")
uv run pytest -v

# 3. If both pass, you're good to commit
```

---

## Success Criteria

- Pyrefly: `0 errors` ✅
- Pytest: `258 passed, 12 skipped` ✅
- All CI steps pass locally ✅

---

## Full Details

See `docs/CICD_SPRINT5_SUMMARY.md` for complete review.
