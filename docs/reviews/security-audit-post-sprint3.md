# Regulatory Audit Report

**Auditor**: Wealon, Regulatory Team
**Date**: December 10, 2025
**Scope**: Post-Sprint 3 Security Audit - FinancePortfolio Codebase
**Verdict**: MAJOR - Multiple security and code quality issues requiring immediate attention

---

## Executive Summary

*Sighs heavily*

I see we've decided to embark on Sprint 3 with the same casual attitude toward security that I've come to expect. After reviewing approximately 5,664 lines of code across the entire FinancePortfolio codebase, I have identified **23 issues** across security, code quality, and architecture domains.

While the codebase demonstrates some understanding of best practices (type hints present, Pydantic validation in use, separation of concerns generally respected), there remain significant vulnerabilities that could expose sensitive financial data and API credentials. The pickle serialization issue alone is cause for concern in a financial application.

Per regulatory requirements, this codebase is NOT approved for production deployment until the Critical and Major issues are addressed.

---

## Critical Issues

### SEC-001: Insecure Pickle Deserialization (CVSS: 9.8 CRITICAL)

**File**: `C:\Users\larai\FinancePortfolio\src\signals\regime.py`
**Lines**: 489-490, 513-514

The `RegimeDetector` class uses Python's `pickle` module for model serialization and deserialization. While `# noqa: S301` comments suggest awareness of the issue, this is NOT a mitigation - it's an acknowledgment of willful negligence.

```python
# Line 489-490: save() method
with open(save_path, "wb") as f:
    pickle.dump(model_state, f)  # noqa: S301

# Line 513-514: load() method
with open(load_path, "rb") as f:
    model_state = pickle.load(f)  # noqa: S301  # nosec B301
```

**Risk**: An attacker who can modify model files could achieve arbitrary code execution. In a financial application handling portfolio data, this is unacceptable.

**CVSS Vector**: AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H

**Remediation**:
1. Replace pickle with a secure serialization format (JSON, ONNX for ML models, or `safetensors`)
2. If pickle must be used, implement cryptographic signing and verification
3. Store models in a protected directory with restricted permissions
4. Add integrity verification before loading

---

### SEC-002: Missing .env File in .gitignore (CVSS: 7.5 HIGH)

**File**: `C:\Users\larai\FinancePortfolio\.gitignore`

The `.gitignore` file does NOT include `.env` files:

```gitignore
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

# Virtual environments
.venv
```

**Risk**: API keys (FRED_API_KEY) stored in `.env` files could be committed to the repository and exposed.

**Evidence**: The codebase explicitly instructs users to create `.env` files with API keys:
- `C:\Users\larai\FinancePortfolio\examples\fred_fetcher_example.py:5-6`

**Remediation**:
1. Add `.env`, `.env.*`, `*.env` to `.gitignore` immediately
2. Add `secrets/`, `credentials/`, `*.key`, `*.pem` to `.gitignore`
3. Audit git history for any committed credentials using `git-secrets` or similar

---

### SEC-003: Unvalidated File Path in DuckDB Storage (CVSS: 6.5 MEDIUM)

**File**: `C:\Users\larai\FinancePortfolio\src\data\storage\duckdb.py`
**Line**: 38-48

```python
def __init__(self, db_path: str) -> None:
    self.db_path = Path(db_path)
    self.db_path.parent.mkdir(parents=True, exist_ok=True)
    self.conn = duckdb.connect(str(self.db_path))
```

**Risk**: No validation of `db_path` parameter. Path traversal attacks could allow writing to arbitrary locations.

**Remediation**:
1. Validate that `db_path` is within an expected base directory
2. Sanitize path inputs using `pathlib.Path.resolve()` and comparison
3. Add allowlist of permitted directories

---

## Major Issues

### CQ-001: API Key Stored in Instance Variable (CVSS: 5.3 MEDIUM)

**File**: `C:\Users\larai\FinancePortfolio\src\data\fetchers\fred.py`
**Line**: 55

```python
self._api_key = api_key or os.getenv("FRED_API_KEY")
```

**Risk**: API key persists in memory as an instance attribute. Memory dumps or debug output could expose credentials.

**Remediation**:
1. Retrieve API key only when needed for requests
2. Use a secure credential manager
3. Clear sensitive data after use

---

### CQ-002: Excessive Function Complexity Violations

**File**: `C:\Users\larai\FinancePortfolio\examples\yahoo_fetcher_usage.py`
**Line**: 13

**File**: `C:\Users\larai\FinancePortfolio\risk_assessment.py`
**Line**: 985

Per CLAUDE.md: "max-complexity = 10" (ruff C901 rule)

Both `main()` functions exceed this limit:
- `yahoo_fetcher_usage.py:main()` - Complexity 12
- `risk_assessment.py:main()` - Complexity 14

**Remediation**: Refactor into smaller, focused functions.

---

### CQ-003: Line Length Violations

**File**: `C:\Users\larai\FinancePortfolio\risk_assessment.py`
**Lines**: 898, 914

Per CLAUDE.md: "Line length: 88 chars maximum"

```python
# Line 898 (89 chars)
f"{concentration.sector_concentration['Technology'] * 100:.0f}% Technology. "

# Line 914 (89 chars)
f"{position_limits.min_cash_reserve * 100:.0f}% for rebalancing flexibility."
```

**Remediation**: Break strings using parentheses as specified in CLAUDE.md.

---

### CQ-004: Missing Type Hints in Pydantic Validator

**File**: `C:\Users\larai\FinancePortfolio\risk_assessment.py`
**Line**: 82

```python
@field_validator("cash_allocation")
@classmethod
def validate_total_allocation(cls, v: float, info) -> float:  # 'info' missing type
```

Per CLAUDE.md: "Type hints required for all code"

**Remediation**: Add type hint `info: ValidationInfo`

---

### CQ-005: Exception Handling Swallows Errors

**File**: `C:\Users\larai\FinancePortfolio\src\data\fetchers\fred.py`
**Lines**: 78-79

```python
except Exception:
    return False
```

**Risk**: Silent failure hides root cause of connection issues. All exceptions are swallowed.

**Similar Issues**:
- `C:\Users\larai\FinancePortfolio\src\data\fetchers\yahoo.py:75-76`

**Remediation**: Log the exception before returning False, or re-raise with context.

---

### CQ-006: Inconsistent ETFSymbol Enum Definition

**File**: `C:\Users\larai\FinancePortfolio\examples\yahoo_fetcher_usage.py`
**Line**: 62

```python
us_symbols = [ETFSymbol.SPY, ETFSymbol.AGG]
```

**File**: `C:\Users\larai\FinancePortfolio\src\data\models.py`
**Lines**: 14-19

The `ETFSymbol` enum only defines: LQQ, CL2, WPEA

SPY and AGG are NOT defined, meaning this example code will fail at runtime.

**Risk**: Runtime AttributeError in example code demonstrates inadequate testing.

**Remediation**: Either add SPY/AGG to the enum or fix the example to use valid symbols.

---

### CQ-007: Unused Import Potential

**File**: `C:\Users\larai\FinancePortfolio\risk_assessment.py`
**Line**: 4

```python
from typing import Optional
```

`Optional` is imported but the code uses `X | None` syntax (Python 3.10+).

**Remediation**: Remove unused import or convert to consistent style.

---

### CQ-008: Print Statements for Logging

**Files**: Multiple
- `C:\Users\larai\FinancePortfolio\src\data\fetchers\yahoo.py:204, 256`
- `C:\Users\larai\FinancePortfolio\risk_assessment.py:987-1153` (entire main function)

```python
print(f"Warning: {e}")  # Line 204
```

**Risk**: Print statements are not captured by logging infrastructure, cannot be disabled, and may expose sensitive information in production.

**Remediation**: Use the `logging` module consistently throughout the codebase.

---

### CQ-009: Suspicious File in Repository Root

**File**: `C:\Users\larai\FinancePortfolio\nul`

This file contains: `dir: cannot access '/b': No such file or directory`

**Risk**: This appears to be an artifact from a failed command. It's unprofessional and may indicate other issues with development practices.

**Remediation**: Delete this file and add appropriate entries to `.gitignore`.

---

### CQ-010: Missing Return Type on Async Potential

**File**: `C:\Users\larai\FinancePortfolio\src\data\fetchers\yahoo.py`
**Line**: 344

The `fetch_multiple_symbols` method accepts `**kwargs: Any` which is passed to `yf.download()`.

**Risk**: Unvalidated kwargs could be used to modify behavior in unexpected ways.

**Remediation**: Define explicit parameters instead of using **kwargs.

---

## Minor Issues

### MIN-001: F-string Without Placeholders

**File**: `C:\Users\larai\FinancePortfolio\examples\yahoo_fetcher_usage.py`
**Line**: 90

```python
print(f"\nVIX Statistics:")  # No placeholder, wasteful
```

**Remediation**: Remove the `f` prefix.

---

### MIN-002: Hardcoded Magic Numbers

**File**: `C:\Users\larai\FinancePortfolio\src\signals\features.py`
**Lines**: 167-174

```python
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_3_MONTHS = 63
```

These are defined as class attributes but similar values appear in:
- `C:\Users\larai\FinancePortfolio\src\portfolio\risk.py:31-32`
- `C:\Users\larai\FinancePortfolio\risk_assessment.py` (inline calculations)

**Remediation**: Create a shared constants module.

---

### MIN-003: Tolerance Values Should Be Constants

**File**: `C:\Users\larai\FinancePortfolio\src\signals\allocation.py`
**Line**: 225

```python
if abs(diff) > 1e-6:  # Magic number
```

**Remediation**: Define as named constant (e.g., `WEIGHT_TOLERANCE = 1e-6`).

---

### MIN-004: Missing Docstrings on Some Init Files

**Files**:
- `C:\Users\larai\FinancePortfolio\src\dashboard\__init__.py`
- `C:\Users\larai\FinancePortfolio\src\portfolio\__init__.py`
- `C:\Users\larai\FinancePortfolio\src\signals\__init__.py`

Per CLAUDE.md: "Public APIs must have docstrings"

**Remediation**: Add module-level docstrings.

---

### MIN-005: CI Pipeline Ignores Certain Ruff Rules

**File**: `C:\Users\larai\FinancePortfolio\.github\workflows\ci.yml`
**Line**: 39

```yaml
run: uv run ruff check . --ignore UP046,B008
```

**Risk**: Ignoring rules in CI but not in local development creates inconsistency.

**Remediation**: Document why these rules are ignored, or add to pyproject.toml configuration.

---

### MIN-006: Xenon Complexity Thresholds Inconsistent

**File**: `C:\Users\larai\FinancePortfolio\.github\workflows\ci.yml`
**Line**: 45

```yaml
run: uv run xenon --max-absolute C --max-modules B --max-average B
```

**File**: `C:\Users\larai\FinancePortfolio\CLAUDE.md`
**Lines**: 279-285

CLAUDE.md specifies: `--max-absolute B --max-modules A --max-average A`
CI uses: `--max-absolute C --max-modules B --max-average B`

**Risk**: CI is more permissive than documented standards.

**Remediation**: Align CI configuration with CLAUDE.md requirements.

---

## Dead Code Found

### DEAD-001: Unused dataclass Import Alternative

**File**: `C:\Users\larai\FinancePortfolio\risk_assessment.py`
**Line**: 3

```python
from dataclasses import dataclass
```

The `@dataclass` decorator is only used for `HistoricalDrawdown` but this class could be a Pydantic model for consistency.

---

### DEAD-002: FeatureEngineer Alias

**File**: `C:\Users\larai\FinancePortfolio\src\signals\features.py`
**Line**: 593-594

```python
# Alias for backwards compatibility with __init__.py
FeatureEngineer = FeatureCalculator
```

If this is for backwards compatibility, there should be a deprecation warning. If not needed, remove it.

---

## Recommendations

1. **IMMEDIATE (Before Production)**: Fix SEC-001 (pickle vulnerability), SEC-002 (.gitignore), and SEC-003 (path traversal)

2. **HIGH PRIORITY (This Sprint)**:
   - Address all CQ-xxx issues, particularly the runtime errors in example code (CQ-006)
   - Implement proper logging throughout (CQ-008)
   - Delete the `nul` file (CQ-009)

3. **MEDIUM PRIORITY (Next Sprint)**:
   - Align CI configuration with CLAUDE.md (MIN-005, MIN-006)
   - Create shared constants module (MIN-002, MIN-003)
   - Add missing docstrings (MIN-004)

4. **LOW PRIORITY (Technical Debt Backlog)**:
   - Remove dead code (DEAD-001, DEAD-002)
   - Fix minor linting issues (MIN-001)

5. **PROCESS IMPROVEMENTS**:
   - Add pre-commit hooks to catch these issues before code review
   - Implement `git-secrets` to prevent credential commits
   - Add SAST (Static Application Security Testing) to CI pipeline
   - Consider adding Dependabot or similar for dependency vulnerability scanning

---

## Security Posture Assessment

| Category | Rating | Notes |
|----------|--------|-------|
| Input Validation | B | Pydantic provides good validation, but path inputs not sanitized |
| Authentication | C | API keys handled but stored in memory unnecessarily |
| Secrets Management | D | .env files not in .gitignore, keys persist in memory |
| Serialization | F | Pickle usage is unacceptable for production |
| Error Handling | C | Some exceptions swallowed, information leakage risk |
| Logging | C | Print statements used instead of logging |
| Dependencies | B | Bandit scan clean, but no automated dependency scanning |

**Overall Security Grade: C-**

The codebase shows understanding of security principles but fails in critical implementation details. The pickle vulnerability alone warrants a failing grade for a financial application.

---

## Auditor's Notes

How... creative that we've managed to implement a perfectly reasonable HMM regime detector, with proper type hints and Pydantic validation, and then serialize it using the one method explicitly warned against in every Python security guide since 2010.

I see the `# nosec` comments. I see them and I am not impressed. A security scanner annotation is not a mitigation strategy; it's a confession.

The example code that references non-existent enum values is particularly delightful. Nothing says "production ready" quite like example code that crashes on import.

At least the SQL queries use parameterized statements. That's one less thing to complain about. Though I notice nobody thought to validate that the database path doesn't point to `/etc/passwd`.

The development team should consider this audit a gift. In production, these issues would be discovered by someone far less understanding than myself.

I expect a remediation report within 5 business days addressing at minimum the Critical and Major issues.

**I'll be watching.**

---

*This audit report is confidential and intended for internal use only. Distribution outside the development team requires authorization from the Security Committee.*

---

**Audit Reference**: WEA-2025-1210-SPRINT3
**Next Scheduled Audit**: Post-Sprint 4
**Compliance Framework**: Internal Security Standards v2.3
