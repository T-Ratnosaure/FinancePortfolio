# Next Steps: Sprint 5 P1 Implementation Guide

**Sprint:** Sprint 5 P1 - Deployment Readiness
**Timeline:** 2 weeks (December 12-26, 2025)
**Goal:** Achieve production-ready infrastructure

---

## Quick Start: What to Do Right Now

### Option 1: Full Sprint 5 P1 (32 hours over 2 weeks)
Follow the detailed plan below for complete production readiness.

### Option 2: Minimum Viable Production (9 hours over 3 days)
Implement only the absolute essentials:
1. Health checks (2h)
2. Environment validator (2h)
3. Deployment workflow (4h)
4. Monitoring (1h)

**Recommendation:** Do Option 2 first, then add remaining P1 items incrementally.

---

## Day-by-Day Implementation Guide

## DAY 1: Environment & Configuration (4 hours)

### Task 1.1: Create .env.example (30 minutes)

**Owner:** Clovis (IT-Core)

```bash
# Navigate to project root
cd /c/Users/larai/FinancePortfolio

# Create .env.example file
cat > .env.example << 'EOF'
# API Keys (REQUIRED)
ANTHROPIC_API_KEY=sk-ant-your-key-here
FRED_API_KEY=your-fred-key-here

# Data Storage (OPTIONAL - defaults shown)
DATA_DIR=./data
DUCKDB_PATH=./data/portfolio.duckdb

# Logging (OPTIONAL - defaults shown)
LOG_LEVEL=INFO
LOG_FORMAT=console

# Monitoring (OPTIONAL)
HEALTHCHECK_IO_URL=https://hc-ping.com/your-uuid-here

# Feature Flags (OPTIONAL)
ENABLE_BACKTESTING=false
ENABLE_PAPER_TRADING=false
EOF

# Verify file created
ls -la .env.example
```

**Acceptance Criteria:**
- [ ] File exists at project root
- [ ] All required env vars documented
- [ ] Comments explain purpose of each var
- [ ] Sensitive values are placeholders (not real keys)

---

### Task 1.2: Implement Environment Validator (2 hours)

**Owner:** Clovis (IT-Core)

**Step 1: Create validator module (1 hour)**

```bash
# Create config directory if it doesn't exist
mkdir -p /c/Users/larai/FinancePortfolio/src/config

# Create validator file
cat > /c/Users/larai/FinancePortfolio/src/config/env_validator.py << 'EOF'
"""Environment configuration validator.

Validates that all required environment variables are present and valid
before starting the application.
"""
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class EnvironmentConfig(BaseSettings):
    """Application environment configuration."""

    # API Keys (REQUIRED)
    anthropic_api_key: str | None = Field(
        None,
        alias="ANTHROPIC_API_KEY",
        description="Anthropic API key for LLM features"
    )
    fred_api_key: str = Field(
        ...,  # Required
        alias="FRED_API_KEY",
        description="FRED API key for macro data"
    )

    # Paths (OPTIONAL)
    data_dir: Path = Field(
        default=Path("./data"),
        alias="DATA_DIR",
        description="Directory for data storage"
    )
    duckdb_path: Path = Field(
        default=Path("./data/portfolio.duckdb"),
        alias="DUCKDB_PATH",
        description="DuckDB database file path"
    )

    # Logging (OPTIONAL)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level"
    )
    log_format: Literal["console", "json"] = Field(
        default="console",
        alias="LOG_FORMAT",
        description="Log output format"
    )

    # Monitoring (OPTIONAL)
    healthcheck_io_url: str | None = Field(
        None,
        alias="HEALTHCHECK_IO_URL",
        description="Healthchecks.io ping URL"
    )

    @field_validator("data_dir", "duckdb_path")
    @classmethod
    def validate_path_parent_exists(cls, v: Path) -> Path:
        """Ensure parent directory of path exists."""
        if not v.parent.exists():
            msg = f"Parent directory does not exist: {v.parent}"
            raise ValueError(msg)
        return v

    @field_validator("fred_api_key")
    @classmethod
    def validate_fred_key(cls, v: str) -> str:
        """Validate FRED API key format."""
        if len(v) < 10:
            msg = "FRED API key appears invalid (too short)"
            raise ValueError(msg)
        return v

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"


def validate_environment() -> EnvironmentConfig:
    """Validate environment configuration.

    Returns:
        Validated environment configuration

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If required files/directories missing
    """
    try:
        config = EnvironmentConfig()
        return config
    except Exception as e:
        msg = f"Environment validation failed: {e}"
        raise ValueError(msg) from e


if __name__ == "__main__":
    # CLI usage: python -m src.config.env_validator
    try:
        config = validate_environment()
        print("✅ Environment validation PASSED")
        print(f"\nConfiguration:")
        print(f"  FRED API Key: {'*' * 20}{config.fred_api_key[-4:]}")
        print(f"  Data Directory: {config.data_dir}")
        print(f"  DuckDB Path: {config.duckdb_path}")
        print(f"  Log Level: {config.log_level}")
        print(f"  Log Format: {config.log_format}")
        if config.healthcheck_io_url:
            print(f"  Monitoring: Enabled")
    except Exception as e:
        print(f"❌ Environment validation FAILED")
        print(f"\nError: {e}")
        print(f"\nPlease check your .env file and ensure all required variables are set.")
        print(f"See .env.example for reference.")
        exit(1)
EOF
```

**Step 2: Add tests (1 hour)**

```bash
# Create test file
cat > /c/Users/larai/FinancePortfolio/tests/test_config/test_env_validator.py << 'EOF'
"""Tests for environment validator."""
import os
from pathlib import Path

import pytest

from src.config.env_validator import EnvironmentConfig, validate_environment


def test_valid_minimal_config(monkeypatch, tmp_path):
    """Test valid configuration with minimal required fields."""
    monkeypatch.setenv("FRED_API_KEY", "test_key_12345")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    config = validate_environment()

    assert config.fred_api_key == "test_key_12345"
    assert config.data_dir == tmp_path


def test_missing_required_field_raises_error(monkeypatch):
    """Test that missing FRED API key raises error."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Environment validation failed"):
        validate_environment()


def test_invalid_fred_key_raises_error(monkeypatch, tmp_path):
    """Test that too-short FRED key raises error."""
    monkeypatch.setenv("FRED_API_KEY", "short")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="FRED API key appears invalid"):
        validate_environment()


def test_nonexistent_parent_directory_raises_error(monkeypatch):
    """Test that nonexistent parent directory raises error."""
    monkeypatch.setenv("FRED_API_KEY", "test_key_12345")
    monkeypatch.setenv("DATA_DIR", "/nonexistent/path/data")

    with pytest.raises(ValueError):
        validate_environment()
EOF

# Create __init__.py for test package
mkdir -p /c/Users/larai/FinancePortfolio/tests/test_config
touch /c/Users/larai/FinancePortfolio/tests/test_config/__init__.py
```

**Step 3: Integrate into main.py**

```python
# Add to top of main.py (after imports)
from src.config.env_validator import validate_environment

def main():
    """Main entry point."""
    # Validate environment before doing anything else
    try:
        config = validate_environment()
        logger.info("Environment validation passed")
    except ValueError as e:
        logger.error(f"Environment validation failed: {e}")
        logger.error("Please check your .env file. See .env.example for reference.")
        sys.exit(1)

    # Rest of main() code...
```

**Acceptance Criteria:**
- [ ] EnvironmentConfig class created with Pydantic
- [ ] All required fields validated
- [ ] Path validation works
- [ ] Tests passing (3+ tests)
- [ ] Integrated into main.py
- [ ] CLI tool works: `python -m src.config.env_validator`

---

### Task 1.3: Add pydantic-settings dependency (5 minutes)

```bash
cd /c/Users/larai/FinancePortfolio
uv add pydantic-settings
```

---

### Task 1.4: Test environment validator (15 minutes)

```bash
# Test with missing key
unset FRED_API_KEY
python -m src.config.env_validator
# Should fail with helpful error

# Test with valid key
export FRED_API_KEY="your_real_key_here"
python -m src.config.env_validator
# Should pass

# Run tests
uv run pytest tests/test_config/test_env_validator.py -v
```

---

## DAY 2: Health Checks & Monitoring (3 hours)

### Task 2.1: Implement Health Check System (2 hours)

**Owner:** Sophie (Data)

**Step 1: Create health check module (1 hour)**

```python
# File: src/common/health.py
"""Health check system for monitoring application status."""
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.data.storage.duckdb_storage import DuckDBStorage


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    name: str = Field(description="Component name")
    status: HealthStatus = Field(description="Health status")
    message: str | None = Field(None, description="Status message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")
    checked_at: datetime = Field(default_factory=datetime.now, description="Check timestamp")


class SystemHealth(BaseModel):
    """Overall system health status."""
    status: HealthStatus = Field(description="Overall health status")
    components: list[ComponentHealth] = Field(description="Component health checks")
    checked_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.status == HealthStatus.HEALTHY


class HealthChecker:
    """System health checker."""

    def __init__(self, storage: DuckDBStorage):
        """Initialize health checker.

        Args:
            storage: DuckDB storage instance
        """
        self.storage = storage

    def check_database_connectivity(self) -> ComponentHealth:
        """Check database connectivity."""
        try:
            # Try a simple query
            conn = self.storage._get_connection()
            conn.execute("SELECT 1").fetchone()
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection successful"
            )
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {e}"
            )

    def check_data_freshness(self, max_age_days: int = 7) -> ComponentHealth:
        """Check data freshness.

        Args:
            max_age_days: Maximum acceptable data age in days

        Returns:
            Component health status
        """
        try:
            # Check most recent price data
            conn = self.storage._get_connection()
            result = conn.execute(
                """
                SELECT MAX(date) as latest_date
                FROM raw_prices
                """
            ).fetchone()

            if not result or not result[0]:
                return ComponentHealth(
                    name="data_freshness",
                    status=HealthStatus.UNHEALTHY,
                    message="No price data found in database"
                )

            latest_date = date.fromisoformat(result[0])
            age_days = (date.today() - latest_date).days

            if age_days <= max_age_days:
                status = HealthStatus.HEALTHY
                message = f"Data is fresh ({age_days} days old)"
            elif age_days <= max_age_days * 2:
                status = HealthStatus.DEGRADED
                message = f"Data is stale ({age_days} days old)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Data is very stale ({age_days} days old)"

            return ComponentHealth(
                name="data_freshness",
                status=status,
                message=message,
                details={"latest_date": str(latest_date), "age_days": age_days}
            )
        except Exception as e:
            return ComponentHealth(
                name="data_freshness",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check data freshness: {e}"
            )

    def check_overall_health(self) -> SystemHealth:
        """Check overall system health.

        Returns:
            System health status with all component checks
        """
        components = [
            self.check_database_connectivity(),
            self.check_data_freshness(),
        ]

        # Determine overall status
        if any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in components):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return SystemHealth(
            status=overall_status,
            components=components
        )


# CLI usage
if __name__ == "__main__":
    import sys
    from src.data.storage.duckdb_storage import DuckDBStorage

    storage = DuckDBStorage()
    checker = HealthChecker(storage)
    health = checker.check_overall_health()

    print(f"Overall Status: {health.status.value.upper()}")
    print(f"\nComponents:")
    for component in health.components:
        print(f"  {component.name}: {component.status.value}")
        if component.message:
            print(f"    {component.message}")
        if component.details:
            for key, value in component.details.items():
                print(f"    {key}: {value}")

    # Exit with error code if unhealthy
    sys.exit(0 if health.is_healthy else 1)
```

**Step 2: Add tests (1 hour)**

```python
# File: tests/test_common/test_health.py
"""Tests for health check system."""
import pytest
from datetime import date, timedelta

from src.common.health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    SystemHealth
)
from src.data.storage.duckdb_storage import DuckDBStorage
from src.data.models import DailyPrice, ETFSymbol


def test_database_connectivity_healthy(tmp_path):
    """Test database connectivity check when healthy."""
    db_path = tmp_path / "test.duckdb"
    storage = DuckDBStorage(str(db_path))
    checker = HealthChecker(storage)

    health = checker.check_database_connectivity()

    assert health.name == "database"
    assert health.status == HealthStatus.HEALTHY


def test_data_freshness_healthy(tmp_path):
    """Test data freshness check with recent data."""
    db_path = tmp_path / "test.duckdb"
    storage = DuckDBStorage(str(db_path))
    checker = HealthChecker(storage)

    # Add recent price data
    recent_price = DailyPrice(
        symbol=ETFSymbol.LQQ.value,
        date=date.today() - timedelta(days=1),
        open=100.0,
        high=105.0,
        low=99.0,
        close=103.0,
        volume=1000000
    )
    storage.store_prices([recent_price])

    health = checker.check_data_freshness(max_age_days=7)

    assert health.name == "data_freshness"
    assert health.status == HealthStatus.HEALTHY
    assert health.details["age_days"] <= 7


def test_data_freshness_stale(tmp_path):
    """Test data freshness check with stale data."""
    db_path = tmp_path / "test.duckdb"
    storage = DuckDBStorage(str(db_path))
    checker = HealthChecker(storage)

    # Add old price data
    old_price = DailyPrice(
        symbol=ETFSymbol.LQQ.value,
        date=date.today() - timedelta(days=10),
        open=100.0,
        high=105.0,
        low=99.0,
        close=103.0,
        volume=1000000
    )
    storage.store_prices([old_price])

    health = checker.check_data_freshness(max_age_days=7)

    assert health.name == "data_freshness"
    assert health.status == HealthStatus.DEGRADED
    assert health.details["age_days"] > 7


def test_overall_health_all_healthy(tmp_path):
    """Test overall health when all components healthy."""
    db_path = tmp_path / "test.duckdb"
    storage = DuckDBStorage(str(db_path))
    checker = HealthChecker(storage)

    # Add recent data
    recent_price = DailyPrice(
        symbol=ETFSymbol.LQQ.value,
        date=date.today() - timedelta(days=1),
        open=100.0,
        high=105.0,
        low=99.0,
        close=103.0,
        volume=1000000
    )
    storage.store_prices([recent_price])

    health = checker.check_overall_health()

    assert health.status == HealthStatus.HEALTHY
    assert health.is_healthy
    assert len(health.components) >= 2
```

**Acceptance Criteria:**
- [ ] HealthChecker class implemented
- [ ] Database connectivity check works
- [ ] Data freshness check works
- [ ] Overall health aggregation works
- [ ] Tests passing (4+ tests)
- [ ] CLI tool works: `python -m src.common.health`

---

### Task 2.2: Setup Monitoring with Healthchecks.io (1 hour)

**Owner:** Lamine (CI/CD)

**Step 1: Create Healthchecks.io account (10 minutes)**

1. Go to https://healthchecks.io/
2. Sign up for free account (20 checks, no credit card)
3. Create new check: "FinancePortfolio Daily Data Update"
4. Set schedule: Every 24 hours
5. Copy ping URL

**Step 2: Add monitoring to daily data fetch (20 minutes)**

```python
# Add to data fetching script
import os
import requests

def ping_healthcheck(success: bool = True):
    """Ping Healthchecks.io to report job status.

    Args:
        success: Whether job succeeded
    """
    url = os.getenv("HEALTHCHECK_IO_URL")
    if not url:
        return  # Monitoring disabled

    try:
        if success:
            requests.get(url, timeout=10)
        else:
            requests.get(f"{url}/fail", timeout=10)
    except Exception:
        # Don't fail job if monitoring fails
        pass

# In data fetch script
try:
    # Fetch data
    fetcher.fetch_all()
    ping_healthcheck(success=True)
except Exception as e:
    logger.error(f"Data fetch failed: {e}")
    ping_healthcheck(success=False)
    raise
```

**Step 3: Add to .env.example and documentation (10 minutes)**

```bash
# .env.example
HEALTHCHECK_IO_URL=https://hc-ping.com/your-uuid-here
```

**Step 4: Test monitoring (20 minutes)**

```bash
# Set real Healthchecks.io URL
export HEALTHCHECK_IO_URL="your_real_url_here"

# Test success ping
python -c "
import os
import requests
url = os.getenv('HEALTHCHECK_IO_URL')
requests.get(url, timeout=10)
print('Success ping sent')
"

# Check dashboard - should see green check

# Test failure ping
python -c "
import os
import requests
url = os.getenv('HEALTHCHECK_IO_URL')
requests.get(f'{url}/fail', timeout=10)
print('Failure ping sent')
"

# Check dashboard - should see red X and get email alert
```

**Acceptance Criteria:**
- [ ] Healthchecks.io account created
- [ ] Check configured for daily data updates
- [ ] Ping function implemented
- [ ] Success pings work
- [ ] Failure pings work and send alerts
- [ ] URL added to .env.example

---

## DAY 3: Pre-Commit Hooks (1 hour)

### Task 3.1: Setup Pre-Commit Framework (1 hour)

**Owner:** Lamine (CI/CD)

**Step 1: Install pre-commit (10 minutes)**

```bash
cd /c/Users/larai/FinancePortfolio

# Add pre-commit to dev dependencies
uv add --dev pre-commit

# Install git hooks
uv run pre-commit install

# Verify installation
uv run pre-commit --version
```

**Step 2: Create .pre-commit-config.yaml (20 minutes)**

```yaml
# .pre-commit-config.yaml
repos:
  # Fast checks first (fail-fast)
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key  # Security: prevent credential leaks

  # Import sorting (MUST run before ruff format)
  - repo: https://github.com/PyCQA/isort
    rev: 5.13.2
    hooks:
      - id: isort
        name: isort (import sorting)

  # Ruff formatting and linting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.14.8
    hooks:
      - id: ruff-format  # Code formatting
      - id: ruff         # Linting
        args: [--fix, --exit-non-zero-on-fix]

  # Type checking (optional - manual stage due to speed)
  - repo: local
    hooks:
      - id: pyrefly
        name: pyrefly (type checking)
        entry: uv run pyrefly check
        language: system
        types: [python]
        pass_filenames: false
        stages: [manual]  # Only run with --hook-stage manual
```

**Step 3: Test pre-commit hooks (20 minutes)**

```bash
# Test on all files
uv run pre-commit run --all-files

# Should see all hooks pass:
# ✅ trailing-whitespace
# ✅ end-of-file-fixer
# ✅ check-yaml
# ✅ check-added-large-files
# ✅ check-merge-conflict
# ✅ detect-private-key
# ✅ isort
# ✅ ruff-format
# ✅ ruff

# Test type checking (manual)
uv run pre-commit run --hook-stage manual

# Should see:
# ✅ pyrefly
```

**Step 4: Document in CLAUDE.md (10 minutes)**

Add to CLAUDE.md:

```markdown
## Pre-Commit Hooks

Pre-commit hooks run automatically before each commit to enforce quality standards.

### Setup (one-time per developer)

```bash
uv run pre-commit install
```

### Usage

Hooks run automatically on `git commit`. If checks fail, commit is aborted.

**Emergency bypass (use sparingly):**
```bash
git commit --no-verify -m "emergency: bypass hooks"
```

**Run type checking before push:**
```bash
uv run pre-commit run --hook-stage manual --all-files
```

**Update hooks to latest versions:**
```bash
uv run pre-commit autoupdate
```
```

**Acceptance Criteria:**
- [ ] pre-commit installed
- [ ] .pre-commit-config.yaml created
- [ ] Git hooks installed
- [ ] All hooks pass on current codebase
- [ ] Documented in CLAUDE.md

---

## DAY 4-5: Integration Tests (6 hours)

### Task 4.1: Setup Integration Test Infrastructure (2 hours)

**Owner:** Sophie (Data)

See detailed plan in: `docs/reviews/sprint5-cicd-deployment-review.md` Section 5

**Key files to create:**
- `tests/integration/conftest.py`
- `tests/integration/test_yahoo_api.py`
- `tests/integration/test_fred_api.py`
- `tests/integration/test_data_pipeline.py`

---

## DAY 6-7: Deployment Workflow (4 hours)

### Task 5.1: Create Deployment Workflow (4 hours)

**Owner:** Lamine (CI/CD)

See detailed plan in: `docs/reviews/sprint5-cicd-deployment-review.md`

**Key deliverable:**
- `.github/workflows/deploy.yml`

---

## Week 2: Documentation & Polish

See detailed plan in: `docs/reviews/sprint5-cicd-deployment-review.md` Section 7

---

## Quick Reference: Commands

### Environment Validation
```bash
# Validate environment
python -m src.config.env_validator

# Expected output if valid:
# ✅ Environment validation PASSED
```

### Health Checks
```bash
# Check system health
python -m src.common.health

# Expected output if healthy:
# Overall Status: HEALTHY
#
# Components:
#   database: healthy
#     Database connection successful
#   data_freshness: healthy
#     Data is fresh (1 days old)
```

### Pre-Commit Hooks
```bash
# Install hooks
uv run pre-commit install

# Run all hooks
uv run pre-commit run --all-files

# Run type checking
uv run pre-commit run --hook-stage manual
```

### Integration Tests
```bash
# Run integration tests
uv run pytest tests/integration/ -v --integration

# Skip integration tests (default)
uv run pytest tests/integration/ -v
# (tests will be skipped without --integration flag)
```

---

## Success Criteria

Sprint 5 P1 is DONE when:

- [ ] All files created as specified
- [ ] All tests passing (unit + integration)
- [ ] CI/CD pipeline includes all new checks
- [ ] Documentation complete and accurate
- [ ] Deployment workflow tested in staging
- [ ] Health checks reporting correctly
- [ ] Monitoring configured and alerting
- [ ] Pre-commit hooks working for all developers

---

## Get Help

**Stuck? Check these resources:**
- Full technical review: `docs/reviews/sprint5-cicd-deployment-review.md`
- Deployment readiness: `DEPLOYMENT_READINESS.md`
- CI/CD priorities: `docs/CI_CD_PRIORITIES.md`

**Questions about specific tasks:**
- Environment validation: See `src/config/env_validator.py` docstrings
- Health checks: See `src/common/health.py` docstrings
- Pre-commit: See `.pre-commit-config.yaml` comments

---

**Good luck with Sprint 5 P1!**

**- Lamine, CI/CD Expert**
