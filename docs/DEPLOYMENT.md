# Deployment Infrastructure Guide

## Overview

This document outlines the complete deployment infrastructure for the PEA Portfolio Management System, designed for **cost-effectiveness, reliability, and maintainability** for individual retail investors.

**Philosophy**: If it's not tested, it's not ready for production.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Testing Strategy](#testing-strategy)
3. [CI/CD Pipeline](#cicd-pipeline)
4. [Deployment Options](#deployment-options)
5. [Scheduling Strategy](#scheduling-strategy)
6. [Monitoring & Alerting](#monitoring--alerting)
7. [Security](#security)
8. [Backup & Recovery](#backup--recovery)
9. [Cost Analysis](#cost-analysis)

---

## Architecture Overview

### Design Principles

1. **Test-Driven**: All deployment changes are tested before production
2. **Fail-Fast**: Catch issues early in the pipeline
3. **Reproducible**: No manual steps, fully automated
4. **Observable**: Comprehensive logging and monitoring
5. **Cost-Conscious**: Optimize for retail investor budget

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Repository                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Feature    │→ │      PR      │→ │    Master    │      │
│  │   Branches   │  │   + Tests    │  │   + Deploy   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline (GitHub Actions)           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Stage 1: Quality Gates                                │  │
│  │  • Format check (ruff format)                         │  │
│  │  • Linting (ruff check)                               │  │
│  │  • Type checking (pyrefly check)                      │  │
│  │  • Security scan (bandit)                             │  │
│  │  • Complexity check (xenon)                           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Stage 2: Testing                                      │  │
│  │  • Unit tests                                         │  │
│  │  • Integration tests                                  │  │
│  │  • Backtest validation tests                          │  │
│  │  • Coverage reports (>80% required)                   │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Stage 3: Build & Package                              │  │
│  │  • Create deployment artifact                         │  │
│  │  • Version tagging                                    │  │
│  │  • Generate release notes                             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Deployment Target                           │
│  Option A: Local (Recommended for retail investors)         │
│  Option B: Cloud Functions (AWS Lambda / GCP Cloud Run)     │
│  Option C: VPS (DigitalOcean / Hetzner)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Testing Strategy

### Test Pyramid

```
           ┌─────────────────┐
           │  E2E / Backtest │  ← 10% (Most expensive)
           │   Validation    │
           └─────────────────┘
         ┌─────────────────────┐
         │   Integration Tests  │  ← 30%
         │  (API, Data Layer)   │
         └─────────────────────┘
    ┌──────────────────────────────┐
    │      Unit Tests               │  ← 60% (Fastest)
    │  (Models, Utils, Business)    │
    └──────────────────────────────┘
```

### 1. Unit Tests

**Coverage**: Business logic, utilities, calculations

**Example Structure**:
```
tests/
├── unit/
│   ├── test_portfolio_calculator.py
│   ├── test_signal_generator.py
│   ├── test_risk_manager.py
│   └── test_pea_compliance.py
```

**Key Tests**:
- Portfolio calculation accuracy
- Signal generation logic
- Risk metrics computation
- PEA eligibility validation
- Performance calculations

**Run**: `uv run pytest tests/unit -v --cov=src`

### 2. Integration Tests

**Coverage**: Data fetching, API interactions, file I/O

**Example Structure**:
```
tests/
├── integration/
│   ├── test_yfinance_integration.py
│   ├── test_boursorama_scraper.py
│   ├── test_database_operations.py
│   └── test_report_generation.py
```

**Key Tests**:
- Yahoo Finance API connectivity
- Data caching mechanisms
- Report generation with real data
- Database read/write operations
- Email notification sending (with mocks)

**Run**: `uv run pytest tests/integration -v --cov-append`

### 3. Backtest Validation Tests

**Coverage**: Strategy validation, historical accuracy

**Example Structure**:
```
tests/
├── backtest/
│   ├── test_momentum_strategy.py
│   ├── test_value_strategy.py
│   ├── test_combined_strategy.py
│   └── test_historical_accuracy.py
```

**Key Tests**:
- Strategy returns match expected patterns
- Drawdown calculations are accurate
- Sharpe ratio computations
- Transaction cost modeling
- Rebalancing logic validation

**Run**: `uv run pytest tests/backtest -v --tb=short`

### 4. Test Configuration

**pytest.ini**:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80
    --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    backtest: Backtest validation tests
    slow: Slow tests (deselect with '-m "not slow"')
    requires_api: Tests requiring API keys
```

### 5. Pre-Commit Testing

**Local Testing Script** (`scripts/test.sh`):
```bash
#!/bin/bash
set -e

echo "Running quality checks..."
uv run ruff format . --check
uv run ruff check .
uv run pyrefly check

echo "Running tests..."
uv run pytest tests/unit -v
uv run pytest tests/integration -v -m "not requires_api"

echo "All checks passed!"
```

---

## CI/CD Pipeline

### Enhanced CI Pipeline

**`.github/workflows/ci-enhanced.yml`**:

This pipeline includes:
1. **Quality Gates**: Format, lint, type check, security, complexity
2. **Testing Matrix**: Test across multiple scenarios
3. **Coverage Enforcement**: Minimum 80% coverage
4. **Artifact Generation**: Build deployment packages
5. **Performance Testing**: Backtest runtime validation

### Deployment Pipeline

**`.github/workflows/deploy.yml`**:

Triggered on:
- Tags matching `v*.*.*`
- Manual workflow dispatch

Stages:
1. **Pre-deployment validation**: All tests must pass
2. **Build**: Create deployment artifact
3. **Deploy to staging**: Test in staging environment
4. **Smoke tests**: Validate deployment
5. **Deploy to production**: Promote to production
6. **Post-deployment validation**: Health checks

---

## Deployment Options

### Option A: Local Deployment (Recommended)

**Best For**: Retail investors with a home computer or always-on device

**Pros**:
- Zero hosting costs
- Full control
- No cold start delays
- Privacy (data stays local)

**Cons**:
- Requires always-on device
- Manual maintenance
- Limited redundancy

**Setup**:

1. **Windows Task Scheduler** (Windows):
```xml
<!-- Task: Daily Signal Generation -->
<Task>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2025-01-01T07:00:00</StartBoundary>
      <DaysOfWeek>
        <Monday/>
        <Tuesday/>
        <Wednesday/>
        <Thursday/>
        <Friday/>
      </DaysOfWeek>
    </CalendarTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>C:\Users\larai\FinancePortfolio\.venv\Scripts\python.exe</Command>
      <Arguments>-m financeportfolio.signal_generator</Arguments>
      <WorkingDirectory>C:\Users\larai\FinancePortfolio</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
```

2. **Cron** (Linux/Mac):
```bash
# crontab -e
# Daily signal generation at 7 AM (weekdays)
0 7 * * 1-5 cd /home/user/FinancePortfolio && /home/user/FinancePortfolio/.venv/bin/python -m financeportfolio.signal_generator

# Weekly portfolio report (Sunday at 9 AM)
0 9 * * 0 cd /home/user/FinancePortfolio && /home/user/FinancePortfolio/.venv/bin/python -m financeportfolio.report_generator

# Data cache refresh (daily at 6 AM)
0 6 * * 1-5 cd /home/user/FinancePortfolio && /home/user/FinancePortfolio/.venv/bin/python -m financeportfolio.data_updater
```

**Cost**: $0/month (uses existing hardware)

### Option B: Cloud Functions (Serverless)

**Best For**: Those without always-on hardware, occasional execution

**AWS Lambda**:
- Runtime: Python 3.12 container
- Memory: 512MB
- Timeout: 15 minutes
- Invocations: ~60/month (daily + weekly jobs)
- Cost: ~$0.20/month (within free tier)

**GCP Cloud Run**:
- Container: Python 3.12
- Memory: 512MB
- CPU: 1
- Invocations: ~60/month
- Cost: ~$0 (within free tier)

**Setup** (AWS Lambda with CloudWatch Events):
```yaml
# serverless.yml
service: pea-portfolio

provider:
  name: aws
  runtime: python3.12
  region: eu-west-1
  memorySize: 512
  timeout: 900
  environment:
    ANTHROPIC_API_KEY: ${env:ANTHROPIC_API_KEY}
    DATA_BUCKET: pea-portfolio-data

functions:
  signalGenerator:
    handler: financeportfolio.signal_generator.handler
    events:
      - schedule:
          rate: cron(0 7 ? * MON-FRI *)
          description: 'Daily signal generation'

  reportGenerator:
    handler: financeportfolio.report_generator.handler
    events:
      - schedule:
          rate: cron(0 9 ? * SUN *)
          description: 'Weekly portfolio report'

  dataUpdater:
    handler: financeportfolio.data_updater.handler
    events:
      - schedule:
          rate: cron(0 6 ? * MON-FRI *)
          description: 'Daily data cache refresh'
```

**Cost**: $0-2/month

### Option C: VPS (Virtual Private Server)

**Best For**: Those wanting 24/7 availability with full control

**Providers**:
- Hetzner Cloud: 4.15 EUR/month (2 vCPU, 2GB RAM)
- DigitalOcean: $6/month (1 vCPU, 1GB RAM)
- Oracle Cloud: Free tier (ARM instance)

**Setup**:
```bash
# Install dependencies
sudo apt update
sudo apt install python3.12 python3.12-venv git

# Clone and setup
cd /opt
git clone https://github.com/yourusername/FinancePortfolio.git
cd FinancePortfolio
python3.12 -m venv .venv
.venv/bin/pip install uv
.venv/bin/uv sync

# Setup systemd service
sudo cp deployment/pea-portfolio.service /etc/systemd/system/
sudo systemctl enable pea-portfolio.timer
sudo systemctl start pea-portfolio.timer
```

**Cost**: $0-6/month

---

## Scheduling Strategy

### Job Types

#### 1. Data Update Job
**Frequency**: Daily, 6:00 AM (before signal generation)
**Duration**: ~5 minutes
**Purpose**: Refresh market data cache

```python
# financeportfolio/data_updater.py
"""
Daily data cache refresh job.
"""
def update_market_data():
    """Fetch and cache latest market data for all tracked securities."""
    # Fetch CAC 40 eligible stocks
    # Update price data
    # Cache fundamental data
    # Log completion
```

#### 2. Signal Generation Job
**Frequency**: Daily, 7:00 AM (weekdays)
**Duration**: ~10 minutes
**Purpose**: Generate trading signals

```python
# financeportfolio/signal_generator.py
"""
Daily signal generation job.
"""
def generate_signals():
    """Generate buy/sell signals based on current portfolio and market data."""
    # Load current portfolio
    # Calculate signals
    # Apply risk constraints
    # Generate actionable recommendations
    # Send notification if action required
```

#### 3. Report Generation Job
**Frequency**: Weekly, Sunday 9:00 AM
**Duration**: ~15 minutes
**Purpose**: Generate comprehensive portfolio report

```python
# financeportfolio/report_generator.py
"""
Weekly portfolio report generation.
"""
def generate_weekly_report():
    """Generate comprehensive portfolio performance report."""
    # Calculate weekly returns
    # Generate charts
    # Compare to benchmarks
    # Send email report
```

#### 4. Backup Job
**Frequency**: Daily, 11:00 PM
**Duration**: ~2 minutes
**Purpose**: Backup portfolio data and configurations

```python
# financeportfolio/backup.py
"""
Daily backup job.
"""
def backup_data():
    """Backup portfolio data, configurations, and historical signals."""
    # Backup portfolio state
    # Backup historical data
    # Backup configurations
    # Upload to cloud storage
```

### Scheduling Configuration

**GitHub Actions (Cloud Execution)**:
```yaml
# .github/workflows/scheduled-jobs.yml
name: Scheduled Jobs

on:
  schedule:
    # Data update: 6:00 AM UTC (weekdays)
    - cron: '0 6 * * 1-5'
    # Signal generation: 7:00 AM UTC (weekdays)
    - cron: '0 7 * * 1-5'
    # Weekly report: Sunday 9:00 AM UTC
    - cron: '0 9 * * 0'
    # Daily backup: 11:00 PM UTC
    - cron: '0 23 * * *'
  workflow_dispatch:
    inputs:
      job_type:
        description: 'Job to run'
        required: true
        type: choice
        options:
          - data_update
          - signal_generation
          - weekly_report
          - backup
```

---

## Monitoring & Alerting

### Monitoring Strategy

#### 1. Application Logs

**Structure**:
```python
# financeportfolio/logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configure structured logging."""
    logger = logging.getLogger('financeportfolio')
    logger.setLevel(logging.INFO)

    # File handler with rotation
    handler = RotatingFileHandler(
        'logs/portfolio.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=30  # Keep 30 days
    )

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
```

**Log Events**:
- Job start/completion
- Data fetch successes/failures
- Signal generations
- Errors and exceptions
- Performance metrics

#### 2. Health Checks

**Endpoint** (for VPS/Cloud deployments):
```python
# financeportfolio/health.py
from datetime import datetime, timedelta

def health_check():
    """Check system health."""
    checks = {
        'last_data_update': check_last_data_update(),
        'last_signal_generation': check_last_signal(),
        'data_cache_status': check_cache_freshness(),
        'api_connectivity': check_api_access(),
    }

    all_healthy = all(checks.values())

    return {
        'status': 'healthy' if all_healthy else 'degraded',
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    }
```

#### 3. Alerting Rules

**Critical Alerts** (immediate notification):
- Signal generation failure
- Data fetch failure for >24 hours
- Portfolio calculation errors
- API key expiration

**Warning Alerts** (daily digest):
- Unusual portfolio movements
- Low data quality
- Performance degradation

**Notification Channels**:
1. Email (primary)
2. Telegram bot (optional)
3. Discord webhook (optional)

### Monitoring Tools

#### Option A: Self-Hosted (Free)

**Healthchecks.io** (free tier):
- Simple HTTP ping monitoring
- 20 checks free
- Email/SMS alerts

```python
# financeportfolio/monitoring.py
import requests

def ping_healthcheck(job_name: str, status: str = 'success'):
    """Ping healthchecks.io to confirm job completion."""
    url = f"https://hc-ping.com/{HEALTHCHECK_ID}/{job_name}/{status}"
    requests.get(url, timeout=10)
```

#### Option B: Cloud Monitoring

**AWS CloudWatch** (if using Lambda):
- Automatic log collection
- Custom metrics
- Alarms

**Google Cloud Monitoring** (if using Cloud Run):
- Automatic metrics
- Log-based alerts
- Uptime checks

**Cost**: $0-2/month

---

## Security

### Secrets Management

#### 1. GitHub Secrets (for CI/CD)

```yaml
# .github/workflows/ci-enhanced.yml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  YAHOO_FINANCE_API_KEY: ${{ secrets.YAHOO_FINANCE_API_KEY }}
  EMAIL_PASSWORD: ${{ secrets.EMAIL_PASSWORD }}
```

**Required Secrets**:
- `ANTHROPIC_API_KEY`: Claude API key
- `YAHOO_FINANCE_API_KEY`: Financial data API key
- `EMAIL_PASSWORD`: Email notification password
- `BACKUP_S3_KEY`: AWS S3 backup credentials (if using cloud backup)

#### 2. Local Secrets (for local deployment)

**`.env` file** (NEVER commit):
```env
ANTHROPIC_API_KEY=sk-ant-...
YAHOO_FINANCE_API_KEY=...
EMAIL_USER=your_email@example.com
EMAIL_PASSWORD=...
BACKUP_S3_BUCKET=pea-portfolio-backup
BACKUP_S3_KEY=...
BACKUP_S3_SECRET=...
```

**Load with python-dotenv**:
```python
# financeportfolio/config.py
from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
YAHOO_FINANCE_API_KEY = os.getenv('YAHOO_FINANCE_API_KEY')
```

#### 3. Cloud Secrets (for cloud deployment)

**AWS Secrets Manager**:
```bash
# Store secret
aws secretsmanager create-secret \
  --name pea-portfolio/anthropic-api-key \
  --secret-string "sk-ant-..."

# Retrieve in Lambda
import boto3
client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='pea-portfolio/anthropic-api-key')
```

**GCP Secret Manager**:
```bash
# Store secret
echo -n "sk-ant-..." | gcloud secrets create anthropic-api-key --data-file=-

# Retrieve in Cloud Run
from google.cloud import secretmanager
client = secretmanager.SecretManagerServiceClient()
name = "projects/PROJECT_ID/secrets/anthropic-api-key/versions/latest"
response = client.access_secret_version(request={"name": name})
```

### Security Best Practices

1. **Never commit secrets** to version control
2. **Rotate API keys** every 90 days
3. **Use least privilege** IAM roles
4. **Enable 2FA** on all accounts
5. **Encrypt backups** at rest
6. **Use HTTPS** for all API calls
7. **Audit logs** regularly
8. **Keep dependencies updated** (Dependabot)

---

## Backup & Recovery

### Backup Strategy

#### What to Backup

1. **Portfolio State**:
   - Current holdings
   - Transaction history
   - Performance metrics

2. **Historical Data**:
   - Price data cache
   - Signal history
   - Backtest results

3. **Configuration**:
   - Strategy parameters
   - Risk constraints
   - Asset universe

#### Backup Frequency

- **Portfolio state**: Daily (after signal generation)
- **Historical data**: Weekly
- **Configuration**: On change
- **Full backup**: Weekly

#### Backup Storage

**Option A: Local Backup** (Free):
```python
# financeportfolio/backup.py
import shutil
from datetime import datetime

def backup_local():
    """Create local backup."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f'backups/backup_{timestamp}'

    # Backup data directory
    shutil.copytree('data/', f'{backup_dir}/data/')

    # Backup config
    shutil.copy('config.yaml', f'{backup_dir}/config.yaml')

    # Compress
    shutil.make_archive(backup_dir, 'zip', backup_dir)
    shutil.rmtree(backup_dir)
```

**Option B: Cloud Backup** (Recommended):

**AWS S3** (99.999999999% durability):
```python
# financeportfolio/backup.py
import boto3
from datetime import datetime

def backup_to_s3():
    """Backup to AWS S3."""
    s3 = boto3.client('s3')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    bucket = 'pea-portfolio-backup'

    # Upload portfolio state
    s3.upload_file(
        'data/portfolio.json',
        bucket,
        f'portfolio/portfolio_{timestamp}.json'
    )

    # Upload transaction history
    s3.upload_file(
        'data/transactions.csv',
        bucket,
        f'transactions/transactions_{timestamp}.csv'
    )
```

**Cost**: ~$0.10/month (5GB storage)

**Google Cloud Storage**:
```python
from google.cloud import storage

def backup_to_gcs():
    """Backup to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket('pea-portfolio-backup')

    blob = bucket.blob(f'portfolio/portfolio_{timestamp}.json')
    blob.upload_from_filename('data/portfolio.json')
```

**Cost**: ~$0.10/month (5GB storage)

#### Retention Policy

- **Daily backups**: Keep for 30 days
- **Weekly backups**: Keep for 1 year
- **Monthly backups**: Keep indefinitely

#### Recovery Procedure

```python
# financeportfolio/recovery.py
def restore_from_backup(backup_date: str):
    """Restore portfolio state from backup."""
    # Download backup from S3/GCS
    # Verify backup integrity
    # Restore data files
    # Validate portfolio state
    # Log recovery
```

---

## Cost Analysis

### Total Cost of Ownership (Monthly)

#### Scenario 1: Local Deployment (Recommended for Retail)

| Item | Cost |
|------|------|
| Hardware (existing PC) | $0 |
| Electricity (~1W avg) | ~$0.20 |
| Cloud backup (AWS S3) | $0.10 |
| Monitoring (Healthchecks.io free) | $0 |
| **Total** | **$0.30/month** |

**Annual Cost**: ~$3.60

#### Scenario 2: Serverless (AWS Lambda)

| Item | Cost |
|------|------|
| Lambda invocations (60/month) | $0 (free tier) |
| Lambda duration (512MB, 5min avg) | $0.20 |
| CloudWatch Logs | $0.10 |
| S3 Storage | $0.10 |
| **Total** | **$0.40/month** |

**Annual Cost**: ~$4.80

#### Scenario 3: VPS (Budget)

| Item | Cost |
|------|------|
| Hetzner Cloud VPS | $4.15 |
| Backups | $0.10 |
| Monitoring | $0 |
| **Total** | **$4.25/month** |

**Annual Cost**: ~$51

### Cost Optimization Tips

1. **Use free tiers**: AWS, GCP, Oracle Cloud have generous free tiers
2. **Local execution**: If you have always-on hardware, use it
3. **Minimize API calls**: Cache data aggressively
4. **Compress backups**: Reduces storage costs
5. **Use spot instances**: For non-critical workloads
6. **Monitor usage**: Set billing alerts

---

## Implementation Checklist

### Phase 1: Testing Infrastructure
- [ ] Create test directory structure
- [ ] Write unit tests for core logic
- [ ] Write integration tests for data fetching
- [ ] Write backtest validation tests
- [ ] Configure pytest
- [ ] Achieve 80%+ test coverage
- [ ] Setup pre-commit hooks

### Phase 2: CI/CD Pipeline
- [ ] Enhance CI workflow with testing matrix
- [ ] Add deployment workflow
- [ ] Configure GitHub secrets
- [ ] Setup branch protection rules
- [ ] Add status badges to README
- [ ] Test pipeline end-to-end

### Phase 3: Deployment
- [ ] Choose deployment option (local/cloud/VPS)
- [ ] Setup scheduled jobs
- [ ] Configure monitoring
- [ ] Setup alerting
- [ ] Implement backup automation
- [ ] Test recovery procedure

### Phase 4: Security
- [ ] Setup secrets management
- [ ] Enable 2FA on all accounts
- [ ] Rotate initial API keys
- [ ] Audit IAM permissions
- [ ] Enable encryption at rest
- [ ] Setup audit logging

### Phase 5: Production Validation
- [ ] Run end-to-end test with real data
- [ ] Validate signal generation
- [ ] Test alert notifications
- [ ] Verify backup/restore
- [ ] Monitor for 1 week
- [ ] Document any issues

---

## Conclusion

This deployment infrastructure is designed to be:
- **Reliable**: Comprehensive testing catches issues early
- **Cost-Effective**: Multiple options from $0.30 to $5/month
- **Secure**: Proper secrets management and encryption
- **Observable**: Full logging and monitoring
- **Recoverable**: Automated backups with tested recovery

**Recommended Starting Point**: Local deployment with cloud backup ($0.30/month)

**Next Steps**:
1. Implement testing infrastructure
2. Enhance CI/CD pipeline
3. Choose and setup deployment option
4. Configure monitoring
5. Test end-to-end

Remember: **If it's not tested, it's not ready for production!**
