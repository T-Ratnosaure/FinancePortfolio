"""Pytest fixtures for paper trading tests."""

import tempfile
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

from src.paper_trading.models import SessionConfig
from src.paper_trading.storage import PaperTradingStorage


@pytest.fixture
def temp_db_path() -> Path:
    """Create a temporary database path (path only, not the file)."""
    temp_dir = Path(tempfile.gettempdir())
    return temp_dir / f"test_paper_trading_{uuid.uuid4().hex}.duckdb"


@pytest.fixture
def storage(temp_db_path: Path) -> PaperTradingStorage:
    """Create a paper trading storage instance."""
    return PaperTradingStorage(temp_db_path)


@pytest.fixture
def sample_config() -> SessionConfig:
    """Create a sample session configuration."""
    return SessionConfig(
        session_name="Test Session",
        initial_capital=Decimal("10000.00"),
        currency="EUR",
        auto_rebalance=False,
        rebalance_threshold=0.05,
    )


@pytest.fixture
def session_id() -> str:
    """Create a sample session ID."""
    return "test-session-001"
