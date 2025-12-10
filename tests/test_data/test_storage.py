"""Tests for DuckDB storage layer."""

import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from src.data.models import (
    DailyPrice,
    ETFSymbol,
    MacroIndicator,
    Position,
    Trade,
    TradeAction,
)
from src.data.storage.duckdb import DuckDBStorage


@pytest.fixture
def temp_db() -> str:
    """Create a temporary database file path (not the file itself)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_portfolio.duckdb")


@pytest.fixture
def storage(temp_db: str) -> DuckDBStorage:
    """Create a DuckDBStorage instance with temporary database."""
    db = DuckDBStorage(temp_db)
    yield db
    db.close()


class TestDuckDBStorageInit:
    """Tests for DuckDB storage initialization."""

    def test_creates_database_file(self, temp_db: str) -> None:
        """Test that database file is created."""
        storage = DuckDBStorage(temp_db)
        assert Path(temp_db).exists()
        storage.close()

    def test_creates_schemas(self, storage: DuckDBStorage) -> None:
        """Test that all schemas are created."""
        result = storage.conn.execute(
            "SELECT schema_name FROM information_schema.schemata"
        ).fetchall()
        schema_names = [row[0] for row in result]
        assert "raw" in schema_names
        assert "cleaned" in schema_names
        assert "analytics" in schema_names

    def test_context_manager(self, temp_db: str) -> None:
        """Test context manager usage."""
        with DuckDBStorage(temp_db) as storage:
            assert storage.conn is not None


class TestPriceOperations:
    """Tests for price data operations."""

    def test_insert_prices(self, storage: DuckDBStorage) -> None:
        """Test inserting daily price data."""
        prices = [
            DailyPrice(
                symbol=ETFSymbol.LQQ,
                date=date(2024, 1, 15),
                open=Decimal("100.00"),
                high=Decimal("102.00"),
                low=Decimal("99.00"),
                close=Decimal("101.00"),
                volume=10000,
                adjusted_close=Decimal("101.00"),
            ),
            DailyPrice(
                symbol=ETFSymbol.LQQ,
                date=date(2024, 1, 16),
                open=Decimal("101.00"),
                high=Decimal("103.00"),
                low=Decimal("100.50"),
                close=Decimal("102.50"),
                volume=12000,
                adjusted_close=Decimal("102.50"),
            ),
        ]
        count = storage.insert_prices(prices)
        assert count == 2

    def test_get_prices(self, storage: DuckDBStorage) -> None:
        """Test querying price data."""
        prices = [
            DailyPrice(
                symbol=ETFSymbol.LQQ,
                date=date(2024, 1, 15),
                open=Decimal("100.00"),
                high=Decimal("102.00"),
                low=Decimal("99.00"),
                close=Decimal("101.00"),
                volume=10000,
                adjusted_close=Decimal("101.00"),
            ),
        ]
        storage.insert_prices(prices)

        result = storage.get_prices(
            ETFSymbol.LQQ.value, date(2024, 1, 1), date(2024, 1, 31)
        )
        assert len(result) == 1
        assert result[0].symbol == ETFSymbol.LQQ

    def test_get_latest_prices(self, storage: DuckDBStorage) -> None:
        """Test getting latest prices for each ETF."""
        prices = [
            DailyPrice(
                symbol=ETFSymbol.LQQ,
                date=date(2024, 1, 15),
                open=Decimal("100.00"),
                high=Decimal("102.00"),
                low=Decimal("99.00"),
                close=Decimal("101.00"),
                volume=10000,
                adjusted_close=Decimal("101.00"),
            ),
            DailyPrice(
                symbol=ETFSymbol.LQQ,
                date=date(2024, 1, 16),
                open=Decimal("101.00"),
                high=Decimal("103.00"),
                low=Decimal("100.50"),
                close=Decimal("102.50"),
                volume=12000,
                adjusted_close=Decimal("102.50"),
            ),
        ]
        storage.insert_prices(prices)

        latest = storage.get_latest_prices()
        assert ETFSymbol.LQQ.value in latest
        assert latest[ETFSymbol.LQQ.value].date == date(2024, 1, 16)


class TestMacroOperations:
    """Tests for macro indicator operations."""

    def test_insert_macro_indicators(self, storage: DuckDBStorage) -> None:
        """Test inserting macro indicator data."""
        indicators = [
            MacroIndicator(
                indicator_name="VIX",
                date=date(2024, 1, 15),
                value=15.5,
                source="FRED",
            ),
            MacroIndicator(
                indicator_name="DGS10",
                date=date(2024, 1, 15),
                value=4.25,
                source="FRED",
            ),
        ]
        count = storage.insert_macro(indicators)
        assert count == 2


class TestPositionOperations:
    """Tests for position operations."""

    def test_insert_and_get_positions(self, storage: DuckDBStorage) -> None:
        """Test inserting and retrieving positions."""
        position = Position(
            symbol=ETFSymbol.WPEA,
            shares=100.0,
            average_cost=50.0,
            current_price=55.0,
            market_value=5500.0,
            unrealized_pnl=500.0,
            weight=0.60,
        )
        storage.insert_position(position)

        positions = storage.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == ETFSymbol.WPEA
        assert positions[0].shares == 100.0


class TestTradeOperations:
    """Tests for trade operations."""

    def test_insert_trade(self, storage: DuckDBStorage) -> None:
        """Test inserting a trade."""
        trade = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime(2024, 1, 15, 10, 30, 0),
            action=TradeAction.BUY,
            shares=10.0,
            price=100.0,
            total_value=1000.0,
            commission=1.0,
        )
        storage.insert_trade(trade)

        trades = storage.get_trades()
        assert len(trades) == 1
        assert trades[0].action == TradeAction.BUY
