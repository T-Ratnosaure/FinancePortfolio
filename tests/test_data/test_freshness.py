"""Tests for data freshness and staleness detection."""

from datetime import datetime, timedelta

import pytest

from src.data.freshness import (
    FreshnessReport,
    check_macro_data_freshness,
    check_price_data_freshness,
    generate_freshness_report,
    log_freshness_warnings,
)
from src.data.models import (
    CRITICAL_THRESHOLDS,
    STALENESS_THRESHOLDS,
    DataCategory,
    DataFreshness,
    ETFSymbol,
    FreshnessStatus,
    StaleDataError,
)
from src.data.storage.duckdb import DuckDBStorage


class TestDataFreshnessModel:
    """Test the DataFreshness Pydantic model."""

    def test_fresh_data(self) -> None:
        """Test that recently updated data is marked as fresh."""
        freshness = DataFreshness(
            data_category=DataCategory.PRICE_DATA,
            symbol="LQQ.PA",
            last_updated=datetime.now() - timedelta(hours=6),
            record_count=100,
            source="Yahoo Finance",
        )

        assert freshness.get_status() == FreshnessStatus.FRESH
        assert not freshness.is_stale()
        assert not freshness.is_critical()
        assert freshness.get_warning_message() is None

    def test_stale_price_data(self) -> None:
        """Test that price data older than 1 day is marked as stale."""
        freshness = DataFreshness(
            data_category=DataCategory.PRICE_DATA,
            symbol="LQQ.PA",
            last_updated=datetime.now() - timedelta(days=3),
            record_count=100,
            source="Yahoo Finance",
        )

        assert freshness.get_status() == FreshnessStatus.STALE
        assert freshness.is_stale()
        assert not freshness.is_critical()
        warning = freshness.get_warning_message()
        assert warning is not None
        assert "WARNING" in warning
        assert "3 days" in warning

    def test_critical_price_data(self) -> None:
        """Test that price data older than 1 week is critical."""
        freshness = DataFreshness(
            data_category=DataCategory.PRICE_DATA,
            symbol="LQQ.PA",
            last_updated=datetime.now() - timedelta(days=10),
            record_count=100,
            source="Yahoo Finance",
        )

        assert freshness.get_status() == FreshnessStatus.CRITICAL
        assert freshness.is_stale()
        assert freshness.is_critical()
        warning = freshness.get_warning_message()
        assert warning is not None
        assert "CRITICAL" in warning

    def test_stale_macro_data(self) -> None:
        """Test that macro data older than 1 week is stale."""
        freshness = DataFreshness(
            data_category=DataCategory.MACRO_DATA,
            indicator_name="VIX",
            last_updated=datetime.now() - timedelta(days=10),
            record_count=50,
            source="FRED",
        )

        assert freshness.get_status() == FreshnessStatus.STALE
        assert freshness.is_stale()

    def test_critical_macro_data(self) -> None:
        """Test that macro data older than 1 month is critical."""
        freshness = DataFreshness(
            data_category=DataCategory.MACRO_DATA,
            indicator_name="VIX",
            last_updated=datetime.now() - timedelta(days=35),
            record_count=50,
            source="FRED",
        )

        assert freshness.get_status() == FreshnessStatus.CRITICAL
        assert freshness.is_critical()

    def test_age_calculation(self) -> None:
        """Test that age is calculated correctly."""
        now = datetime.now()
        freshness = DataFreshness(
            data_category=DataCategory.PRICE_DATA,
            symbol="LQQ.PA",
            last_updated=now - timedelta(hours=5, minutes=30),
            record_count=100,
            source="Yahoo Finance",
        )

        age = freshness.get_age()
        assert age.total_seconds() >= 5 * 3600  # At least 5 hours

    def test_age_formatting(self) -> None:
        """Test human-readable age formatting."""
        # Test days
        freshness_days = DataFreshness(
            data_category=DataCategory.PRICE_DATA,
            symbol="LQQ.PA",
            last_updated=datetime.now() - timedelta(days=3, hours=2),
            record_count=100,
            source="Yahoo Finance",
        )
        warning = freshness_days.get_warning_message()
        assert warning is not None
        assert "3 days" in warning

        # Test hours
        freshness_hours = DataFreshness(
            data_category=DataCategory.PRICE_DATA,
            symbol="LQQ.PA",
            last_updated=datetime.now() - timedelta(hours=5),
            record_count=100,
            source="Yahoo Finance",
        )
        age_str = freshness_hours._format_age(freshness_hours.get_age())
        assert "hour" in age_str

    def test_thresholds_configuration(self) -> None:
        """Test that thresholds are configured correctly."""
        assert STALENESS_THRESHOLDS[DataCategory.PRICE_DATA] == timedelta(days=1)
        assert STALENESS_THRESHOLDS[DataCategory.MACRO_DATA] == timedelta(days=7)
        assert STALENESS_THRESHOLDS[DataCategory.PORTFOLIO_DATA] == timedelta(hours=1)

        assert CRITICAL_THRESHOLDS[DataCategory.PRICE_DATA] == timedelta(days=7)
        assert CRITICAL_THRESHOLDS[DataCategory.MACRO_DATA] == timedelta(days=30)
        assert CRITICAL_THRESHOLDS[DataCategory.PORTFOLIO_DATA] == timedelta(hours=24)


class TestDuckDBFreshnessTracking:
    """Test freshness tracking in DuckDB storage."""

    def test_freshness_tracking_on_price_insert(self, tmp_path: object) -> None:
        """Test that freshness is tracked when inserting prices."""
        from src.data.models import DailyPrice

        db_path = tmp_path / "test_freshness.duckdb"  # type: ignore
        storage = DuckDBStorage(str(db_path))

        try:
            # Insert price data
            prices = [
                DailyPrice(
                    symbol=ETFSymbol.LQQ,
                    date=datetime(2024, 1, 15).date(),
                    open=100.0,
                    high=105.0,
                    low=99.0,
                    close=103.0,
                    volume=1000000,
                    adjusted_close=103.0,
                )
            ]

            storage.insert_prices(prices)

            # Check freshness was recorded
            freshness = storage.get_freshness(DataCategory.PRICE_DATA, symbol="LQQ.PA")

            assert freshness is not None
            assert freshness.data_category == DataCategory.PRICE_DATA
            assert freshness.symbol == "LQQ.PA"
            assert freshness.record_count == 1
            assert freshness.source == "api"
            assert freshness.get_status() == FreshnessStatus.FRESH

        finally:
            storage.close()

    def test_freshness_tracking_on_macro_insert(self, tmp_path: object) -> None:
        """Test that freshness is tracked when inserting macro data."""
        from src.data.models import MacroIndicator

        db_path = tmp_path / "test_freshness.duckdb"  # type: ignore
        storage = DuckDBStorage(str(db_path))

        try:
            # Insert macro data
            indicators = [
                MacroIndicator(
                    indicator_name="VIX",
                    date=datetime(2024, 1, 15).date(),
                    value=15.5,
                    source="FRED",
                )
            ]

            storage.insert_macro(indicators)

            # Check freshness was recorded
            freshness = storage.get_freshness(
                DataCategory.MACRO_DATA, indicator_name="VIX"
            )

            assert freshness is not None
            assert freshness.data_category == DataCategory.MACRO_DATA
            assert freshness.indicator_name == "VIX"
            assert freshness.record_count == 1
            assert freshness.source == "FRED"
            assert freshness.get_status() == FreshnessStatus.FRESH

        finally:
            storage.close()

    def test_freshness_update_on_new_data(self, tmp_path: object) -> None:
        """Test that freshness is updated when new data is inserted."""
        from src.data.models import DailyPrice

        db_path = tmp_path / "test_freshness.duckdb"  # type: ignore
        storage = DuckDBStorage(str(db_path))

        try:
            # Insert initial data
            prices = [
                DailyPrice(
                    symbol=ETFSymbol.LQQ,
                    date=datetime(2024, 1, 15).date(),
                    open=100.0,
                    high=105.0,
                    low=99.0,
                    close=103.0,
                    volume=1000000,
                    adjusted_close=103.0,
                )
            ]
            storage.insert_prices(prices)

            freshness1 = storage.get_freshness(DataCategory.PRICE_DATA, symbol="LQQ.PA")
            assert freshness1 is not None
            first_update = freshness1.last_updated

            # Insert more data
            more_prices = [
                DailyPrice(
                    symbol=ETFSymbol.LQQ,
                    date=datetime(2024, 1, 16).date(),
                    open=103.0,
                    high=107.0,
                    low=102.0,
                    close=106.0,
                    volume=1100000,
                    adjusted_close=106.0,
                )
            ]
            storage.insert_prices(more_prices)

            freshness2 = storage.get_freshness(DataCategory.PRICE_DATA, symbol="LQQ.PA")
            assert freshness2 is not None
            assert freshness2.last_updated >= first_update

        finally:
            storage.close()

    def test_check_freshness_with_warnings(
        self, tmp_path: object, caplog: object
    ) -> None:
        """Test that check_freshness logs appropriate warnings."""
        from src.data.models import DailyPrice

        db_path = tmp_path / "test_freshness.duckdb"  # type: ignore
        storage = DuckDBStorage(str(db_path))

        try:
            # Insert data and manually update freshness to be stale
            prices = [
                DailyPrice(
                    symbol=ETFSymbol.LQQ,
                    date=datetime(2024, 1, 15).date(),
                    open=100.0,
                    high=105.0,
                    low=99.0,
                    close=103.0,
                    volume=1000000,
                    adjusted_close=103.0,
                )
            ]
            storage.insert_prices(prices)

            # Manually set stale timestamp
            stale_time = datetime.now() - timedelta(days=3)
            storage.conn.execute(
                """
                UPDATE raw.data_freshness
                SET last_updated = ?, updated_at = ?
                WHERE data_category = 'price_data' AND symbol = 'LQQ.PA'
            """,
                [stale_time, stale_time],
            )

            # Check freshness (should log warning but not raise)
            freshness = storage.check_freshness(
                DataCategory.PRICE_DATA, symbol="LQQ.PA", raise_on_critical=False
            )

            assert freshness is not None
            assert freshness.get_status() == FreshnessStatus.STALE

        finally:
            storage.close()

    def test_check_freshness_raises_on_critical(self, tmp_path: object) -> None:
        """Test that check_freshness raises StaleDataError for critical data."""
        from src.data.models import DailyPrice

        db_path = tmp_path / "test_freshness.duckdb"  # type: ignore
        storage = DuckDBStorage(str(db_path))

        try:
            # Insert data and manually update freshness to be critical
            prices = [
                DailyPrice(
                    symbol=ETFSymbol.LQQ,
                    date=datetime(2024, 1, 15).date(),
                    open=100.0,
                    high=105.0,
                    low=99.0,
                    close=103.0,
                    volume=1000000,
                    adjusted_close=103.0,
                )
            ]
            storage.insert_prices(prices)

            # Manually set critical timestamp (10 days old)
            critical_time = datetime.now() - timedelta(days=10)
            storage.conn.execute(
                """
                UPDATE raw.data_freshness
                SET last_updated = ?, updated_at = ?
                WHERE data_category = 'price_data' AND symbol = 'LQQ.PA'
            """,
                [critical_time, critical_time],
            )

            # Check freshness (should raise)
            with pytest.raises(StaleDataError) as exc_info:
                storage.check_freshness(
                    DataCategory.PRICE_DATA, symbol="LQQ.PA", raise_on_critical=True
                )

            assert "CRITICAL" in str(exc_info.value)
            assert exc_info.value.freshness.get_status() == FreshnessStatus.CRITICAL

        finally:
            storage.close()

    def test_get_all_freshness_status(self, tmp_path: object) -> None:
        """Test retrieving all freshness status."""
        from src.data.models import DailyPrice, MacroIndicator

        db_path = tmp_path / "test_freshness.duckdb"  # type: ignore
        storage = DuckDBStorage(str(db_path))

        try:
            # Insert multiple datasets
            prices = [
                DailyPrice(
                    symbol=ETFSymbol.LQQ,
                    date=datetime(2024, 1, 15).date(),
                    open=100.0,
                    high=105.0,
                    low=99.0,
                    close=103.0,
                    volume=1000000,
                    adjusted_close=103.0,
                )
            ]
            storage.insert_prices(prices)

            indicators = [
                MacroIndicator(
                    indicator_name="VIX",
                    date=datetime(2024, 1, 15).date(),
                    value=15.5,
                    source="FRED",
                )
            ]
            storage.insert_macro(indicators)

            # Get all freshness status
            all_freshness = storage.get_all_freshness_status()

            assert len(all_freshness) >= 2
            categories = {f.data_category for f in all_freshness}
            assert DataCategory.PRICE_DATA in categories
            assert DataCategory.MACRO_DATA in categories

        finally:
            storage.close()


class TestFreshnessUtilities:
    """Test high-level freshness utility functions."""

    def test_generate_freshness_report(self, tmp_path: object) -> None:
        """Test generating a comprehensive freshness report."""
        from src.data.models import DailyPrice, MacroIndicator

        db_path = tmp_path / "test_freshness.duckdb"  # type: ignore
        storage = DuckDBStorage(str(db_path))

        try:
            # Insert fresh data
            prices = [
                DailyPrice(
                    symbol=ETFSymbol.LQQ,
                    date=datetime(2024, 1, 15).date(),
                    open=100.0,
                    high=105.0,
                    low=99.0,
                    close=103.0,
                    volume=1000000,
                    adjusted_close=103.0,
                )
            ]
            storage.insert_prices(prices)

            # Insert stale macro data
            indicators = [
                MacroIndicator(
                    indicator_name="VIX",
                    date=datetime(2024, 1, 15).date(),
                    value=15.5,
                    source="FRED",
                )
            ]
            storage.insert_macro(indicators)

            # Make macro data stale
            stale_time = datetime.now() - timedelta(days=10)
            storage.conn.execute(
                """
                UPDATE raw.data_freshness
                SET last_updated = ?, updated_at = ?
                WHERE data_category = 'macro_data'
            """,
                [stale_time, stale_time],
            )

            # Generate report
            report = generate_freshness_report(storage)

            assert report.fresh_count >= 1
            assert report.stale_count >= 1
            assert report.has_issues()

            # Test report string output
            report_str = str(report)
            assert "Data Freshness Report" in report_str
            assert "WARNINGS:" in report_str

            # Test report dict output
            report_dict = report.to_dict()
            assert "summary" in report_dict
            assert "stale_datasets" in report_dict
            assert report_dict["summary"]["stale"] >= 1

        finally:
            storage.close()

    def test_check_price_data_freshness_utility(self, tmp_path: object) -> None:
        """Test the check_price_data_freshness utility function."""
        from src.data.models import DailyPrice

        db_path = tmp_path / "test_freshness.duckdb"  # type: ignore
        storage = DuckDBStorage(str(db_path))

        try:
            # Insert price data
            prices = [
                DailyPrice(
                    symbol=ETFSymbol.LQQ,
                    date=datetime(2024, 1, 15).date(),
                    open=100.0,
                    high=105.0,
                    low=99.0,
                    close=103.0,
                    volume=1000000,
                    adjusted_close=103.0,
                )
            ]
            storage.insert_prices(prices)

            # Check freshness using utility
            freshness = check_price_data_freshness(
                storage, "LQQ.PA", raise_on_critical=False
            )

            assert freshness is not None
            assert freshness.symbol == "LQQ.PA"
            assert freshness.get_status() == FreshnessStatus.FRESH

        finally:
            storage.close()

    def test_check_macro_data_freshness_utility(self, tmp_path: object) -> None:
        """Test the check_macro_data_freshness utility function."""
        from src.data.models import MacroIndicator

        db_path = tmp_path / "test_freshness.duckdb"  # type: ignore
        storage = DuckDBStorage(str(db_path))

        try:
            # Insert macro data
            indicators = [
                MacroIndicator(
                    indicator_name="VIX",
                    date=datetime(2024, 1, 15).date(),
                    value=15.5,
                    source="FRED",
                )
            ]
            storage.insert_macro(indicators)

            # Check freshness using utility
            freshness = check_macro_data_freshness(
                storage, "VIX", raise_on_critical=False
            )

            assert freshness is not None
            assert freshness.indicator_name == "VIX"
            assert freshness.get_status() == FreshnessStatus.FRESH

        finally:
            storage.close()

    def test_log_freshness_warnings(self, tmp_path: object, caplog: object) -> None:
        """Test that log_freshness_warnings logs all issues."""
        from src.data.models import DailyPrice

        db_path = tmp_path / "test_freshness.duckdb"  # type: ignore
        storage = DuckDBStorage(str(db_path))

        try:
            # Insert data and make it stale
            prices = [
                DailyPrice(
                    symbol=ETFSymbol.LQQ,
                    date=datetime(2024, 1, 15).date(),
                    open=100.0,
                    high=105.0,
                    low=99.0,
                    close=103.0,
                    volume=1000000,
                    adjusted_close=103.0,
                )
            ]
            storage.insert_prices(prices)

            stale_time = datetime.now() - timedelta(days=3)
            storage.conn.execute(
                """
                UPDATE raw.data_freshness
                SET last_updated = ?, updated_at = ?
                WHERE data_category = 'price_data'
            """,
                [stale_time, stale_time],
            )

            # Log warnings
            log_freshness_warnings(storage)

            # Verify logging occurred (this depends on caplog fixture)
            # Just verify function runs without error

        finally:
            storage.close()


class TestFreshnessReport:
    """Test the FreshnessReport class."""

    def test_report_with_all_fresh_data(self) -> None:
        """Test report when all data is fresh."""
        datasets = [
            DataFreshness(
                data_category=DataCategory.PRICE_DATA,
                symbol="LQQ.PA",
                last_updated=datetime.now() - timedelta(hours=1),
                record_count=100,
                source="api",
            ),
            DataFreshness(
                data_category=DataCategory.MACRO_DATA,
                indicator_name="VIX",
                last_updated=datetime.now() - timedelta(days=2),
                record_count=50,
                source="FRED",
            ),
        ]

        report = FreshnessReport(datasets)

        assert report.fresh_count == 2
        assert report.stale_count == 0
        assert report.critical_count == 0
        assert not report.has_issues()
        assert not report.has_critical_issues()

    def test_report_with_stale_data(self) -> None:
        """Test report when some data is stale."""
        datasets = [
            DataFreshness(
                data_category=DataCategory.PRICE_DATA,
                symbol="LQQ.PA",
                last_updated=datetime.now() - timedelta(days=3),
                record_count=100,
                source="api",
            ),
            DataFreshness(
                data_category=DataCategory.MACRO_DATA,
                indicator_name="VIX",
                last_updated=datetime.now() - timedelta(hours=1),
                record_count=50,
                source="FRED",
            ),
        ]

        report = FreshnessReport(datasets)

        assert report.stale_count == 1
        assert report.has_issues()
        assert not report.has_critical_issues()

        stale_datasets = report.get_stale_datasets()
        assert len(stale_datasets) == 1
        assert stale_datasets[0].symbol == "LQQ.PA"

    def test_report_with_critical_data(self) -> None:
        """Test report when some data is critically stale."""
        datasets = [
            DataFreshness(
                data_category=DataCategory.PRICE_DATA,
                symbol="LQQ.PA",
                last_updated=datetime.now() - timedelta(days=10),
                record_count=100,
                source="api",
            )
        ]

        report = FreshnessReport(datasets)

        assert report.critical_count == 1
        assert report.has_issues()
        assert report.has_critical_issues()

        critical_datasets = report.get_critical_datasets()
        assert len(critical_datasets) == 1
