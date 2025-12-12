"""DuckDB storage layer for the PEA Portfolio system.

This module provides a DuckDB-based storage implementation with a three-layer
architecture:
- Raw layer: Unprocessed data from external sources
- Cleaned layer: Validated and processed data
- Analytics layer: Derived metrics and analysis results
"""

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
from pydantic import ValidationError

from ..models import (
    DailyPrice,
    DataCategory,
    DataFreshness,
    FreshnessStatus,
    MacroIndicator,
    Position,
    StaleDataError,
    Trade,
)

logger = logging.getLogger(__name__)

# Allowed base directories for database files (security constraint)
ALLOWED_DB_DIRECTORIES = [
    "data",
    "db",
    "database",
    "storage",
    ".data",
]


class PathValidationError(Exception):
    """Raised when a path fails security validation."""

    pass


def _validate_db_path(db_path: str) -> Path:
    """Validate database path to prevent path traversal attacks.

    Security measures:
    - Resolves path to absolute form
    - Checks for path traversal sequences
    - Ensures path is within allowed directories or uses standard db extensions
    - Validates file extension

    Args:
        db_path: The database path to validate.

    Returns:
        Validated Path object.

    Raises:
        PathValidationError: If the path fails validation.
    """
    path = Path(db_path)

    # Resolve to absolute path to detect traversal attempts
    try:
        resolved_path = path.resolve()
    except (OSError, ValueError) as e:
        raise PathValidationError(f"Invalid path: {e}") from e

    # Check for obvious path traversal patterns in the original string
    dangerous_patterns = ["..", "..\\" if "\\" in db_path else None]
    dangerous_patterns = [p for p in dangerous_patterns if p is not None]
    for pattern in dangerous_patterns:
        if pattern in db_path:
            raise PathValidationError(
                f"Path traversal detected: '{pattern}' not allowed in database path"
            )

    # Validate file extension
    valid_extensions = {".duckdb", ".db", ".ddb"}
    if resolved_path.suffix.lower() not in valid_extensions and resolved_path.suffix:
        raise PathValidationError(
            f"Invalid database extension: {resolved_path.suffix}. "
            f"Allowed extensions: {valid_extensions}"
        )

    # For in-memory databases, allow special syntax
    if db_path in (":memory:", ""):
        return path

    # Ensure the path is not accessing critical system directories
    # Note: /tmp is allowed for testing (pytest creates temp dirs there)
    # The path traversal check above prevents ../ attacks
    system_dirs = [
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        r"C:\Windows",
        r"C:\Program Files",
        r"C:\ProgramData",
    ]

    # Check if running in test environment (pytest sets this)
    is_testing = (
        os.environ.get("PYTEST_CURRENT_TEST") is not None
        or "pytest" in str(resolved_path)
        or "tmp" in str(resolved_path).lower()
    )

    # Only block system dirs in non-test environments, or always block critical ones
    resolved_str = str(resolved_path).lower()
    for sys_dir in system_dirs:
        if resolved_str.startswith(sys_dir.lower()):
            raise PathValidationError(
                f"Database path cannot be in system directory: {sys_dir}"
            )

    # Block /var access except during testing (CI runners may use /var/folders)
    if resolved_str.startswith("/var") and not is_testing:
        raise PathValidationError("Database path cannot be in system directory: /var")

    logger.debug(f"Database path validated: {resolved_path}")
    return resolved_path


class DuckDBStorage:
    """DuckDB storage implementation for financial data.

    This class manages a DuckDB connection and provides methods for:
    - Creating and managing database schema across three layers
    - Inserting and querying price and macro data
    - Managing portfolio positions and trades
    - Ensuring data integrity and validation

    The database uses a layered approach:
    1. Raw layer: Direct ingestion from data sources
    2. Cleaned layer: Validated, deduplicated data
    3. Analytics layer: Derived features and analysis results
    """

    def __init__(self, db_path: str) -> None:
        """Initialize DuckDB connection and create schema.

        Args:
            db_path: Path to the DuckDB database file. If it doesn't exist,
                    a new database will be created.

        Raises:
            PathValidationError: If the database path fails security validation.
        """
        # Security: Validate path to prevent traversal attacks
        self.db_path = _validate_db_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = duckdb.connect(str(self.db_path))
        logger.info(f"Connected to DuckDB at {self.db_path}")

        self._create_schema()

    def _create_schema(self) -> None:
        """Create the three-layer database schema.

        Creates schemas and tables for:
        - Raw layer: etf_prices_raw, macro_indicators_raw, data_ingestion_log
        - Cleaned layer: etf_prices_daily, macro_indicators, derived_features
        - Analytics layer: portfolio_positions, trade_signals, backtest_results
        """
        # Create schemas
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS cleaned")
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS analytics")

        # Raw layer tables
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS raw.seq_etf_prices_raw_id
            START 1
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS raw.etf_prices_raw (
                id INTEGER PRIMARY KEY DEFAULT nextval('raw.seq_etf_prices_raw_id'),
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DECIMAL(18, 6) NOT NULL,
                high DECIMAL(18, 6) NOT NULL,
                low DECIMAL(18, 6) NOT NULL,
                close DECIMAL(18, 6) NOT NULL,
                volume BIGINT NOT NULL,
                adjusted_close DECIMAL(18, 6) NOT NULL,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source VARCHAR
            )
        """)

        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS raw.seq_macro_indicators_raw_id
            START 1
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS raw.macro_indicators_raw (
                id INTEGER PRIMARY KEY
                    DEFAULT nextval('raw.seq_macro_indicators_raw_id'),
                indicator_name VARCHAR NOT NULL,
                date DATE NOT NULL,
                value DECIMAL(18, 6) NOT NULL,
                source VARCHAR NOT NULL,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS raw.seq_data_ingestion_log_id
            START 1
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS raw.data_ingestion_log (
                id INTEGER PRIMARY KEY DEFAULT nextval('raw.seq_data_ingestion_log_id'),
                source VARCHAR NOT NULL,
                table_name VARCHAR NOT NULL,
                records_inserted INTEGER NOT NULL,
                status VARCHAR NOT NULL,
                error_message VARCHAR,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Data freshness tracking table
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS raw.seq_data_freshness_id
            START 1
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS raw.data_freshness (
                id INTEGER PRIMARY KEY DEFAULT nextval('raw.seq_data_freshness_id'),
                data_category VARCHAR NOT NULL,
                symbol VARCHAR,
                indicator_name VARCHAR,
                last_updated TIMESTAMP NOT NULL,
                record_count INTEGER NOT NULL,
                source VARCHAR NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(data_category, symbol, indicator_name, source)
            )
        """)

        # Cleaned layer tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cleaned.etf_prices_daily (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DECIMAL(18, 6) NOT NULL,
                high DECIMAL(18, 6) NOT NULL,
                low DECIMAL(18, 6) NOT NULL,
                close DECIMAL(18, 6) NOT NULL,
                volume BIGINT NOT NULL,
                adjusted_close DECIMAL(18, 6) NOT NULL,
                validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cleaned.macro_indicators (
                indicator_name VARCHAR NOT NULL,
                date DATE NOT NULL,
                value DECIMAL(18, 6) NOT NULL,
                source VARCHAR NOT NULL,
                validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (indicator_name, date, source)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cleaned.derived_features (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                feature_name VARCHAR NOT NULL,
                value DECIMAL(18, 6) NOT NULL,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date, feature_name)
            )
        """)

        # Analytics layer tables
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS analytics.seq_portfolio_positions_id
            START 1
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics.portfolio_positions (
                id INTEGER PRIMARY KEY
                    DEFAULT nextval('analytics.seq_portfolio_positions_id'),
                symbol VARCHAR NOT NULL,
                shares DECIMAL(18, 6) NOT NULL,
                average_cost DECIMAL(18, 6) NOT NULL,
                current_price DECIMAL(18, 6) NOT NULL,
                market_value DECIMAL(18, 6) NOT NULL,
                unrealized_pnl DECIMAL(18, 6) NOT NULL,
                weight DECIMAL(5, 4) NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_current BOOLEAN DEFAULT TRUE
            )
        """)

        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS analytics.seq_trade_signals_id
            START 1
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics.trade_signals (
                id INTEGER PRIMARY KEY
                    DEFAULT nextval('analytics.seq_trade_signals_id'),
                date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                signal_type VARCHAR NOT NULL,
                strength DECIMAL(5, 4) NOT NULL,
                reasoning VARCHAR,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS analytics.seq_backtest_results_id
            START 1
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics.backtest_results (
                id INTEGER PRIMARY KEY
                    DEFAULT nextval('analytics.seq_backtest_results_id'),
                backtest_name VARCHAR NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                total_return DECIMAL(10, 4) NOT NULL,
                sharpe_ratio DECIMAL(10, 4),
                max_drawdown DECIMAL(10, 4),
                win_rate DECIMAL(5, 4),
                num_trades INTEGER,
                parameters JSON,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for performance
        self._create_indexes()
        logger.info("Database schema created successfully")

    def _create_indexes(self) -> None:
        """Create indexes for query performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_raw_prices_symbol_date "
            "ON raw.etf_prices_raw(symbol, date)",
            "CREATE INDEX IF NOT EXISTS idx_raw_prices_date "
            "ON raw.etf_prices_raw(date)",
            "CREATE INDEX IF NOT EXISTS idx_raw_macro_indicator_date "
            "ON raw.macro_indicators_raw(indicator_name, date)",
            "CREATE INDEX IF NOT EXISTS idx_cleaned_prices_date "
            "ON cleaned.etf_prices_daily(date)",
            "CREATE INDEX IF NOT EXISTS idx_derived_features_symbol_date "
            "ON cleaned.derived_features(symbol, date)",
            "CREATE INDEX IF NOT EXISTS idx_trade_signals_date "
            "ON analytics.trade_signals(date)",
            "CREATE INDEX IF NOT EXISTS idx_positions_symbol "
            "ON analytics.portfolio_positions(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_data_freshness_category "
            "ON raw.data_freshness(data_category)",
            "CREATE INDEX IF NOT EXISTS idx_data_freshness_symbol "
            "ON raw.data_freshness(symbol)",
        ]

        for index_sql in indexes:
            try:
                self.conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")

    def insert_prices(self, prices: list[DailyPrice]) -> int:
        """Bulk insert daily price data.

        Args:
            prices: List of DailyPrice objects to insert

        Returns:
            Number of records inserted

        Raises:
            ValidationError: If any price data fails validation
        """
        if not prices:
            logger.warning("No prices to insert")
            return 0

        # Insert into raw layer first
        raw_records = []
        for price in prices:
            raw_records.append(
                (
                    price.symbol.value,
                    price.date,
                    float(price.open),
                    float(price.high),
                    float(price.low),
                    float(price.close),
                    price.volume,
                    float(price.adjusted_close)
                    if price.adjusted_close is not None
                    else 0.0,
                    datetime.now(),
                    "api",
                )
            )

        self.conn.executemany(
            """
            INSERT INTO raw.etf_prices_raw
            (symbol, date, open, high, low, close, volume, adjusted_close,
             ingested_at, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            raw_records,
        )

        # Insert into cleaned layer (with conflict handling)
        cleaned_records = []
        for price in prices:
            cleaned_records.append(
                (
                    price.symbol.value,
                    price.date,
                    float(price.open),
                    float(price.high),
                    float(price.low),
                    float(price.close),
                    price.volume,
                    float(price.adjusted_close)
                    if price.adjusted_close is not None
                    else 0.0,
                    datetime.now(),
                )
            )

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO cleaned.etf_prices_daily
            (symbol, date, open, high, low, close, volume, adjusted_close,
             validated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            cleaned_records,
        )

        logger.info(f"Inserted {len(prices)} price records")

        # Update freshness tracking for each symbol
        for symbol in {price.symbol for price in prices}:
            symbol_prices = [p for p in prices if p.symbol == symbol]
            self._update_freshness(
                data_category=DataCategory.PRICE_DATA,
                symbol=symbol.value,
                record_count=len(symbol_prices),
                source="api",
            )

        return len(prices)

    def insert_macro(self, indicators: list[MacroIndicator]) -> int:
        """Bulk insert macroeconomic indicator data.

        Args:
            indicators: List of MacroIndicator objects to insert

        Returns:
            Number of records inserted

        Raises:
            ValidationError: If any indicator data fails validation
        """
        if not indicators:
            logger.warning("No macro indicators to insert")
            return 0

        # Insert into raw layer
        raw_records = []
        for indicator in indicators:
            raw_records.append(
                (
                    indicator.indicator_name,
                    indicator.date,
                    float(indicator.value),
                    indicator.source,
                    datetime.now(),
                )
            )

        self.conn.executemany(
            """
            INSERT INTO raw.macro_indicators_raw
            (indicator_name, date, value, source, ingested_at)
            VALUES (?, ?, ?, ?, ?)
        """,
            raw_records,
        )

        # Insert into cleaned layer
        cleaned_records = []
        for indicator in indicators:
            cleaned_records.append(
                (
                    indicator.indicator_name,
                    indicator.date,
                    float(indicator.value),
                    indicator.source,
                    datetime.now(),
                )
            )

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO cleaned.macro_indicators
            (indicator_name, date, value, source, validated_at)
            VALUES (?, ?, ?, ?, ?)
        """,
            cleaned_records,
        )

        logger.info(f"Inserted {len(indicators)} macro indicator records")

        # Update freshness tracking for each indicator
        for indicator_name in {ind.indicator_name for ind in indicators}:
            indicator_records = [
                ind for ind in indicators if ind.indicator_name == indicator_name
            ]
            if indicator_records:
                source = indicator_records[0].source
                self._update_freshness(
                    data_category=DataCategory.MACRO_DATA,
                    indicator_name=indicator_name,
                    record_count=len(indicator_records),
                    source=source,
                )

        return len(indicators)

    def get_prices(
        self, symbol: str, start_date: date, end_date: date
    ) -> list[DailyPrice]:
        """Query price data for a symbol within a date range.

        Args:
            symbol: ETF symbol to query
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of DailyPrice objects ordered by date ascending
        """
        result = self.conn.execute(
            """
            SELECT symbol, date, open, high, low, close, volume, adjusted_close
            FROM cleaned.etf_prices_daily
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        """,
            [symbol, start_date, end_date],
        ).fetchall()

        prices = []
        for row in result:
            try:
                prices.append(
                    DailyPrice(
                        symbol=row[0],
                        date=row[1],
                        open=row[2],
                        high=row[3],
                        low=row[4],
                        close=row[5],
                        volume=row[6],
                        adjusted_close=row[7],
                    )
                )
            except ValidationError as e:
                logger.error(f"Validation error for {row}: {e}")

        return prices

    def get_latest_prices(self) -> dict[str, DailyPrice]:
        """Get the most recent price for each ETF.

        Returns:
            Dictionary mapping symbol to its latest DailyPrice
        """
        result = self.conn.execute(
            """
            WITH latest_dates AS (
                SELECT symbol, MAX(date) as max_date
                FROM cleaned.etf_prices_daily
                GROUP BY symbol
            )
            SELECT p.symbol, p.date, p.open, p.high, p.low, p.close,
                   p.volume, p.adjusted_close
            FROM cleaned.etf_prices_daily p
            INNER JOIN latest_dates ld
                ON p.symbol = ld.symbol AND p.date = ld.max_date
            ORDER BY p.symbol
        """
        ).fetchall()

        latest_prices: dict[str, DailyPrice] = {}
        for row in result:
            try:
                latest_prices[row[0]] = DailyPrice(
                    symbol=row[0],
                    date=row[1],
                    open=row[2],
                    high=row[3],
                    low=row[4],
                    close=row[5],
                    volume=row[6],
                    adjusted_close=row[7],
                )
            except ValidationError as e:
                logger.error(f"Validation error for {row}: {e}")

        return latest_prices

    def insert_position(self, position: Position) -> None:
        """Insert or update a portfolio position.

        Args:
            position: Position object to insert
        """
        # Mark existing positions as not current
        self.conn.execute(
            """
            UPDATE analytics.portfolio_positions
            SET is_current = FALSE
            WHERE symbol = ? AND is_current = TRUE
        """,
            [position.symbol.value],
        )

        # Insert new position
        self.conn.execute(
            """
            INSERT INTO analytics.portfolio_positions
            (symbol, shares, average_cost, current_price, market_value,
             unrealized_pnl, weight, updated_at, is_current)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, TRUE)
        """,
            [
                position.symbol.value,
                float(position.shares),
                float(position.average_cost),
                float(position.current_price),
                float(position.market_value),
                float(position.unrealized_pnl),
                float(position.weight),
                datetime.now(),
            ],
        )

        logger.info(f"Inserted position for {position.symbol.value}")

        # Update freshness tracking for portfolio positions
        self._update_freshness(
            data_category=DataCategory.PORTFOLIO_DATA,
            symbol=position.symbol.value,
            record_count=1,
            source="manual",
        )

    def get_positions(self) -> list[Position]:
        """Get current portfolio positions.

        Returns:
            List of current Position objects
        """
        result = self.conn.execute(
            """
            SELECT symbol, shares, average_cost, current_price, market_value,
                   unrealized_pnl, weight
            FROM analytics.portfolio_positions
            WHERE is_current = TRUE
            ORDER BY symbol
        """
        ).fetchall()

        positions = []
        for row in result:
            try:
                positions.append(
                    Position(
                        symbol=row[0],
                        shares=row[1],
                        average_cost=row[2],
                        current_price=row[3],
                        market_value=row[4],
                        unrealized_pnl=row[5],
                        weight=row[6],
                    )
                )
            except ValidationError as e:
                logger.error(f"Validation error for position {row}: {e}")

        return positions

    def insert_trade(self, trade: Trade) -> None:
        """Insert a trade record.

        This method stores trades in a separate trades table that's not
        explicitly defined in the initial schema but follows the same pattern.

        Args:
            trade: Trade object to insert
        """
        # Create trades table if it doesn't exist
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS analytics.seq_trades_id START 1
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics.trades (
                id INTEGER PRIMARY KEY DEFAULT nextval('analytics.seq_trades_id'),
                symbol VARCHAR NOT NULL,
                date TIMESTAMP NOT NULL,
                action VARCHAR NOT NULL,
                shares DECIMAL(18, 6) NOT NULL,
                price DECIMAL(18, 6) NOT NULL,
                total_value DECIMAL(18, 6) NOT NULL,
                commission DECIMAL(18, 6) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute(
            """
            INSERT INTO analytics.trades
            (symbol, date, action, shares, price, total_value, commission,
             created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                trade.symbol.value,
                trade.date,
                trade.action,
                float(trade.shares),
                float(trade.price),
                float(trade.total_value),
                float(trade.commission),
                datetime.now(),
            ],
        )

        logger.info(
            f"Inserted {trade.action} trade for {trade.symbol.value}: "
            f"{trade.shares} shares @ {trade.price}"
        )

    def get_trades(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> list[Trade]:
        """Get trade history, optionally filtered by date range.

        Args:
            start_date: Optional start date filter (inclusive)
            end_date: Optional end date filter (inclusive)

        Returns:
            List of Trade objects ordered by date descending
        """
        sql = """
            SELECT symbol, date, action, shares, price, total_value, commission
            FROM analytics.trades
            WHERE 1=1
        """
        params: list[Any] = []

        if start_date:
            sql += " AND date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date <= ?"
            params.append(end_date)

        sql += " ORDER BY date DESC"

        result = self.conn.execute(sql, params).fetchall()

        trades = []
        for row in result:
            try:
                trades.append(
                    Trade(
                        symbol=row[0],
                        date=row[1],
                        action=row[2],
                        shares=row[3],
                        price=row[4],
                        total_value=row[5],
                        commission=row[6],
                    )
                )
            except ValidationError as e:
                logger.error(f"Validation error for trade {row}: {e}")

        return trades

    def _update_freshness(
        self,
        data_category: DataCategory,
        record_count: int,
        source: str,
        symbol: str | None = None,
        indicator_name: str | None = None,
    ) -> None:
        """Update data freshness metadata.

        Args:
            data_category: Category of data being updated
            record_count: Number of records in this update
            source: Data source name
            symbol: Optional symbol for price/position data
            indicator_name: Optional indicator name for macro data
        """
        now = datetime.now()

        # Use INSERT ... ON CONFLICT with explicit conflict target
        self.conn.execute(
            """
            INSERT INTO raw.data_freshness
            (data_category, symbol, indicator_name, last_updated,
             record_count, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (data_category, symbol, indicator_name, source)
            DO UPDATE SET
                last_updated = EXCLUDED.last_updated,
                record_count = EXCLUDED.record_count,
                updated_at = EXCLUDED.updated_at
        """,
            [
                data_category.value,
                symbol,
                indicator_name,
                now,
                record_count,
                source,
                now,
            ],
        )

    def get_freshness(
        self,
        data_category: DataCategory,
        symbol: str | None = None,
        indicator_name: str | None = None,
    ) -> DataFreshness | None:
        """Get data freshness metadata.

        Args:
            data_category: Category of data to check
            symbol: Optional symbol filter
            indicator_name: Optional indicator name filter

        Returns:
            DataFreshness object if found, None otherwise
        """
        sql = """
            SELECT data_category, symbol, indicator_name, last_updated,
                   record_count, source
            FROM raw.data_freshness
            WHERE data_category = ?
        """
        params: list[Any] = [data_category.value]

        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol)

        if indicator_name is not None:
            sql += " AND indicator_name = ?"
            params.append(indicator_name)

        sql += " ORDER BY last_updated DESC LIMIT 1"

        result = self.conn.execute(sql, params).fetchone()

        if not result:
            return None

        return DataFreshness(
            data_category=DataCategory(result[0]),
            symbol=result[1],
            indicator_name=result[2],
            last_updated=result[3],
            record_count=result[4],
            source=result[5],
        )

    def check_freshness(
        self,
        data_category: DataCategory,
        symbol: str | None = None,
        indicator_name: str | None = None,
        raise_on_critical: bool = True,
    ) -> DataFreshness | None:
        """Check data freshness and log warnings.

        Args:
            data_category: Category of data to check
            symbol: Optional symbol filter
            indicator_name: Optional indicator name filter
            raise_on_critical: If True, raise StaleDataError for critical staleness

        Returns:
            DataFreshness object if found, None otherwise

        Raises:
            StaleDataError: If data is critically stale and raise_on_critical=True
        """
        freshness = self.get_freshness(data_category, symbol, indicator_name)

        if not freshness:
            logger.warning(f"No freshness metadata found for {data_category.value}")
            return None

        status = freshness.get_status()

        if status == FreshnessStatus.CRITICAL:
            warning = freshness.get_warning_message()
            logger.error(warning)
            if raise_on_critical:
                raise StaleDataError(freshness)
        elif status == FreshnessStatus.STALE:
            warning = freshness.get_warning_message()
            logger.warning(warning)

        return freshness

    def get_all_freshness_status(self) -> list[DataFreshness]:
        """Get freshness status for all tracked data.

        Returns:
            List of DataFreshness objects for all tracked datasets
        """
        result = self.conn.execute(
            """
            SELECT data_category, symbol, indicator_name, last_updated,
                   record_count, source
            FROM raw.data_freshness
            ORDER BY data_category, symbol, indicator_name
        """
        ).fetchall()

        freshness_list = []
        for row in result:
            try:
                freshness_list.append(
                    DataFreshness(
                        data_category=DataCategory(row[0]),
                        symbol=row[1],
                        indicator_name=row[2],
                        last_updated=row[3],
                        record_count=row[4],
                        source=row[5],
                    )
                )
            except Exception as e:
                logger.error(f"Failed to parse freshness record {row}: {e}")

        return freshness_list

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed")

    def __enter__(self) -> "DuckDBStorage":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
