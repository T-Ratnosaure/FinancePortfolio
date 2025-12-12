"""Data freshness utilities for staleness detection and monitoring.

This module provides high-level utilities for checking and monitoring data
freshness across the entire portfolio system.
"""

import logging
from datetime import datetime
from typing import Any

from .models import DataCategory, DataFreshness, FreshnessStatus
from .storage.duckdb import DuckDBStorage

logger = logging.getLogger(__name__)


class FreshnessReport:
    """Report on data freshness status across all datasets.

    Attributes:
        timestamp: When the report was generated
        fresh_count: Number of datasets with fresh data
        stale_count: Number of datasets with stale data
        critical_count: Number of datasets with critically stale data
        datasets: List of all DataFreshness objects
    """

    def __init__(self, datasets: list[DataFreshness]) -> None:
        """Initialize freshness report.

        Args:
            datasets: List of DataFreshness objects to analyze
        """
        self.timestamp = datetime.now()
        self.datasets = datasets

        self.fresh_count = sum(
            1 for d in datasets if d.get_status() == FreshnessStatus.FRESH
        )
        self.stale_count = sum(
            1 for d in datasets if d.get_status() == FreshnessStatus.STALE
        )
        self.critical_count = sum(
            1 for d in datasets if d.get_status() == FreshnessStatus.CRITICAL
        )

    def has_issues(self) -> bool:
        """Check if there are any freshness issues.

        Returns:
            True if any data is stale or critical
        """
        return self.stale_count > 0 or self.critical_count > 0

    def has_critical_issues(self) -> bool:
        """Check if there are any critical freshness issues.

        Returns:
            True if any data is critically stale
        """
        return self.critical_count > 0

    def get_stale_datasets(self) -> list[DataFreshness]:
        """Get list of stale datasets.

        Returns:
            List of DataFreshness objects that are stale (not critical)
        """
        return [d for d in self.datasets if d.get_status() == FreshnessStatus.STALE]

    def get_critical_datasets(self) -> list[DataFreshness]:
        """Get list of critically stale datasets.

        Returns:
            List of DataFreshness objects that are critically stale
        """
        return [d for d in self.datasets if d.get_status() == FreshnessStatus.CRITICAL]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary.

        Returns:
            Dictionary representation of the report
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_datasets": len(self.datasets),
                "fresh": self.fresh_count,
                "stale": self.stale_count,
                "critical": self.critical_count,
            },
            "stale_datasets": [
                {
                    "category": d.data_category.value,
                    "symbol": d.symbol,
                    "indicator": d.indicator_name,
                    "age": str(d.get_age()),
                    "last_updated": d.last_updated.isoformat(),
                    "message": d.get_warning_message(),
                }
                for d in self.get_stale_datasets()
            ],
            "critical_datasets": [
                {
                    "category": d.data_category.value,
                    "symbol": d.symbol,
                    "indicator": d.indicator_name,
                    "age": str(d.get_age()),
                    "last_updated": d.last_updated.isoformat(),
                    "message": d.get_warning_message(),
                }
                for d in self.get_critical_datasets()
            ],
        }

    def __str__(self) -> str:
        """Generate human-readable report.

        Returns:
            Formatted string report
        """
        lines = [
            f"Data Freshness Report - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            f"Total Datasets: {len(self.datasets)}",
            f"  Fresh: {self.fresh_count}",
            f"  Stale: {self.stale_count}",
            f"  Critical: {self.critical_count}",
        ]

        if self.critical_count > 0:
            lines.append("\nCRITICAL ISSUES:")
            lines.append("-" * 70)
            for dataset in self.get_critical_datasets():
                lines.append(f"  {dataset.get_warning_message()}")

        if self.stale_count > 0:
            lines.append("\nWARNINGS:")
            lines.append("-" * 70)
            for dataset in self.get_stale_datasets():
                lines.append(f"  {dataset.get_warning_message()}")

        return "\n".join(lines)


def generate_freshness_report(storage: DuckDBStorage) -> FreshnessReport:
    """Generate a comprehensive freshness report.

    Args:
        storage: DuckDB storage instance to query

    Returns:
        FreshnessReport object with analysis of all datasets
    """
    datasets = storage.get_all_freshness_status()
    return FreshnessReport(datasets)


def check_price_data_freshness(
    storage: DuckDBStorage, symbol: str, raise_on_critical: bool = True
) -> DataFreshness | None:
    """Check freshness of price data for a specific symbol.

    Args:
        storage: DuckDB storage instance
        symbol: ETF symbol to check
        raise_on_critical: If True, raise StaleDataError for critical staleness

    Returns:
        DataFreshness object if found, None otherwise

    Raises:
        StaleDataError: If data is critically stale and raise_on_critical=True
    """
    return storage.check_freshness(
        DataCategory.PRICE_DATA, symbol=symbol, raise_on_critical=raise_on_critical
    )


def check_macro_data_freshness(
    storage: DuckDBStorage, indicator_name: str, raise_on_critical: bool = True
) -> DataFreshness | None:
    """Check freshness of macro data for a specific indicator.

    Args:
        storage: DuckDB storage instance
        indicator_name: Indicator name to check (e.g., 'VIX', 'TREASURY_10Y')
        raise_on_critical: If True, raise StaleDataError for critical staleness

    Returns:
        DataFreshness object if found, None otherwise

    Raises:
        StaleDataError: If data is critically stale and raise_on_critical=True
    """
    return storage.check_freshness(
        DataCategory.MACRO_DATA,
        indicator_name=indicator_name,
        raise_on_critical=raise_on_critical,
    )


def check_portfolio_freshness(
    storage: DuckDBStorage, raise_on_critical: bool = True
) -> DataFreshness | None:
    """Check freshness of portfolio position data.

    Args:
        storage: DuckDB storage instance
        raise_on_critical: If True, raise StaleDataError for critical staleness

    Returns:
        DataFreshness object if found, None otherwise

    Raises:
        StaleDataError: If data is critically stale and raise_on_critical=True
    """
    return storage.check_freshness(
        DataCategory.PORTFOLIO_DATA, raise_on_critical=raise_on_critical
    )


def log_freshness_warnings(storage: DuckDBStorage) -> None:
    """Log warnings for all stale data.

    This function checks all tracked datasets and logs warnings for any
    that are stale or critical, without raising exceptions.

    Args:
        storage: DuckDB storage instance
    """
    datasets = storage.get_all_freshness_status()

    for dataset in datasets:
        status = dataset.get_status()
        if status == FreshnessStatus.STALE:
            logger.warning(dataset.get_warning_message())
        elif status == FreshnessStatus.CRITICAL:
            logger.error(dataset.get_warning_message())
