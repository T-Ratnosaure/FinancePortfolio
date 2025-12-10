"""Base fetcher abstract class for data sources."""

from abc import ABC, abstractmethod
from datetime import date


class FetchError(Exception):
    """Base exception for data fetching errors."""

    def __init__(self, message: str, source: str | None = None) -> None:
        """Initialize fetch error.

        Args:
            message: Error description
            source: Data source name (e.g., 'Yahoo Finance', 'FRED')
        """
        self.source = source
        super().__init__(f"[{source}] {message}" if source else message)


class RateLimitError(FetchError):
    """Exception raised when rate limit is exceeded."""

    pass


class DataNotAvailableError(FetchError):
    """Exception raised when requested data is not available."""

    pass


class BaseFetcher(ABC):
    """Abstract base class for data fetchers.

    Defines the interface that all data fetchers must implement.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the fetcher with required credentials/configuration."""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that the connection to the data source is working.

        Returns:
            True if connection is valid, False otherwise.
        """
        pass

    def _validate_date_range(self, start_date: date, end_date: date) -> None:
        """Validate that date range is valid.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Raises:
            ValueError: If date range is invalid
        """
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must be before end_date ({end_date})"
            )
