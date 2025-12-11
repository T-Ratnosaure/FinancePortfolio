"""Centralized logging configuration for FinancePortfolio.

This module provides a unified logging setup for the entire application,
supporting both development and production environments with appropriate
formatters and handlers.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        import json
        from datetime import datetime

        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "[36m",  # Cyan
        "INFO": "[32m",  # Green
        "WARNING": "[33m",  # Yellow
        "ERROR": "[31m",  # Red
        "CRITICAL": "[35m",  # Magenta
        "RESET": "[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for console.

        Args:
            record: Log record to format

        Returns:
            Colored formatted log string
        """
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )

        # Format the message
        formatted = super().format(record)

        # Reset levelname for other formatters
        record.levelname = levelname

        return formatted


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    use_json: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Configure logging for the application.

    This function should be called once at application startup to set up
    the logging infrastructure. It configures the root logger with
    appropriate handlers and formatters based on the environment.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            If not provided, reads from LOG_LEVEL environment variable.
            Defaults to INFO.
        log_file: Optional path to log file. If provided, adds a rotating
            file handler.
        use_json: If True, use JSON formatter. If False, use console
            formatter. Can be overridden by LOG_FORMAT environment variable.
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)

    Example:
        >>> from src.config.logging import setup_logging
        >>> setup_logging()  # Use defaults
        >>> setup_logging(level="DEBUG", log_file=Path("logs/app.log"))
    """
    # Determine log level
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Validate log level
    numeric_level = getattr(logging, level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Determine if we should use JSON format
    log_format = os.getenv("LOG_FORMAT", "").lower()
    if log_format == "json":
        use_json = True
    elif log_format == "console":
        use_json = False

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if use_json:
        console_formatter = JSONFormatter()
    else:
        console_formatter = ConsoleFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log_file is provided
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)

        # Always use JSON format for file logs
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Log initialization message
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging initialized: level={level}, "
        f"format={'JSON' if use_json else 'console'}, "
        f"file={'enabled' if log_file else 'disabled'}"
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.

    This is a convenience wrapper around logging.getLogger() that ensures
    consistent logger naming throughout the application.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
