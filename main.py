"""Main entry point for FinancePortfolio application."""

import logging

from src.config.logging import setup_logging

# Initialize logging at application start
setup_logging()

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the main application."""
    logger.info("Starting FinancePortfolio application")
    logger.info("Application initialized successfully")


if __name__ == "__main__":
    main()
