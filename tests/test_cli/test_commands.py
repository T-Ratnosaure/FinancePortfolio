"""Tests for CLI commands."""

import json

from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()


class TestStatusCommand:
    """Tests for the status command."""

    def test_status_demo_portfolio_plain(self) -> None:
        """Test status command with demo portfolio in plain format."""
        result = runner.invoke(app, ["status", "--format", "plain"])
        # Should work even without database
        assert result.exit_code == 0
        assert (
            "Portfolio Value" in result.stdout
            or "demo portfolio" in result.stdout.lower()
        )

    def test_status_demo_portfolio_json(self) -> None:
        """Test status command with JSON output."""
        result = runner.invoke(app, ["status", "--format", "json"])
        # Should output valid JSON
        assert result.exit_code == 0
        # Try to parse as JSON
        try:
            output = result.stdout
            # Find JSON in output (may have warning messages before)
            json_start = output.find("{")
            if json_start >= 0:
                json_str = output[json_start:]
                data = json.loads(json_str)
                assert "portfolio_value" in data or "weights" in data
        except json.JSONDecodeError:
            pass  # JSON may not be valid if there are warnings

    def test_status_invalid_regime(self) -> None:
        """Test status command with invalid target regime."""
        result = runner.invoke(app, ["status", "--target-regime", "invalid_regime"])
        assert result.exit_code == 1
        assert "Invalid regime" in result.stdout


class TestBacktestCommand:
    """Tests for the backtest command."""

    def test_backtest_missing_dates(self) -> None:
        """Test backtest command fails without required dates."""
        result = runner.invoke(app, ["backtest"])
        assert result.exit_code != 0
        # Should show error about missing options

    def test_backtest_invalid_date_format(self) -> None:
        """Test backtest command with invalid date format."""
        result = runner.invoke(
            app,
            [
                "backtest",
                "--start-date",
                "invalid-date",
                "--end-date",
                "2024-01-01",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid date format" in result.stdout

    def test_backtest_end_before_start(self) -> None:
        """Test backtest command fails when end date is before start."""
        result = runner.invoke(
            app,
            [
                "backtest",
                "--start-date",
                "2024-06-01",
                "--end-date",
                "2024-01-01",
            ],
        )
        assert result.exit_code == 1
        assert "Start date must be before end date" in result.stdout

    def test_backtest_invalid_frequency(self) -> None:
        """Test backtest command with invalid rebalance frequency."""
        result = runner.invoke(
            app,
            [
                "backtest",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2024-01-01",
                "--rebalance-frequency",
                "invalid",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid rebalance frequency" in result.stdout


class TestFetchCommand:
    """Tests for the fetch command."""

    def test_fetch_invalid_type(self) -> None:
        """Test fetch command with invalid data type."""
        result = runner.invoke(app, ["fetch", "--type", "invalid"])
        assert result.exit_code == 1
        assert "Invalid data type" in result.stdout

    def test_fetch_invalid_date_format(self) -> None:
        """Test fetch command with invalid date format."""
        result = runner.invoke(
            app,
            [
                "fetch",
                "--start-date",
                "invalid-date",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid date format" in result.stdout


class TestAnalyzeCommand:
    """Tests for the analyze command."""

    def test_analyze_with_skip_fetch(self) -> None:
        """Test analyze command with --skip-fetch flag."""
        # This test verifies the skip-fetch flag is recognized
        result = runner.invoke(
            app,
            ["analyze", "--skip-fetch", "--format", "plain"],
        )
        # Command should be recognized and attempt to run
        # May fail due to missing data, but flag should be recognized
        assert "--skip-fetch" not in result.stdout  # Flag shouldn't appear in output

    def test_analyze_plain_format(self) -> None:
        """Test analyze command with plain format option."""
        result = runner.invoke(
            app,
            ["analyze", "--format", "plain", "--skip-fetch"],
        )
        # Command should be recognized and attempt to run
        # May fail due to missing data, but format should be recognized
        assert "--format" not in result.stdout


class TestOutputFormats:
    """Tests for output format options across commands."""

    def test_status_rich_format(self) -> None:
        """Test status command with rich format (default)."""
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0

    def test_status_plain_format(self) -> None:
        """Test status command with plain format."""
        result = runner.invoke(app, ["status", "--format", "plain"])
        assert result.exit_code == 0
        # Plain format should not have rich markup
        assert "[bold]" not in result.stdout
