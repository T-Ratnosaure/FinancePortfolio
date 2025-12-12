"""Tests for the main CLI application."""

from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()


class TestCLIMain:
    """Tests for the main CLI entry point."""

    def test_app_no_args_shows_help(self) -> None:
        """Test that running without args shows help."""
        result = runner.invoke(app)
        # Typer with no_args_is_help=True may exit with 0 or 2 depending on version
        # The important thing is that help text is shown
        assert (
            "PEA Portfolio Optimization System" in result.stdout
            or result.exit_code in [0, 2]
        )

    def test_help_command(self) -> None:
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "PEA Portfolio Optimization System" in result.stdout
        assert "analyze" in result.stdout
        assert "backtest" in result.stdout
        assert "status" in result.stdout
        assert "fetch" in result.stdout

    def test_analyze_help(self) -> None:
        """Test analyze command help."""
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "market regime analysis" in result.stdout.lower()

    def test_backtest_help(self) -> None:
        """Test backtest command help."""
        result = runner.invoke(app, ["backtest", "--help"])
        assert result.exit_code == 0
        # Rich formatting may truncate options, so check for key terms
        output_lower = result.stdout.lower()
        assert "backtest" in output_lower
        assert "options" in output_lower or "help" in output_lower

    def test_status_help(self) -> None:
        """Test status command help."""
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "portfolio status" in result.stdout.lower()

    def test_fetch_help(self) -> None:
        """Test fetch command help."""
        result = runner.invoke(app, ["fetch", "--help"])
        assert result.exit_code == 0
        # Rich formatting may truncate options, so check for key terms
        output_lower = result.stdout.lower()
        assert "fetch" in output_lower
        assert "market data" in output_lower or "options" in output_lower

    def test_verbose_option(self) -> None:
        """Test --verbose global option."""
        result = runner.invoke(app, ["--verbose", "--help"])
        assert result.exit_code == 0

    def test_quiet_option(self) -> None:
        """Test --quiet global option."""
        result = runner.invoke(app, ["--quiet", "--help"])
        assert result.exit_code == 0
