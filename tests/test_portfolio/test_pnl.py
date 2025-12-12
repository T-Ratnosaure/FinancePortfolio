"""Tests for PnL tracking, attribution, and reconciliation."""

from datetime import date, datetime

import pandas as pd
import pytest

from src.data.models import ETFSymbol, Position, Regime, Trade, TradeAction
from src.portfolio.pnl import (
    DailyPnL,
    PeriodPnL,
    PnLAttribution,
    PnLCalculator,
    PnLReconciler,
    ReconciliationResult,
)


class TestDailyPnL:
    """Test DailyPnL model."""

    def test_valid_daily_pnl(self) -> None:
        """Test creation of valid DailyPnL."""
        pnl = DailyPnL(
            date=date(2024, 1, 15),
            total_pnl=100.0,
            realized_pnl=60.0,
            unrealized_pnl=40.0,
            transaction_costs=5.0,
            net_pnl=95.0,
        )

        assert pnl.date == date(2024, 1, 15)
        assert pnl.total_pnl == 100.0
        assert pnl.realized_pnl == 60.0
        assert pnl.unrealized_pnl == 40.0
        assert pnl.transaction_costs == 5.0
        assert pnl.net_pnl == 95.0

    def test_pnl_consistency_validation(self) -> None:
        """Test that PnL components must be consistent."""
        with pytest.raises(ValueError, match="should equal realized_pnl"):
            DailyPnL(
                date=date(2024, 1, 15),
                total_pnl=100.0,
                realized_pnl=60.0,
                unrealized_pnl=50.0,  # Should be 40.0
                transaction_costs=5.0,
                net_pnl=95.0,
            )

        with pytest.raises(ValueError, match="should equal total_pnl"):
            DailyPnL(
                date=date(2024, 1, 15),
                total_pnl=100.0,
                realized_pnl=60.0,
                unrealized_pnl=40.0,
                transaction_costs=5.0,
                net_pnl=90.0,  # Should be 95.0
            )

    def test_round_values(self) -> None:
        """Test rounding of PnL values."""
        pnl = DailyPnL(
            date=date(2024, 1, 15),
            total_pnl=100.567,
            realized_pnl=60.234,
            unrealized_pnl=40.333,
            transaction_costs=5.123,
            net_pnl=95.444,
        )

        rounded = pnl.round_values()
        assert rounded.total_pnl == 100.57
        assert rounded.realized_pnl == 60.23
        assert rounded.unrealized_pnl == 40.33
        assert rounded.transaction_costs == 5.12
        assert rounded.net_pnl == 95.44

    def test_negative_pnl(self) -> None:
        """Test DailyPnL with negative values (losses)."""
        pnl = DailyPnL(
            date=date(2024, 1, 15),
            total_pnl=-50.0,
            realized_pnl=-30.0,
            unrealized_pnl=-20.0,
            transaction_costs=5.0,
            net_pnl=-55.0,
        )

        assert pnl.total_pnl == -50.0
        assert pnl.net_pnl == -55.0


class TestPeriodPnL:
    """Test PeriodPnL model."""

    def test_valid_period_pnl(self) -> None:
        """Test creation of valid PeriodPnL."""
        pnl = PeriodPnL(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            total_pnl=500.0,
            realized_pnl=300.0,
            unrealized_pnl=200.0,
            transaction_costs=25.0,
            net_pnl=475.0,
            num_trading_days=21,
        )

        assert pnl.start_date == date(2024, 1, 1)
        assert pnl.end_date == date(2024, 1, 31)
        assert pnl.total_pnl == 500.0
        assert pnl.num_trading_days == 21

    def test_invalid_date_range(self) -> None:
        """Test that end_date must be >= start_date."""
        with pytest.raises(ValueError, match="end_date must be"):
            PeriodPnL(
                start_date=date(2024, 1, 31),
                end_date=date(2024, 1, 1),  # Before start
                total_pnl=500.0,
                realized_pnl=300.0,
                unrealized_pnl=200.0,
                transaction_costs=25.0,
                net_pnl=475.0,
                num_trading_days=21,
            )

    def test_period_pnl_with_daily_breakdown(self) -> None:
        """Test PeriodPnL with daily PnL records."""
        daily1 = DailyPnL(
            date=date(2024, 1, 15),
            total_pnl=100.0,
            realized_pnl=60.0,
            unrealized_pnl=40.0,
            transaction_costs=5.0,
            net_pnl=95.0,
        )
        daily2 = DailyPnL(
            date=date(2024, 1, 16),
            total_pnl=50.0,
            realized_pnl=20.0,
            unrealized_pnl=30.0,
            transaction_costs=2.0,
            net_pnl=48.0,
        )

        period = PeriodPnL(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 16),
            total_pnl=150.0,
            realized_pnl=80.0,
            unrealized_pnl=70.0,
            transaction_costs=7.0,
            net_pnl=143.0,
            num_trading_days=2,
            daily_pnls=[daily1, daily2],
        )

        assert len(period.daily_pnls) == 2
        assert period.num_trading_days == 2


class TestPnLAttribution:
    """Test PnLAttribution model."""

    def test_attribution_creation(self) -> None:
        """Test creation of attribution breakdown."""
        attribution = PnLAttribution(
            by_symbol={"LQQ.PA": 100.0, "CL2.PA": 50.0},
            by_regime={"risk_on": 120.0, "neutral": 30.0},
            by_factor={"beta": 80.0, "alpha": 70.0},
        )

        assert attribution.by_symbol["LQQ.PA"] == 100.0
        assert attribution.by_regime["risk_on"] == 120.0
        assert attribution.by_factor["alpha"] == 70.0

    def test_attribution_rounding(self) -> None:
        """Test rounding of attribution values."""
        attribution = PnLAttribution(
            by_symbol={"LQQ.PA": 100.567, "CL2.PA": 50.234},
            by_regime={"risk_on": 120.999},
            by_factor={"beta": 80.123},
        )

        rounded = attribution.round_values()
        assert rounded.by_symbol["LQQ.PA"] == 100.57
        assert rounded.by_symbol["CL2.PA"] == 50.23
        assert rounded.by_regime["risk_on"] == 121.00
        assert rounded.by_factor["beta"] == 80.12


class TestReconciliationResult:
    """Test ReconciliationResult model."""

    def test_successful_reconciliation(self) -> None:
        """Test reconciliation result when values match."""
        result = ReconciliationResult(
            matches=True,
            calculated_pnl=1000.50,
            expected_pnl=1000.45,
            difference=0.05,
            tolerance=0.10,
            message="PnL reconciled successfully",
        )

        assert result.matches is True
        assert result.difference == 0.05

    def test_failed_reconciliation(self) -> None:
        """Test reconciliation result when values don't match."""
        result = ReconciliationResult(
            matches=False,
            calculated_pnl=1000.50,
            expected_pnl=900.00,
            difference=100.50,
            tolerance=1.00,
            message="PnL reconciliation FAILED",
        )

        assert result.matches is False
        assert result.difference == 100.50

    def test_difference_validation(self) -> None:
        """Test that difference is validated correctly."""
        with pytest.raises(ValueError, match="difference"):
            ReconciliationResult(
                matches=True,
                calculated_pnl=1000.0,
                expected_pnl=900.0,
                difference=50.0,  # Should be 100.0
                tolerance=1.0,
                message="Test",
            )


class TestPnLCalculator:
    """Test PnLCalculator class."""

    @pytest.fixture
    def calculator(self) -> PnLCalculator:
        """Create PnLCalculator instance."""
        return PnLCalculator()

    @pytest.fixture
    def sample_positions(self) -> dict[str, Position]:
        """Create sample positions for testing."""
        return {
            "LQQ.PA": Position(
                symbol=ETFSymbol.LQQ,
                shares=10.0,
                average_cost=100.0,
                current_price=105.0,
                market_value=1050.0,
                unrealized_pnl=50.0,
                weight=0.5,
            ),
            "CL2.PA": Position(
                symbol=ETFSymbol.CL2,
                shares=5.0,
                average_cost=200.0,
                current_price=210.0,
                market_value=1050.0,
                unrealized_pnl=50.0,
                weight=0.5,
            ),
        }

    def test_calculate_daily_pnl_no_trades(
        self,
        calculator: PnLCalculator,
        sample_positions: dict[str, Position],
    ) -> None:
        """Test daily PnL calculation with no trades."""
        prices_today = {"LQQ.PA": 105.0, "CL2.PA": 210.0}
        prices_yesterday = {"LQQ.PA": 100.0, "CL2.PA": 200.0}

        pnl = calculator.calculate_daily_pnl(
            positions=sample_positions,
            prices_today=prices_today,
            prices_yesterday=prices_yesterday,
            trades_today=[],
            pnl_date=date(2024, 1, 15),
        )

        # Unrealized: LQQ (105-100)*10 + CL2 (210-200)*5 = 50 + 50 = 100
        assert pnl.unrealized_pnl == 100.0
        assert pnl.realized_pnl == 0.0
        assert pnl.transaction_costs == 0.0
        assert pnl.total_pnl == 100.0
        assert pnl.net_pnl == 100.0

    def test_calculate_daily_pnl_with_sell_trade(
        self,
        calculator: PnLCalculator,
        sample_positions: dict[str, Position],
    ) -> None:
        """Test daily PnL calculation with a sell trade."""
        prices_today = {"LQQ.PA": 105.0, "CL2.PA": 210.0}
        prices_yesterday = {"LQQ.PA": 100.0, "CL2.PA": 200.0}

        sell_trade = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime(2024, 1, 15, 10, 0),
            action=TradeAction.SELL,
            shares=2.0,
            price=105.0,
            total_value=210.0,
            commission=1.99,
        )

        pnl = calculator.calculate_daily_pnl(
            positions=sample_positions,
            prices_today=prices_today,
            prices_yesterday=prices_yesterday,
            trades_today=[sell_trade],
            pnl_date=date(2024, 1, 15),
        )

        # Realized: (105-100)*2 = 10.0
        # Unrealized: LQQ (105-100)*10 + CL2 (210-200)*5 = 100.0
        # Costs: 1.99
        assert pnl.realized_pnl == 10.0
        assert pnl.unrealized_pnl == 100.0
        assert pnl.transaction_costs == 1.99
        assert pnl.total_pnl == 110.0
        assert pnl.net_pnl == 108.01

    def test_calculate_daily_pnl_with_buy_trade(
        self,
        calculator: PnLCalculator,
        sample_positions: dict[str, Position],
    ) -> None:
        """Test daily PnL calculation with a buy trade (costs only)."""
        prices_today = {"LQQ.PA": 105.0, "CL2.PA": 210.0}
        prices_yesterday = {"LQQ.PA": 100.0, "CL2.PA": 200.0}

        buy_trade = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime(2024, 1, 15, 10, 0),
            action=TradeAction.BUY,
            shares=5.0,
            price=105.0,
            total_value=525.0,
            commission=2.50,
        )

        pnl = calculator.calculate_daily_pnl(
            positions=sample_positions,
            prices_today=prices_today,
            prices_yesterday=prices_yesterday,
            trades_today=[buy_trade],
            pnl_date=date(2024, 1, 15),
        )

        # No realized PnL on buy
        assert pnl.realized_pnl == 0.0
        assert pnl.unrealized_pnl == 100.0
        assert pnl.transaction_costs == 2.50
        assert pnl.net_pnl == 97.50

    def test_calculate_daily_pnl_missing_prices(
        self,
        calculator: PnLCalculator,
        sample_positions: dict[str, Position],
    ) -> None:
        """Test daily PnL calculation with missing prices."""
        prices_today = {"LQQ.PA": 105.0}  # CL2 missing
        prices_yesterday = {"LQQ.PA": 100.0}

        pnl = calculator.calculate_daily_pnl(
            positions=sample_positions,
            prices_today=prices_today,
            prices_yesterday=prices_yesterday,
            trades_today=[],
            pnl_date=date(2024, 1, 15),
        )

        # Only LQQ has price change
        assert pnl.unrealized_pnl == 50.0
        assert pnl.total_pnl == 50.0

    def test_calculate_period_pnl(self, calculator: PnLCalculator) -> None:
        """Test period PnL calculation."""
        # Create position snapshots for 3 days
        position_day1 = {
            "LQQ.PA": Position(
                symbol=ETFSymbol.LQQ,
                shares=10.0,
                average_cost=100.0,
                current_price=100.0,
                market_value=1000.0,
                unrealized_pnl=0.0,
                weight=1.0,
            )
        }

        # Price history
        dates = pd.date_range(start="2024-01-01", periods=3, freq="D")
        price_history = {
            "LQQ.PA": pd.Series([100.0, 102.0, 105.0], index=dates),
        }

        # No trades
        trades: list[Trade] = []

        period_pnl = calculator.calculate_period_pnl(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            position_history=[position_day1] * 3,
            price_history=price_history,
            trade_log=trades,
        )

        # Day 1: no change (first day)
        # Day 2: (102-100)*10 = 20
        # Day 3: (105-102)*10 = 30
        # Total unrealized: 50
        assert period_pnl.num_trading_days == 3
        assert period_pnl.realized_pnl == 0.0
        assert period_pnl.transaction_costs == 0.0

    def test_calculate_period_pnl_invalid_dates(
        self,
        calculator: PnLCalculator,
    ) -> None:
        """Test period PnL with invalid date range."""
        with pytest.raises(ValueError, match="end_date must be"):
            calculator.calculate_period_pnl(
                start_date=date(2024, 1, 31),
                end_date=date(2024, 1, 1),
                position_history=[],
                price_history={},
                trade_log=[],
            )

    def test_attribute_by_symbol(self, calculator: PnLCalculator) -> None:
        """Test PnL attribution by symbol."""
        positions = {
            "LQQ.PA": Position(
                symbol=ETFSymbol.LQQ,
                shares=10.0,
                average_cost=100.0,
                current_price=105.0,
                market_value=1050.0,
                unrealized_pnl=50.0,
                weight=0.5,
            ),
            "CL2.PA": Position(
                symbol=ETFSymbol.CL2,
                shares=5.0,
                average_cost=200.0,
                current_price=210.0,
                market_value=1050.0,
                unrealized_pnl=50.0,
                weight=0.5,
            ),
        }

        dates = pd.date_range(start="2024-01-01", periods=2, freq="D")
        price_history = {
            "LQQ.PA": pd.Series([100.0, 105.0], index=dates),
            "CL2.PA": pd.Series([200.0, 210.0], index=dates),
        }

        attribution = calculator.attribute_by_symbol(
            daily_pnls=[],
            positions=positions,
            price_history=price_history,
        )

        # LQQ: (105-100)*10 = 50
        # CL2: (210-200)*5 = 50
        assert attribution["LQQ.PA"] == 50.0
        assert attribution["CL2.PA"] == 50.0

    def test_attribute_by_regime(self, calculator: PnLCalculator) -> None:
        """Test PnL attribution by regime."""
        daily_pnls = [
            DailyPnL(
                date=date(2024, 1, 1),
                total_pnl=100.0,
                realized_pnl=50.0,
                unrealized_pnl=50.0,
                transaction_costs=5.0,
                net_pnl=95.0,
            ),
            DailyPnL(
                date=date(2024, 1, 2),
                total_pnl=50.0,
                realized_pnl=20.0,
                unrealized_pnl=30.0,
                transaction_costs=2.0,
                net_pnl=48.0,
            ),
            DailyPnL(
                date=date(2024, 1, 3),
                total_pnl=-30.0,
                realized_pnl=-20.0,
                unrealized_pnl=-10.0,
                transaction_costs=3.0,
                net_pnl=-33.0,
            ),
        ]

        regime_history = {
            date(2024, 1, 1): Regime.RISK_ON,
            date(2024, 1, 2): Regime.RISK_ON,
            date(2024, 1, 3): Regime.RISK_OFF,
        }

        attribution = calculator.attribute_by_regime(daily_pnls, regime_history)

        # Risk-on: 95 + 48 = 143
        # Risk-off: -33
        assert attribution["risk_on"] == 143.0
        assert attribution["risk_off"] == -33.0
        assert attribution["neutral"] == 0.0

    def test_calculate_alpha_beta(self, calculator: PnLCalculator) -> None:
        """Test alpha and beta calculation."""
        portfolio_returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        benchmark_returns = pd.Series([0.008, 0.015, -0.005, 0.025])

        factors = calculator.calculate_alpha_beta(portfolio_returns, benchmark_returns)

        assert "beta" in factors
        assert "alpha" in factors
        assert "beta_pnl" in factors
        assert isinstance(factors["beta"], float)
        assert isinstance(factors["alpha"], float)

    def test_calculate_alpha_beta_insufficient_data(
        self,
        calculator: PnLCalculator,
    ) -> None:
        """Test alpha/beta calculation with insufficient data."""
        portfolio_returns = pd.Series([0.01])
        benchmark_returns = pd.Series([0.008])

        factors = calculator.calculate_alpha_beta(portfolio_returns, benchmark_returns)

        assert factors["beta"] == 0.0
        assert factors["alpha"] == 0.0

    def test_calculate_alpha_beta_mismatched_length(
        self,
        calculator: PnLCalculator,
    ) -> None:
        """Test alpha/beta calculation with mismatched series length."""
        portfolio_returns = pd.Series([0.01, 0.02])
        benchmark_returns = pd.Series([0.008])

        with pytest.raises(ValueError, match="same length"):
            calculator.calculate_alpha_beta(portfolio_returns, benchmark_returns)


class TestPnLReconciler:
    """Test PnLReconciler class."""

    @pytest.fixture
    def reconciler(self) -> PnLReconciler:
        """Create PnLReconciler instance."""
        return PnLReconciler()

    def test_reconcile_within_tolerance(self, reconciler: PnLReconciler) -> None:
        """Test reconciliation when values are within tolerance."""
        result = reconciler.reconcile(
            calculated_pnl=1000.50,
            expected_pnl=1000.45,
            tolerance=0.10,
        )

        assert result.matches is True
        assert result.difference == 0.05
        assert result.calculated_pnl == 1000.50
        assert result.expected_pnl == 1000.45

    def test_reconcile_exceeds_tolerance(self, reconciler: PnLReconciler) -> None:
        """Test reconciliation when difference exceeds tolerance."""
        result = reconciler.reconcile(
            calculated_pnl=1000.50,
            expected_pnl=900.00,
            tolerance=1.00,
        )

        assert result.matches is False
        assert result.difference == 100.50

    def test_reconcile_exact_match(self, reconciler: PnLReconciler) -> None:
        """Test reconciliation with exact match."""
        result = reconciler.reconcile(
            calculated_pnl=1000.00,
            expected_pnl=1000.00,
            tolerance=0.01,
        )

        assert result.matches is True
        assert result.difference == 0.0

    def test_reconcile_negative_tolerance(self, reconciler: PnLReconciler) -> None:
        """Test that negative tolerance raises error."""
        with pytest.raises(ValueError, match="tolerance must be positive"):
            reconciler.reconcile(
                calculated_pnl=1000.0,
                expected_pnl=1000.0,
                tolerance=-0.01,
            )

    def test_validate_pnl_components_valid(self, reconciler: PnLReconciler) -> None:
        """Test validation of valid PnL components."""
        pnl = DailyPnL(
            date=date(2024, 1, 15),
            total_pnl=100.0,
            realized_pnl=60.0,
            unrealized_pnl=40.0,
            transaction_costs=5.0,
            net_pnl=95.0,
        )

        errors = reconciler.validate_pnl_components(pnl)
        assert len(errors) == 0

    def test_validate_pnl_components_invalid_total(
        self,
        reconciler: PnLReconciler,
    ) -> None:
        """Test validation catches invalid total PnL."""
        # Create with inconsistent values by bypassing validation
        pnl = DailyPnL.model_construct(
            date=date(2024, 1, 15),
            total_pnl=100.0,
            realized_pnl=60.0,
            unrealized_pnl=50.0,  # Should be 40.0
            transaction_costs=5.0,
            net_pnl=95.0,
        )

        errors = reconciler.validate_pnl_components(pnl)
        assert len(errors) > 0
        assert any("Total PnL mismatch" in err for err in errors)

    def test_validate_pnl_components_invalid_net(
        self,
        reconciler: PnLReconciler,
    ) -> None:
        """Test validation catches invalid net PnL."""
        pnl = DailyPnL.model_construct(
            date=date(2024, 1, 15),
            total_pnl=100.0,
            realized_pnl=60.0,
            unrealized_pnl=40.0,
            transaction_costs=5.0,
            net_pnl=100.0,  # Should be 95.0
        )

        errors = reconciler.validate_pnl_components(pnl)
        assert len(errors) > 0
        assert any("Net PnL mismatch" in err for err in errors)

    def test_validate_pnl_components_negative_costs(
        self,
        reconciler: PnLReconciler,
    ) -> None:
        """Test validation catches negative transaction costs."""
        pnl = DailyPnL.model_construct(
            date=date(2024, 1, 15),
            total_pnl=100.0,
            realized_pnl=60.0,
            unrealized_pnl=40.0,
            transaction_costs=-5.0,  # Should be positive
            net_pnl=105.0,
        )

        errors = reconciler.validate_pnl_components(pnl)
        assert len(errors) > 0
        assert any("cannot be negative" in err for err in errors)
