"""Tests for Pydantic data models."""

from datetime import date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.data.models import (
    PEA_ETFS,
    AllocationRecommendation,
    DailyPrice,
    ETFInfo,
    ETFSymbol,
    MacroIndicator,
    Position,
    Regime,
    Trade,
    TradeAction,
)


class TestETFSymbol:
    """Tests for ETFSymbol enum."""

    def test_pea_symbols_exist(self) -> None:
        """Verify all PEA ETF symbols are defined."""
        assert ETFSymbol.LQQ.value == "LQQ.PA"
        assert ETFSymbol.CL2.value == "CL2.PA"
        assert ETFSymbol.WPEA.value == "WPEA.PA"

    def test_symbol_count(self) -> None:
        """Verify we have exactly 3 PEA ETFs."""
        assert len(ETFSymbol) == 3


class TestETFInfo:
    """Tests for ETFInfo model."""

    def test_valid_etf_info(self) -> None:
        """Test creating valid ETF info."""
        info = ETFInfo(
            symbol=ETFSymbol.LQQ,
            isin="FR0010342592",
            name="Amundi Nasdaq-100 Daily (2x) Leveraged UCITS ETF",
            leverage=2,
            ter=0.006,
            pea_eligible=True,
            accumulating=True,
        )
        assert info.symbol == ETFSymbol.LQQ
        assert info.leverage == 2
        assert info.ter == 0.006

    def test_invalid_isin_format(self) -> None:
        """Test that invalid ISIN format is rejected."""
        with pytest.raises(ValidationError):
            ETFInfo(
                symbol=ETFSymbol.LQQ,
                isin="invalid",
                name="Test",
                leverage=2,
                ter=0.006,
            )

    def test_pea_etfs_registry(self) -> None:
        """Test the PEA_ETFS registry contains all ETFs."""
        assert len(PEA_ETFS) == 3
        assert ETFSymbol.LQQ in PEA_ETFS
        assert ETFSymbol.CL2 in PEA_ETFS
        assert ETFSymbol.WPEA in PEA_ETFS


class TestDailyPrice:
    """Tests for DailyPrice model."""

    def test_valid_price(self) -> None:
        """Test creating valid daily price."""
        price = DailyPrice(
            symbol=ETFSymbol.LQQ,
            date=date(2024, 1, 15),
            open=Decimal("100.50"),
            high=Decimal("102.00"),
            low=Decimal("99.50"),
            close=Decimal("101.25"),
            volume=10000,
            adjusted_close=Decimal("101.25"),
        )
        assert price.symbol == ETFSymbol.LQQ
        assert price.close == Decimal("101.25")

    def test_high_less_than_low_rejected(self) -> None:
        """Test that high < low is rejected."""
        with pytest.raises(ValidationError, match="high must be >= low"):
            DailyPrice(
                symbol=ETFSymbol.LQQ,
                date=date(2024, 1, 15),
                open=Decimal("100.00"),
                high=Decimal("99.00"),  # Less than low
                low=Decimal("100.00"),
                close=Decimal("100.00"),
                volume=10000,
            )

    def test_high_less_than_close_rejected(self) -> None:
        """Test that high < close is rejected."""
        with pytest.raises(ValidationError, match="high must be >= open and close"):
            DailyPrice(
                symbol=ETFSymbol.LQQ,
                date=date(2024, 1, 15),
                open=Decimal("100.00"),
                high=Decimal("100.00"),
                low=Decimal("99.00"),
                close=Decimal("101.00"),  # Higher than high
                volume=10000,
            )


class TestMacroIndicator:
    """Tests for MacroIndicator model."""

    def test_valid_indicator(self) -> None:
        """Test creating valid macro indicator."""
        indicator = MacroIndicator(
            indicator_name="VIX",
            date=date(2024, 1, 15),
            value=15.5,
            source="FRED",
        )
        assert indicator.indicator_name == "VIX"
        assert indicator.value == 15.5

    def test_default_source(self) -> None:
        """Test default source is FRED."""
        indicator = MacroIndicator(
            indicator_name="DGS10",
            date=date(2024, 1, 15),
            value=4.25,
        )
        assert indicator.source == "FRED"


class TestRegime:
    """Tests for Regime enum."""

    def test_regime_values(self) -> None:
        """Test regime enum values."""
        assert Regime.RISK_ON.value == "risk_on"
        assert Regime.NEUTRAL.value == "neutral"
        assert Regime.RISK_OFF.value == "risk_off"


class TestAllocationRecommendation:
    """Tests for AllocationRecommendation model."""

    def test_valid_allocation(self) -> None:
        """Test creating valid allocation."""
        alloc = AllocationRecommendation(
            date=date(2024, 1, 15),
            regime=Regime.NEUTRAL,
            lqq_weight=0.10,
            cl2_weight=0.10,
            wpea_weight=0.60,
            cash_weight=0.20,
            confidence=0.75,
        )
        assert alloc.regime == Regime.NEUTRAL
        assert alloc.confidence == 0.75

    def test_weights_must_sum_to_one(self) -> None:
        """Test that weights must sum to 1."""
        with pytest.raises(ValidationError, match="Weights must sum to 1.0"):
            AllocationRecommendation(
                date=date(2024, 1, 15),
                regime=Regime.NEUTRAL,
                lqq_weight=0.10,
                cl2_weight=0.10,
                wpea_weight=0.30,  # Total = 0.70
                cash_weight=0.20,
                confidence=0.75,
            )

    def test_leveraged_max_30_percent(self) -> None:
        """Test that leveraged ETFs cannot exceed 30%."""
        with pytest.raises(ValidationError, match="exceeds 30% limit"):
            AllocationRecommendation(
                date=date(2024, 1, 15),
                regime=Regime.RISK_ON,
                lqq_weight=0.20,
                cl2_weight=0.20,  # Total leveraged = 40%
                wpea_weight=0.50,
                cash_weight=0.10,
                confidence=0.75,
            )

    def test_cash_minimum_10_percent(self) -> None:
        """Test that cash must be at least 10%."""
        with pytest.raises(ValidationError):
            AllocationRecommendation(
                date=date(2024, 1, 15),
                regime=Regime.RISK_ON,
                lqq_weight=0.15,
                cl2_weight=0.15,
                wpea_weight=0.65,
                cash_weight=0.05,  # Below 10% minimum
                confidence=0.75,
            )


class TestPosition:
    """Tests for Position model."""

    def test_valid_position(self) -> None:
        """Test creating valid position."""
        pos = Position(
            symbol=ETFSymbol.WPEA,
            shares=100.0,
            average_cost=50.0,
            current_price=55.0,
            market_value=5500.0,
            unrealized_pnl=500.0,
            weight=0.60,
        )
        assert pos.symbol == ETFSymbol.WPEA
        assert pos.unrealized_pnl == 500.0


class TestTrade:
    """Tests for Trade model."""

    def test_valid_buy_trade(self) -> None:
        """Test creating valid buy trade."""
        trade = Trade(
            symbol=ETFSymbol.LQQ,
            date=datetime(2024, 1, 15, 10, 30, 0),
            action=TradeAction.BUY,
            shares=10.0,
            price=100.0,
            total_value=1000.0,
            commission=1.0,
        )
        assert trade.action == TradeAction.BUY
        assert trade.total_value == 1000.0

    def test_total_value_validation(self) -> None:
        """Test that total_value must match shares * price."""
        with pytest.raises(ValidationError, match="total_value"):
            Trade(
                symbol=ETFSymbol.LQQ,
                date=datetime(2024, 1, 15, 10, 30, 0),
                action=TradeAction.BUY,
                shares=10.0,
                price=100.0,
                total_value=500.0,  # Should be 1000
                commission=0.0,
            )
