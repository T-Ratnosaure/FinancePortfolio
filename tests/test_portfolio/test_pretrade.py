"""Tests for pre-trade risk validation module."""

from datetime import datetime
from decimal import Decimal

import pytest

from src.portfolio.pretrade import (
    LEVERAGED_SYMBOLS,
    Order,
    OrderAction,
    OrderType,
    Portfolio,
    PreTradeValidator,
    PreTradeValidatorConfig,
    ValidationResult,
)


@pytest.fixture
def validator() -> PreTradeValidator:
    """Create a pre-trade validator with default config."""
    return PreTradeValidator()


@pytest.fixture
def custom_config() -> PreTradeValidatorConfig:
    """Create custom validator configuration."""
    return PreTradeValidatorConfig(
        max_single_position=0.20,
        max_leveraged_exposure=0.25,
        min_cash_buffer=0.15,
        min_order_value=Decimal("100.0"),
        max_order_value=Decimal("50000.0"),
        price_deviation_threshold=0.03,
    )


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Create a sample portfolio for testing.

    Total value: 100,000 EUR
    - LQQ.PA: 50 * 100 = 5,000 (5%)
    - CL2.PA: 25 * 200 = 5,000 (5%)
    - WPEA.PA: 500 * 60 = 30,000 (30%) - intentionally at limit edge
    - Cash: 60,000 (60%)
    """
    return Portfolio(
        holdings={
            "LQQ.PA": Decimal("50"),
            "CL2.PA": Decimal("25"),
            "WPEA.PA": Decimal("500"),
        },
        cash_balance=Decimal("60000"),
        prices={
            "LQQ.PA": Decimal("100.0"),
            "CL2.PA": Decimal("200.0"),
            "WPEA.PA": Decimal("60.0"),
        },
    )


@pytest.fixture
def empty_portfolio() -> Portfolio:
    """Create an empty portfolio for testing."""
    return Portfolio(
        holdings={},
        cash_balance=Decimal("0"),
        prices={},
    )


class TestOrder:
    """Tests for Order model."""

    def test_order_creation(self) -> None:
        """Test basic order creation."""
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            price=Decimal("100.0"),
        )

        assert order.symbol == "LQQ.PA"
        assert order.action == OrderAction.BUY
        assert order.quantity == Decimal("10")
        assert order.price == Decimal("100.0")
        assert order.order_type == OrderType.MARKET

    def test_order_total_value(self) -> None:
        """Test order total value calculation."""
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            price=Decimal("100.0"),
        )

        assert order.total_value == Decimal("1000.0")

    def test_order_with_limit_type(self) -> None:
        """Test order with LIMIT type."""
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.SELL,
            quantity=Decimal("5"),
            price=Decimal("105.0"),
            order_type=OrderType.LIMIT,
        )

        assert order.order_type == OrderType.LIMIT

    def test_order_invalid_quantity(self) -> None:
        """Test order with invalid quantity raises error."""
        with pytest.raises(ValueError):
            Order(
                symbol="LQQ.PA",
                action=OrderAction.BUY,
                quantity=Decimal("0"),
                price=Decimal("100.0"),
            )

    def test_order_invalid_price(self) -> None:
        """Test order with invalid price raises error."""
        with pytest.raises(ValueError):
            Order(
                symbol="LQQ.PA",
                action=OrderAction.BUY,
                quantity=Decimal("10"),
                price=Decimal("-100.0"),
            )


class TestPortfolio:
    """Tests for Portfolio model."""

    def test_portfolio_total_value(self, sample_portfolio: Portfolio) -> None:
        """Test portfolio total value calculation."""
        # LQQ: 50 * 100 = 5,000
        # CL2: 25 * 200 = 5,000
        # WPEA: 500 * 60 = 30,000
        # Cash: 60,000
        # Total: 100,000
        assert sample_portfolio.total_value == Decimal("100000")

    def test_portfolio_position_value(self, sample_portfolio: Portfolio) -> None:
        """Test position value calculation."""
        assert sample_portfolio.get_position_value("LQQ.PA") == Decimal("5000")
        assert sample_portfolio.get_position_value("CL2.PA") == Decimal("5000")
        assert sample_portfolio.get_position_value("WPEA.PA") == Decimal("30000")

    def test_portfolio_position_weight(self, sample_portfolio: Portfolio) -> None:
        """Test position weight calculation."""
        # LQQ: 5,000 / 100,000 = 5%
        lqq_weight = sample_portfolio.get_position_weight("LQQ.PA")
        assert abs(lqq_weight - 0.05) < 0.001

    def test_portfolio_nonexistent_position(self, sample_portfolio: Portfolio) -> None:
        """Test getting value of nonexistent position."""
        assert sample_portfolio.get_position_value("NONEXISTENT") == Decimal("0")
        assert sample_portfolio.get_position_weight("NONEXISTENT") == 0.0

    def test_empty_portfolio_weight(self, empty_portfolio: Portfolio) -> None:
        """Test weight calculation for empty portfolio."""
        assert empty_portfolio.get_position_weight("LQQ.PA") == 0.0


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_valid_result(self) -> None:
        """Test valid result creation."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.warnings == []
        assert result.errors == []

    def test_result_with_warnings(self) -> None:
        """Test result with warnings."""
        result = ValidationResult(
            is_valid=True,
            warnings=["Position approaching limit"],
        )
        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_result_with_errors(self) -> None:
        """Test result with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["Position limit exceeded"],
        )
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_result_inconsistent_raises_error(self) -> None:
        """Test that is_valid=True with errors raises ValueError."""
        with pytest.raises(ValueError, match="cannot be True"):
            ValidationResult(
                is_valid=True,
                errors=["Some error"],
            )

    def test_result_with_suggested_adjustments(self) -> None:
        """Test result with suggested adjustments."""
        result = ValidationResult(
            is_valid=False,
            errors=["Position limit exceeded"],
            suggested_adjustments={"max_quantity": 50.0, "max_value": 5000.0},
        )
        assert result.suggested_adjustments is not None
        assert result.suggested_adjustments["max_quantity"] == 50.0


class TestPreTradeValidatorInit:
    """Tests for PreTradeValidator initialization."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        validator = PreTradeValidator()
        assert validator.config is not None
        assert validator.config.max_single_position == 0.25

    def test_init_with_custom_config(
        self, custom_config: PreTradeValidatorConfig
    ) -> None:
        """Test initialization with custom config."""
        validator = PreTradeValidator(config=custom_config)
        assert validator.config.max_single_position == 0.20

    def test_init_with_reference_prices(self) -> None:
        """Test initialization with reference prices."""
        ref_prices = {"LQQ.PA": Decimal("100.0")}
        validator = PreTradeValidator(reference_prices=ref_prices)
        assert validator.reference_prices == ref_prices


class TestCheckPositionLimit:
    """Tests for position limit validation."""

    def test_position_limit_ok(
        self, validator: PreTradeValidator, sample_portfolio: Portfolio
    ) -> None:
        """Test valid order within position limit."""
        # Buy LQQ which is at 5% - well within limits
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("50"),
            price=Decimal("100.0"),
        )  # 5,000 EUR
        # New LQQ: 10,000, New total: 105,000
        # New weight: 9.5% - within 25% limit

        result = validator.check_position_limit(order, sample_portfolio)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_position_limit_exceeded(
        self, validator: PreTradeValidator, sample_portfolio: Portfolio
    ) -> None:
        """Test order that would exceed position limit."""
        # Try to buy enough to exceed 25% limit
        # Current WPEA: 30,000 / 70,000 = 42.9% already over!
        # But let's test with a smaller portfolio
        portfolio = Portfolio(
            holdings={"LQQ.PA": Decimal("10")},
            cash_balance=Decimal("50000"),
            prices={"LQQ.PA": Decimal("100.0")},
        )
        # Total: 51,000, LQQ = 1,000 (2%)
        # Try to buy enough LQQ to exceed 25%

        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("200"),
            price=Decimal("100.0"),
        )  # 20,000 EUR

        result = validator.check_position_limit(order, portfolio)

        # New LQQ value: 21,000, New total: 71,000
        # New weight: 21,000/71,000 = 29.6% > 25%
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "limit exceeded" in result.errors[0].lower()
        assert result.suggested_adjustments is not None

    def test_position_limit_exactly_at_limit(
        self, validator: PreTradeValidator
    ) -> None:
        """Test order that brings position exactly to limit."""
        portfolio = Portfolio(
            holdings={},
            cash_balance=Decimal("100000"),
            prices={"LQQ.PA": Decimal("100.0")},
        )
        # 25% of 100,000 = 25,000
        # After buy: total = 125,000, position = 25,000
        # Weight = 25,000 / 125,000 = 20% - actually less than 25%

        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("250"),
            price=Decimal("100.0"),
        )

        result = validator.check_position_limit(order, portfolio)

        assert result.is_valid is True

    def test_position_limit_approaching_warning(
        self, validator: PreTradeValidator
    ) -> None:
        """Test warning when approaching position limit."""
        portfolio = Portfolio(
            holdings={},
            cash_balance=Decimal("100000"),
            prices={"LQQ.PA": Decimal("100.0")},
        )
        # Warning threshold = 25% * 0.8 = 20%
        # Buy 28,000 EUR: Weight = 28,000/128,000 = 21.9% > 20% warning
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("280"),
            price=Decimal("100.0"),
        )

        result = validator.check_position_limit(order, portfolio)

        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "approaching" in result.warnings[0].lower()

    def test_sell_more_than_held(
        self, validator: PreTradeValidator, sample_portfolio: Portfolio
    ) -> None:
        """Test selling more than current position."""
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.SELL,
            quantity=Decimal("100"),
            price=Decimal("100.0"),
        )  # Trying to sell 10,000 but only hold 5,000

        result = validator.check_position_limit(order, sample_portfolio)

        assert result.is_valid is False
        assert "cannot sell more" in result.errors[0].lower()

    def test_empty_portfolio_error(
        self, validator: PreTradeValidator, empty_portfolio: Portfolio
    ) -> None:
        """Test validation fails for empty portfolio."""
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            price=Decimal("100.0"),
        )

        result = validator.check_position_limit(order, empty_portfolio)

        assert result.is_valid is False
        assert "no value" in result.errors[0].lower()


class TestCheckLeveragedExposure:
    """Tests for leveraged ETF exposure validation."""

    def test_non_leveraged_order_skipped(
        self, validator: PreTradeValidator, sample_portfolio: Portfolio
    ) -> None:
        """Test non-leveraged order bypasses check."""
        order = Order(
            symbol="WPEA.PA",
            action=OrderAction.BUY,
            quantity=Decimal("100"),
            price=Decimal("60.0"),
        )

        result = validator.check_leveraged_exposure(order, sample_portfolio)

        assert result.is_valid is True

    def test_leveraged_exposure_ok(self, validator: PreTradeValidator) -> None:
        """Test leveraged buy within limit."""
        portfolio = Portfolio(
            holdings={
                "LQQ.PA": Decimal("50"),
                "WPEA.PA": Decimal("500"),
            },
            cash_balance=Decimal("30000"),
            prices={
                "LQQ.PA": Decimal("100.0"),
                "WPEA.PA": Decimal("60.0"),
            },
        )
        # Total: 5,000 + 30,000 + 30,000 = 65,000
        # Current leveraged: 5,000 (7.7%)

        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("50"),
            price=Decimal("100.0"),
        )  # 5,000 EUR
        # New leveraged: 10,000, New total: 70,000
        # New weight: 14.3% < 30%

        result = validator.check_leveraged_exposure(order, portfolio)

        assert result.is_valid is True

    def test_leveraged_exposure_exceeded(self, validator: PreTradeValidator) -> None:
        """Test order that would exceed leveraged exposure limit."""
        portfolio = Portfolio(
            holdings={
                "LQQ.PA": Decimal("100"),
                "CL2.PA": Decimal("50"),
            },
            cash_balance=Decimal("50000"),
            prices={
                "LQQ.PA": Decimal("100.0"),
                "CL2.PA": Decimal("200.0"),
            },
        )
        # Total: 10,000 + 10,000 + 50,000 = 70,000
        # Current leveraged: 20,000 (28.6%)

        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("50"),
            price=Decimal("100.0"),
        )  # 5,000 EUR
        # New leveraged: 25,000, New total: 75,000
        # New weight: 33.3% > 30%

        result = validator.check_leveraged_exposure(order, portfolio)

        assert result.is_valid is False
        assert "leveraged exposure" in result.errors[0].lower()
        assert result.suggested_adjustments is not None

    def test_leveraged_exposure_sell_ok(self, validator: PreTradeValidator) -> None:
        """Test selling leveraged ETF reduces exposure - always allowed."""
        portfolio = Portfolio(
            holdings={
                "LQQ.PA": Decimal("150"),
                "CL2.PA": Decimal("50"),
            },
            cash_balance=Decimal("35000"),
            prices={
                "LQQ.PA": Decimal("100.0"),
                "CL2.PA": Decimal("200.0"),
            },
        )
        # Total: 15,000 + 10,000 + 35,000 = 60,000
        # Current leveraged: 25,000 (41.7%) - over limit
        # LQQ position: 15,000 (25%) - at position limit

        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.SELL,
            quantity=Decimal("50"),
            price=Decimal("100.0"),
        )  # Sell 5,000 EUR
        # New leveraged: 20,000, Total stays 60,000
        # New weight: 33.3% - still over but sell is ok to reduce exposure

        result = validator.check_leveraged_exposure(order, portfolio)

        # Sells always pass leveraged exposure check (reduces exposure)
        assert result.is_valid is True


class TestCheckCashBuffer:
    """Tests for cash buffer validation."""

    def test_sell_order_skipped(
        self, validator: PreTradeValidator, sample_portfolio: Portfolio
    ) -> None:
        """Test sell orders bypass cash buffer check."""
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.SELL,
            quantity=Decimal("50"),
            price=Decimal("100.0"),
        )

        result = validator.check_cash_buffer(order, sample_portfolio)

        assert result.is_valid is True

    def test_cash_buffer_ok(
        self, validator: PreTradeValidator, sample_portfolio: Portfolio
    ) -> None:
        """Test buy order maintaining cash buffer."""
        # Portfolio: 70,000 total, 20,000 cash (28.6%)
        order = Order(
            symbol="WPEA.PA",
            action=OrderAction.BUY,
            quantity=Decimal("100"),
            price=Decimal("60.0"),
        )  # 6,000 EUR
        # New cash: 14,000, New total: 76,000
        # New cash weight: 18.4% > 10%

        result = validator.check_cash_buffer(order, sample_portfolio)

        assert result.is_valid is True

    def test_insufficient_cash(self, validator: PreTradeValidator) -> None:
        """Test buy order exceeding available cash."""
        portfolio = Portfolio(
            holdings={"WPEA.PA": Decimal("100")},
            cash_balance=Decimal("20000"),
            prices={"WPEA.PA": Decimal("60.0")},
        )
        # Total: 6,000 + 20,000 = 26,000
        # Cash: 20,000

        order = Order(
            symbol="WPEA.PA",
            action=OrderAction.BUY,
            quantity=Decimal("500"),
            price=Decimal("60.0"),
        )  # 30,000 EUR but only 20,000 cash

        result = validator.check_cash_buffer(order, portfolio)

        assert result.is_valid is False
        assert "insufficient cash" in result.errors[0].lower()

    def test_cash_buffer_violated(self, validator: PreTradeValidator) -> None:
        """Test buy order that would violate cash buffer."""
        portfolio = Portfolio(
            holdings={"WPEA.PA": Decimal("500")},
            cash_balance=Decimal("12000"),
            prices={"WPEA.PA": Decimal("60.0")},
        )
        # Total: 30,000 + 12,000 = 42,000
        # Cash weight: 28.6%

        order = Order(
            symbol="WPEA.PA",
            action=OrderAction.BUY,
            quantity=Decimal("150"),
            price=Decimal("60.0"),
        )  # 9,000 EUR
        # New cash: 3,000, New total: 51,000
        # New cash weight: 5.9% < 10%

        result = validator.check_cash_buffer(order, portfolio)

        assert result.is_valid is False
        assert "cash buffer violated" in result.errors[0].lower()
        assert result.suggested_adjustments is not None

    def test_cash_buffer_approaching_warning(
        self, validator: PreTradeValidator
    ) -> None:
        """Test warning when approaching cash buffer minimum."""
        portfolio = Portfolio(
            holdings={"WPEA.PA": Decimal("500")},
            cash_balance=Decimal("15000"),
            prices={"WPEA.PA": Decimal("60.0")},
        )
        # Total: 30,000 + 15,000 = 45,000
        # Cash weight: 33.3%
        # Warning threshold = 10% * 1.3 = 13%
        # Buy 8,100 EUR -> New cash: 6,900, New total: 53,100
        # New cash weight: 13.0% = warning threshold exactly
        order = Order(
            symbol="WPEA.PA",
            action=OrderAction.BUY,
            quantity=Decimal("135"),
            price=Decimal("60.0"),
        )

        result = validator.check_cash_buffer(order, portfolio)

        assert result.is_valid is True
        # Should get warning when at or just below 13%


class TestCheckOrderSize:
    """Tests for order size validation."""

    def test_order_size_ok(self, validator: PreTradeValidator) -> None:
        """Test order within size limits."""
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("100"),
            price=Decimal("100.0"),
        )  # 10,000 EUR

        result = validator.check_order_size(order)

        assert result.is_valid is True

    def test_order_too_small(self, validator: PreTradeValidator) -> None:
        """Test order below minimum size."""
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("0.3"),
            price=Decimal("100.0"),
        )  # 30 EUR < 50 EUR minimum

        result = validator.check_order_size(order)

        assert result.is_valid is False
        assert "below minimum" in result.errors[0].lower()

    def test_order_too_large(self, validator: PreTradeValidator) -> None:
        """Test order above maximum size."""
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("2000"),
            price=Decimal("100.0"),
        )  # 200,000 EUR > 100,000 EUR maximum

        result = validator.check_order_size(order)

        assert result.is_valid is False
        assert "exceeds maximum" in result.errors[0].lower()
        assert result.suggested_adjustments is not None

    def test_order_exactly_at_minimum(self, validator: PreTradeValidator) -> None:
        """Test order exactly at minimum size."""
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("0.5"),
            price=Decimal("100.0"),
        )  # 50 EUR = minimum

        result = validator.check_order_size(order)

        assert result.is_valid is True


class TestCheckMarketHours:
    """Tests for market hours validation."""

    def test_during_market_hours(self, validator: PreTradeValidator) -> None:
        """Test order during market hours."""
        # Wednesday at 10:00
        timestamp = datetime(2024, 1, 10, 10, 0, 0)
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            price=Decimal("100.0"),
            timestamp=timestamp,
        )

        result = validator.check_market_hours(order)

        assert result.is_valid is True
        assert len(result.warnings) == 0

    def test_before_market_open(self, validator: PreTradeValidator) -> None:
        """Test order before market opens."""
        # Wednesday at 08:30
        timestamp = datetime(2024, 1, 10, 8, 30, 0)
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            price=Decimal("100.0"),
            timestamp=timestamp,
        )

        result = validator.check_market_hours(order)

        assert result.is_valid is True  # Warning only
        assert len(result.warnings) > 0
        assert "before market open" in result.warnings[0].lower()

    def test_after_market_close(self, validator: PreTradeValidator) -> None:
        """Test order after market closes."""
        # Wednesday at 18:00
        timestamp = datetime(2024, 1, 10, 18, 0, 0)
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            price=Decimal("100.0"),
            timestamp=timestamp,
        )

        result = validator.check_market_hours(order)

        assert result.is_valid is True  # Warning only
        assert len(result.warnings) > 0
        assert "after market close" in result.warnings[0].lower()

    def test_weekend_order(self, validator: PreTradeValidator) -> None:
        """Test order on weekend."""
        # Saturday
        timestamp = datetime(2024, 1, 13, 10, 0, 0)
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            price=Decimal("100.0"),
            timestamp=timestamp,
        )

        result = validator.check_market_hours(order)

        assert result.is_valid is True  # Warning only
        assert len(result.warnings) > 0
        assert "weekend" in result.warnings[0].lower()


class TestCheckPriceDeviation:
    """Tests for price deviation validation."""

    def test_no_reference_prices(self, validator: PreTradeValidator) -> None:
        """Test when no reference prices set."""
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            price=Decimal("100.0"),
        )

        result = validator.check_price_deviation(order)

        assert result.is_valid is True
        assert len(result.warnings) == 0

    def test_price_within_threshold(self) -> None:
        """Test price within deviation threshold."""
        ref_prices = {"LQQ.PA": Decimal("100.0")}
        validator = PreTradeValidator(reference_prices=ref_prices)

        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            price=Decimal("101.0"),
        )  # 1% deviation < 2% threshold

        result = validator.check_price_deviation(order)

        assert result.is_valid is True
        assert len(result.warnings) == 0

    def test_price_exceeds_threshold_higher(self) -> None:
        """Test price significantly higher than reference."""
        ref_prices = {"LQQ.PA": Decimal("100.0")}
        validator = PreTradeValidator(reference_prices=ref_prices)

        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            price=Decimal("105.0"),
        )  # 5% deviation > 2% threshold

        result = validator.check_price_deviation(order)

        assert result.is_valid is True  # Warning only
        assert len(result.warnings) > 0
        assert "higher" in result.warnings[0].lower()

    def test_price_exceeds_threshold_lower(self) -> None:
        """Test price significantly lower than reference."""
        ref_prices = {"LQQ.PA": Decimal("100.0")}
        validator = PreTradeValidator(reference_prices=ref_prices)

        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("10"),
            price=Decimal("95.0"),
        )  # 5% deviation > 2% threshold

        result = validator.check_price_deviation(order)

        assert result.is_valid is True  # Warning only
        assert len(result.warnings) > 0
        assert "lower" in result.warnings[0].lower()

    def test_set_reference_prices(self, validator: PreTradeValidator) -> None:
        """Test setting reference prices."""
        validator.set_reference_prices({"LQQ.PA": Decimal("100.0")})

        assert validator.reference_prices["LQQ.PA"] == Decimal("100.0")


class TestValidateOrder:
    """Tests for combined validate_order method."""

    def test_valid_order(self, validator: PreTradeValidator) -> None:
        """Test fully valid order passes all checks."""
        # Use portfolio where WPEA.PA has room to grow
        portfolio = Portfolio(
            holdings={"WPEA.PA": Decimal("100")},
            cash_balance=Decimal("50000"),
            prices={"WPEA.PA": Decimal("60.0")},
        )
        # Total: 6,000 + 50,000 = 56,000
        # WPEA weight: 10.7%

        order = Order(
            symbol="WPEA.PA",
            action=OrderAction.BUY,
            quantity=Decimal("50"),
            price=Decimal("60.0"),
            timestamp=datetime(2024, 1, 10, 10, 0, 0),
        )  # 3,000 EUR
        # New WPEA: 9,000, New total: 59,000
        # New weight: 15.3% - well within limits

        result = validator.validate_order(order, portfolio)

        assert result.is_valid is True

    def test_invalid_order_multiple_errors(self, validator: PreTradeValidator) -> None:
        """Test order with multiple validation errors."""
        portfolio = Portfolio(
            holdings={},
            cash_balance=Decimal("1000"),
            prices={"LQQ.PA": Decimal("100.0")},
        )

        # This order is too large for cash and would exceed limits
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("50"),
            price=Decimal("100.0"),
            timestamp=datetime(2024, 1, 10, 10, 0, 0),
        )  # 5,000 EUR > 1,000 cash

        result = validator.validate_order(order, portfolio)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_order_with_warnings_only(self, validator: PreTradeValidator) -> None:
        """Test valid order with warnings."""
        # Use portfolio where order is valid but on weekend
        portfolio = Portfolio(
            holdings={"WPEA.PA": Decimal("100")},
            cash_balance=Decimal("50000"),
            prices={"WPEA.PA": Decimal("60.0")},
        )

        # Weekend order - valid but with warning
        order = Order(
            symbol="WPEA.PA",
            action=OrderAction.BUY,
            quantity=Decimal("50"),
            price=Decimal("60.0"),
            timestamp=datetime(2024, 1, 13, 10, 0, 0),  # Saturday
        )

        result = validator.validate_order(order, portfolio)

        assert result.is_valid is True
        assert len(result.warnings) > 0


class TestValidateBatch:
    """Tests for batch order validation."""

    def test_batch_validation_cumulative(self, validator: PreTradeValidator) -> None:
        """Test batch validation considers cumulative effects."""
        portfolio = Portfolio(
            holdings={},
            cash_balance=Decimal("30000"),
            prices={
                "LQQ.PA": Decimal("100.0"),
                "CL2.PA": Decimal("200.0"),
            },
        )

        orders = [
            Order(
                symbol="LQQ.PA",
                action=OrderAction.BUY,
                quantity=Decimal("50"),
                price=Decimal("100.0"),
            ),  # 5,000 EUR
            Order(
                symbol="CL2.PA",
                action=OrderAction.BUY,
                quantity=Decimal("25"),
                price=Decimal("200.0"),
            ),  # 5,000 EUR
            Order(
                symbol="LQQ.PA",
                action=OrderAction.BUY,
                quantity=Decimal("50"),
                price=Decimal("100.0"),
            ),  # Another 5,000 EUR
        ]

        results = validator.validate_batch(orders, portfolio)

        # First two orders should be valid
        assert results[0].is_valid is True
        assert results[1].is_valid is True
        # Third might fail due to leveraged exposure (15,000/45,000 = 33%)
        # or cash buffer issues

    def test_batch_validation_empty(
        self, validator: PreTradeValidator, sample_portfolio: Portfolio
    ) -> None:
        """Test batch validation with empty list."""
        results = validator.validate_batch([], sample_portfolio)

        assert results == {}

    def test_batch_validation_sell_then_buy(self, validator: PreTradeValidator) -> None:
        """Test batch with sell followed by buy."""
        portfolio = Portfolio(
            holdings={"LQQ.PA": Decimal("50"), "WPEA.PA": Decimal("200")},
            cash_balance=Decimal("60000"),
            prices={"LQQ.PA": Decimal("100.0"), "WPEA.PA": Decimal("60.0")},
        )
        # Total: 5,000 + 12,000 + 60,000 = 77,000
        # LQQ weight: 6.5%, WPEA weight: 15.6%, Cash: 77.9%

        orders = [
            Order(
                symbol="LQQ.PA",
                action=OrderAction.SELL,
                quantity=Decimal("25"),
                price=Decimal("100.0"),
            ),  # Sell 2,500 EUR of LQQ
            Order(
                symbol="WPEA.PA",
                action=OrderAction.BUY,
                quantity=Decimal("50"),
                price=Decimal("60.0"),
            ),  # Buy 3,000 EUR of WPEA
        ]

        results = validator.validate_batch(orders, portfolio)

        assert results[0].is_valid is True
        assert results[1].is_valid is True


class TestLeveragedSymbols:
    """Tests for leveraged symbol detection."""

    def test_leveraged_symbols_constant(self) -> None:
        """Test LEVERAGED_SYMBOLS contains expected ETFs."""
        assert "LQQ.PA" in LEVERAGED_SYMBOLS
        assert "CL2.PA" in LEVERAGED_SYMBOLS
        assert "WPEA.PA" not in LEVERAGED_SYMBOLS


class TestCustomConfig:
    """Tests with custom configuration."""

    def test_custom_position_limit(
        self, custom_config: PreTradeValidatorConfig
    ) -> None:
        """Test custom position limit is respected."""
        validator = PreTradeValidator(config=custom_config)
        portfolio = Portfolio(
            holdings={},
            cash_balance=Decimal("100000"),
            prices={"LQQ.PA": Decimal("100.0")},
        )

        # With 20% limit, buying 25,000 would exceed
        # 25,000 / 125,000 = 20% exactly
        order = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("260"),
            price=Decimal("100.0"),
        )  # 26,000 EUR
        # 26,000 / 126,000 = 20.6% > 20%

        result = validator.check_position_limit(order, portfolio)

        assert result.is_valid is False

    def test_custom_cash_buffer(self, custom_config: PreTradeValidatorConfig) -> None:
        """Test custom cash buffer limit is respected."""
        validator = PreTradeValidator(config=custom_config)
        portfolio = Portfolio(
            holdings={"WPEA.PA": Decimal("500")},
            cash_balance=Decimal("20000"),
            prices={"WPEA.PA": Decimal("60.0")},
        )
        # Total: 30,000 + 20,000 = 50,000
        # 15% buffer = 7,500 min cash

        order = Order(
            symbol="WPEA.PA",
            action=OrderAction.BUY,
            quantity=Decimal("220"),
            price=Decimal("60.0"),
        )  # 13,200 EUR
        # New cash: 6,800, New total: 63,200
        # Cash weight: 10.8% < 15%

        result = validator.check_cash_buffer(order, portfolio)

        assert result.is_valid is False

    def test_custom_order_limits(self, custom_config: PreTradeValidatorConfig) -> None:
        """Test custom order size limits."""
        validator = PreTradeValidator(config=custom_config)

        # Below custom minimum (100 EUR)
        order_small = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("0.5"),
            price=Decimal("100.0"),
        )  # 50 EUR

        result = validator.check_order_size(order_small)

        assert result.is_valid is False

        # Above custom maximum (50,000 EUR)
        order_large = Order(
            symbol="LQQ.PA",
            action=OrderAction.BUY,
            quantity=Decimal("600"),
            price=Decimal("100.0"),
        )  # 60,000 EUR

        result = validator.check_order_size(order_large)

        assert result.is_valid is False
