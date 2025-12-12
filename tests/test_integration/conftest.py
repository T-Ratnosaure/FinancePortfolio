"""Shared fixtures for integration tests.

Provides realistic mock data and common setup for integration testing.
"""

from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.data.models import DailyPrice, ETFSymbol, MacroIndicator


@pytest.fixture
def mock_price_data() -> pd.DataFrame:
    """Generate mock price data for PEA ETFs.

    Returns:
        DataFrame with columns: date, symbol, close, volume
        Covers 10 years of daily data (2500 trading days)
    """
    rng = np.random.default_rng(42)
    start_date = date(2015, 1, 1)

    # Generate 2500 trading days
    dates = pd.date_range(start=start_date, periods=2500, freq="B")

    symbols = [ETFSymbol.LQQ.value, ETFSymbol.CL2.value, ETFSymbol.WPEA.value]

    data = []
    for symbol in symbols:
        # Generate realistic price series
        base_price = 100.0
        drift = 0.0003  # 7.5% annual drift
        vol = (
            0.015 if symbol == ETFSymbol.WPEA.value else 0.030
        )  # Higher vol for leveraged

        prices = [base_price]
        for _ in range(len(dates) - 1):
            shock = rng.normal(drift, vol)
            new_price = prices[-1] * (1 + shock)
            prices.append(new_price)

        for i, d in enumerate(dates):
            data.append(
                {
                    "date": d.date(),
                    "symbol": symbol,
                    "close": prices[i],
                    "volume": int(rng.integers(100000, 1000000)),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def mock_vix_data() -> pd.DataFrame:
    """Generate mock VIX data.

    Returns:
        DataFrame with columns: date, vix
        Covers 10 years with regime-like patterns
    """
    rng = np.random.default_rng(42)
    start_date = date(2015, 1, 1)
    dates = pd.date_range(start=start_date, periods=2500, freq="B")

    # Generate VIX with regime patterns
    # Low VIX: ~12, Medium: ~18, High: ~30
    vix_values = []
    regime = 0  # 0 = low, 1 = neutral, 2 = high

    for i in range(len(dates)):
        # Switch regimes occasionally
        if i % 500 == 0 and i > 0:
            regime = (regime + 1) % 3

        if regime == 0:  # RISK_ON
            base_vix = 12.0
            vol = 2.0
        elif regime == 1:  # NEUTRAL
            base_vix = 18.0
            vol = 3.0
        else:  # RISK_OFF
            base_vix = 30.0
            vol = 5.0

        vix = max(9.0, base_vix + rng.normal(0, vol))
        vix_values.append(vix)

    return pd.DataFrame({"date": [d.date() for d in dates], "vix": vix_values})


@pytest.fixture
def mock_treasury_data() -> pd.DataFrame:
    """Generate mock Treasury yield data.

    Returns:
        DataFrame with columns: date, treasury_2y, treasury_10y
    """
    rng = np.random.default_rng(42)
    start_date = date(2015, 1, 1)
    dates = pd.date_range(start=start_date, periods=2500, freq="B")

    # Generate yields with realistic ranges
    t2y = 2.0 + rng.normal(0, 0.5, size=len(dates))
    t10y = 2.5 + rng.normal(0, 0.6, size=len(dates))

    # Ensure 10Y > 2Y most of the time (normal curve)
    t10y = np.maximum(t10y, t2y + 0.1)

    return pd.DataFrame(
        {
            "date": [d.date() for d in dates],
            "treasury_2y": t2y,
            "treasury_10y": t10y,
        }
    )


@pytest.fixture
def mock_hy_spread_data() -> pd.DataFrame:
    """Generate mock high-yield spread data.

    Returns:
        DataFrame with columns: date, hy_spread
    """
    rng = np.random.default_rng(42)
    start_date = date(2015, 1, 1)
    dates = pd.date_range(start=start_date, periods=2500, freq="B")

    # Generate spreads (3-6% typical range)
    hy_spread = 4.0 + rng.normal(0, 1.0, size=len(dates))
    hy_spread = np.maximum(hy_spread, 2.0)  # Floor at 2%

    return pd.DataFrame({"date": [d.date() for d in dates], "hy_spread": hy_spread})


@pytest.fixture
def mock_daily_prices() -> list[DailyPrice]:
    """Generate mock DailyPrice objects for testing.

    Returns:
        List of DailyPrice Pydantic models
    """
    rng = np.random.default_rng(42)
    start_date = date(2015, 1, 1)
    prices = []

    for i in range(300):  # 300 trading days
        current_date = start_date + timedelta(days=i)

        for symbol in [ETFSymbol.LQQ, ETFSymbol.CL2, ETFSymbol.WPEA]:
            base = 100.0
            price_val = Decimal(str(base + rng.uniform(-5, 10)))

            prices.append(
                DailyPrice(
                    symbol=symbol,
                    date=current_date,
                    open=price_val * Decimal("0.99"),
                    high=price_val * Decimal("1.02"),
                    low=price_val * Decimal("0.98"),
                    close=price_val,
                    volume=int(rng.integers(100000, 1000000)),
                    adjusted_close=price_val,
                )
            )

    return prices


@pytest.fixture
def mock_macro_indicators() -> list[MacroIndicator]:
    """Generate mock MacroIndicator objects.

    Returns:
        List of MacroIndicator Pydantic models for VIX, yields, spreads
    """
    rng = np.random.default_rng(42)
    start_date = date(2015, 1, 1)
    indicators = []

    for i in range(300):
        current_date = start_date + timedelta(days=i)

        indicators.extend(
            [
                MacroIndicator(
                    indicator_name="VIX",
                    date=current_date,
                    value=12.0 + rng.normal(0, 3.0),
                    source="CBOE",
                ),
                MacroIndicator(
                    indicator_name="DGS2",
                    date=current_date,
                    value=2.0 + rng.normal(0, 0.3),
                    source="FRED",
                ),
                MacroIndicator(
                    indicator_name="DGS10",
                    date=current_date,
                    value=2.5 + rng.normal(0, 0.3),
                    source="FRED",
                ),
                MacroIndicator(
                    indicator_name="BAMLH0A0HYM2",
                    date=current_date,
                    value=4.0 + rng.normal(0, 0.5),
                    source="FRED",
                ),
            ]
        )

    return indicators
