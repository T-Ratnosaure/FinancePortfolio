"""Example: Data freshness and staleness detection.

This example demonstrates how to use the data freshness tracking system
to detect and warn about stale data before making investment decisions.
"""

from datetime import datetime, timedelta

from src.data.freshness import (
    check_price_data_freshness,
    generate_freshness_report,
)
from src.data.models import DailyPrice, ETFSymbol, MacroIndicator
from src.data.storage.duckdb import DuckDBStorage


def main() -> None:
    """Demonstrate data freshness tracking."""
    print("=" * 80)
    print("Data Freshness Detection Example")
    print("=" * 80)

    # Create a temporary database
    storage = DuckDBStorage("data/freshness_example.duckdb")

    try:
        # 1. Insert some price data
        print("\n1. Inserting fresh price data...")
        prices = [
            DailyPrice(
                symbol=ETFSymbol.LQQ,
                date=datetime.now().date(),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
                adjusted_close=103.0,
            ),
            DailyPrice(
                symbol=ETFSymbol.WPEA,
                date=datetime.now().date(),
                open=50.0,
                high=52.0,
                low=49.5,
                close=51.5,
                volume=500000,
                adjusted_close=51.5,
            ),
        ]
        storage.insert_prices(prices)
        print("  Price data inserted successfully")

        # 2. Insert some macro data
        print("\n2. Inserting fresh macro data...")
        indicators = [
            MacroIndicator(
                indicator_name="VIX",
                date=datetime.now().date(),
                value=15.5,
                source="FRED",
            ),
            MacroIndicator(
                indicator_name="TREASURY_10Y",
                date=datetime.now().date(),
                value=4.25,
                source="FRED",
            ),
        ]
        storage.insert_macro(indicators)
        print("  Macro data inserted successfully")

        # 3. Check freshness status
        print("\n3. Checking data freshness...")
        freshness_lqq = check_price_data_freshness(
            storage, "LQQ.PA", raise_on_critical=False
        )
        if freshness_lqq:
            print(f"  LQQ.PA status: {freshness_lqq.get_status().value}")
            print(f"  Last updated: {freshness_lqq.last_updated}")
            print(f"  Age: {freshness_lqq.get_age()}")

        # 4. Generate comprehensive report
        print("\n4. Generating freshness report...")
        report = generate_freshness_report(storage)
        print(report)

        # 5. Simulate stale data scenario
        print("\n5. Simulating stale data scenario...")
        print("  (Manually setting data to be 3 days old)")

        stale_time = datetime.now() - timedelta(days=3)
        storage.conn.execute(
            """
            UPDATE raw.data_freshness
            SET last_updated = ?, updated_at = ?
            WHERE symbol = 'LQQ.PA'
        """,
            [stale_time, stale_time],
        )

        # Check freshness again
        print("\n6. Checking freshness after simulating staleness...")
        freshness_lqq_stale = check_price_data_freshness(
            storage, "LQQ.PA", raise_on_critical=False
        )

        if freshness_lqq_stale:
            print(f"  LQQ.PA status: {freshness_lqq_stale.get_status().value}")
            warning = freshness_lqq_stale.get_warning_message()
            if warning:
                print(f"  {warning}")

        # 7. Generate report with stale data
        print("\n7. Generating report with stale data...")
        stale_report = generate_freshness_report(storage)
        print(stale_report)

        # 8. Show report as dictionary
        print("\n8. Report as JSON-compatible dictionary:")
        print(f"  Summary: {stale_report.to_dict()['summary']}")
        if stale_report.has_issues():
            print(f"  Issues found: {stale_report.stale_count} stale datasets")

    finally:
        storage.close()
        print("\n" + "=" * 80)
        print("Example completed")
        print("=" * 80)


if __name__ == "__main__":
    main()
