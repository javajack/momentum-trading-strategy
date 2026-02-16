#!/usr/bin/env python3
"""
Test script for dynamic rebalancing implementation.
Runs a 12-month backtest with the new features enabled.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from fortress.backtest import BacktestConfig, BacktestEngine
from fortress.config import load_config
from fortress.universe import Universe


def load_cached_data(cache_dir: str = ".cache") -> dict:
    """Load historical data from cache."""
    data = {}
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"Cache directory {cache_dir} not found!")
        return data

    parquet_files = list(cache_path.glob("*.parquet"))
    print(f"Found {len(parquet_files)} cached data files")

    # Mapping for index/special symbols (underscores to spaces)
    symbol_map = {
        "NIFTY_50": "NIFTY 50",
        "INDIA_VIX": "INDIA VIX",
        "NIFTY_MIDCAP_100": "NIFTY MIDCAP 100",
        "NIFTY_BANK": "NIFTY BANK",
        "NIFTY_IT": "NIFTY IT",
        "NIFTY_AUTO": "NIFTY AUTO",
        "NIFTY_PHARMA": "NIFTY PHARMA",
        "NIFTY_METAL": "NIFTY METAL",
        "NIFTY_FMCG": "NIFTY FMCG",
        "NIFTY_REALTY": "NIFTY REALTY",
        "NIFTY_ENERGY": "NIFTY ENERGY",
        "NIFTY_INFRA": "NIFTY INFRA",
        "NIFTY_MEDIA": "NIFTY MEDIA",
        "NIFTY_COMMODITIES": "NIFTY COMMODITIES",
        "NIFTY_CONSUMPTION": "NIFTY CONSUMPTION",
        "NIFTY_SERV_SECTOR": "NIFTY SERV SECTOR",
    }

    for file in parquet_files:
        symbol = file.stem
        # Map special symbols to their expected names
        mapped_symbol = symbol_map.get(symbol, symbol)
        try:
            df = pd.read_parquet(file)
            if not df.empty:
                data[mapped_symbol] = df
        except Exception as e:
            print(f"Error loading {symbol}: {e}")

    print(f"Loaded {len(data)} symbols from cache")

    # Verify critical symbols
    if "NIFTY 50" in data:
        print(
            f"  NIFTY 50: {data['NIFTY 50'].index.min().date()} to {data['NIFTY 50'].index.max().date()}"
        )
    if "INDIA VIX" in data:
        print(
            f"  INDIA VIX: {data['INDIA VIX'].index.min().date()} to {data['INDIA VIX'].index.max().date()}"
        )

    return data


def run_backtest(months: int = 12, use_dynamic: bool = True):
    """Run backtest with specified parameters."""

    print("=" * 70)
    print(
        f"BACKTEST: {months} months, Dynamic Rebalancing: {'ENABLED' if use_dynamic else 'DISABLED'}"
    )
    print("=" * 70)

    # Load config
    config = load_config("config.yaml")

    # Override dynamic rebalancing setting
    config.dynamic_rebalance.enabled = use_dynamic

    # Load universe
    universe = Universe("stock-universe.json")

    # Load cached data
    data = load_cached_data()

    if len(data) < 50:
        print("ERROR: Not enough cached data. Run fetch_and_backtest.py first.")
        return None

    # Calculate dates - use Feb 1, 2026 as end date (latest available data)
    end_date = datetime(2026, 2, 1)
    start_date = end_date - timedelta(days=months * 30)  # Dynamic based on months param

    print(f"\nDate Range: {start_date.date()} to {end_date.date()}")
    print(f"Strategy: {config.active_strategy}")
    print(f"Dynamic Rebalance: {config.dynamic_rebalance.enabled}")
    if config.dynamic_rebalance.enabled:
        print(f"  - Min days between: {config.dynamic_rebalance.min_days_between}")
        print(f"  - Max days between: {config.dynamic_rebalance.max_days_between}")
        print(
            f"  - Regime transition trigger: {config.dynamic_rebalance.regime_transition_trigger}"
        )
        print(f"  - VIX recovery trigger: {config.dynamic_rebalance.vix_recovery_trigger}")
        print(f"  - Crash avoidance trigger: {config.dynamic_rebalance.crash_avoidance_trigger}")
        print(f"  - Breadth thrust trigger: {config.dynamic_rebalance.breadth_thrust_trigger}")
    print(f"Adaptive Lookback: {config.adaptive_lookback.enabled}")

    # Create backtest config
    bt_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=2000000,
        rebalance_days=21,
        transaction_cost=0.003,
        target_positions=12,
        min_positions=10,
        max_positions=15,
        strategy_name=config.active_strategy,
        use_stop_loss=True,
    )

    # Run backtest
    print("\nRunning backtest...")
    try:
        engine = BacktestEngine(
            universe=universe,
            historical_data=data,
            config=bt_config,
            app_config=config,
            strategy_name=config.active_strategy,
        )
        result = engine.run()

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Initial Capital:  Rs {result.initial_capital:,.0f}")
        print(f"Final Value:      Rs {result.final_value:,.0f}")
        print(f"Total Profit:     Rs {result.total_profit:,.0f}")
        print(f"Total Return:     {result.total_return:.2%}")
        print(f"CAGR:             {result.cagr:.2%}")
        print(f"Max Drawdown:     {result.max_drawdown:.2%}")
        print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print(f"Win Rate:         {result.win_rate:.1%}")
        print(f"Total Trades:     {result.total_trades}")

        # Benchmark comparison
        nifty_return = next(
            (r for n, r in (result.benchmark_returns or []) if n.startswith("Nifty 50")), None
        )
        if nifty_return is not None:
            alpha = result.total_return - nifty_return
            print(f"\nNIFTY 50 Return:  {nifty_return:.2%}")
            print(f"Alpha:            {alpha:.2%}")

        # Regime analysis
        if result.regime_history is not None and len(result.regime_history) > 0:
            print("\nRegime Distribution:")
            regime_df = pd.DataFrame(result.regime_history)
            if "regime" in regime_df.columns:
                regime_counts = regime_df["regime"].value_counts()
                total_days = len(regime_df)
                for regime, count in regime_counts.items():
                    pct = count / total_days * 100
                    print(f"  {regime}: {count} days ({pct:.1f}%)")

        # Dynamic rebalancing analysis
        if hasattr(engine, "_rebalance_triggers_log") and engine._rebalance_triggers_log:
            print("\nDynamic Rebalance Triggers:")
            trigger_counts = {}
            for entry in engine._rebalance_triggers_log:
                for trigger in entry.get("triggers_fired", []):
                    trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
            for trigger, count in sorted(trigger_counts.items(), key=lambda x: -x[1]):
                print(f"  {trigger}: {count} times")
            print(f"  Total dynamic rebalances: {len(engine._rebalance_triggers_log)}")

        return result

    except Exception as e:
        import traceback

        print(f"\nERROR during backtest: {e}")
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    # Configure test period
    TEST_MONTHS = 48  # Change this to test different periods

    print("\n" + "=" * 70)
    print(f"DYNAMIC REBALANCING TEST - {TEST_MONTHS} MONTH BACKTEST")
    print("=" * 70 + "\n")

    # Run with dynamic rebalancing ENABLED
    result_dynamic = run_backtest(months=TEST_MONTHS, use_dynamic=True)

    print("\n" + "=" * 70)
    print("COMPARISON: Running with FIXED rebalancing for comparison...")
    print("=" * 70 + "\n")

    # Run with dynamic rebalancing DISABLED (fixed 21-day)
    result_fixed = run_backtest(months=TEST_MONTHS, use_dynamic=False)

    # Compare results
    if result_dynamic and result_fixed:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Metric':<25} {'Dynamic':>15} {'Fixed':>15} {'Diff':>15}")
        print("-" * 70)
        print(
            f"{'Total Return':<25} {result_dynamic.total_return:>14.2%} {result_fixed.total_return:>14.2%} {result_dynamic.total_return - result_fixed.total_return:>+14.2%}"
        )
        print(
            f"{'CAGR':<25} {result_dynamic.cagr:>14.2%} {result_fixed.cagr:>14.2%} {result_dynamic.cagr - result_fixed.cagr:>+14.2%}"
        )
        print(
            f"{'Max Drawdown':<25} {result_dynamic.max_drawdown:>14.2%} {result_fixed.max_drawdown:>14.2%} {result_dynamic.max_drawdown - result_fixed.max_drawdown:>+14.2%}"
        )
        print(
            f"{'Sharpe Ratio':<25} {result_dynamic.sharpe_ratio:>14.2f} {result_fixed.sharpe_ratio:>14.2f} {result_dynamic.sharpe_ratio - result_fixed.sharpe_ratio:>+14.2f}"
        )
        print(
            f"{'Total Trades':<25} {result_dynamic.total_trades:>15} {result_fixed.total_trades:>15} {result_dynamic.total_trades - result_fixed.total_trades:>+15}"
        )

        nifty_dyn = next(
            (r for n, r in (result_dynamic.benchmark_returns or []) if n.startswith("Nifty 50")),
            None,
        )
        nifty_fix = next(
            (r for n, r in (result_fixed.benchmark_returns or []) if n.startswith("Nifty 50")), None
        )
        if nifty_dyn and nifty_fix:
            alpha_dynamic = result_dynamic.total_return - nifty_dyn
            alpha_fixed = result_fixed.total_return - nifty_fix
            print(
                f"{'Alpha vs NIFTY':<25} {alpha_dynamic:>14.2%} {alpha_fixed:>14.2%} {alpha_dynamic - alpha_fixed:>+14.2%}"
            )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
