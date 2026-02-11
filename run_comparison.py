#!/usr/bin/env python3
"""Run backtest comparison: 10-day vs 21-day rebalance, 5-year range."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from fortress.backtest import BacktestConfig, BacktestEngine
from fortress.config import load_config
from fortress.universe import Universe


def load_cached_data(cache_dir: Path) -> dict:
    """Load all cached historical data from .cache/"""
    historical_data = {}
    cache_meta_file = cache_dir / "cache_meta.json"

    if not cache_meta_file.exists():
        print(f"ERROR: {cache_meta_file} not found")
        return {}

    cache_meta = json.loads(cache_meta_file.read_text())

    for symbol in cache_meta.keys():
        safe_name = symbol.replace(" ", "_").replace("&", "_")
        cache_file = cache_dir / f"{safe_name}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            historical_data[symbol] = df

    return historical_data


def run_backtest(universe, historical_data, config, start_date, end_date, rebalance_days, dynamic=True):
    """Run a single backtest and return the result."""
    from copy import deepcopy
    config = deepcopy(config)
    config.dynamic_rebalance.enabled = dynamic
    bt_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=config.portfolio.initial_capital,
        rebalance_days=rebalance_days,
        transaction_cost=config.costs.transaction_cost,
        target_positions=config.position_sizing.target_positions,
        min_score_percentile=config.pure_momentum.min_score_percentile,
        min_52w_high_prox=config.pure_momentum.min_52w_high_prox,
        initial_stop_loss=config.risk.initial_stop_loss,
        trailing_stop=config.risk.trailing_stop,
        weight_6m=config.pure_momentum.weight_6m,
        weight_12m=config.pure_momentum.weight_12m,
        strategy_name="dual_momentum",
    )

    engine = BacktestEngine(
        universe=universe,
        historical_data=historical_data,
        config=bt_config,
        app_config=config,
        strategy_name="dual_momentum",
    )

    return engine.run()


def display_result(label, result):
    """Display backtest results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total Return:     {result.total_return:>10.2%}")
    print(f"  CAGR:             {result.cagr:>10.2%}")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown:>10.2%}")
    print(f"  Win Rate:         {result.win_rate:>10.2%}")
    print(f"  Total Trades:     {result.total_trades:>10d}")
    print(f"  Initial Capital:  {result.initial_capital:>10,.0f}")
    print(f"  Final Value:      {result.final_value:>10,.0f}")
    print(f"  Peak Value:       {result.peak_value:>10,.0f}")
    print(f"  Total Profit:     {result.total_profit:>10,.0f}")
    if result.nifty_50_return is not None:
        print(f"  Nifty 50 Return:  {result.nifty_50_return:>10.2%}")
    if result.nifty_midcap_100_return is not None:
        print(f"  Midcap 100 Return:{result.nifty_midcap_100_return:>10.2%}")
    print(f"  Regime Changes:   {result.regime_transitions:>10d}")
    if result.time_in_regime:
        print(f"  Time in Regimes:")
        for regime, pct in result.time_in_regime.items():
            print(f"    {regime:18s} {pct:>6.1%}")
    print(f"{'='*60}")


def main():
    print("\n" + "=" * 60)
    print("  FORTRESS MOMENTUM - Backtest Comparison")
    print("  Strategy: dual_momentum | Range: 5 years")
    print("=" * 60)

    # Load config and universe
    config = load_config("config.yaml")
    universe = Universe("stock-universe.json")

    # Load cached data
    cache_dir = Path(".cache")
    print(f"\nLoading cached data from {cache_dir}...")
    historical_data = load_cached_data(cache_dir)
    print(f"Loaded {len(historical_data)} symbols")

    if len(historical_data) < 50:
        print("ERROR: Insufficient cached data. Run menu option 2 first to populate cache.")
        sys.exit(1)

    # Date range: last 5 years
    yesterday = datetime.now() - timedelta(days=1)
    while yesterday.weekday() >= 5:
        yesterday -= timedelta(days=1)
    end_date = yesterday
    start_date = end_date - relativedelta(months=60)

    print(f"\nBacktest period: {start_date.date()} to {end_date.date()}")
    print(f"Initial capital:  {config.portfolio.initial_capital:,.0f}")

    # Run dynamic (default)
    print("\n>>> Running DYNAMIC rebalance (default mode)...")
    result_dyn = run_backtest(universe, historical_data, config, start_date, end_date, 21, dynamic=True)
    display_result("Dynamic Rebalance (Default)", result_dyn)

    # Run 21-day fixed rebalance
    print("\n>>> Running 21-day FIXED rebalance...")
    result_21 = run_backtest(universe, historical_data, config, start_date, end_date, 21, dynamic=False)
    display_result("21-Day Fixed Rebalance (Monthly)", result_21)

    # Run 10-day fixed rebalance
    print("\n>>> Running 10-day FIXED rebalance...")
    result_10 = run_backtest(universe, historical_data, config, start_date, end_date, 10, dynamic=False)
    display_result("10-Day Fixed Rebalance (Bi-Weekly)", result_10)

    # Side-by-side comparison
    print(f"\n{'='*72}")
    print(f"  COMPARISON: Dynamic vs Fixed-21 vs Fixed-10")
    print(f"{'='*72}")
    print(f"  {'Metric':<22s} {'Dynamic':>12s} {'Fixed-21':>12s} {'Fixed-10':>12s}")
    print(f"  {'-'*70}")

    metrics = [
        ("Total Return", result_dyn.total_return, result_21.total_return, result_10.total_return, True),
        ("CAGR", result_dyn.cagr, result_21.cagr, result_10.cagr, True),
        ("Sharpe Ratio", result_dyn.sharpe_ratio, result_21.sharpe_ratio, result_10.sharpe_ratio, False),
        ("Max Drawdown", result_dyn.max_drawdown, result_21.max_drawdown, result_10.max_drawdown, True),
        ("Win Rate", result_dyn.win_rate, result_21.win_rate, result_10.win_rate, True),
        ("Total Trades", result_dyn.total_trades, result_21.total_trades, result_10.total_trades, False),
        ("Final Value", result_dyn.final_value, result_21.final_value, result_10.final_value, False),
    ]

    for name, vd, v21, v10, is_pct in metrics:
        if is_pct:
            print(f"  {name:<22s} {vd:>11.2%} {v21:>11.2%} {v10:>11.2%}")
        elif isinstance(vd, int):
            print(f"  {name:<22s} {vd:>11d} {v21:>11d} {v10:>11d}")
        else:
            print(f"  {name:<22s} {vd:>11.2f} {v21:>11.2f} {v10:>11.2f}")

    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
