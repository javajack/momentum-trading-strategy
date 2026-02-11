#!/usr/bin/env python3
"""Quick analysis script with immediate output."""

import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from fortress.config import load_config
from fortress.universe import Universe
from fortress.backtest import BacktestConfig, BacktestEngine

def load_data():
    data = {}
    symbol_map = {
        'NIFTY_50': 'NIFTY 50', 'INDIA_VIX': 'INDIA VIX',
        'NIFTY_MIDCAP_100': 'NIFTY MIDCAP 100',
    }
    for f in Path('.cache').glob('*.parquet'):
        symbol = f.stem
        mapped = symbol_map.get(symbol, symbol)
        df = pd.read_parquet(f)
        if not df.empty:
            data[mapped] = df
    return data

def run_test(data, universe, start, end, dynamic=True, min_days=5, vix_decline=0.15, label=""):
    config = load_config('config.yaml')
    config.dynamic_rebalance.enabled = dynamic
    config.dynamic_rebalance.min_days_between = min_days
    config.dynamic_rebalance.vix_recovery_decline = vix_decline

    bt_config = BacktestConfig(
        start_date=start, end_date=end,
        initial_capital=2000000, rebalance_days=21,
        strategy_name='simple', use_stop_loss=True,
    )

    engine = BacktestEngine(universe=universe, historical_data=data,
                           config=bt_config, app_config=config)
    result = engine.run()

    triggers = {}
    total_rebal = 0
    if hasattr(engine, '_rebalance_triggers_log'):
        total_rebal = len(engine._rebalance_triggers_log)
        for entry in engine._rebalance_triggers_log:
            for t in entry.get('triggers_fired', []):
                triggers[t] = triggers.get(t, 0) + 1

    return {
        'label': label,
        'return': result.total_return,
        'alpha': result.total_return - (result.nifty_50_return or 0),
        'cagr': result.cagr,
        'max_dd': result.max_drawdown,
        'sharpe': result.sharpe_ratio,
        'trades': result.total_trades,
        'rebalances': total_rebal,
        'triggers': triggers,
    }

print("Loading data...")
data = load_data()
universe = Universe('stock-universe.json')
print(f"Loaded {len(data)} symbols\n")

# ============================================================================
print("=" * 80)
print("ANALYSIS 1: VIX_RECOVERY THRESHOLD (48 months: Feb 2022 - Feb 2026)")
print("=" * 80)

start, end = datetime(2022, 2, 1), datetime(2026, 2, 1)
print(f"\n{'Threshold':<12} {'Return':>10} {'Alpha':>10} {'MaxDD':>10} {'VIX Trig':>10} {'Trades':>8}")
print("-" * 60)

for thresh in [0.10, 0.15, 0.20, 0.25, 0.30]:
    r = run_test(data, universe, start, end, dynamic=True, vix_decline=thresh)
    vix = r['triggers'].get('VIX_RECOVERY', 0)
    print(f"{thresh:.0%:<12} {r['return']:>9.1%} {r['alpha']:>9.1%} {r['max_dd']:>9.1%} {vix:>10} {r['trades']:>8}")

# Test VIX disabled
r_no_vix = run_test(data, universe, start, end, dynamic=True, vix_decline=0.15)
# Manually disable VIX for comparison
config = load_config('config.yaml')
config.dynamic_rebalance.enabled = True
config.dynamic_rebalance.vix_recovery_trigger = False
bt_config = BacktestConfig(start_date=start, end_date=end, initial_capital=2000000,
                          rebalance_days=21, strategy_name='simple', use_stop_loss=True)
engine = BacktestEngine(universe=universe, historical_data=data, config=bt_config, app_config=config)
result = engine.run()
rebal = len(engine._rebalance_triggers_log) if hasattr(engine, '_rebalance_triggers_log') else 0
print(f"\nVIX DISABLED: Return={result.total_return:.1%}, Alpha={result.total_return - (result.nifty_50_return or 0):.1%}, Rebalances={rebal}, Trades={result.total_trades}")

# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 2: MIN_DAYS_BETWEEN TUNING (48 months)")
print("=" * 80)

print(f"\n{'Min Days':<10} {'Return':>10} {'Alpha':>10} {'MaxDD':>10} {'Sharpe':>8} {'Rebal':>8} {'Trades':>8}")
print("-" * 64)

for min_d in [5, 7, 10, 14, 21]:
    r = run_test(data, universe, start, end, dynamic=True, min_days=min_d)
    print(f"{min_d:<10} {r['return']:>9.1%} {r['alpha']:>9.1%} {r['max_dd']:>9.1%} {r['sharpe']:>7.2f} {r['rebalances']:>8} {r['trades']:>8}")

# Fixed baseline
r_fixed = run_test(data, universe, start, end, dynamic=False)
print(f"{'Fixed':<10} {r_fixed['return']:>9.1%} {r_fixed['alpha']:>9.1%} {r_fixed['max_dd']:>9.1%} {r_fixed['sharpe']:>7.2f} {r_fixed['rebalances']:>8} {r_fixed['trades']:>8}")

# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 3: MARKET PHASES (with optimized min_days=10)")
print("=" * 80)

phases = [
    ("COVID Crash", datetime(2020, 1, 1), datetime(2020, 4, 30)),
    ("V-Recovery", datetime(2020, 4, 1), datetime(2020, 12, 31)),
    ("2021 Bull", datetime(2021, 1, 1), datetime(2021, 10, 31)),
    ("2022 Correction", datetime(2022, 1, 1), datetime(2022, 6, 30)),
    ("2023-24 Rally", datetime(2023, 4, 1), datetime(2024, 6, 30)),
    ("Recent 2024-25", datetime(2024, 7, 1), datetime(2026, 2, 1)),
]

print(f"\n{'Phase':<18} {'Dyn Ret':>10} {'Fix Ret':>10} {'Diff':>10} {'Dyn DD':>10} {'Fix DD':>10} {'Winner':>10}")
print("-" * 78)

dyn_wins, fix_wins = 0, 0
for name, s, e in phases:
    r_dyn = run_test(data, universe, s, e, dynamic=True, min_days=10)
    r_fix = run_test(data, universe, s, e, dynamic=False)
    diff = r_dyn['return'] - r_fix['return']
    winner = "Dynamic" if diff > 0.01 else ("Fixed" if diff < -0.01 else "Tie")
    if winner == "Dynamic": dyn_wins += 1
    elif winner == "Fixed": fix_wins += 1
    print(f"{name:<18} {r_dyn['return']:>9.1%} {r_fix['return']:>9.1%} {diff:>+9.1%} {r_dyn['max_dd']:>9.1%} {r_fix['max_dd']:>9.1%} {winner:>10}")

print(f"\nOverall: Dynamic wins {dyn_wins}, Fixed wins {fix_wins}")

# ============================================================================
print("\n" + "=" * 80)
print("FINAL: OPTIMIZED DYNAMIC vs FIXED (min_days=10, vix_decline=20%)")
print("=" * 80)

periods = [
    ("12M", datetime(2025, 2, 1), datetime(2026, 2, 1)),
    ("24M", datetime(2024, 2, 1), datetime(2026, 2, 1)),
    ("48M", datetime(2022, 2, 1), datetime(2026, 2, 1)),
]

print(f"\n{'Period':<8} {'Opt Dyn':>10} {'Fixed':>10} {'Diff':>10} {'Opt DD':>10} {'Fix DD':>10} {'DD Diff':>10}")
print("-" * 68)

for name, s, e in periods:
    config = load_config('config.yaml')
    config.dynamic_rebalance.enabled = True
    config.dynamic_rebalance.min_days_between = 10
    config.dynamic_rebalance.vix_recovery_decline = 0.20

    bt_config = BacktestConfig(start_date=s, end_date=e, initial_capital=2000000,
                              rebalance_days=21, strategy_name='simple', use_stop_loss=True)
    engine = BacktestEngine(universe=universe, historical_data=data, config=bt_config, app_config=config)
    r_opt = engine.run()

    r_fix = run_test(data, universe, s, e, dynamic=False)

    diff_ret = r_opt.total_return - r_fix['return']
    diff_dd = r_opt.max_drawdown - r_fix['max_dd']
    print(f"{name:<8} {r_opt.total_return:>9.1%} {r_fix['return']:>9.1%} {diff_ret:>+9.1%} {r_opt.max_drawdown:>9.1%} {r_fix['max_dd']:>9.1%} {diff_dd:>+9.1%}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
