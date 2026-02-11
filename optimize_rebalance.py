#!/usr/bin/env python3
"""
Optimize rebalance period for FORTRESS MOMENTUM strategies.

Tests different rebalance intervals (in trading days) to find optimal frequency.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from fortress.backtest import BacktestConfig, BacktestEngine
from fortress.config import load_config
from fortress.universe import Universe


def load_cached_data(cache_dir: Path, universe: Universe) -> dict:
    """Load cached historical data including benchmark indices."""
    historical_data = {}

    if not cache_dir.exists():
        print(f"[ERROR] Cache directory not found: {cache_dir}")
        return historical_data

    all_stocks = universe.get_all_stocks()
    symbols = [s.zerodha_symbol for s in all_stocks]

    # Add benchmark indices and defensive instruments
    benchmark_symbols = ["NIFTY 50", "NIFTY MIDCAP 100", "INDIA VIX", "GOLDBEES", "LIQUIDCASE"]
    symbols.extend(benchmark_symbols)

    loaded = 0
    for symbol in symbols:
        cache_file = cache_dir / f"{symbol.replace(' ', '_')}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                historical_data[symbol] = df
                loaded += 1
            except Exception:
                pass

    print(f"Loaded {loaded} symbols from cache")
    return historical_data


def load_data(months: int = 36):
    """Load historical data for backtesting."""
    config = load_config("config.yaml")
    universe = Universe(config.paths.universe_file)

    # Load from cache
    cache_dir = Path(config.paths.data_cache)
    historical_data = load_cached_data(cache_dir, universe)

    if len(historical_data) < 50:
        print("[ERROR] Insufficient cached data. Run 'python fetch_and_backtest.py' first.")
        sys.exit(1)

    # Calculate date range
    yesterday = datetime.now() - timedelta(days=1)
    while yesterday.weekday() >= 5:
        yesterday -= timedelta(days=1)
    end_date = yesterday
    start_date = end_date - timedelta(days=months * 30)

    return universe, historical_data, start_date, end_date, config


def test_rebalance_period(
    universe: Universe,
    historical_data: dict,
    start_date: datetime,
    end_date: datetime,
    rebalance_days: int,
    strategy: str,
    config,
) -> dict:
    """Test a specific rebalance period."""

    bt_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=config.portfolio.initial_capital,
        rebalance_days=rebalance_days,
        transaction_cost=config.costs.transaction_cost,
        target_positions=config.position_sizing.target_positions,
        min_positions=config.position_sizing.min_positions,
        max_positions=config.position_sizing.max_positions,
        use_stop_loss=True,
        initial_stop_loss=config.risk.initial_stop_loss,
        trailing_stop=config.risk.trailing_stop,
        trailing_activation=config.risk.trailing_activation,
        strategy_name=strategy,
    )

    engine = BacktestEngine(
        universe=universe,
        historical_data=historical_data,
        config=bt_config,
        app_config=config,
    )

    result = engine.run()

    return {
        "rebalance_days": rebalance_days,
        "rebalance_weeks": round(rebalance_days / 5, 1),
        "strategy": strategy,
        "total_return": result.total_return,
        "cagr": result.cagr,
        "sharpe": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
    }


def main():
    # Test periods (in trading days)
    # 5 = weekly, 10 = bi-weekly, 15 = 3 weeks, 21 = monthly, 42 = bi-monthly, 63 = quarterly
    rebalance_periods = [5, 10, 15, 21, 30, 42, 63]
    strategies = ["classic", "allweather"]

    print("=" * 90)
    print("REBALANCE PERIOD OPTIMIZATION")
    print("=" * 90)
    print("\nNOTE: Days are TRADING DAYS (weekends/holidays excluded)")
    print("  5 trading days ≈ 1 week")
    print("  21 trading days ≈ 1 month")
    print("  63 trading days ≈ 3 months (quarterly)")
    print()

    # Load data once
    universe, historical_data, start_date, end_date, config = load_data(months=36)

    results = []

    for strategy in strategies:
        print(f"\n{'='*90}")
        print(f"Testing {strategy.upper()} strategy with different rebalance periods...")
        print(f"{'='*90}")

        for period in rebalance_periods:
            weeks = period / 5
            print(f"\n  Testing {period} trading days (~{weeks:.1f} weeks)...")

            try:
                result = test_rebalance_period(
                    universe, historical_data, start_date, end_date, period, strategy, config
                )
                results.append(result)

                print(f"    Return: {result['total_return']*100:+.1f}%  |  "
                      f"Sharpe: {result['sharpe']:.2f}  |  "
                      f"MaxDD: {result['max_drawdown']*100:.1f}%  |  "
                      f"Trades: {result['total_trades']}")
            except Exception as e:
                print(f"    ERROR: {e}")

    # Create summary DataFrame
    df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("SUMMARY: REBALANCE PERIOD COMPARISON")
    print("=" * 90)

    for strategy in strategies:
        print(f"\n{strategy.upper()} Strategy:")
        print("-" * 80)
        strategy_df = df[df["strategy"] == strategy].copy()
        strategy_df = strategy_df.sort_values("rebalance_days")

        print(f"{'Period':<12} {'Weeks':<8} {'Return':<12} {'CAGR':<10} {'Sharpe':<10} {'MaxDD':<10} {'Trades':<10} {'WinRate':<10}")
        print("-" * 80)

        for _, row in strategy_df.iterrows():
            print(f"{row['rebalance_days']:<12} {row['rebalance_weeks']:<8} "
                  f"{row['total_return']*100:+.1f}%      {row['cagr']*100:+.1f}%     "
                  f"{row['sharpe']:.2f}       {row['max_drawdown']*100:.1f}%      "
                  f"{row['total_trades']:<10} {row['win_rate']*100:.1f}%")

        # Find optimal
        best_sharpe = strategy_df.loc[strategy_df["sharpe"].idxmax()]
        best_return = strategy_df.loc[strategy_df["total_return"].idxmax()]

        print(f"\n  Best Sharpe Ratio: {best_sharpe['rebalance_days']} days (~{best_sharpe['rebalance_weeks']} weeks) - Sharpe: {best_sharpe['sharpe']:.2f}")
        print(f"  Best Total Return: {best_return['rebalance_days']} days (~{best_return['rebalance_weeks']} weeks) - Return: {best_return['total_return']*100:+.1f}%")

    # Cross-strategy comparison at each period
    print("\n" + "=" * 90)
    print("PERIOD-BY-PERIOD COMPARISON (Classic vs AllWeather)")
    print("=" * 90)

    for period in rebalance_periods:
        classic = df[(df["strategy"] == "classic") & (df["rebalance_days"] == period)]
        allweather = df[(df["strategy"] == "allweather") & (df["rebalance_days"] == period)]

        if len(classic) > 0 and len(allweather) > 0:
            c = classic.iloc[0]
            a = allweather.iloc[0]

            return_winner = "Classic" if c["total_return"] > a["total_return"] else "AllWeather"
            sharpe_winner = "Classic" if c["sharpe"] > a["sharpe"] else "AllWeather"

            print(f"\n{period} days (~{period/5:.1f} weeks):")
            print(f"  Classic:    Return {c['total_return']*100:+.1f}%  Sharpe {c['sharpe']:.2f}  Trades {c['total_trades']}")
            print(f"  AllWeather: Return {a['total_return']*100:+.1f}%  Sharpe {a['sharpe']:.2f}  Trades {a['total_trades']}")
            print(f"  Winner: {return_winner} (return) / {sharpe_winner} (sharpe)")

    print("\n" + "=" * 90)
    print("RECOMMENDATION")
    print("=" * 90)

    # Overall recommendation based on Sharpe ratio
    best_overall = df.loc[df["sharpe"].idxmax()]
    print(f"\nOptimal setup: {best_overall['strategy'].upper()} with {best_overall['rebalance_days']} trading days "
          f"(~{best_overall['rebalance_weeks']} weeks)")
    print(f"  Expected Sharpe: {best_overall['sharpe']:.2f}")
    print(f"  Expected Return: {best_overall['total_return']*100:+.1f}%")
    print(f"  Expected MaxDD:  {best_overall['max_drawdown']*100:.1f}%")
    print(f"  Expected Trades: {best_overall['total_trades']} over 3 years")


if __name__ == "__main__":
    main()
