#!/usr/bin/env python3
"""
Test different rebalancing frequencies with dynamic rebalancing DISABLED.
Compares 5-day (weekly) vs 21-day (monthly) rebalancing.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent))

from fortress.config import load_config
from fortress.universe import Universe
from fortress.backtest import BacktestConfig, BacktestEngine, BacktestResult

console = Console()


def load_cached_data(cache_dir: Path) -> dict:
    """Load all cached historical data."""
    historical_data = {}
    cache_meta_file = cache_dir / "cache_meta.json"

    if not cache_meta_file.exists():
        return {}

    cache_meta = json.loads(cache_meta_file.read_text())

    for symbol in cache_meta.keys():
        cache_file = cache_dir / f"{symbol.replace(' ', '_')}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                historical_data[symbol] = df
            except:
                pass

    return historical_data


def run_backtest_with_rebalance(
    universe: Universe,
    historical_data: dict,
    config,
    start_date: datetime,
    end_date: datetime,
    rebalance_days: int,
) -> BacktestResult:
    """Run backtest with specific rebalance frequency and dynamic rebalancing DISABLED."""

    # Create a modified config with dynamic_rebalance disabled
    modified_config = deepcopy(config)
    modified_config.dynamic_rebalance.enabled = False  # DISABLE dynamic rebalancing

    bt_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=config.portfolio.initial_capital,
        rebalance_days=rebalance_days,
        transaction_cost=config.costs.transaction_cost,
        target_positions=config.position_sizing.target_positions,
        min_positions=config.position_sizing.min_positions,
        use_stop_loss=True,
        initial_stop_loss=config.risk.initial_stop_loss,
        trailing_stop=config.risk.trailing_stop,
        trailing_activation=config.risk.trailing_activation,
        min_score_percentile=config.pure_momentum.min_score_percentile,
        min_52w_high_prox=config.pure_momentum.min_52w_high_prox,
        strategy_name="dual_momentum",
    )

    engine = BacktestEngine(
        universe=universe,
        historical_data=historical_data,
        config=bt_config,
        app_config=modified_config,  # Use modified config with dynamic rebalancing disabled
    )

    return engine.run()


def main():
    console.print(Panel("[bold cyan]REBALANCING FREQUENCY COMPARISON[/bold cyan]"))
    console.print("[dim]Testing 5-day (weekly) vs 21-day (monthly) with dynamic rebalancing DISABLED[/dim]\n")

    # Load config and data
    config = load_config()
    universe = Universe(config.paths.universe_file)
    cache_dir = Path(config.paths.data_cache)

    console.print("Loading cached data...")
    historical_data = load_cached_data(cache_dir)
    console.print(f"[green]Loaded {len(historical_data)} symbols[/green]\n")

    # Define periods
    periods = [
        ("2016-2020", datetime(2016, 1, 1), datetime(2020, 1, 1)),
        ("2020-2024", datetime(2020, 1, 1), datetime(2024, 1, 1)),
    ]

    # Rebalancing frequencies to test
    frequencies = [
        (5, "Weekly (5 days)"),
        (10, "Bi-weekly (10 days)"),
        (21, "Monthly (21 days)"),
    ]

    results = []

    for period_name, start_date, end_date in periods:
        console.print(f"\n[bold cyan]═══ {period_name} ═══[/bold cyan]")

        for rebal_days, rebal_name in frequencies:
            console.print(f"  Running {rebal_name}...", end=" ")

            try:
                result = run_backtest_with_rebalance(
                    universe, historical_data, config,
                    start_date, end_date, rebal_days
                )
                console.print(f"[green]{result.total_return:+.1%} return, {result.sharpe_ratio:.2f} Sharpe[/green]")

                results.append({
                    "period": period_name,
                    "rebalance": rebal_name,
                    "rebal_days": rebal_days,
                    "total_return": result.total_return,
                    "cagr": result.cagr,
                    "sharpe": result.sharpe_ratio,
                    "max_dd": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "trades": result.total_trades,
                    "final_value": result.final_value,
                })
            except Exception as e:
                console.print(f"[red]Failed: {e}[/red]")

    # Results table
    console.print("\n")
    console.print(Panel("[bold]RESULTS COMPARISON[/bold]"))

    # Period 1 table
    table1 = Table(title="2016-2020 Performance by Rebalancing Frequency")
    table1.add_column("Rebalance", style="cyan")
    table1.add_column("Return", justify="right")
    table1.add_column("CAGR", justify="right")
    table1.add_column("Sharpe", justify="right")
    table1.add_column("Max DD", justify="right")
    table1.add_column("Win Rate", justify="right")
    table1.add_column("Trades", justify="right")

    for r in results:
        if r["period"] == "2016-2020":
            ret_color = "green" if r["total_return"] > 0 else "red"
            table1.add_row(
                r["rebalance"],
                f"[{ret_color}]{r['total_return']:+.1%}[/{ret_color}]",
                f"{r['cagr']:.1%}",
                f"{r['sharpe']:.2f}",
                f"{r['max_dd']:.1%}",
                f"{r['win_rate']:.0%}",
                str(r["trades"]),
            )

    console.print(table1)

    # Period 2 table
    table2 = Table(title="2020-2024 Performance by Rebalancing Frequency")
    table2.add_column("Rebalance", style="cyan")
    table2.add_column("Return", justify="right")
    table2.add_column("CAGR", justify="right")
    table2.add_column("Sharpe", justify="right")
    table2.add_column("Max DD", justify="right")
    table2.add_column("Win Rate", justify="right")
    table2.add_column("Trades", justify="right")

    for r in results:
        if r["period"] == "2020-2024":
            ret_color = "green" if r["total_return"] > 0 else "red"
            table2.add_row(
                r["rebalance"],
                f"[{ret_color}]{r['total_return']:+.1%}[/{ret_color}]",
                f"{r['cagr']:.1%}",
                f"{r['sharpe']:.2f}",
                f"{r['max_dd']:.1%}",
                f"{r['win_rate']:.0%}",
                str(r["trades"]),
            )

    console.print(table2)

    # Summary comparison
    console.print("\n[bold]KEY INSIGHTS:[/bold]")

    # Find best for each period
    p1_results = [r for r in results if r["period"] == "2016-2020"]
    p2_results = [r for r in results if r["period"] == "2020-2024"]

    if p1_results:
        best_p1 = max(p1_results, key=lambda x: x["sharpe"])
        console.print(f"\n[cyan]2016-2020 Best Sharpe:[/cyan] {best_p1['rebalance']} ({best_p1['sharpe']:.2f})")

    if p2_results:
        best_p2 = max(p2_results, key=lambda x: x["sharpe"])
        console.print(f"[cyan]2020-2024 Best Sharpe:[/cyan] {best_p2['rebalance']} ({best_p2['sharpe']:.2f})")

    # Transaction cost impact
    console.print("\n[bold]TRANSACTION COST IMPACT:[/bold]")
    for period in ["2016-2020", "2020-2024"]:
        period_results = [r for r in results if r["period"] == period]
        if len(period_results) >= 2:
            weekly = next((r for r in period_results if r["rebal_days"] == 5), None)
            monthly = next((r for r in period_results if r["rebal_days"] == 21), None)
            if weekly and monthly:
                trade_diff = weekly["trades"] - monthly["trades"]
                cost_diff = trade_diff * 0.003 * config.portfolio.initial_capital / monthly["trades"]  # Approx extra cost
                console.print(f"  {period}: Weekly has {trade_diff:+d} more trades than monthly")


if __name__ == "__main__":
    main()
