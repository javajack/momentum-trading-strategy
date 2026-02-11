#!/usr/bin/env python3
"""
Test if dynamic rebalancing bridges the gap between optimal frequencies.
Compares:
1. Fixed Monthly (21 days) - best for 2016-2020
2. Fixed Bi-weekly (10 days) - best for 2020-2024
3. Dynamic Rebalancing - auto-adjusts based on regime
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


def run_backtest(
    universe: Universe,
    historical_data: dict,
    config,
    start_date: datetime,
    end_date: datetime,
    rebalance_days: int,
    dynamic_enabled: bool,
) -> BacktestResult:
    """Run backtest with specific settings."""

    # Create a modified config
    modified_config = deepcopy(config)
    modified_config.dynamic_rebalance.enabled = dynamic_enabled

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
        app_config=modified_config,
    )

    return engine.run()


def main():
    console.print(Panel("[bold cyan]DYNAMIC REBALANCING TEST[/bold cyan]"))
    console.print("[dim]Does dynamic rebalancing adapt to different market conditions?[/dim]\n")

    # Load config and data
    config = load_config()
    universe = Universe(config.paths.universe_file)
    cache_dir = Path(config.paths.data_cache)

    # Show current dynamic rebalance settings
    console.print("[bold]Current Dynamic Rebalance Config:[/bold]")
    dr = config.dynamic_rebalance
    console.print(f"  enabled: {dr.enabled}")
    console.print(f"  min_days_between: {dr.min_days_between}")
    console.print(f"  max_days_between: {dr.max_days_between}")
    console.print(f"  regime_transition_trigger: {dr.regime_transition_trigger}")
    console.print(f"  vix_recovery_trigger: {dr.vix_recovery_trigger}")
    console.print(f"  drawdown_trigger: {dr.drawdown_trigger}")
    console.print(f"  drawdown_threshold: {dr.drawdown_threshold}")
    console.print()

    console.print("Loading cached data...")
    historical_data = load_cached_data(cache_dir)
    console.print(f"[green]Loaded {len(historical_data)} symbols[/green]\n")

    # Define periods
    periods = [
        ("2016-2020", datetime(2016, 1, 1), datetime(2020, 1, 1)),
        ("2020-2024", datetime(2020, 1, 1), datetime(2024, 1, 1)),
    ]

    # Test configurations
    test_configs = [
        (21, False, "Fixed Monthly (21d)"),
        (10, False, "Fixed Bi-weekly (10d)"),
        (5, False, "Fixed Weekly (5d)"),
        (21, True, "Dynamic (base 21d)"),
        (10, True, "Dynamic (base 10d)"),
    ]

    results = []

    for period_name, start_date, end_date in periods:
        console.print(f"\n[bold cyan]{'═' * 50}[/bold cyan]")
        console.print(f"[bold cyan]  {period_name}[/bold cyan]")
        console.print(f"[bold cyan]{'═' * 50}[/bold cyan]")

        for rebal_days, dynamic, config_name in test_configs:
            console.print(f"  Testing {config_name}...", end=" ")

            try:
                result = run_backtest(
                    universe, historical_data, config,
                    start_date, end_date, rebal_days, dynamic
                )

                # Count rebalance events from regime history if available
                rebal_count = "N/A"
                if result.regime_history is not None and len(result.regime_history) > 0:
                    rebal_count = len(result.regime_history)

                console.print(f"[green]{result.total_return:+.1%} | Sharpe: {result.sharpe_ratio:.2f} | Trades: {result.total_trades}[/green]")

                results.append({
                    "period": period_name,
                    "config": config_name,
                    "rebal_days": rebal_days,
                    "dynamic": dynamic,
                    "total_return": result.total_return,
                    "cagr": result.cagr,
                    "sharpe": result.sharpe_ratio,
                    "max_dd": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "trades": result.total_trades,
                    "final_value": result.final_value,
                    "rebal_events": rebal_count,
                })
            except Exception as e:
                console.print(f"[red]Failed: {e}[/red]")

    # Results tables
    console.print("\n")
    console.print(Panel("[bold]RESULTS COMPARISON[/bold]"))

    for period in ["2016-2020", "2020-2024"]:
        period_results = [r for r in results if r["period"] == period]

        table = Table(title=f"{period} Performance Comparison")
        table.add_column("Configuration", style="cyan")
        table.add_column("Return", justify="right")
        table.add_column("CAGR", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Win %", justify="right")
        table.add_column("Trades", justify="right")

        # Find best sharpe for highlighting
        best_sharpe = max(r["sharpe"] for r in period_results) if period_results else 0
        best_return = max(r["total_return"] for r in period_results) if period_results else 0

        for r in period_results:
            ret_style = "bold green" if r["total_return"] == best_return else ("green" if r["total_return"] > 0 else "red")
            sharpe_style = "bold yellow" if r["sharpe"] == best_sharpe else ""

            # Mark dynamic configs
            config_name = r["config"]
            if r["dynamic"]:
                config_name = f"[magenta]{config_name}[/magenta]"

            table.add_row(
                config_name,
                f"[{ret_style}]{r['total_return']:+.1%}[/{ret_style}]",
                f"{r['cagr']:.1%}",
                f"[{sharpe_style}]{r['sharpe']:.2f}[/{sharpe_style}]",
                f"{r['max_dd']:.1%}",
                f"{r['win_rate']:.0%}",
                str(r["trades"]),
            )

        console.print(table)
        console.print()

    # Analysis
    console.print(Panel("[bold]ANALYSIS: Does Dynamic Rebalancing Help?[/bold]"))

    for period in ["2016-2020", "2020-2024"]:
        period_results = [r for r in results if r["period"] == period]

        fixed_monthly = next((r for r in period_results if r["config"] == "Fixed Monthly (21d)"), None)
        fixed_biweekly = next((r for r in period_results if r["config"] == "Fixed Bi-weekly (10d)"), None)
        fixed_weekly = next((r for r in period_results if r["config"] == "Fixed Weekly (5d)"), None)
        dynamic_21 = next((r for r in period_results if r["config"] == "Dynamic (base 21d)"), None)
        dynamic_10 = next((r for r in period_results if r["config"] == "Dynamic (base 10d)"), None)

        console.print(f"\n[bold cyan]{period}:[/bold cyan]")

        # Find best fixed
        fixed_results = [r for r in period_results if not r["dynamic"]]
        best_fixed = max(fixed_results, key=lambda x: x["sharpe"]) if fixed_results else None

        # Find best dynamic
        dynamic_results = [r for r in period_results if r["dynamic"]]
        best_dynamic = max(dynamic_results, key=lambda x: x["sharpe"]) if dynamic_results else None

        if best_fixed and best_dynamic:
            console.print(f"  Best Fixed:   {best_fixed['config']} → {best_fixed['total_return']:+.1%} return, {best_fixed['sharpe']:.2f} Sharpe")
            console.print(f"  Best Dynamic: {best_dynamic['config']} → {best_dynamic['total_return']:+.1%} return, {best_dynamic['sharpe']:.2f} Sharpe")

            if best_dynamic["sharpe"] > best_fixed["sharpe"]:
                improvement = (best_dynamic["sharpe"] - best_fixed["sharpe"]) / best_fixed["sharpe"] * 100
                console.print(f"  [green]✓ Dynamic improved Sharpe by {improvement:.1f}%[/green]")
            elif best_dynamic["sharpe"] < best_fixed["sharpe"]:
                decline = (best_fixed["sharpe"] - best_dynamic["sharpe"]) / best_fixed["sharpe"] * 100
                console.print(f"  [yellow]✗ Dynamic underperformed by {decline:.1f}% Sharpe[/yellow]")
            else:
                console.print(f"  [dim]≈ Dynamic matched fixed performance[/dim]")

            # Return comparison
            if best_dynamic["total_return"] > best_fixed["total_return"]:
                ret_diff = best_dynamic["total_return"] - best_fixed["total_return"]
                console.print(f"  [green]✓ Dynamic captured {ret_diff:.1%} more return[/green]")
            else:
                ret_diff = best_fixed["total_return"] - best_dynamic["total_return"]
                console.print(f"  [yellow]✗ Dynamic missed {ret_diff:.1%} return vs best fixed[/yellow]")

    # Final verdict
    console.print("\n" + "─" * 60)
    console.print("[bold]VERDICT:[/bold]")

    # Calculate overall scores
    p1_dynamic = [r for r in results if r["period"] == "2016-2020" and r["dynamic"]]
    p1_fixed = [r for r in results if r["period"] == "2016-2020" and not r["dynamic"]]
    p2_dynamic = [r for r in results if r["period"] == "2020-2024" and r["dynamic"]]
    p2_fixed = [r for r in results if r["period"] == "2020-2024" and not r["dynamic"]]

    best_p1_fixed = max(p1_fixed, key=lambda x: x["sharpe"]) if p1_fixed else None
    best_p1_dynamic = max(p1_dynamic, key=lambda x: x["sharpe"]) if p1_dynamic else None
    best_p2_fixed = max(p2_fixed, key=lambda x: x["sharpe"]) if p2_fixed else None
    best_p2_dynamic = max(p2_dynamic, key=lambda x: x["sharpe"]) if p2_dynamic else None

    console.print("""
Dynamic rebalancing triggers on:
  • Regime transitions (normal → caution → defensive)
  • VIX recovery (fear subsiding after spike)
  • Portfolio drawdown exceeding threshold

The question is: Does this adaptive approach outperform a fixed schedule?
""")

    if best_p1_dynamic and best_p1_fixed and best_p2_dynamic and best_p2_fixed:
        # Check if dynamic is consistently good (or at least not much worse)
        p1_ok = best_p1_dynamic["sharpe"] >= best_p1_fixed["sharpe"] * 0.9
        p2_ok = best_p2_dynamic["sharpe"] >= best_p2_fixed["sharpe"] * 0.9

        if p1_ok and p2_ok:
            console.print("[green]✓ Dynamic rebalancing performs well across BOTH market conditions![/green]")
            console.print("  It adapts reasonably to both low-momentum (2016-2020) and high-momentum (2020-2024) periods.")
        elif p1_ok:
            console.print("[yellow]⚠ Dynamic works well for low-momentum periods but underperforms in strong trends.[/yellow]")
            console.print("  Consider using faster fixed rebalancing (bi-weekly) in confirmed bull markets.")
        elif p2_ok:
            console.print("[yellow]⚠ Dynamic works well for trending markets but underperforms in choppy conditions.[/yellow]")
            console.print("  The regime detection may need tuning for sideways markets.")
        else:
            console.print("[red]✗ Dynamic rebalancing underperforms in both conditions.[/red]")
            console.print("  Fixed frequency with manual regime assessment may be better.")


if __name__ == "__main__":
    main()
