#!/usr/bin/env python3
"""
FORTRESS MOMENTUM Optimizer

Comprehensive grid search optimization for pure momentum strategy parameters.
Targets all-weather performance across bullish, sideways, and bearish markets.

Usage:
    uv run python optimize_momentum.py [--quick] [--full] [--regime]
"""

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from fortress.backtest import BacktestConfig, BacktestEngine, BacktestResult
from fortress.config import Config, load_config
from fortress.universe import Universe

console = Console()


# =============================================================================
# PARAMETER GRIDS
# =============================================================================

# Quick grid for fast testing (~50 combinations)
QUICK_PARAM_GRID = {
    "rebalance_days": [15, 21],
    "weight_6m": [0.50, 0.60],
    "min_score_percentile": [93, 95],
    "min_52w_high_prox": [0.85],
    "target_positions": [15],
    "initial_stop_loss": [0.15],
    "trailing_stop": [0.12],
}

# Standard grid for thorough optimization (~108 combinations)
# Fixed monthly rebalance (21 days) - proven optimal
STANDARD_PARAM_GRID = {
    "rebalance_days": [21],  # Monthly only - proven optimal
    "weight_6m": [0.40, 0.50, 0.60],
    "min_score_percentile": [90, 93, 95],
    "min_52w_high_prox": [0.80, 0.85, 0.90],
    "target_positions": [12, 15, 18],
    "initial_stop_loss": [0.12, 0.15],
    "trailing_stop": [0.10, 0.12],
}

# Full grid for exhaustive search (~2000+ combinations)
FULL_PARAM_GRID = {
    "rebalance_days": [10, 15, 21, 30, 42],
    "weight_6m": [0.30, 0.40, 0.50, 0.60, 0.70],
    "min_score_percentile": [88, 90, 93, 95, 97],
    "min_52w_high_prox": [0.75, 0.80, 0.85, 0.90],
    "target_positions": [10, 12, 15, 18, 20],
    "initial_stop_loss": [0.10, 0.12, 0.15, 0.18, 0.20],
    "trailing_stop": [0.08, 0.10, 0.12, 0.15],
}

# Regime threshold grid (for hybrid mode optimization)
REGIME_PARAM_GRID = {
    "vix_caution": [18, 20, 22],
    "vix_defensive": [23, 25, 28],
    "nifty_caution_1m": [-0.02, -0.03, -0.04],
    "nifty_defensive_3m": [-0.08, -0.10, -0.12],
}


@dataclass
class OptimizationResult:
    """Result of a single parameter combination test."""

    params: Dict[str, Any]
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int

    # Period-specific metrics (for robustness check)
    bear_period_return: Optional[float] = None
    sideways_period_return: Optional[float] = None
    bull_period_return: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        result = {}
        for key, value in self.params.items():
            result[f"param_{key}"] = value
        result["total_return"] = self.total_return
        result["cagr"] = self.cagr
        result["sharpe_ratio"] = self.sharpe_ratio
        result["max_drawdown"] = self.max_drawdown
        result["win_rate"] = self.win_rate
        result["total_trades"] = self.total_trades
        if self.bear_period_return is not None:
            result["bear_period_return"] = self.bear_period_return
        if self.sideways_period_return is not None:
            result["sideways_period_return"] = self.sideways_period_return
        if self.bull_period_return is not None:
            result["bull_period_return"] = self.bull_period_return
        return result


class MomentumOptimizer:
    """
    Grid search optimizer for pure momentum strategy parameters.

    Optimizes for all-weather performance across different market conditions.
    """

    def __init__(
        self,
        universe: Universe,
        historical_data: Dict[str, pd.DataFrame],
        base_config: Config,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 1600000,
    ):
        self.universe = universe
        self.historical_data = historical_data
        self.base_config = base_config
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        # Market period definitions for robustness testing
        self.market_periods = {
            "bear_2022": (datetime(2022, 2, 1), datetime(2022, 6, 30)),
            "sideways_2023": (datetime(2023, 1, 1), datetime(2023, 3, 31)),
            "bull_2023": (datetime(2023, 4, 1), datetime(2023, 12, 31)),
        }

    def generate_combinations(
        self,
        param_grid: Dict[str, List],
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def run_single_backtest(
        self,
        params: Dict[str, Any],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BacktestResult:
        """Run a single backtest with given parameters."""
        start = start_date or self.start_date
        end = end_date or self.end_date

        # Calculate weight_12m from weight_6m (must sum to 1.0)
        weight_6m = params.get("weight_6m", 0.50)
        weight_12m = 1.0 - weight_6m

        # Determine rebalance_days
        rebalance_days = params.get("rebalance_days", 21)

        # Create backtest config with parameter overrides
        bt_config = BacktestConfig(
            start_date=start,
            end_date=end,
            initial_capital=self.initial_capital,
            rebalance_days=rebalance_days,
            transaction_cost=self.base_config.costs.transaction_cost,
            target_positions=params.get("target_positions", 15),
            min_positions=params.get("min_positions", 12),
            use_stop_loss=True,
            initial_stop_loss=params.get("initial_stop_loss", 0.18),
            trailing_stop=params.get("trailing_stop", 0.15),
            trailing_activation=params.get("trailing_activation", 0.08),
            min_score_percentile=params.get("min_score_percentile", 95),
            min_52w_high_prox=params.get("min_52w_high_prox", 0.85),
            weight_6m=weight_6m,
            weight_12m=weight_12m,
        )

        # Create a modified config with updated pure_momentum settings
        # We need to temporarily modify the app_config for NMS calculation params
        # Since Config is frozen, we create a new one with updated values
        config_data = {
            "zerodha": {"api_key": self.base_config.zerodha.api_key, "api_secret": self.base_config.zerodha.api_secret},
            "portfolio": {"initial_capital": self.base_config.portfolio.initial_capital, "max_positions": self.base_config.portfolio.max_positions},
            "pure_momentum": {
                "lookback_6m": self.base_config.pure_momentum.lookback_6m,
                "lookback_12m": self.base_config.pure_momentum.lookback_12m,
                "lookback_volatility": self.base_config.pure_momentum.lookback_volatility,
                "skip_recent_days": self.base_config.pure_momentum.skip_recent_days,
                "weight_6m": weight_6m,
                "weight_12m": weight_12m,
                "min_score_percentile": params.get("min_score_percentile", 95),
                "min_52w_high_prox": params.get("min_52w_high_prox", 0.85),
                "min_volume_ratio": self.base_config.pure_momentum.min_volume_ratio,
                "min_daily_turnover": self.base_config.pure_momentum.min_daily_turnover,
                "min_hold_percentile": self.base_config.pure_momentum.min_hold_percentile,
                "max_days_without_gain": self.base_config.pure_momentum.max_days_without_gain,
                "min_gain_threshold": self.base_config.pure_momentum.min_gain_threshold,
            },
            "position_sizing": {
                "method": self.base_config.position_sizing.method,
                "max_single_position": self.base_config.position_sizing.max_single_position,
                "min_single_position": self.base_config.position_sizing.min_single_position,
                "max_sector_exposure": self.base_config.position_sizing.max_sector_exposure,
                "target_positions": params.get("target_positions", 15),
                "min_positions": params.get("min_positions", 12),
                "max_positions": params.get("max_positions", 20),
            },
            "risk": {
                "initial_stop_loss": params.get("initial_stop_loss", 0.18),
                "trailing_stop": params.get("trailing_stop", 0.15),
                "trailing_activation": params.get("trailing_activation", 0.08),
                "max_single_position": self.base_config.risk.max_single_position,
                "hard_max_position": self.base_config.risk.hard_max_position,
                "max_sector_exposure": self.base_config.risk.max_sector_exposure,
                "hard_max_sector": self.base_config.risk.hard_max_sector,
                "max_drawdown_warning": self.base_config.risk.max_drawdown_warning,
                "max_drawdown_halt": self.base_config.risk.max_drawdown_halt,
                "daily_loss_limit": self.base_config.risk.daily_loss_limit,
            },
            "rebalancing": {
                "frequency": self.base_config.rebalancing.frequency,
                "day": self.base_config.rebalancing.day,
                "min_trade_value": self.base_config.rebalancing.min_trade_value,
            },
            "costs": {"transaction_cost": self.base_config.costs.transaction_cost},
            "paths": {
                "universe_file": self.base_config.paths.universe_file,
                "log_dir": self.base_config.paths.log_dir,
                "data_cache": self.base_config.paths.data_cache,
            },
            "excluded_symbols": list(self.base_config.excluded_symbols),
        }

        app_config = Config(**config_data)

        # Run backtest
        engine = BacktestEngine(
            universe=self.universe,
            historical_data=self.historical_data,
            config=bt_config,
            app_config=app_config,
        )

        return engine.run()

    def run_optimization(
        self,
        param_grid: Dict[str, List],
        max_dd_filter: float = -0.25,
        include_period_analysis: bool = False,
    ) -> List[OptimizationResult]:
        """
        Run grid search optimization over all parameter combinations.

        Args:
            param_grid: Parameter grid to search
            max_dd_filter: Filter out results with drawdown worse than this
            include_period_analysis: Run additional backtests for market periods

        Returns:
            List of results sorted by Sharpe ratio (descending)
        """
        combinations = self.generate_combinations(param_grid)
        total = len(combinations)

        console.print(f"\n[bold cyan]Starting optimization with {total} parameter combinations[/bold cyan]")
        console.print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        console.print(f"Max DD filter: {max_dd_filter:.1%}")

        results: List[OptimizationResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Testing parameter combinations...",
                total=total,
            )

            for i, params in enumerate(combinations):
                try:
                    # Run main backtest
                    result = self.run_single_backtest(params)

                    # Skip if drawdown is too severe
                    if result.max_drawdown < max_dd_filter:
                        progress.update(task, completed=i + 1)
                        continue

                    # Create optimization result
                    opt_result = OptimizationResult(
                        params=params,
                        total_return=result.total_return,
                        cagr=result.cagr,
                        sharpe_ratio=result.sharpe_ratio,
                        max_drawdown=result.max_drawdown,
                        win_rate=result.win_rate,
                        total_trades=result.total_trades,
                    )

                    # Optionally run period-specific backtests
                    if include_period_analysis:
                        # Bear period (Feb-Jun 2022)
                        bear_start, bear_end = self.market_periods["bear_2022"]
                        if bear_start >= self.start_date:
                            bear_result = self.run_single_backtest(params, bear_start, bear_end)
                            opt_result.bear_period_return = bear_result.total_return

                        # Sideways period (Jan-Mar 2023)
                        sw_start, sw_end = self.market_periods["sideways_2023"]
                        if sw_start >= self.start_date and sw_end <= self.end_date:
                            sw_result = self.run_single_backtest(params, sw_start, sw_end)
                            opt_result.sideways_period_return = sw_result.total_return

                        # Bull period (Apr-Dec 2023)
                        bull_start, bull_end = self.market_periods["bull_2023"]
                        if bull_start >= self.start_date and bull_end <= self.end_date:
                            bull_result = self.run_single_backtest(params, bull_start, bull_end)
                            opt_result.bull_period_return = bull_result.total_return

                    results.append(opt_result)

                except Exception as e:
                    console.print(f"[dim red]Error with params {params}: {e}[/dim red]")

                progress.update(task, completed=i + 1)

        # Sort by Sharpe ratio descending
        results.sort(key=lambda x: x.sharpe_ratio, reverse=True)

        console.print(f"\n[green]Completed! {len(results)} valid results (passed DD filter)[/green]")

        return results

    def display_top_results(
        self,
        results: List[OptimizationResult],
        n: int = 20,
        title: str = "Top Results by Sharpe Ratio",
    ) -> None:
        """Display top N results in a formatted table."""
        table = Table(title=title, show_lines=True)

        # Add columns
        table.add_column("#", justify="center", style="cyan", width=3)
        table.add_column("Rebal", justify="right", width=5)
        table.add_column("W6M", justify="right", width=5)
        table.add_column("Pctl", justify="right", width=5)
        table.add_column("52W%", justify="right", width=5)
        table.add_column("Pos", justify="right", width=4)
        table.add_column("ISL", justify="right", width=5)
        table.add_column("TSL", justify="right", width=5)
        table.add_column("Sharpe", justify="right", style="bold", width=7)
        table.add_column("CAGR", justify="right", width=8)
        table.add_column("MaxDD", justify="right", width=8)
        table.add_column("Win%", justify="right", width=6)
        table.add_column("Trades", justify="right", width=6)

        for i, result in enumerate(results[:n], 1):
            params = result.params

            # Color code metrics
            sharpe_color = "green" if result.sharpe_ratio > 6.0 else "yellow" if result.sharpe_ratio > 4.0 else "white"
            cagr_color = "green" if result.cagr > 0.28 else "yellow" if result.cagr > 0.20 else "white"
            dd_color = "green" if result.max_drawdown > -0.15 else "yellow" if result.max_drawdown > -0.20 else "red"
            win_color = "green" if result.win_rate > 0.50 else "yellow" if result.win_rate > 0.45 else "white"

            table.add_row(
                str(i),
                str(params.get("rebalance_days", "")),
                f"{params.get('weight_6m', 0.5):.2f}",
                str(int(params.get("min_score_percentile", 95))),
                f"{params.get('min_52w_high_prox', 0.85):.2f}",
                str(params.get("target_positions", 15)),
                f"{params.get('initial_stop_loss', 0.15):.2f}",
                f"{params.get('trailing_stop', 0.12):.2f}",
                f"[{sharpe_color}]{result.sharpe_ratio:.2f}[/{sharpe_color}]",
                f"[{cagr_color}]{result.cagr:.1%}[/{cagr_color}]",
                f"[{dd_color}]{result.max_drawdown:.1%}[/{dd_color}]",
                f"[{win_color}]{result.win_rate:.0%}[/{win_color}]",
                str(result.total_trades),
            )

        console.print(table)

    def save_results(
        self,
        results: List[OptimizationResult],
        filepath: str,
    ) -> None:
        """Save results to CSV file."""
        if not results:
            return

        rows = [r.to_dict() for r in results]

        path = Path(filepath)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        console.print(f"[green]Results saved to {filepath}[/green]")

    def analyze_best_params(
        self,
        results: List[OptimizationResult],
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Analyze common patterns in top performing parameters."""
        if len(results) < top_n:
            top_n = len(results)

        top_results = results[:top_n]

        # Aggregate parameter values
        param_values: Dict[str, List] = {}
        for result in top_results:
            for key, value in result.params.items():
                if key not in param_values:
                    param_values[key] = []
                param_values[key].append(value)

        # Calculate statistics for each parameter
        analysis = {}
        for param, values in param_values.items():
            if isinstance(values[0], (int, float)):
                analysis[param] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "mode": max(set(values), key=values.count),
                    "values": values,
                }
            else:
                analysis[param] = {
                    "mode": max(set(values), key=values.count),
                    "values": values,
                }

        return analysis

    def print_parameter_analysis(
        self,
        results: List[OptimizationResult],
        top_n: int = 10,
    ) -> None:
        """Print analysis of winning parameter patterns."""
        analysis = self.analyze_best_params(results, top_n)

        console.print(Panel(
            f"[bold]Parameter Analysis (Top {top_n} Results)[/bold]",
            style="cyan",
        ))

        table = Table(show_header=True, header_style="bold")
        table.add_column("Parameter", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Mode", justify="right", style="green")

        for param, stats in analysis.items():
            if "mean" in stats:
                table.add_row(
                    param,
                    f"{stats['mean']:.3f}" if isinstance(stats['mean'], float) else str(stats['mean']),
                    str(stats['min']),
                    str(stats['max']),
                    str(stats['mode']),
                )
            else:
                table.add_row(
                    param,
                    "-",
                    "-",
                    "-",
                    str(stats['mode']),
                )

        console.print(table)

        # Print recommended config
        console.print("\n[bold cyan]Recommended Configuration:[/bold cyan]")
        for param, stats in analysis.items():
            mode = stats['mode']
            console.print(f"  {param}: {mode}")


def load_data_for_optimization(
    config: Config,
    months: int = 48,
) -> Tuple[Universe, Dict[str, pd.DataFrame], datetime, datetime]:
    """
    Load universe and historical data for optimization.

    Uses cached parquet files from the .cache directory if available.
    Requires running 'fortress-legacy backtest' first to populate the cache.
    """
    import json
    from pathlib import Path

    # Load universe
    universe = Universe(config.paths.universe_file)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(months=months)
    start_date = start_date.to_pydatetime()

    console.print(f"[cyan]Loading data for {months} months: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}[/cyan]")

    # Get all symbols needed
    all_stocks = universe.get_all_stocks()
    stock_symbols = [s.zerodha_symbol for s in all_stocks]

    index_symbols = ["NIFTY 50", "INDIA VIX"]
    for sector in universe.get_valid_sectors(3):
        idx_info = universe.get_sector_index(sector)
        if idx_info and idx_info.symbol:
            index_symbols.append(idx_info.symbol)

    all_symbols = list(set(stock_symbols + index_symbols))
    console.print(f"[cyan]Total symbols needed: {len(all_symbols)}[/cyan]")

    # Load from cache directory
    cache_dir = Path(config.paths.data_cache)
    if not cache_dir.exists():
        console.print("[red]Cache directory not found. Please run 'fortress-legacy backtest' first to populate the cache.[/red]")
        sys.exit(1)

    historical_data: Dict[str, pd.DataFrame] = {}
    missing_symbols = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Loading cached data...", total=len(all_symbols))

        for symbol in all_symbols:
            cache_file = cache_dir / f"{symbol.replace(' ', '_')}.parquet"

            if cache_file.exists():
                try:
                    df = pd.read_parquet(cache_file)
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_convert(None)

                    # Filter to date range
                    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
                    df_filtered = df[mask]

                    if len(df_filtered) > 100:  # Need sufficient history
                        historical_data[symbol] = df_filtered
                except Exception as e:
                    missing_symbols.append(symbol)
            else:
                missing_symbols.append(symbol)

            progress.update(task, advance=1)

    console.print(f"[green]Loaded data for {len(historical_data)} symbols[/green]")

    if missing_symbols and len(missing_symbols) < 20:
        console.print(f"[yellow]Missing {len(missing_symbols)} symbols: {missing_symbols[:10]}...[/yellow]")
    elif missing_symbols:
        console.print(f"[yellow]Missing {len(missing_symbols)} symbols (run backtest first to cache them)[/yellow]")

    if len(historical_data) < 50:
        console.print("[red]Insufficient cached data for optimization (need at least 50 symbols).[/red]")
        console.print("[yellow]Please run 'fortress-legacy backtest' first to populate the cache.[/yellow]")
        sys.exit(1)

    return universe, historical_data, start_date, end_date


def run_baseline(
    optimizer: MomentumOptimizer,
) -> OptimizationResult:
    """Run baseline backtest with current config."""
    console.print("\n[bold cyan]Running baseline backtest with current config...[/bold cyan]")

    # Current config parameters
    baseline_params = {
        "rebalance_days": 21,
        "weight_6m": 0.50,
        "min_score_percentile": 95,
        "min_52w_high_prox": 0.85,
        "target_positions": 15,
        "initial_stop_loss": 0.15,
        "trailing_stop": 0.12,
    }

    result = optimizer.run_single_backtest(baseline_params)

    opt_result = OptimizationResult(
        params=baseline_params,
        total_return=result.total_return,
        cagr=result.cagr,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown=result.max_drawdown,
        win_rate=result.win_rate,
        total_trades=result.total_trades,
    )

    console.print(Panel(
        f"[bold]Baseline Results[/bold]\n\n"
        f"Total Return: [cyan]{result.total_return:.1%}[/cyan]\n"
        f"CAGR: [cyan]{result.cagr:.1%}[/cyan]\n"
        f"Sharpe Ratio: [cyan]{result.sharpe_ratio:.2f}[/cyan]\n"
        f"Max Drawdown: [{'green' if result.max_drawdown > -0.20 else 'red'}]{result.max_drawdown:.1%}[/{'green' if result.max_drawdown > -0.20 else 'red'}]\n"
        f"Win Rate: [cyan]{result.win_rate:.1%}[/cyan]\n"
        f"Total Trades: [cyan]{result.total_trades}[/cyan]",
        title="Baseline (Current Config)",
        style="cyan",
    ))

    return opt_result


def main():
    parser = argparse.ArgumentParser(
        description="FORTRESS MOMENTUM Optimizer - Grid search for optimal parameters"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick optimization (~50 combinations)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full exhaustive optimization (~2000+ combinations)",
    )
    parser.add_argument(
        "--regime",
        action="store_true",
        help="Also optimize regime detection thresholds",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=48,
        help="Number of months for backtest (default: 48)",
    )
    parser.add_argument(
        "--periods",
        action="store_true",
        help="Include period-specific analysis (bear/sideways/bull)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimization_results.csv",
        help="Output CSV file for results",
    )

    args = parser.parse_args()

    console.print(Panel(
        "[bold]FORTRESS MOMENTUM Optimizer[/bold]\n\n"
        "Grid search optimization for all-weather performance",
        style="cyan",
    ))

    # Load config
    try:
        config = load_config("config.yaml")
    except FileNotFoundError:
        console.print("[red]Error: config.yaml not found[/red]")
        sys.exit(1)

    # Load data
    universe, historical_data, start_date, end_date = load_data_for_optimization(
        config, args.months
    )

    # Create optimizer
    optimizer = MomentumOptimizer(
        universe=universe,
        historical_data=historical_data,
        base_config=config,
        start_date=start_date,
        end_date=end_date,
    )

    # Run baseline
    baseline = run_baseline(optimizer)

    # Select parameter grid
    if args.quick:
        param_grid = QUICK_PARAM_GRID
        grid_name = "Quick"
    elif args.full:
        param_grid = FULL_PARAM_GRID
        grid_name = "Full"
    else:
        param_grid = STANDARD_PARAM_GRID
        grid_name = "Standard"

    # Add regime parameters if requested
    if args.regime:
        param_grid.update(REGIME_PARAM_GRID)
        grid_name += " + Regime"

    combinations = list(itertools.product(*param_grid.values()))
    console.print(f"\n[cyan]{grid_name} grid: {len(combinations)} combinations[/cyan]")

    # Run optimization (relax DD filter to -35% to include more results)
    results = optimizer.run_optimization(
        param_grid,
        max_dd_filter=-0.35,
        include_period_analysis=args.periods,
    )

    if not results:
        console.print("[red]No results passed the filters![/red]")
        sys.exit(1)

    # Display results
    optimizer.display_top_results(results, n=20)

    # Parameter analysis
    optimizer.print_parameter_analysis(results, top_n=10)

    # Compare best vs baseline
    best = results[0]
    console.print(Panel(
        f"[bold]Best vs Baseline Comparison[/bold]\n\n"
        f"{'Metric':<15} {'Baseline':>12} {'Best':>12} {'Improvement':>12}\n"
        f"{'-'*51}\n"
        f"{'CAGR':<15} {baseline.cagr:>11.1%} {best.cagr:>11.1%} {(best.cagr - baseline.cagr):>+11.1%}\n"
        f"{'Sharpe':<15} {baseline.sharpe_ratio:>12.2f} {best.sharpe_ratio:>12.2f} {(best.sharpe_ratio - baseline.sharpe_ratio):>+12.2f}\n"
        f"{'Max DD':<15} {baseline.max_drawdown:>11.1%} {best.max_drawdown:>11.1%} {(best.max_drawdown - baseline.max_drawdown):>+11.1%}\n"
        f"{'Win Rate':<15} {baseline.win_rate:>11.1%} {best.win_rate:>11.1%} {(best.win_rate - baseline.win_rate):>+11.1%}\n",
        title="Optimization Results",
        style="green" if best.sharpe_ratio > baseline.sharpe_ratio else "yellow",
    ))

    # Print best parameters for config.yaml
    console.print("\n[bold cyan]Best Parameters for config.yaml:[/bold cyan]")
    console.print("```yaml")
    console.print("pure_momentum:")
    weight_6m = best.params.get("weight_6m", 0.50)
    console.print(f"  weight_6m: {weight_6m:.2f}")
    console.print(f"  weight_12m: {1.0 - weight_6m:.2f}")
    console.print(f"  min_score_percentile: {best.params.get('min_score_percentile', 95)}")
    console.print(f"  min_52w_high_prox: {best.params.get('min_52w_high_prox', 0.85)}")
    console.print("")
    console.print("position_sizing:")
    console.print(f"  target_positions: {best.params.get('target_positions', 15)}")
    console.print("")
    console.print("risk:")
    console.print(f"  initial_stop_loss: {best.params.get('initial_stop_loss', 0.15)}")
    console.print(f"  trailing_stop: {best.params.get('trailing_stop', 0.12)}")
    console.print("")
    console.print(f"# Rebalance every {best.params.get('rebalance_days', 21)} trading days")
    console.print("```")

    # Save results
    optimizer.save_results(results, args.output)

    console.print("\n[green]Optimization complete![/green]")


if __name__ == "__main__":
    main()
