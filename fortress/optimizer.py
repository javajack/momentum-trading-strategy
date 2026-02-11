"""
Grid search optimizer for FORTRESS MOMENTUM strategy parameters.

Systematically tests parameter combinations to find optimal NMS settings.
"""

import csv
import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from .backtest import BacktestConfig, BacktestEngine
from .config import Config
from .universe import Universe


# Parameter grid for pure momentum (NMS-based) strategy optimization
NMS_PARAM_GRID = {
    # Rebalance frequency
    "rebalance_days": [15, 21, 42],
    # NMS weight allocation (6M vs 12M momentum)
    "weight_6m": [0.40, 0.50, 0.60, 0.70],
    # Entry filters
    "min_score_percentile": [90, 93, 95, 97],
    "min_52w_high_prox": [0.80, 0.85, 0.90],
    # Position sizing
    "target_positions": [12, 15, 18, 20],
    # Stop loss parameters
    "initial_stop_loss": [0.12, 0.15, 0.18],
    "trailing_stop": [0.10, 0.12, 0.15],
}


# Quick NMS grid for fast testing (~50 combinations)
NMS_QUICK_GRID = {
    "rebalance_days": [15, 21],
    "weight_6m": [0.50, 0.60],
    "min_score_percentile": [93, 95],
    "min_52w_high_prox": [0.85],
    "target_positions": [15],
    "initial_stop_loss": [0.15],
    "trailing_stop": [0.12],
}


# Full NMS grid for exhaustive search (~2000+ combinations)
NMS_FULL_GRID = {
    "rebalance_days": [10, 15, 21, 30, 42],
    "weight_6m": [0.30, 0.40, 0.50, 0.60, 0.70],
    "min_score_percentile": [88, 90, 93, 95, 97],
    "min_52w_high_prox": [0.75, 0.80, 0.85, 0.90],
    "target_positions": [10, 12, 15, 18, 20],
    "initial_stop_loss": [0.10, 0.12, 0.15, 0.18, 0.20],
    "trailing_stop": [0.08, 0.10, 0.12, 0.15],
}


@dataclass
class GridSearchResult:
    """Result of a single parameter combination test."""

    params: Dict[str, Any]
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    total_return: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        result = {}
        # Flatten params
        for key, value in self.params.items():
            if isinstance(value, tuple):
                result[key] = str(value)
            else:
                result[key] = value
        # Add metrics
        result["cagr"] = self.cagr
        result["sharpe_ratio"] = self.sharpe_ratio
        result["max_drawdown"] = self.max_drawdown
        result["win_rate"] = self.win_rate
        result["total_trades"] = self.total_trades
        result["total_return"] = self.total_return
        return result


class GridSearchEngine:
    """
    Systematic parameter optimization through grid search.

    Tests all combinations of parameters and ranks by Sharpe ratio.
    """

    def __init__(
        self,
        universe: Universe,
        historical_data: Dict[str, pd.DataFrame],
        base_config: Config,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 1600000,
        console: Optional[Console] = None,
    ):
        """
        Initialize grid search engine.

        Args:
            universe: Stock universe
            historical_data: Dict mapping symbol to OHLC DataFrame
            base_config: Base configuration to use
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            console: Rich console for output
        """
        self.universe = universe
        self.historical_data = historical_data
        self.base_config = base_config
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.console = console or Console()

    def generate_combinations(
        self,
        param_grid: Optional[Dict[str, List]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from grid.

        Args:
            param_grid: Dict of param_name -> list of values

        Returns:
            List of parameter dicts
        """
        grid = param_grid or NMS_PARAM_GRID

        # Get all keys and values
        keys = list(grid.keys())
        values = list(grid.values())

        # Generate cartesian product
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def run_single_backtest(
        self,
        params: Dict[str, Any],
    ) -> GridSearchResult:
        """
        Run backtest with specific parameters.

        Args:
            params: Parameter dict

        Returns:
            GridSearchResult with metrics
        """
        # Determine rebalance days
        rebalance_days = params.get("rebalance_days", 21)

        # Create backtest config with overrides
        bt_config = BacktestConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            rebalance_days=rebalance_days,
            transaction_cost=self.base_config.costs.transaction_cost,
            target_positions=params.get("target_positions", 15),
            min_positions=params.get("min_positions", 12),
            use_stop_loss=params.get("use_stop_loss", True),
            initial_stop_loss=params.get("initial_stop_loss", 0.18),
            trailing_stop=params.get("trailing_stop", 0.15),
            trailing_activation=params.get("trailing_activation", 0.08),
            min_score_percentile=params.get("min_score_percentile", 95),
            min_52w_high_prox=params.get("min_52w_high_prox", 0.85),
            weight_6m=params.get("weight_6m"),
            weight_12m=1.0 - params.get("weight_6m", 0.5) if "weight_6m" in params else None,
        )

        # Create modified app_config if NMS parameters are provided
        app_config = self.base_config
        if "weight_6m" in params:
            app_config = self._create_modified_config(params)

        # Run backtest
        engine = BacktestEngine(
            universe=self.universe,
            historical_data=self.historical_data,
            config=bt_config,
            app_config=app_config,
        )

        result = engine.run()

        return GridSearchResult(
            params=params,
            cagr=result.cagr,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            total_trades=result.total_trades,
            total_return=result.total_return,
        )

    def _create_modified_config(self, params: Dict[str, Any]) -> Config:
        """
        Create a modified Config with updated pure_momentum settings.

        Since Config is frozen, we create a new one with updated values.
        """
        weight_6m = params.get("weight_6m", 0.50)
        weight_12m = 1.0 - weight_6m

        config_data = {
            "zerodha": {
                "api_key": self.base_config.zerodha.api_key,
                "api_secret": self.base_config.zerodha.api_secret,
            },
            "portfolio": {
                "initial_capital": self.base_config.portfolio.initial_capital,
                "max_positions": self.base_config.portfolio.max_positions,
            },
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

        return Config(**config_data)

    def run_grid_search(
        self,
        param_grid: Optional[Dict[str, List]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        max_dd_filter: Optional[float] = None,
    ) -> List[GridSearchResult]:
        """
        Run grid search over all parameter combinations.

        Args:
            param_grid: Optional custom parameter grid
            progress_callback: Optional callback(current, total) for progress
            max_dd_filter: Filter out results with drawdown worse than this (e.g., -0.25)

        Returns:
            List of results sorted by Sharpe ratio (descending)
        """
        combinations = self.generate_combinations(param_grid)
        total = len(combinations)

        results: List[GridSearchResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Testing parameter combinations...",
                total=total,
            )

            for i, params in enumerate(combinations):
                try:
                    result = self.run_single_backtest(params)

                    # Apply max drawdown filter if specified
                    if max_dd_filter is not None and result.max_drawdown < max_dd_filter:
                        progress.update(task, completed=i + 1)
                        continue

                    results.append(result)
                except Exception as e:
                    # Log error but continue
                    self.console.print(
                        f"[dim red]Error with params {params}: {e}[/dim red]"
                    )

                progress.update(task, completed=i + 1)

                if progress_callback:
                    progress_callback(i + 1, total)

        # Sort by Sharpe ratio descending
        results.sort(key=lambda x: x.sharpe_ratio, reverse=True)

        return results

    def run_nms_grid_search(
        self,
        grid_type: str = "standard",
        max_dd_filter: float = -0.25,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[GridSearchResult]:
        """
        Run grid search optimized for pure momentum (NMS) strategy.

        Args:
            grid_type: "quick", "standard", or "full"
            max_dd_filter: Filter out results with drawdown worse than this
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of results sorted by Sharpe ratio (descending)
        """
        # Select grid
        if grid_type == "quick":
            param_grid = NMS_QUICK_GRID.copy()
        elif grid_type == "full":
            param_grid = NMS_FULL_GRID.copy()
        else:
            param_grid = NMS_PARAM_GRID.copy()

        combinations = self.generate_combinations(param_grid)
        self.console.print(f"[cyan]NMS Grid Search: {len(combinations)} combinations[/cyan]")

        return self.run_grid_search(
            param_grid,
            progress_callback=progress_callback,
            max_dd_filter=max_dd_filter,
        )

    def save_results(
        self,
        results: List[GridSearchResult],
        filepath: str,
    ) -> None:
        """
        Save results to CSV file.

        Args:
            results: List of GridSearchResult
            filepath: Output CSV path
        """
        if not results:
            return

        # Convert to dicts
        rows = [r.to_dict() for r in results]

        # Write CSV
        path = Path(filepath)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def display_results(
        self,
        results: List[GridSearchResult],
        n: int = 20,
        title: str = "Top NMS Results (by Sharpe Ratio)",
    ) -> None:
        """
        Display top N results for NMS (pure momentum) optimization.

        Args:
            results: List of GridSearchResult
            n: Number of results to display
            title: Table title
        """
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

        self.console.print(table)

    def analyze_parameter_patterns(
        self,
        results: List[GridSearchResult],
        top_n: int = 10,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze common patterns in top performing parameters.

        Args:
            results: List of GridSearchResult
            top_n: Number of top results to analyze

        Returns:
            Dict with statistics for each parameter
        """
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

    def display_parameter_analysis(
        self,
        results: List[GridSearchResult],
        top_n: int = 10,
    ) -> None:
        """
        Display analysis of winning parameter patterns.

        Args:
            results: List of GridSearchResult
            top_n: Number of top results to analyze
        """
        analysis = self.analyze_parameter_patterns(results, top_n)

        self.console.print(f"\n[bold cyan]Parameter Analysis (Top {top_n} Results)[/bold cyan]")

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
                    f"{stats['mean']:.3f}" if isinstance(stats['mean'], float) else str(int(stats['mean'])),
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

        self.console.print(table)

        # Print recommended config
        self.console.print("\n[bold cyan]Recommended Configuration (mode values):[/bold cyan]")
        for param, stats in analysis.items():
            mode = stats['mode']
            self.console.print(f"  {param}: {mode}")

    def get_best_params(
        self,
        results: List[GridSearchResult],
        metric: str = "sharpe_ratio",
    ) -> Optional[Dict[str, Any]]:
        """
        Get best parameters by specified metric.

        Args:
            results: List of GridSearchResult
            metric: Metric to optimize ("sharpe_ratio", "cagr", "win_rate")

        Returns:
            Best parameter dict or None
        """
        if not results:
            return None

        if metric == "sharpe_ratio":
            best = max(results, key=lambda x: x.sharpe_ratio)
        elif metric == "cagr":
            best = max(results, key=lambda x: x.cagr)
        elif metric == "win_rate":
            best = max(results, key=lambda x: x.win_rate)
        elif metric == "max_drawdown":
            best = max(results, key=lambda x: x.max_drawdown)  # Less negative is better
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return best.params
