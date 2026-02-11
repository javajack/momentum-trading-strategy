#!/usr/bin/env python3
"""
Compare backtest performance across two time periods:
1. 2019-2020 (or earliest available to 2020)
2. Jan 2020 - Jan 2024

Analyzes why performance differs between these periods.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
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
        console.print("[red]No cached data found. Run fetch_and_backtest.py first.[/red]")
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
            except Exception as e:
                pass

    return historical_data


def run_backtest(
    universe: Universe,
    historical_data: dict,
    config,
    start_date: datetime,
    end_date: datetime,
    strategy_name: str = "dual_momentum",
    rebalance_days: int = 5,  # Changed to weekly (5 trading days)
) -> BacktestResult:
    """Run a single backtest."""
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
        strategy_name=strategy_name,
    )

    engine = BacktestEngine(
        universe=universe,
        historical_data=historical_data,
        config=bt_config,
        app_config=config,
    )

    return engine.run()


def analyze_market_conditions(historical_data: dict, start_date: datetime, end_date: datetime) -> dict:
    """Analyze market conditions during a period."""

    # Try to get NIFTY 50 data
    nifty_data = None
    for key in ['NIFTY 50', 'NIFTY50', 'NSE:NIFTY50']:
        if key in historical_data:
            nifty_data = historical_data[key]
            break

    if nifty_data is None:
        # Try to find any index
        for key, df in historical_data.items():
            if 'NIFTY' in key.upper():
                nifty_data = df
                break

    analysis = {
        "period": f"{start_date.date()} to {end_date.date()}",
        "days": (end_date - start_date).days,
    }

    if nifty_data is not None:
        period_data = nifty_data[(nifty_data.index >= pd.Timestamp(start_date)) &
                                  (nifty_data.index <= pd.Timestamp(end_date))]

        if len(period_data) > 10:
            start_price = period_data['close'].iloc[0]
            end_price = period_data['close'].iloc[-1]

            analysis["nifty_return"] = (end_price / start_price - 1) * 100
            analysis["nifty_start"] = start_price
            analysis["nifty_end"] = end_price

            # Calculate volatility
            returns = period_data['close'].pct_change().dropna()
            analysis["daily_volatility"] = returns.std() * 100
            analysis["annualized_volatility"] = returns.std() * np.sqrt(252) * 100

            # Max drawdown
            rolling_max = period_data['close'].expanding().max()
            drawdowns = (period_data['close'] - rolling_max) / rolling_max
            analysis["max_drawdown"] = drawdowns.min() * 100

            # Trend analysis - percentage of up days
            analysis["up_days_pct"] = (returns > 0).sum() / len(returns) * 100

            # Average monthly return
            monthly = period_data['close'].resample('M').last().pct_change().dropna()
            analysis["avg_monthly_return"] = monthly.mean() * 100
            analysis["positive_months"] = (monthly > 0).sum()
            analysis["negative_months"] = (monthly < 0).sum()

    return analysis


def analyze_stock_universe(historical_data: dict, start_date: datetime, end_date: datetime) -> dict:
    """Analyze stock-level characteristics."""

    returns = []
    volatilities = []
    momentum_quality = []

    for symbol, df in historical_data.items():
        if 'NIFTY' in symbol.upper():
            continue

        period_data = df[(df.index >= pd.Timestamp(start_date)) &
                         (df.index <= pd.Timestamp(end_date))]

        if len(period_data) < 60:  # Need at least 60 days
            continue

        start_price = period_data['close'].iloc[0]
        end_price = period_data['close'].iloc[-1]
        total_return = (end_price / start_price - 1) * 100
        returns.append(total_return)

        daily_returns = period_data['close'].pct_change().dropna()
        vol = daily_returns.std() * np.sqrt(252) * 100
        volatilities.append(vol)

        # Momentum quality: check if stocks with positive momentum continued
        mid_idx = len(period_data) // 2
        first_half = period_data.iloc[:mid_idx]
        second_half = period_data.iloc[mid_idx:]

        first_ret = (first_half['close'].iloc[-1] / first_half['close'].iloc[0] - 1)
        second_ret = (second_half['close'].iloc[-1] / second_half['close'].iloc[0] - 1)

        # Momentum continuation: positive first half followed by positive second half
        if first_ret > 0.05:  # >5% in first half
            momentum_quality.append(1 if second_ret > 0 else 0)

    return {
        "avg_stock_return": np.mean(returns) if returns else 0,
        "median_stock_return": np.median(returns) if returns else 0,
        "stocks_positive": sum(1 for r in returns if r > 0),
        "stocks_negative": sum(1 for r in returns if r <= 0),
        "total_stocks": len(returns),
        "avg_volatility": np.mean(volatilities) if volatilities else 0,
        "momentum_continuation_rate": np.mean(momentum_quality) * 100 if momentum_quality else 0,
        "top_quartile_avg": np.mean(sorted(returns, reverse=True)[:len(returns)//4]) if returns else 0,
        "bottom_quartile_avg": np.mean(sorted(returns)[:len(returns)//4]) if returns else 0,
    }


def print_comparison_table(result1: BacktestResult, result2: BacktestResult,
                           period1_name: str, period2_name: str):
    """Print comparison table for two backtest results."""

    table = Table(title="Backtest Performance Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column(period1_name, justify="right")
    table.add_column(period2_name, justify="right")
    table.add_column("Difference", justify="right")

    metrics = [
        ("Total Return", f"{result1.total_return:.1%}", f"{result2.total_return:.1%}",
         f"{(result2.total_return - result1.total_return):.1%}"),
        ("CAGR", f"{result1.cagr:.1%}", f"{result2.cagr:.1%}",
         f"{(result2.cagr - result1.cagr):.1%}"),
        ("Sharpe Ratio", f"{result1.sharpe_ratio:.2f}", f"{result2.sharpe_ratio:.2f}",
         f"{(result2.sharpe_ratio - result1.sharpe_ratio):.2f}"),
        ("Max Drawdown", f"{result1.max_drawdown:.1%}", f"{result2.max_drawdown:.1%}",
         f"{(result2.max_drawdown - result1.max_drawdown):.1%}"),
        ("Win Rate", f"{result1.win_rate:.0%}", f"{result2.win_rate:.0%}",
         f"{(result2.win_rate - result1.win_rate):.0%}"),
        ("Total Trades", f"{result1.total_trades}", f"{result2.total_trades}",
         f"{result2.total_trades - result1.total_trades}"),
        ("Final Value", f"₹{result1.final_value:,.0f}", f"₹{result2.final_value:,.0f}",
         f"₹{(result2.final_value - result1.final_value):,.0f}"),
    ]

    for metric, v1, v2, diff in metrics:
        # Color the difference
        if "Return" in metric or "CAGR" in metric or "Sharpe" in metric or "Win" in metric:
            diff_color = "green" if float(diff.replace('%', '')) > 0 else "red"
        elif "Drawdown" in metric:
            diff_color = "green" if float(diff.replace('%', '')) < 0 else "red"
        else:
            diff_color = "white"

        table.add_row(metric, v1, v2, f"[{diff_color}]{diff}[/{diff_color}]")

    console.print(table)


def main():
    console.print(Panel("[bold cyan]PERIOD COMPARISON: 2016-2020 vs 2020-2024[/bold cyan]"))

    # Load config and data
    config = load_config()
    universe = Universe(config.paths.universe_file)
    cache_dir = Path(config.paths.data_cache)

    console.print("\n[bold]Loading cached data...[/bold]")
    historical_data = load_cached_data(cache_dir)

    if len(historical_data) < 50:
        console.print("[red]Insufficient cached data. Run fetch_and_backtest.py first.[/red]")
        return

    console.print(f"[green]Loaded {len(historical_data)} symbols[/green]")

    # Determine data range
    earliest = min(df.index[0] for df in historical_data.values() if len(df) > 0)
    latest = max(df.index[-1] for df in historical_data.values() if len(df) > 0)

    console.print(f"Data range: {earliest.date()} to {latest.date()}")

    # Define periods
    # Period 1: Early data to end of 2020 (or 2019 to 2020 to match user's request)
    # Need ~280 days for NMS warmup
    usable_start = earliest + pd.Timedelta(days=300)

    # Period 1: 2016-01-01 to 2020-01-01 (4 years like Period 2)
    p1_start = max(datetime(2016, 1, 1), usable_start.to_pydatetime())
    p1_end = datetime(2020, 1, 1)

    # Period 2: Jan 2020 to Jan 2024
    p2_start = datetime(2020, 1, 1)
    p2_end = min(datetime(2024, 1, 1), latest.to_pydatetime())

    console.print(f"\n[bold]Period 1:[/bold] {p1_start.date()} to {p1_end.date()} (~{(p1_end - p1_start).days} days)")
    console.print(f"[bold]Period 2:[/bold] {p2_start.date()} to {p2_end.date()} (~{(p2_end - p2_start).days} days)")

    # Check if period 1 is valid
    if p1_start >= p1_end:
        console.print(f"[yellow]Warning: Period 1 dates invalid. Adjusting...[/yellow]")
        p1_start = usable_start.to_pydatetime()
        p1_end = datetime(2020, 12, 31) if p1_start.year < 2020 else p1_start + pd.Timedelta(days=365)
        console.print(f"[bold]Adjusted Period 1:[/bold] {p1_start.date()} to {p1_end.date()}")

    # ===== MARKET CONDITION ANALYSIS =====
    console.print("\n" + "="*60)
    console.print("[bold]MARKET CONDITION ANALYSIS[/bold]")
    console.print("="*60)

    market1 = analyze_market_conditions(historical_data, p1_start, p1_end)
    market2 = analyze_market_conditions(historical_data, p2_start, p2_end)

    market_table = Table(title="Market Conditions Comparison")
    market_table.add_column("Metric", style="cyan")
    market_table.add_column(f"Period 1\n{p1_start.date()} to {p1_end.date()}", justify="right")
    market_table.add_column(f"Period 2\n{p2_start.date()} to {p2_end.date()}", justify="right")

    market_metrics = [
        ("NIFTY Return", f"{market1.get('nifty_return', 0):.1f}%", f"{market2.get('nifty_return', 0):.1f}%"),
        ("Annualized Volatility", f"{market1.get('annualized_volatility', 0):.1f}%", f"{market2.get('annualized_volatility', 0):.1f}%"),
        ("Max Drawdown", f"{market1.get('max_drawdown', 0):.1f}%", f"{market2.get('max_drawdown', 0):.1f}%"),
        ("Up Days %", f"{market1.get('up_days_pct', 0):.1f}%", f"{market2.get('up_days_pct', 0):.1f}%"),
        ("Avg Monthly Return", f"{market1.get('avg_monthly_return', 0):.2f}%", f"{market2.get('avg_monthly_return', 0):.2f}%"),
        ("Positive Months", f"{market1.get('positive_months', 0)}", f"{market2.get('positive_months', 0)}"),
        ("Negative Months", f"{market1.get('negative_months', 0)}", f"{market2.get('negative_months', 0)}"),
    ]

    for metric, v1, v2 in market_metrics:
        market_table.add_row(metric, v1, v2)

    console.print(market_table)

    # ===== STOCK UNIVERSE ANALYSIS =====
    console.print("\n" + "="*60)
    console.print("[bold]STOCK UNIVERSE CHARACTERISTICS[/bold]")
    console.print("="*60)

    stocks1 = analyze_stock_universe(historical_data, p1_start, p1_end)
    stocks2 = analyze_stock_universe(historical_data, p2_start, p2_end)

    stocks_table = Table(title="Stock Universe Analysis")
    stocks_table.add_column("Metric", style="cyan")
    stocks_table.add_column(f"Period 1", justify="right")
    stocks_table.add_column(f"Period 2", justify="right")

    stock_metrics = [
        ("Avg Stock Return", f"{stocks1.get('avg_stock_return', 0):.1f}%", f"{stocks2.get('avg_stock_return', 0):.1f}%"),
        ("Median Stock Return", f"{stocks1.get('median_stock_return', 0):.1f}%", f"{stocks2.get('median_stock_return', 0):.1f}%"),
        ("Stocks Up", f"{stocks1.get('stocks_positive', 0)}", f"{stocks2.get('stocks_positive', 0)}"),
        ("Stocks Down", f"{stocks1.get('stocks_negative', 0)}", f"{stocks2.get('stocks_negative', 0)}"),
        ("Avg Volatility", f"{stocks1.get('avg_volatility', 0):.1f}%", f"{stocks2.get('avg_volatility', 0):.1f}%"),
        ("Momentum Continuation", f"{stocks1.get('momentum_continuation_rate', 0):.1f}%", f"{stocks2.get('momentum_continuation_rate', 0):.1f}%"),
        ("Top 25% Avg Return", f"{stocks1.get('top_quartile_avg', 0):.1f}%", f"{stocks2.get('top_quartile_avg', 0):.1f}%"),
        ("Bottom 25% Avg Return", f"{stocks1.get('bottom_quartile_avg', 0):.1f}%", f"{stocks2.get('bottom_quartile_avg', 0):.1f}%"),
    ]

    for metric, v1, v2 in stock_metrics:
        stocks_table.add_row(metric, v1, v2)

    console.print(stocks_table)

    # ===== RUN BACKTESTS =====
    console.print("\n" + "="*60)
    console.print("[bold]RUNNING BACKTESTS[/bold]")
    console.print("="*60)

    console.print(f"\n[cyan]Running Period 1 backtest ({p1_start.date()} to {p1_end.date()})...[/cyan]")
    try:
        result1 = run_backtest(universe, historical_data, config, p1_start, p1_end, "dual_momentum")
        console.print(f"[green]✓ Period 1: {result1.total_return:+.1%} return, {result1.sharpe_ratio:.2f} Sharpe[/green]")
    except Exception as e:
        console.print(f"[red]Period 1 backtest failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        result1 = None

    console.print(f"\n[cyan]Running Period 2 backtest ({p2_start.date()} to {p2_end.date()})...[/cyan]")
    try:
        result2 = run_backtest(universe, historical_data, config, p2_start, p2_end, "dual_momentum")
        console.print(f"[green]✓ Period 2: {result2.total_return:+.1%} return, {result2.sharpe_ratio:.2f} Sharpe[/green]")
    except Exception as e:
        console.print(f"[red]Period 2 backtest failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        result2 = None

    # ===== COMPARISON =====
    if result1 and result2:
        console.print("\n" + "="*60)
        console.print("[bold]PERFORMANCE COMPARISON[/bold]")
        console.print("="*60 + "\n")

        print_comparison_table(result1, result2,
                              f"Period 1\n({p1_start.year}-{p1_end.year})",
                              f"Period 2\n({p2_start.year}-{p2_end.year})")

        # ===== ANALYSIS AND INSIGHTS =====
        console.print("\n" + "="*60)
        console.print("[bold]WHY THE PERFORMANCE DIFFERS[/bold]")
        console.print("="*60)

        insights = []

        # 1. Market trend
        nifty_diff = market2.get('nifty_return', 0) - market1.get('nifty_return', 0)
        if abs(nifty_diff) > 20:
            if nifty_diff > 0:
                insights.append(f"[green]✓ STRONGER BULL MARKET:[/green] Period 2 saw NIFTY return {market2.get('nifty_return', 0):.1f}% vs {market1.get('nifty_return', 0):.1f}% in Period 1. Momentum strategies thrive in strong uptrends.")
            else:
                insights.append(f"[red]✗ WEAKER MARKET:[/red] Period 1 had weaker overall market ({market1.get('nifty_return', 0):.1f}%) compared to Period 2 ({market2.get('nifty_return', 0):.1f}%).")

        # 2. Volatility
        vol_diff = market2.get('annualized_volatility', 0) - market1.get('annualized_volatility', 0)
        if abs(vol_diff) > 5:
            if vol_diff > 0:
                insights.append(f"[yellow]⚠ HIGHER VOLATILITY:[/yellow] Period 2 had higher volatility ({market2.get('annualized_volatility', 0):.1f}%) which can increase both gains and losses with momentum.")
            else:
                insights.append(f"[yellow]⚠ LOWER VOLATILITY:[/yellow] Period 1 had lower volatility ({market1.get('annualized_volatility', 0):.1f}%) - quieter markets make it harder for momentum to generate alpha.")

        # 3. Momentum continuation
        mom_diff = stocks2.get('momentum_continuation_rate', 0) - stocks1.get('momentum_continuation_rate', 0)
        if abs(mom_diff) > 5:
            if mom_diff > 0:
                insights.append(f"[green]✓ BETTER MOMENTUM PERSISTENCE:[/green] In Period 2, {stocks2.get('momentum_continuation_rate', 0):.0f}% of momentum stocks continued their trend vs {stocks1.get('momentum_continuation_rate', 0):.0f}% in Period 1.")
            else:
                insights.append(f"[red]✗ POOR MOMENTUM PERSISTENCE:[/red] In Period 1, only {stocks1.get('momentum_continuation_rate', 0):.0f}% of momentum stocks continued their trend. Momentum reversals hurt the strategy.")

        # 4. Dispersion (top vs bottom quartile)
        disp1 = stocks1.get('top_quartile_avg', 0) - stocks1.get('bottom_quartile_avg', 0)
        disp2 = stocks2.get('top_quartile_avg', 0) - stocks2.get('bottom_quartile_avg', 0)

        if disp2 > disp1 * 1.3:
            insights.append(f"[green]✓ WIDER STOCK DISPERSION:[/green] Period 2 had better stock selection opportunity (top vs bottom quartile spread: {disp2:.0f}% vs {disp1:.0f}%). Momentum works best when winners and losers are clearly differentiated.")
        elif disp1 > disp2 * 1.3:
            insights.append(f"[red]✗ NARROW DISPERSION:[/red] Period 1 had less differentiation between winners/losers ({disp1:.0f}% spread vs {disp2:.0f}%). Stock picking adds less value.")

        # 5. Drawdowns
        dd_diff = market2.get('max_drawdown', 0) - market1.get('max_drawdown', 0)
        if abs(dd_diff) > 10:
            insights.append(f"[yellow]⚠ DIFFERENT DRAWDOWN PROFILE:[/yellow] Max drawdowns differed significantly ({market1.get('max_drawdown', 0):.1f}% vs {market2.get('max_drawdown', 0):.1f}%). Large drawdowns can hurt momentum (stocks get stopped out at lows).")

        # 6. Period length
        days_diff = (p2_end - p2_start).days - (p1_end - p1_start).days
        if abs(days_diff) > 180:
            insights.append(f"[yellow]⚠ DIFFERENT PERIOD LENGTH:[/yellow] Period 2 is {'longer' if days_diff > 0 else 'shorter'} ({(p2_end - p2_start).days} vs {(p1_end - p1_start).days} days). Longer periods allow compounding but also more exposure to regime changes.")

        # Print insights
        if insights:
            for i, insight in enumerate(insights, 1):
                console.print(f"\n{i}. {insight}")
        else:
            console.print("\n[dim]Market conditions were relatively similar between periods.[/dim]")

        # Key takeaway
        console.print("\n" + "-"*60)
        console.print("[bold]KEY TAKEAWAY:[/bold]")

        if result2.total_return > result1.total_return:
            console.print(f"""
Momentum performed better in Period 2 ({p2_start.year}-{p2_end.year}) because:
• The overall market trend was {'stronger' if market2.get('nifty_return', 0) > market1.get('nifty_return', 0) else 'weaker'}
  (NIFTY: {market2.get('nifty_return', 0):.1f}% vs {market1.get('nifty_return', 0):.1f}%)
• Momentum continuation rate: {stocks2.get('momentum_continuation_rate', 0):.0f}% vs {stocks1.get('momentum_continuation_rate', 0):.0f}%
• Stock dispersion: {disp2:.0f}% vs {disp1:.0f}%

[bold yellow]Momentum strategies work best when:[/bold yellow]
1. The market has a clear upward trend
2. Winners continue to win (momentum persists)
3. There's clear differentiation between strong and weak stocks
4. Volatility is moderate (not too high, not too low)
""")
        else:
            console.print(f"""
Momentum performed better in Period 1 ({p1_start.year}-{p1_end.year}) because:
• Stock dispersion was {'wider' if disp1 > disp2 else 'narrower'} ({disp1:.0f}% vs {disp2:.0f}%)
• Momentum continuation: {stocks1.get('momentum_continuation_rate', 0):.0f}% vs {stocks2.get('momentum_continuation_rate', 0):.0f}%

Period 2 may have suffered from:
• Regime changes (COVID crash + recovery in 2020)
• Higher volatility leading to more stop-outs
• Momentum reversals during corrections
""")

        # Regime analysis if available
        if result1.regime_history is not None and len(result1.regime_history) > 0:
            console.print("\n[bold]Regime Distribution - Period 1:[/bold]")
            regime1 = result1.regime_history['regime'].value_counts()
            for regime, count in regime1.items():
                console.print(f"  {regime}: {count} days ({count/len(result1.regime_history)*100:.1f}%)")

        if result2.regime_history is not None and len(result2.regime_history) > 0:
            console.print("\n[bold]Regime Distribution - Period 2:[/bold]")
            regime2 = result2.regime_history['regime'].value_counts()
            for regime, count in regime2.items():
                console.print(f"  {regime}: {count} days ({count/len(result2.regime_history)*100:.1f}%)")


if __name__ == "__main__":
    main()
