#!/usr/bin/env python3
"""
Fetch 48 months of historical data and run comprehensive backtests.

This script:
1. Authenticates with Zerodha
2. Fetches historical data for all stocks and indices
3. Runs backtests comparing momentum vs sector rotation
4. Analyzes results across different market periods
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

sys.path.insert(0, str(Path(__file__).parent))

from fortress.config import load_config
from fortress.universe import Universe
from fortress.auth import ZerodhaAuth
from fortress.instruments import InstrumentMapper
from fortress.market_data import MarketDataProvider
from fortress.backtest import BacktestConfig, BacktestEngine, BacktestResult

console = Console()


def with_retry(func, max_retries=3, delay=2.0):
    """Retry a function on failure."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise e


def fetch_historical_data(
    kite,
    universe: Universe,
    mapper: InstrumentMapper,
    cache_dir: Path,
    months: int = 48,
) -> dict:
    """Fetch historical data for all symbols."""

    # Calculate date range
    end_date = datetime.now() - timedelta(days=1)
    while end_date.weekday() >= 5:
        end_date -= timedelta(days=1)

    start_date = end_date - timedelta(days=months * 30)
    data_start = start_date - timedelta(days=30)  # Extra buffer

    console.print(f"\n[bold]Data Fetch Configuration:[/bold]")
    console.print(f"  Period: {start_date.date()} to {end_date.date()}")
    console.print(f"  Duration: {months} months (~{months * 30} days)")

    # Get all symbols
    all_stocks = universe.get_all_stocks()
    stock_symbols = [s.zerodha_symbol for s in all_stocks]

    # Get sector indices
    valid_sectors = universe.get_valid_sectors()
    index_symbols = []
    for sector in valid_sectors:
        idx = universe.get_sector_index(sector)
        if idx and idx.symbol:
            index_symbols.append(idx.symbol)

    # Add benchmark
    benchmark = universe.benchmark
    if benchmark.symbol not in index_symbols:
        index_symbols.append(benchmark.symbol)

    all_symbols = list(set(stock_symbols + index_symbols))
    console.print(f"  Total symbols: {len(all_symbols)}")

    # Setup cache
    cache_dir.mkdir(exist_ok=True)
    cache_meta_file = cache_dir / "cache_meta.json"

    # Load existing cache metadata
    cache_meta = {}
    if cache_meta_file.exists():
        try:
            cache_meta = json.loads(cache_meta_file.read_text())
        except:
            cache_meta = {}

    # Check what needs fetching
    end_str = end_date.strftime("%Y-%m-%d")
    start_str = data_start.strftime("%Y-%m-%d")

    symbols_to_fetch = []
    for symbol in all_symbols:
        cache_file = cache_dir / f"{symbol.replace(' ', '_')}.parquet"

        # Check if cache is valid and covers our range
        if cache_file.exists() and symbol in cache_meta:
            cached_start = cache_meta[symbol].get("start_date", "")
            cached_end = cache_meta[symbol].get("end_date", "")

            # Re-fetch if cache doesn't cover our date range
            if cached_start <= start_str and cached_end >= end_str:
                continue

        symbols_to_fetch.append(symbol)

    console.print(f"  Already cached: {len(all_symbols) - len(symbols_to_fetch)}")
    console.print(f"  Need to fetch: {len(symbols_to_fetch)}")

    if not symbols_to_fetch:
        console.print("\n[green]All data already cached![/green]")
        # Load from cache
        historical_data = {}
        for symbol in all_symbols:
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

    # Fetch missing data
    console.print(f"\n[yellow]Fetching {len(symbols_to_fetch)} symbols from Zerodha...[/yellow]")
    console.print("[dim]This may take 10-20 minutes. Data will be cached for future use.[/dim]\n")

    market_data = MarketDataProvider(kite, mapper)
    historical_data = {}
    failed_symbols = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Fetching data...", total=len(symbols_to_fetch))

        for i, symbol in enumerate(symbols_to_fetch):
            progress.update(task, description=f"[cyan]{symbol[:20]:<20}")

            try:
                df = with_retry(
                    lambda s=symbol: market_data.get_historical(
                        s, data_start, end_date, interval="day", check_quality=False
                    ),
                    max_retries=3,
                    delay=2.0,
                )

                if df is not None and len(df) > 0:
                    # Normalize timezone
                    if df.index.tz is not None:
                        df.index = df.index.tz_convert(None)

                    # Save to cache
                    cache_file = cache_dir / f"{symbol.replace(' ', '_')}.parquet"
                    df.to_parquet(cache_file)

                    # Update metadata
                    cache_meta[symbol] = {
                        "start_date": df.index[0].strftime("%Y-%m-%d"),
                        "end_date": df.index[-1].strftime("%Y-%m-%d"),
                        "rows": len(df),
                    }

                    historical_data[symbol] = df

            except Exception as e:
                failed_symbols.append((symbol, str(e)[:50]))

            progress.advance(task)

            # Rate limiting
            time.sleep(0.35)

    # Save cache metadata
    cache_meta_file.write_text(json.dumps(cache_meta, indent=2))

    if failed_symbols:
        console.print(f"\n[yellow]Failed to fetch {len(failed_symbols)} symbols:[/yellow]")
        for sym, err in failed_symbols[:10]:
            console.print(f"  [dim]{sym}: {err}[/dim]")
        if len(failed_symbols) > 10:
            console.print(f"  [dim]... and {len(failed_symbols) - 10} more[/dim]")

    # Load all cached data
    console.print(f"\n[green]Loading all cached data...[/green]")
    for symbol in all_symbols:
        if symbol in historical_data:
            continue
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

    console.print(f"[green]Loaded {len(historical_data)} symbols total[/green]")
    return historical_data


def run_backtest(
    universe: Universe,
    historical_data: dict,
    config,
    start_date: datetime,
    end_date: datetime,
    strategy_mode: str,
) -> BacktestResult:
    """Run a single backtest."""
    bt_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=config.portfolio.initial_capital,
        rebalance_days=5,
        top_sectors=config.rotation.top_sectors,
        stocks_per_sector=config.rotation.stocks_per_sector,
        transaction_cost=config.costs.transaction_cost,
        strategy_mode=strategy_mode,
        target_positions=config.position_sizing.target_positions,
        min_positions=config.position_sizing.min_positions,
        use_stop_loss=True,
        initial_stop_loss=config.risk.initial_stop_loss,
        trailing_stop=config.risk.trailing_stop,
        trailing_activation=config.risk.trailing_activation,
        min_score_percentile=config.pure_momentum.min_score_percentile,
        min_52w_high_prox=config.pure_momentum.min_52w_high_prox,
    )

    engine = BacktestEngine(
        universe=universe,
        historical_data=historical_data,
        config=bt_config,
        app_config=config,
    )

    return engine.run()


def analyze_results(results: list):
    """Analyze and display backtest results."""

    console.print("\n")
    console.print(Panel("[bold]COMPREHENSIVE BACKTEST RESULTS[/bold]", style="green"))

    # Results table
    table = Table(title="Strategy Performance Across Market Periods")
    table.add_column("Period", style="cyan")
    table.add_column("Strategy", style="bold")
    table.add_column("Return", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("Monthly", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Trades", justify="right")

    for r in results:
        ret_color = "green" if r["return"] > 0 else "red"
        sharpe_color = "green" if r["sharpe"] > 1.0 else "yellow" if r["sharpe"] > 0 else "red"

        # Calculate monthly return from CAGR
        monthly = r["cagr"] / 12 if r["cagr"] else 0

        table.add_row(
            r["period"],
            r["strategy"],
            f"[{ret_color}]{r['return']:+.1%}[/{ret_color}]",
            f"{r['cagr']:.1%}",
            f"{monthly:.1%}",
            f"[{sharpe_color}]{r['sharpe']:.2f}[/{sharpe_color}]",
            f"{r['max_dd']:.1%}",
            f"{r['win_rate']:.0%}",
            str(r["trades"]),
        )

    console.print(table)

    # Summary by strategy
    console.print("\n[bold]Summary Statistics:[/bold]")

    for strategy in ["Momentum (NMS)", "Sector (RRV)", "Hybrid"]:
        strat_results = [r for r in results if r["strategy"] == strategy]
        if strat_results:
            avg_ret = sum(r["return"] for r in strat_results) / len(strat_results)
            avg_cagr = sum(r["cagr"] for r in strat_results) / len(strat_results)
            avg_sharpe = sum(r["sharpe"] for r in strat_results) / len(strat_results)
            avg_dd = sum(r["max_dd"] for r in strat_results) / len(strat_results)

            console.print(f"\n  [cyan]{strategy}:[/cyan]")
            console.print(f"    Avg Return: {avg_ret:+.1%}")
            console.print(f"    Avg CAGR: {avg_cagr:.1%}")
            console.print(f"    Avg Monthly: {avg_cagr/12:.1%}")
            console.print(f"    Avg Sharpe: {avg_sharpe:.2f}")
            console.print(f"    Avg Max DD: {avg_dd:.1%}")


def main():
    console.print(Panel("[bold cyan]FORTRESS MOMENTUM - 48 Month Backtest[/bold cyan]"))

    # Load config
    config = load_config()
    universe = Universe(config.paths.universe_file)
    cache_dir = Path(config.paths.data_cache)

    # Authenticate
    console.print("\n[bold]Step 1: Authentication[/bold]")
    auth = ZerodhaAuth(config.zerodha.api_key, config.zerodha.api_secret)

    # Check if already authenticated (cached token)
    if auth.is_authenticated():
        console.print("[green]✓ Using cached access token[/green]")
        kite = auth.get_kite()
    else:
        console.print("[yellow]No cached token found. Starting interactive login...[/yellow]")
        try:
            kite = auth.login_interactive()
            console.print("[green]✓ Logged in successfully[/green]")
        except Exception as e:
            console.print(f"[red]Authentication failed: {e}[/red]")
            return

    # Load instruments
    console.print("\n[bold]Step 2: Loading Instruments[/bold]")
    mapper = InstrumentMapper(kite, universe)
    with console.status("[bold green]Loading instrument tokens..."):
        mapper.load_instruments()
    console.print("[green]✓ Instruments loaded[/green]")

    # Fetch data
    console.print("\n[bold]Step 3: Fetching Historical Data (48 months)[/bold]")
    historical_data = fetch_historical_data(kite, universe, mapper, cache_dir, months=48)

    if len(historical_data) < 50:
        console.print("[red]Insufficient data for backtest[/red]")
        return

    # Determine data range
    earliest = min(df.index[0] for df in historical_data.values() if len(df) > 0)
    latest = max(df.index[-1] for df in historical_data.values() if len(df) > 0)

    console.print(f"\n[bold]Data Range:[/bold] {earliest.date()} to {latest.date()}")

    # For NMS, need 280 days lookback
    usable_start = earliest + pd.Timedelta(days=280)

    console.print(f"[bold]Usable Start (after lookback):[/bold] {usable_start.date()}")

    # Define test periods
    console.print("\n[bold]Step 4: Running Backtests[/bold]")

    test_periods = []

    # Try to create meaningful test periods
    # Bull market periods (if data available)
    if usable_start <= pd.Timestamp("2021-06-01"):
        test_periods.append(("2021 Bull Run (Jun-Dec)", datetime(2021, 6, 1), datetime(2021, 12, 31)))

    if usable_start <= pd.Timestamp("2022-01-01"):
        test_periods.append(("2022 Correction (Jan-Jun)", datetime(2022, 1, 1), datetime(2022, 6, 30)))

    if usable_start <= pd.Timestamp("2022-07-01"):
        test_periods.append(("2022 Recovery (Jul-Dec)", datetime(2022, 7, 1), datetime(2022, 12, 31)))

    if usable_start <= pd.Timestamp("2023-01-01"):
        test_periods.append(("2023 Bull (Jan-Sep)", datetime(2023, 1, 1), datetime(2023, 9, 30)))

    if usable_start <= pd.Timestamp("2023-10-01"):
        test_periods.append(("2023 Q4 (Oct-Dec)", datetime(2023, 10, 1), datetime(2023, 12, 31)))

    if usable_start <= pd.Timestamp("2024-01-01"):
        test_periods.append(("2024 H1 (Jan-Jun)", datetime(2024, 1, 1), datetime(2024, 6, 30)))

    if usable_start <= pd.Timestamp("2024-07-01"):
        test_periods.append(("2024 Correction (Jul-Dec)", datetime(2024, 7, 1), datetime(2024, 12, 31)))

    # Full period
    test_periods.append(("Full Period", usable_start.to_pydatetime(), latest.to_pydatetime()))

    results = []

    for period_name, start_date, end_date in test_periods:
        # Adjust end date if beyond data
        if end_date > latest.to_pydatetime():
            end_date = latest.to_pydatetime()

        # Skip if start is before usable
        if start_date < usable_start.to_pydatetime():
            start_date = usable_start.to_pydatetime()

        # Skip if period is too short
        if (end_date - start_date).days < 30:
            continue

        console.print(f"\n  [cyan]{period_name}[/cyan] ({start_date.date()} to {end_date.date()})")

        # Momentum strategy
        try:
            mom_result = run_backtest(
                universe, historical_data, config,
                start_date, end_date, "pure_momentum"
            )
            console.print(f"    Momentum: {mom_result.total_return:+.1%} (Sharpe: {mom_result.sharpe_ratio:.2f})")
            results.append({
                "period": period_name,
                "strategy": "Momentum (NMS)",
                "return": mom_result.total_return,
                "cagr": mom_result.cagr,
                "sharpe": mom_result.sharpe_ratio,
                "max_dd": mom_result.max_drawdown,
                "win_rate": mom_result.win_rate,
                "trades": mom_result.total_trades,
            })
        except Exception as e:
            console.print(f"    [red]Momentum error: {str(e)[:50]}[/red]")

        # Sector rotation
        try:
            sec_result = run_backtest(
                universe, historical_data, config,
                start_date, end_date, "sector_rotation"
            )
            console.print(f"    Sector:   {sec_result.total_return:+.1%} (Sharpe: {sec_result.sharpe_ratio:.2f})")
            results.append({
                "period": period_name,
                "strategy": "Sector (RRV)",
                "return": sec_result.total_return,
                "cagr": sec_result.cagr,
                "sharpe": sec_result.sharpe_ratio,
                "max_dd": sec_result.max_drawdown,
                "win_rate": sec_result.win_rate,
                "trades": sec_result.total_trades,
            })
        except Exception as e:
            console.print(f"    [red]Sector error: {str(e)[:50]}[/red]")

        # Hybrid strategy (momentum in bull/normal, sector in caution/defensive)
        try:
            hyb_result = run_backtest(
                universe, historical_data, config,
                start_date, end_date, "hybrid"
            )
            regime_info = ""
            if hyb_result.regime_history is not None and len(hyb_result.regime_history) > 0:
                regime_counts = hyb_result.regime_history["regime"].value_counts()
                regime_info = f" [{', '.join(f'{k}:{v}' for k, v in regime_counts.items())}]"
            console.print(f"    Hybrid:   {hyb_result.total_return:+.1%} (Sharpe: {hyb_result.sharpe_ratio:.2f}){regime_info}")
            results.append({
                "period": period_name,
                "strategy": "Hybrid",
                "return": hyb_result.total_return,
                "cagr": hyb_result.cagr,
                "sharpe": hyb_result.sharpe_ratio,
                "max_dd": hyb_result.max_drawdown,
                "win_rate": hyb_result.win_rate,
                "trades": hyb_result.total_trades,
            })
        except Exception as e:
            console.print(f"    [red]Hybrid error: {str(e)[:50]}[/red]")

    # Analyze results
    if results:
        analyze_results(results)

        # Save results
        results_df = pd.DataFrame(results)
        results_file = f"backtest_48mo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_file, index=False)
        console.print(f"\n[dim]Results saved to {results_file}[/dim]")
    else:
        console.print("[red]No backtest results generated[/red]")

    # Final recommendations
    console.print("\n[bold]Key Takeaways:[/bold]")
    console.print("  1. Momentum excels in bull markets, may underperform in corrections")
    console.print("  2. Sector rotation provides more defensive positioning in bear markets")
    console.print("  3. Hybrid automatically switches: momentum (bull/normal) → sector (caution/defensive)")
    console.print("  4. Look at Sharpe ratio for risk-adjusted returns (>1.0 is good)")
    console.print("  5. Max drawdown matters - lower is better for capital preservation")


if __name__ == "__main__":
    main()
