#!/usr/bin/env python3
"""
Fetch historical data from 2014 to end of 2020.
Skips symbols that already have data from 2014.
Validates complete data coverage from 2014 to present.
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from fortress.config import load_config
from fortress.universe import Universe
from fortress.auth import ZerodhaAuth
from fortress.instruments import InstrumentMapper
from fortress.market_data import MarketDataProvider

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


def fetch_in_chunks(market_data, symbol, start_date, end_date, chunk_days=1800):
    """Fetch data in chunks to avoid API limits (max 2000 days per request)."""
    all_dfs = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)

        try:
            df = with_retry(
                lambda s=symbol, st=current_start, en=current_end: market_data.get_historical(
                    s, st, en, interval="day", check_quality=False
                ),
                max_retries=3,
                delay=2.0,
            )

            if df is not None and len(df) > 0:
                all_dfs.append(df)
        except Exception as e:
            console.print(f"[dim]  Chunk {current_start.date()}-{current_end.date()} failed: {str(e)[:40]}[/dim]")

        current_start = current_end + timedelta(days=1)
        time.sleep(0.35)  # Rate limiting between chunks

    if not all_dfs:
        return None

    # Merge all chunks
    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()

    return combined


def merge_with_existing(new_df, cache_file):
    """Merge new data with existing cached data."""
    if not cache_file.exists():
        return new_df

    try:
        existing_df = pd.read_parquet(cache_file)
        existing_df.index = pd.to_datetime(existing_df.index)
        if existing_df.index.tz is not None:
            existing_df.index = existing_df.index.tz_convert(None)

        # Combine old and new data
        combined = pd.concat([new_df, existing_df])
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()
        return combined
    except Exception as e:
        console.print(f"[yellow]Warning merging existing data: {e}[/yellow]")
        return new_df


def validate_data_coverage(cache_dir, cache_meta, required_start="2014-01-01"):
    """Validate data coverage and report gaps."""
    console.print("\n[bold]Validating Data Coverage (2014 to Present)[/bold]")

    required_start_date = datetime.strptime(required_start, "%Y-%m-%d")

    full_coverage = []
    partial_coverage = []
    no_old_data = []

    for symbol, meta in cache_meta.items():
        start_date = datetime.strptime(meta["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(meta["end_date"], "%Y-%m-%d")

        if start_date <= required_start_date + timedelta(days=30):  # Allow 30 day tolerance
            full_coverage.append((symbol, meta["start_date"], meta["end_date"], meta["rows"]))
        elif start_date.year <= 2018:
            partial_coverage.append((symbol, meta["start_date"], meta["end_date"], meta["rows"]))
        else:
            no_old_data.append((symbol, meta["start_date"], meta["end_date"], meta["rows"]))

    # Summary table
    table = Table(title="Data Coverage Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Description")

    table.add_row(
        "[green]Full Coverage[/green]",
        str(len(full_coverage)),
        f"Data from 2014 or earlier"
    )
    table.add_row(
        "[yellow]Partial Coverage[/yellow]",
        str(len(partial_coverage)),
        f"Data from 2015-2018"
    )
    table.add_row(
        "[red]Recent Only[/red]",
        str(len(no_old_data)),
        f"Data from 2019+ (newer stocks/IPOs)"
    )

    console.print(table)

    # Show sample of each category
    if full_coverage:
        console.print(f"\n[green]Sample stocks with full 2014 coverage:[/green]")
        for sym, start, end, rows in sorted(full_coverage, key=lambda x: x[0])[:10]:
            console.print(f"  {sym}: {start} to {end} ({rows} rows)")

    if partial_coverage:
        console.print(f"\n[yellow]Stocks with partial coverage (2015-2018 start):[/yellow]")
        for sym, start, end, rows in sorted(partial_coverage, key=lambda x: x[1])[:10]:
            console.print(f"  {sym}: {start} to {end} ({rows} rows)")

    if no_old_data:
        console.print(f"\n[red]Recent stocks (no 2014 data available - likely IPOs):[/red]")
        for sym, start, end, rows in sorted(no_old_data, key=lambda x: x[1])[:15]:
            console.print(f"  {sym}: {start} to {end} ({rows} rows)")

    return len(full_coverage), len(partial_coverage), len(no_old_data)


def main():
    console.print("[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold blue]")
    console.print("[bold blue]  FORTRESS MOMENTUM - Historical Data Fetch (2014-2020)  [/bold blue]")
    console.print("[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold blue]\n")

    # Load config
    config = load_config("config.yaml")
    universe = Universe(config.paths.universe_file)

    # Setup cache
    cache_dir = Path(config.paths.data_cache)
    cache_dir.mkdir(exist_ok=True)
    cache_meta_file = cache_dir / "cache_meta.json"

    # Load existing cache metadata
    cache_meta = {}
    if cache_meta_file.exists():
        try:
            cache_meta = json.loads(cache_meta_file.read_text())
        except:
            cache_meta = {}

    # Target date range: 2014-01-01 to 2020-12-31
    target_start = datetime(2014, 1, 1)
    target_end = datetime(2020, 12, 31)

    console.print(f"[bold]Target Period:[/bold] {target_start.date()} to {target_end.date()}")
    console.print(f"[dim]Will merge with existing data for complete coverage[/dim]\n")

    # Get all symbols
    all_stocks = universe.get_all_stocks()
    stock_symbols = [s.zerodha_symbol for s in all_stocks]

    # Add benchmarks and indices
    extra_symbols = [
        "NIFTY 50", "NIFTY MIDCAP 100", "INDIA VIX",
        "GOLDBEES", "LIQUIDBEES"
    ]

    all_symbols = list(set(stock_symbols + extra_symbols))
    console.print(f"[bold]Total symbols in universe:[/bold] {len(all_symbols)}")

    # Check what needs fetching
    symbols_to_fetch = []
    already_have_2014 = []

    for symbol in all_symbols:
        if symbol in cache_meta:
            cached_start = cache_meta[symbol].get("start_date", "")
            # Skip if already have data from 2014 (or earlier)
            if cached_start <= "2014-06-01":
                already_have_2014.append(symbol)
                continue
        symbols_to_fetch.append(symbol)

    console.print(f"[green]Already have 2014 data:[/green] {len(already_have_2014)}")
    console.print(f"[yellow]Need to fetch 2014-2020:[/yellow] {len(symbols_to_fetch)}")

    if not symbols_to_fetch:
        console.print("\n[green]All symbols already have data from 2014![/green]")
        validate_data_coverage(cache_dir, cache_meta)
        return

    # Authenticate
    console.print("\n[yellow]Authenticating with Zerodha...[/yellow]")
    auth = ZerodhaAuth(config.zerodha.api_key, config.zerodha.api_secret)

    if auth.is_authenticated():
        console.print("[green]Using cached access token[/green]")
        kite = auth.get_kite()
    else:
        console.print("[yellow]Opening browser for login...[/yellow]")
        kite = auth.login_interactive()

    console.print("[green]✓ Authenticated successfully[/green]\n")

    # Setup instrument mapper
    mapper = InstrumentMapper(kite, universe)
    market_data = MarketDataProvider(kite, mapper)

    # Fetch data
    console.print(f"[bold]Fetching {len(symbols_to_fetch)} symbols...[/bold]")
    console.print("[dim]This will take 20-40 minutes due to API rate limits.[/dim]")
    console.print("[dim]Data is fetched in chunks of ~1800 days to respect 2000-day limit.[/dim]\n")

    failed_symbols = []
    success_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Fetching historical data...", total=len(symbols_to_fetch))

        for i, symbol in enumerate(symbols_to_fetch):
            progress.update(task, description=f"[cyan]{symbol[:20]:<20} ({i+1}/{len(symbols_to_fetch)})")

            try:
                # Fetch 2014-2020 data in chunks
                df = fetch_in_chunks(market_data, symbol, target_start, target_end)

                if df is not None and len(df) > 0:
                    # Normalize timezone
                    if df.index.tz is not None:
                        df.index = df.index.tz_convert(None)

                    # Merge with existing data (2021+ data we already have)
                    cache_file = cache_dir / f"{symbol.replace(' ', '_')}.parquet"
                    merged_df = merge_with_existing(df, cache_file)

                    # Save merged data
                    merged_df.to_parquet(cache_file)

                    # Update metadata
                    cache_meta[symbol] = {
                        "start_date": merged_df.index[0].strftime("%Y-%m-%d"),
                        "end_date": merged_df.index[-1].strftime("%Y-%m-%d"),
                        "rows": len(merged_df),
                    }
                    success_count += 1
                else:
                    # No data available for 2014-2020 (probably newer stock)
                    failed_symbols.append((symbol, "No data for 2014-2020 period"))

            except Exception as e:
                failed_symbols.append((symbol, str(e)[:60]))

            progress.advance(task)

            # Save metadata periodically (every 20 symbols)
            if (i + 1) % 20 == 0:
                cache_meta_file.write_text(json.dumps(cache_meta, indent=2))

    # Final save
    cache_meta_file.write_text(json.dumps(cache_meta, indent=2))

    # Results
    console.print(f"\n[green]✓ Successfully fetched/merged: {success_count} symbols[/green]")

    if failed_symbols:
        console.print(f"\n[yellow]Could not fetch 2014-2020 data for {len(failed_symbols)} symbols:[/yellow]")
        console.print("[dim](These are likely newer stocks/IPOs that didn't exist in 2014)[/dim]")
        for sym, err in sorted(failed_symbols)[:20]:
            console.print(f"  [dim]{sym}: {err}[/dim]")
        if len(failed_symbols) > 20:
            console.print(f"  [dim]... and {len(failed_symbols) - 20} more[/dim]")

    # Validate complete coverage
    # Reload metadata
    cache_meta = json.loads(cache_meta_file.read_text())
    validate_data_coverage(cache_dir, cache_meta)

    console.print("\n[bold green]Done! You now have historical data from 2014 for backtesting.[/bold green]")


if __name__ == "__main__":
    main()
