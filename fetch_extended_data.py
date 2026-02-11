#!/usr/bin/env python3
"""
Fetch extended historical data from Jan 2020 onwards.
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

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

        df = with_retry(
            lambda s=symbol, st=current_start, en=current_end: market_data.get_historical(
                s, st, en, interval="day", check_quality=False
            ),
            max_retries=3,
            delay=2.0,
        )

        if df is not None and len(df) > 0:
            all_dfs.append(df)

        current_start = current_end + timedelta(days=1)
        time.sleep(0.35)  # Rate limiting between chunks

    if not all_dfs:
        return None

    # Merge all chunks
    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()

    return combined


def main():
    console.print("[bold blue]FORTRESS MOMENTUM - Extended Data Fetch[/bold blue]")
    console.print("Fetching data from January 2020 onwards\n")

    # Load config
    config = load_config("config.yaml")
    universe = Universe(config.paths.universe_file)

    # Authenticate
    console.print("[yellow]Authenticating with Zerodha...[/yellow]")
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

    # Calculate date range - from Jan 2019 to now (need 2019 for 12-month lookback in 2020)
    start_date = datetime(2019, 1, 1)
    end_date = datetime.now() - timedelta(days=1)
    while end_date.weekday() >= 5:
        end_date -= timedelta(days=1)

    console.print(f"[bold]Data Fetch Configuration:[/bold]")
    console.print(f"  Period: {start_date.date()} to {end_date.date()}")
    console.print(f"  Duration: ~{(end_date - start_date).days // 30} months")

    # Get all symbols
    all_stocks = universe.get_all_stocks()
    stock_symbols = [s.zerodha_symbol for s in all_stocks]

    # Add benchmarks and defensive instruments
    extra_symbols = [
        "NIFTY 50", "NIFTY MIDCAP 100", "INDIA VIX",
        "GOLDBEES", "LIQUIDCASE", "LIQUIDBEES"
    ]

    all_symbols = list(set(stock_symbols + extra_symbols))
    console.print(f"  Total symbols: {len(all_symbols)}")

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

    # Check what needs fetching (force re-fetch for extended date range)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    symbols_to_fetch = []
    for symbol in all_symbols:
        cache_file = cache_dir / f"{symbol.replace(' ', '_')}.parquet"

        # Check if cache covers from 2020
        if cache_file.exists() and symbol in cache_meta:
            cached_start = cache_meta[symbol].get("start_date", "")
            if cached_start <= start_str:
                continue  # Already have data from 2020

        symbols_to_fetch.append(symbol)

    console.print(f"  Already have 2020 data: {len(all_symbols) - len(symbols_to_fetch)}")
    console.print(f"  Need to fetch: {len(symbols_to_fetch)}")

    if not symbols_to_fetch:
        console.print("\n[green]All data already fetched from 2020![/green]")
        return

    # Fetch data
    console.print(f"\n[yellow]Fetching {len(symbols_to_fetch)} symbols from Zerodha...[/yellow]")
    console.print("[dim]This may take 15-30 minutes. Data will be cached.[/dim]\n")

    market_data = MarketDataProvider(kite, mapper)
    failed_symbols = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Fetching data...", total=len(symbols_to_fetch))

        for symbol in symbols_to_fetch:
            progress.update(task, description=f"[cyan]{symbol[:20]:<20}")

            try:
                # Fetch in chunks to avoid 2000-day API limit
                df = fetch_in_chunks(market_data, symbol, start_date, end_date)

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

            except Exception as e:
                failed_symbols.append((symbol, str(e)[:50]))

            progress.advance(task)
            time.sleep(0.35)  # Rate limiting

    # Save cache metadata
    cache_meta_file.write_text(json.dumps(cache_meta, indent=2))

    console.print(f"\n[green]✓ Fetched {len(symbols_to_fetch) - len(failed_symbols)} symbols[/green]")

    if failed_symbols:
        console.print(f"[yellow]Failed to fetch {len(failed_symbols)} symbols:[/yellow]")
        for sym, err in failed_symbols[:10]:
            console.print(f"  {sym}: {err}")

    # Verify data range
    console.print("\n[bold]Verifying data range:[/bold]")
    for symbol in ["NIFTY_50", "RELIANCE", "HDFCBANK"]:
        cache_file = cache_dir / f"{symbol}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            console.print(f"  {symbol}: {df.index.min().date()} to {df.index.max().date()}")


if __name__ == "__main__":
    main()
