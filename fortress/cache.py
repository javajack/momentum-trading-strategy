"""
Reusable Cache Manager for FORTRESS MOMENTUM.

Provides:
- Fast parallel cache loading
- Concurrent incremental updates with rate limiting
- Session-level memory caching (load once per session)
- Works everywhere (scan, rebalance, backtest)
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# Module-level session cache - persists across CacheManager instances
_SESSION_CACHE: Dict[str, pd.DataFrame] = {}
_SESSION_LAST_DATES: Dict[str, date] = {}
_SESSION_CACHE_LOADED: bool = False
_SESSION_LOCK = Lock()


class CacheManager:
    """
    Centralized cache manager for historical market data.

    Features:
    - Parallel parquet file loading (3-5x faster)
    - Concurrent API updates with rate limiting
    - Session-level memory cache (avoids re-reading files)
    - Incremental updates only fetch missing days

    Usage:
        cache = CacheManager(config, universe, market_data)

        # Fast load (no updates)
        data = cache.load()

        # Load and update if stale
        data = cache.load_and_update()
    """

    # Concurrency settings
    LOAD_WORKERS = 8       # Parallel file reads
    UPDATE_WORKERS = 4     # Concurrent API calls
    API_RATE_LIMIT = 3.0   # Max API calls per second

    def __init__(self, config, universe, market_data=None):
        """
        Initialize cache manager.

        Args:
            config: App config with paths.data_cache
            universe: Universe with get_all_stocks()
            market_data: MarketDataProvider for fetching (optional for load-only)
        """
        self.config = config
        self.universe = universe
        self.market_data = market_data
        self.cache_dir = Path(config.paths.data_cache)
        self.cache_dir.mkdir(exist_ok=True)

        # Build symbol list
        all_stocks = universe.get_all_stocks()
        self._symbols = [s.zerodha_symbol for s in all_stocks]
        self._symbols.extend(["NIFTY 50", "NIFTY MIDCAP 100", "NIFTY SMLCAP 100", "INDIA VIX"])
        self._symbols.extend([config.regime.gold_symbol, config.regime.cash_symbol])

        # Rate limiting state
        self._api_timestamps: List[float] = []
        self._api_lock = Lock()

        # Manifest file tracks last update attempt (handles market holidays)
        self._manifest_file = self.cache_dir / ".cache_manifest.json"

    def _load_single_file(self, symbol: str) -> Optional[Tuple[str, pd.DataFrame, date]]:
        """Load a single parquet file. Returns (symbol, df, last_date) or None."""
        cache_file = self.cache_dir / f"{symbol.replace(' ', '_')}.parquet"
        if not cache_file.exists():
            return None
        try:
            df = pd.read_parquet(cache_file)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            last_date = df.index.max().date()
            return (symbol, df, last_date)
        except Exception:
            return None

    def load(self, silent: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load all cached data (no network calls).

        Uses parallel file reading for 3-5x speedup.
        Session-level caching avoids re-reading files.

        Args:
            silent: If True, don't print status messages

        Returns:
            Dict of symbol -> DataFrame
        """
        global _SESSION_CACHE, _SESSION_LAST_DATES, _SESSION_CACHE_LOADED

        # Return session cache if already loaded
        with _SESSION_LOCK:
            if _SESSION_CACHE_LOADED and _SESSION_CACHE:
                if not silent:
                    console.print(f"[dim]Using session cache ({len(_SESSION_CACHE)} symbols)[/dim]")
                return _SESSION_CACHE

        # Parallel load from disk
        loaded_data: Dict[str, pd.DataFrame] = {}
        loaded_dates: Dict[str, date] = {}

        with ThreadPoolExecutor(max_workers=self.LOAD_WORKERS) as executor:
            futures = {executor.submit(self._load_single_file, symbol): symbol
                      for symbol in self._symbols}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    symbol, df, last_date = result
                    loaded_data[symbol] = df
                    loaded_dates[symbol] = last_date

        # Update session cache
        with _SESSION_LOCK:
            _SESSION_CACHE = loaded_data
            _SESSION_LAST_DATES = loaded_dates
            _SESSION_CACHE_LOADED = True

        if not silent and loaded_data:
            console.print(f"[dim]Loaded {len(loaded_data)} symbols from cache[/dim]")

        return loaded_data

    @property
    def _data(self) -> Dict[str, pd.DataFrame]:
        """Access session cache data."""
        global _SESSION_CACHE
        return _SESSION_CACHE

    @property
    def _last_dates(self) -> Dict[str, date]:
        """Access session cache dates."""
        global _SESSION_LAST_DATES
        return _SESSION_LAST_DATES

    def get_target_date(self) -> date:
        """Get T-1 (last trading day)."""
        today = datetime.now().date()
        target = today - timedelta(days=1)
        while target.weekday() >= 5:  # Skip weekends
            target -= timedelta(days=1)
        return target

    def get_stale_symbols(self) -> List[Tuple[str, Optional[date]]]:
        """
        Get list of symbols that need updating.

        Returns:
            List of (symbol, last_cached_date) for stale symbols
        """
        global _SESSION_CACHE_LOADED

        if not _SESSION_CACHE_LOADED:
            self.load(silent=True)

        target = self.get_target_date()
        stale = []

        for symbol in self._symbols:
            last_date = self._last_dates.get(symbol)
            if last_date is None:
                # Not in cache at all - needs full fetch
                stale.append((symbol, None))
            elif last_date < target:
                stale.append((symbol, last_date))

        return stale

    def is_current(self) -> bool:
        """Check if cache is current (no stale symbols)."""
        return len(self.get_stale_symbols()) == 0

    def _was_already_attempted_today(self) -> bool:
        """Check if we already attempted an update for the current target date today."""
        try:
            if not self._manifest_file.exists():
                return False
            meta = json.loads(self._manifest_file.read_text())
            return (
                meta.get("target") == str(self.get_target_date())
                and meta.get("date") == str(datetime.now().date())
            )
        except Exception:
            return False

    def _save_update_manifest(self, target: date, updated_count: int, stale_count: int):
        """Save manifest recording this update attempt."""
        try:
            self._manifest_file.write_text(json.dumps({
                "target": str(target),
                "date": str(datetime.now().date()),
                "updated": updated_count,
                "stale": stale_count,
            }))
        except Exception:
            pass

    def _rate_limit_wait(self):
        """Wait to respect API rate limit (3 calls/second)."""
        with self._api_lock:
            now = time.time()
            # Remove timestamps older than 1 second
            self._api_timestamps = [t for t in self._api_timestamps if now - t < 1.0]

            if len(self._api_timestamps) >= self.API_RATE_LIMIT:
                # Need to wait
                oldest = self._api_timestamps[0]
                wait_time = 1.0 - (now - oldest) + 0.05  # Small buffer
                if wait_time > 0:
                    time.sleep(wait_time)

            self._api_timestamps.append(time.time())

    def _fetch_incremental(
        self,
        symbol: str,
        last_date: date,
        to_date: datetime,
    ) -> Optional[Tuple[str, pd.DataFrame]]:
        """Fetch incremental data for a symbol. Returns (symbol, merged_df) or None."""
        global _SESSION_CACHE, _SESSION_LAST_DATES

        self._rate_limit_wait()

        try:
            from_dt = datetime.combine(last_date + timedelta(days=1), datetime.min.time())
            df = self.market_data.get_historical(
                symbol, from_dt, to_date, interval="day", quality_level="warn"
            )
            if df is None or len(df) == 0:
                return None

            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)

            # Merge with existing data
            cache_file = self.cache_dir / f"{symbol.replace(' ', '_')}.parquet"
            old_df = _SESSION_CACHE.get(symbol, pd.DataFrame())

            if not old_df.empty:
                merged = pd.concat([old_df[~old_df.index.isin(df.index)], df])
                merged = merged.sort_index()
            else:
                merged = df

            # Save to disk
            merged.to_parquet(cache_file)

            return (symbol, merged)
        except Exception:
            return None

    def _fetch_full(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
    ) -> Optional[Tuple[str, pd.DataFrame]]:
        """Fetch full history for a symbol. Returns (symbol, df) or None."""
        self._rate_limit_wait()

        try:
            df = self.market_data.get_historical(
                symbol, from_date, to_date, interval="day", quality_level="warn"
            )
            if df is None or len(df) == 0:
                return None

            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)

            # Save to disk
            cache_file = self.cache_dir / f"{symbol.replace(' ', '_')}.parquet"
            df.to_parquet(cache_file)

            return (symbol, df)
        except Exception:
            return None

    def update(self, lookback_days: int = 400) -> Dict[str, pd.DataFrame]:
        """
        Update stale symbols with concurrent fetching.

        Args:
            lookback_days: Days of history for full fetches

        Returns:
            Updated data dict
        """
        global _SESSION_CACHE, _SESSION_LAST_DATES

        if self.market_data is None:
            console.print("[yellow]Cannot update - not logged in. Please login first (Option 1).[/yellow]")
            return self._data

        # Load existing cache first
        if not _SESSION_CACHE_LOADED:
            self.load(silent=True)

        stale = self.get_stale_symbols()
        if not stale:
            console.print(f"[dim]Cache is current ({len(self._data)} symbols)[/dim]")
            return self._data

        # Separate incremental vs full fetches
        incremental = [(s, d) for s, d in stale if d is not None]
        full_fetch = [s for s, d in stale if d is None]

        total = len(stale)
        # Always fetch up to T-1 (last completed trading day) to avoid live/incomplete data
        target = self.get_target_date()
        to_date = datetime.combine(target, datetime.max.time())
        from_date_full = to_date - timedelta(days=lookback_days)

        updated_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Syncing...", total=total)

            # Process incremental updates with concurrency
            if incremental:
                with ThreadPoolExecutor(max_workers=self.UPDATE_WORKERS) as executor:
                    futures = {
                        executor.submit(
                            self._fetch_incremental, symbol, last_date, to_date
                        ): symbol
                        for symbol, last_date in incremental
                    }

                    for future in as_completed(futures):
                        symbol = futures[future]
                        progress.update(task, description=f"[cyan]Updating {symbol}...")
                        result = future.result()
                        if result:
                            sym, merged_df = result
                            with _SESSION_LOCK:
                                _SESSION_CACHE[sym] = merged_df
                                _SESSION_LAST_DATES[sym] = merged_df.index.max().date()
                            updated_count += 1
                        progress.update(task, advance=1)

            # Process full fetches with concurrency
            if full_fetch:
                with ThreadPoolExecutor(max_workers=self.UPDATE_WORKERS) as executor:
                    futures = {
                        executor.submit(
                            self._fetch_full, symbol, from_date_full, to_date
                        ): symbol
                        for symbol in full_fetch
                    }

                    for future in as_completed(futures):
                        symbol = futures[future]
                        progress.update(task, description=f"[cyan]Fetching {symbol}...")
                        result = future.result()
                        if result:
                            sym, df = result
                            with _SESSION_LOCK:
                                _SESSION_CACHE[sym] = df
                                _SESSION_LAST_DATES[sym] = df.index.max().date()
                            updated_count += 1
                        progress.update(task, advance=1)

        # Save manifest so subsequent runs (same day, same target) skip the fetch
        self._save_update_manifest(target, updated_count, total)

        # Mark unfetched symbols as current in session cache (holiday / no new data)
        # so within-session subsequent calls don't re-attempt
        if updated_count < total:
            with _SESSION_LOCK:
                for symbol, _ in stale:
                    if _SESSION_LAST_DATES.get(symbol, date.min) < target:
                        _SESSION_LAST_DATES[symbol] = target

        console.print(f"[dim]Cache updated ({len(self._data)} symbols, {updated_count} refreshed)[/dim]")
        return self._data

    def load_and_update(self, lookback_days: int = 400) -> Dict[str, pd.DataFrame]:
        """
        Load cache and update if stale.

        Args:
            lookback_days: Days of history for full fetches

        Returns:
            Current data dict
        """
        self.load(silent=True)

        if self.is_current():
            console.print(f"[dim]Cache is current ({len(self._data)} symbols)[/dim]")
            return self._data

        # Skip re-fetching if we already tried today for the same target
        # (handles market holidays where API returns no new data)
        if self._was_already_attempted_today():
            console.print(f"[dim]Cache is current ({len(self._data)} symbols, already checked today)[/dim]")
            return self._data

        return self.update(lookback_days)

    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        """Get cached data (load if empty)."""
        if not _SESSION_CACHE_LOADED:
            self.load(silent=True)
        return self._data

    def backfill_history(self, required_start: date) -> int:
        """
        Extend cached data backwards so every symbol covers *required_start*.

        For each symbol whose earliest cached date is later than
        *required_start*, fetch the gap from the API and prepend it.

        Args:
            required_start: The earliest date the cache must cover.

        Returns:
            Number of symbols that were backfilled.
        """
        global _SESSION_CACHE, _SESSION_CACHE_LOADED

        if self.market_data is None:
            console.print(
                "[yellow]Cannot backfill — not logged in. "
                "Please login first (Option 1).[/yellow]"
            )
            return 0

        if not _SESSION_CACHE_LOADED:
            self.load(silent=True)

        required_ts = pd.Timestamp(required_start)
        to_backfill: list[str] = []

        for symbol in self._symbols:
            df = _SESSION_CACHE.get(symbol)
            if df is None or df.empty:
                continue  # no existing cache — leave for full-fetch path
            earliest = df.index.min()
            if earliest > required_ts:
                to_backfill.append(symbol)

        if not to_backfill:
            console.print(
                f"[dim]Cache already covers {required_start} "
                f"({len(_SESSION_CACHE)} symbols)[/dim]"
            )
            return 0

        console.print(
            f"[cyan]Backfilling {len(to_backfill)} symbols to {required_start} …[/cyan]"
        )

        from_dt = datetime.combine(required_start, datetime.min.time())
        backfilled = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Backfilling…", total=len(to_backfill))

            with ThreadPoolExecutor(max_workers=self.UPDATE_WORKERS) as executor:
                futures = {}
                for symbol in to_backfill:
                    existing = _SESSION_CACHE[symbol]
                    to_dt = datetime.combine(
                        existing.index.min().date() - timedelta(days=1),
                        datetime.max.time(),
                    )
                    if to_dt <= from_dt:
                        progress.update(task, advance=1)
                        continue
                    futures[
                        executor.submit(
                            self._fetch_full, symbol, from_dt, to_dt
                        )
                    ] = symbol

                for future in as_completed(futures):
                    symbol = futures[future]
                    progress.update(task, description=f"[cyan]Backfill {symbol}…")
                    result = future.result()
                    if result:
                        sym, old_data = result
                        existing = _SESSION_CACHE.get(sym, pd.DataFrame())
                        if not existing.empty:
                            merged = pd.concat(
                                [old_data[~old_data.index.isin(existing.index)], existing]
                            )
                            merged = merged.sort_index()
                        else:
                            merged = old_data
                        # Persist
                        cache_file = self.cache_dir / f"{sym.replace(' ', '_')}.parquet"
                        merged.to_parquet(cache_file)
                        with _SESSION_LOCK:
                            _SESSION_CACHE[sym] = merged
                        backfilled += 1
                    progress.update(task, advance=1)

        console.print(
            f"[green]Backfilled {backfilled}/{len(to_backfill)} symbols "
            f"to {required_start}[/green]"
        )
        return backfilled

    def get_earliest_date(self) -> Optional[date]:
        """Return the earliest date across all cached symbols, or None."""
        global _SESSION_CACHE, _SESSION_CACHE_LOADED
        if not _SESSION_CACHE_LOADED:
            self.load(silent=True)
        earliest = None
        for df in _SESSION_CACHE.values():
            if df is not None and not df.empty:
                d = df.index.min().date()
                if earliest is None or d < earliest:
                    earliest = d
        return earliest

    def invalidate_session_cache(self):
        """Clear session cache (forces reload from disk)."""
        global _SESSION_CACHE, _SESSION_LAST_DATES, _SESSION_CACHE_LOADED
        with _SESSION_LOCK:
            _SESSION_CACHE = {}
            _SESSION_LAST_DATES = {}
            _SESSION_CACHE_LOADED = False
        # Also remove manifest so next load_and_update does a fresh check
        try:
            self._manifest_file.unlink(missing_ok=True)
        except Exception:
            pass


def clear_session_cache():
    """Module-level function to clear the session cache."""
    global _SESSION_CACHE, _SESSION_LAST_DATES, _SESSION_CACHE_LOADED
    with _SESSION_LOCK:
        _SESSION_CACHE = {}
        _SESSION_LAST_DATES = {}
        _SESSION_CACHE_LOADED = False
