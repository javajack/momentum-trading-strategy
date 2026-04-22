"""
Universe — the strategy's view of tradable stocks at a moment in time.

Composes three sources:
    1. **Membership** — nse-universe, queried for `as_of` date within a
       `rank_range` (e.g. (1, 200) = nifty_200-equivalent). Point-in-time,
       survivorship-bias-free.
    2. **Sector map** — stock-sectors.json (built offline by
       tools/build_sectors.py). Classifies every symbol to a sector /
       sub_sector with deterministic fallback to UNCLASSIFIED.
    3. **Static metadata** — market-metadata.json ships the benchmark
       (NIFTY 50), VIX, broad-market indices, sectoral indices (with
       Zerodha instrument tokens), and the hedge registry
       (GOLDBEES / LIQUIDBEES / LIQUIDCASE / etc.).

Invariants preserved from the old loader:
    D2: every member has a non-empty zerodha_symbol / api_format
    D3: every sectoral index has an api_format
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stock:
    """Represents a tradeable stock in the universe."""

    ticker: str
    name: str
    isin: str
    industry: str
    sector: str
    sub_sector: str
    series: str
    zerodha_symbol: str
    api_format: str


@dataclass(frozen=True)
class IndexInfo:
    """Represents a market index."""

    symbol: str
    zerodha_symbol: str
    exchange: str
    segment: str
    instrument_token: Optional[int]
    api_format: str
    maps_to_sector: Optional[str]
    description: str


class UniverseValidationError(Exception):
    """Raised when universe validation fails."""


# Sector assignments for registered hedges that never appear in the
# ranked universe (they're not equity).
_HEDGE_SECTORS = {
    "gold": ("COMMODITIES", "GOLD_ETF"),
    "cash": ("DEBT", "LIQUID_ETF"),
}


def _as_of_default() -> date:
    """Default as-of date. Extracted for test injection."""
    return date.today()


# Process-wide caches for the static artifacts — they don't change within
# a single run, and backtests build hundreds of Universe instances.
_SECTOR_MAP_CACHE: Dict[str, Dict[str, Dict[str, str]]] = {}
_METADATA_CACHE: Dict[str, Dict[str, dict]] = {}


def _nse_universe_singleton():
    """Reuse one NSEUniverse instance across all Universe constructions.

    Opening a DuckDB connection + loading indices.yml per instance is
    ~50 ms each; amortizing across a 240-rebalance backtest matters.
    """
    if not hasattr(_nse_universe_singleton, "_instance"):
        from nse_universe import Universe as NSEUniverse
        _nse_universe_singleton._instance = NSEUniverse()
    return _nse_universe_singleton._instance


class Universe:
    """Tradable stock universe, resolved at a point in time.

    Args:
        as_of: Date for membership resolution. Defaults to today; backtests
            build a fresh Universe per rebalance date for point-in-time
            membership (no survivorship bias).
        rank_range: Inclusive ``(lo, hi)`` rank window from nse-universe.
            ``(1, 200)`` = top-200 (nifty_200 equivalent), the default.
            ``(101, 250)`` = mid-150. Change this to re-target the strategy.
        sectors_path: Path to stock-sectors.json (built by tools/build_sectors.py).
        metadata_path: Path to market-metadata.json (indices + VIX + hedges).

    Backwards compatibility: the legacy ``filepath`` / ``filter_universes``
    kwargs are accepted but ignored so old call sites keep working through
    the Phase-5 migration.
    """

    def __init__(
        self,
        as_of: Optional[date] = None,
        rank_range: Tuple[int, int] = (1, 200),
        sectors_path: str = "stock-sectors.json",
        metadata_path: str = "market-metadata.json",
        *,
        filepath: Optional[str] = None,  # legacy, ignored
        filter_universes: Optional[List[str]] = None,  # legacy, ignored
    ) -> None:
        # Legacy positional call: Universe("stock-universe.json")
        # The positional arg binds to as_of, which is wrong. Detect + unbind.
        if isinstance(as_of, str):
            logger.debug("Universe: ignoring legacy filepath positional %r", as_of)
            as_of = None

        self.as_of: date = as_of or _as_of_default()
        self.rank_range = rank_range

        self._sector_map: Dict[str, Dict[str, str]] = self._load_sector_map(sectors_path)
        self._metadata: Dict[str, dict] = self._load_metadata(metadata_path)
        self._members: List[str] = self._load_members()

        self._stocks_cache: Dict[str, Stock] = {}
        self._hedge_tickers: set[str] = set()
        self._hydrate_stocks()
        self._validate()

    # ---- Composition sources ------------------------------------------------

    @staticmethod
    def _load_sector_map(path: str) -> Dict[str, Dict[str, str]]:
        if path in _SECTOR_MAP_CACHE:
            return _SECTOR_MAP_CACHE[path]
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Sector map not found: {p}. Run tools/build_sectors.py to generate it."
            )
        doc = json.loads(p.read_text()).get("symbols", {})
        _SECTOR_MAP_CACHE[path] = doc
        return doc

    @staticmethod
    def _load_metadata(path: str) -> Dict[str, dict]:
        if path in _METADATA_CACHE:
            return _METADATA_CACHE[path]
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Market metadata not found: {p}")
        doc = json.loads(p.read_text())
        _METADATA_CACHE[path] = doc
        return doc

    def _load_members(self) -> List[str]:
        """Return point-in-time members within ``rank_range`` as of ``as_of``."""
        nse = _nse_universe_singleton()
        df = nse.universe_at(self.as_of)
        lo, hi = self.rank_range
        filtered = df[(df["rank"] >= lo) & (df["rank"] <= hi)]
        return filtered["symbol"].tolist()

    # ---- Stock hydration ----------------------------------------------------

    def _make_stock(
        self,
        ticker: str,
        sector: str,
        sub_sector: str,
        *,
        series: str = "EQ",
    ) -> Stock:
        """Construct a Stock — NSE symbol == Zerodha tradingsymbol for EQ."""
        return Stock(
            ticker=ticker,
            name=ticker,  # nse-universe doesn't carry names
            isin="",
            industry=sector,
            sector=sector,
            sub_sector=sub_sector,
            series=series,
            zerodha_symbol=ticker,
            api_format=f"NSE:{ticker}",
        )

    # Sectors that identify non-equity instruments (index / commodity /
    # debt / international ETFs). Symbols classified here are excluded
    # from the tradable universe even if nse-universe ranks them into
    # the rank window by turnover — they're either user-external ETFs
    # or handled as registered hedges elsewhere.
    _NON_EQUITY_SECTORS = {"DEFENSIVE", "COMMODITIES", "DEBT", "INTERNATIONAL"}

    def _hydrate_stocks(self) -> None:
        # Members of the current rank window — equities only.
        for sym in self._members:
            entry = self._sector_map.get(sym, {})
            sector = entry.get("sector", "UNCLASSIFIED")
            if sector in self._NON_EQUITY_SECTORS:
                continue  # user-external ETFs; hedges (if any) re-added below
            sub_sector = entry.get("sub_sector", "UNCLASSIFIED")
            self._stocks_cache[sym] = self._make_stock(sym, sector, sub_sector)

        # Hedges — always included regardless of rank window.
        for hedge_key, hedge_data in self._metadata.get("hedges", {}).items():
            sym = hedge_data.get("symbol")
            if not sym or sym in self._stocks_cache:
                continue
            sector, sub_sector = _HEDGE_SECTORS.get(hedge_key, ("DEFENSIVE", "DEFENSIVE"))
            self._stocks_cache[sym] = Stock(
                ticker=sym,
                name=hedge_data.get("description", sym),
                isin="",
                industry=sector,
                sector=sector,
                sub_sector=sub_sector,
                series=hedge_data.get("instrument_type", "EQ"),
                zerodha_symbol=hedge_data.get("zerodha_symbol", sym),
                api_format=hedge_data.get("api_format", f"NSE:{sym}"),
            )
            self._hedge_tickers.add(sym)

    def _validate(self) -> None:
        errors: List[str] = []
        for ticker, stock in self._stocks_cache.items():
            if not stock.zerodha_symbol:
                errors.append(f"D2: {ticker} missing zerodha_symbol")
            if not stock.api_format:
                errors.append(f"D2: {ticker} missing api_format")
        for idx_key, idx_data in self._metadata.get("sectoral_indices", {}).items():
            if "api_format" not in idx_data:
                errors.append(f"D3: sectoral index {idx_key} missing api_format")
        if errors:
            raise UniverseValidationError("\n".join(errors))

    # ---- Public API ---------------------------------------------------------

    @property
    def metadata(self) -> dict:
        return self._metadata.get("metadata", {})

    @property
    def benchmark(self) -> IndexInfo:
        b = self._metadata["benchmark"]
        return IndexInfo(
            symbol=b["symbol"],
            zerodha_symbol=b["zerodha_symbol"],
            exchange=b["exchange"],
            segment=b["segment"],
            instrument_token=b.get("instrument_token"),
            api_format=b["api_format"],
            maps_to_sector=None,
            description=b.get("description", ""),
        )

    @property
    def hedge_symbols(self) -> set:
        return {
            h["symbol"]
            for h in self._metadata.get("hedges", {}).values()
            if h.get("symbol")
        }

    def is_managed_symbol(self, symbol: str) -> bool:
        return symbol in self._stocks_cache or symbol in self.hedge_symbols

    def get_all_stocks(self) -> List[Stock]:
        """All tradable stocks (excludes hedge instruments)."""
        return [s for s in self._stocks_cache.values() if s.ticker not in self._hedge_tickers]

    def get_stocks_by_sector(self, sector: str) -> List[Stock]:
        return [s for s in self._stocks_cache.values() if s.sector == sector]

    def get_stocks_by_sub_sector(self, sub_sector: str) -> List[Stock]:
        return [s for s in self._stocks_cache.values() if s.sub_sector == sub_sector]

    def get_sub_sectors(self, sector: Optional[str] = None) -> List[str]:
        if sector:
            return list({s.sub_sector for s in self._stocks_cache.values() if s.sector == sector})
        return list({s.sub_sector for s in self._stocks_cache.values()})

    def get_valid_sectors(self, min_stocks: int = 3) -> List[str]:
        counts: Dict[str, int] = {}
        for stock in self._stocks_cache.values():
            counts[stock.sector] = counts.get(stock.sector, 0) + 1
        return [s for s, c in counts.items() if c >= min_stocks]

    def get_sector_index(self, sector: str) -> Optional[IndexInfo]:
        for idx_key, idx_data in self._metadata.get("sectoral_indices", {}).items():
            if idx_data.get("maps_to_sector") == sector:
                return IndexInfo(
                    symbol=idx_data["symbol"],
                    zerodha_symbol=idx_data.get("zerodha_symbol", idx_data["symbol"]),
                    exchange=idx_data.get("exchange", "NSE"),
                    segment=idx_data.get("segment", "INDICES"),
                    instrument_token=idx_data.get("instrument_token"),
                    api_format=idx_data["api_format"],
                    maps_to_sector=sector,
                    description=idx_data.get("description", ""),
                )
        return None

    def get_hedge(self, hedge_type: str) -> Optional[dict]:
        return self._metadata.get("hedges", {}).get(hedge_type)

    def get_stock(self, ticker: str) -> Optional[Stock]:
        """Return a Stock for any known ticker.

        Falls back to the sector map if the ticker isn't in the current
        rank window — lets Portfolio classify holdings that dropped out
        of the universe recently (or live outside it entirely).
        """
        if ticker in self._stocks_cache:
            return self._stocks_cache[ticker]
        sec_entry = self._sector_map.get(ticker)
        if sec_entry is None:
            return None
        return self._make_stock(
            ticker,
            sec_entry.get("sector", "UNCLASSIFIED"),
            sec_entry.get("sub_sector", "UNCLASSIFIED"),
        )

    def get_api_symbols(self, tickers: List[str]) -> List[str]:
        out: List[str] = []
        for t in tickers:
            stock = self.get_stock(t)
            if stock is not None:
                out.append(stock.api_format)
        return out

    def get_vix(self) -> IndexInfo:
        vix = self._metadata["broad_market_indices"]["INDIA_VIX"]
        return IndexInfo(
            symbol=vix["symbol"],
            zerodha_symbol=vix["zerodha_symbol"],
            exchange=vix["exchange"],
            segment=vix["segment"],
            instrument_token=vix.get("instrument_token"),
            api_format=vix["api_format"],
            maps_to_sector=None,
            description=vix.get("description", ""),
        )
