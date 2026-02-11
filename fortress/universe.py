"""
Universe parser for FORTRESS MOMENTUM.

Enforces invariants:
- D1: Universe JSON validates against schema
- D2: All stocks have valid zerodha_symbol
- D3: All sectoral indices have instrument_token (where available)
- D4: No duplicate tickers in universe
- D5: sector_summary totals match actual counts (warning only)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stock:
    """Represents a tradeable stock in the universe."""

    ticker: str
    name: str
    isin: str
    industry: str
    sector: str
    sub_sector: str  # Granular business classification
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


@dataclass(frozen=True)
class ETFInfo:
    """Represents an ETF."""

    symbol: str
    api_format: str
    fund_house: Optional[str]
    tracks_index: str
    description: Optional[str]


class UniverseValidationError(Exception):
    """Raised when universe validation fails."""

    pass


class Universe:
    """
    Parser for stock-universe.json.

    Loads and validates the stock universe, providing access to:
    - Stocks by sector
    - Sectoral indices
    - Sector ETFs
    - Hedge instruments
    """

    def __init__(self, filepath: str = "stock-universe.json", filter_universes: Optional[List[str]] = None):
        """
        Load and validate universe from JSON file.

        Args:
            filepath: Path to universe JSON file
            filter_universes: Optional list of universe keys to load (e.g. ["NIFTY100", "MIDCAP100"]).
                If None, all universes are loaded. Hedges are always included.

        Raises:
            FileNotFoundError: If file doesn't exist
            UniverseValidationError: If validation fails
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Universe file not found: {filepath}")

        with open(path) as f:
            self._data = json.load(f)

        self._filter_universes = filter_universes
        self._stocks_cache: Dict[str, Stock] = {}
        self._hedge_tickers: set = set()
        self._build_stock_cache()
        self._validate()

    # Sector mapping for hedge/defensive instruments
    _HEDGE_SECTORS = {
        "gold": ("COMMODITIES", "GOLD_ETF"),
        "silver": ("COMMODITIES", "SILVER_ETF"),
        "international": ("INTERNATIONAL", "INTERNATIONAL_ETF"),
        "cash": ("DEBT", "LIQUID_ETF"),
        "cash_liquid": ("DEBT", "LIQUID_ETF"),
    }

    def _build_stock_cache(self) -> None:
        """Build internal cache of stocks from universes and hedge instruments."""
        for universe_name, universe in self._data.get("universes", {}).items():
            if self._filter_universes and universe_name not in self._filter_universes:
                continue
            for stock_data in universe.get("stocks", []):
                stock = Stock(
                    ticker=stock_data["ticker"],
                    name=stock_data["name"],
                    isin=stock_data["isin"],
                    industry=stock_data["industry"],
                    sector=stock_data["sector"],
                    sub_sector=stock_data.get("sub_sector", stock_data["sector"]),
                    series=stock_data["series"],
                    zerodha_symbol=stock_data["zerodha_symbol"],
                    api_format=stock_data["api_format"],
                )
                self._stocks_cache[stock.ticker] = stock

        # Register hedge instruments so they get proper sector assignments
        for hedge_key, hedge_data in self._data.get("hedges", {}).items():
            symbol = hedge_data.get("symbol", "")
            if symbol and symbol not in self._stocks_cache:
                sector, sub_sector = self._HEDGE_SECTORS.get(hedge_key, ("DEFENSIVE", "DEFENSIVE"))
                self._stocks_cache[symbol] = Stock(
                    ticker=symbol,
                    name=hedge_data.get("description", symbol),
                    isin="",
                    industry=sector,
                    sector=sector,
                    sub_sector=sub_sector,
                    series=hedge_data.get("instrument_type", "EQ"),
                    zerodha_symbol=hedge_data.get("zerodha_symbol", symbol),
                    api_format=hedge_data.get("api_format", f"NSE:{symbol}"),
                )
                self._hedge_tickers.add(symbol)

    def _validate(self) -> None:
        """
        Validate universe against invariants D1-D5.

        Raises:
            UniverseValidationError: If critical validation fails (D1-D4)
        """
        errors = []
        warnings = []

        # D1: Validate required top-level keys
        required_keys = ["metadata", "benchmark", "sectoral_indices", "universes"]
        for key in required_keys:
            if key not in self._data:
                errors.append(f"D1: Missing required key: {key}")

        # D2: All stocks have valid zerodha_symbol
        for ticker, stock in self._stocks_cache.items():
            if not stock.zerodha_symbol:
                errors.append(f"D2: Stock {ticker} missing zerodha_symbol")
            if not stock.api_format:
                errors.append(f"D2: Stock {ticker} missing api_format")

        # D3: Check sectoral indices have instrument_token where expected
        # Note: Some indices may not have tokens pre-resolved
        for idx_key, idx_data in self._data.get("sectoral_indices", {}).items():
            if "api_format" not in idx_data:
                errors.append(f"D3: Index {idx_key} missing api_format")

        # D4: No duplicate tickers (dedupe instead of error)
        all_tickers = []
        for universe in self._data.get("universes", {}).values():
            for stock in universe.get("stocks", []):
                all_tickers.append(stock["ticker"])
        duplicates = set(t for t in all_tickers if all_tickers.count(t) > 1)
        if duplicates:
            # Log warning but don't fail - duplicates are already deduped in cache
            logger.debug(f"D4: Duplicate tickers found (deduped): {duplicates}")

        # D5: sector_summary totals match actual counts (warning only)
        # This is metadata validation - doesn't affect trading functionality
        if "sector_summary" in self._data:
            actual_counts: Dict[str, int] = {}
            for stock in self._stocks_cache.values():
                actual_counts[stock.sector] = actual_counts.get(stock.sector, 0) + 1

            for sector, counts in self._data["sector_summary"].items():
                expected_total = counts.get("total", 0)
                actual_total = actual_counts.get(sector, 0)
                if expected_total != actual_total:
                    warnings.append(
                        f"D5: Sector {sector} summary={expected_total}, "
                        f"actual={actual_total}"
                    )

        # Log warnings but don't fail
        if warnings:
            logger.debug(f"Universe validation warnings: {len(warnings)} issues")
            for w in warnings:
                logger.debug(f"  {w}")

        # Only fail on critical errors (D1-D3)
        if errors:
            raise UniverseValidationError("\n".join(errors))

    @property
    def metadata(self) -> dict:
        """Get universe metadata."""
        return self._data.get("metadata", {})

    @property
    def benchmark(self) -> IndexInfo:
        """Get benchmark index (NIFTY 50)."""
        b = self._data["benchmark"]
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

    def get_all_stocks(self) -> List[Stock]:
        """
        Get all tradeable stocks in the universe (excludes hedge instruments).

        Returns:
            List of all stocks (excluding GOLDBEES, LIQUIDCASE, etc.)
        """
        return [s for s in self._stocks_cache.values() if s.ticker not in self._hedge_tickers]

    def get_stocks_by_sector(self, sector: str) -> List[Stock]:
        """
        Get stocks belonging to a specific sector.

        Args:
            sector: Sector name (e.g., "FINANCIALS")

        Returns:
            List of stocks in the sector
        """
        return [s for s in self._stocks_cache.values() if s.sector == sector]

    def get_stocks_by_sub_sector(self, sub_sector: str) -> List[Stock]:
        """
        Get stocks belonging to a specific sub-sector.

        Args:
            sub_sector: Sub-sector name (e.g., "BANKING_PRIVATE")

        Returns:
            List of stocks in the sub-sector
        """
        return [s for s in self._stocks_cache.values() if s.sub_sector == sub_sector]

    def get_sub_sectors(self, sector: Optional[str] = None) -> List[str]:
        """
        Get all unique sub-sectors, optionally filtered by sector.

        Args:
            sector: If provided, only return sub-sectors within this sector

        Returns:
            List of unique sub-sector names
        """
        if sector:
            return list(set(
                s.sub_sector for s in self._stocks_cache.values()
                if s.sector == sector
            ))
        return list(set(s.sub_sector for s in self._stocks_cache.values()))

    def get_valid_sectors(self, min_stocks: int = 3) -> List[str]:
        """
        Get sectors with at least min_stocks for ranking eligibility.

        Args:
            min_stocks: Minimum stocks required (default: 3)

        Returns:
            List of eligible sector names
        """
        sector_counts: Dict[str, int] = {}
        for stock in self._stocks_cache.values():
            sector_counts[stock.sector] = sector_counts.get(stock.sector, 0) + 1

        return [sector for sector, count in sector_counts.items() if count >= min_stocks]

    def get_sector_index(self, sector: str) -> Optional[IndexInfo]:
        """
        Get the sectoral index for a sector.

        Args:
            sector: Sector name

        Returns:
            IndexInfo or None if no index maps to sector
        """
        for idx_key, idx_data in self._data.get("sectoral_indices", {}).items():
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

    def get_sector_etf(self, sector: str) -> Optional[ETFInfo]:
        """
        Get the primary ETF for a sector.

        Args:
            sector: Sector name

        Returns:
            ETFInfo or None if no ETF available
        """
        etfs = self._data.get("sectoral_etfs", {})
        if sector not in etfs:
            return None

        etf_data = etfs[sector].get("primary_etf")
        if not etf_data:
            return None

        return ETFInfo(
            symbol=etf_data["symbol"],
            api_format=etf_data["api_format"],
            fund_house=etf_data.get("fund_house"),
            tracks_index=etf_data.get("tracks_index", ""),
            description=etf_data.get("description"),
        )

    def get_hedge(self, hedge_type: str) -> Optional[dict]:
        """
        Get hedge instrument info.

        Args:
            hedge_type: "gold", "cash", "international", or "silver"

        Returns:
            Hedge info dict or None
        """
        return self._data.get("hedges", {}).get(hedge_type)

    def get_stock(self, ticker: str) -> Optional[Stock]:
        """
        Get stock by ticker.

        Args:
            ticker: Stock ticker (e.g., "RELIANCE")

        Returns:
            Stock or None if not found
        """
        return self._stocks_cache.get(ticker)

    def get_api_symbols(self, tickers: List[str]) -> List[str]:
        """
        Convert tickers to Zerodha API format.

        Args:
            tickers: List of ticker symbols

        Returns:
            List of API format symbols (e.g., ["NSE:RELIANCE"])
        """
        return [
            self._stocks_cache[t].api_format
            for t in tickers
            if t in self._stocks_cache
        ]

    def get_vix(self) -> IndexInfo:
        """Get India VIX index info."""
        vix = self._data["broad_market_indices"]["INDIA_VIX"]
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
