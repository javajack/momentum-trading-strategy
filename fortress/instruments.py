"""
Instrument token mapper for FORTRESS MOMENTUM.

Enforces invariants:
- D3: All sectoral indices have instrument_token (pre-resolved)
- D8: All required sectors have index mapping
"""

from typing import Dict, List, Optional

import pandas as pd

from .universe import Universe


class InstrumentMapper:
    """
    Maps symbols to Zerodha instrument tokens.
    Pre-loads index tokens from universe.json to avoid API lookups.
    """

    # Pre-resolved index tokens from universe.json (D3)
    # These are stable and don't change
    INDEX_TOKENS: Dict[str, int] = {
        # Broad market
        "NIFTY 50": 256265,
        "NIFTY 100": 260617,
        "NIFTY JUNIOR": 260361,
        "NIFTY MIDCAP 100": 256777,
        "NIFTY MIDCAP 50": 260873,
        "INDIA VIX": 264969,
        # Sectoral indices
        "NIFTY BANK": 260105,
        "NIFTY IT": 259849,
        "NIFTY PHARMA": 262409,
        "NIFTY AUTO": 263433,
        "NIFTY METAL": 263689,
        "NIFTY ENERGY": 261641,
        "NIFTY FMCG": 261897,
        "NIFTY REALTY": 261129,
        "NIFTY MEDIA": 263945,
        "NIFTY CONSUMPTION": 257545,
        "NIFTY PSU BANK": 262921,
        "NIFTY COMMODITIES": 257289,
        "NIFTY SERV SECTOR": 263177,
        "NIFTY PSE": 262665,
        # Smallcap indices (used for breadth calculations)
        "NIFTY SMLCAP 50": 266761,
        "NIFTY SMLCAP 100": 267017,
    }

    def __init__(self, kite, universe: Universe):
        """
        Initialize instrument mapper.

        Args:
            kite: Authenticated KiteConnect instance
            universe: Loaded Universe instance
        """
        self.kite = kite
        self.universe = universe
        self._instrument_df: Optional[pd.DataFrame] = None
        self._token_cache: Dict[str, int] = dict(self.INDEX_TOKENS)
        self._tick_size_cache: Dict[str, float] = {}
        self._lot_size_cache: Dict[str, int] = {}

    def load_instruments(self) -> None:
        """
        Load instrument dump from Kite - call once daily.

        This populates the cache with all universe stock tokens.
        """
        instruments = self.kite.instruments("NSE")
        self._instrument_df = pd.DataFrame(instruments)

        # Pre-populate cache for all universe stocks
        for stock in self.universe.get_all_stocks():
            match = self._instrument_df[
                (self._instrument_df["tradingsymbol"] == stock.zerodha_symbol)
                & (self._instrument_df["segment"] == "NSE")
            ]
            if not match.empty:
                self._token_cache[stock.zerodha_symbol] = int(match.iloc[0]["instrument_token"])

    def get_token(self, symbol: str) -> Optional[int]:
        """
        Get instrument token for a symbol.

        Args:
            symbol: Trading symbol (e.g., "RELIANCE" or "NIFTY 50")

        Returns:
            Instrument token or None if not found
        """
        # Check cache first (includes pre-resolved indices)
        if symbol in self._token_cache:
            return self._token_cache[symbol]

        # Load instruments if not done
        if self._instrument_df is None:
            self.load_instruments()

        # Fallback lookup
        if self._instrument_df is not None:
            match = self._instrument_df[self._instrument_df["tradingsymbol"] == symbol]
            if not match.empty:
                token = int(match.iloc[0]["instrument_token"])
                self._token_cache[symbol] = token
                return token

        return None

    def get_tokens_bulk(self, symbols: List[str]) -> Dict[str, int]:
        """
        Get tokens for multiple symbols efficiently.

        Args:
            symbols: List of trading symbols

        Returns:
            Dict mapping symbol to token
        """
        result = {}
        missing = []

        for symbol in symbols:
            if symbol in self._token_cache:
                result[symbol] = self._token_cache[symbol]
            else:
                missing.append(symbol)

        if missing:
            if self._instrument_df is None:
                self.load_instruments()

            for symbol in missing:
                token = self.get_token(symbol)
                if token:
                    result[symbol] = token

        return result

    def get_api_format(self, symbol: str, exchange: str = "NSE") -> str:
        """
        Convert symbol to API format (e.g., NSE:RELIANCE).

        Args:
            symbol: Trading symbol
            exchange: Exchange name (default: NSE)

        Returns:
            API format string
        """
        return f"{exchange}:{symbol}"

    def get_lot_size(self, symbol: str) -> int:
        """
        Get lot size for a symbol from instruments dump.

        Args:
            symbol: Trading symbol

        Returns:
            Lot size (1 for equity by default)
        """
        if symbol in self._lot_size_cache:
            return self._lot_size_cache[symbol]

        if self._instrument_df is None:
            self.load_instruments()

        if self._instrument_df is not None:
            match = self._instrument_df[self._instrument_df["tradingsymbol"] == symbol]
            if not match.empty:
                lot_size = int(match.iloc[0]["lot_size"])
                self._lot_size_cache[symbol] = lot_size
                return lot_size

        return 1  # Default for equity

    def get_tick_size(self, symbol: str) -> float:
        """
        Get tick size for a symbol (NSE equity typically 0.05).

        Args:
            symbol: Trading symbol

        Returns:
            Tick size (0.05 for NSE equity by default)
        """
        if symbol in self._tick_size_cache:
            return self._tick_size_cache[symbol]

        if self._instrument_df is None:
            self.load_instruments()

        if self._instrument_df is not None:
            match = self._instrument_df[self._instrument_df["tradingsymbol"] == symbol]
            if not match.empty:
                tick_size = float(match.iloc[0]["tick_size"])
                self._tick_size_cache[symbol] = tick_size
                return tick_size

        return 0.05  # Default NSE tick size

    def round_to_tick(self, price: float, symbol: str) -> float:
        """
        Round price to nearest tick increment.

        Args:
            price: Price to round
            symbol: Trading symbol

        Returns:
            Price rounded to nearest tick
        """
        tick_size = self.get_tick_size(symbol)
        return round(price / tick_size) * tick_size

    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if symbol exists in universe or instruments.

        Args:
            symbol: Trading symbol

        Returns:
            True if valid
        """
        # Check universe first
        if self.universe.get_stock(symbol):
            return True

        # Check index tokens
        if symbol in self.INDEX_TOKENS:
            return True

        # Check loaded instruments
        if self._instrument_df is not None:
            match = self._instrument_df[self._instrument_df["tradingsymbol"] == symbol]
            return not match.empty

        return False
