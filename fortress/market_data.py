"""
Market data provider for FORTRESS MOMENTUM.

Enforces invariants:
- D6: Historical data has no gaps > 5 days
- D7: Price data is adjusted for corporate actions
- O6: Rate limit max 3 API calls/second
"""

import logging
from datetime import datetime
from typing import Dict, List, Literal, Optional

import pandas as pd

from .instruments import InstrumentMapper
from .utils import rate_limit

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when data quality checks fail."""

    pass


class MarketDataProvider:
    """
    Fetches and caches OHLC data from Zerodha Kite API.

    Rate limited to max 3 requests/second (O6).
    """

    MAX_GAP_DAYS = 5  # D6: Max allowed gap between trading days

    def __init__(self, kite, instrument_mapper: InstrumentMapper):
        """
        Initialize market data provider.

        Args:
            kite: Authenticated KiteConnect instance
            instrument_mapper: InstrumentMapper instance
        """
        self.kite = kite
        self.mapper = instrument_mapper
        self._cache: Dict[str, pd.DataFrame] = {}

    @rate_limit(calls=3, period=1.0)
    def get_historical(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = "day",
        check_quality: bool = True,
        quality_level: Optional[Literal["strict", "warn", "none"]] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data.

        Args:
            symbol: Trading symbol (e.g., "RELIANCE" or "NIFTY 50")
            from_date: Start date
            to_date: End date
            interval: "minute", "day", "week", "month"
            check_quality: Whether to validate data quality (D6) - deprecated, use quality_level
            quality_level: "strict" (raises error), "warn" (logs warning), "none" (no checks)
                          If specified, takes precedence over check_quality

        Returns:
            DataFrame with columns: date (index), open, high, low, close, volume

        Raises:
            ValueError: If symbol not found
            DataQualityError: If data has gaps > 5 days (when quality_level="strict")
        """
        cache_key = f"{symbol}_{from_date.date()}_{to_date.date()}_{interval}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        token = self.mapper.get_token(symbol)
        if token is None:
            raise ValueError(f"Unknown symbol: {symbol}")

        data = self.kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
        )

        if not data:
            raise ValueError(f"No data returned for {symbol}")

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        # D7: Kite API returns adjusted data by default
        # No additional adjustment needed

        # D6: Check for data gaps
        # Determine quality check mode
        if quality_level is not None:
            do_strict = quality_level == "strict"
            do_warn = quality_level == "warn"
        else:
            # Backwards compatibility: check_quality=True means strict
            do_strict = check_quality
            do_warn = False

        if interval == "day" and (do_strict or do_warn):
            issues = self._check_data_quality(df, symbol)
            if issues:
                if do_strict:
                    raise DataQualityError(issues)
                else:
                    logger.warning(f"Data quality issues for {symbol}: {issues}")

        self._cache[cache_key] = df
        return df

    def _check_data_quality(self, df: pd.DataFrame, symbol: str) -> Optional[str]:
        """
        Check data quality and return any issues found.

        Args:
            df: OHLC DataFrame
            symbol: Symbol for messages

        Returns:
            String describing quality issues, or None if no issues
        """
        if len(df) < 2:
            return None

        dates = df.index.to_series()
        gaps = dates.diff().dropna()

        # Find gaps > MAX_GAP_DAYS (excluding weekends)
        # A gap of 3-4 days is normal (weekends + holidays)
        max_gap = gaps.max().days if len(gaps) > 0 else 0

        if max_gap > self.MAX_GAP_DAYS:
            return (
                f"D6 violation: {symbol} has gap of {max_gap} days "
                f"(max allowed: {self.MAX_GAP_DAYS})"
            )

        return None

    def _validate_data_quality(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Validate data quality (D6). Raises error on issues.

        Args:
            df: OHLC DataFrame
            symbol: Symbol for error messages

        Raises:
            DataQualityError: If gaps > MAX_GAP_DAYS found
        """
        issues = self._check_data_quality(df, symbol)
        if issues:
            raise DataQualityError(issues)

    @rate_limit(calls=3, period=1.0)
    def get_ltp(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get last traded prices for multiple symbols.

        Args:
            symbols: List of trading symbols

        Returns:
            Dict mapping symbol to LTP
        """
        api_symbols = [self.mapper.get_api_format(s) for s in symbols]
        quotes = self.kite.ltp(api_symbols)

        result = {}
        for symbol in symbols:
            api_format = self.mapper.get_api_format(symbol)
            if api_format in quotes:
                result[symbol] = quotes[api_format]["last_price"]

        return result

    @rate_limit(calls=3, period=1.0)
    def get_ohlc(self, symbols: List[str]) -> Dict[str, dict]:
        """
        Get OHLC snapshot for multiple symbols.

        Args:
            symbols: List of trading symbols

        Returns:
            Dict mapping symbol to OHLC dict
        """
        api_symbols = [self.mapper.get_api_format(s) for s in symbols]
        return self.kite.ohlc(api_symbols)

    @rate_limit(calls=3, period=1.0)
    def get_quote(self, symbols: List[str]) -> Dict[str, dict]:
        """
        Get full quote for multiple symbols.

        Args:
            symbols: List of trading symbols

        Returns:
            Dict mapping API symbol to quote dict
        """
        api_symbols = [self.mapper.get_api_format(s) for s in symbols]
        return self.kite.quote(api_symbols)

    def get_vix(self) -> float:
        """
        Get current India VIX value.

        Returns:
            VIX value
        """
        ltp = self.get_ltp(["INDIA VIX"])
        return ltp.get("INDIA VIX", 15.0)  # Default to 15 if unavailable

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()


class BacktestDataProvider:
    """
    Data provider for backtesting that pre-loads data.

    Uses the same interface as MarketDataProvider but loads
    all required data upfront to avoid API calls during backtest.
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize with pre-loaded data.

        Args:
            data: Dict mapping symbol to OHLC DataFrame
        """
        self._data = data

    def get_historical(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = "day",
        check_quality: bool = False,
    ) -> pd.DataFrame:
        """
        Get historical data from pre-loaded cache.

        Args:
            symbol: Trading symbol
            from_date: Start date
            to_date: End date
            interval: Ignored (always day)
            check_quality: Ignored

        Returns:
            Filtered DataFrame
        """
        if symbol not in self._data:
            raise ValueError(f"No data for symbol: {symbol}")

        df = self._data[symbol]

        # Filter to date range - convert to pandas Timestamp for comparison
        ts_from = pd.Timestamp(from_date)
        ts_to = pd.Timestamp(to_date)
        mask = (df.index >= ts_from) & (df.index <= ts_to)
        return df.loc[mask].copy()

    def get_ltp(self, symbols: List[str]) -> Dict[str, float]:
        """Get last prices from pre-loaded data."""
        result = {}
        for symbol in symbols:
            if symbol in self._data and len(self._data[symbol]) > 0:
                result[symbol] = self._data[symbol]["close"].iloc[-1]
        return result
