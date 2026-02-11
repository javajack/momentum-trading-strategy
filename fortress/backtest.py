"""
Backtesting framework for FORTRESS MOMENTUM.

Enforces invariants:
- C7: No look-ahead bias in backtest
- P6: Backtest and live use same calculation code

Supports pluggable strategies via StrategyRegistry.

Performance optimizations:
- NMS result caching per (ticker, date) pair
- Pre-filtered data views by date
- Parallel NMS calculation (optional)
- Dynamic rebalancing optimization (60x speedup):
  - Pre-computed market breadth cache (O(1) lookup vs O(n×m) per day)
  - Pre-computed 21-day market returns cache
  - Pre-computed 20-day VIX peak cache
  - Fast O(1) trigger check to skip ~85% of days early
  - deque for O(1) breadth history management
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config, get_default_config
from .indicators import (
    calculate_drawdown,
    calculate_normalized_momentum_score,
    calculate_bull_recovery_signals,
    calculate_market_breadth,
    detect_market_regime,
    detect_vix_recovery,
    detect_breadth_thrust,
    detect_momentum_crash,
    should_trigger_rebalance,
    MarketRegime,
    NMSResult,
    RegimeResult,
    BullRecoverySignals,
    RebalanceTrigger,
)
from .portfolio import BacktestPortfolio
from .risk_governor import RiskGovernor
from .strategy import StrategyRegistry, StockScore
from .universe import Universe
from .utils import renormalize_with_caps


class BacktestMarketDataAdapter:
    """
    Adapter that makes historical data look like MarketDataProvider.

    Enforces C7 (no look-ahead) by filtering data to as_of_date.
    """

    def __init__(self, historical_data: Dict[str, pd.DataFrame]):
        self.data = historical_data
        self._as_of_date: Optional[datetime] = None

    def set_as_of_date(self, date: datetime):
        """Set the current simulation date (for look-ahead prevention)."""
        self._as_of_date = date

    def get_historical(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = "day",
        check_quality: bool = False,
    ) -> pd.DataFrame:
        """
        Get historical data up to the as_of_date (no look-ahead).

        C7: Only returns data available at as_of_date.
        """
        if symbol not in self.data:
            return pd.DataFrame()

        df = self.data[symbol]

        # Use the earlier of to_date and as_of_date (no look-ahead)
        effective_to = to_date
        if self._as_of_date is not None:
            ts_as_of = pd.Timestamp(self._as_of_date)
            ts_to = pd.Timestamp(to_date)
            if ts_as_of < ts_to:
                effective_to = self._as_of_date

        ts_from = pd.Timestamp(from_date)
        ts_to = pd.Timestamp(effective_to)

        mask = (df.index >= ts_from) & (df.index <= ts_to)
        return df[mask].copy()


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""

    start_date: datetime
    end_date: datetime
    initial_capital: float = 1600000
    rebalance_days: int = 21  # Monthly rebalance
    transaction_cost: float = 0.003  # 0.3% round trip

    # Position management - OPTIMIZED for quality focus
    target_positions: int = 12
    min_positions: int = 10
    max_positions: int = 15

    # Sector diversification (max stocks per sector, 0 = no limit)
    # Default 3: better diversification across sectors
    max_stocks_per_sector: int = 3

    # Stop loss settings
    use_stop_loss: bool = True
    initial_stop_loss: float = 0.18  # 18% initial stop
    trailing_stop: float = 0.15  # 15% trailing stop
    trailing_activation: float = 0.08

    # Entry filters (must match MomentumEngine filters)
    min_score_percentile: float = 95  # Top 5% by NMS
    min_52w_high_prox: float = 0.85  # 85% of 52-week high
    min_volume_ratio: float = 1.1  # 20d vol >= 1.1x 50d vol
    min_daily_turnover: float = 20_000_000  # Rs 2 Cr minimum

    # NMS weight overrides (None = use app_config)
    weight_6m: Optional[float] = None
    weight_12m: Optional[float] = None

    # Exit percentile threshold
    exit_percentile: Optional[float] = None

    # Regime detection settings
    use_regime_detection: bool = True
    compare_benchmarks: bool = True

    # Per-profile gold override
    profile_max_gold: Optional[float] = None

    # Strategy selection
    strategy_name: str = "dual_momentum"


@dataclass
class Trade:
    """Record of a single trade."""

    date: datetime
    symbol: str
    sector: str
    action: str  # "BUY" or "SELL"
    quantity: int
    price: float
    value: float
    cost: float  # Transaction cost


@dataclass
class RebalanceRecord:
    """Record of a single rebalance decision for the trail log."""

    # Timing
    date: datetime
    rebalance_number: int
    portfolio_value: float

    # Regime reasoning
    regime: str
    nifty_price: float
    nifty_sma: float
    trend_above_sma: bool
    vix_value: float
    breadth_value: float

    # Sleeve weights (after all scaling)
    equity_weight: float
    gold_weight: float
    liquid_weight: float

    # Scaling factors applied
    vol_scale: float
    breadth_scale: float
    gold_exhaustion_scale: float

    # Equity picks: list of (symbol, score, weight)
    equity_picks: List[Tuple[str, float, float]]

    # Gate failures: list of (symbol, score)
    gate_failures: List[Tuple[str, float]]

    # Execution
    trade_count: int


@dataclass
class BacktestResult:
    """Results of a backtest run."""

    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: pd.Series
    trades: List[Trade]
    sector_allocations: pd.DataFrame

    # Capital amounts
    initial_capital: float = 0.0
    final_value: float = 0.0
    peak_value: float = 0.0
    total_profit: float = 0.0

    # Regime detection results
    regime_history: Optional[pd.DataFrame] = None
    regime_transitions: int = 0
    time_in_regime: Optional[Dict[str, float]] = None

    # Benchmark comparison
    nifty_50_return: Optional[float] = None
    nifty_midcap_100_return: Optional[float] = None

    # Strategy used
    strategy_name: str = "dual_momentum"

    # Rebalance trail log
    rebalance_trail: List[RebalanceRecord] = None

    def __post_init__(self):
        if self.rebalance_trail is None:
            self.rebalance_trail = []


class BacktestEngine:
    """
    Historical simulation for pure momentum strategy.

    Key features:
    - C7: Look-ahead bias prevention (uses only data available at decision time)
    - P6: Uses same calculation code as live trading
    - Realistic transaction costs
    - Monthly rebalance
    """

    def __init__(
        self,
        universe: Universe,
        historical_data: Dict[str, pd.DataFrame],
        config: Optional[BacktestConfig] = None,
        app_config: Optional[Config] = None,
        strategy_name: Optional[str] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            universe: Loaded Universe instance
            historical_data: Dict mapping symbol to OHLC DataFrame
            config: Backtest configuration
            app_config: Application config for strategy parameters
            strategy_name: Strategy to use (overrides config.strategy_name)
        """
        self.universe = universe
        self.data = historical_data
        # Default end_date to T-1 (yesterday, skip weekends) to avoid live data
        if config is None:
            yesterday = datetime.now() - timedelta(days=1)
            while yesterday.weekday() >= 5:
                yesterday -= timedelta(days=1)
            config = BacktestConfig(
                start_date=datetime(2024, 1, 1),
                end_date=yesterday,
            )
        self.config = config
        self.app_config = app_config or get_default_config()
        self.risk_governor = RiskGovernor(
            self.app_config.risk, self.app_config.portfolio
        )

        # Initialize strategy
        effective_strategy = strategy_name or self.config.strategy_name
        self.strategy = StrategyRegistry.get(effective_strategy, self.app_config)
        self._strategy_name = effective_strategy

        # Regime tracking for hysteresis (track previous result for state)
        self._previous_regime_result: Optional[RegimeResult] = None

        # Drawdown tracking for AllWeather recovery mode
        self._current_drawdown: float = 0.0

        # Create market data adapter for strategy
        self._market_data_adapter = BacktestMarketDataAdapter(self.data)

        # Performance optimization: NMS cache per (ticker, date_str)
        # Avoids redundant NMS calculations across multiple calls
        self._nms_cache: Dict[str, Tuple[float, NMSResult]] = {}
        self._nms_cache_hits = 0
        self._nms_cache_misses = 0

        # Pre-compute date-filtered data views for common dates (lazy populated)
        self._date_filtered_data: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Dynamic rebalancing state
        self._last_rebalance_date: Optional[datetime] = None
        self._days_since_rebalance: int = 0
        self._vix_peak_20d: float = 15.0
        self._breadth_history: Deque[float] = deque(maxlen=15)  # O(1) append
        self._rebalance_triggers_log: List[dict] = []

        # Performance optimization: Pre-computed caches for dynamic rebalancing
        # These eliminate expensive O(n×m) calculations on every trading day
        self._breadth_cache: Dict[str, float] = {}  # date_str -> breadth value
        self._market_return_cache: pd.Series = pd.Series(dtype=float)  # 21-day returns
        self._vix_peak_cache: pd.Series = pd.Series(dtype=float)  # 20-day rolling max
        self._regime_cache: Dict[str, RegimeResult] = {}  # date_str -> regime
        self._nifty_200sma_cache: pd.Series = pd.Series(dtype=float)  # 200-day SMA
        self._gold_200sma_cache: pd.Series = pd.Series(dtype=float)  # Gold 200-day SMA

        # Per-profile gold override
        self._profile_max_gold = config.profile_max_gold

        # Trail log: intermediate data from the last rebalance call
        self._trail_ranked_stocks: List[StockScore] = []
        self._trail_vol_scale: float = 1.0
        self._trail_breadth_scale: float = 1.0
        self._trail_gold_exhaustion_scale: float = 1.0

        # Pre-filtered symbol lists for O(1) lookups
        self._excluded_set: set = set(self.app_config.excluded_symbols) | {
            "NIFTY 50", "INDIA VIX", "NIFTY MIDCAP 100",
            self.app_config.regime.gold_symbol,
            self.app_config.regime.cash_symbol,
        }
        self._stock_symbols: List[str] = []  # Populated in _initialize_caches
        self._caches_initialized: bool = False

        # P2: Pre-built close price Series for O(log n) .asof() lookups
        self._close_prices: Dict[str, pd.Series] = {
            symbol: df["close"] for symbol, df in self.data.items() if len(df) > 0
        }

        # E9: Minimum hold period (only hard stop during first N days)
        strategy_cfg = getattr(self.app_config, "strategy_dual_momentum", None)
        self._min_hold_days: int = getattr(strategy_cfg, "min_hold_days", 10) if strategy_cfg else 10

    def _get_trading_days(self) -> pd.DatetimeIndex:
        """
        Get actual trading days from the historical data.

        Uses a reference symbol (benchmark or first available) to determine
        which days the market was open.

        Returns:
            DatetimeIndex of trading days
        """
        # Try to find a symbol with good data coverage
        reference_symbols = ["NIFTY 50", "RELIANCE", "INFY", "SBIN"]

        for symbol in reference_symbols:
            if symbol in self.data and len(self.data[symbol]) > 0:
                return self.data[symbol].index

        # Fall back to first available symbol
        for symbol, df in self.data.items():
            if len(df) > 0:
                return df.index

        return pd.DatetimeIndex([])

    def _initialize_caches(self) -> None:
        """
        Pre-compute all expensive calculations once at backtest start.

        This is the key optimization for dynamic rebalancing performance.
        Instead of computing O(n×m) operations on every trading day,
        we compute them once upfront and use O(1) lookups during simulation.

        Pre-computes:
        - Market breadth for all dates (% stocks above 50-day MA)
        - 21-day rolling market returns for NIFTY
        - 20-day rolling VIX peak values
        """
        if self._caches_initialized:
            return

        # Get stock symbols (excluding indices and special symbols)
        self._stock_symbols = [
            s for s in self.data.keys()
            if s not in self._excluded_set
        ]

        # Pre-compute market breadth for all dates
        self._breadth_cache = self._precompute_breadth()

        # Pre-compute 21-day market returns
        self._market_return_cache = self._precompute_market_returns()

        # Pre-compute VIX 20-day rolling peak
        self._vix_peak_cache = self._precompute_vix_peaks()

        # Pre-compute NIFTY 200-day SMA for trend guard
        self._nifty_200sma_cache = self._precompute_nifty_200sma()

        # Pre-compute Gold SMA for exhaustion scaling
        self._gold_200sma_cache = self._precompute_gold_200sma()

        self._caches_initialized = True

    def _precompute_breadth(self) -> Dict[str, float]:
        """
        Pre-compute market breadth for all trading dates.

        Vectorized: computes 50-day MA on each stock's own data (no NaN gaps),
        then aligns to common DataFrame for a single boolean comparison.

        Returns:
            Dict mapping date string to breadth value (0.0 to 1.0)
        """
        if not self._stock_symbols:
            return {}

        # Compute MA on each stock's own consecutive data, then align
        close_dict = {}
        ma_dict = {}
        for symbol in self._stock_symbols:
            if symbol in self.data and len(self.data[symbol]) >= 50:
                s = self.data[symbol]["close"]
                close_dict[symbol] = s
                ma_dict[symbol] = s.rolling(50).mean()

        if not close_dict:
            return {}

        close_df = pd.DataFrame(close_dict)
        ma_50_df = pd.DataFrame(ma_dict)

        # Vectorized: count stocks above their 50-day MA per date
        above_ma = (close_df > ma_50_df) & close_df.notna() & ma_50_df.notna()
        valid = close_df.notna() & ma_50_df.notna()

        above_count = above_ma.sum(axis=1)
        valid_count = valid.sum(axis=1)

        breadth_series = above_count / valid_count.replace(0, np.nan)
        breadth_series = breadth_series.fillna(0.5)

        return {str(d.date()): v for d, v in breadth_series.items()}

    def _precompute_market_returns(self) -> pd.Series:
        """
        Pre-compute 21-day rolling returns for NIFTY 50.

        Used for fast O(1) lookup of market momentum in trigger checks.

        Returns:
            Series of 21-day returns indexed by date
        """
        nifty_symbol = "NIFTY 50"
        if nifty_symbol not in self.data:
            return pd.Series(dtype=float)

        nifty_close = self.data[nifty_symbol]["close"]
        return nifty_close.pct_change(21)

    def _precompute_vix_peaks(self) -> pd.Series:
        """
        Pre-compute 20-day rolling maximum VIX.

        Used for VIX recovery detection (comparing current VIX to recent peak).

        Returns:
            Series of 20-day VIX peaks indexed by date
        """
        vix_symbol = "INDIA VIX"
        if vix_symbol not in self.data:
            return pd.Series(dtype=float)

        vix_close = self.data[vix_symbol]["close"]
        return vix_close.rolling(20).max()

    def _precompute_nifty_200sma(self) -> pd.Series:
        """Pre-compute 200-day SMA for NIFTY 50 (trend guard)."""
        nifty_symbol = "NIFTY 50"
        if nifty_symbol not in self.data:
            return pd.Series(dtype=float)
        return self.data[nifty_symbol]["close"].rolling(200).mean()

    def _precompute_gold_200sma(self) -> pd.Series:
        """Pre-compute SMA for GOLDBEES (gold exhaustion scaling)."""
        gold_symbol = self.app_config.regime.gold_symbol
        sma_period = self.app_config.regime.gold_exhaustion_sma_period
        if gold_symbol not in self.data:
            return pd.Series(dtype=float)
        return self.data[gold_symbol]["close"].rolling(sma_period).mean()

    def _is_market_uptrend(self, date: datetime) -> bool:
        """O(1) check: is NIFTY 50 above its 200-day SMA?"""
        if self._nifty_200sma_cache.empty:
            return False
        ts_date = pd.Timestamp(date)
        if ts_date not in self._nifty_200sma_cache.index:
            return False
        sma_val = self._nifty_200sma_cache.loc[ts_date]
        if pd.isna(sma_val):
            return False
        nifty_close = self.data["NIFTY 50"].loc[ts_date, "close"]
        return bool(nifty_close > sma_val)

    def _get_cached_breadth(self, date: datetime) -> float:
        """O(1) breadth lookup from pre-computed cache."""
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)
        return self._breadth_cache.get(date_str, 0.5)

    def _get_past_trading_date(self, as_of_date: datetime, n_days: int) -> Optional[datetime]:
        """Get the trading date N trading days before as_of_date.

        Uses the pre-computed trading day index for O(1) lookup.
        Returns None if not enough history.
        """
        trading_days = self._get_trading_days()
        ts_date = pd.Timestamp(as_of_date)
        idx = trading_days.searchsorted(ts_date, side="right") - 1
        target_idx = idx - n_days
        if target_idx < 0:
            return None
        return trading_days[target_idx].to_pydatetime()

    def _get_cached_market_return(self, date: datetime) -> float:
        """O(1) market return lookup from pre-computed cache."""
        ts_date = pd.Timestamp(date)
        if ts_date in self._market_return_cache.index:
            val = self._market_return_cache.loc[ts_date]
            return val if not pd.isna(val) else 0.0
        return 0.0

    def _get_cached_vix_peak(self, date: datetime) -> float:
        """O(1) VIX peak lookup from pre-computed cache."""
        ts_date = pd.Timestamp(date)
        if ts_date in self._vix_peak_cache.index:
            val = self._vix_peak_cache.loc[ts_date]
            return val if not pd.isna(val) else 15.0
        return 15.0

    def _get_vix_at_date(self, date: datetime) -> float:
        """Get current VIX value at date (O(1) lookup)."""
        vix_symbol = "INDIA VIX"
        if vix_symbol not in self.data:
            return 15.0

        ts_date = pd.Timestamp(date)
        vix_df = self.data[vix_symbol]
        if ts_date in vix_df.index:
            return vix_df.loc[ts_date, "close"]

        # Fall back to most recent value before date
        mask = vix_df.index <= ts_date
        available = vix_df[mask]
        if len(available) > 0:
            return available["close"].iloc[-1]
        return 15.0

    def _should_check_rebalance_fast(
        self,
        date: datetime,
        days_since_last: int,
    ) -> bool:
        """
        Fast O(1) check to determine if we should evaluate full rebalance triggers.

        This is the key optimization: skip expensive calculations on ~85% of days
        by using pre-computed caches for quick checks.

        Returns True only if:
        - Past max_days_between (must rebalance)
        - VIX recovery detected (worth checking)
        - Significant drawdown (worth checking)
        - Market crash detected (worth checking)

        Returns False to skip full trigger evaluation.
        """
        dynamic_config = self.app_config.dynamic_rebalance

        # Always skip if under minimum days
        if days_since_last < dynamic_config.min_days_between:
            return False

        # Always check if past maximum days
        if days_since_last >= dynamic_config.max_days_between:
            return True

        # Quick VIX recovery check (O(1) lookups)
        vix_today = self._get_vix_at_date(date)
        vix_peak = self._get_cached_vix_peak(date)

        if vix_peak > dynamic_config.vix_spike_threshold:
            # VIX was elevated recently - check for recovery
            vix_decline = (vix_peak - vix_today) / vix_peak if vix_peak > 0 else 0
            if vix_decline >= dynamic_config.vix_recovery_decline:
                return True  # VIX recovery - worth full check

        # Quick drawdown check
        if self._current_drawdown <= -dynamic_config.drawdown_threshold:
            return True  # Significant drawdown - worth full check

        # Quick market crash check (O(1) lookup)
        market_return = self._get_cached_market_return(date)
        if market_return <= dynamic_config.crash_threshold:
            return True  # Market crash - worth full check

        # No urgent triggers detected - skip expensive calculations
        return False

    def _get_rebalance_dates(self) -> List[datetime]:
        """
        Get dates when rebalancing should occur based on trading days.

        Uses actual trading days from market data, rebalancing every N trading days.
        This ensures weekends and holidays are automatically skipped.

        C7: Only uses dates within backtest range.

        Returns:
            List of rebalance dates (actual trading days)
        """
        trading_days = self._get_trading_days()

        if len(trading_days) == 0:
            return []

        # Convert config dates to pandas Timestamps for comparison
        ts_start = pd.Timestamp(self.config.start_date)
        ts_end = pd.Timestamp(self.config.end_date)

        # Filter trading days to backtest range
        mask = (trading_days >= ts_start) & (trading_days <= ts_end)
        valid_days = trading_days[mask]

        if len(valid_days) == 0:
            return []

        # Select every N-th trading day for rebalancing
        rebalance_interval = self.config.rebalance_days
        rebalance_dates = []

        for i in range(0, len(valid_days), rebalance_interval):
            # Convert to Python datetime for consistency
            rebalance_dates.append(valid_days[i].to_pydatetime())

        return rebalance_dates

    def _detect_regime(
        self,
        as_of_date: datetime,
    ) -> Optional[RegimeResult]:
        """
        Detect market regime at the given date.

        C7: Only uses data available at as_of_date.

        Uses enhanced regime detection with:
        - Multi-timeframe position (21/63/126 day weighted)
        - Hysteresis for transition confirmation
        - Graduated allocation based on stress score

        Args:
            as_of_date: Date for regime detection

        Returns:
            RegimeResult or None if insufficient data
        """
        if not self.config.use_regime_detection:
            return None

        regime_config = self.app_config.regime
        if not regime_config.enabled:
            return None

        # Get Nifty 50 prices
        nifty_symbol = "NIFTY 50"
        if nifty_symbol not in self.data:
            return None

        nifty_df = self.data[nifty_symbol]
        ts_date = pd.Timestamp(as_of_date)
        mask = nifty_df.index <= ts_date
        nifty_available = nifty_df[mask]

        if len(nifty_available) < 63:  # Need at least 3 months
            return None

        nifty_prices = nifty_available["close"]

        # Get VIX value and history (use default if not available)
        vix_symbol = "INDIA VIX"
        vix_value = 15.0  # Default calm market
        vix_history = None

        if vix_symbol in self.data:
            vix_df = self.data[vix_symbol]
            vix_mask = vix_df.index <= ts_date
            vix_available = vix_df[vix_mask]
            if len(vix_available) > 0:
                vix_value = vix_available["close"].iloc[-1]
                if len(vix_available) >= 10:
                    vix_history = vix_available["close"]

        # Pass previous result for hysteresis tracking
        return detect_market_regime(
            nifty_prices,
            vix_value,
            regime_config,
            previous_result=self._previous_regime_result,
            vix_history=vix_history,
        )

    def _calculate_bull_recovery_signals(
        self,
        as_of_date: datetime,
        regime: RegimeResult,
    ) -> Optional[BullRecoverySignals]:
        """
        Calculate bull recovery signals for the strategy.

        Bull recovery mode activates when:
        1. VIX is declining from elevated levels (fear subsiding)
        2. Position momentum is positive (market improving)

        This helps capture more upside during V-shaped recoveries.
        """
        nifty_symbol = "NIFTY 50"
        vix_symbol = "INDIA VIX"

        if nifty_symbol not in self.data:
            return None

        # Get Nifty prices up to as_of_date
        nifty_df = self.data[nifty_symbol]
        ts_date = pd.Timestamp(as_of_date)
        nifty_mask = nifty_df.index <= ts_date
        nifty_available = nifty_df[nifty_mask]

        if len(nifty_available) < 63:  # Need at least 3 months of data
            return None

        nifty_prices = nifty_available["close"]

        # Get VIX history
        vix_history = None
        vix_value = regime.vix_level

        if vix_symbol in self.data:
            vix_df = self.data[vix_symbol]
            vix_mask = vix_df.index <= ts_date
            vix_available = vix_df[vix_mask]
            if len(vix_available) >= 20:
                vix_history = vix_available["close"].iloc[-30:]  # Last 30 days

        # Calculate returns
        return_1m = 0.0
        return_3m = 0.0

        if len(nifty_prices) >= 21:
            price_1m_ago = nifty_prices.iloc[-21]
            return_1m = (nifty_prices.iloc[-1] - price_1m_ago) / price_1m_ago

        if len(nifty_prices) >= 63:
            price_3m_ago = nifty_prices.iloc[-63]
            return_3m = (nifty_prices.iloc[-1] - price_3m_ago) / price_3m_ago

        # Use position momentum from regime if available
        position_momentum = regime.position_momentum if hasattr(regime, "position_momentum") else 0.0

        # Calculate bull recovery signals
        return calculate_bull_recovery_signals(
            nifty_prices=nifty_prices,
            vix_value=vix_value,
            vix_history=vix_history,
            position_momentum=position_momentum,
            return_1m=return_1m,
            return_3m=return_3m,
        )

    def _should_skip_gold(self, as_of_date: datetime) -> Tuple[bool, str]:
        """
        Check if GOLDBEES should be skipped from defensive allocation.

        Two modes controlled by config.regime.gold_skip_logic:
        - "downtrend": Skip gold only if price < 50-SMA (downtrend).
          Volatile gold during crisis is flight-to-safety = good.
        - "volatile": Legacy logic - skip if recent vol > 1.5x average.

        Args:
            as_of_date: Date to check as of

        Returns:
            Tuple of (should_skip, reason_string)
        """
        gold_symbol = self.app_config.regime.gold_symbol

        if gold_symbol not in self.data:
            return (False, "")

        gold_df = self.data[gold_symbol]
        ts_date = pd.Timestamp(as_of_date)
        mask = gold_df.index <= ts_date
        available = gold_df[mask]

        if len(available) < 50:
            return (False, "")  # Not enough data

        skip_logic = self.app_config.regime.gold_skip_logic

        if skip_logic == "downtrend":
            # Skip gold only if it's in a downtrend (below 50-SMA)
            # Volatile gold during crisis is GOOD (flight to safety)
            current_price = available["close"].iloc[-1]
            sma_50 = available["close"].iloc[-50:].mean()
            if current_price < sma_50:
                reason = f"Gold in downtrend: {current_price:.1f} < 50-SMA {sma_50:.1f}"
                return (True, reason)
            return (False, "")
        else:
            # Legacy volatile check
            returns = available["close"].pct_change().dropna()
            recent_vol = returns.iloc[-10:].std() * (252 ** 0.5)
            avg_vol = returns.std() * (252 ** 0.5)
            if recent_vol > avg_vol * 1.5 and recent_vol > 0.15:
                reason = f"Recent vol {recent_vol:.0%} vs normal {avg_vol:.0%} ({recent_vol/avg_vol:.1f}x)"
                return (True, reason)
            return (False, "")

    def _calculate_gold_exhaustion_scale(self, as_of_date: datetime) -> float:
        """
        Calculate gold exhaustion scaling factor (GE1).

        Linearly scales gold allocation from 1.0 to 0.0 as gold moves
        from threshold_low to threshold_high above its 200-SMA.

        Returns:
            Float between 0.0 and 1.0
        """
        regime_config = self.app_config.regime
        if not regime_config.use_gold_exhaustion_scaling:
            return 1.0

        if self._gold_200sma_cache.empty:
            return 1.0

        gold_symbol = regime_config.gold_symbol
        if gold_symbol not in self.data:
            return 1.0

        ts_date = pd.Timestamp(as_of_date)
        if ts_date not in self._gold_200sma_cache.index:
            return 1.0

        sma_val = self._gold_200sma_cache.loc[ts_date]
        if pd.isna(sma_val) or sma_val <= 0:
            return 1.0

        gold_price = self.data[gold_symbol].loc[ts_date, "close"]
        deviation = (gold_price - sma_val) / sma_val

        if deviation <= regime_config.gold_exhaustion_threshold_low:
            return 1.0
        elif deviation >= regime_config.gold_exhaustion_threshold_high:
            return 0.0
        else:
            # Linear interpolation from 1.0 → 0.0
            span = regime_config.gold_exhaustion_threshold_high - regime_config.gold_exhaustion_threshold_low
            return 1.0 - (deviation - regime_config.gold_exhaustion_threshold_low) / span

    def _redirect_freed_weight(
        self, weights: Dict[str, float], freed: float, as_of_date: datetime
    ) -> None:
        """Redirect freed gold weight to equities pro-rata (uptrend) or cash (downtrend)."""
        regime_config = self.app_config.regime
        cash_sym = regime_config.cash_symbol
        if (
            regime_config.redirect_freed_to_equity_in_uptrend
            and self._is_market_uptrend(as_of_date)
        ):
            defensive = {regime_config.gold_symbol, cash_sym}
            equity_weights = {t: w for t, w in weights.items() if t not in defensive and w > 0}
            total_eq = sum(equity_weights.values())
            if total_eq > 0:
                for t, w in equity_weights.items():
                    weights[t] += freed * (w / total_eq)
                return
        weights[cash_sym] = weights.get(cash_sym, 0.0) + freed

    def _calculate_benchmark_returns(
        self,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate benchmark returns over the backtest period.

        Returns:
            Tuple of (nifty_50_return, nifty_midcap_100_return)
        """
        nifty_return = None
        midcap_return = None

        ts_start = pd.Timestamp(self.config.start_date)
        ts_end = pd.Timestamp(self.config.end_date)

        # Nifty 50
        nifty_symbol = "NIFTY 50"
        if nifty_symbol in self.data:
            df = self.data[nifty_symbol]
            mask = (df.index >= ts_start) & (df.index <= ts_end)
            available = df[mask]
            if len(available) >= 2:
                start_price = available["close"].iloc[0]
                end_price = available["close"].iloc[-1]
                nifty_return = (end_price - start_price) / start_price

        # Nifty Midcap 100
        midcap_symbol = "NIFTY MIDCAP 100"
        if midcap_symbol in self.data:
            df = self.data[midcap_symbol]
            mask = (df.index >= ts_start) & (df.index <= ts_end)
            available = df[mask]
            if len(available) >= 2:
                start_price = available["close"].iloc[0]
                end_price = available["close"].iloc[-1]
                midcap_return = (end_price - start_price) / start_price

        return nifty_return, midcap_return

    def _get_price_at_date(
        self,
        symbol: str,
        date: datetime,
        lookback_days: int = 5,
    ) -> Optional[float]:
        """
        Get price for symbol at or before date.

        C7: Only uses data available at the time.

        Args:
            symbol: Stock symbol
            date: Target date
            lookback_days: Days to look back if exact date unavailable

        Returns:
            Price or None if not found
        """
        series = self._close_prices.get(symbol)
        if series is None:
            return None

        ts_date = pd.Timestamp(date)
        val = series.asof(ts_date)
        if pd.isna(val):
            return None
        return float(val)

    def _get_prices_at_date(
        self,
        symbols: List[str],
        date: datetime,
    ) -> Dict[str, float]:
        """Get prices for multiple symbols at date."""
        ts_date = pd.Timestamp(date)
        result = {}
        for s in symbols:
            series = self._close_prices.get(s)
            if series is not None:
                val = series.asof(ts_date)
                if not pd.isna(val):
                    result[s] = float(val)
        return result

    def _get_filtered_data(self, symbol: str, as_of_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get pre-filtered data for a symbol up to as_of_date (cached).

        Performance optimization: caches filtered views to avoid repeated slicing.
        """
        date_str = as_of_date.strftime("%Y-%m-%d")

        # Check date-level cache first
        if date_str not in self._date_filtered_data:
            self._date_filtered_data[date_str] = {}

        date_cache = self._date_filtered_data[date_str]

        if symbol in date_cache:
            return date_cache[symbol]

        # Compute and cache
        if symbol not in self.data:
            return None

        df = self.data[symbol]
        ts_date = pd.Timestamp(as_of_date)
        mask = df.index <= ts_date
        filtered = df[mask]

        date_cache[symbol] = filtered
        return filtered

    def _calculate_stock_nms(
        self,
        ticker: str,
        as_of_date: datetime,
    ) -> Optional[Tuple[float, NMSResult]]:
        """
        Calculate stock NMS using data available at as_of_date.

        C7: No look-ahead - only uses data up to as_of_date.

        Performance: Uses caching to avoid redundant calculations.

        Args:
            ticker: Stock ticker
            as_of_date: Date for calculation

        Returns:
            Tuple of (NMS score, full NMSResult) or None
        """
        # Check cache first
        cache_key = f"{ticker}_{as_of_date.strftime('%Y-%m-%d')}"
        if cache_key in self._nms_cache:
            self._nms_cache_hits += 1
            return self._nms_cache[cache_key]

        self._nms_cache_misses += 1

        stock = self.universe.get_stock(ticker)
        if not stock or stock.zerodha_symbol not in self.data:
            return None

        # Use cached filtered data
        available = self._get_filtered_data(stock.zerodha_symbol, as_of_date)

        if available is None or len(available) < 280:  # Need 252 + buffer
            return None

        prices = available["close"]
        volumes = available["volume"] if "volume" in available.columns else pd.Series([1e6] * len(prices), index=prices.index)

        # Get config values, with backtest config overrides taking precedence
        pm = self.app_config.pure_momentum

        # Use backtest config overrides if provided
        weight_6m = self.config.weight_6m if self.config.weight_6m is not None else pm.weight_6m
        weight_12m = self.config.weight_12m if self.config.weight_12m is not None else pm.weight_12m

        result = calculate_normalized_momentum_score(
            prices=prices,
            volumes=volumes,
            lookback_6m=pm.lookback_6m,
            lookback_12m=pm.lookback_12m,
            lookback_volatility=pm.lookback_volatility,
            skip_recent_days=pm.skip_recent_days,
            weight_6m=weight_6m,
            weight_12m=weight_12m,
        )

        if result is None:
            return None

        # Cache the result
        cached_result = (result.nms, result)
        self._nms_cache[cache_key] = cached_result

        return cached_result

    def _select_momentum_portfolio(
        self,
        as_of_date: datetime,
        regime: Optional[RegimeResult] = None,
    ) -> Dict[str, float]:
        """
        Select portfolio using pure momentum (NMS) ranking.

        C7: No look-ahead bias.

        If regime is provided and not BULLISH/NORMAL, equity weights are scaled
        down and defensive positions (gold + cash symbols) are added.

        Args:
            as_of_date: Date for calculation
            regime: Optional regime detection result for defensive allocation

        Returns:
            Dict mapping ticker to target weight
        """
        all_stocks = self.universe.get_all_stocks()

        # Excluded symbols
        excluded = set(self.app_config.excluded_symbols)

        # Calculate NMS for all stocks using parallel processing
        stock_nms: List[Tuple[str, str, float, NMSResult]] = []  # ticker, sector, nms, result

        # Filter eligible stocks first
        eligible_stocks = [s for s in all_stocks if s.ticker not in excluded]

        # Use ThreadPoolExecutor for parallel NMS calculation
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._calculate_stock_nms, stock.ticker, as_of_date): stock
                for stock in eligible_stocks
            }

            for future in as_completed(futures):
                stock = futures[future]
                try:
                    nms_data = future.result()
                    if nms_data:
                        nms_score, nms_result = nms_data
                        stock_nms.append((stock.ticker, stock.sector, nms_score, nms_result))
                except Exception:
                    continue

        if not stock_nms:
            return {}

        # Sort by NMS descending
        stock_nms.sort(key=lambda x: x[2], reverse=True)

        # Calculate percentiles
        total_stocks = len(stock_nms)

        # Apply entry filters and select top stocks with sector diversification
        selected = []
        sector_counts: Dict[str, int] = {}  # Track stocks per sector
        max_per_sector = self.config.max_stocks_per_sector

        for i, (ticker, sector, nms, result) in enumerate(stock_nms):
            percentile = 100 * (total_stocks - i) / total_stocks

            # Check percentile threshold
            if percentile < self.config.min_score_percentile:
                if len(selected) >= self.config.min_positions:
                    break

            # Check sector limit (if enabled)
            if max_per_sector > 0:
                current_sector_count = sector_counts.get(sector, 0)
                if current_sector_count >= max_per_sector:
                    # Skip this stock - sector is full
                    continue

            # Check entry filters (must match MomentumEngine filters)
            passes_filters = (
                result.high_52w_proximity >= self.config.min_52w_high_prox
                and result.above_50ema
                and result.above_200sma
                and result.volume_surge >= self.config.min_volume_ratio
                and result.daily_turnover >= self.config.min_daily_turnover
            )

            if passes_filters or len(selected) < self.config.min_positions:
                selected.append((ticker, sector, nms))
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            if len(selected) >= self.config.target_positions:
                break

        if not selected:
            return {}

        # Calculate weights (momentum-weighted)
        min_nms = min(s[2] for s in selected)
        adjusted_scores = [s[2] - min_nms + 0.01 for s in selected]
        total_score = sum(adjusted_scores)

        # Position sizing config
        ps = self.app_config.position_sizing
        max_weight = ps.max_single_position
        min_weight = ps.min_single_position
        max_sector = ps.max_sector_exposure

        weights = {}
        for i, (ticker, sector, nms) in enumerate(selected):
            raw_weight = adjusted_scores[i] / total_score
            weight = max(min_weight, min(max_weight, raw_weight))
            weights[ticker] = weight

        # Normalize with iterative capping to maintain position limits
        # Uses shared utility for logic parity with live rebalance (momentum_engine.py)
        weights = renormalize_with_caps(weights, max_weight, min_weight)

        # Apply sector limits by capping overweight sectors
        ticker_sectors = {t: s for t, s, _ in selected}
        capped_sectors: set = set()  # Sectors that have been capped

        # Iteratively cap sectors and redistribute to uncapped sectors
        max_iterations = 10
        for _ in range(max_iterations):
            # Calculate current sector weights
            sector_weights: Dict[str, float] = {}
            for ticker, weight in weights.items():
                sector = ticker_sectors[ticker]
                sector_weights[sector] = sector_weights.get(sector, 0) + weight

            # Find newly overweight sectors (excluding already capped)
            newly_capped = False
            for sector, sw in sector_weights.items():
                if sector in capped_sectors:
                    continue
                if sw > max_sector:
                    # Cap this sector
                    capped_sectors.add(sector)
                    scale = max_sector / sw
                    for ticker in weights:
                        if ticker_sectors[ticker] == sector:
                            weights[ticker] *= scale
                    newly_capped = True

            if not newly_capped:
                break  # No new sectors to cap

            # Normalize only uncapped sectors to make weights sum to 1.0
            capped_total = sum(
                w for t, w in weights.items()
                if ticker_sectors[t] in capped_sectors
            )
            uncapped_total = sum(
                w for t, w in weights.items()
                if ticker_sectors[t] not in capped_sectors
            )

            # Uncapped sectors should sum to (1.0 - capped_total)
            target_uncapped = 1.0 - capped_total
            if uncapped_total > 0 and target_uncapped > 0:
                scale = target_uncapped / uncapped_total
                for ticker in weights:
                    if ticker_sectors[ticker] not in capped_sectors:
                        weights[ticker] *= scale

        # Re-enforce per-position caps after sector capping may have inflated weights
        weights = renormalize_with_caps(weights, max_weight, min_weight)

        # Apply regime-based allocation adjustments
        if regime and regime.equity_weight < 1.0:
            # Scale down equity weights
            equity_scale = regime.equity_weight
            for ticker in weights:
                weights[ticker] *= equity_scale

            # Add defensive positions (with volatility check for parity with live mode)
            regime_config = self.app_config.regime
            if regime.gold_weight > 0:
                gold_volatile, _ = self._should_skip_gold(as_of_date)
                if gold_volatile:
                    # Skip GOLDBEES, redistribute to equity (same as live mode)
                    redistrib_scale = 1.0 / (1.0 - regime.gold_weight)
                    for ticker in weights:
                        weights[ticker] *= redistrib_scale
                else:
                    # Apply gold exhaustion scaling (GE1)
                    gold_exhaust_scale = self._calculate_gold_exhaustion_scale(as_of_date)
                    effective_gold = regime.gold_weight * gold_exhaust_scale
                    if effective_gold > 0.001:
                        weights[regime_config.gold_symbol] = effective_gold
                    # Freed gold weight → equities (uptrend) or cash (downtrend)
                    freed_gold = regime.gold_weight - effective_gold
                    if freed_gold > 0.001:
                        self._redirect_freed_weight(weights, freed_gold, as_of_date)
            if regime.cash_weight > 0:
                weights[regime_config.cash_symbol] = weights.get(regime_config.cash_symbol, 0.0) + regime.cash_weight

        return weights

    def _calculate_vol_scale(self) -> float:
        """
        Calculate portfolio-level volatility scaling factor (E2).

        Scales equity inversely to realized portfolio vol:
        vol_scale = clamp(target_vol / realized_vol, floor, 1.0)

        When vol spikes to 30% (crisis), scale = 15/30 = 0.50 → halves equity.
        When vol is normal (12-15%), scale = 1.0 → no drag.

        Returns:
            Float between vol_scale_floor and 1.0
        """
        regime_config = self.app_config.regime
        if not regime_config.use_vol_targeting:
            return 1.0

        lookback = regime_config.vol_lookback_days
        if len(self._portfolio_daily_returns) < lookback:
            return 1.0  # Not enough history yet

        recent_returns = list(self._portfolio_daily_returns)[-lookback:]
        realized_vol = np.std(recent_returns) * (252 ** 0.5)

        if realized_vol < 0.01:
            return 1.0  # Avoid division by near-zero

        target_vol = regime_config.target_portfolio_vol
        raw_scale = target_vol / realized_vol
        return min(1.0, max(regime_config.vol_scale_floor, raw_scale))

    def _calculate_breadth_scale(self, as_of_date: datetime) -> float:
        """
        Calculate breadth-based exposure scaling factor (E3).

        Linear interpolation between breadth_low and breadth_full:
        breadth >= full → 1.0 (broad rally)
        breadth <= low  → min_scale (narrow market)
        between → linear interpolation

        Returns:
            Float between breadth_min_scale and 1.0
        """
        regime_config = self.app_config.regime
        if not regime_config.use_breadth_scaling:
            return 1.0

        breadth = self._calculate_market_breadth(as_of_date)

        if breadth >= regime_config.breadth_full:
            return 1.0
        elif breadth <= regime_config.breadth_low:
            return regime_config.breadth_min_scale
        else:
            # Linear interpolation
            t = (breadth - regime_config.breadth_low) / (regime_config.breadth_full - regime_config.breadth_low)
            return regime_config.breadth_min_scale + t * (1.0 - regime_config.breadth_min_scale)

    def _get_effective_sector_cap(self, regime: Optional[RegimeResult]) -> float:
        """
        Get effective sector cap based on regime (E4).

        BULLISH/NORMAL: use default max_sector_exposure (30%)
        CAUTION: use caution_max_sector (25%)
        DEFENSIVE: use defensive_max_sector (20%)

        Returns:
            Float representing maximum sector weight
        """
        sizing_config = self.app_config.position_sizing
        if not sizing_config.use_dynamic_sector_caps or regime is None:
            return sizing_config.max_sector_exposure

        if regime.regime == MarketRegime.DEFENSIVE:
            return sizing_config.defensive_max_sector
        elif regime.regime == MarketRegime.CAUTION:
            return sizing_config.caution_max_sector
        else:
            return sizing_config.max_sector_exposure

    def _select_portfolio_via_strategy(
        self,
        as_of_date: datetime,
        regime: Optional[RegimeResult] = None,
        portfolio_value: Optional[float] = None,
        profile_max_gold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Select portfolio using the pluggable strategy interface.

        C7: No look-ahead bias.

        Args:
            as_of_date: Date for calculation
            regime: Optional regime detection result for defensive allocation
            portfolio_value: Current portfolio value (defaults to initial_capital)

        Returns:
            Dict mapping ticker to target weight
        """
        # Set the as_of_date for the market data adapter
        self._market_data_adapter.set_as_of_date(as_of_date)

        # Pass regime to strategy for adaptive parameters (AllWeather only)
        if hasattr(self.strategy, "set_regime") and regime is not None:
            self.strategy.set_regime(regime)

        # Pass drawdown to strategy for recovery mode detection (AllWeather only)
        if hasattr(self.strategy, "set_drawdown"):
            self.strategy.set_drawdown(self._current_drawdown, as_of_date)

        # Get current positions for the strategy
        current_positions: Dict[str, float] = {}

        # Rank stocks using the strategy
        ranked_stocks = self.strategy.rank_stocks(
            as_of_date=as_of_date,
            universe=self.universe,
            market_data=self._market_data_adapter,
            filter_entry=True,
        )

        if not ranked_stocks:
            return {}

        # Trail: capture ranked stocks for the trail log
        self._trail_ranked_stocks = ranked_stocks

        # Select portfolio using the strategy
        effective_value = portfolio_value if portfolio_value is not None else self.config.initial_capital
        weights = self.strategy.select_portfolio(
            ranked_stocks=ranked_stocks,
            portfolio_value=effective_value,
            current_positions=current_positions,
            max_positions=self.config.target_positions,
            max_per_sector=self.config.max_stocks_per_sector,
        )

        # Apply sector weight limits with dynamic caps (E4)
        max_sector_exposure = self._get_effective_sector_cap(regime)
        ticker_to_sector = {s.ticker: s.sector for s in ranked_stocks}
        # Include defensive symbols (gold + cash) from universe
        for ticker in weights:
            if ticker not in ticker_to_sector:
                stock = self.universe.get_stock(ticker)
                ticker_to_sector[ticker] = stock.sector if stock else "UNKNOWN"
        capped_sectors: set = set()
        for _ in range(10):  # Iterative capping
            sector_weights: Dict[str, float] = {}
            for ticker, w in weights.items():
                sec = ticker_to_sector.get(ticker, "UNKNOWN")
                sector_weights[sec] = sector_weights.get(sec, 0) + w
            newly_capped = False
            for sec, sw in sector_weights.items():
                if sec in capped_sectors or sw <= max_sector_exposure:
                    continue
                capped_sectors.add(sec)
                scale = max_sector_exposure / sw
                for ticker in weights:
                    if ticker_to_sector.get(ticker) == sec:
                        weights[ticker] *= scale
                newly_capped = True
            if not newly_capped:
                break
            # Redistribute excess to uncapped tickers
            capped_total = sum(w for t, w in weights.items() if ticker_to_sector.get(t) in capped_sectors)
            uncapped_total = sum(w for t, w in weights.items() if ticker_to_sector.get(t) not in capped_sectors)
            target_uncapped = 1.0 - capped_total
            if uncapped_total > 0 and target_uncapped > 0:
                scale_up = target_uncapped / uncapped_total
                for ticker in weights:
                    if ticker_to_sector.get(ticker) not in capped_sectors:
                        weights[ticker] *= scale_up

        # Per-profile gold cap: clamp regime gold weight before allocation
        if regime and profile_max_gold is not None and regime.gold_weight > profile_max_gold:
            freed = regime.gold_weight - profile_max_gold
            regime.gold_weight = profile_max_gold
            regime.equity_weight = min(1.0, regime.equity_weight + freed)
            regime.cash_weight = max(0.0, 1.0 - regime.equity_weight - regime.gold_weight)

        # Change 5: Recovery equity override — cap stress when drawdown + improving breadth
        regime_cfg = self.app_config.regime
        if (regime and regime_cfg.use_recovery_equity_override
                and self._current_drawdown < regime_cfg.recovery_override_dd_threshold):
            breadth_improving = False
            lookback_days = 10
            current_b = self._get_cached_breadth(as_of_date)
            past_date = self._get_past_trading_date(as_of_date, lookback_days)
            if past_date is not None:
                past_b = self._get_cached_breadth(past_date)
                if current_b - past_b > regime_cfg.recovery_override_breadth_improvement:
                    breadth_improving = True
            if breadth_improving:
                import math
                capped_stress = min(regime.stress_score, regime_cfg.recovery_override_max_stress)
                steepness = regime_cfg.allocation_curve_steepness
                stress_curve = math.pow(capped_stress, 1.0 / steepness) if steepness > 0 else capped_stress
                override_equity = 1.0 - stress_curve * (1.0 - regime_cfg.min_equity_allocation)
                effective_max_gold = profile_max_gold if profile_max_gold is not None else regime_cfg.max_gold_allocation
                override_gold = stress_curve * effective_max_gold
                regime.equity_weight = max(regime.equity_weight, override_equity)
                regime.gold_weight = min(regime.gold_weight, override_gold)
                regime.cash_weight = max(0.0, 1.0 - regime.equity_weight - regime.gold_weight)

        # Apply regime-based allocation adjustments
        if regime and regime.equity_weight < 1.0:
            equity_scale = regime.equity_weight
            for ticker in weights:
                weights[ticker] *= equity_scale

            # Add defensive positions (with gold skip check)
            regime_config = self.app_config.regime
            if regime.gold_weight > 0:
                gold_volatile, _ = self._should_skip_gold(as_of_date)
                if gold_volatile:
                    # Skip GOLDBEES, redistribute to equity (same as live mode)
                    redistrib_scale = 1.0 / (1.0 - regime.gold_weight)
                    for ticker in weights:
                        weights[ticker] *= redistrib_scale
                else:
                    # Apply gold exhaustion scaling (GE1)
                    gold_exhaust_scale = self._calculate_gold_exhaustion_scale(as_of_date)
                    effective_gold = regime.gold_weight * gold_exhaust_scale
                    if effective_gold > 0.001:
                        weights[regime_config.gold_symbol] = effective_gold
                    # Freed gold weight → equities (uptrend) or cash (downtrend)
                    freed_gold = regime.gold_weight - effective_gold
                    if freed_gold > 0.001:
                        self._redirect_freed_weight(weights, freed_gold, as_of_date)
            if regime.cash_weight > 0:
                weights[regime_config.cash_symbol] = weights.get(regime_config.cash_symbol, 0.0) + regime.cash_weight

        # Apply portfolio-level vol targeting (E2) and breadth scaling (E3)
        # These scale equity weights down when vol is high or breadth is narrow
        vol_scale = self._calculate_vol_scale()
        breadth_scale = self._calculate_breadth_scale(as_of_date)

        # Trail: capture scaling factors
        self._trail_vol_scale = vol_scale
        self._trail_breadth_scale = breadth_scale

        # Combined scale with trend-aware floor to prevent over-de-risking
        combined_scale = vol_scale * breadth_scale
        if self._is_market_uptrend(as_of_date):
            combined_floor = self.app_config.regime.trend_scale_floor
        else:
            combined_floor = self.app_config.regime.combined_scale_floor
        effective_scale = max(combined_floor, combined_scale)

        if effective_scale < 1.0:
            # Scale down equity positions only (not gold/cash defensive symbols)
            regime_config = self.app_config.regime
            defensive_symbols = {regime_config.gold_symbol, regime_config.cash_symbol}
            equity_before = sum(w for t, w in weights.items() if t not in defensive_symbols)
            for ticker in list(weights.keys()):
                if ticker not in defensive_symbols:
                    weights[ticker] *= effective_scale
            # Redirect freed equity to cash_symbol (Change 1)
            equity_after = sum(w for t, w in weights.items() if t not in defensive_symbols)
            freed_weight = equity_before - equity_after
            if freed_weight > 0.01:
                cash_sym = self.app_config.regime.cash_symbol
                weights[cash_sym] = weights.get(cash_sym, 0) + freed_weight

        # Catch-all: redirect any unallocated weight to cash_symbol
        # (e.g. when fewer stocks qualify than needed to fill 100% at max_single_position)
        total_weight = sum(weights.values())
        if total_weight < 0.999:
            shortfall = 1.0 - total_weight
            cash_sym = self.app_config.regime.cash_symbol
            weights[cash_sym] = weights.get(cash_sym, 0) + shortfall

        return weights

    def _calculate_market_breadth(self, as_of_date: datetime) -> float:
        """
        Calculate market breadth (% of stocks above 50-day MA).

        This is used for breadth thrust detection.

        OPTIMIZED: Uses pre-computed cache when available for O(1) lookup.
        Falls back to on-demand calculation if cache miss (shouldn't happen
        in normal backtest flow).

        Args:
            as_of_date: Date to calculate breadth for

        Returns:
            Float from 0 to 1 representing % of stocks above 50-day MA
        """
        # Use pre-computed cache if available (O(1) lookup)
        if self._caches_initialized:
            return self._get_cached_breadth(as_of_date)

        # Fallback: compute on-demand (O(n×m) - only for edge cases)
        above_50ma = 0
        valid_stocks = 0

        for symbol, df in self.data.items():
            # Skip indices and special symbols
            if symbol in self._excluded_set:
                continue

            if df.empty or len(df) < 50:
                continue

            try:
                ts_date = pd.Timestamp(as_of_date)
                mask = df.index <= ts_date
                available = df[mask]

                if len(available) < 50:
                    continue

                close = available["close"].iloc[-1]
                ma_50 = available["close"].iloc[-50:].mean()

                valid_stocks += 1
                if close > ma_50:
                    above_50ma += 1

            except Exception:
                continue

        return above_50ma / valid_stocks if valid_stocks > 0 else 0.5

    def _check_dynamic_rebalance(
        self,
        date: datetime,
        days_since_last: int,
        current_regime: Optional[RegimeResult],
        previous_regime: Optional[MarketRegime],
        portfolio_drawdown: float,
    ) -> RebalanceTrigger:
        """
        Check if dynamic rebalancing should be triggered.

        Evaluates multiple event-driven triggers beyond fixed intervals.

        Args:
            date: Current date
            days_since_last: Trading days since last rebalance
            current_regime: Current market regime
            previous_regime: Previous regime for transition detection
            portfolio_drawdown: Current portfolio drawdown (negative number)

        Returns:
            RebalanceTrigger with decision and reason
        """
        dynamic_config = self.app_config.dynamic_rebalance

        # If dynamic rebalancing is disabled, only use fixed interval
        if not dynamic_config.enabled:
            should_rebal = days_since_last >= self.config.rebalance_days
            return RebalanceTrigger(
                should_rebalance=should_rebal,
                reason=f"Fixed interval ({self.config.rebalance_days} days)",
                days_since_last=days_since_last,
                triggers_fired=["REGULAR_INTERVAL"] if should_rebal else [],
                urgency="MEDIUM" if should_rebal else "LOW",
            )

        # Get VIX level
        vix_level = 15.0
        if current_regime is not None:
            vix_level = current_regime.vix_level

        # Get market 1-month return (O(1) from pre-computed cache)
        market_1m_return = self._get_cached_market_return(date)

        # Check breadth thrust
        breadth_thrust = False
        if dynamic_config.breadth_thrust_trigger and len(self._breadth_history) >= 11:
            breadth_series = pd.Series(list(self._breadth_history))  # Convert deque to list for Series
            thrust_result = detect_breadth_thrust(
                breadth_history=breadth_series,
                thrust_low=dynamic_config.breadth_thrust_low,
                thrust_high=dynamic_config.breadth_thrust_high,
                max_days=dynamic_config.breadth_thrust_days,
            )
            breadth_thrust = thrust_result.is_thrust

        # Map current regime to MarketRegime enum if available
        current_regime_enum = current_regime.regime if current_regime else None

        return should_trigger_rebalance(
            days_since_last=days_since_last,
            current_regime=current_regime_enum,
            previous_regime=previous_regime,
            vix_level=vix_level,
            vix_peak_20d=self._vix_peak_20d,
            portfolio_drawdown=portfolio_drawdown,
            market_1m_return=market_1m_return,
            breadth_thrust=breadth_thrust,
            min_days_between=dynamic_config.min_days_between,
            max_days_between=dynamic_config.max_days_between,
            vix_recovery_decline=dynamic_config.vix_recovery_decline,
            vix_spike_threshold=dynamic_config.vix_spike_threshold,
            drawdown_threshold=dynamic_config.drawdown_threshold,
            crash_threshold=dynamic_config.crash_threshold,
        )

    def _update_dynamic_state(
        self,
        date: datetime,
        vix_level: float,
    ) -> None:
        """
        Update state tracking for dynamic rebalancing.

        OPTIMIZED: Uses pre-computed caches for O(1) lookups.
        Updates VIX peak and breadth history.
        """
        # Update VIX peak - use cached value if available (O(1))
        if self._caches_initialized:
            cached_peak = self._get_cached_vix_peak(date)
            self._vix_peak_20d = max(cached_peak, vix_level)
        else:
            self._vix_peak_20d = max(self._vix_peak_20d, vix_level)

        # Calculate and track breadth (O(1) via cache)
        current_breadth = self._calculate_market_breadth(date)
        # deque with maxlen=15 handles trimming automatically (O(1) append)
        self._breadth_history.append(current_breadth)

    def run(self) -> BacktestResult:
        """
        Execute backtest simulation.

        C7: Uses only data available at each decision point.

        Supports both fixed-interval and dynamic (event-driven) rebalancing.

        Returns:
            BacktestResult with performance metrics
        """
        portfolio = BacktestPortfolio(self.config.initial_capital, self.universe)

        # Get all trading days for dynamic rebalancing
        trading_days = self._get_trading_days()
        ts_start = pd.Timestamp(self.config.start_date)
        ts_end = pd.Timestamp(self.config.end_date)
        mask = (trading_days >= ts_start) & (trading_days <= ts_end)
        all_trading_days = [d.to_pydatetime() for d in trading_days[mask]]

        # Build trading day index for O(1) trading-day distance lookups
        trading_day_index = {d: i for i, d in enumerate(all_trading_days)}

        # For fixed rebalancing, get the scheduled dates
        fixed_rebalance_dates = self._get_rebalance_dates()

        equity_values: List[tuple] = []
        trades: List[Trade] = []
        sector_history: List[dict] = []

        # Stop loss tracking with strategy support
        # Stores entry trading day index for accurate days_held calculation
        stop_loss_entries: Dict[str, Tuple[float, float, int]] = {}  # ticker -> (entry_price, peak_price, entry_day_idx)

        # Regime tracking
        regime_history: List[dict] = []
        prev_regime: Optional[MarketRegime] = None

        # Reset previous regime result for hysteresis tracking
        self._previous_regime_result = None

        # Drawdown tracking for AllWeather recovery mode
        peak_value = self.config.initial_capital
        self._current_drawdown = 0.0

        # Clear caches for fresh run
        self._nms_cache.clear()
        self._date_filtered_data.clear()
        self._nms_cache_hits = 0
        self._nms_cache_misses = 0

        # Dynamic rebalancing state
        self._last_rebalance_date = None
        self._days_since_rebalance = 0
        self._vix_peak_20d = 15.0
        self._breadth_history = deque(maxlen=15)  # O(1) append with auto-trim
        self._rebalance_triggers_log = []

        # Portfolio daily returns for vol targeting (E2)
        self._portfolio_daily_returns: Deque[float] = deque(maxlen=63)
        self._prev_portfolio_value = self.config.initial_capital

        # Rebalance trail log
        rebalance_trail: List[RebalanceRecord] = []
        rebalance_number = 0

        # PERFORMANCE OPTIMIZATION: Pre-compute expensive calculations
        # This is the key optimization for dynamic rebalancing (60x speedup)
        use_dynamic = self.app_config.dynamic_rebalance.enabled
        if use_dynamic:
            self._caches_initialized = False  # Reset for fresh computation
            self._initialize_caches()

        # Always iterate all trading days for daily equity tracking, stop loss
        # checking, and accurate drawdown/Sharpe calculations.
        # For fixed rebalancing, determine rebalance eligibility per-date.
        fixed_rebalance_set = set(fixed_rebalance_dates) if not use_dynamic else set()

        for date in all_trading_days:
            # Update prices
            all_symbols = list(self.data.keys())
            prices = self._get_prices_at_date(all_symbols, date)
            portfolio.update_prices(prices)

            # Record equity
            current_value = portfolio.get_total_value()
            equity_values.append((date, current_value))

            # Track daily returns for vol targeting (E2)
            if self._prev_portfolio_value > 0:
                daily_ret = (current_value - self._prev_portfolio_value) / self._prev_portfolio_value
                self._portfolio_daily_returns.append(daily_ret)
            self._prev_portfolio_value = current_value

            # Update drawdown tracking for AllWeather recovery mode
            if current_value > peak_value:
                peak_value = current_value
            self._current_drawdown = (current_value - peak_value) / peak_value if peak_value > 0 else 0.0

            # Increment days since last rebalance (in trading days, not calendar days)
            if self._last_rebalance_date is not None:
                self._days_since_rebalance = trading_day_index[date] - trading_day_index[self._last_rebalance_date]
            else:
                self._days_since_rebalance = self.app_config.dynamic_rebalance.max_days_between  # Force first rebalance

            # Daily stop loss check: update peak prices and sell breached positions
            # This runs every trading day, independent of rebalancing schedule
            if self.config.use_stop_loss and portfolio.positions:
                for symbol in list(portfolio.positions.keys()):
                    if symbol not in stop_loss_entries:
                        continue
                    entry_price, peak_price, entry_idx = stop_loss_entries[symbol]
                    current_price = prices.get(symbol, 0)
                    if current_price <= 0:
                        continue

                    # Update peak price daily (not just on rebalance days)
                    if current_price > peak_price:
                        peak_price = current_price
                        stop_loss_entries[symbol] = (entry_price, peak_price, entry_idx)

                    days_held = trading_day_index[date] - entry_idx
                    gain_from_entry = (current_price - entry_price) / entry_price
                    stop_config = self.strategy.get_stop_loss_config(symbol, gain_from_entry)

                    triggered = False
                    # Check initial stop loss (always active, even during min hold)
                    if gain_from_entry <= -stop_config.initial_stop:
                        triggered = True
                    # E9: Skip trailing stop during minimum hold period
                    elif days_held >= self._min_hold_days and gain_from_entry >= stop_config.trailing_activation:
                        gain_from_peak = (current_price - peak_price) / peak_price
                        if gain_from_peak <= -stop_config.trailing_stop:
                            triggered = True

                    if triggered:
                        # Immediate sell on stop loss breach
                        pos = portfolio.positions[symbol]
                        price = prices.get(symbol, pos.current_price)
                        if price > 0:
                            cost = pos.value * self.config.transaction_cost
                            portfolio.sell(symbol, pos.quantity, price)
                            portfolio.cash -= cost
                            trades.append(
                                Trade(
                                    date=date,
                                    symbol=symbol,
                                    sector=pos.sector,
                                    action="SELL",
                                    quantity=pos.quantity,
                                    price=price,
                                    value=pos.quantity * price,
                                    cost=cost,
                                )
                            )
                            stop_loss_entries.pop(symbol, None)

            # PERFORMANCE OPTIMIZATION: Fast O(1) check to skip ~85% of days early
            # This avoids expensive regime detection and trigger evaluation on most days
            if use_dynamic and not self._should_check_rebalance_fast(date, self._days_since_rebalance):
                # Update breadth history even on skipped days (O(1) from cache)
                current_breadth = self._get_cached_breadth(date)
                self._breadth_history.append(current_breadth)
                continue

            # Detect market regime (uses self._previous_regime_result for hysteresis)
            # This is expensive, so only done when fast check passes
            regime = self._detect_regime(date)

            # Update dynamic state (VIX peak, breadth) on days we're checking
            if use_dynamic and regime:
                self._update_dynamic_state(date, regime.vix_level)

            # Check if we should rebalance
            should_rebalance = False
            rebalance_reason = ""

            if use_dynamic:
                # Dynamic rebalancing - check triggers (using cached values)
                trigger_result = self._check_dynamic_rebalance(
                    date=date,
                    days_since_last=self._days_since_rebalance,
                    current_regime=regime,
                    previous_regime=prev_regime,
                    portfolio_drawdown=self._current_drawdown,
                )
                should_rebalance = trigger_result.should_rebalance
                rebalance_reason = trigger_result.reason

                if should_rebalance:
                    self._rebalance_triggers_log.append({
                        "date": date,
                        "reason": rebalance_reason,
                        "triggers_fired": trigger_result.triggers_fired,
                        "urgency": trigger_result.urgency,
                        "days_since_last": self._days_since_rebalance,
                    })
            else:
                # Fixed rebalancing - rebalance only on scheduled dates
                if date in fixed_rebalance_set:
                    should_rebalance = True
                    rebalance_reason = f"Fixed interval ({self.config.rebalance_days} days)"

            # Skip if not rebalancing
            if not should_rebalance:
                continue

            # Update last rebalance date
            self._last_rebalance_date = date

            if regime:
                # Store for next iteration's hysteresis
                self._previous_regime_result = regime

                # IMPORTANT: Set regime on strategy BEFORE update_bull_recovery_state
                # so that crash recovery mode can access current VIX level
                if hasattr(self.strategy, "set_regime"):
                    self.strategy.set_regime(regime)

                # Enhanced regime history with new fields
                regime_history.append({
                    "date": date,
                    "regime": regime.regime.value,
                    "nifty_52w_position": regime.nifty_52w_position,
                    "vix_level": regime.vix_level,
                    "nifty_3m_return": regime.nifty_3m_return,
                    "equity_weight": regime.equity_weight,
                    "gold_weight": regime.gold_weight,
                    "cash_weight": regime.cash_weight,
                    # New multi-timeframe fields
                    "position_short": regime.position_short,
                    "position_medium": regime.position_medium,
                    "position_long": regime.position_long,
                    "return_1m": regime.return_1m,
                    "stress_score": regime.stress_score,
                    # Hysteresis tracking
                    "pending_regime": regime.pending_regime.value if regime.pending_regime else None,
                    "confirmation_days": regime.confirmation_days,
                    "transition_blocked": regime.transition_blocked,
                })
                prev_regime = regime.regime

                # Calculate and pass bull recovery signals to strategy
                # This enables bull recovery mode during V-shaped recoveries
                bull_recovery_signals = self._calculate_bull_recovery_signals(date, regime)
                if bull_recovery_signals and hasattr(self.strategy, "update_bull_recovery_state"):
                    self.strategy.update_bull_recovery_state(date, bull_recovery_signals)

                # Update market mode for hybrid strategy adaptive scoring
                # This enables detection of strong bull markets and gradual corrections
                if hasattr(self.strategy, "update_market_mode"):
                    self.strategy.update_market_mode(self.data, date)

                # NEW: Update crash avoidance state for SimpleStrategy
                if hasattr(self.strategy, "update_crash_avoidance_state"):
                    nifty_symbol = "NIFTY 50"
                    vix_symbol = "INDIA VIX"
                    if nifty_symbol in self.data:
                        nifty_df = self.data[nifty_symbol]
                        ts_date = pd.Timestamp(date)
                        nifty_mask = nifty_df.index <= ts_date
                        nifty_available = nifty_df[nifty_mask]
                        if len(nifty_available) >= 63:
                            nifty_prices = nifty_available["close"]
                            vix_history = None
                            if vix_symbol in self.data:
                                vix_df = self.data[vix_symbol]
                                vix_mask = vix_df.index <= ts_date
                                vix_available = vix_df[vix_mask]
                                if len(vix_available) >= 5:
                                    vix_history = vix_available["close"]
                            self.strategy.update_crash_avoidance_state(
                                as_of_date=date,
                                market_prices=nifty_prices,
                                vix_level=regime.vix_level,
                                vix_history=vix_history,
                            )

                # NEW: Update adaptive lookback for SimpleStrategy
                if hasattr(self.strategy, "update_adaptive_lookback"):
                    self.strategy.update_adaptive_lookback(regime.vix_level)

                # NEW: Update breadth state for SimpleStrategy
                if hasattr(self.strategy, "update_breadth_state"):
                    current_breadth = self._calculate_market_breadth(date)
                    self.strategy.update_breadth_state(current_breadth)

            # Get target portfolio using strategy interface
            target_weights = self._select_portfolio_via_strategy(
                date, regime, portfolio_value=portfolio.get_total_value(),
                profile_max_gold=self._profile_max_gold,
            )

            if not target_weights:
                continue

            # Note: stop losses are now checked daily (above) with immediate sells.
            # Positions stopped out before this rebalance are already sold.

            # Apply crash avoidance position scaling if strategy supports it
            position_scale = 1.0
            if hasattr(self.strategy, "get_position_scale"):
                position_scale = self.strategy.get_position_scale()
                if position_scale < 1.0:
                    # Scale down all equity positions, keep defensive positions full
                    defensive_syms = {self.app_config.regime.gold_symbol, self.app_config.regime.cash_symbol}
                    eq_before = sum(w for t, w in target_weights.items() if t not in defensive_syms)
                    for ticker in list(target_weights.keys()):
                        if ticker not in defensive_syms:
                            target_weights[ticker] *= position_scale
                    # Redirect freed equity to cash_symbol (Change 1)
                    eq_after = sum(w for t, w in target_weights.items() if t not in defensive_syms)
                    freed = eq_before - eq_after
                    if freed > 0.01:
                        cash_sym = self.app_config.regime.cash_symbol
                        target_weights[cash_sym] = target_weights.get(cash_sym, 0) + freed

            # Calculate target positions
            # FIX: Use total portfolio value (positions + cash) for sizing
            # This ensures cash gets reinvested on rebalance, not left idle
            # The strategy weights already account for allocation decisions
            portfolio_value = portfolio.get_total_value()
            target_positions: Dict[str, int] = {}

            for ticker, weight in target_weights.items():
                if ticker not in prices or prices[ticker] <= 0:
                    continue
                target_value = portfolio_value * weight
                target_qty = int(target_value / prices[ticker])
                if target_qty > 0:
                    target_positions[ticker] = target_qty

            # Track trades for this rebalance (for trail log)
            trades_before = len(trades)

            # Execute sells first (full liquidation of positions not in target)
            for symbol in list(portfolio.positions.keys()):
                if symbol not in target_positions:
                    pos = portfolio.positions[symbol]
                    price = prices.get(symbol, pos.current_price)
                    if price > 0:
                        cost = pos.value * self.config.transaction_cost
                        portfolio.sell(symbol, pos.quantity, price)
                        portfolio.cash -= cost
                        trades.append(
                            Trade(
                                date=date,
                                symbol=symbol,
                                sector=pos.sector,
                                action="SELL",
                                quantity=pos.quantity,
                                price=price,
                                value=pos.quantity * price,
                                cost=cost,
                            )
                        )
                        # Remove from stop loss tracking
                        stop_loss_entries.pop(symbol, None)

            # Execute partial sells (reduce positions that exceed target)
            for ticker, target_qty in target_positions.items():
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    current_qty = pos.quantity
                    if current_qty > target_qty:
                        # Sell excess shares
                        sell_qty = current_qty - target_qty
                        price = prices.get(ticker, pos.current_price)
                        sector = pos.sector
                        if price > 0:
                            cost = sell_qty * price * self.config.transaction_cost
                            portfolio.sell(ticker, sell_qty, price)
                            portfolio.cash -= cost
                            trades.append(
                                Trade(
                                    date=date,
                                    symbol=ticker,
                                    sector=sector,
                                    action="SELL",
                                    quantity=sell_qty,
                                    price=price,
                                    value=sell_qty * price,
                                    cost=cost,
                                )
                            )

            # Execute buys
            for ticker, target_qty in target_positions.items():
                current_qty = 0
                if ticker in portfolio.positions:
                    current_qty = portfolio.positions[ticker].quantity

                delta = target_qty - current_qty
                if delta > 0 and ticker in prices:
                    price = prices[ticker]
                    stock = self.universe.get_stock(ticker)
                    sector = stock.sector if stock else "UNKNOWN"
                    cost = delta * price * self.config.transaction_cost

                    if portfolio.buy(ticker, delta, price, sector):
                        portfolio.cash -= cost
                        trades.append(
                            Trade(
                                date=date,
                                symbol=ticker,
                                sector=sector,
                                action="BUY",
                                quantity=delta,
                                price=price,
                                value=delta * price,
                                cost=cost,
                            )
                        )
                        # Track for stop loss (entry_price, peak_price, entry_day_idx)
                        if self.config.use_stop_loss:
                            if ticker not in stop_loss_entries:
                                stop_loss_entries[ticker] = (price, price, trading_day_index[date])

            # Sweep residual cash to cash_symbol (parity with live executor)
            cash_sym = self.app_config.regime.cash_symbol
            if portfolio.cash > 0 and cash_sym in prices and prices[cash_sym] > 0:
                sweep_qty = int(portfolio.cash / prices[cash_sym])
                if sweep_qty > 0:
                    sweep_price = prices[cash_sym]
                    sweep_cost = sweep_qty * sweep_price * self.config.transaction_cost
                    stock = self.universe.get_stock(cash_sym)
                    sweep_sector = stock.sector if stock else "Cash"
                    if portfolio.buy(cash_sym, sweep_qty, sweep_price, sweep_sector):
                        portfolio.cash -= sweep_cost
                        trades.append(
                            Trade(
                                date=date,
                                symbol=cash_sym,
                                sector=sweep_sector,
                                action="BUY",
                                quantity=sweep_qty,
                                price=sweep_price,
                                value=sweep_qty * sweep_price,
                                cost=sweep_cost,
                            )
                        )

            # Build rebalance trail record
            rebalance_number += 1
            regime_config = self.app_config.regime
            defensive_symbols = {regime_config.gold_symbol, regime_config.cash_symbol}

            # Get NIFTY price and SMA at this date
            nifty_price = 0.0
            nifty_sma = 0.0
            ts_date = pd.Timestamp(date)
            if "NIFTY 50" in self.data:
                nifty_df = self.data["NIFTY 50"]
                if ts_date in nifty_df.index:
                    nifty_price = float(nifty_df.loc[ts_date, "close"])
                if not self._nifty_200sma_cache.empty and ts_date in self._nifty_200sma_cache.index:
                    val = self._nifty_200sma_cache.loc[ts_date]
                    if not pd.isna(val):
                        nifty_sma = float(val)

            # Compute final sleeve weights from target_weights
            eq_w = sum(w for t, w in target_weights.items() if t not in defensive_symbols)
            gold_w = target_weights.get(regime_config.gold_symbol, 0.0)
            liquid_w = target_weights.get(regime_config.cash_symbol, 0.0)

            # Equity picks: (symbol, score, weight) sorted by weight desc
            ranked_map = {s.ticker: s.score for s in self._trail_ranked_stocks}
            equity_picks = [
                (t, ranked_map.get(t, 0.0), w)
                for t, w in sorted(target_weights.items(), key=lambda x: -x[1])
                if t not in defensive_symbols
            ]

            # Gate failures: stocks that were ranked but didn't pass entry filters
            gate_failures = [
                (s.ticker, s.score)
                for s in self._trail_ranked_stocks
                if not s.passes_entry_filters
            ][:8]  # Limit to top 8

            breadth_val = self._calculate_market_breadth(date)

            trail_record = RebalanceRecord(
                date=date,
                rebalance_number=rebalance_number,
                portfolio_value=portfolio.get_total_value(),
                regime=regime.regime.value if regime else "unknown",
                nifty_price=nifty_price,
                nifty_sma=nifty_sma,
                trend_above_sma=nifty_price > nifty_sma if nifty_sma > 0 else False,
                vix_value=regime.vix_level if regime else 0.0,
                breadth_value=breadth_val,
                equity_weight=eq_w,
                gold_weight=gold_w,
                liquid_weight=liquid_w,
                vol_scale=self._trail_vol_scale,
                breadth_scale=self._trail_breadth_scale,
                gold_exhaustion_scale=self._trail_gold_exhaustion_scale,
                equity_picks=equity_picks,
                gate_failures=gate_failures,
                trade_count=len(trades) - trades_before,
            )
            rebalance_trail.append(trail_record)

            # Reset trail state for next rebalance
            self._trail_gold_exhaustion_scale = 1.0

            # Record sector allocation
            snapshot = portfolio.get_snapshot()
            sector_alloc = {"date": date}
            for sector, weight in snapshot.get_sector_weights().items():
                sector_alloc[sector] = weight
            sector_history.append(sector_alloc)

            # Clear date-filtered cache after each rebalance (no cross-date benefit)
            self._date_filtered_data.clear()

        # Final equity value (avoid duplicate key if last trading day == end_date)
        final_date = self.config.end_date
        prices = self._get_prices_at_date(list(self.data.keys()), final_date)
        portfolio.update_prices(prices)
        if not equity_values or equity_values[-1][0] != final_date:
            equity_values.append((final_date, portfolio.get_total_value()))

        # Build equity curve
        equity_curve = pd.Series(
            {d: v for d, v in equity_values},
            name="equity",
        )

        # Calculate benchmark returns
        nifty_return, midcap_return = None, None
        if self.config.compare_benchmarks:
            nifty_return, midcap_return = self._calculate_benchmark_returns()

        # Calculate regime statistics
        regime_df = pd.DataFrame(regime_history) if regime_history else None
        regime_transitions = 0
        time_in_regime: Dict[str, float] = {}

        if regime_df is not None and len(regime_df) > 1:
            # Count regime transitions
            regime_df["regime_changed"] = regime_df["regime"] != regime_df["regime"].shift(1)
            regime_transitions = regime_df["regime_changed"].sum() - 1  # First is always "change"

            # Calculate time in each regime (as percentage)
            regime_counts = regime_df["regime"].value_counts()
            total_periods = len(regime_df)
            for regime_name, count in regime_counts.items():
                time_in_regime[regime_name] = count / total_periods

        # Calculate metrics
        return self._calculate_metrics(
            equity_curve,
            trades,
            sector_history,
            regime_df,
            regime_transitions,
            time_in_regime,
            nifty_return,
            midcap_return,
            self._strategy_name,
            rebalance_trail,
        )

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Trade],
        sector_history: List[dict],
        regime_history: Optional[pd.DataFrame] = None,
        regime_transitions: int = 0,
        time_in_regime: Optional[Dict[str, float]] = None,
        nifty_50_return: Optional[float] = None,
        nifty_midcap_100_return: Optional[float] = None,
        strategy_name: str = "dual_momentum",
        rebalance_trail: Optional[List[RebalanceRecord]] = None,
    ) -> BacktestResult:
        """Calculate performance metrics."""
        if len(equity_curve) < 2:
            return BacktestResult(
                total_return=0,
                cagr=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0,
                equity_curve=equity_curve,
                trades=trades,
                sector_allocations=pd.DataFrame(sector_history),
                initial_capital=self.config.initial_capital,
                final_value=self.config.initial_capital,
                peak_value=self.config.initial_capital,
                total_profit=0,
                regime_history=regime_history,
                regime_transitions=regime_transitions,
                time_in_regime=time_in_regime,
                nifty_50_return=nifty_50_return,
                nifty_midcap_100_return=nifty_midcap_100_return,
                strategy_name=strategy_name,
                rebalance_trail=rebalance_trail or [],
            )

        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # CAGR
        start = equity_curve.index[0]
        end = equity_curve.index[-1]
        years = (end - start).days / 365.25
        if years > 0:
            cagr = (1 + total_return) ** (1 / years) - 1
        else:
            cagr = 0

        # Sharpe ratio (with correct annualization for return frequency)
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            # Detect actual return frequency from median gap between observations
            time_diffs = pd.Series(equity_curve.index).diff().dropna().dt.days
            median_gap_days = time_diffs.median()
            # Approximate periods per year (252 trading days / gap in trading days)
            # For daily: ~252, for 21-day: ~12
            trading_day_gap = max(1, round(median_gap_days * 252 / 365.25))
            periods_per_year = 252 / trading_day_gap

            # Subtract risk-free rate (6% annual, consistent with indicators.py)
            risk_free_per_period = 0.06 / periods_per_year
            excess_returns = returns - risk_free_per_period

            sharpe = (excess_returns.mean() * periods_per_year) / (returns.std() * np.sqrt(periods_per_year))
        else:
            sharpe = 0

        # Max drawdown
        _, max_dd = calculate_drawdown(equity_curve)

        # Win rate (from trades) - FIFO matching with quantity tracking
        if trades:
            # Each queue entry is (price, remaining_quantity)
            buy_price_queues: Dict[str, deque] = {}
            wins = 0
            closed = 0

            for trade in trades:
                if trade.action == "BUY":
                    if trade.symbol not in buy_price_queues:
                        buy_price_queues[trade.symbol] = deque()
                    buy_price_queues[trade.symbol].append((trade.price, trade.quantity))
                else:  # SELL
                    if trade.symbol not in buy_price_queues:
                        continue
                    sell_remaining = trade.quantity
                    while sell_remaining > 0 and buy_price_queues[trade.symbol]:
                        buy_price, buy_qty = buy_price_queues[trade.symbol][0]
                        matched = min(sell_remaining, buy_qty)
                        if trade.price > buy_price:
                            wins += matched
                        closed += matched
                        sell_remaining -= matched
                        buy_qty -= matched
                        if buy_qty <= 0:
                            buy_price_queues[trade.symbol].popleft()
                        else:
                            buy_price_queues[trade.symbol][0] = (buy_price, buy_qty)

            win_rate = wins / closed if closed > 0 else 0
        else:
            win_rate = 0

        # Calculate capital amounts
        initial_capital = self.config.initial_capital
        final_value = equity_curve.iloc[-1]
        peak_value = equity_curve.max()
        total_profit = final_value - initial_capital

        return BacktestResult(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=len(trades),
            equity_curve=equity_curve,
            trades=trades,
            sector_allocations=pd.DataFrame(sector_history),
            initial_capital=initial_capital,
            final_value=final_value,
            peak_value=peak_value,
            total_profit=total_profit,
            regime_history=regime_history,
            regime_transitions=regime_transitions,
            time_in_regime=time_in_regime,
            nifty_50_return=nifty_50_return,
            nifty_midcap_100_return=nifty_midcap_100_return,
            strategy_name=strategy_name,
            rebalance_trail=rebalance_trail or [],
        )
