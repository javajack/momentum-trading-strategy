"""
Pure Momentum Engine for FORTRESS MOMENTUM.

Implements Nifty 500 Momentum 50 style stock selection:
- Direct stock ranking by Normalized Momentum Score (NMS)
- No sector-first approach
- Entry/exit filter chain
- Stop loss tracking integration

Now supports pluggable strategies for logic parity between live and backtest.

Performance optimizations:
- Parallel NMS calculation using ThreadPoolExecutor
- Session-level cache integration for zero-fetch ranking
- Batch processing for multiple stocks
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .config import (
    Config,
    PositionSizingConfig,
    PureMomentumConfig,
    RegimeConfig,
    RiskConfig,
)
from .indicators import (
    BullRecoverySignals,
    MarketRegime,
    NMSResult,
    RegimeResult,
    calculate_bull_recovery_signals,
    calculate_exit_triggers,
    calculate_normalized_momentum_score,
    detect_market_regime,
)
from .market_data import MarketDataProvider
from .universe import Stock, Universe
from .utils import renormalize_with_caps

if TYPE_CHECKING:
    from .strategy import BaseStrategy, StockScore


class ExitReason(Enum):
    """Reasons for position exit."""

    MOMENTUM_DECAY = "momentum_decay"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TREND_BREAK = "trend_break"
    TIME_DECAY = "time_decay"
    MANUAL = "manual"


@dataclass
class StockMomentum:
    """
    Stock with momentum score and entry filter results.

    Represents a single stock's momentum analysis for the pure momentum strategy.
    """

    ticker: str
    name: str
    sector: str
    sub_sector: str  # Granular business classification
    zerodha_symbol: str

    # NMS components
    nms: float  # Normalized Momentum Score
    return_6m: float  # 6-month simple return
    return_12m: float  # 12-month simple return
    volatility_6m: float  # 6-month annualized volatility
    adj_return_6m: float  # Volatility-adjusted 6M return
    adj_return_12m: float  # Volatility-adjusted 12M return

    # Entry filter results
    high_52w_proximity: float  # Price / 52-week high
    above_50ema: bool  # Price > 50-day EMA
    above_200sma: bool  # Price > 200-day SMA
    volume_surge: float  # 20d avg / 50d avg volume
    daily_turnover: float  # Average daily turnover (₹)

    # Ranking
    rank: int = 0  # Overall rank by NMS
    percentile: float = 0.0  # NMS percentile (0-100)

    # Entry filter status
    passes_filters: bool = False
    filter_failures: List[str] = field(default_factory=list)

    # Current price for position sizing
    current_price: float = 0.0

    def __post_init__(self):
        """Evaluate entry filters after initialization."""
        self._evaluate_filters()

    def _evaluate_filters(self):
        """Check which entry filters pass/fail."""
        failures = []

        if self.high_52w_proximity < 0.85:
            failures.append(f"52w high: {self.high_52w_proximity:.0%} < 85%")

        if not self.above_50ema:
            failures.append("Below 50-day EMA")

        if not self.above_200sma:
            failures.append("Below 200-day SMA")

        if self.volume_surge < 1.1:
            failures.append(f"Volume: {self.volume_surge:.2f}x < 1.1x")

        if self.daily_turnover < 20_000_000:
            failures.append(f"Turnover: ₹{self.daily_turnover / 1e7:.1f}Cr < ₹2Cr")

        self.filter_failures = failures
        self.passes_filters = len(failures) == 0


@dataclass
class PositionTracker:
    """Tracks a position for stop loss and exit trigger monitoring."""

    ticker: str
    sector: str
    entry_price: float
    entry_date: datetime
    entry_nms: float  # NMS at entry
    quantity: int
    current_price: float = 0.0
    peak_price: float = 0.0  # Highest price since entry
    current_nms: float = 0.0  # Current NMS
    nms_percentile: float = 0.0  # Current NMS percentile
    days_held: int = 0

    def update(
        self,
        current_price: float,
        current_nms: float,
        nms_percentile: float,
        as_of_date: datetime,
    ) -> None:
        """Update position with latest market data."""
        self.current_price = current_price
        self.peak_price = max(self.peak_price, current_price)
        self.current_nms = current_nms
        self.nms_percentile = nms_percentile
        self.days_held = (as_of_date - self.entry_date).days

    @property
    def current_gain(self) -> float:
        """Current gain/loss from entry."""
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def gain_from_peak(self) -> float:
        """Current price vs peak price."""
        return (
            (self.current_price - self.peak_price) / self.peak_price if self.peak_price > 0 else 0.0
        )

    @property
    def current_value(self) -> float:
        """Current position value."""
        return self.current_price * self.quantity


class MomentumEngine:
    """
    Pure momentum stock selection engine.

    Ranks ALL stocks in the universe by Normalized Momentum Score (NMS)
    without sector constraints. Implements entry filters and exit triggers
    following Nifty 500 Momentum 50 methodology.

    Now supports pluggable strategies via the strategy parameter.
    When a strategy is provided, it delegates to the strategy for all
    ranking, selection, and exit trigger logic to ensure parity with backtest.
    """

    # Concurrency settings for parallel NMS calculation
    NMS_WORKERS = 8  # Parallel threads for NMS computation

    def __init__(
        self,
        universe: Universe,
        market_data: MarketDataProvider,
        momentum_config: Optional[PureMomentumConfig] = None,
        sizing_config: Optional[PositionSizingConfig] = None,
        risk_config: Optional[RiskConfig] = None,
        regime_config: Optional[RegimeConfig] = None,
        strategy: Optional["BaseStrategy"] = None,
        app_config: Optional[Config] = None,
        cached_data: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        """
        Initialize momentum engine.

        Args:
            universe: Stock universe
            market_data: Market data provider
            momentum_config: Pure momentum settings
            sizing_config: Position sizing settings
            risk_config: Risk management settings
            regime_config: Market regime detection settings
            strategy: Optional strategy for pluggable momentum logic
            app_config: Full application config (used with strategy)
            cached_data: Optional pre-loaded cache data (for performance)
        """
        self.universe = universe
        self.market_data = market_data
        self.momentum_config = momentum_config or PureMomentumConfig()
        self.sizing_config = sizing_config or PositionSizingConfig()
        self.risk_config = risk_config or RiskConfig()
        self.regime_config = regime_config or RegimeConfig()

        # Strategy support for logic parity with backtest
        self._strategy = strategy
        self._app_config = app_config

        # Cached data for fast ranking (avoids API calls)
        self._cached_data: Optional[Dict[str, pd.DataFrame]] = cached_data

        # Position tracking for stop loss management
        self._positions: Dict[str, PositionTracker] = {}

        # Regime tracking for hysteresis
        self._previous_regime_result: Optional[RegimeResult] = None

        # Drawdown tracking for live mode parity with backtest
        self._current_drawdown: float = 0.0
        self._peak_portfolio_value: float = 0.0

        # Recovery override hysteresis (FIX 6)
        self._recovery_override_active: bool = False
        self._recovery_override_confirm_days: int = 0

        # Smoothed breadth for scaling (FIX 8)
        self._breadth_ema: Optional[float] = None

        # Excluded symbols (ETFs, hedges) — config-based + hardcoded external ETFs
        self._excluded_symbols: Set[str] = {
            "LIQUIDCASE",
            "LIQUIDBEES",
            "NIFTYBEES",
            "JUNIORBEES",
            "MID150BEES",
            "HDFCSML250",
            "GOLDBEES",
            "HANGSENGBEES",
            self.regime_config.gold_symbol,  # Parity with backtest._excluded_set
            self.regime_config.cash_symbol,
        }

        # Portfolio daily returns for vol targeting (E2)
        self._portfolio_daily_returns: List[float] = []
        self._prev_portfolio_value: float = 0.0

        # (Breadth history removed — recovery override now uses point-in-time computation)

    @property
    def strategy(self) -> Optional["BaseStrategy"]:
        """Get the current strategy."""
        return self._strategy

    def set_strategy(self, strategy: "BaseStrategy") -> None:
        """Set or change the strategy."""
        self._strategy = strategy

    def set_cached_data(self, cached_data: Dict[str, pd.DataFrame]) -> None:
        """Set cached data for fast ranking (avoids API calls)."""
        self._cached_data = cached_data

    def _calculate_single_nms(
        self,
        stock: Stock,
        as_of_date: datetime,
    ) -> Optional[StockMomentum]:
        """
        Calculate NMS for a single stock using cached data.

        Thread-safe helper for parallel NMS calculation.
        """
        # Skip excluded symbols
        if stock.ticker in self._excluded_symbols:
            return None

        try:
            # Try cached data first (fast path)
            df = None
            if self._cached_data and stock.zerodha_symbol in self._cached_data:
                df = self._cached_data[stock.zerodha_symbol]
                # Filter to as_of_date
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                df = df[df.index <= pd.Timestamp(as_of_date)]

            # Fallback to API if no cache or insufficient data
            if df is None or len(df) < self.momentum_config.lookback_6m:
                lookback_days = (
                    self.momentum_config.lookback_12m + self.momentum_config.skip_recent_days + 30
                )
                from_date = as_of_date - timedelta(days=int(lookback_days * 1.5))
                df = self.market_data.get_historical(
                    symbol=stock.zerodha_symbol,
                    from_date=from_date,
                    to_date=as_of_date,
                    interval="day",
                    check_quality=False,
                )

            if df is None or df.empty or len(df) < self.momentum_config.lookback_6m:
                return None

            prices = df["close"]
            volumes = df["volume"]

            # Calculate NMS
            nms_result = calculate_normalized_momentum_score(
                prices=prices,
                volumes=volumes,
                lookback_6m=self.momentum_config.lookback_6m,
                lookback_12m=self.momentum_config.lookback_12m,
                lookback_volatility=self.momentum_config.lookback_volatility,
                skip_recent_days=self.momentum_config.skip_recent_days,
                weight_6m=self.momentum_config.weight_6m,
                weight_12m=self.momentum_config.weight_12m,
            )

            if nms_result is None:
                return None

            # Create StockMomentum
            return StockMomentum(
                ticker=stock.ticker,
                name=stock.name,
                sector=stock.sector,
                sub_sector=stock.sub_sector,
                zerodha_symbol=stock.zerodha_symbol,
                nms=nms_result.nms,
                return_6m=nms_result.return_6m,
                return_12m=nms_result.return_12m,
                volatility_6m=nms_result.volatility_6m,
                adj_return_6m=nms_result.adj_return_6m,
                adj_return_12m=nms_result.adj_return_12m,
                high_52w_proximity=nms_result.high_52w_proximity,
                above_50ema=nms_result.above_50ema,
                above_200sma=nms_result.above_200sma,
                volume_surge=nms_result.volume_surge,
                daily_turnover=nms_result.daily_turnover,
                current_price=prices.iloc[-1],
            )

        except Exception:
            return None

    def _convert_stock_score_to_momentum(self, score: "StockScore") -> StockMomentum:
        """Convert a StockScore (from strategy) to StockMomentum for backward compatibility."""
        return StockMomentum(
            ticker=score.ticker,
            name=score.name,
            sector=score.sector,
            sub_sector=score.sub_sector,
            zerodha_symbol=score.zerodha_symbol,
            nms=score.score,  # Map score to nms
            return_6m=score.return_6m,
            return_12m=score.return_12m,
            volatility_6m=score.volatility,
            adj_return_6m=score.extra_metrics.get("adj_return_6m", 0.0),
            adj_return_12m=score.extra_metrics.get("adj_return_12m", 0.0),
            high_52w_proximity=score.high_52w_proximity,
            above_50ema=score.above_50ema,
            above_200sma=score.above_200sma,
            volume_surge=score.volume_surge,
            daily_turnover=score.daily_turnover,
            rank=score.rank,
            percentile=score.percentile,
            current_price=score.current_price,
        )

    def rank_all_stocks(
        self,
        as_of_date: datetime,
        filter_entry: bool = True,
    ) -> List[StockMomentum]:
        """
        Rank all stocks in universe by Normalized Momentum Score.

        This is the core function of the pure momentum strategy - no sector
        grouping, just direct stock ranking by NMS.

        If a strategy is set, delegates to strategy.rank_stocks() to ensure
        logic parity between live and backtest.

        Args:
            as_of_date: Date for calculation
            filter_entry: If True, only return stocks passing entry filters

        Returns:
            List of StockMomentum sorted by NMS descending
        """
        # If strategy is set, delegate to it for logic parity
        if self._strategy is not None:
            stock_scores = self._strategy.rank_stocks(
                as_of_date=as_of_date,
                universe=self.universe,
                market_data=self.market_data,
                filter_entry=filter_entry,
            )
            # Convert StockScore to StockMomentum for backward compatibility
            return [self._convert_stock_score_to_momentum(s) for s in stock_scores]

        # Original implementation (when no strategy is set)
        # Uses parallel NMS calculation for 3-5x speedup
        all_stocks = self.universe.get_all_stocks()
        results: List[StockMomentum] = []

        # Use parallel NMS calculation with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.NMS_WORKERS) as executor:
            futures = {
                executor.submit(self._calculate_single_nms, stock, as_of_date): stock
                for stock in all_stocks
            }

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        # Sort by NMS descending
        results.sort(key=lambda x: x.nms, reverse=True)

        # Apply entry filter if requested (BEFORE calculating percentiles)
        if filter_entry:
            results = [s for s in results if s.passes_filters]

        # Assign ranks and percentiles (AFTER filtering for accurate percentiles)
        total = len(results)
        if total == 0:
            return results  # Empty list, nothing to rank
        for i, stock in enumerate(results):
            stock.rank = i + 1
            stock.percentile = 100 * (total - i) / total

        return results

    def select_top_stocks(
        self,
        as_of_date: datetime,
        n: Optional[int] = None,
        min_percentile: Optional[float] = None,
        max_per_sector: int = 3,
    ) -> List[StockMomentum]:
        """
        Select top N stocks by NMS that pass all entry filters.

        If a strategy is set, delegates to strategy for selection logic
        to ensure parity between live and backtest.

        Args:
            as_of_date: Date for calculation
            n: Number of stocks to select (default: target_positions from config)
            min_percentile: Minimum NMS percentile (default: from config)
            max_per_sector: Maximum stocks per sector (default: 3)

        Returns:
            Top stocks passing all filters with sector diversification
        """
        n = n or self.sizing_config.target_positions
        min_percentile = min_percentile or self.momentum_config.min_score_percentile

        # Get all stocks with entry filter applied (uses strategy if set)
        all_ranked = self.rank_all_stocks(as_of_date, filter_entry=True)

        # Select with sector diversification (same logic as backtest)
        selected = []
        sector_counts: Dict[str, int] = {}

        for stock in all_ranked:
            # Check percentile threshold
            if stock.percentile < min_percentile:
                if len(selected) >= self.sizing_config.min_positions:
                    break

            # Check sector limit
            if max_per_sector > 0:
                current_count = sector_counts.get(stock.sector, 0)
                if current_count >= max_per_sector:
                    continue  # Skip - sector is full

            selected.append(stock)
            sector_counts[stock.sector] = sector_counts.get(stock.sector, 0) + 1

            if len(selected) >= n:
                break

        # If not enough stocks, relax percentile requirement
        if len(selected) < self.sizing_config.min_positions:
            for stock in all_ranked:
                if stock in selected:
                    continue
                if max_per_sector > 0:
                    current_count = sector_counts.get(stock.sector, 0)
                    if current_count >= max_per_sector:
                        continue
                selected.append(stock)
                sector_counts[stock.sector] = sector_counts.get(stock.sector, 0) + 1
                if len(selected) >= self.sizing_config.min_positions:
                    break

        return selected[:n]

    def calculate_target_weights(
        self,
        selected_stocks: List[StockMomentum],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Calculate target weight for each selected stock.

        Supports three methods:
        - equal: Equal weight across all positions
        - momentum_weighted: Weight proportional to NMS
        - inverse_volatility: Weight inversely proportional to volatility

        Args:
            selected_stocks: Stocks to allocate to
            portfolio_value: Total portfolio value

        Returns:
            Dict mapping ticker to target weight (0-1)
        """
        if not selected_stocks:
            return {}

        method = self.sizing_config.method
        n = len(selected_stocks)

        if method == "equal":
            base_weight = 1.0 / n
            return {s.ticker: base_weight for s in selected_stocks}

        elif method == "momentum_weighted":
            # Normalize NMS scores to positive values
            min_nms = min(s.nms for s in selected_stocks)
            adjusted_scores = [s.nms - min_nms + 0.01 for s in selected_stocks]
            total_score = sum(adjusted_scores)

            weights = {}
            for i, stock in enumerate(selected_stocks):
                raw_weight = adjusted_scores[i] / total_score
                # Apply min/max constraints
                weight = max(
                    self.sizing_config.min_single_position,
                    min(self.sizing_config.max_single_position, raw_weight),
                )
                weights[stock.ticker] = weight

            # Renormalize with iterative capping to maintain position limits
            return self._renormalize_with_caps(weights)

        elif method == "inverse_volatility":
            # Weight inversely proportional to volatility
            inv_vols = [1.0 / max(s.volatility_6m, 0.10) for s in selected_stocks]
            total_inv_vol = sum(inv_vols)

            weights = {}
            for i, stock in enumerate(selected_stocks):
                raw_weight = inv_vols[i] / total_inv_vol
                weight = max(
                    self.sizing_config.min_single_position,
                    min(self.sizing_config.max_single_position, raw_weight),
                )
                weights[stock.ticker] = weight

            # Renormalize with iterative capping to maintain position limits
            return self._renormalize_with_caps(weights)

        else:
            # Fallback to equal weight
            base_weight = 1.0 / n
            return {s.ticker: base_weight for s in selected_stocks}

    def _renormalize_with_caps(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Renormalize weights to sum to 1.0 while respecting position limits.

        Delegates to shared utility function for logic parity with backtest.
        """
        return renormalize_with_caps(
            weights,
            max_weight=self.sizing_config.max_single_position,
            min_weight=self.sizing_config.min_single_position,
        )

    def apply_sector_limits(
        self,
        weights: Dict[str, float],
        selected_stocks: List[StockMomentum],
        max_sector_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Apply sector concentration limits to weights.

        Uses iterative capping algorithm (same as backtest) to ensure
        no sector exceeds max_sector_exposure while properly redistributing
        excess weight to other sectors.

        Args:
            weights: Initial weight allocation
            selected_stocks: Selected stocks (for sector mapping)

        Returns:
            Adjusted weights respecting sector limits
        """
        # Build ticker -> sector mapping
        ticker_to_sector = {s.ticker: s.sector for s in selected_stocks}
        max_sector = (
            max_sector_override
            if max_sector_override is not None
            else self.sizing_config.max_sector_exposure
        )
        adjusted_weights = dict(weights)
        capped_sectors: set = set()

        # Iteratively cap sectors and redistribute to uncapped sectors
        max_iterations = 10
        for _ in range(max_iterations):
            # Calculate current sector weights
            sector_weights: Dict[str, float] = {}
            for ticker, weight in adjusted_weights.items():
                sector = ticker_to_sector.get(ticker, "UNKNOWN")
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
                    for ticker in adjusted_weights:
                        if ticker_to_sector.get(ticker) == sector:
                            adjusted_weights[ticker] *= scale
                    newly_capped = True

            if not newly_capped:
                break  # No new sectors to cap

            # Normalize only uncapped sectors to make weights sum to 1.0
            capped_total = sum(
                w for t, w in adjusted_weights.items() if ticker_to_sector.get(t) in capped_sectors
            )
            uncapped_total = sum(
                w
                for t, w in adjusted_weights.items()
                if ticker_to_sector.get(t) not in capped_sectors
            )

            # Uncapped sectors should sum to (1.0 - capped_total)
            target_uncapped = 1.0 - capped_total
            if uncapped_total > 0 and target_uncapped > 0:
                scale = target_uncapped / uncapped_total
                for ticker in adjusted_weights:
                    if ticker_to_sector.get(ticker) not in capped_sectors:
                        adjusted_weights[ticker] *= scale

        return adjusted_weights

    def check_exit_triggers(
        self,
        positions: Dict[str, PositionTracker],
        as_of_date: datetime,
    ) -> List[Tuple[str, ExitReason, str]]:
        """
        Check exit triggers for all tracked positions.

        If a strategy is set, uses strategy's check_exit_triggers() and
        get_stop_loss_config() for logic parity with backtest.

        Exit Triggers:
        1. Momentum decay: NMS falls below 50th percentile
        2. Stop loss: configurable from entry price
        3. Trailing stop: configurable from peak (after activation threshold)
        4. Trend break: Price < 50-day EMA
        5. Time decay: >60 days held without target gain

        Args:
            positions: Current position trackers
            as_of_date: Date for calculation

        Returns:
            List of (ticker, exit_reason, description) for positions to exit
        """
        exits: List[Tuple[str, ExitReason, str]] = []

        # First, update all positions with current data
        for ticker, pos in positions.items():
            try:
                stock = self.universe.get_stock(ticker)
                if not stock:
                    continue

                # Fetch current data
                from_date = as_of_date - timedelta(days=300)
                df = self.market_data.get_historical(
                    symbol=stock.zerodha_symbol,
                    from_date=from_date,
                    to_date=as_of_date,
                    interval="day",
                    check_quality=False,
                )

                if df.empty:
                    continue

                current_price = df["close"].iloc[-1]

                # Calculate current NMS
                nms_result = calculate_normalized_momentum_score(
                    prices=df["close"],
                    volumes=df["volume"],
                    lookback_6m=self.momentum_config.lookback_6m,
                    lookback_12m=self.momentum_config.lookback_12m,
                    lookback_volatility=self.momentum_config.lookback_volatility,
                    skip_recent_days=self.momentum_config.skip_recent_days,
                    weight_6m=self.momentum_config.weight_6m,
                    weight_12m=self.momentum_config.weight_12m,
                )

                if nms_result:
                    # Update position
                    pos.update(
                        current_price=current_price,
                        current_nms=nms_result.nms,
                        nms_percentile=pos.nms_percentile,
                        as_of_date=as_of_date,
                    )

                    # If strategy is set, use its exit trigger logic
                    if self._strategy is not None:
                        # Convert NMS result to StockScore for strategy
                        from .strategy import StockScore

                        stock_score = StockScore(
                            ticker=ticker,
                            sector=pos.sector,
                            sub_sector="",
                            zerodha_symbol=stock.zerodha_symbol,
                            name=stock.name,
                            score=nms_result.nms,
                            rank=0,
                            percentile=pos.nms_percentile,
                            passes_entry_filters=True,
                            return_6m=nms_result.return_6m,
                            return_12m=nms_result.return_12m,
                            volatility=nms_result.volatility_6m,
                            high_52w_proximity=nms_result.high_52w_proximity,
                            above_50ema=nms_result.above_50ema,
                            above_200sma=nms_result.above_200sma,
                            volume_surge=nms_result.volume_surge,
                            daily_turnover=nms_result.daily_turnover,
                            current_price=current_price,
                        )

                        exit_signal = self._strategy.check_exit_triggers(
                            ticker=ticker,
                            entry_price=pos.entry_price,
                            current_price=current_price,
                            peak_price=pos.peak_price,
                            days_held=pos.days_held,
                            stock_score=stock_score,
                            nms_percentile=pos.nms_percentile,
                        )

                        if exit_signal.should_exit:
                            # Map exit_type string to ExitReason enum
                            exit_type_map = {
                                "stop_loss": ExitReason.STOP_LOSS,
                                "trailing_stop": ExitReason.TRAILING_STOP,
                                "momentum_decay": ExitReason.MOMENTUM_DECAY,
                                "trend_break": ExitReason.TREND_BREAK,
                                "time_decay": ExitReason.TIME_DECAY,
                                "rs_floor": ExitReason.MOMENTUM_DECAY,
                                "exhaustion": ExitReason.MOMENTUM_DECAY,
                            }
                            exit_type = exit_type_map.get(exit_signal.exit_type, ExitReason.MANUAL)
                            exits.append((ticker, exit_type, exit_signal.reason))
                    else:
                        # Original implementation (when no strategy is set)
                        should_exit, reason = calculate_exit_triggers(
                            nms_result=nms_result,
                            entry_price=pos.entry_price,
                            current_price=current_price,
                            peak_price=pos.peak_price,
                            days_held=pos.days_held,
                            nms_percentile=pos.nms_percentile,
                            initial_stop_loss=self.risk_config.initial_stop_loss,
                            trailing_stop=self.risk_config.trailing_stop,
                            trailing_activation=self.risk_config.trailing_activation,
                            max_days_without_gain=self.momentum_config.max_days_without_gain,
                            min_gain_threshold=self.momentum_config.min_gain_threshold,
                            min_nms_percentile=self.momentum_config.min_hold_percentile,
                        )

                        if should_exit:
                            # Determine exit reason type
                            if "Stop loss" in reason:
                                exit_type = ExitReason.STOP_LOSS
                            elif "Trailing stop" in reason:
                                exit_type = ExitReason.TRAILING_STOP
                            elif "Momentum decay" in reason:
                                exit_type = ExitReason.MOMENTUM_DECAY
                            elif "Trend break" in reason:
                                exit_type = ExitReason.TREND_BREAK
                            elif "Time decay" in reason:
                                exit_type = ExitReason.TIME_DECAY
                            else:
                                exit_type = ExitReason.MANUAL

                            exits.append((ticker, exit_type, reason))

            except Exception:
                continue

        return exits

    def get_stop_loss_config(self, ticker: str, current_gain: float):
        """
        Get stop loss configuration for a position.

        If a strategy is set, uses strategy's get_stop_loss_config()
        for tiered/adaptive stops.

        Args:
            ticker: Stock ticker
            current_gain: Current gain from entry

        Returns:
            StopLossConfig or dict with stop loss parameters
        """
        if self._strategy is not None:
            return self._strategy.get_stop_loss_config(ticker, current_gain)

        # Original implementation
        return {
            "initial_stop": self.risk_config.initial_stop_loss,
            "trailing_stop": self.risk_config.trailing_stop,
            "trailing_activation": self.risk_config.trailing_activation,
        }

    def update_position_percentiles(
        self,
        positions: Dict[str, PositionTracker],
        all_stocks: List[StockMomentum],
    ) -> None:
        """
        Update NMS percentiles for tracked positions.

        Args:
            positions: Current position trackers
            all_stocks: Full ranked stock list (for percentile calculation)
        """
        # Build NMS lookup
        nms_lookup = {s.ticker: s for s in all_stocks}

        for ticker, pos in positions.items():
            if ticker in nms_lookup:
                stock = nms_lookup[ticker]
                pos.current_nms = stock.nms
                pos.nms_percentile = stock.percentile

    def register_position(
        self,
        ticker: str,
        sector: str,
        entry_price: float,
        quantity: int,
        entry_date: datetime,
        entry_nms: float,
    ) -> None:
        """
        Register a new position for stop loss tracking.

        Args:
            ticker: Stock ticker
            sector: Stock sector
            entry_price: Entry price
            quantity: Number of shares
            entry_date: Date of entry
            entry_nms: NMS at entry
        """
        self._positions[ticker] = PositionTracker(
            ticker=ticker,
            sector=sector,
            entry_price=entry_price,
            entry_date=entry_date,
            entry_nms=entry_nms,
            quantity=quantity,
            peak_price=entry_price,
        )

    def remove_position(self, ticker: str) -> None:
        """Remove a position from tracking."""
        if ticker in self._positions:
            del self._positions[ticker]

    def get_position(self, ticker: str) -> Optional[PositionTracker]:
        """Get position tracker for a ticker."""
        return self._positions.get(ticker)

    @property
    def tracked_positions(self) -> Dict[str, PositionTracker]:
        """Get all tracked positions."""
        return self._positions.copy()

    def get_sector_exposure(
        self,
        selected_stocks: List[StockMomentum],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate sector exposure from weights.

        Args:
            selected_stocks: Selected stocks
            weights: Weight allocation

        Returns:
            Dict mapping sector to total weight
        """
        sector_weights: Dict[str, float] = {}
        ticker_to_sector = {s.ticker: s.sector for s in selected_stocks}

        for ticker, weight in weights.items():
            sector = ticker_to_sector.get(ticker, "UNKNOWN")
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        return sector_weights

    def detect_current_regime(
        self,
        as_of_date: datetime,
    ) -> Optional[RegimeResult]:
        """
        Detect current market regime using live data.

        Uses enhanced regime detection with:
        - Multi-timeframe position (21/63/126 day weighted composite)
        - Hysteresis for transition confirmation
        - Graduated allocation based on stress score

        Args:
            as_of_date: Date for calculation

        Returns:
            RegimeResult or None if regime detection is disabled or data unavailable
        """
        if not self.regime_config.enabled:
            return None

        # Fetch Nifty 50 historical data (need ~1 year for full analysis)
        lookback_days = 365
        from_date = as_of_date - timedelta(days=lookback_days)

        try:
            nifty_df = self.market_data.get_historical(
                symbol="NIFTY 50",
                from_date=from_date,
                to_date=as_of_date,
                interval="day",
                check_quality=False,
            )

            if nifty_df.empty or len(nifty_df) < 63:  # Need at least 3 months
                return None

            nifty_prices = nifty_df["close"]

            # Fetch VIX (use default if not available)
            vix_value = 15.0  # Default calm market
            vix_history = None
            try:
                vix_df = self.market_data.get_historical(
                    symbol="INDIA VIX",
                    from_date=as_of_date - timedelta(days=30),
                    to_date=as_of_date,
                    interval="day",
                    check_quality=False,
                )
                if not vix_df.empty:
                    vix_value = vix_df["close"].iloc[-1]
                    if len(vix_df) >= 10:
                        vix_history = vix_df["close"]
            except Exception:
                pass  # Use default VIX

            # Use enhanced regime detection with hysteresis
            result = detect_market_regime(
                nifty_prices,
                vix_value,
                self.regime_config,
                previous_result=self._previous_regime_result,
                vix_history=vix_history,
            )

            # Store for next call's hysteresis
            if result is not None:
                self._previous_regime_result = result

            return result

        except Exception:
            return None

    def update_portfolio_value(self, portfolio_value: float) -> None:
        """Track portfolio value for vol targeting (E2). Call daily."""
        if self._prev_portfolio_value > 0:
            daily_ret = (portfolio_value - self._prev_portfolio_value) / self._prev_portfolio_value
            self._portfolio_daily_returns.append(daily_ret)
            # Keep last 63 days
            if len(self._portfolio_daily_returns) > 63:
                self._portfolio_daily_returns = self._portfolio_daily_returns[-63:]
        self._prev_portfolio_value = portfolio_value

    def _calculate_vol_scale(self) -> float:
        """Calculate portfolio-level vol scaling (E2) for live mode."""
        if not self.regime_config.use_vol_targeting:
            return 1.0
        lookback = self.regime_config.vol_lookback_days
        if len(self._portfolio_daily_returns) < lookback:
            return 1.0
        recent_returns = self._portfolio_daily_returns[-lookback:]
        realized_vol = np.std(recent_returns) * (252**0.5)
        if realized_vol < 0.01:
            return 1.0
        raw_scale = self.regime_config.target_portfolio_vol / realized_vol
        return min(1.0, max(self.regime_config.vol_scale_floor, raw_scale))

    def _calculate_breadth_scale(self, as_of_date: datetime) -> float:
        """Calculate breadth-based exposure scaling (E3) for live mode."""
        if not self.regime_config.use_breadth_scaling:
            return 1.0
        try:
            raw_breadth = self._compute_live_breadth(as_of_date)
        except Exception:
            return 1.0

        # Smooth breadth with 5-day EMA to reduce oscillation in sideways markets (FIX 8)
        alpha = 2.0 / (5 + 1)
        if self._breadth_ema is None:
            self._breadth_ema = raw_breadth
        else:
            self._breadth_ema = alpha * raw_breadth + (1 - alpha) * self._breadth_ema
        breadth = self._breadth_ema

        cfg = self.regime_config
        if breadth >= cfg.breadth_full:
            return 1.0
        elif breadth <= cfg.breadth_low:
            return cfg.breadth_min_scale
        else:
            t = (breadth - cfg.breadth_low) / (cfg.breadth_full - cfg.breadth_low)
            return cfg.breadth_min_scale + t * (1.0 - cfg.breadth_min_scale)

    def _compute_live_breadth(self, as_of_date: datetime) -> float:
        """Compute market breadth from cached or live data."""
        above_50ma = 0
        valid_stocks = 0

        data_source = self._cached_data or {}
        for symbol in self.universe.get_all_symbols():
            if symbol in self._excluded_symbols:
                continue
            df = data_source.get(symbol)
            if df is None or df.empty or len(df) < 50:
                continue
            try:
                ts_date = pd.Timestamp(as_of_date)
                available = df[df.index <= ts_date]
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

    def _get_effective_sector_cap(self, regime: Optional[RegimeResult]) -> float:
        """Get effective sector cap based on regime (E4)."""
        if not self.sizing_config.use_dynamic_sector_caps or regime is None:
            return self.sizing_config.max_sector_exposure
        if regime.regime == MarketRegime.DEFENSIVE:
            return self.sizing_config.defensive_max_sector
        elif regime.regime == MarketRegime.CAUTION:
            return self.sizing_config.caution_max_sector
        else:
            return self.sizing_config.max_sector_exposure

    def _is_market_uptrend(self, as_of_date: datetime) -> bool:
        """Check if NIFTY 50 is above its 200-day SMA (structural uptrend)."""
        nifty_symbol = "NIFTY 50"
        try:
            if self._cached_data and nifty_symbol in self._cached_data:
                df = self._cached_data[nifty_symbol]
                df = df.loc[:as_of_date]
            else:
                df = self.market_data.get_historical(
                    symbol=nifty_symbol,
                    from_date=as_of_date - timedelta(days=400),
                    to_date=as_of_date,
                    interval="day",
                    check_quality=False,
                )
            if df.empty or len(df) < 200:
                return False
            sma_200 = df["close"].iloc[-200:].mean()
            current_price = df["close"].iloc[-1]
            return bool(current_price > sma_200)
        except Exception:
            return False

    def _should_skip_gold(self, as_of_date: datetime) -> Tuple[bool, str]:
        """
        Check if gold ETF should be skipped from defensive allocation.

        Two modes controlled by config.regime.gold_skip_logic:
        - "downtrend": Skip only if price < 50-SMA (volatile gold in crisis = good)
        - "volatile": Legacy logic - skip if recent vol > 1.5x average

        Returns:
            Tuple of (should_skip, reason_string)
        """
        gold_symbol = self.regime_config.gold_symbol
        try:
            gold_df = self.market_data.get_historical(
                symbol=gold_symbol,
                from_date=as_of_date - timedelta(days=90),
                to_date=as_of_date,
                interval="day",
                check_quality=False,
            )
            if gold_df.empty or len(gold_df) < 50:
                return (False, "")

            skip_logic = self.regime_config.gold_skip_logic

            if skip_logic == "downtrend":
                current_price = gold_df["close"].iloc[-1]
                sma_50 = gold_df["close"].iloc[-50:].mean()
                if current_price < sma_50:
                    return (True, f"Gold downtrend: {current_price:.1f} < 50-SMA {sma_50:.1f}")
                return (False, "")
            else:
                returns = gold_df["close"].pct_change().dropna()
                recent_vol = returns.iloc[-10:].std() * (252**0.5)
                avg_vol = returns.std() * (252**0.5)
                if recent_vol > avg_vol * 1.5 and recent_vol > 0.15:
                    return (
                        True,
                        f"Gold volatile: recent_vol {recent_vol:.3f} > 1.5x avg {avg_vol:.3f}",
                    )
                return (False, "")
        except Exception:
            return (False, "")

    def _calculate_gold_exhaustion_scale(self, as_of_date: datetime) -> float:
        """
        Calculate gold exhaustion scaling factor (GE1).

        Linearly scales gold allocation from 1.0 to 0.0 as gold moves
        from threshold_low to threshold_high above its SMA.

        Returns:
            Float between 0.0 and 1.0
        """
        if not self.regime_config.use_gold_exhaustion_scaling:
            return 1.0

        gold_symbol = self.regime_config.gold_symbol
        sma_period = self.regime_config.gold_exhaustion_sma_period
        try:
            if self._cached_data and gold_symbol in self._cached_data:
                df = self._cached_data[gold_symbol]
                df = df.loc[:as_of_date]
            else:
                df = self.market_data.get_historical(
                    symbol=gold_symbol,
                    from_date=as_of_date - timedelta(days=sma_period * 2),
                    to_date=as_of_date,
                    interval="day",
                    check_quality=False,
                )
            if df.empty or len(df) < sma_period:
                return 1.0

            sma_val = df["close"].iloc[-sma_period:].mean()
            if sma_val <= 0:
                return 1.0

            current_price = df["close"].iloc[-1]
            deviation = (current_price - sma_val) / sma_val

            low = self.regime_config.gold_exhaustion_threshold_low
            high = self.regime_config.gold_exhaustion_threshold_high
            if deviation <= low:
                return 1.0
            elif deviation >= high:
                return 0.0
            else:
                return 1.0 - (deviation - low) / (high - low)
        except Exception:
            return 1.0

    def _redirect_freed_weight(
        self, weights: Dict[str, float], freed: float, as_of_date: datetime
    ) -> None:
        """Redirect freed gold weight to equities pro-rata (uptrend) or cash (downtrend)."""
        cash_sym = self.regime_config.cash_symbol
        if self.regime_config.redirect_freed_to_equity_in_uptrend and self._is_market_uptrend(
            as_of_date
        ):
            defensive = {self.regime_config.gold_symbol, cash_sym}
            equity_weights = {t: w for t, w in weights.items() if t not in defensive and w > 0}
            total_eq = sum(equity_weights.values())
            if total_eq > 0:
                for t, w in equity_weights.items():
                    weights[t] += freed * (w / total_eq)
                return
        weights[cash_sym] = weights.get(cash_sym, 0.0) + freed

    def reset_regime_state(self) -> None:
        """
        Reset regime hysteresis state.

        Call this when starting a new session or after extended market closure
        to reset the regime tracking state.
        """
        self._previous_regime_result = None

    def update_drawdown(self, portfolio_value: float) -> float:
        """
        Update drawdown tracking based on current portfolio value.

        Args:
            portfolio_value: Current total portfolio value

        Returns:
            Current drawdown as a negative percentage (e.g., -0.10 = -10%)
        """
        # Update peak if new high
        if portfolio_value > self._peak_portfolio_value:
            self._peak_portfolio_value = portfolio_value

        # Calculate current drawdown
        if self._peak_portfolio_value > 0:
            self._current_drawdown = (
                portfolio_value - self._peak_portfolio_value
            ) / self._peak_portfolio_value
        else:
            self._current_drawdown = 0.0

        return self._current_drawdown

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

        # Fetch Nifty prices
        lookback_days = 100
        from_date = as_of_date - timedelta(days=lookback_days)

        try:
            nifty_df = self.market_data.get_historical(
                symbol=nifty_symbol,
                from_date=from_date,
                to_date=as_of_date,
                interval="day",
                check_quality=False,
            )

            if nifty_df.empty or len(nifty_df) < 63:  # Need at least 3 months
                return None

            nifty_prices = nifty_df["close"]

            # Get VIX history
            vix_history = None
            vix_value = regime.vix_level

            try:
                vix_df = self.market_data.get_historical(
                    symbol=vix_symbol,
                    from_date=as_of_date - timedelta(days=40),
                    to_date=as_of_date,
                    interval="day",
                    check_quality=False,
                )
                if not vix_df.empty and len(vix_df) >= 20:
                    vix_history = vix_df["close"].iloc[-30:]  # Last 30 days
            except Exception:
                pass  # Use default vix_history = None

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
            position_momentum = (
                regime.position_momentum if hasattr(regime, "position_momentum") else 0.0
            )

            # Calculate bull recovery signals
            return calculate_bull_recovery_signals(
                nifty_prices=nifty_prices,
                vix_value=vix_value,
                vix_history=vix_history,
                position_momentum=position_momentum,
                return_1m=return_1m,
                return_3m=return_3m,
            )

        except Exception:
            return None

    def _get_crash_avoidance_data(
        self, as_of_date: datetime
    ) -> Tuple[Optional["pd.Series"], Optional["pd.Series"]]:
        """Extract NIFTY 50 prices and VIX history for crash avoidance state updates."""
        nifty_symbol = "NIFTY 50"
        vix_symbol = "INDIA VIX"
        lookback_days = 100

        try:
            from_date = as_of_date - timedelta(days=lookback_days)
            nifty_df = self.market_data.get_historical(
                symbol=nifty_symbol,
                from_date=from_date,
                to_date=as_of_date,
                interval="day",
                check_quality=False,
            )
            if nifty_df.empty or len(nifty_df) < 63:
                return None, None

            nifty_prices = nifty_df["close"]

            vix_history = None
            try:
                vix_df = self.market_data.get_historical(
                    symbol=vix_symbol,
                    from_date=as_of_date - timedelta(days=40),
                    to_date=as_of_date,
                    interval="day",
                    check_quality=False,
                )
                if not vix_df.empty and len(vix_df) >= 5:
                    vix_history = vix_df["close"]
            except Exception:
                pass

            return nifty_prices, vix_history
        except Exception:
            return None, None

    def select_portfolio_with_regime(
        self,
        as_of_date: datetime,
        portfolio_value: float,
        max_per_sector: int = 3,
        profile_max_gold: Optional[float] = None,
    ) -> Tuple[Dict[str, float], Optional[RegimeResult]]:
        """
        Select portfolio with regime-based allocation.

        Combines momentum stock selection with market regime detection
        to automatically adjust allocation during defensive markets.

        LIVE MODE PARITY: Now passes regime, drawdown, and bull recovery signals
        to the strategy for adaptive parameter adjustment (same as backtest).

        Args:
            as_of_date: Date for calculation
            portfolio_value: Total portfolio value
            max_per_sector: Maximum stocks per sector

        Returns:
            Tuple of (weights dict, regime result)
        """
        # Detect current market regime
        regime = self.detect_current_regime(as_of_date)

        # Update drawdown tracking
        self.update_drawdown(portfolio_value)

        # Pass regime to strategy for adaptive parameters (LIVE MODE PARITY)
        if regime and self._strategy is not None:
            if hasattr(self._strategy, "set_regime"):
                self._strategy.set_regime(regime)

            # Pass drawdown for recovery mode
            if hasattr(self._strategy, "set_drawdown"):
                self._strategy.set_drawdown(self._current_drawdown, as_of_date)

            # Calculate and pass bull recovery signals
            if hasattr(self._strategy, "update_bull_recovery_state"):
                bull_recovery_signals = self._calculate_bull_recovery_signals(as_of_date, regime)
                if bull_recovery_signals:
                    self._strategy.update_bull_recovery_state(as_of_date, bull_recovery_signals)

            # Parity Fix 1a: Update crash avoidance state (matches backtest.py:1750-1772)
            if hasattr(self._strategy, "update_crash_avoidance_state"):
                nifty_prices, vix_history = self._get_crash_avoidance_data(as_of_date)
                if nifty_prices is not None:
                    self._strategy.update_crash_avoidance_state(
                        as_of_date=as_of_date,
                        market_prices=nifty_prices,
                        vix_level=regime.vix_level,
                        vix_history=vix_history,
                    )

            # Parity Fix 1b: Update adaptive lookback (matches backtest.py:1775-1776)
            if hasattr(self._strategy, "update_adaptive_lookback"):
                self._strategy.update_adaptive_lookback(regime.vix_level)

            # Parity Fix 1c: Update breadth state (matches backtest.py:1779-1781)
            if hasattr(self._strategy, "update_breadth_state"):
                try:
                    breadth = self._compute_live_breadth(as_of_date)
                    self._strategy.update_breadth_state(breadth)
                except Exception:
                    pass

        # Select top momentum stocks
        top_stocks = self.select_top_stocks(
            as_of_date=as_of_date,
            n=self.sizing_config.target_positions,
            max_per_sector=max_per_sector,
        )

        if not top_stocks:
            return {}, regime

        # Calculate target weights
        raw_weights = self.calculate_target_weights(
            selected_stocks=top_stocks,
            portfolio_value=portfolio_value,
        )

        # Apply sector limits with dynamic caps (E4)
        effective_sector_cap = self._get_effective_sector_cap(regime)
        weights = self.apply_sector_limits(
            weights=raw_weights,
            selected_stocks=top_stocks,
            max_sector_override=effective_sector_cap,
        )

        # Per-profile gold cap: clamp regime gold weight before allocation
        if regime and profile_max_gold is not None and regime.gold_weight > profile_max_gold:
            freed = regime.gold_weight - profile_max_gold
            regime.gold_weight = profile_max_gold
            regime.equity_weight = min(1.0, regime.equity_weight + freed)
            regime.cash_weight = max(0.0, 1.0 - regime.equity_weight - regime.gold_weight)

        # Change 5: Recovery equity override with hysteresis (FIX 6)
        # Cap stress when drawdown + improving breadth, with confirmation period
        rcfg = self.regime_config
        if regime and rcfg.use_recovery_equity_override:
            conditions_met = False
            if self._current_drawdown < rcfg.recovery_override_dd_threshold:
                try:
                    current_b = self._compute_live_breadth(as_of_date)
                    past_date = as_of_date - timedelta(days=14)  # ~10 trading days
                    past_b = self._compute_live_breadth(past_date)
                    if current_b - past_b > rcfg.recovery_override_breadth_improvement:
                        conditions_met = True
                except Exception:
                    pass

            required_days = rcfg.recovery_override_confirmation_days
            if conditions_met:
                self._recovery_override_confirm_days = min(
                    self._recovery_override_confirm_days + 1, required_days + 1
                )
                if self._recovery_override_confirm_days >= required_days:
                    self._recovery_override_active = True
            else:
                self._recovery_override_confirm_days = max(
                    self._recovery_override_confirm_days - 1, -(required_days + 1)
                )
                if self._recovery_override_confirm_days <= -required_days:
                    self._recovery_override_active = False

            if self._recovery_override_active:
                import math

                capped_stress = min(regime.stress_score, rcfg.recovery_override_max_stress)
                steepness = rcfg.allocation_curve_steepness
                stress_curve = (
                    math.pow(capped_stress, 1.0 / steepness) if steepness > 0 else capped_stress
                )
                override_equity = 1.0 - stress_curve * (1.0 - rcfg.min_equity_allocation)
                effective_max_gold = (
                    profile_max_gold if profile_max_gold is not None else rcfg.max_gold_allocation
                )
                override_gold = stress_curve * effective_max_gold
                regime.equity_weight = max(regime.equity_weight, override_equity)
                regime.gold_weight = min(regime.gold_weight, override_gold)
                regime.cash_weight = max(0.0, 1.0 - regime.equity_weight - regime.gold_weight)

        # Apply regime-based allocation adjustments
        if regime and regime.equity_weight < 1.0:
            # Scale down equity weights
            equity_scale = regime.equity_weight
            for ticker in weights:
                weights[ticker] *= equity_scale

            # Add defensive positions (with gold skip check)
            if regime.gold_weight > 0:
                gold_volatile, _ = self._should_skip_gold(as_of_date)
                if gold_volatile:
                    redistrib_scale = 1.0 / (1.0 - regime.gold_weight)
                    for ticker in weights:
                        weights[ticker] *= redistrib_scale
                else:
                    # Apply gold exhaustion scaling (GE1)
                    gold_exhaust_scale = self._calculate_gold_exhaustion_scale(as_of_date)
                    effective_gold = regime.gold_weight * gold_exhaust_scale
                    if effective_gold > 0.001:
                        weights[self.regime_config.gold_symbol] = effective_gold
                    # Freed gold weight → equities (uptrend) or cash (downtrend)
                    freed_gold = regime.gold_weight - effective_gold
                    if freed_gold > 0.001:
                        self._redirect_freed_weight(weights, freed_gold, as_of_date)
            if regime.cash_weight > 0:
                weights[self.regime_config.cash_symbol] = (
                    weights.get(self.regime_config.cash_symbol, 0.0) + regime.cash_weight
                )

        # Apply vol targeting (E2) and breadth scaling (E3)
        vol_scale = self._calculate_vol_scale()
        breadth_scale = self._calculate_breadth_scale(as_of_date)
        combined_scale = vol_scale * breadth_scale
        if self._is_market_uptrend(as_of_date):
            combined_floor = self.regime_config.trend_scale_floor
        else:
            combined_floor = self.regime_config.combined_scale_floor
        effective_scale = max(combined_floor, combined_scale)

        if effective_scale < 1.0:
            defensive_symbols = {self.regime_config.gold_symbol, self.regime_config.cash_symbol}
            equity_before = sum(w for t, w in weights.items() if t not in defensive_symbols)
            for ticker in list(weights.keys()):
                if ticker not in defensive_symbols:
                    weights[ticker] *= effective_scale
            # Redirect freed equity to cash_symbol (Change 1)
            equity_after = sum(w for t, w in weights.items() if t not in defensive_symbols)
            freed_weight = equity_before - equity_after
            if freed_weight > 0.01:
                cash_sym = self.regime_config.cash_symbol
                weights[cash_sym] = weights.get(cash_sym, 0) + freed_weight

        # Parity Fix 2: Apply crash avoidance position scaling (matches backtest.py:1796-1810)
        if self._strategy is not None and hasattr(self._strategy, "get_position_scale"):
            position_scale = self._strategy.get_position_scale()
            if position_scale < 1.0:
                defensive_symbols = {self.regime_config.gold_symbol, self.regime_config.cash_symbol}
                eq_before = sum(w for t, w in weights.items() if t not in defensive_symbols)
                for ticker in list(weights.keys()):
                    if ticker not in defensive_symbols:
                        weights[ticker] *= position_scale
                eq_after = sum(w for t, w in weights.items() if t not in defensive_symbols)
                freed = eq_before - eq_after
                if freed > 0.01:
                    cash_sym = self.regime_config.cash_symbol
                    weights[cash_sym] = weights.get(cash_sym, 0) + freed

        # Catch-all: redirect any unallocated weight to cash_symbol
        # (e.g. when fewer stocks qualify than needed to fill 100% at max_single_position)
        total_weight = sum(weights.values())
        if total_weight < 0.999:
            shortfall = 1.0 - total_weight
            cash_sym = self.regime_config.cash_symbol
            weights[cash_sym] = weights.get(cash_sym, 0) + shortfall

        return weights, regime
