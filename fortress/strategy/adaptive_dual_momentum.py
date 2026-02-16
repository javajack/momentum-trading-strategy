"""
Adaptive Dual Momentum Strategy for FORTRESS MOMENTUM.

Research-backed approach combining:
1. Dual Momentum (Antonacci): Absolute + Relative momentum
2. Multi-timeframe regime detection with stress score
3. Recovery modes (bull, general, crash) for capturing rebounds
4. Tiered adaptive stops to let winners run
5. Adaptive trend break protection with buffer
6. Volatility targeting for position sizing

Based on research showing:
- Dual Momentum: +440 bps annually vs index (Antonacci)
- Quality + Momentum: 93% outperformance rate
- Volatility Targeting: Reduces max DD by 6.6%, can double Sharpe
- Recovery modes: Critical for capturing bull market rebounds

See strategies/README.md for detailed documentation.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

from ..indicators import (
    BullRecoverySignals,
    MarketRegime,
    RegimeResult,
    SimpleRegime,
    SimpleRegimeResult,
    calculate_adaptive_lookback,
    calculate_bull_recovery_signals,
    calculate_exhaustion_score,
    calculate_momentum_acceleration,
    calculate_normalized_momentum_score,
    calculate_position_momentum,
    calculate_relative_strength,
    detect_breadth_thrust,
    detect_momentum_crash,
    detect_simple_regime,
    detect_vix_recovery,
)
from ..utils import renormalize_with_caps
from .base import BaseStrategy, ExitSignal, StockScore, StopLossConfig
from .registry import StrategyRegistry

if TYPE_CHECKING:
    from ..config import Config
    from ..market_data import MarketDataProvider
    from ..universe import Universe


@dataclass
class RecoveryModeState:
    """Track recovery mode after significant drawdown."""

    is_active: bool = False
    triggered_date: Optional[datetime] = None
    trigger_drawdown: float = 0.0
    days_active: int = 0


@dataclass
class BullRecoveryState:
    """Track bull recovery mode during market rebounds."""

    is_active: bool = False
    triggered_date: Optional[datetime] = None
    days_active: int = 0
    recovery_strength: float = 0.0


@dataclass
class CrashRecoveryState:
    """Track crash recovery mode after extreme VIX spikes (>50)."""

    is_active: bool = False
    triggered_date: Optional[datetime] = None
    trigger_vix: float = 0.0
    days_active: int = 0


@dataclass
class CrashAvoidanceState:
    """
    Track momentum crash avoidance mode.

    Activated when market drops >10% in 1 month - research shows
    momentum strategies suffer "momentum crash" in these conditions.
    Switch to contrarian/reduced exposure for 1-3 months.

    Source: ScienceDirect - Momentum Crash Research
    """

    is_active: bool = False
    triggered_date: Optional[datetime] = None
    market_1m_return: float = 0.0
    days_active: int = 0
    contrarian_mode: bool = False
    position_scale: float = 1.0  # Scale down position sizes


@dataclass
class AdaptiveLookbackState:
    """Track adaptive lookback periods based on market conditions."""

    lookback_6m: int = 126
    lookback_12m: int = 252
    mode: str = "NORMAL"  # "NORMAL", "RECOVERY", "VOLATILE"


@dataclass
class AdaptedParameters:
    """Parameters adapted for current market regime."""

    # Entry thresholds
    min_rs_threshold: float
    min_52w_high_prox: float
    ema_buffer: float  # How far below 50 EMA is acceptable for ENTRY
    skip_200sma_check: bool  # Skip 200 SMA filter during crash recovery

    # Exit thresholds
    rs_exit_threshold: float

    # Stop loss widths (tiered)
    initial_stop_loss: float
    tier1_trailing: float
    tier2_trailing: float
    tier3_trailing: float
    tier4_trailing: float

    # Trend break thresholds
    trend_break_buffer: float
    trend_break_confirm_days: int

    # Context info
    stress_score: float
    recovery_mode_active: bool
    bull_recovery_active: bool
    crash_recovery_active: bool


class AdaptiveDualMomentumStrategy(BaseStrategy):
    """
    Adaptive Dual Momentum strategy with regime adaptation.

    Entry Logic (Dual Momentum):
        1. Absolute Momentum: NMS > 0 (stock has positive momentum)
        2. Relative Momentum: RS > threshold (stock beats benchmark)
        3. Trend Filter: Price > 50-day EMA (with adaptive buffer)
        4. Liquidity: Daily turnover >= Rs 1 Cr
        5. Optional: 52W high proximity filter (adaptive by regime)

    Scoring:
        Score = NMS * (1 + rs_weight * (RS - 1))
        RS adjustment boosts outperformers, penalizes laggards

    Exit Rules:
        1. Hard Stop: -15% from entry
        2. Tiered Trailing Stop: 12-22% from peak (based on gain level)
        3. Trend Break: Price below 50-EMA with buffer + confirmation
        4. RS Floor: Exit if RS drops below 0.95

    Position Sizing:
        Volatility-targeted: Scale inversely to stock volatility
        High volatility -> smaller positions -> same risk exposure
        Low volatility -> larger positions -> capture more upside

    Regime Adaptation:
        - Uses full RegimeResult with stress score (0-1)
        - Bullish (stress < 0.3): Relaxed filters, wider stops
        - Defensive (stress > 0.6): Stricter filters, tighter stops
        - Recovery modes: Bull, General, Crash for capturing rebounds
    """

    def __init__(self, config: Optional["Config"] = None):
        super().__init__(config)
        self._excluded_symbols: Set[str] = {
            "LIQUIDCASE",
            "LIQUIDBEES",
            "NIFTYBEES",
            "JUNIORBEES",
            "MID150BEES",
            "HDFCSML250",
            "GOLDBEES",
            "HANGSENGBEES",
        }

        # Regime tracking
        self._current_regime: Optional[RegimeResult] = None
        self._simple_regime: Optional[SimpleRegimeResult] = None
        self._current_vix: float = 15.0  # Default to calm

        # Recovery state tracking
        self._recovery_state = RecoveryModeState()
        self._bull_recovery_state = BullRecoveryState()
        self._crash_recovery_state = CrashRecoveryState()
        self._current_drawdown: float = 0.0
        self._vix_history: Optional[object] = None

        # I1: Sideways market state
        self._is_sideways: bool = False

        # Crash avoidance state (NEW - momentum crash protection)
        self._crash_avoidance_state = CrashAvoidanceState()

        # Adaptive lookback state (NEW - dynamic lookback periods)
        self._adaptive_lookback_state = AdaptiveLookbackState()

        # Breadth tracking for thrust detection (NEW)
        self._breadth_history: List[float] = []
        self._breadth_thrust_active: bool = False

        # Track days below 50-EMA for trend break confirmation
        self._days_below_ema: Dict[str, int] = {}

        # Track peak prices for trailing stops
        self._peak_prices: Dict[str, float] = {}

        # P3: Cached config values dict (config never changes during a run)
        self._cached_config_values: Optional[dict] = None

    @property
    def name(self) -> str:
        return "dual_momentum"

    @property
    def description(self) -> str:
        return "Adaptive dual momentum with regime detection and recovery modes"

    def set_vix(self, vix_value: float) -> None:
        """Update current VIX for regime detection."""
        self._current_vix = vix_value

    def set_regime(self, regime) -> None:
        """
        Update current market regime.

        Accepts either RegimeResult (full) or SimpleRegimeResult for flexibility.
        """
        if isinstance(regime, RegimeResult):
            self._current_regime = regime
        elif isinstance(regime, SimpleRegimeResult):
            self._simple_regime = regime

    def set_sideways(self, is_sideways: bool) -> None:
        """I1: Update sideways market state for buffer widening."""
        self._is_sideways = is_sideways

    def set_vix_history(self, vix_history) -> None:
        """Store VIX history for bull recovery detection."""
        self._vix_history = vix_history

    def set_drawdown(self, drawdown: float, as_of_date: datetime) -> None:
        """Update drawdown and check recovery mode."""
        self._current_drawdown = drawdown
        self._recovery_state = self._check_recovery_mode(as_of_date)

    def update_bull_recovery_state(
        self,
        as_of_date: datetime,
        bull_recovery_signals: BullRecoverySignals,
    ) -> None:
        """
        Update bull recovery state based on market signals.

        Also updates crash recovery state if VIX is extremely high.
        """
        cfg = self._get_config_values()

        if not cfg.get("use_recovery_modes", True):
            return

        self._bull_recovery_state = self._check_bull_recovery_mode(
            as_of_date, bull_recovery_signals
        )

        # Also check crash recovery mode (triggered by extreme VIX)
        vix_level = self._current_vix
        if self._current_regime is not None:
            vix_level = self._current_regime.vix_level

        self._crash_recovery_state = self._check_crash_recovery_mode(as_of_date, vix_level)

    def _check_recovery_mode(self, as_of_date: datetime) -> RecoveryModeState:
        """
        Check if general recovery mode should be active based on drawdown.
        """
        cfg = self._get_config_values()

        if not cfg.get("use_recovery_modes", True):
            return RecoveryModeState()

        trigger = cfg.get("recovery_drawdown_trigger", -0.07)
        duration = cfg.get("recovery_duration_days", 60)

        # Check if duration expired first (if currently active)
        if self._recovery_state.is_active:
            if self._recovery_state.triggered_date is None:
                return RecoveryModeState()
            # Approximate trading days from calendar days (252 trading / 365.25 calendar)
            calendar_days = (as_of_date - self._recovery_state.triggered_date).days
            days_elapsed = int(calendar_days * 252 / 365.25)
            if days_elapsed >= duration:
                return RecoveryModeState()
            return RecoveryModeState(
                is_active=True,
                triggered_date=self._recovery_state.triggered_date,
                trigger_drawdown=self._recovery_state.trigger_drawdown,
                days_active=days_elapsed,
            )

        # Check if should trigger new recovery mode
        if self._current_drawdown <= trigger:
            return RecoveryModeState(
                is_active=True,
                triggered_date=as_of_date,
                trigger_drawdown=self._current_drawdown,
                days_active=0,
            )

        return RecoveryModeState()

    def _check_bull_recovery_mode(
        self,
        as_of_date: datetime,
        bull_recovery_signals: BullRecoverySignals,
    ) -> BullRecoveryState:
        """
        Check if bull recovery mode should be active.
        """
        cfg = self._get_config_values()

        if not cfg.get("use_bull_recovery_mode", True):
            return BullRecoveryState()

        duration = cfg.get("bull_recovery_duration_days", 60)

        # Check if duration expired first (if currently active)
        if self._bull_recovery_state.is_active:
            if self._bull_recovery_state.triggered_date is None:
                return BullRecoveryState()
            # Approximate trading days from calendar days (252 trading / 365.25 calendar)
            calendar_days = (as_of_date - self._bull_recovery_state.triggered_date).days
            days_elapsed = int(calendar_days * 252 / 365.25)
            if days_elapsed >= duration:
                return BullRecoveryState()
            # Maintain or update strength
            new_strength = max(
                self._bull_recovery_state.recovery_strength * 0.95,
                bull_recovery_signals.recovery_strength,
            )
            return BullRecoveryState(
                is_active=True,
                triggered_date=self._bull_recovery_state.triggered_date,
                days_active=days_elapsed,
                recovery_strength=new_strength,
            )

        # Check if should trigger new bull recovery mode
        if bull_recovery_signals.is_bull_recovery:
            return BullRecoveryState(
                is_active=True,
                triggered_date=as_of_date,
                days_active=0,
                recovery_strength=bull_recovery_signals.recovery_strength,
            )

        return BullRecoveryState()

    def _check_crash_recovery_mode(
        self, as_of_date: datetime, vix_level: float
    ) -> CrashRecoveryState:
        """
        Check if crash recovery mode should be active (VIX spike > 50).
        """
        cfg = self._get_config_values()

        if not cfg.get("use_crash_recovery_mode", True):
            return CrashRecoveryState()

        trigger = cfg.get("crash_recovery_vix_trigger", 50.0)
        duration = cfg.get("crash_recovery_duration_days", 90)

        # Check if duration expired first (if currently active)
        if self._crash_recovery_state.is_active:
            if self._crash_recovery_state.triggered_date is None:
                return CrashRecoveryState()
            # Approximate trading days from calendar days (252 trading / 365.25 calendar)
            calendar_days = (as_of_date - self._crash_recovery_state.triggered_date).days
            days_elapsed = int(calendar_days * 252 / 365.25)
            if days_elapsed >= duration:
                return CrashRecoveryState()
            return CrashRecoveryState(
                is_active=True,
                triggered_date=self._crash_recovery_state.triggered_date,
                trigger_vix=self._crash_recovery_state.trigger_vix,
                days_active=days_elapsed,
            )

        # Check if should trigger new crash recovery mode
        if vix_level >= trigger:
            return CrashRecoveryState(
                is_active=True,
                triggered_date=as_of_date,
                trigger_vix=vix_level,
                days_active=0,
            )

        return CrashRecoveryState()

    def update_crash_avoidance_state(
        self,
        as_of_date: datetime,
        market_prices: "pd.Series",
        vix_level: float,
        vix_history: Optional["pd.Series"] = None,
    ) -> None:
        """
        Update crash avoidance state based on market crash detection.

        Momentum crash avoidance mode activates when:
        1. Market drops >10% in 1 month
        2. VIX spikes significantly

        In these conditions, momentum strategies suffer "momentum crash"
        where past losers outperform past winners for 1-3 months.

        Source: ScienceDirect - Momentum Crash Research
        """
        # Get config for crash avoidance settings
        cfg = self._get_config_values()

        # Check if crash avoidance is enabled
        if not cfg.get("use_crash_avoidance", True):
            self._crash_avoidance_state = CrashAvoidanceState()
            return

        # Detect momentum crash conditions
        crash_signal = detect_momentum_crash(
            market_prices=market_prices,
            vix_current=vix_level,
            vix_history=vix_history,
            crash_threshold=cfg.get("crash_avoidance_threshold", -0.07),
            vix_spike_threshold=cfg.get("crash_avoidance_vix_threshold", 30.0),
            early_warning_threshold=cfg.get("crash_early_warning_threshold", -0.05),
            early_warning_3m_threshold=cfg.get("crash_early_warning_3m_threshold", -0.08),
        )

        duration = cfg.get("crash_avoidance_duration", 60)
        # E6: Use lighter scale for early-warning (REDUCE_MOMENTUM from slow grind)
        early_warning_scale = cfg.get("crash_early_warning_scale", 0.80)
        position_scale = cfg.get("crash_avoidance_position_scale", 0.6)

        # Check if duration expired first (if currently active)
        if self._crash_avoidance_state.is_active:
            if self._crash_avoidance_state.triggered_date is None:
                self._crash_avoidance_state = CrashAvoidanceState()
                return

            days_elapsed = (as_of_date - self._crash_avoidance_state.triggered_date).days
            if days_elapsed >= duration:
                logger.info(f"Crash avoidance mode expired after {days_elapsed} days")
                self._crash_avoidance_state = CrashAvoidanceState()
                return

            # Continue in crash avoidance mode
            self._crash_avoidance_state = CrashAvoidanceState(
                is_active=True,
                triggered_date=self._crash_avoidance_state.triggered_date,
                market_1m_return=crash_signal.market_1m_return,
                days_active=days_elapsed,
                contrarian_mode=crash_signal.recommendation == "CONTRARIAN",
                position_scale=position_scale,
            )
            return

        # Check if should trigger new crash avoidance mode
        if crash_signal.is_crash:
            # E6: Use lighter scale for early-warning slow grind vs full crash
            effective_scale = position_scale
            if (
                crash_signal.recommendation == "REDUCE_MOMENTUM"
                and not crash_signal.volatility_spike
            ):
                # Early warning (slow grind): lighter reduction
                effective_scale = early_warning_scale
            logger.warning(
                f"Crash avoidance ACTIVATED: 1M return={crash_signal.market_1m_return:.1%}, "
                f"recommendation={crash_signal.recommendation}, scale={effective_scale:.0%}"
            )
            self._crash_avoidance_state = CrashAvoidanceState(
                is_active=True,
                triggered_date=as_of_date,
                market_1m_return=crash_signal.market_1m_return,
                days_active=0,
                contrarian_mode=crash_signal.recommendation == "CONTRARIAN",
                position_scale=effective_scale,
            )
        else:
            self._crash_avoidance_state = CrashAvoidanceState()

    def update_adaptive_lookback(
        self,
        vix_level: float,
    ) -> None:
        """
        Update adaptive lookback periods based on market conditions.

        Research shows optimal lookbacks vary with conditions:
        - Post-crash/recovery: Shorter lookbacks capture V-shaped rebounds
        - High volatility: Longer lookbacks reduce whipsaws
        - Normal markets: Standard lookbacks work well

        Source: Dynamic Momentum Learning (arXiv:2106.08420)
        """
        cfg = self._get_config_values()

        # Check if adaptive lookback is enabled
        if not cfg.get("use_adaptive_lookback", True):
            self._adaptive_lookback_state = AdaptiveLookbackState(
                lookback_6m=cfg["lookback_6m"],
                lookback_12m=cfg["lookback_12m"],
                mode="NORMAL",
            )
            return

        # Calculate adaptive lookbacks based on conditions
        lookback_6m, lookback_12m, mode = calculate_adaptive_lookback(
            base_lookback_6m=cfg["lookback_6m"],
            base_lookback_12m=cfg["lookback_12m"],
            portfolio_drawdown=self._current_drawdown,
            vix_level=vix_level,
            drawdown_threshold=cfg.get("adaptive_lookback_dd_threshold", 0.05),
            vix_threshold=cfg.get("adaptive_lookback_vix_threshold", 30.0),
            recovery_multiplier=cfg.get("adaptive_lookback_recovery_mult", 0.5),
            volatile_multiplier=cfg.get("adaptive_lookback_volatile_mult", 1.5),
        )

        # Update state
        if mode != self._adaptive_lookback_state.mode:
            logger.info(
                f"Adaptive lookback changed: {self._adaptive_lookback_state.mode} -> {mode} "
                f"(6M: {self._adaptive_lookback_state.lookback_6m} -> {lookback_6m}, "
                f"12M: {self._adaptive_lookback_state.lookback_12m} -> {lookback_12m})"
            )

        self._adaptive_lookback_state = AdaptiveLookbackState(
            lookback_6m=lookback_6m,
            lookback_12m=lookback_12m,
            mode=mode,
        )

    def update_breadth_state(
        self,
        current_breadth: float,
    ) -> None:
        """
        Update breadth tracking and check for breadth thrust.

        Breadth thrust is a powerful bullish signal when breadth moves
        from oversold to overbought very quickly (within 10 days).
        """
        import pandas as pd

        # Add to history (keep last 15 days)
        self._breadth_history.append(current_breadth)
        if len(self._breadth_history) > 15:
            self._breadth_history = self._breadth_history[-15:]

        # Check for breadth thrust if we have enough history
        if len(self._breadth_history) >= 11:
            cfg = self._get_config_values()
            breadth_series = pd.Series(self._breadth_history)

            thrust_result = detect_breadth_thrust(
                breadth_history=breadth_series,
                thrust_low=cfg.get("breadth_thrust_low", 0.40),
                thrust_high=cfg.get("breadth_thrust_high", 0.615),
                max_days=cfg.get("breadth_thrust_days", 10),
            )

            if thrust_result.is_thrust and not self._breadth_thrust_active:
                logger.info(
                    f"Breadth THRUST detected: {thrust_result.breadth_10d_ago:.1%} -> "
                    f"{thrust_result.current_breadth:.1%} in {thrust_result.days_to_thrust} days"
                )
            self._breadth_thrust_active = thrust_result.is_thrust

    def get_position_scale(self) -> float:
        """
        Get position size multiplier based on crash avoidance state.

        Returns 1.0 in normal conditions, reduced scale during crash avoidance.
        """
        if self._crash_avoidance_state.is_active:
            return self._crash_avoidance_state.position_scale
        return 1.0

    def is_in_crash_avoidance(self) -> bool:
        """Check if crash avoidance mode is active."""
        return self._crash_avoidance_state.is_active

    def is_breadth_thrust_active(self) -> bool:
        """Check if breadth thrust signal is active."""
        return self._breadth_thrust_active

    def get_effective_lookbacks(self) -> tuple:
        """
        Get effective lookback periods (may be adaptive).

        Returns:
            Tuple of (lookback_6m, lookback_12m)
        """
        lb_6m = self._adaptive_lookback_state.lookback_6m
        lb_12m = self._adaptive_lookback_state.lookback_12m

        # I7: Regime-aware lookback — shorten in CAUTION/DEFENSIVE
        cfg = self._get_config_values()
        if cfg.get("use_regime_adaptive_lookback", False) and self._current_regime:
            regime = self._current_regime.regime
            if regime in (MarketRegime.CAUTION, MarketRegime.DEFENSIVE):
                mult = cfg.get("regime_lookback_mult", 0.50)
                lb_6m = max(21, int(lb_6m * mult))
                lb_12m = max(42, int(lb_12m * mult))

        return (lb_6m, lb_12m)

    def _get_config_values(self) -> dict:
        """Get configuration values with defaults. Cached after first call."""
        if self._cached_config_values is not None:
            return self._cached_config_values

        defaults = {
            # Dual Momentum
            "min_rs_threshold": 1.05,
            "rs_weight": 0.25,
            "min_nms_for_entry": 0.0,
            "rs_exit_threshold": 0.94,
            # Simple Regime
            "vix_bullish_threshold": 18.0,
            "vix_defensive_threshold": 25.0,
            # Feature toggles
            "use_adaptive_parameters": True,
            "use_recovery_modes": True,
            "use_tiered_stops": True,
            "use_full_regime": True,
            # Recovery settings
            "recovery_drawdown_trigger": -0.07,
            "recovery_duration_days": 60,
            "recovery_filter_relaxation": 0.25,
            "use_bull_recovery_mode": True,
            "bull_recovery_filter_relaxation": 0.25,
            "bull_recovery_vix_threshold": 20.0,
            "bull_recovery_momentum_threshold": 0.003,
            "bull_recovery_duration_days": 60,
            "use_crash_recovery_mode": True,
            "crash_recovery_vix_trigger": 50.0,
            "crash_recovery_duration_days": 90,
            "crash_recovery_52w_mult": 0.75,
            "crash_recovery_ema_buffer": 0.15,
            # Tiered stops
            "tier1_threshold": 0.08,
            "tier2_threshold": 0.20,
            "tier3_threshold": 0.50,
            "tier1_trailing": 0.12,
            "tier2_trailing": 0.14,
            "tier3_trailing": 0.16,
            "tier4_trailing": 0.22,
            # Regime multipliers
            "rs_bullish_mult": 0.85,
            "rs_defensive_mult": 1.05,
            "stop_bullish_mult": 1.25,
            "stop_defensive_mult": 0.85,
            # Trend break
            "trend_break_buffer": 0.035,
            "trend_break_days": 2,
            "trend_break_buffer_bullish_mult": 1.67,
            "trend_break_buffer_defensive_mult": 0.0,
            "trend_break_confirm_bullish_mult": 1.5,
            "trend_break_confirm_defensive_mult": 0.5,
            # Stop Losses (legacy)
            "hard_stop": 0.15,
            "trailing_stop": 0.15,
            "trailing_activation": 0.08,
            "defensive_trailing_stop": 0.10,
            # Volatility Targeting
            "target_volatility": 0.15,
            "max_vol_scale": 1.5,
            "high_vol_threshold": 0.25,
            "high_vol_reduction": 0.70,
            # Entry Filters
            "min_daily_turnover": 10_000_000,
            "defensive_rs_boost": 0.10,
            "min_52w_high_prox": 0.85,
            "high_52w_bullish_mult": 0.90,
            "high_52w_defensive_mult": 1.05,
            # Partial filter passing
            "use_partial_filter_passing": True,
            "partial_filter_min_passed": 2,
            "partial_filter_score_penalty": 0.04,
            # NMS Parameters (from pure_momentum config)
            "lookback_6m": 126,
            "lookback_12m": 252,
            "lookback_volatility": 126,
            "skip_recent_days": 5,
            "weight_6m": 0.50,
            "weight_12m": 0.50,
            # Position sizing
            "max_single_position": 0.08,
            "min_single_position": 0.03,
            "max_sector_exposure": 0.30,
            # === NEW: Crash Avoidance Settings ===
            "use_crash_avoidance": True,
            "crash_avoidance_threshold": -0.07,  # Market 1M return threshold (E6: -0.10→-0.07)
            "crash_avoidance_vix_threshold": 30.0,  # VIX spike threshold
            "crash_avoidance_duration": 60,  # Days to maintain mode
            "crash_avoidance_position_scale": 0.6,  # Scale down positions 40%
            # E6: Early warning crash detection
            "crash_early_warning_threshold": -0.05,  # 1M return for early warning
            "crash_early_warning_3m_threshold": -0.08,  # 3M return confirmation
            "crash_early_warning_scale": 0.80,  # Reduce to 80% positions (not 60%)
            # === NEW: Adaptive Lookback Settings ===
            "use_adaptive_lookback": True,
            "adaptive_lookback_dd_threshold": 0.05,  # DD to trigger recovery mode
            "adaptive_lookback_vix_threshold": 30.0,  # VIX to trigger volatile mode
            "adaptive_lookback_recovery_mult": 0.5,  # 50% shorter lookbacks
            "adaptive_lookback_volatile_mult": 1.5,  # 50% longer lookbacks
            # === NEW: Breadth Thrust Settings ===
            "breadth_thrust_low": 0.40,  # Starting breadth level
            "breadth_thrust_high": 0.615,  # Thrust confirmation level
            "breadth_thrust_days": 10,  # Max days for thrust
            # === NEW: Sector Momentum Filter (E5) ===
            "use_sector_momentum": True,
            "sector_momentum_lookback": 126,  # 6 months
            "sector_exclude_bottom": 3,  # Exclude bottom 3 sectors
            # === E9: Minimum Hold Period ===
            "min_hold_days": 3,  # Only hard stop during first 3 days
            # === I3: Momentum Deceleration Filter ===
            "deceleration_penalty": 0.12,  # 12% score penalty for decelerating stocks
            "deceleration_threshold": 0.85,  # Acceleration ratio below which penalty applies
            # === I7: Regime-Aware NMS Lookback ===
            "use_regime_adaptive_lookback": True,
            "regime_lookback_mult": 0.50,
            # === I1: Sideways Market Detection ===
            "use_sideways_detection": True,
            "sideways_hold_days": 7,
            "sideways_buffer_mult": 1.5,
            "sideways_rebalance_days": 12,
        }

        # Override with config if available
        if self._config is not None:
            # Strategy-specific config
            if hasattr(self._config, "strategy_dual_momentum"):
                sc = self._config.strategy_dual_momentum
                for key in defaults:
                    if hasattr(sc, key):
                        defaults[key] = getattr(sc, key)

            # NMS parameters from pure_momentum
            if hasattr(self._config, "pure_momentum"):
                pm = self._config.pure_momentum
                for key in [
                    "lookback_6m",
                    "lookback_12m",
                    "lookback_volatility",
                    "skip_recent_days",
                    "weight_6m",
                    "weight_12m",
                ]:
                    if hasattr(pm, key):
                        defaults[key] = getattr(pm, key)

            # Position sizing from position_sizing
            if hasattr(self._config, "position_sizing"):
                ps = self._config.position_sizing
                for key in ["max_single_position", "min_single_position", "max_sector_exposure"]:
                    if hasattr(ps, key):
                        defaults[key] = getattr(ps, key)

        self._cached_config_values = defaults
        return defaults

    def _get_stress_score(self) -> float:
        """
        Get current stress score (0 = bullish, 1 = defensive).

        Uses full RegimeResult if available, otherwise derives from SimpleRegimeResult.
        """
        cfg = self._get_config_values()

        # Prefer full regime result
        if cfg.get("use_full_regime", True) and self._current_regime is not None:
            return self._current_regime.stress_score

        # Fall back to simple regime
        if self._simple_regime is not None:
            if self._simple_regime.regime == SimpleRegime.BULLISH:
                return 0.15  # Low stress
            elif self._simple_regime.regime == SimpleRegime.DEFENSIVE:
                return 0.75  # High stress
            else:
                return 0.40  # Normal stress

        # Default to neutral
        return 0.40

    def _adaptive_param(
        self,
        base: float,
        bullish_mult: float,
        defensive_mult: float,
    ) -> float:
        """
        Scale parameter based on stress level.

        Args:
            base: Base parameter value from config
            bullish_mult: Multiplier when stress = 0 (fully bullish)
            defensive_mult: Multiplier when stress = 1 (fully defensive)

        Returns:
            Adapted parameter value
        """
        cfg = self._get_config_values()

        if not cfg.get("use_adaptive_parameters", True):
            return base

        stress = self._get_stress_score()

        # Apply adaptation curve (higher = more aggressive at extremes)
        adaptation_curve = 2.0
        curved_stress = stress ** (1.0 / adaptation_curve)

        # Linear interpolation between bullish and defensive multipliers
        adjustment = bullish_mult + curved_stress * (defensive_mult - bullish_mult)

        # Recovery mode: bias toward bullish parameters
        if cfg.get("use_recovery_modes", True) and self._recovery_state.is_active:
            relax = cfg.get("recovery_filter_relaxation", 0.25)
            if defensive_mult > bullish_mult:
                adjustment -= relax * (defensive_mult - bullish_mult)
            else:
                adjustment += relax * (bullish_mult - defensive_mult)

        # Bull recovery mode: even more aggressive relaxation
        if cfg.get("use_bull_recovery_mode", True) and self._bull_recovery_state.is_active:
            relax = (
                cfg.get("bull_recovery_filter_relaxation", 0.25)
                * self._bull_recovery_state.recovery_strength
            )
            if defensive_mult > bullish_mult:
                adjustment -= relax * (defensive_mult - bullish_mult)
            else:
                adjustment += relax * (bullish_mult - defensive_mult)

        return base * adjustment

    def _get_adapted_parameters(self) -> AdaptedParameters:
        """Get all parameters adapted for current regime."""
        cfg = self._get_config_values()
        stress = self._get_stress_score()

        # Start with regime-adaptive values
        min_52w = self._adaptive_param(
            cfg["min_52w_high_prox"],
            cfg.get("high_52w_bullish_mult", 0.90),
            cfg.get("high_52w_defensive_mult", 1.05),
        )
        ema_buffer = self._adaptive_param(
            cfg["trend_break_buffer"],
            cfg.get("trend_break_buffer_bullish_mult", 1.67),
            cfg.get("trend_break_buffer_defensive_mult", 0.0),
        )

        # I1: Widen trend break buffer in sideways markets
        if self._is_sideways:
            ema_buffer *= cfg.get("sideways_buffer_mult", 1.5)

        # Stop loss multipliers
        stop_bullish = cfg.get("stop_bullish_mult", 1.25)
        stop_defensive = cfg.get("stop_defensive_mult", 0.85)

        # CRASH RECOVERY MODE: Most aggressive relaxation
        skip_200sma = False
        if self._crash_recovery_state.is_active and cfg.get("use_crash_recovery_mode", True):
            min_52w = cfg["min_52w_high_prox"] * cfg.get("crash_recovery_52w_mult", 0.75)
            ema_buffer = cfg.get("crash_recovery_ema_buffer", 0.15)
            skip_200sma = True

        # Compute tier trailing stops
        tier1 = self._adaptive_param(cfg.get("tier1_trailing", 0.12), stop_bullish, stop_defensive)
        tier2 = self._adaptive_param(cfg.get("tier2_trailing", 0.14), stop_bullish, stop_defensive)
        tier3 = self._adaptive_param(cfg.get("tier3_trailing", 0.16), stop_bullish, stop_defensive)
        tier4 = self._adaptive_param(cfg.get("tier4_trailing", 0.22), stop_bullish, stop_defensive)

        # Change 4: Widen stops during recovery modes (differentiated by type)
        # Bull recovery: full widening (market recovering, give room)
        # General recovery: moderate widening
        # Crash recovery: no widening (crashes need tight stops)
        if self._bull_recovery_state.is_active:
            recovery_mult = cfg.get("stop_recovery_mult", 1.50)
        elif self._recovery_state.is_active:
            recovery_mult = cfg.get("stop_recovery_mult_general", 1.25)
        else:
            recovery_mult = None  # No widening (crash or no recovery)

        if recovery_mult is not None:
            tier1 = max(tier1, cfg.get("tier1_trailing", 0.12) * recovery_mult)
            tier2 = max(tier2, cfg.get("tier2_trailing", 0.14) * recovery_mult)
            tier3 = max(tier3, cfg.get("tier3_trailing", 0.16) * recovery_mult)
            tier4 = max(tier4, cfg.get("tier4_trailing", 0.22) * recovery_mult)

        return AdaptedParameters(
            # Entry thresholds
            min_rs_threshold=self._adaptive_param(
                cfg["min_rs_threshold"],
                cfg.get("rs_bullish_mult", 0.85),
                cfg.get("rs_defensive_mult", 1.05),
            ),
            min_52w_high_prox=min_52w,
            ema_buffer=ema_buffer,
            skip_200sma_check=skip_200sma,
            # Exit thresholds
            rs_exit_threshold=cfg.get("rs_exit_threshold", 0.94),
            # Stop loss widths (using tiered, with recovery widening)
            initial_stop_loss=self._adaptive_param(
                cfg["hard_stop"],
                stop_bullish,
                stop_defensive,
            ),
            tier1_trailing=tier1,
            tier2_trailing=tier2,
            tier3_trailing=tier3,
            tier4_trailing=tier4,
            # Trend break thresholds
            trend_break_buffer=ema_buffer,
            trend_break_confirm_days=max(
                1,
                round(
                    self._adaptive_param(
                        float(cfg.get("trend_break_days", 2)),
                        cfg.get("trend_break_confirm_bullish_mult", 1.5),
                        cfg.get("trend_break_confirm_defensive_mult", 0.5),
                    )
                ),
            ),
            # Context info
            stress_score=stress,
            recovery_mode_active=self._recovery_state.is_active,
            bull_recovery_active=self._bull_recovery_state.is_active,
            crash_recovery_active=self._crash_recovery_state.is_active,
        )

    def _is_in_bullish_or_recovery(self) -> bool:
        """Check if conditions allow relaxed filters."""
        stress = self._get_stress_score()
        return (
            stress < 0.35
            or self._recovery_state.is_active
            or self._bull_recovery_state.is_active
            or self._crash_recovery_state.is_active
        )

    def _calculate_sector_momentum(
        self,
        as_of_date: datetime,
        universe: "Universe",
        market_data: "MarketDataProvider",
    ) -> Dict[str, float]:
        """
        Calculate momentum for each sector using sectoral index data (E5).

        Returns dict mapping sector name to 6M return of its sectoral index.
        Sectors without index data get neutral (0.0) momentum.
        """
        cfg = self._get_config_values()
        lookback = cfg.get("sector_momentum_lookback", 126)
        from_date = as_of_date - timedelta(days=int(lookback * 1.8))

        sector_momentum: Dict[str, float] = {}

        for sector in universe.get_valid_sectors():
            idx_info = universe.get_sector_index(sector)
            if idx_info is None:
                sector_momentum[sector] = 0.0  # Neutral if no index
                continue

            try:
                idx_df = market_data.get_historical(
                    symbol=idx_info.symbol,
                    from_date=from_date,
                    to_date=as_of_date,
                    interval="day",
                    check_quality=False,
                )
                if idx_df is None or len(idx_df) < lookback:
                    sector_momentum[sector] = 0.0
                    continue

                close = idx_df["close"]
                sector_momentum[sector] = close.iloc[-1] / close.iloc[-lookback] - 1
            except Exception:
                sector_momentum[sector] = 0.0

        return sector_momentum

    def _get_bottom_sectors(
        self,
        as_of_date: datetime,
        universe: "Universe",
        market_data: "MarketDataProvider",
    ) -> Set[str]:
        """
        Get bottom N sectors by momentum (E5), regime-aware.

        Returns set of sector names that should receive a score penalty.
        During any active recovery mode, returns empty set (skip filter).
        Exclusion count scales by regime: bullish=3, caution=2, defensive=1.
        """
        cfg = self._get_config_values()
        if not cfg.get("use_sector_momentum", True):
            return set()

        # During drawdown/crash recovery: skip sector filter
        # Bull recovery keeps filter active (market is bullish, diversification helps)
        if self._recovery_state.is_active or self._crash_recovery_state.is_active:
            return set()

        sector_momentum = self._calculate_sector_momentum(as_of_date, universe, market_data)
        if not sector_momentum:
            return set()

        # Scale exclusion count by regime
        stress = self._get_stress_score()
        if stress < 0.35:  # bullish
            n_exclude = cfg.get("sector_exclude_bullish", 3)
        elif stress < 0.65:  # caution
            n_exclude = cfg.get("sector_exclude_caution", 2)
        else:  # defensive
            n_exclude = cfg.get("sector_exclude_defensive", 1)

        if n_exclude <= 0:
            return set()

        sorted_sectors = sorted(sector_momentum.items(), key=lambda x: x[1])
        return {s[0] for s in sorted_sectors[:n_exclude]}

    def rank_stocks(
        self,
        as_of_date: datetime,
        universe: "Universe",
        market_data: "MarketDataProvider",
        filter_entry: bool = True,
    ) -> List[StockScore]:
        """
        Rank stocks using enhanced dual momentum scoring.

        Score = NMS * (1 + rs_weight * (RS - 1))

        Entry filters (all must pass, or partial in bullish/recovery):
        1. NMS > 0 (absolute momentum)
        2. RS > threshold (relative momentum, adaptive)
        3. Price > 50-day EMA (with adaptive buffer)
        4. 52W high proximity (adaptive)
        5. Turnover >= minimum (liquidity)
        """
        cfg = self._get_config_values()
        adapted = self._get_adapted_parameters()

        # Get effective lookbacks (may be adaptive)
        effective_6m, effective_12m = self.get_effective_lookbacks()

        # Calculate lookback period for data fetch (use max possible lookback)
        max_lookback = max(cfg["lookback_12m"], effective_12m, 378)  # 378 = max volatile lookback
        lookback_days = max_lookback + cfg["skip_recent_days"] + 30
        from_date = as_of_date - timedelta(days=int(lookback_days * 1.5))

        # Sector momentum filter (E5): identify bottom sectors to exclude
        bottom_sectors = self._get_bottom_sectors(as_of_date, universe, market_data)

        # Get benchmark data for RS calculation
        benchmark_prices = None
        try:
            bench_df = market_data.get_historical(
                symbol="NIFTY 50",
                from_date=from_date,
                to_date=as_of_date,
                interval="day",
                check_quality=False,
            )
            if bench_df is not None and len(bench_df) >= 252:
                benchmark_prices = bench_df["close"]
        except Exception as e:
            logger.warning(f"Could not get benchmark data: {e}")

        scored_stocks: List[StockScore] = []

        for stock in universe.get_all_stocks():
            ticker = stock.ticker

            # Skip excluded symbols
            if ticker in self._excluded_symbols:
                continue

            try:
                # Get price and volume data
                df = market_data.get_historical(
                    symbol=stock.zerodha_symbol,
                    from_date=from_date,
                    to_date=as_of_date,
                    interval="day",
                    check_quality=False,
                )

                if df is None or df.empty:
                    continue

                prices = df["close"]
                volumes = df["volume"]

                if len(prices) < cfg["lookback_12m"]:
                    continue
            except Exception:
                continue

            # Calculate NMS using adaptive lookbacks
            nms_result = calculate_normalized_momentum_score(
                prices=prices,
                volumes=volumes,
                lookback_6m=effective_6m,
                lookback_12m=effective_12m,
                lookback_volatility=cfg["lookback_volatility"],
                skip_recent_days=cfg["skip_recent_days"],
                weight_6m=cfg["weight_6m"],
                weight_12m=cfg["weight_12m"],
            )

            if nms_result is None:
                continue

            # Calculate Relative Strength
            rs_result = None
            if benchmark_prices is not None:
                rs_result = calculate_relative_strength(
                    stock_prices=prices,
                    benchmark_prices=benchmark_prices,
                )

            # Default RS if calculation failed
            rs_composite = rs_result.rs_composite if rs_result else 1.0

            # Calculate exhaustion for distance from 50 EMA
            exhaustion_result = calculate_exhaustion_score(prices, volumes)
            distance_from_50ema = 0.0
            if exhaustion_result is not None:
                distance_from_50ema = exhaustion_result.distance_from_50ema

            # --- Entry Filter Checks ---
            filter_reasons = []
            filter_passes = {
                "nms": True,
                "rs": True,
                "ema": True,
                "high_52w": True,
                "turnover": True,
                "sma200": True,
            }

            # 1. Absolute Momentum: NMS > threshold
            if nms_result.nms <= cfg["min_nms_for_entry"]:
                filter_passes["nms"] = False
                filter_reasons.append(f"NMS {nms_result.nms:.2f} <= {cfg['min_nms_for_entry']}")

            # 2. Relative Momentum: RS > threshold (adaptive)
            rs_threshold = adapted.min_rs_threshold
            if rs_composite < rs_threshold:
                filter_passes["rs"] = False
                filter_reasons.append(f"RS {rs_composite:.2f} < {rs_threshold:.2f}")

            # 3. 50 EMA filter with adaptive buffer
            if distance_from_50ema < -adapted.ema_buffer:
                filter_passes["ema"] = False
                buffer_pct = adapted.ema_buffer * 100
                filter_reasons.append(
                    f"Below 50-EMA: {distance_from_50ema:.1%} (buffer: {buffer_pct:.0f}%)"
                )
            elif not nms_result.above_50ema and adapted.ema_buffer == 0:
                filter_passes["ema"] = False
                filter_reasons.append("Below 50-EMA")

            # 4. 52W high proximity (adaptive)
            if nms_result.high_52w_proximity < adapted.min_52w_high_prox:
                filter_passes["high_52w"] = False
                filter_reasons.append(
                    f"52W high: {nms_result.high_52w_proximity:.0%} < {adapted.min_52w_high_prox:.0%}"
                )

            # 5. Liquidity: Minimum turnover
            if nms_result.daily_turnover < cfg["min_daily_turnover"]:
                filter_passes["turnover"] = False
                filter_reasons.append(
                    f"Turnover {nms_result.daily_turnover / 1e6:.1f}M < {cfg['min_daily_turnover'] / 1e6:.1f}M"
                )

            # 6. 200-SMA trend filter (skipped during crash recovery)
            if not adapted.skip_200sma_check and not nms_result.above_200sma:
                filter_passes["sma200"] = False
                filter_reasons.append("Below 200-SMA")

            # 7. Sector momentum filter (E5): soft penalty instead of hard exclude
            in_bottom_sector = bool(bottom_sectors and stock.sector in bottom_sectors)
            if in_bottom_sector:
                filter_reasons.append(
                    f"Sector '{stock.sector}' in bottom {len(bottom_sectors)} (penalty)"
                )

            # Count how many core filters pass (NMS, RS, EMA)
            core_filters_passed = sum(
                [filter_passes["nms"], filter_passes["rs"], filter_passes["ema"]]
            )
            all_filters_passed = all(filter_passes.values())

            # Apply partial filter passing logic
            is_bullish_or_recovery = self._is_in_bullish_or_recovery()
            passes = False

            if all_filters_passed:
                passes = True
            elif cfg.get("use_partial_filter_passing", True) and is_bullish_or_recovery:
                # In bullish/recovery: allow partial passes if core filters >= min required
                min_required = cfg.get("partial_filter_min_passed", 2)
                if core_filters_passed >= min_required and filter_passes["turnover"]:
                    passes = True
                    filter_reasons.append(f"Partial pass: {core_filters_passed}/3 core filters")

            # --- Enhanced Dual Momentum Scoring ---
            # Score = NMS * (1 + rs_weight * (RS - 1))
            base_score = nms_result.nms
            rs_adjustment = (rs_composite - 1.0) * cfg["rs_weight"]
            score = base_score * (1.0 + rs_adjustment)

            # Apply penalty for partial pass
            if passes and not all_filters_passed:
                score *= 1.0 - cfg.get("partial_filter_score_penalty", 0.04)

            # Apply sector momentum soft penalty (Change 2)
            if in_bottom_sector:
                score *= 1.0 - cfg.get("sector_momentum_penalty", 0.15)

            # I3: Momentum deceleration penalty — penalize stocks with fading momentum
            decel_penalty = cfg.get("deceleration_penalty", 0.0)
            if decel_penalty > 0:
                accel = calculate_momentum_acceleration(prices, short_period=21, medium_period=63)
                decel_threshold = cfg.get("deceleration_threshold", 0.85)
                if accel < decel_threshold:
                    score *= 1.0 - decel_penalty

            # Create StockScore
            stock_score = StockScore(
                ticker=ticker,
                sector=stock.sector,
                sub_sector=stock.sub_sector,
                zerodha_symbol=stock.zerodha_symbol,
                name=stock.name,
                score=score,
                passes_entry_filters=passes,
                filter_reasons=filter_reasons,
                return_6m=nms_result.return_6m,
                return_12m=nms_result.return_12m,
                volatility=nms_result.volatility_6m,
                high_52w_proximity=nms_result.high_52w_proximity,
                above_50ema=nms_result.above_50ema,
                above_200sma=nms_result.above_200sma,
                volume_surge=nms_result.volume_surge,
                daily_turnover=nms_result.daily_turnover,
                current_price=prices.iloc[-1],
                extra_metrics={
                    "nms": nms_result.nms,
                    "rs_composite": rs_composite,
                    "rs_21d": rs_result.rs_21d if rs_result else 1.0,
                    "rs_63d": rs_result.rs_63d if rs_result else 1.0,
                    "distance_from_50ema": distance_from_50ema,
                    "adapted_rs_threshold": rs_threshold,
                    "adapted_52w_threshold": adapted.min_52w_high_prox,
                },
            )

            scored_stocks.append(stock_score)

        # Sort by score (descending)
        scored_stocks.sort(key=lambda x: x.score, reverse=True)

        # Filter if requested
        if filter_entry:
            scored_stocks = [s for s in scored_stocks if s.passes_entry_filters]

        # Assign ranks and percentiles
        total = len(scored_stocks)
        for i, stock in enumerate(scored_stocks):
            stock.rank = i + 1
            stock.percentile = 100 * (total - i) / total if total > 0 else 0

        return scored_stocks

    def select_portfolio(
        self,
        ranked_stocks: List[StockScore],
        portfolio_value: float,
        current_positions: Dict[str, float],
        max_positions: int,
        max_per_sector: int,
    ) -> Dict[str, float]:
        """
        Select portfolio from ranked stocks with sector diversification.
        """
        cfg = self._get_config_values()

        # Filter to passing stocks only
        candidates = [s for s in ranked_stocks if s.passes_entry_filters]

        # Select stocks respecting sector limits
        selected: List[StockScore] = []
        sector_count: Dict[str, int] = defaultdict(int)

        for stock in candidates:
            if len(selected) >= max_positions:
                break

            # Check sector limit
            if sector_count[stock.sector] >= max_per_sector:
                continue

            selected.append(stock)
            sector_count[stock.sector] += 1

        if not selected:
            return {}

        # Calculate weights with volatility targeting
        weights = self.calculate_weights(selected, portfolio_value)

        return weights

    def calculate_weights(
        self,
        selected_stocks: List[StockScore],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Calculate volatility-targeted weights.

        High volatility stocks get smaller weights.
        Low volatility stocks get larger weights.
        """
        if not selected_stocks:
            return {}

        cfg = self._get_config_values()
        target_vol = cfg["target_volatility"]
        max_scale = cfg["max_vol_scale"]
        high_vol_threshold = cfg["high_vol_threshold"]
        high_vol_reduction = cfg["high_vol_reduction"]

        raw_weights: Dict[str, float] = {}

        for stock in selected_stocks:
            # Get stock volatility (annualized)
            stock_vol = max(stock.volatility, 0.05)  # Floor at 5%

            # Volatility ratio: target_vol / actual_vol
            vol_ratio = target_vol / stock_vol

            # Scale weight (capped at max_scale)
            base_weight = 1.0  # Equal weight base
            scaled_weight = base_weight * min(vol_ratio, max_scale)

            # Additional reduction in high volatility environment
            if stock_vol > high_vol_threshold:
                scaled_weight *= high_vol_reduction

            raw_weights[stock.ticker] = scaled_weight

        # Normalize to sum to 1.0
        total = sum(raw_weights.values())
        if total > 0:
            weights = {k: v / total for k, v in raw_weights.items()}
        else:
            n = len(selected_stocks)
            weights = {s.ticker: 1.0 / n for s in selected_stocks}

        # Apply position size limits
        max_pos = cfg["max_single_position"]
        min_pos = cfg["min_single_position"]

        # Cap at max and redistribute
        excess = 0.0
        for ticker, weight in weights.items():
            if weight > max_pos:
                excess += weight - max_pos
                weights[ticker] = max_pos
            elif weight < min_pos:
                excess += weight
                weights[ticker] = 0.0

        # Remove zero-weight positions
        weights = {k: v for k, v in weights.items() if v > 0}

        # Redistribute excess proportionally
        if excess > 0 and weights:
            remaining_total = sum(weights.values())
            if remaining_total > 0:
                for ticker in weights:
                    if weights[ticker] < max_pos:
                        add_weight = excess * (weights[ticker] / remaining_total)
                        weights[ticker] = min(weights[ticker] + add_weight, max_pos)

        # Final normalization with iterative capping to maintain position limits
        # Uses shared utility for logic parity with backtest and live rebalance
        return renormalize_with_caps(weights, max_pos, min_pos)

    def check_exit_triggers(
        self,
        ticker: str,
        entry_price: float,
        current_price: float,
        peak_price: float,
        days_held: int,
        stock_score: Optional[StockScore],
        nms_percentile: float,
    ) -> ExitSignal:
        """
        Check exit triggers with adaptive thresholds.

        1. Hard Stop: -15% from entry
        2. Tiered Trailing Stop: 12-22% from peak (based on gain tier)
        3. Trend Break: Price below 50-EMA with buffer + confirmation
        4. RS Floor: Exit if RS drops below 0.95
        """
        cfg = self._get_config_values()
        adapted = self._get_adapted_parameters()

        # Calculate gains
        gain = (current_price - entry_price) / entry_price
        drawdown_from_peak = (current_price - peak_price) / peak_price if peak_price > 0 else 0

        # Get stop config for current gain tier
        stop_config = self.get_stop_loss_config(ticker, gain)

        # --- Rule 1: Hard Stop (always active, even during min hold period) ---
        if gain <= -stop_config.initial_stop:
            self._days_below_ema.pop(ticker, None)
            return ExitSignal(
                should_exit=True,
                reason=f"Hard stop: {gain:.1%} loss (threshold: {-stop_config.initial_stop:.1%})",
                exit_type="stop_loss",
                urgency="immediate",
            )

        # E9: Minimum hold period — skip soft exits during first N days
        min_hold = cfg.get("min_hold_days", 3)
        if days_held < min_hold:
            return ExitSignal(
                should_exit=False,
                reason=f"Hold (min hold: {days_held}/{min_hold} days)",
                exit_type="none",
                urgency="normal",
            )

        # --- Rule 2: Trailing Stop (tiered) ---
        if gain >= stop_config.trailing_activation:
            if drawdown_from_peak <= -stop_config.trailing_stop:
                tier_info = ""
                if stop_config.tiers:
                    tier_info = f" [{stop_config.tiers.get('current_tier', 'N/A')}]"
                self._days_below_ema.pop(ticker, None)
                return ExitSignal(
                    should_exit=True,
                    reason=f"Trailing stop{tier_info}: {drawdown_from_peak:.1%} from peak",
                    exit_type="trailing_stop",
                    urgency="immediate",
                )

        # --- Rule 3: Trend Break with buffer + confirmation ---
        if stock_score is not None:
            distance_from_ema = stock_score.extra_metrics.get("distance_from_50ema", 0.0)

            # Check if below the buffered threshold
            below_buffer = distance_from_ema < -adapted.trend_break_buffer

            if below_buffer:
                self._days_below_ema[ticker] = self._days_below_ema.get(ticker, 0) + 1
            else:
                self._days_below_ema[ticker] = 0

            # Check if confirmation days requirement is met
            days_below = self._days_below_ema.get(ticker, 0)
            if days_below >= adapted.trend_break_confirm_days:
                self._days_below_ema[ticker] = 0
                buffer_pct = adapted.trend_break_buffer * 100
                return ExitSignal(
                    should_exit=True,
                    reason=f"Trend break: {distance_from_ema:.1%} below 50-EMA (buffer: {buffer_pct:.0f}%, {days_below}d confirmed)",
                    exit_type="trend_break",
                    urgency="next_rebalance",
                )

        # --- Rule 4: RS Floor ---
        if stock_score is not None and "rs_composite" in stock_score.extra_metrics:
            rs = stock_score.extra_metrics["rs_composite"]
            if rs < adapted.rs_exit_threshold:
                self._days_below_ema.pop(ticker, None)
                return ExitSignal(
                    should_exit=True,
                    reason=f"RS floor: {rs:.2f} < {adapted.rs_exit_threshold:.2f}",
                    exit_type="rs_floor",
                    urgency="next_rebalance",
                )

        # Hold
        return ExitSignal(
            should_exit=False,
            reason="Hold",
            exit_type="none",
            urgency="normal",
        )

    def get_stop_loss_config(
        self,
        ticker: str,
        current_gain: float,
    ) -> StopLossConfig:
        """
        Get tiered adaptive stop loss configuration.

        Tiers (let winners run):
        - Tier 1 (<8%): 12% trailing
        - Tier 2 (8-20%): 14% trailing
        - Tier 3 (20-50%): 16% trailing
        - Tier 4 (>50%): 22% trailing

        Regime adaptation:
        - Bullish: Wider stops (1.25x)
        - Defensive: Tighter stops (0.85x)
        """
        cfg = self._get_config_values()
        adapted = self._get_adapted_parameters()

        if not cfg.get("use_tiered_stops", True):
            # Fall back to simple stops
            stress = self._get_stress_score()
            if stress > 0.6:
                trailing = cfg.get("defensive_trailing_stop", 0.10)
            else:
                trailing = cfg.get("trailing_stop", 0.15)

            return StopLossConfig(
                initial_stop=cfg["hard_stop"],
                trailing_stop=trailing,
                trailing_activation=cfg["trailing_activation"],
                use_tiered=False,
                tiers=None,
            )

        # Tiered stops based on current gain
        tier1_threshold = cfg.get("tier1_threshold", 0.08)
        tier2_threshold = cfg.get("tier2_threshold", 0.20)
        tier3_threshold = cfg.get("tier3_threshold", 0.50)

        if current_gain >= tier3_threshold:  # >= 50%
            trailing = adapted.tier4_trailing
            tier = "tier4"
        elif current_gain >= tier2_threshold:  # >= 20%
            trailing = adapted.tier3_trailing
            tier = "tier3"
        elif current_gain >= tier1_threshold:  # >= 8%
            trailing = adapted.tier2_trailing
            tier = "tier2"
        else:
            trailing = adapted.tier1_trailing
            tier = "tier1"

        return StopLossConfig(
            initial_stop=adapted.initial_stop_loss,
            trailing_stop=trailing,
            trailing_activation=cfg["trailing_activation"],
            use_tiered=True,
            tiers={
                "current_tier": tier,
                "tier1": adapted.tier1_trailing,
                "tier2": adapted.tier2_trailing,
                "tier3": adapted.tier3_trailing,
                "tier4": adapted.tier4_trailing,
            },
        )

    def get_config_schema(self) -> Dict:
        """Return strategy-specific config parameters."""
        return {
            "strategy_dual_momentum": {
                "min_rs_threshold": "Base RS for entry (1.05 = beat index by 5%)",
                "rs_weight": "RS weight for score boost (0.25)",
                "use_adaptive_parameters": "Enable regime-adaptive parameter scaling",
                "use_recovery_modes": "Enable recovery modes (bull, general, crash)",
                "use_tiered_stops": "Enable tiered trailing stops based on gain level",
                "tier1_trailing": "Trailing stop for tier 1 (<8% gain): 12%",
                "tier2_trailing": "Trailing stop for tier 2 (8-20% gain): 14%",
                "tier3_trailing": "Trailing stop for tier 3 (20-50% gain): 16%",
                "tier4_trailing": "Trailing stop for tier 4 (>50% gain): 22%",
                "trend_break_buffer": "Base buffer below 50 EMA (3%)",
                "trend_break_days": "Base confirmation days (2)",
            }
        }


# Auto-register strategy
StrategyRegistry.register(AdaptiveDualMomentumStrategy)
