"""
Configuration models for FORTRESS MOMENTUM.

Pure momentum strategy configuration only.
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ZerodhaConfig(BaseModel):
    """Zerodha API credentials."""

    api_key: str = Field(default="", description="Kite Connect API key")
    api_secret: str = Field(default="", description="Kite Connect API secret")

    @field_validator("api_key", "api_secret")
    @classmethod
    def check_not_placeholder(cls, v: str) -> str:
        if v in ("your_api_key_here", "your_api_secret_here"):
            raise ValueError("Please set actual API credentials in config.yaml")
        return v


class PortfolioConfig(BaseModel):
    """Portfolio settings."""

    initial_capital: float = Field(default=1600000, gt=0)
    max_positions: int = Field(default=20, ge=1, le=50)


class PureMomentumConfig(BaseModel):
    """
    Pure momentum strategy configuration (Nifty 500 Momentum 50 style).

    Uses Normalized Momentum Score (NMS) for volatility-adjusted stock ranking.
    """

    # NMS calculation parameters
    lookback_6m: int = Field(default=126, ge=20, description="6-month lookback in trading days")
    lookback_12m: int = Field(default=252, ge=100, description="12-month lookback in trading days")
    lookback_volatility: int = Field(default=126, ge=20, description="Volatility calculation lookback")
    skip_recent_days: int = Field(default=5, ge=0, le=20, description="Skip recent days to avoid reversal")
    weight_6m: float = Field(default=0.40, ge=0, le=1, description="Weight for 6M adjusted return")
    weight_12m: float = Field(default=0.60, ge=0, le=1, description="Weight for 12M adjusted return")

    # Entry filters
    min_score_percentile: float = Field(default=95, ge=50, le=100, description="Min NMS percentile for entry")
    min_52w_high_prox: float = Field(default=0.85, ge=0.5, le=1.0, description="Min proximity to 52-week high")
    min_volume_ratio: float = Field(default=1.1, ge=0.5, le=5.0, description="Min 20d/50d volume ratio")
    min_daily_turnover: float = Field(default=20_000_000, ge=0, description="Min avg daily turnover")

    # Exit triggers
    min_hold_percentile: float = Field(default=50, ge=0, le=100, description="Exit if NMS falls below this")
    max_days_without_gain: int = Field(default=60, ge=10, description="Max days to hold without target gain")
    min_gain_threshold: float = Field(default=0.10, ge=0, le=1, description="Target gain for time-based exit")

    @field_validator("weight_12m")
    @classmethod
    def weights_sum_to_one(cls, v: float, info) -> float:
        weight_6m = info.data.get("weight_6m", 0.50)
        if abs(weight_6m + v - 1.0) > 0.01:
            raise ValueError("weight_6m + weight_12m must equal 1.0")
        return v


class PositionSizingConfig(BaseModel):
    """Position sizing configuration."""

    method: str = Field(default="momentum_weighted", description="Sizing method")
    max_single_position: float = Field(default=0.08, gt=0, le=0.20)
    min_single_position: float = Field(default=0.03, gt=0, le=0.10)
    max_sector_exposure: float = Field(default=0.30, gt=0, le=0.50)
    target_positions: int = Field(default=15, ge=5, le=30)
    min_positions: int = Field(default=12, ge=5, le=20)
    max_positions: int = Field(default=20, ge=10, le=50)

    # Dynamic sector caps (E4) - tighter caps in non-bullish regimes
    use_dynamic_sector_caps: bool = Field(
        default=True,
        description="Reduce sector caps in CAUTION/DEFENSIVE regimes",
    )
    caution_max_sector: float = Field(
        default=0.25,
        ge=0.15,
        le=0.40,
        description="Max sector exposure in CAUTION regime (25%)",
    )
    defensive_max_sector: float = Field(
        default=0.20,
        ge=0.10,
        le=0.30,
        description="Max sector exposure in DEFENSIVE regime (20%)",
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid = ("equal", "momentum_weighted", "inverse_volatility")
        if v.lower() not in valid:
            raise ValueError(f"method must be one of {valid}")
        return v.lower()


class RiskConfig(BaseModel):
    """Risk management limits."""

    max_single_position: float = Field(default=0.08, gt=0, le=1)
    hard_max_position: float = Field(default=0.12, gt=0, le=1)
    max_sector_exposure: float = Field(default=0.35, gt=0, le=1)
    hard_max_sector: float = Field(default=0.45, gt=0, le=1)
    max_drawdown_warning: float = Field(default=0.15, gt=0, le=1)
    max_drawdown_halt: float = Field(default=0.25, gt=0, le=1)
    daily_loss_limit: float = Field(default=0.03, gt=0, le=1)

    # Stop loss settings
    initial_stop_loss: float = Field(default=0.18, gt=0, le=0.5, description="Initial stop loss")
    trailing_stop: float = Field(default=0.15, gt=0, le=0.5, description="Trailing stop")
    trailing_activation: float = Field(default=0.08, gt=0, le=0.5, description="Gain to activate trailing")


class RebalancingConfig(BaseModel):
    """Rebalancing schedule configuration."""

    frequency: str = Field(default="monthly", description="Rebalance frequency")
    rebalance_days: int = Field(default=21, ge=1, le=252, description="Trading days between rebalances (21 = monthly)")
    day: str = Field(default="friday", description="Day of week for rebalancing")
    min_trade_value: float = Field(default=10000, ge=0, description="Minimum trade value")

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, v: str) -> str:
        valid = ("daily", "weekly", "monthly")
        if v.lower() not in valid:
            raise ValueError(f"frequency must be one of {valid}")
        return v.lower()


class CostsConfig(BaseModel):
    """Transaction cost settings."""

    transaction_cost: float = Field(default=0.003, ge=0, le=0.1)


class PathsConfig(BaseModel):
    """File path settings."""

    universe_file: str = Field(default="stock-universe.json")
    log_dir: str = Field(default="logs")
    data_cache: str = Field(default=".cache")


class RegimeConfig(BaseModel):
    """
    Market regime detection configuration.

    Enhanced multi-signal approach with:
    1. Multi-timeframe range position (21/63/126 day weighted composite)
    2. VIX levels with graduated thresholds
    3. Return signals (1-month and 3-month)
    4. Bidirectional transitions with hysteresis
    5. Graduated allocation based on stress score

    Defensive allocation kicks in during CAUTION/DEFENSIVE regimes.
    """

    enabled: bool = Field(default=True, description="Enable regime detection")

    # Multi-timeframe lookbacks (NEW - faster detection)
    lookback_short: int = Field(
        default=21,
        ge=10,
        le=42,
        description="Short-term lookback (1 month fast signal)",
    )
    lookback_medium: int = Field(
        default=63,
        ge=42,
        le=126,
        description="Medium-term lookback (3 months intermediate)",
    )
    lookback_long: int = Field(
        default=126,
        ge=63,
        le=252,
        description="Long-term lookback (6 months trend confirmation)",
    )
    weight_short: float = Field(
        default=0.30,
        ge=0.0,
        le=0.5,
        description="Weight for short-term signal (30% - increased for faster response)",
    )
    weight_medium: float = Field(
        default=0.35,
        ge=0.0,
        le=0.6,
        description="Weight for medium-term signal (35%)",
    )
    weight_long: float = Field(
        default=0.35,
        ge=0.0,
        le=0.7,
        description="Weight for long-term signal (35% - reduced for faster response)",
    )

    # Adjusted entry thresholds (tighter bands for faster response)
    bullish_threshold: float = Field(
        default=0.65,
        ge=0.5,
        le=1.0,
        description="Above this = BULLISH regime (was 0.70)",
    )
    caution_threshold: float = Field(
        default=0.45,
        ge=0.2,
        le=0.7,
        description="Below this = CAUTION regime (was 0.50)",
    )
    defensive_threshold: float = Field(
        default=0.25,
        ge=0.1,
        le=0.5,
        description="Below this = DEFENSIVE regime (was 0.30)",
    )

    # Recovery thresholds (REDUCED asymmetric penalty for faster recovery)
    bullish_recovery_threshold: float = Field(
        default=0.70,
        ge=0.6,
        le=1.0,
        description="Recovery to BULLISH requires this level (reduced from 0.75)",
    )
    normal_recovery_threshold: float = Field(
        default=0.48,
        ge=0.4,
        le=0.8,
        description="Recovery to NORMAL requires this level (E7: 0.52→0.48)",
    )
    caution_recovery_threshold: float = Field(
        default=0.32,
        ge=0.2,
        le=0.6,
        description="Recovery from DEFENSIVE to CAUTION (reduced from 0.40)",
    )

    # Hysteresis settings (confirmation periods - reduced asymmetry)
    upgrade_confirmation_days: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Days to confirm upgrade (more defensive)",
    )
    downgrade_confirmation_days: int = Field(
        default=3,
        ge=1,
        le=15,
        description="Days to confirm downgrade (E7: 4→3)",
    )

    # Position momentum settings (NEW - detect trend reversals faster)
    use_position_momentum: bool = Field(
        default=True,
        description="Enable position momentum signal for faster recovery detection",
    )
    position_momentum_period: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Days to measure position rate-of-change",
    )
    position_momentum_recovery_bonus: float = Field(
        default=0.10,
        ge=0.0,
        le=0.15,
        description="Recovery threshold reduction when momentum > 0.005/day (E7/E10: 0.05→0.10)",
    )

    # VIX recovery accelerator settings (NEW - detect VIX mean-reversion)
    use_vix_recovery_accelerator: bool = Field(
        default=True,
        description="Enable VIX mean-reversion detection for faster recovery",
    )
    vix_recovery_spike_threshold: float = Field(
        default=25.0,
        ge=20.0,
        le=40.0,
        description="VIX level considered a spike",
    )
    vix_recovery_decline_rate: float = Field(
        default=0.10,
        ge=0.05,
        le=0.25,
        description="Minimum VIX decline from peak to trigger recovery signal",
    )
    vix_recovery_bonus: float = Field(
        default=0.03,
        ge=0.0,
        le=0.10,
        description="Recovery threshold reduction during VIX mean-reversion",
    )

    # Adaptive hysteresis settings (NEW - strong signals reduce confirmation)
    adaptive_hysteresis: bool = Field(
        default=True,
        description="Strong signals reduce confirmation period by 1 day",
    )
    strong_signal_bonus: float = Field(
        default=0.10,
        ge=0.05,
        le=0.20,
        description="Position must exceed threshold by this amount for strong signal",
    )

    # Short-term return settings (NEW - faster early detection)
    use_return_10d: bool = Field(
        default=True,
        description="Include 10-day return in stress calculation",
    )
    return_10d_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=0.30,
        description="Weight for 10-day return in stress score",
    )

    # VIX thresholds (ADJUSTED - raised to avoid over-reaction)
    vix_elevated: float = Field(
        default=18.0,
        ge=12.0,
        le=25.0,
        description="VIX early warning level",
    )
    vix_caution: float = Field(
        default=22.0,
        ge=15.0,
        le=35.0,
        description="VIX above this upgrades to CAUTION (was 20.0)",
    )
    vix_defensive: float = Field(
        default=28.0,
        ge=20.0,
        le=50.0,
        description="VIX above this forces DEFENSIVE (was 25.0)",
    )
    vix_normal: float = Field(
        default=19.0,
        ge=10.0,
        le=25.0,
        description="VIX below this allows recovery to NORMAL (E7: 16→19)",
    )
    vix_calm: float = Field(
        default=14.0,
        ge=8.0,
        le=18.0,
        description="VIX below this allows full recovery",
    )

    # Return thresholds (ENHANCED - includes 1-month warning)
    return_warning: float = Field(
        default=-0.03,
        ge=-0.10,
        le=0.0,
        description="1-month return warning threshold",
    )
    return_caution: float = Field(
        default=-0.05,
        ge=-0.20,
        le=0.0,
        description="3M return below this upgrades to CAUTION",
    )
    return_defensive: float = Field(
        default=-0.10,
        ge=-0.30,
        le=-0.05,
        description="3M return below this forces DEFENSIVE",
    )
    return_recovery_normal: float = Field(
        default=0.01,
        ge=0.0,
        le=0.10,
        description="Return above this allows recovery to NORMAL (E7: 0.03→0.01)",
    )
    return_recovery_bullish: float = Field(
        default=0.08,
        ge=0.03,
        le=0.20,
        description="Return above this allows recovery to BULLISH",
    )

    # E7: 2-of-3 recovery gate
    recovery_require_all_conditions: bool = Field(
        default=False,
        description="If false, recovery requires 2-of-3 conditions (position, VIX, returns). If true, all 3.",
    )

    # E10: Configurable stress score weights (replacing hardcoded 50/25/25)
    stress_weight_position: float = Field(
        default=0.40,
        ge=0.20,
        le=0.60,
        description="Position weight in stress score (E10: 0.50→0.40)",
    )
    stress_weight_vix: float = Field(
        default=0.30,
        ge=0.15,
        le=0.50,
        description="VIX weight in stress score (E10: 0.25→0.30)",
    )
    stress_weight_returns: float = Field(
        default=0.30,
        ge=0.15,
        le=0.50,
        description="Returns weight in stress score (E10: 0.25→0.30)",
    )

    # Graduated allocation settings (NEW - smooth transitions)
    use_graduated_allocation: bool = Field(
        default=True,
        description="Use smooth stress-based allocation instead of fixed steps",
    )
    min_equity_allocation: float = Field(
        default=0.60,
        ge=0.3,
        le=0.7,
        description="Minimum equity allocation at max stress",
    )
    max_gold_allocation: float = Field(
        default=0.15,
        ge=0.1,
        le=0.35,
        description="Maximum gold allocation at max stress",
    )
    max_cash_allocation: float = Field(
        default=0.0,
        ge=0.0,
        le=0.35,
        description="Maximum cash allocation at max stress (0 = no cash, user manages manually)",
    )
    allocation_curve_steepness: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Steepness of allocation curve (higher = more aggressive at extremes)",
    )

    # Defensive allocation - CAUTION regime (used when graduated is disabled)
    # Cash share moved to equity: 90% equity + 10% gold (was 80/10/10)
    caution_equity: float = Field(default=0.90, ge=0.5, le=1.0)
    caution_gold: float = Field(default=0.10, ge=0.0, le=0.25)
    caution_cash: float = Field(default=0.0, ge=0.0, le=0.25)

    # Defensive allocation - DEFENSIVE regime (used when graduated is disabled)
    # Cash share moved to equity: 80% equity + 20% gold (was 60/20/20)
    defensive_equity: float = Field(default=0.80, ge=0.3, le=1.0)
    defensive_gold: float = Field(default=0.20, ge=0.1, le=0.35)
    defensive_cash: float = Field(default=0.0, ge=0.0, le=0.35)

    # Gold skip logic: "downtrend" (skip if price < 50-SMA) or "volatile" (legacy: skip if volatile)
    gold_skip_logic: str = Field(
        default="downtrend",
        description="When to skip gold: 'downtrend' (price < 50-SMA) or 'volatile' (legacy high-vol check)",
    )

    # Gold exhaustion scaling (GE1): reduce gold allocation when gold is overextended above 200-SMA
    use_gold_exhaustion_scaling: bool = Field(
        default=True,
        description="Scale down gold allocation when gold is far above 200-SMA",
    )
    gold_exhaustion_sma_period: int = Field(
        default=200,
        ge=50,
        le=400,
        description="SMA period for gold exhaustion check",
    )
    gold_exhaustion_threshold_low: float = Field(
        default=0.15,
        ge=0.0,
        le=0.50,
        description="Gold deviation below this: full allocation (scale=1.0)",
    )
    gold_exhaustion_threshold_high: float = Field(
        default=0.40,
        ge=0.10,
        le=1.0,
        description="Gold deviation above this: zero allocation (scale=0.0)",
    )

    # Portfolio-level volatility targeting (E2)
    use_vol_targeting: bool = Field(
        default=True,
        description="Scale equity inversely to realized portfolio volatility",
    )
    target_portfolio_vol: float = Field(
        default=0.15,
        ge=0.05,
        le=0.30,
        description="Target annualized portfolio volatility (15%)",
    )
    vol_lookback_days: int = Field(
        default=21,
        ge=10,
        le=63,
        description="Lookback days for realized vol calculation",
    )
    vol_scale_floor: float = Field(
        default=0.40,
        ge=0.20,
        le=0.80,
        description="Minimum vol scale (never reduce equity below 40% of regime target)",
    )

    # Breadth-based exposure scaling (E3)
    use_breadth_scaling: bool = Field(
        default=True,
        description="Scale equity based on market breadth (% stocks above 50-DMA)",
    )
    breadth_full: float = Field(
        default=0.50,
        ge=0.30,
        le=0.80,
        description="Breadth above this = full equity (broad rally)",
    )
    breadth_low: float = Field(
        default=0.30,
        ge=0.20,
        le=0.50,
        description="Breadth below this = minimum scale (narrow market)",
    )
    breadth_min_scale: float = Field(
        default=0.50,
        ge=0.30,
        le=0.80,
        description="Minimum breadth scale when breadth is very low",
    )

    # Combined floor for vol + breadth scaling
    combined_scale_floor: float = Field(
        default=0.50,
        ge=0.20,
        le=0.70,
        description="Never reduce equity below 50% of regime target (vol × breadth floor)",
    )
    trend_scale_floor: float = Field(
        default=0.80,
        ge=0.50,
        le=1.0,
        description="Combined scale floor when NIFTY above 200-SMA (uptrend guard)",
    )

    # Recovery equity override (Change 5)
    use_recovery_equity_override: bool = Field(
        default=True,
        description="Cap stress when drawdown + improving breadth detected",
    )
    recovery_override_dd_threshold: float = Field(
        default=-0.05,
        ge=-0.20,
        le=-0.02,
        description="Drawdown threshold to activate recovery override (-5%)",
    )
    recovery_override_breadth_improvement: float = Field(
        default=0.05,
        ge=0.02,
        le=0.15,
        description="Required breadth improvement over 10 entries to trigger override",
    )
    recovery_override_max_stress: float = Field(
        default=0.35,
        ge=0.20,
        le=0.50,
        description="Cap stress to this value during recovery override",
    )

    # Defensive instruments
    gold_symbol: str = Field(default="GOLDBEES", description="Gold ETF symbol")
    cash_symbol: str = Field(default="LIQUIDBEES", description="Liquid fund ETF symbol")
    redirect_freed_to_equity_in_uptrend: bool = Field(
        default=True,
        description="In uptrend, redirect freed gold weight to equities pro-rata instead of cash",
    )

    @field_validator("defensive_threshold")
    @classmethod
    def defensive_less_than_caution(cls, v: float, info) -> float:
        caution = info.data.get("caution_threshold", 0.45)
        if v >= caution:
            raise ValueError("defensive_threshold must be < caution_threshold")
        return v

    @field_validator("caution_threshold")
    @classmethod
    def caution_less_than_bullish(cls, v: float, info) -> float:
        bullish = info.data.get("bullish_threshold", 0.65)
        if v >= bullish:
            raise ValueError("caution_threshold must be < bullish_threshold")
        return v

    @field_validator("weight_long")
    @classmethod
    def weights_sum_to_one(cls, v: float, info) -> float:
        weight_short = info.data.get("weight_short", 0.20)
        weight_medium = info.data.get("weight_medium", 0.35)
        if abs(weight_short + weight_medium + v - 1.0) > 0.01:
            raise ValueError("weight_short + weight_medium + weight_long must equal 1.0")
        return v

    @field_validator("normal_recovery_threshold")
    @classmethod
    def normal_recovery_greater_than_caution(cls, v: float, info) -> float:
        caution = info.data.get("caution_threshold", 0.45)
        if v <= caution:
            raise ValueError("normal_recovery_threshold must be > caution_threshold")
        return v

    @field_validator("bullish_recovery_threshold")
    @classmethod
    def bullish_recovery_greater_than_bullish(cls, v: float, info) -> float:
        bullish = info.data.get("bullish_threshold", 0.65)
        if v <= bullish:
            raise ValueError("bullish_recovery_threshold must be > bullish_threshold")
        return v


class DynamicRebalanceConfig(BaseModel):
    """
    Configuration for dynamic (event-driven) rebalancing.

    Research-backed triggers that supplement fixed-interval rebalancing:
    1. Regime Transition: Immediate rebalance on regime change
    2. Breadth Thrust: Aggressive entry on breadth improvement
    3. VIX Recovery: Opportunity rebalance when VIX declines from spike
    4. Drawdown Trigger: Defensive rebalance on portfolio drawdown
    5. Crash Avoidance: Switch to contrarian mode on market crash

    Sources:
    - Regime-Switching Signals Research (arXiv:2402.05272)
    - CME Momentum Research (improving-time-series-momentum)
    """

    enabled: bool = Field(
        default=True,
        description="Enable dynamic rebalancing triggers"
    )
    min_days_between: int = Field(
        default=7,
        ge=1,
        le=15,
        description="Minimum trading days between rebalances (E8: tuned)"
    )
    max_days_between: int = Field(
        default=30,
        ge=15,
        le=63,
        description="Force rebalance after this many days"
    )

    # Regime transition trigger
    regime_transition_trigger: bool = Field(
        default=True,
        description="Trigger rebalance on regime change (e.g., NORMAL→CAUTION)"
    )

    # VIX recovery trigger
    vix_recovery_trigger: bool = Field(
        default=True,
        description="Trigger rebalance when VIX recovers from spike"
    )
    vix_recovery_decline: float = Field(
        default=0.15,
        ge=0.10,
        le=0.30,
        description="VIX decline percentage from peak to trigger (15%)"
    )
    vix_spike_threshold: float = Field(
        default=25.0,
        ge=20.0,
        le=40.0,
        description="VIX level considered a spike"
    )

    # Portfolio drawdown trigger
    drawdown_trigger: bool = Field(
        default=True,
        description="Trigger defensive rebalance on portfolio drawdown"
    )
    drawdown_threshold: float = Field(
        default=0.10,
        ge=0.05,
        le=0.20,
        description="Portfolio drawdown threshold to trigger (10%)"
    )

    # Crash avoidance trigger (momentum crash detection)
    crash_avoidance_trigger: bool = Field(
        default=True,
        description="Trigger contrarian mode on market crash"
    )
    crash_threshold: float = Field(
        default=-0.07,
        ge=-0.20,
        le=-0.05,
        description="Market 1-month return threshold for crash (E6: -0.10→-0.07)"
    )
    crash_avoidance_duration: int = Field(
        default=60,
        ge=30,
        le=120,
        description="Trading days to maintain crash avoidance mode"
    )
    crash_position_scale: float = Field(
        default=0.6,
        ge=0.3,
        le=0.9,
        description="Position size multiplier during crash avoidance (0.6 = 40% reduction)"
    )

    # Breadth thrust trigger
    breadth_thrust_trigger: bool = Field(
        default=True,
        description="Trigger aggressive entry on breadth thrust"
    )
    breadth_thrust_low: float = Field(
        default=0.40,
        ge=0.25,
        le=0.50,
        description="Breadth level to start measuring thrust (40%)"
    )
    breadth_thrust_high: float = Field(
        default=0.615,
        ge=0.55,
        le=0.75,
        description="Breadth level that confirms thrust (61.5%)"
    )
    breadth_thrust_days: int = Field(
        default=10,
        ge=5,
        le=15,
        description="Max days for breadth to move from low to high"
    )


class AdaptiveLookbackConfig(BaseModel):
    """
    Configuration for adaptive momentum lookback periods.

    Research shows that optimal lookback periods vary with market conditions:
    - Post-crash/recovery: Shorter lookbacks capture V-shaped rebounds faster
    - High volatility: Longer lookbacks reduce whipsaws
    - Normal markets: Standard lookbacks work well

    Sources:
    - Dynamic Momentum Learning (arXiv:2106.08420)
    - ReSolve Asset Management (half-life-of-optimal-lookback-horizon)
    """

    enabled: bool = Field(
        default=True,
        description="Enable adaptive lookback periods"
    )

    # Normal market lookbacks (standard)
    normal_6m: int = Field(
        default=126,
        ge=63,
        le=189,
        description="6-month lookback in normal conditions"
    )
    normal_12m: int = Field(
        default=252,
        ge=126,
        le=378,
        description="12-month lookback in normal conditions"
    )

    # Recovery mode lookbacks (shorter for faster signals)
    recovery_6m: int = Field(
        default=63,
        ge=42,
        le=126,
        description="6-month lookback in recovery mode (faster)"
    )
    recovery_12m: int = Field(
        default=126,
        ge=63,
        le=189,
        description="12-month lookback in recovery mode (faster)"
    )

    # High volatility lookbacks (longer to reduce whipsaws)
    volatile_6m: int = Field(
        default=189,
        ge=126,
        le=252,
        description="6-month lookback in high volatility (slower)"
    )
    volatile_12m: int = Field(
        default=315,
        ge=252,
        le=378,
        description="12-month lookback in high volatility (slower)"
    )

    # Thresholds for switching modes
    drawdown_threshold: float = Field(
        default=0.05,
        ge=0.03,
        le=0.10,
        description="Drawdown level to switch to recovery lookbacks (5%)"
    )
    vix_volatile_threshold: float = Field(
        default=30.0,
        ge=25.0,
        le=40.0,
        description="VIX level to switch to volatile lookbacks"
    )

    # Multiplier approach (alternative to fixed lookbacks)
    use_multipliers: bool = Field(
        default=False,
        description="Use multipliers instead of fixed values"
    )
    recovery_multiplier: float = Field(
        default=0.5,
        ge=0.3,
        le=0.7,
        description="Lookback multiplier in recovery (0.5 = 50% shorter)"
    )
    volatile_multiplier: float = Field(
        default=1.5,
        ge=1.2,
        le=2.0,
        description="Lookback multiplier in volatile (1.5 = 50% longer)"
    )


class AdaptiveDualMomentumConfig(BaseModel):
    """
    Configuration for the Adaptive Dual Momentum strategy.

    Research-backed approach combining:
    1. Dual Momentum (Antonacci): Absolute + Relative momentum
    2. Multi-timeframe regime detection with stress score
    3. Recovery modes (bull, general, crash) for capturing rebounds
    4. Tiered adaptive stops to let winners run
    5. Adaptive trend break protection with buffer
    6. Volatility targeting for position sizing
    7. Dynamic rebalancing triggers
    8. Adaptive lookback periods
    9. Breadth-based regime enhancement
    10. Momentum crash avoidance

    See fortress/strategy/README.md for detailed documentation.
    """

    # === Dual Momentum Settings ===
    # Entry requires: NMS > 0 (absolute) AND RS > threshold (relative)
    min_rs_threshold: float = Field(
        default=1.05,
        ge=0.8,
        le=1.3,
        description="Base RS for entry (1.05 = beat index by 5%)"
    )
    rs_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=0.5,
        description="RS weight for score boost (score = NMS * (1 + rs_weight * (RS - 1)))"
    )
    min_nms_for_entry: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Minimum NMS for entry (0 = positive momentum)"
    )
    rs_exit_threshold: float = Field(
        default=0.94,
        ge=0.7,
        le=1.2,
        description="Exit if RS drops below this (E8: tuned)"
    )

    # === Simple Regime Detection (VIX-based) ===
    # BULLISH: VIX < bullish_threshold AND trend up
    # DEFENSIVE: VIX > defensive_threshold OR trend down
    # NEUTRAL: otherwise
    vix_bullish_threshold: float = Field(
        default=18.0,
        ge=10.0,
        le=25.0,
        description="VIX below this + uptrend = BULLISH"
    )
    vix_defensive_threshold: float = Field(
        default=25.0,
        ge=18.0,
        le=40.0,
        description="VIX above this = DEFENSIVE"
    )

    # === Feature Toggles ===
    use_adaptive_parameters: bool = Field(
        default=True,
        description="Enable regime-adaptive parameter scaling"
    )
    use_recovery_modes: bool = Field(
        default=True,
        description="Enable recovery modes (bull, general, crash)"
    )
    use_tiered_stops: bool = Field(
        default=True,
        description="Enable tiered trailing stops based on gain level"
    )
    use_full_regime: bool = Field(
        default=True,
        description="Use full RegimeResult instead of SimpleRegimeResult for adaptive params"
    )

    # === Recovery Mode Settings ===
    # General recovery: triggered by portfolio drawdown
    recovery_drawdown_trigger: float = Field(
        default=-0.07,
        ge=-0.30,
        le=-0.03,
        description="Drawdown level that triggers recovery mode (-7%)"
    )
    recovery_duration_days: int = Field(
        default=60,
        ge=14,
        le=120,
        description="Duration of recovery mode in days"
    )
    recovery_filter_relaxation: float = Field(
        default=0.25,
        ge=0.0,
        le=0.40,
        description="Filter relaxation during recovery (25%)"
    )

    # Bull recovery: triggered by VIX decline + positive momentum
    use_bull_recovery_mode: bool = Field(
        default=True,
        description="Enable bull recovery mode for V-shaped rebounds"
    )
    bull_recovery_filter_relaxation: float = Field(
        default=0.25,
        ge=0.0,
        le=0.50,
        description="Filter relaxation during bull recovery (25%)"
    )
    bull_recovery_vix_threshold: float = Field(
        default=20.0,
        ge=15.0,
        le=35.0,
        description="VIX must be declining from this level to trigger"
    )
    bull_recovery_momentum_threshold: float = Field(
        default=0.003,
        ge=0.0,
        le=0.01,
        description="Min position momentum per day to confirm recovery"
    )
    bull_recovery_duration_days: int = Field(
        default=60,
        ge=21,
        le=120,
        description="How long bull recovery mode lasts"
    )

    # Crash recovery: triggered by extreme VIX spike (>50)
    use_crash_recovery_mode: bool = Field(
        default=True,
        description="Enable crash recovery mode after VIX spikes >50"
    )
    crash_recovery_vix_trigger: float = Field(
        default=50.0,
        ge=35.0,
        le=80.0,
        description="VIX level that triggers crash recovery mode"
    )
    crash_recovery_duration_days: int = Field(
        default=90,
        ge=30,
        le=180,
        description="Duration of crash recovery mode"
    )
    crash_recovery_52w_mult: float = Field(
        default=0.75,
        ge=0.60,
        le=0.90,
        description="52W high multiplier during crash recovery (85% × 0.75 = 64%)"
    )
    crash_recovery_ema_buffer: float = Field(
        default=0.15,
        ge=0.05,
        le=0.25,
        description="Allow entries 15% below 50 EMA during crash recovery"
    )

    # === Tiered Stops Settings ===
    # Let winners run with progressively wider stops
    tier1_threshold: float = Field(
        default=0.08,
        ge=0.0,
        le=0.15,
        description="Gain threshold for tier 1->2 transition (8%)"
    )
    tier2_threshold: float = Field(
        default=0.20,
        ge=0.10,
        le=0.35,
        description="Gain threshold for tier 2->3 transition (20%)"
    )
    tier3_threshold: float = Field(
        default=0.50,
        ge=0.30,
        le=0.70,
        description="Gain threshold for tier 3->4 transition (50%)"
    )
    tier1_trailing: float = Field(
        default=0.12,
        ge=0.08,
        le=0.20,
        description="Trailing stop for tier 1 (<8% gain): 12%"
    )
    tier2_trailing: float = Field(
        default=0.14,
        ge=0.08,
        le=0.25,
        description="Trailing stop for tier 2 (8-20% gain): 14%"
    )
    tier3_trailing: float = Field(
        default=0.16,
        ge=0.10,
        le=0.30,
        description="Trailing stop for tier 3 (20-50% gain): 16%"
    )
    tier4_trailing: float = Field(
        default=0.22,
        ge=0.15,
        le=0.40,
        description="Trailing stop for tier 4 (>50% gain): 22%"
    )

    # === Regime Multipliers ===
    # Adjust thresholds based on regime (bullish relaxes, defensive tightens)
    rs_bullish_mult: float = Field(
        default=0.85,
        ge=0.75,
        le=1.0,
        description="RS threshold multiplier in bullish (1.05 × 0.85 = 0.89)"
    )
    rs_defensive_mult: float = Field(
        default=1.05,
        ge=1.0,
        le=1.20,
        description="RS threshold multiplier in defensive (1.05 × 1.05 = 1.10)"
    )
    stop_bullish_mult: float = Field(
        default=1.25,
        ge=1.0,
        le=1.5,
        description="Stop loss width multiplier in bullish (wider stops)"
    )
    stop_defensive_mult: float = Field(
        default=0.85,
        ge=0.6,
        le=1.0,
        description="Stop loss width multiplier in defensive (tighter stops)"
    )

    # === Trend Break Protection ===
    trend_break_buffer: float = Field(
        default=0.035,
        ge=0.0,
        le=0.10,
        description="Base buffer below 50 EMA before trend break exit (E8: tuned)"
    )
    trend_break_days: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Base days below 50-EMA to confirm trend break"
    )
    trend_break_buffer_bullish_mult: float = Field(
        default=1.67,
        ge=1.0,
        le=3.0,
        description="Buffer multiplier in bullish (3% × 1.67 = 5%)"
    )
    trend_break_buffer_defensive_mult: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Buffer multiplier in defensive (0% = immediate)"
    )
    trend_break_confirm_bullish_mult: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="Confirm days multiplier in bullish (2 × 1.5 = 3 days)"
    )
    trend_break_confirm_defensive_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confirm days multiplier in defensive (2 × 0.5 = 1 day)"
    )

    # === Legacy Stop Loss Settings (fallback) ===
    hard_stop: float = Field(
        default=0.15,
        ge=0.10,
        le=0.25,
        description="Hard stop loss from entry (-15%)"
    )
    trailing_stop: float = Field(
        default=0.15,
        ge=0.10,
        le=0.25,
        description="Trailing stop from peak (15%) - used when tiered disabled"
    )
    trailing_activation: float = Field(
        default=0.08,
        ge=0.05,
        le=0.40,
        description="Gain required to activate trailing (8%)"
    )
    defensive_trailing_stop: float = Field(
        default=0.10,
        ge=0.05,
        le=0.20,
        description="Tighter trailing stop in DEFENSIVE regime (10%)"
    )

    # === Volatility Targeting ===
    # Scale position size inversely to volatility
    target_volatility: float = Field(
        default=0.15,
        ge=0.05,
        le=0.30,
        description="Target annualized volatility (15%)"
    )
    max_vol_scale: float = Field(
        default=1.5,
        ge=1.0,
        le=2.0,
        description="Maximum position size multiplier (1.5x)"
    )
    high_vol_threshold: float = Field(
        default=0.25,
        ge=0.15,
        le=0.40,
        description="Volatility above this triggers reduction"
    )
    high_vol_reduction: float = Field(
        default=0.70,
        ge=0.50,
        le=0.90,
        description="Position multiplier in high volatility (0.7x)"
    )

    # === Entry Filters ===
    min_daily_turnover: float = Field(
        default=10_000_000,
        ge=1_000_000,
        le=100_000_000,
        description="Minimum daily turnover Rs 1 Cr"
    )
    defensive_rs_boost: float = Field(
        default=0.10,
        ge=0.0,
        le=0.30,
        description="Extra RS required in DEFENSIVE regime (+10%)"
    )
    min_52w_high_prox: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Minimum proximity to 52-week high (85%)"
    )
    high_52w_bullish_mult: float = Field(
        default=0.90,
        ge=0.80,
        le=1.0,
        description="52W high multiplier in bullish (85% × 0.90 = 76.5%)"
    )
    high_52w_defensive_mult: float = Field(
        default=1.05,
        ge=1.0,
        le=1.15,
        description="52W high multiplier in defensive (85% × 1.05 = 89.3%)"
    )

    # === Partial Filter Passing ===
    use_partial_filter_passing: bool = Field(
        default=True,
        description="Allow 2 of 3 entry filters to pass in bullish/recovery"
    )
    partial_filter_min_passed: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Minimum filters required (2 of 3)"
    )
    partial_filter_score_penalty: float = Field(
        default=0.04,
        ge=0.0,
        le=0.20,
        description="Score penalty when only partial filters pass (4%)"
    )

    # === Rebalancing ===
    rebalance_days: int = Field(
        default=21,
        ge=7,
        le=63,
        description="Days between rebalances (21 = monthly)"
    )

    # === Sector Momentum Filter (E5) ===
    use_sector_momentum: bool = Field(
        default=True,
        description="Penalize stocks from bottom-performing sectors",
    )
    sector_momentum_lookback: int = Field(
        default=63,
        ge=42,
        le=252,
        description="Lookback days for sector momentum (3 months)",
    )
    sector_exclude_bottom: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of bottom sectors to exclude (bullish default)",
    )
    sector_exclude_bullish: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Bottom sectors to penalize in bullish regime",
    )
    sector_exclude_caution: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Bottom sectors to penalize in caution regime",
    )
    sector_exclude_defensive: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Bottom sectors to penalize in defensive regime",
    )
    sector_momentum_penalty: float = Field(
        default=0.15,
        ge=0.0,
        le=0.50,
        description="Score penalty for stocks in bottom sectors (15%)",
    )

    # === E9: Minimum Hold Period ===
    min_hold_days: int = Field(
        default=3,
        ge=0,
        le=30,
        description="Minimum days before soft exits (trailing/trend/RS). Hard stop always active.",
    )

    # === Recovery Stop Widening (Change 4) ===
    stop_recovery_mult: float = Field(
        default=1.50,
        ge=1.0,
        le=2.0,
        description="Trailing stop multiplier during recovery modes (1.5 = 50% wider)",
    )

    @field_validator("vix_defensive_threshold")
    @classmethod
    def defensive_greater_than_bullish(cls, v: float, info) -> float:
        bullish = info.data.get("vix_bullish_threshold", 18.0)
        if v <= bullish:
            raise ValueError("vix_defensive_threshold must be > vix_bullish_threshold")
        return v


class StrategyOverrides(BaseModel):
    """Per-profile strategy parameter overrides. None = use global defaults."""

    target_portfolio_vol: Optional[float] = Field(default=None, ge=0.10, le=0.30)
    stop_loss_multiplier: Optional[float] = Field(default=None, ge=0.5, le=2.0)
    rs_exit_threshold: Optional[float] = Field(default=None, ge=0.7, le=1.2)
    min_52w_high_prox: Optional[float] = Field(default=None, ge=0.5, le=1.0)
    min_hold_days: Optional[int] = Field(default=None, ge=0, le=30)
    trend_break_buffer: Optional[float] = Field(default=None, ge=0.0, le=0.10)
    min_days_between: Optional[int] = Field(default=None, ge=1, le=30)
    sector_exclude_bullish: Optional[int] = Field(default=None, ge=0, le=5)
    sector_exclude_caution: Optional[int] = Field(default=None, ge=0, le=5)
    sector_exclude_defensive: Optional[int] = Field(default=None, ge=0, le=5)


class ProfileConfig(BaseModel):
    """Per-portfolio profile overrides."""

    name: str = Field(default="primary", description="Human-readable profile name")
    universe_filter: List[str] = Field(
        default_factory=lambda: ["NIFTY100", "MIDCAP100"],
        description="Universe keys to include (e.g. ['NIFTY100', 'MIDCAP100'])",
    )
    initial_capital: float = Field(default=2000000, gt=0, description="Initial capital for this profile")
    target_positions: int = Field(default=12, ge=5, le=30)
    min_positions: int = Field(default=10, ge=5, le=20)
    max_positions: int = Field(default=15, ge=10, le=50)
    max_single_position: float = Field(default=0.10, gt=0, le=0.20)
    max_gold_allocation: Optional[float] = Field(
        default=None, ge=0.0, le=0.35,
        description="Per-profile gold cap. None=use global. 0.0=no gold hedge.",
    )
    strategy_overrides: Optional[StrategyOverrides] = Field(
        default=None,
        description="Per-profile strategy parameter overrides",
    )
    state_file: str = Field(default="strategy_state.json", description="State file name (inside .cache/)")


class Config(BaseModel):
    """Main configuration model for FORTRESS MOMENTUM."""

    zerodha: ZerodhaConfig = Field(default_factory=ZerodhaConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    pure_momentum: PureMomentumConfig = Field(default_factory=PureMomentumConfig)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    rebalancing: RebalancingConfig = Field(default_factory=RebalancingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)

    # Dynamic rebalancing and adaptive lookback configs (NEW)
    dynamic_rebalance: DynamicRebalanceConfig = Field(
        default_factory=DynamicRebalanceConfig
    )
    adaptive_lookback: AdaptiveLookbackConfig = Field(
        default_factory=AdaptiveLookbackConfig
    )

    # Strategy selection
    active_strategy: str = Field(
        default="dual_momentum",
        description="Active strategy: 'dual_momentum'"
    )

    # Strategy-specific configs
    strategy_dual_momentum: AdaptiveDualMomentumConfig = Field(
        default_factory=AdaptiveDualMomentumConfig
    )

    # Excluded symbols (ETFs, liquid funds) - Note: GOLDBEES/LIQUIDCASE
    # are excluded from momentum ranking but used for defensive allocation
    excluded_symbols: List[str] = Field(
        default_factory=lambda: [
            "LIQUIDCASE", "LIQUIDBEES", "LIQUIDETF",
            "NIFTYBEES", "JUNIORBEES", "MID150BEES",
            "HDFCSML250", "GOLDBEES", "HANGSENGBEES",
        ]
    )

    # Portfolio profiles (optional — backward-compatible)
    profiles: Dict[str, ProfileConfig] = Field(default_factory=dict)
    active_profile: str = Field(default="primary", description="Currently selected profile name")

    model_config = {"frozen": True}

    def get_profile(self, name: Optional[str] = None) -> ProfileConfig:
        """Get a profile by name, falling back to defaults from portfolio/position_sizing.

        If no profiles are configured, returns a default profile built from
        existing portfolio and position_sizing config values (backward-compatible).
        """
        profile_name = name or self.active_profile
        if self.profiles and profile_name in self.profiles:
            return self.profiles[profile_name]
        # Backward-compatible: build default profile from existing config sections
        return ProfileConfig(
            name="primary",
            universe_filter=["NIFTY100", "MIDCAP100"],
            initial_capital=self.portfolio.initial_capital,
            target_positions=self.position_sizing.target_positions,
            min_positions=self.position_sizing.min_positions,
            max_positions=self.position_sizing.max_positions,
            max_single_position=self.position_sizing.max_single_position,
            state_file="strategy_state.json",
        )

    def get_profile_names(self) -> List[str]:
        """Get list of available profile names."""
        if self.profiles:
            return list(self.profiles.keys())
        return ["primary"]

    def with_profile_overrides(self, profile_name: Optional[str] = None) -> "Config":
        """Return a Config copy with profile strategy overrides applied.

        For profiles without overrides (e.g. primary), returns self (identity).
        """
        profile = self.get_profile(profile_name)
        overrides = profile.strategy_overrides
        if overrides is None:
            return self

        updates: Dict = {}
        sdm_updates: Dict = {}
        pm_updates: Dict = {}
        regime_updates: Dict = {}
        dyn_updates: Dict = {}

        if overrides.target_portfolio_vol is not None:
            regime_updates["target_portfolio_vol"] = overrides.target_portfolio_vol
        if overrides.stop_loss_multiplier is not None:
            m = overrides.stop_loss_multiplier
            sdm = self.strategy_dual_momentum
            sdm_updates.update(
                tier1_trailing=round(sdm.tier1_trailing * m, 4),
                tier2_trailing=round(sdm.tier2_trailing * m, 4),
                tier3_trailing=round(sdm.tier3_trailing * m, 4),
                tier4_trailing=round(sdm.tier4_trailing * m, 4),
                hard_stop=round(sdm.hard_stop * m, 4),
            )
        for attr in ("rs_exit_threshold", "min_52w_high_prox", "min_hold_days",
                     "trend_break_buffer", "sector_exclude_bullish",
                     "sector_exclude_caution", "sector_exclude_defensive"):
            val = getattr(overrides, attr, None)
            if val is not None:
                sdm_updates[attr] = val
        # min_52w_high_prox also lives in pure_momentum (used by backtest config)
        if overrides.min_52w_high_prox is not None:
            pm_updates["min_52w_high_prox"] = overrides.min_52w_high_prox
        if overrides.min_days_between is not None:
            dyn_updates["min_days_between"] = overrides.min_days_between

        if sdm_updates:
            updates["strategy_dual_momentum"] = self.strategy_dual_momentum.model_copy(
                update=sdm_updates
            )
        if pm_updates:
            updates["pure_momentum"] = self.pure_momentum.model_copy(update=pm_updates)
        if regime_updates:
            updates["regime"] = self.regime.model_copy(update=regime_updates)
        if dyn_updates:
            updates["dynamic_rebalance"] = self.dynamic_rebalance.model_copy(
                update=dyn_updates
            )

        return self.model_copy(update=updates) if updates else self


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file with environment variable override support.

    Environment variables take precedence over config file values:
    - ZERODHA_API_KEY: Override zerodha.api_key
    - ZERODHA_API_SECRET: Override zerodha.api_secret
    """
    import os

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Override with environment variables if present
    if "ZERODHA_API_KEY" in os.environ:
        if "zerodha" not in data:
            data["zerodha"] = {}
        data["zerodha"]["api_key"] = os.environ["ZERODHA_API_KEY"]
    if "ZERODHA_API_SECRET" in os.environ:
        if "zerodha" not in data:
            data["zerodha"] = {}
        data["zerodha"]["api_secret"] = os.environ["ZERODHA_API_SECRET"]

    # Parse profiles section into ProfileConfig objects
    if "profiles" in data and isinstance(data["profiles"], dict):
        parsed_profiles = {}
        for pname, pdata in data["profiles"].items():
            if isinstance(pdata, dict):
                parsed_profiles[pname] = ProfileConfig(name=pname, **pdata)
        data["profiles"] = parsed_profiles

    return Config(**data)


def get_default_config() -> Config:
    """Get configuration with all defaults."""
    return Config()
