"""
Technical indicators for FORTRESS MOMENTUM.

Pure momentum strategy using Normalized Momentum Score (NMS).
Market regime detection for defensive allocation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import RegimeConfig


class MarketRegime(Enum):
    """
    Market regime classification for defensive allocation.

    Based on Nifty 52-week range position with VIX and return modifiers.
    """

    BULLISH = "bullish"      # > 70% of 52W range, full equity
    NORMAL = "normal"        # 50-70% of 52W range, full equity
    CAUTION = "caution"      # 30-50% of 52W range, or VIX > 20, reduced equity
    DEFENSIVE = "defensive"  # < 30% of 52W range, or VIX > 25, heavy defensive


@dataclass
class RegimeResult:
    """
    Result of market regime detection.

    Contains the detected regime and all contributing signals,
    plus the target allocation weights.

    Enhanced to support:
    - Multi-timeframe position tracking
    - Hysteresis for regime transitions
    - Stress score for graduated allocation
    """

    regime: MarketRegime
    nifty_52w_position: float      # 0-1, composite weighted position
    vix_level: float               # Current VIX value
    nifty_3m_return: float         # 3-month return
    equity_weight: float           # Target equity allocation (0-1)
    gold_weight: float             # Target gold allocation (0-1)
    cash_weight: float             # Target cash allocation (0-1)

    # Signal details for debugging
    primary_regime: MarketRegime   # Regime from position alone
    vix_upgrade: bool              # Was regime upgraded due to VIX?
    return_upgrade: bool           # Was regime upgraded due to returns?

    # Multi-timeframe position tracking (NEW)
    position_short: float = 0.0    # 21-day position (fast signal)
    position_medium: float = 0.0   # 63-day position (intermediate)
    position_long: float = 0.0     # 126-day position (trend)
    return_1m: float = 0.0         # 1-month return (faster warning)

    # Hysteresis tracking (NEW)
    pending_regime: Optional[MarketRegime] = None  # Regime awaiting confirmation
    confirmation_days: int = 0     # Days signal has been consistent
    transition_blocked: bool = False  # Was transition blocked by hysteresis?

    # Stress score for graduated allocation (NEW)
    stress_score: float = 0.0      # 0-1 composite stress indicator

    # Agile regime detection signals (NEW)
    position_momentum: float = 0.0      # Rate of change in position
    return_10d: float = 0.0             # 10-day return for faster detection
    vix_recovering: bool = False        # Is VIX mean-reverting from spike?
    vix_recovery_strength: float = 0.0  # How strong is VIX recovery (0-1)?
    momentum_recovery_bonus: float = 0.0  # Applied momentum bonus to thresholds
    vix_recovery_bonus: float = 0.0       # Applied VIX recovery bonus to thresholds

    def __str__(self) -> str:
        modifiers = []
        if self.vix_upgrade:
            modifiers.append(f"VIX={self.vix_level:.1f}")
        if self.return_upgrade:
            modifiers.append(f"3M={self.nifty_3m_return:.1%}")
        if self.transition_blocked:
            modifiers.append(f"pending={self.pending_regime.value if self.pending_regime else 'none'}")

        modifier_str = f" [{', '.join(modifiers)}]" if modifiers else ""
        return (
            f"{self.regime.value.upper()}{modifier_str}: "
            f"Equity={self.equity_weight:.0%}, "
            f"Gold={self.gold_weight:.0%}, "
            f"Cash={self.cash_weight:.0%} "
            f"(stress={self.stress_score:.0%})"
        )


@dataclass
class NMSResult:
    """
    Result of Normalized Momentum Score calculation.

    Based on Nifty 500 Momentum 50 methodology:
    - Volatility-adjusted returns over 6M and 12M periods
    - Skips most recent 5 days to avoid short-term reversal
    """

    nms: float                    # Composite normalized momentum score
    return_6m: float              # 6-month simple return
    return_12m: float             # 12-month simple return
    volatility_6m: float          # 6-month annualized volatility
    adj_return_6m: float          # Volatility-adjusted 6M return
    adj_return_12m: float         # Volatility-adjusted 12M return
    high_52w_proximity: float     # Current price / 52-week high (0-1)
    above_50ema: bool             # Price > 50-day EMA
    above_200sma: bool            # Price > 200-day SMA
    volume_surge: float           # 20-day avg volume / 50-day avg volume
    daily_turnover: float         # Average daily turnover (rupees)

    def passes_entry_filters(
        self,
        min_52w_prox: float = 0.85,
        min_volume_ratio: float = 1.1,
        min_daily_turnover: float = 20_000_000,
    ) -> bool:
        """Check if stock passes all entry filters."""
        return (
            self.high_52w_proximity >= min_52w_prox
            and self.above_50ema
            and self.above_200sma
            and self.volume_surge >= min_volume_ratio
            and self.daily_turnover >= min_daily_turnover
        )


def calculate_drawdown(prices: pd.Series) -> Tuple[float, float]:
    """
    Calculate current and maximum drawdown.

    Args:
        prices: Series of prices or portfolio values

    Returns:
        Tuple of (current_drawdown, max_drawdown) as negative decimals
    """
    if len(prices) == 0:
        return (0.0, 0.0)

    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max

    current_drawdown = drawdown.iloc[-1]
    max_drawdown = drawdown.min()

    return (current_drawdown, max_drawdown)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.06,
    trading_days_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default: 6%)
        trading_days_year: Trading days per year

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    daily_rf = risk_free_rate / trading_days_year
    excess_returns = returns - daily_rf

    annualized_return = excess_returns.mean() * trading_days_year
    annualized_vol = returns.std() * np.sqrt(trading_days_year)

    return annualized_return / annualized_vol if annualized_vol > 0 else 0.0


def calculate_normalized_momentum_score(
    prices: pd.Series,
    volumes: pd.Series,
    lookback_6m: int = 126,
    lookback_12m: int = 252,
    lookback_volatility: int = 126,
    skip_recent_days: int = 5,
    weight_6m: float = 0.50,
    weight_12m: float = 0.50,
    trading_days_year: int = 252,
    min_volatility_floor: float = 0.10,
) -> Optional[NMSResult]:
    """
    Calculate Normalized Momentum Score (NMS) following Nifty 500 Momentum 50 methodology.

    NMS uses volatility-adjusted returns to normalize momentum across stocks with
    different risk profiles. Higher volatility stocks need proportionally higher
    returns to achieve the same score.

    Formula:
        adj_return_6m = return_6m / max(volatility_6m, 0.10)
        adj_return_12m = return_12m / max(volatility_6m, 0.10)
        NMS = weight_6m * adj_6m + weight_12m * adj_12m

    Args:
        prices: Series of closing prices with DateTimeIndex
        volumes: Series of trading volumes with DateTimeIndex
        lookback_6m: Days for 6-month return (default: 126)
        lookback_12m: Days for 12-month return (default: 252)
        lookback_volatility: Days for volatility calculation (default: 126)
        skip_recent_days: Skip most recent N days (avoid reversal, default: 5)
        weight_6m: Weight for 6-month adjusted return (default: 0.50)
        weight_12m: Weight for 12-month adjusted return (default: 0.50)
        trading_days_year: Trading days per year (default: 252)
        min_volatility_floor: Minimum volatility to prevent division issues (default: 0.10)

    Returns:
        NMSResult with all components, or None if insufficient data
    """
    # Verify weights sum to 1.0
    assert abs(weight_6m + weight_12m - 1.0) < 1e-10, "NMS weights must sum to 1.0"

    # Need at least 12M + skip days of data
    min_required = lookback_12m + skip_recent_days
    if len(prices) < min_required:
        return None

    # Skip most recent days (avoids short-term reversal)
    if skip_recent_days > 0:
        prices_calc = prices.iloc[:-skip_recent_days]
        volumes_calc = volumes.iloc[:-skip_recent_days] if len(volumes) > skip_recent_days else volumes
    else:
        prices_calc = prices
        volumes_calc = volumes

    current_price = prices_calc.iloc[-1]

    # Calculate simple returns (not log returns for momentum scoring)
    if len(prices_calc) >= lookback_6m:
        return_6m = (current_price / prices_calc.iloc[-lookback_6m]) - 1
    else:
        return_6m = 0.0

    if len(prices_calc) >= lookback_12m:
        return_12m = (current_price / prices_calc.iloc[-lookback_12m]) - 1
    else:
        return_12m = return_6m  # Fall back to 6M if not enough data

    # Calculate volatility (using log returns, annualized)
    log_returns = np.log(prices_calc / prices_calc.shift(1)).dropna()
    if len(log_returns) >= lookback_volatility:
        volatility_6m = log_returns.iloc[-lookback_volatility:].std() * np.sqrt(trading_days_year)
    else:
        volatility_6m = log_returns.std() * np.sqrt(trading_days_year) if len(log_returns) > 0 else 0.20

    # Calculate separate 12M volatility for 12M adjusted return
    lookback_12m_vol = min(lookback_12m, len(log_returns))
    if lookback_12m_vol >= lookback_12m:
        volatility_12m = log_returns.iloc[-lookback_12m:].std() * np.sqrt(trading_days_year)
    else:
        volatility_12m = volatility_6m  # Fallback to 6M volatility if insufficient data

    # Apply volatility floors
    volatility_6m_adj = max(volatility_6m, min_volatility_floor)
    volatility_12m_adj = max(volatility_12m, min_volatility_floor)

    # Volatility-adjusted returns (use matching volatility period for each return period)
    adj_return_6m = return_6m / volatility_6m_adj
    adj_return_12m = return_12m / volatility_12m_adj

    # Composite NMS
    nms = weight_6m * adj_return_6m + weight_12m * adj_return_12m

    # Calculate 52-week high proximity using full prices (not skip_recent_days)
    # Consistent with 50-EMA and 200-SMA filters which already use full prices
    high_52w = prices.iloc[-252:].max() if len(prices) >= 252 else prices.max()
    high_52w_proximity = prices.iloc[-1] / high_52w if high_52w > 0 else 0.0

    # Calculate 50-day EMA
    ema_50 = prices.ewm(span=50, adjust=False).mean().iloc[-1]
    above_50ema = prices.iloc[-1] > ema_50

    # Calculate 200-day SMA
    if len(prices) >= 200:
        sma_200 = prices.iloc[-200:].mean()
        above_200sma = prices.iloc[-1] > sma_200
    else:
        above_200sma = True  # Assume OK if not enough data

    # Calculate volume surge (20-day vs 50-day average)
    if len(volumes) >= 50:
        avg_vol_20 = volumes.iloc[-20:].mean()
        avg_vol_50 = volumes.iloc[-50:].mean()
        volume_surge = avg_vol_20 / avg_vol_50 if avg_vol_50 > 0 else 1.0
    else:
        volume_surge = 1.0

    # Calculate average daily turnover (price * volume)
    if len(prices) >= 20 and len(volumes) >= 20:
        daily_turnover = (prices.iloc[-20:] * volumes.iloc[-20:]).mean()
    else:
        daily_turnover = 0.0

    return NMSResult(
        nms=nms,
        return_6m=return_6m,
        return_12m=return_12m,
        volatility_6m=volatility_6m,
        adj_return_6m=adj_return_6m,
        adj_return_12m=adj_return_12m,
        high_52w_proximity=high_52w_proximity,
        above_50ema=above_50ema,
        above_200sma=above_200sma,
        volume_surge=volume_surge,
        daily_turnover=daily_turnover,
    )


def calculate_exit_triggers(
    nms_result: NMSResult,
    entry_price: float,
    current_price: float,
    peak_price: float,
    days_held: int,
    nms_percentile: float,
    initial_stop_loss: float = 0.15,
    trailing_stop: float = 0.12,
    trailing_activation: float = 0.08,
    max_days_without_gain: int = 60,
    min_gain_threshold: float = 0.10,
    min_nms_percentile: float = 50.0,
) -> Tuple[bool, str]:
    """
    Check if any exit trigger is activated.

    Exit Triggers:
    1. Momentum decay: NMS falls below 50th percentile
    2. Stop loss: -15% from entry price
    3. Trailing stop: -12% from peak (after +8% gain)
    4. Trend break: Price < 50-day EMA
    5. Time decay: >60 days held without +10% gain

    Args:
        nms_result: Current NMS calculation result
        entry_price: Price at which position was entered
        current_price: Current market price
        peak_price: Highest price since entry
        days_held: Number of days position has been held
        nms_percentile: Current NMS percentile rank (0-100)
        initial_stop_loss: Initial stop loss threshold (default: 15%)
        trailing_stop: Trailing stop threshold (default: 12%)
        trailing_activation: Gain needed to activate trailing stop (default: 8%)
        max_days_without_gain: Max days to hold without target gain (default: 60)
        min_gain_threshold: Target gain for time-based exit (default: 10%)
        min_nms_percentile: Minimum NMS percentile to hold (default: 50)

    Returns:
        Tuple of (should_exit, reason)
    """
    current_gain = (current_price - entry_price) / entry_price
    gain_from_peak = (current_price - peak_price) / peak_price

    # 1. Momentum decay
    if nms_percentile < min_nms_percentile:
        return (True, f"Momentum decay: NMS at {nms_percentile:.0f}th percentile")

    # 2. Initial stop loss
    if current_gain <= -initial_stop_loss:
        return (True, f"Stop loss: {current_gain:.1%} from entry")

    # 3. Trailing stop (only if gain exceeded activation threshold)
    if current_gain >= trailing_activation and gain_from_peak <= -trailing_stop:
        return (True, f"Trailing stop: {gain_from_peak:.1%} from peak")

    # 4. Trend break
    if not nms_result.above_50ema:
        return (True, "Trend break: Price below 50-day EMA")

    # 5. Time decay
    if days_held > max_days_without_gain and current_gain < min_gain_threshold:
        return (True, f"Time decay: {days_held} days held, only {current_gain:.1%} gain")

    return (False, "OK")


def calculate_range_position(prices: pd.Series, lookback: int) -> float:
    """
    Calculate position within high-low range over lookback period.

    Position = (Current - Low) / (High - Low)
    Returns value between 0 and 1.

    Args:
        prices: Series of prices with DateTimeIndex
        lookback: Number of days to look back

    Returns:
        Position as fraction (0 = at low, 1 = at high)
    """
    if len(prices) < lookback:
        lookback = len(prices)

    if lookback < 2:
        return 0.5  # Default to middle if insufficient data

    recent_prices = prices.iloc[-lookback:]
    high = recent_prices.max()
    low = recent_prices.min()
    current = prices.iloc[-1]

    if high == low:
        return 0.5  # Avoid division by zero

    return (current - low) / (high - low)


def calculate_composite_position(
    prices: pd.Series,
    lookback_short: int,
    lookback_medium: int,
    lookback_long: int,
    weight_short: float,
    weight_medium: float,
    weight_long: float,
) -> Tuple[float, float, float, float]:
    """
    Calculate weighted composite position from multiple timeframes.

    This provides faster detection than single 52-week lookback while
    maintaining stability through longer-term confirmation.

    Args:
        prices: Series of prices with DateTimeIndex
        lookback_short: Days for short-term (e.g., 21 = 1 month)
        lookback_medium: Days for medium-term (e.g., 63 = 3 months)
        lookback_long: Days for long-term (e.g., 126 = 6 months)
        weight_short: Weight for short-term signal
        weight_medium: Weight for medium-term signal
        weight_long: Weight for long-term signal

    Returns:
        Tuple of (composite_position, position_short, position_medium, position_long)
    """
    position_short = calculate_range_position(prices, lookback_short)
    position_medium = calculate_range_position(prices, lookback_medium)
    position_long = calculate_range_position(prices, lookback_long)

    composite = (
        weight_short * position_short +
        weight_medium * position_medium +
        weight_long * position_long
    )

    return (composite, position_short, position_medium, position_long)


def calculate_position_momentum(
    prices: pd.Series,
    lookback: int = 21,
    momentum_period: int = 5,
) -> float:
    """
    Calculate rate-of-change in range position to detect trend reversals faster.

    Positive momentum = position improving (bullish signal)
    Negative momentum = position declining (bearish signal)

    Args:
        prices: Series of prices with DateTimeIndex
        lookback: Lookback period for position calculation (default: 21 days)
        momentum_period: Days to measure rate-of-change (default: 5)

    Returns:
        Position momentum (change per day, typically -0.02 to +0.02)
    """
    if len(prices) < lookback + momentum_period:
        return 0.0

    # Current position
    current_position = calculate_range_position(prices, lookback)

    # Position from momentum_period days ago
    prices_prior = prices.iloc[:-momentum_period]
    if len(prices_prior) < lookback:
        return 0.0

    previous_position = calculate_range_position(prices_prior, lookback)

    # Rate of change per day
    return (current_position - previous_position) / momentum_period


def calculate_vix_recovery_signal(
    vix_series: pd.Series,
    spike_threshold: float = 25.0,
    decline_rate: float = 0.10,
) -> Tuple[bool, float]:
    """
    Detect VIX spike-and-decline pattern indicating mean-reversion.

    VIX is mean-reverting. When VIX spikes then rapidly declines,
    this signals reduced fear and potential market recovery.

    Args:
        vix_series: Series of VIX values with DateTimeIndex
        spike_threshold: VIX level considered a spike (default: 25.0)
        decline_rate: Minimum decline from peak to trigger signal (default: 10%)

    Returns:
        Tuple of (is_recovering: bool, recovery_strength: 0-1)
    """
    if len(vix_series) < 10:
        return (False, 0.0)

    # Find recent peak in last 10 days
    recent_peak = vix_series.iloc[-10:].max()
    current_vix = vix_series.iloc[-1]

    # Check if we had a spike
    had_spike = recent_peak >= spike_threshold

    # Check if VIX is declining from peak
    is_declining = current_vix < recent_peak * (1 - decline_rate)

    if had_spike and is_declining:
        # Recovery strength: how much has VIX declined from peak
        # Normalize to 0-1 where 20% decline = 1.0
        decline_pct = (recent_peak - current_vix) / recent_peak
        recovery_strength = min(decline_pct / 0.20, 1.0)
        return (True, float(recovery_strength))

    return (False, 0.0)


def calculate_rs_trend(
    stock_prices: pd.Series,
    benchmark_prices: pd.Series,
    lookback: int = 10,
) -> Tuple[float, float]:
    """
    Calculate RS trend (is relative strength improving?).

    This helps identify "execution discipline" stocks - those consistently
    outperforming even in sideways markets.

    Args:
        stock_prices: Series of stock prices with DateTimeIndex
        benchmark_prices: Series of benchmark prices (e.g., NIFTY 50)
        lookback: Days to measure RS trend (default: 10)

    Returns:
        Tuple of (current_rs, rs_change) where:
        - current_rs: Current relative strength value
        - rs_change: Change in RS over lookback period (positive = improving)
    """
    # Use date-based alignment for accurate RS calculation
    common_dates = stock_prices.index.intersection(benchmark_prices.index)
    if len(common_dates) < lookback + 21:
        return (1.0, 0.0)

    try:
        # Get aligned prices on common dates
        stock_aligned = stock_prices.loc[common_dates]
        bench_aligned = benchmark_prices.loc[common_dates]

        # Calculate current RS (21-day)
        stock_return_now = (stock_aligned.iloc[-1] / stock_aligned.iloc[-21]) - 1
        bench_return_now = (bench_aligned.iloc[-1] / bench_aligned.iloc[-21]) - 1

        if abs(bench_return_now) < 0.0001:
            current_rs = 1.0
        else:
            current_rs = (1 + stock_return_now) / (1 + bench_return_now)

        # Calculate RS from lookback days ago
        stock_return_then = (stock_aligned.iloc[-lookback-1] / stock_aligned.iloc[-lookback-21]) - 1
        bench_return_then = (bench_aligned.iloc[-lookback-1] / bench_aligned.iloc[-lookback-21]) - 1

        if abs(bench_return_then) < 0.0001:
            previous_rs = 1.0
        else:
            previous_rs = (1 + stock_return_then) / (1 + bench_return_then)

        # RS change (positive = improving)
        rs_change = current_rs - previous_rs

        return (float(current_rs), float(rs_change))
    except (IndexError, ZeroDivisionError, KeyError):
        return (1.0, 0.0)


def calculate_stress_score(
    composite_position: float,
    vix_level: float,
    return_1m: float,
    return_3m: float,
    config: "RegimeConfig",
    return_10d: float = 0.0,
    position_momentum: float = 0.0,
) -> float:
    """
    Calculate composite stress score for graduated allocation.

    Stress score ranges from 0 (calm market) to 1 (extreme stress).
    Uses weighted combination of position, VIX, and returns.

    Enhanced with:
    - 10-day return for faster early detection
    - Position momentum for recovery detection

    Args:
        composite_position: Multi-timeframe weighted position (0-1)
        vix_level: Current VIX value
        return_1m: 1-month return
        return_3m: 3-month return
        config: RegimeConfig with thresholds
        return_10d: 10-day return for faster detection (optional)
        position_momentum: Rate of change in position (optional)

    Returns:
        Stress score between 0 and 1
    """
    # Position stress: lower position = higher stress
    # Map position from [0, 1] to stress [1, 0], with curve
    position_stress = 1.0 - composite_position

    # VIX stress: normalize VIX to 0-1 range
    # VIX 14 (calm) -> 0, VIX 30+ -> 1
    vix_min = config.vix_calm
    vix_max = 35.0  # Extreme fear level
    vix_stress = np.clip((vix_level - vix_min) / (vix_max - vix_min), 0, 1)

    # Return stress: negative returns = higher stress
    # Include 10-day return if enabled
    if config.use_return_10d and return_10d != 0.0:
        # Weighted combination of returns (10d gets partial weight)
        base_weight = 1.0 - config.return_10d_weight
        worst_base_return = min(return_1m, return_3m)
        worst_return = base_weight * worst_base_return + config.return_10d_weight * return_10d
    else:
        worst_return = min(return_1m, return_3m)

    # Map returns from [+10%, -15%] to stress [0, 1]
    return_stress = np.clip((0.10 - worst_return) / 0.25, 0, 1)

    # Momentum adjustment: positive momentum reduces stress
    # Strong positive momentum (> 0.01/day) can reduce stress by up to 10%
    momentum_adjustment = 0.0
    if config.use_position_momentum and position_momentum > 0.005:
        momentum_adjustment = min(position_momentum * 5, 0.10)  # Max 10% reduction

    # Weighted combination (E10: configurable weights, default 40/30/30)
    w_pos = getattr(config, "stress_weight_position", 0.40)
    w_vix = getattr(config, "stress_weight_vix", 0.30)
    w_ret = getattr(config, "stress_weight_returns", 0.30)
    stress = (
        w_pos * position_stress +
        w_vix * vix_stress +
        w_ret * return_stress
    )

    # Apply momentum adjustment
    stress = max(0, stress - momentum_adjustment)

    return float(np.clip(stress, 0, 1))


def calculate_graduated_allocation(
    stress_score: float,
    config: "RegimeConfig",
) -> Tuple[float, float, float]:
    """
    Calculate smooth allocation based on stress score.

    Uses a smooth curve to transition between full equity and defensive
    allocation, avoiding cliff effects from fixed thresholds.

    Formula:
        equity = 1.0 - stress × (1.0 - min_equity)
        Remaining split between gold and cash

    Args:
        stress_score: Composite stress indicator (0-1)
        config: RegimeConfig with allocation limits

    Returns:
        Tuple of (equity_weight, gold_weight, cash_weight)
    """
    # Apply curve steepness to stress score
    # Higher steepness = more aggressive response at extremes
    steepness = config.allocation_curve_steepness
    curved_stress = np.power(stress_score, 1.0 / steepness)

    # Calculate equity allocation
    # At stress=0: equity=100%, at stress=1: equity=min_equity
    max_reduction = 1.0 - config.min_equity_allocation
    equity_weight = 1.0 - (curved_stress * max_reduction)

    # Defensive allocation goes to gold only (cash = 0, user manages manually)
    # Any defensive allocation beyond max_gold stays as equity
    defensive_allocation = 1.0 - equity_weight
    gold_weight = min(defensive_allocation, config.max_gold_allocation)
    cash_weight = 0.0  # No cash allocation - user manages cash manually

    # If defensive_allocation > max_gold, add remainder back to equity
    # (don't normalize - that would push gold above its cap)
    equity_weight = 1.0 - gold_weight - cash_weight

    return (equity_weight, gold_weight, cash_weight)


def _determine_regime_from_signals(
    composite_position: float,
    vix_level: float,
    return_3m: float,
    return_1m: float,
    config: "RegimeConfig",
) -> Tuple[MarketRegime, MarketRegime, bool, bool]:
    """
    Determine regime from current signals (without hysteresis).

    Returns the signal regime (what signals currently indicate)
    and the primary regime (from position alone).

    Args:
        composite_position: Multi-timeframe weighted position
        vix_level: Current VIX value
        return_3m: 3-month return
        return_1m: 1-month return
        config: RegimeConfig with thresholds

    Returns:
        Tuple of (signal_regime, primary_regime, vix_upgrade, return_upgrade)
    """
    # Step 1: Determine primary regime from composite position
    if composite_position >= config.bullish_threshold:
        primary_regime = MarketRegime.BULLISH
    elif composite_position >= config.caution_threshold:
        primary_regime = MarketRegime.NORMAL
    elif composite_position >= config.defensive_threshold:
        primary_regime = MarketRegime.CAUTION
    else:
        primary_regime = MarketRegime.DEFENSIVE

    # Step 2: Apply secondary signal upgrades (can only make more defensive)
    signal_regime = primary_regime
    vix_upgrade = False
    return_upgrade = False

    # VIX can upgrade regime severity
    if vix_level >= config.vix_defensive:
        if signal_regime != MarketRegime.DEFENSIVE:
            signal_regime = MarketRegime.DEFENSIVE
            vix_upgrade = True
    elif vix_level >= config.vix_caution:
        if signal_regime in (MarketRegime.BULLISH, MarketRegime.NORMAL):
            signal_regime = MarketRegime.CAUTION
            vix_upgrade = True

    # Return signals can upgrade regime severity
    # Use 3M return for DEFENSIVE, 1M for early CAUTION warning
    if return_3m <= config.return_defensive:
        if signal_regime != MarketRegime.DEFENSIVE:
            signal_regime = MarketRegime.DEFENSIVE
            return_upgrade = True
    elif return_3m <= config.return_caution or return_1m <= config.return_warning:
        if signal_regime in (MarketRegime.BULLISH, MarketRegime.NORMAL):
            signal_regime = MarketRegime.CAUTION
            return_upgrade = True

    return (signal_regime, primary_regime, vix_upgrade, return_upgrade)


def _check_recovery_conditions(
    current_regime: MarketRegime,
    composite_position: float,
    vix_level: float,
    return_3m: float,
    config: "RegimeConfig",
    position_momentum: float = 0.0,
    vix_recovering: bool = False,
    vix_recovery_strength: float = 0.0,
) -> Tuple[Optional[MarketRegime], float, float]:
    """
    Check if conditions allow recovery to a less defensive regime.

    Recovery requires stronger signals than entry (asymmetric thresholds)
    and supportive VIX/return conditions.

    Enhanced with:
    - Position momentum bonus: reduces thresholds when position improving
    - VIX recovery bonus: reduces thresholds during VIX mean-reversion

    Args:
        current_regime: Current market regime
        composite_position: Multi-timeframe weighted position
        vix_level: Current VIX value
        return_3m: 3-month return
        config: RegimeConfig with recovery thresholds
        position_momentum: Rate of change in position (optional)
        vix_recovering: Is VIX mean-reverting from spike?
        vix_recovery_strength: Strength of VIX recovery (0-1)

    Returns:
        Tuple of (target_regime, momentum_bonus_applied, vix_bonus_applied)
    """
    # Calculate recovery threshold bonuses
    momentum_bonus = 0.0
    vix_bonus = 0.0

    # Position momentum bonus: when momentum > 0.005/day, reduce threshold
    if config.use_position_momentum and position_momentum > 0.005:
        momentum_bonus = config.position_momentum_recovery_bonus

    # VIX recovery bonus: when VIX is mean-reverting from spike
    if config.use_vix_recovery_accelerator and vix_recovering:
        vix_bonus = config.vix_recovery_bonus * vix_recovery_strength

    total_bonus = momentum_bonus + vix_bonus

    # E7: 2-of-3 recovery gate helper
    require_all = getattr(config, "recovery_require_all_conditions", True)

    def _check_recovery(position_ok: bool, vix_ok: bool, return_ok: bool) -> bool:
        """Check recovery conditions: all 3 or 2-of-3 based on config."""
        if require_all:
            return position_ok and vix_ok and return_ok
        # 2-of-3 gate: any 2 conditions sufficient
        return sum([position_ok, vix_ok, return_ok]) >= 2

    # Recovery from DEFENSIVE to CAUTION
    if current_regime == MarketRegime.DEFENSIVE:
        adjusted_threshold = config.caution_recovery_threshold - total_bonus
        if _check_recovery(
            composite_position >= adjusted_threshold,
            vix_level <= config.vix_caution,
            return_3m >= config.return_caution,
        ):
            return (MarketRegime.CAUTION, momentum_bonus, vix_bonus)

    # Recovery from CAUTION to NORMAL
    elif current_regime == MarketRegime.CAUTION:
        adjusted_threshold = config.normal_recovery_threshold - total_bonus
        if _check_recovery(
            composite_position >= adjusted_threshold,
            vix_level <= config.vix_normal,
            return_3m >= config.return_recovery_normal,
        ):
            return (MarketRegime.NORMAL, momentum_bonus, vix_bonus)

    # Recovery from NORMAL to BULLISH
    elif current_regime == MarketRegime.NORMAL:
        adjusted_threshold = config.bullish_recovery_threshold - total_bonus
        if _check_recovery(
            composite_position >= adjusted_threshold,
            vix_level <= config.vix_calm,
            return_3m >= config.return_recovery_bullish,
        ):
            return (MarketRegime.BULLISH, momentum_bonus, vix_bonus)

    return (None, momentum_bonus, vix_bonus)


def _regime_severity(regime: MarketRegime) -> int:
    """Get numeric severity for regime comparison."""
    severity_map = {
        MarketRegime.BULLISH: 0,
        MarketRegime.NORMAL: 1,
        MarketRegime.CAUTION: 2,
        MarketRegime.DEFENSIVE: 3,
    }
    return severity_map.get(regime, 1)


def evaluate_regime_transition(
    current_regime: MarketRegime,
    signal_regime: MarketRegime,
    composite_position: float,
    vix_level: float,
    return_3m: float,
    previous_pending: Optional[MarketRegime],
    previous_confirmation_days: int,
    config: "RegimeConfig",
    position_momentum: float = 0.0,
    vix_recovering: bool = False,
    vix_recovery_strength: float = 0.0,
) -> Tuple[MarketRegime, Optional[MarketRegime], int, bool, float, float]:
    """
    Apply hysteresis to regime transitions.

    Upgrades (more defensive) require upgrade_confirmation_days.
    Downgrades (less defensive) require downgrade_confirmation_days.

    Enhanced with:
    - Adaptive hysteresis: strong signals reduce confirmation by 1 day
    - Position momentum bonus for recovery
    - VIX recovery accelerator for recovery

    Args:
        current_regime: Current active regime
        signal_regime: Regime indicated by current signals
        composite_position: Multi-timeframe position (for recovery check)
        vix_level: Current VIX (for recovery check)
        return_3m: 3-month return (for recovery check)
        previous_pending: Previously pending regime (if any)
        previous_confirmation_days: Days the previous pending was consistent
        config: RegimeConfig with confirmation periods
        position_momentum: Rate of change in position
        vix_recovering: Is VIX mean-reverting from spike?
        vix_recovery_strength: Strength of VIX recovery (0-1)

    Returns:
        Tuple of (final_regime, pending_regime, confirmation_days, transition_blocked,
                  momentum_bonus, vix_bonus)
    """
    current_severity = _regime_severity(current_regime)
    signal_severity = _regime_severity(signal_regime)

    # Check for recovery conditions (downgrade path) with bonuses
    recovery_result = _check_recovery_conditions(
        current_regime=current_regime,
        composite_position=composite_position,
        vix_level=vix_level,
        return_3m=return_3m,
        config=config,
        position_momentum=position_momentum,
        vix_recovering=vix_recovering,
        vix_recovery_strength=vix_recovery_strength,
    )
    recovery_target, momentum_bonus, vix_bonus = recovery_result

    # Determine target regime and confirmation requirement
    if signal_severity > current_severity:
        # Upgrade (more defensive) - shorter confirmation
        target_regime = signal_regime
        required_days = config.upgrade_confirmation_days
        is_recovery = False
    elif recovery_target is not None:
        # Recovery (less defensive) - longer confirmation
        target_regime = recovery_target
        required_days = config.downgrade_confirmation_days
        is_recovery = True
    else:
        # No change needed
        return (current_regime, None, 0, False, momentum_bonus, vix_bonus)

    # Adaptive hysteresis: strong signals reduce required days by 1
    if config.adaptive_hysteresis:
        if is_recovery:
            # For recovery, check if position exceeds threshold + bonus
            threshold_map = {
                MarketRegime.CAUTION: config.caution_recovery_threshold,
                MarketRegime.NORMAL: config.normal_recovery_threshold,
                MarketRegime.BULLISH: config.bullish_recovery_threshold,
            }
            threshold = threshold_map.get(target_regime, 0.5)
            if composite_position >= threshold + config.strong_signal_bonus:
                required_days = max(1, required_days - 1)
        else:
            # For upgrades (more defensive), check if position is well below threshold
            threshold_map = {
                MarketRegime.CAUTION: config.caution_threshold,
                MarketRegime.DEFENSIVE: config.defensive_threshold,
            }
            threshold = threshold_map.get(target_regime, 0.5)
            if composite_position <= threshold - config.strong_signal_bonus:
                required_days = max(1, required_days - 1)

    # Check if this is the same pending regime as before
    if previous_pending == target_regime:
        # Continuing to confirm the same transition
        new_confirmation_days = previous_confirmation_days + 1
        if new_confirmation_days >= required_days:
            # Transition confirmed!
            return (target_regime, None, 0, False, momentum_bonus, vix_bonus)
        else:
            # Still pending
            return (current_regime, target_regime, new_confirmation_days, True, momentum_bonus, vix_bonus)
    else:
        # New pending regime - start counting
        return (current_regime, target_regime, 1, True, momentum_bonus, vix_bonus)


def detect_market_regime(
    nifty_prices: pd.Series,
    vix_value: float,
    config: "RegimeConfig",
    previous_result: Optional[RegimeResult] = None,
    vix_history: Optional[pd.Series] = None,
) -> RegimeResult:
    """
    Detect market regime using enhanced multi-signal approach.

    Enhancements over original:
    1. Multi-timeframe position (30/35/35 weighted composite - more responsive)
    2. Bidirectional transitions with adaptive hysteresis
    3. Recovery detection with reduced asymmetric thresholds
    4. Graduated allocation based on stress score
    5. Position momentum for faster recovery detection (NEW)
    6. VIX mean-reversion accelerator (NEW)
    7. 10-day return for faster early detection (NEW)

    Primary Signal: Multi-timeframe composite position
        Position = 0.30 × Position_21d + 0.35 × Position_63d + 0.35 × Position_126d

        BULLISH:    Position >= 65%  -> Low stress allocation
        NORMAL:     Position 45-65%  -> Low stress allocation
        CAUTION:    Position 25-45%  -> Medium stress allocation
        DEFENSIVE:  Position < 25%   -> High stress allocation

    Secondary Signals (can upgrade severity):
        - VIX > 22: Upgrade to CAUTION
        - VIX > 28: Force DEFENSIVE
        - 1M return < -3%: Early warning
        - 3M return < -5%: Upgrade to CAUTION
        - 3M return < -10%: Force DEFENSIVE

    Agile Recovery Signals (NEW):
        - Position momentum > 0.005/day: Reduce recovery thresholds
        - VIX spike then decline: Accelerate recovery

    Hysteresis:
        - Upgrades (more defensive): 3 days confirmation
        - Downgrades (less defensive): 4 days confirmation (reduced from 5)
        - Adaptive: Strong signals reduce by 1 day

    Args:
        nifty_prices: Series of Nifty 50 prices with DatetimeIndex
        vix_value: Current India VIX value
        config: RegimeConfig with thresholds and allocations
        previous_result: Previous RegimeResult for hysteresis tracking
        vix_history: Optional VIX history for recovery detection (last 10+ days)

    Returns:
        RegimeResult with detected regime and allocation weights
    """
    # Calculate multi-timeframe composite position
    composite_position, position_short, position_medium, position_long = calculate_composite_position(
        prices=nifty_prices,
        lookback_short=config.lookback_short,
        lookback_medium=config.lookback_medium,
        lookback_long=config.lookback_long,
        weight_short=config.weight_short,
        weight_medium=config.weight_medium,
        weight_long=config.weight_long,
    )

    # Calculate position momentum (NEW - rate of change for faster recovery)
    position_momentum = 0.0
    if config.use_position_momentum:
        position_momentum = calculate_position_momentum(
            prices=nifty_prices,
            lookback=config.lookback_short,
            momentum_period=config.position_momentum_period,
        )

    # Calculate 10-day return (NEW - faster early detection)
    return_10d = 0.0
    if config.use_return_10d and len(nifty_prices) >= 10:
        price_10d_ago = nifty_prices.iloc[-10]
        return_10d = (nifty_prices.iloc[-1] - price_10d_ago) / price_10d_ago

    # Calculate 1-month return (faster warning signal)
    if len(nifty_prices) >= 21:
        price_1m_ago = nifty_prices.iloc[-21]
        return_1m = (nifty_prices.iloc[-1] - price_1m_ago) / price_1m_ago
    else:
        return_1m = 0.0

    # Calculate 3-month return
    if len(nifty_prices) >= 63:
        price_3m_ago = nifty_prices.iloc[-63]
        return_3m = (nifty_prices.iloc[-1] - price_3m_ago) / price_3m_ago
    else:
        return_3m = 0.0

    # Calculate VIX recovery signal (NEW - detect mean-reversion)
    vix_recovering = False
    vix_recovery_strength = 0.0
    if config.use_vix_recovery_accelerator and vix_history is not None:
        vix_recovering, vix_recovery_strength = calculate_vix_recovery_signal(
            vix_series=vix_history,
            spike_threshold=config.vix_recovery_spike_threshold,
            decline_rate=config.vix_recovery_decline_rate,
        )

    # Determine signal regime from current signals (without hysteresis)
    signal_regime, primary_regime, vix_upgrade, return_upgrade = _determine_regime_from_signals(
        composite_position=composite_position,
        vix_level=vix_value,
        return_3m=return_3m,
        return_1m=return_1m,
        config=config,
    )

    # Apply hysteresis if we have previous result
    momentum_bonus = 0.0
    vix_bonus = 0.0
    if previous_result is not None:
        current_regime = previous_result.regime
        previous_pending = previous_result.pending_regime
        previous_confirmation_days = previous_result.confirmation_days

        transition_result = evaluate_regime_transition(
            current_regime=current_regime,
            signal_regime=signal_regime,
            composite_position=composite_position,
            vix_level=vix_value,
            return_3m=return_3m,
            previous_pending=previous_pending,
            previous_confirmation_days=previous_confirmation_days,
            config=config,
            position_momentum=position_momentum,
            vix_recovering=vix_recovering,
            vix_recovery_strength=vix_recovery_strength,
        )
        final_regime, pending_regime, confirmation_days, transition_blocked, momentum_bonus, vix_bonus = transition_result
    else:
        # No previous result - use signal regime directly
        final_regime = signal_regime
        pending_regime = None
        confirmation_days = 0
        transition_blocked = False

    # Calculate stress score with new signals
    stress_score = calculate_stress_score(
        composite_position=composite_position,
        vix_level=vix_value,
        return_1m=return_1m,
        return_3m=return_3m,
        config=config,
        return_10d=return_10d,
        position_momentum=position_momentum,
    )

    # Determine allocation weights
    if config.use_graduated_allocation:
        # Graduated allocation based on stress score
        equity_weight, gold_weight, cash_weight = calculate_graduated_allocation(
            stress_score=stress_score,
            config=config,
        )
    else:
        # Fixed allocation based on regime (original behavior)
        if final_regime == MarketRegime.BULLISH:
            equity_weight = 1.0
            gold_weight = 0.0
            cash_weight = 0.0
        elif final_regime == MarketRegime.NORMAL:
            equity_weight = 1.0
            gold_weight = 0.0
            cash_weight = 0.0
        elif final_regime == MarketRegime.CAUTION:
            equity_weight = config.caution_equity  # 0.90
            gold_weight = config.caution_gold      # 0.10
            cash_weight = 0.0                      # No cash - user manages manually
        else:  # DEFENSIVE
            equity_weight = config.defensive_equity  # 0.80
            gold_weight = config.defensive_gold      # 0.20
            cash_weight = 0.0                        # No cash - user manages manually

    return RegimeResult(
        regime=final_regime,
        nifty_52w_position=composite_position,
        vix_level=vix_value,
        nifty_3m_return=return_3m,
        equity_weight=equity_weight,
        gold_weight=gold_weight,
        cash_weight=cash_weight,
        primary_regime=primary_regime,
        vix_upgrade=vix_upgrade,
        return_upgrade=return_upgrade,
        # Multi-timeframe tracking
        position_short=position_short,
        position_medium=position_medium,
        position_long=position_long,
        return_1m=return_1m,
        # Hysteresis tracking
        pending_regime=pending_regime,
        confirmation_days=confirmation_days,
        transition_blocked=transition_blocked,
        # Stress score
        stress_score=stress_score,
        # Agile regime signals (NEW)
        position_momentum=position_momentum,
        return_10d=return_10d,
        vix_recovering=vix_recovering,
        vix_recovery_strength=vix_recovery_strength,
        momentum_recovery_bonus=momentum_bonus,
        vix_recovery_bonus=vix_bonus,
    )


# =============================================================================
# Simple Regime Detection for SimpleStrategy
# =============================================================================


class SimpleRegime(Enum):
    """
    Simple 3-state market regime for SimpleStrategy.

    Based on VIX levels and Nifty trend (price vs 200-day SMA).
    Much simpler than the full MarketRegime with hysteresis.
    """

    BULLISH = "bullish"      # VIX < 18 AND trend up -> full equity
    NEUTRAL = "neutral"      # Neither bullish nor defensive
    DEFENSIVE = "defensive"  # VIX > 25 OR trend down -> reduce exposure


@dataclass
class SimpleRegimeResult:
    """
    Result of simple regime detection.

    Contains the detected regime and key metrics for logging.
    """

    regime: SimpleRegime
    vix_level: float
    trend_up: bool           # Price > 200-day SMA
    sma_200: float           # 200-day SMA value
    current_price: float     # Current Nifty price

    def __str__(self) -> str:
        trend_str = "uptrend" if self.trend_up else "downtrend"
        return f"{self.regime.value.upper()} (VIX={self.vix_level:.1f}, {trend_str})"


def detect_simple_regime(
    nifty_prices: pd.Series,
    vix_value: float,
    vix_bullish_threshold: float = 18.0,
    vix_defensive_threshold: float = 25.0,
) -> SimpleRegimeResult:
    """
    Detect market regime using simple VIX + trend approach.

    This is a simplified regime detection for the SimpleStrategy:
    - BULLISH: VIX < 18 AND price > 200-day SMA
    - DEFENSIVE: VIX > 25 OR price < 200-day SMA
    - NEUTRAL: Otherwise

    Based on RegimeFolio research showing VIX-based regime achieves
    495% return with Sharpe 1.88.

    Args:
        nifty_prices: Series of Nifty 50 prices with DatetimeIndex
        vix_value: Current India VIX value
        vix_bullish_threshold: VIX below this (with uptrend) = BULLISH
        vix_defensive_threshold: VIX above this = DEFENSIVE

    Returns:
        SimpleRegimeResult with detected regime
    """
    current_price = nifty_prices.iloc[-1]

    # Calculate 200-day SMA
    if len(nifty_prices) >= 200:
        sma_200 = nifty_prices.iloc[-200:].mean()
    else:
        # Fall back to available data
        sma_200 = nifty_prices.mean()

    trend_up = current_price > sma_200

    # Determine regime
    if vix_value < vix_bullish_threshold and trend_up:
        regime = SimpleRegime.BULLISH
    elif vix_value > vix_defensive_threshold or not trend_up:
        regime = SimpleRegime.DEFENSIVE
    else:
        regime = SimpleRegime.NEUTRAL

    return SimpleRegimeResult(
        regime=regime,
        vix_level=vix_value,
        trend_up=trend_up,
        sma_200=sma_200,
        current_price=current_price,
    )


# =============================================================================
# Enhanced Indicators for AllWeather Strategy
# =============================================================================


@dataclass
class RelativeStrengthResult:
    """
    Result of Relative Strength calculation.

    RS measures how a stock performs relative to a benchmark index.
    RS > 1.0 means outperforming, RS < 1.0 means underperforming.
    """

    rs_21d: float         # 21-day (1-month) RS ratio
    rs_63d: float         # 63-day (3-month) RS ratio
    rs_126d: float        # 126-day (6-month) RS ratio
    rs_composite: float   # Weighted composite RS


def calculate_relative_strength(
    stock_prices: pd.Series,
    benchmark_prices: pd.Series,
    lookback_short: int = 21,
    lookback_medium: int = 63,
    lookback_long: int = 126,
    weight_short: float = 0.30,
    weight_medium: float = 0.40,
    weight_long: float = 0.30,
) -> Optional[RelativeStrengthResult]:
    """
    Calculate Relative Strength (RS) vs benchmark.

    RS = (stock_return / benchmark_return) over a period.
    RS > 1.0 indicates outperformance.

    The composite uses multiple timeframes for robustness:
    - Short (21d): Captures recent momentum shift
    - Medium (63d): Primary signal
    - Long (126d): Trend confirmation

    Args:
        stock_prices: Stock price series with DateTimeIndex
        benchmark_prices: Benchmark price series (e.g., NIFTY 50)
        lookback_short: Days for short-term RS (default: 21)
        lookback_medium: Days for medium-term RS (default: 63)
        lookback_long: Days for long-term RS (default: 126)
        weight_short: Weight for short-term RS (default: 0.30)
        weight_medium: Weight for medium-term RS (default: 0.40)
        weight_long: Weight for long-term RS (default: 0.30)

    Returns:
        RelativeStrengthResult or None if insufficient data
    """
    if len(stock_prices) < lookback_long or len(benchmark_prices) < lookback_long:
        return None

    # Calculate returns over each period using date-based alignment
    # This ensures we compare returns over the same calendar period
    def calc_rs(prices_stock: pd.Series, prices_bench: pd.Series, days: int) -> float:
        if len(prices_stock) < days or len(prices_bench) < days:
            return 1.0

        # Get the most recent common date
        common_dates = prices_stock.index.intersection(prices_bench.index)
        if len(common_dates) < days:
            return 1.0

        # Use date-based lookup instead of positional
        end_date = common_dates[-1]
        # Find the date approximately 'days' trading days ago
        start_idx = max(0, len(common_dates) - days)
        start_date = common_dates[start_idx]

        try:
            stock_end = prices_stock.loc[end_date]
            stock_start = prices_stock.loc[start_date]
            bench_end = prices_bench.loc[end_date]
            bench_start = prices_bench.loc[start_date]

            if stock_start <= 0 or bench_start <= 0:
                return 1.0

            stock_return = (stock_end / stock_start) - 1
            bench_return = (bench_end / bench_start) - 1
        except KeyError:
            return 1.0

        # Avoid division by zero or very small returns
        if abs(bench_return) < 0.001:
            return 1.0 if stock_return >= 0 else 0.99

        # RS is the ratio of returns + 1 to handle negative returns properly
        # This gives RS > 1 for outperformance, RS < 1 for underperformance
        stock_factor = 1 + stock_return
        bench_factor = 1 + bench_return

        if bench_factor <= 0:
            return 1.0

        return stock_factor / bench_factor

    rs_21d = calc_rs(stock_prices, benchmark_prices, lookback_short)
    rs_63d = calc_rs(stock_prices, benchmark_prices, lookback_medium)
    rs_126d = calc_rs(stock_prices, benchmark_prices, lookback_long)

    rs_composite = (
        weight_short * rs_21d +
        weight_medium * rs_63d +
        weight_long * rs_126d
    )

    return RelativeStrengthResult(
        rs_21d=rs_21d,
        rs_63d=rs_63d,
        rs_126d=rs_126d,
        rs_composite=rs_composite,
    )


def calculate_momentum_acceleration(
    prices: pd.Series,
    short_period: int = 21,
    medium_period: int = 63,
) -> float:
    """
    Calculate momentum acceleration.

    Acceleration = short-term momentum / medium-term momentum.
    > 1.0 indicates accelerating momentum (bullish)
    < 1.0 indicates decelerating momentum (bearish)

    Args:
        prices: Price series with DateTimeIndex
        short_period: Days for short-term momentum (default: 21)
        medium_period: Days for medium-term momentum (default: 63)

    Returns:
        Acceleration ratio (> 1.0 = accelerating, < 1.0 = decelerating)
    """
    if len(prices) < medium_period:
        return 1.0

    # Annualized momentum = return / sqrt(period)
    # This normalizes for time so we can compare different periods
    short_return = (prices.iloc[-1] / prices.iloc[-short_period]) - 1
    medium_return = (prices.iloc[-1] / prices.iloc[-medium_period]) - 1

    # Normalize by sqrt of period for fair comparison
    short_momentum = short_return / np.sqrt(short_period / 252)
    medium_momentum = medium_return / np.sqrt(medium_period / 252)

    # Avoid division by zero - check (1 + medium_momentum) not just medium_momentum
    # This handles cases where medium_momentum is close to -1
    denominator = 1 + medium_momentum
    if abs(denominator) < 0.01:
        return 1.0 if short_momentum >= 0 else 0.5

    # Ratio > 1.0 means short-term is stronger than medium-term
    acceleration = (1 + short_momentum) / denominator

    # Clip to reasonable range
    return float(np.clip(acceleration, 0.5, 2.0))


@dataclass
class ExhaustionResult:
    """
    Result of exhaustion detection.

    Identifies potential trend exhaustion based on:
    - Extended move from moving averages
    - RSI extremes
    - Volume patterns
    """

    exhaustion_score: float  # 0-100, higher = more exhausted
    distance_from_20ema: float  # Percent above/below 20 EMA
    distance_from_50ema: float  # Percent above/below 50 EMA
    rsi_14: float             # 14-day RSI
    volume_exhaustion: float  # Volume spike indicator


def calculate_exhaustion_score(
    prices: pd.Series,
    volumes: pd.Series,
) -> Optional[ExhaustionResult]:
    """
    Calculate exhaustion score to detect potential trend exhaustion.

    An exhausted trend may be due for a pullback or reversal.
    High exhaustion scores suggest caution for new entries.

    Components:
    1. Distance from moving averages (extended moves exhaust)
    2. RSI extremes (overbought/oversold)
    3. Volume patterns (climax volume often marks exhaustion)

    Args:
        prices: Price series with DateTimeIndex
        volumes: Volume series with DateTimeIndex

    Returns:
        ExhaustionResult or None if insufficient data
    """
    if len(prices) < 50 or len(volumes) < 50:
        return None

    current_price = prices.iloc[-1]

    # Calculate EMAs
    ema_20 = prices.ewm(span=20, adjust=False).mean().iloc[-1]
    ema_50 = prices.ewm(span=50, adjust=False).mean().iloc[-1]

    distance_20 = (current_price - ema_20) / ema_20
    distance_50 = (current_price - ema_50) / ema_50

    # Calculate RSI
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean().iloc[-1]
    avg_loss = loss.rolling(14).mean().iloc[-1]

    if avg_loss == 0:
        rsi_14 = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_14 = 100 - (100 / (1 + rs))

    # Volume exhaustion (recent volume vs average)
    avg_vol_20 = volumes.iloc[-20:].mean()
    avg_vol_50 = volumes.iloc[-50:].mean()

    if avg_vol_50 > 0:
        volume_exhaustion = avg_vol_20 / avg_vol_50
    else:
        volume_exhaustion = 1.0

    # Calculate composite exhaustion score
    # Higher distance from EMAs = more exhaustion
    distance_score = min(50, abs(distance_20) * 200 + abs(distance_50) * 100)

    # RSI extremes add to exhaustion
    if rsi_14 >= 70:
        rsi_score = (rsi_14 - 70) * 1.5  # Up to 45 points
    elif rsi_14 <= 30:
        rsi_score = (30 - rsi_14) * 1.5
    else:
        rsi_score = 0

    # Volume climax adds to exhaustion
    if volume_exhaustion > 2.0:
        vol_score = min(20, (volume_exhaustion - 1.0) * 10)
    else:
        vol_score = 0

    exhaustion_score = min(100, distance_score + rsi_score + vol_score)

    return ExhaustionResult(
        exhaustion_score=exhaustion_score,
        distance_from_20ema=distance_20,
        distance_from_50ema=distance_50,
        rsi_14=rsi_14,
        volume_exhaustion=volume_exhaustion,
    )


@dataclass
class BullRecoverySignals:
    """
    Signals indicating a bull market recovery from crash/correction.

    Bull recovery is detected when:
    1. VIX is declining from elevated levels (fear subsiding)
    2. Position momentum is positive (market improving)
    3. Recent returns show upward trend

    This helps AllWeather strategy capture more upside during V-shaped recoveries.
    """

    is_bull_recovery: bool        # Are we in a bull recovery?
    vix_declining: bool           # Is VIX declining from spike?
    vix_decline_pct: float        # How much has VIX declined (0-1)?
    momentum_positive: bool       # Is position momentum positive?
    position_momentum: float      # Current position momentum
    return_1m: float              # 1-month return
    return_3m: float              # 3-month return
    recovery_strength: float      # Composite recovery strength (0-1)


def calculate_bull_recovery_signals(
    nifty_prices: pd.Series,
    vix_value: float,
    vix_history: Optional[pd.Series],
    position_momentum: float,
    return_1m: float,
    return_3m: float,
    vix_threshold: float = 20.0,
    momentum_threshold: float = 0.003,
) -> BullRecoverySignals:
    """
    Detect if market is recovering from a crash/correction (bull recovery).

    Bull recovery signals help AllWeather capture more upside during V-shaped
    recoveries by providing extra filter relaxation when:
    1. VIX is declining from elevated levels (fear subsiding)
    2. Position momentum is positive and strong (market improving)
    3. Recent returns are recovering from negative

    Args:
        nifty_prices: Series of Nifty 50 prices with DatetimeIndex
        vix_value: Current India VIX value
        vix_history: Optional VIX history for spike detection (last 20+ days)
        position_momentum: Current position momentum (rate of change)
        return_1m: 1-month return
        return_3m: 3-month return
        vix_threshold: VIX level to consider as elevated (default: 20.0)
        momentum_threshold: Min momentum to confirm recovery (default: 0.003)

    Returns:
        BullRecoverySignals with detection results
    """
    # Check VIX decline from spike
    vix_declining = False
    vix_decline_pct = 0.0

    if vix_history is not None and len(vix_history) >= 20:
        # Find the peak VIX in last 20 days
        recent_peak = vix_history.iloc[-20:].max()

        # Check if peak was above threshold
        if recent_peak >= vix_threshold:
            # Check if VIX has declined meaningfully
            decline = (recent_peak - vix_value) / recent_peak
            if decline >= 0.10:  # At least 10% decline from peak
                vix_declining = True
                vix_decline_pct = min(decline, 0.50)  # Cap at 50% for scoring

    # Check position momentum
    momentum_positive = position_momentum > momentum_threshold

    # Calculate recovery strength (0-1 composite)
    # Higher strength = more filter relaxation warranted
    recovery_strength = 0.0

    # VIX contribution (0-0.4)
    if vix_declining:
        recovery_strength += min(vix_decline_pct * 0.8, 0.4)

    # Momentum contribution (0-0.3)
    if momentum_positive:
        momentum_score = min(position_momentum / 0.01, 1.0)  # Normalize to 0-1
        recovery_strength += momentum_score * 0.3

    # Return recovery contribution (0-0.3)
    # Returns improving from negative is a strong signal
    if return_1m > 0 and return_3m < 0.05:
        # 1M positive but 3M not yet fully recovered
        recovery_strength += 0.2
    if return_1m > 0.03:
        recovery_strength += 0.1

    # Is this a bull recovery?
    # Requires either: VIX declining + positive momentum, or very strong momentum
    is_bull_recovery = (
        (vix_declining and momentum_positive) or
        (position_momentum > 0.008)  # Very strong momentum alone
    )

    return BullRecoverySignals(
        is_bull_recovery=is_bull_recovery,
        vix_declining=vix_declining,
        vix_decline_pct=vix_decline_pct,
        momentum_positive=momentum_positive,
        position_momentum=position_momentum,
        return_1m=return_1m,
        return_3m=return_3m,
        recovery_strength=min(recovery_strength, 1.0),
    )


# =============================================================================
# MARKET MODE DETECTION (for Hybrid Strategy)
# =============================================================================

class MarketMode(Enum):
    """
    Market mode classification for adaptive hybrid strategy.

    More granular than MarketRegime - specifically designed to adapt
    entry filters and scoring weights based on market conditions.
    """

    STRONG_BULL = "strong_bull"  # Breadth > 70%, VIX < 15, strong momentum
    BULL = "bull"                # Breadth > 55%, VIX < 20, positive momentum
    NEUTRAL = "neutral"          # Breadth 40-55%, moderate conditions
    CORRECTION = "correction"    # Breadth < 40%, declining trend, negative momentum
    CRISIS = "crisis"            # VIX > 30 or severe drawdown


@dataclass
class MarketBreadth:
    """Market breadth indicators calculated from universe data."""

    pct_above_50ma: float       # % of stocks above 50-day MA (0-1)
    pct_above_200ma: float      # % of stocks above 200-day MA (0-1)
    pct_positive_momentum: float  # % of stocks with positive 21-day momentum (0-1)
    avg_distance_from_high: float  # Average distance from 52W high (0-1)
    breadth_momentum: float     # Change in breadth over last 10 days
    sample_size: int            # Number of stocks used in calculation


@dataclass
class MarketModeResult:
    """
    Result of market mode detection with adaptive parameters.

    Provides specific multipliers and thresholds for hybrid strategy
    to adapt its behavior based on current market conditions.
    """

    mode: MarketMode
    breadth: MarketBreadth
    vix_level: float
    index_return_1m: float
    index_return_3m: float
    index_momentum: float       # Index position momentum

    # Adaptive scoring weights (should sum to 1.0)
    wp_weight: float            # Weekly Persistence weight
    rs_weight: float            # Relative Strength weight
    dh_weight: float            # Daily Health weight

    # Adaptive entry thresholds
    min_score_mult: float       # Multiplier for min_hybrid_score
    max_rank_mult: float        # Multiplier for max_entry_rank
    position_size_mult: float   # Multiplier for position sizes

    # Confidence and debug info
    confidence: float           # How confident are we in this mode (0-1)
    signals: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.mode.value.upper()}: "
            f"Breadth={self.breadth.pct_above_50ma:.0%}, "
            f"VIX={self.vix_level:.1f}, "
            f"Weights=WP{self.wp_weight:.0%}/RS{self.rs_weight:.0%}/DH{self.dh_weight:.0%}"
        )


def calculate_market_breadth(
    universe_data: Dict[str, pd.DataFrame],
    as_of_date: Optional[pd.Timestamp] = None,
) -> MarketBreadth:
    """
    Calculate market breadth indicators from universe price data.

    Args:
        universe_data: Dict mapping symbol to DataFrame with 'close' column
        as_of_date: Date to calculate breadth for (default: latest)

    Returns:
        MarketBreadth with breadth indicators
    """
    above_50ma = 0
    above_200ma = 0
    positive_momentum = 0
    distances_from_high = []
    breadth_10d_ago = 0
    valid_stocks = 0

    for symbol, df in universe_data.items():
        if df.empty or len(df) < 252:
            continue

        try:
            # Get data up to as_of_date if specified
            if as_of_date is not None:
                df = df[df.index <= as_of_date]
                if df.empty:
                    continue

            close = df['close'].iloc[-1]
            ma_50 = df['close'].rolling(50).mean().iloc[-1]
            ma_200 = df['close'].rolling(200).mean().iloc[-1]

            # Above 50 MA
            if close > ma_50:
                above_50ma += 1

            # Above 200 MA
            if close > ma_200:
                above_200ma += 1

            # Positive 21-day momentum
            if len(df) >= 21:
                ret_21d = (close / df['close'].iloc[-21]) - 1
                if ret_21d > 0:
                    positive_momentum += 1

            # Distance from 52W high
            high_52w = df['close'].rolling(252).max().iloc[-1]
            if high_52w > 0:
                dist = (high_52w - close) / high_52w
                distances_from_high.append(dist)

            # Breadth 10 days ago (for momentum)
            if len(df) >= 60:
                ma_50_10d_ago = df['close'].rolling(50).mean().iloc[-10]
                close_10d_ago = df['close'].iloc[-10]
                if close_10d_ago > ma_50_10d_ago:
                    breadth_10d_ago += 1

            valid_stocks += 1

        except Exception:
            continue

    if valid_stocks == 0:
        return MarketBreadth(
            pct_above_50ma=0.5,
            pct_above_200ma=0.5,
            pct_positive_momentum=0.5,
            avg_distance_from_high=0.1,
            breadth_momentum=0.0,
            sample_size=0,
        )

    pct_above_50 = above_50ma / valid_stocks
    pct_above_200 = above_200ma / valid_stocks
    pct_pos_mom = positive_momentum / valid_stocks
    avg_dist = np.mean(distances_from_high) if distances_from_high else 0.1
    breadth_10d = breadth_10d_ago / valid_stocks if valid_stocks > 0 else 0.5
    breadth_mom = pct_above_50 - breadth_10d  # Positive = improving

    return MarketBreadth(
        pct_above_50ma=pct_above_50,
        pct_above_200ma=pct_above_200,
        pct_positive_momentum=pct_pos_mom,
        avg_distance_from_high=avg_dist,
        breadth_momentum=breadth_mom,
        sample_size=valid_stocks,
    )


def detect_market_mode(
    breadth: MarketBreadth,
    vix_level: float,
    index_return_1m: float,
    index_return_3m: float,
    index_momentum: float,
    stress_score: float,
) -> MarketModeResult:
    """
    Detect current market mode and return adaptive parameters.

    This is the core intelligence for hybrid strategy adaptation.
    It classifies the market into one of 5 modes and provides
    specific parameter adjustments for each.

    Args:
        breadth: Market breadth indicators
        vix_level: Current VIX level
        index_return_1m: Index 1-month return
        index_return_3m: Index 3-month return
        index_momentum: Index position momentum
        stress_score: Current regime stress score (0-1)

    Returns:
        MarketModeResult with mode and adaptive parameters
    """
    signals = {
        'breadth_50ma': breadth.pct_above_50ma,
        'breadth_200ma': breadth.pct_above_200ma,
        'breadth_momentum': breadth.breadth_momentum,
        'vix': vix_level,
        'return_1m': index_return_1m,
        'return_3m': index_return_3m,
        'index_momentum': index_momentum,
        'stress': stress_score,
    }

    # Determine market mode based on multiple signals
    mode = MarketMode.NEUTRAL
    confidence = 0.5

    # CRISIS: High VIX or severe negative returns (unchanged - this is correct)
    if vix_level >= 30 or index_return_3m <= -0.15:
        mode = MarketMode.CRISIS
        confidence = 0.9 if vix_level >= 35 else 0.7

    # STRONG_BULL: Relaxed thresholds - focus on broad participation and low fear
    # Changed: breadth 70%->60%, VIX 15->18, return 8%->5%, momentum optional
    elif (breadth.pct_above_50ma >= 0.60 and
          vix_level <= 18 and
          index_return_3m >= 0.05):
        mode = MarketMode.STRONG_BULL
        confidence = 0.85 if breadth.breadth_momentum >= 0 else 0.70

    # BULL: More achievable conditions for normal bull markets
    # Changed: breadth 55%->45%, added positive 3M return as alternative
    elif ((breadth.pct_above_50ma >= 0.45 and vix_level <= 22 and index_return_1m >= 0) or
          (breadth.pct_above_200ma >= 0.55 and index_return_3m >= 0.03 and vix_level <= 22)):
        mode = MarketMode.BULL
        confidence = 0.75

    # CORRECTION: Tighter triggers - need multiple confirming signals
    # Changed: breadth alone 40%->30%, require stress OR momentum decline
    elif ((breadth.pct_above_50ma <= 0.30 and stress_score >= 0.5) or
          (breadth.breadth_momentum <= -0.08 and index_return_1m <= -0.05) or
          (index_return_3m <= -0.10 and stress_score >= 0.65)):
        mode = MarketMode.CORRECTION
        confidence = 0.80 if breadth.pct_above_50ma <= 0.25 else 0.65

    # NEUTRAL: Everything else - normal market conditions
    else:
        mode = MarketMode.NEUTRAL
        confidence = 0.60

    # Set adaptive parameters based on mode
    if mode == MarketMode.STRONG_BULL:
        # Very aggressive: Heavy RS focus, significantly relaxed filters
        wp_weight, rs_weight, dh_weight = 0.25, 0.50, 0.25
        min_score_mult = 0.80      # 75 * 0.80 = 60 (more entries)
        max_rank_mult = 1.80       # 15 * 1.80 = 27 (wider net)
        position_mult = 1.15       # Larger positions

    elif mode == MarketMode.BULL:
        # Aggressive: RS emphasis, relaxed filters
        wp_weight, rs_weight, dh_weight = 0.30, 0.45, 0.25
        min_score_mult = 0.85      # 75 * 0.85 = 63.75
        max_rank_mult = 1.50       # 15 * 1.50 = 22.5
        position_mult = 1.10

    elif mode == MarketMode.NEUTRAL:
        # Normal: Match classic strategy behavior
        wp_weight, rs_weight, dh_weight = 0.35, 0.40, 0.25
        min_score_mult = 0.95      # 75 * 0.95 = 71.25 (slightly relaxed)
        max_rank_mult = 1.20       # 15 * 1.20 = 18
        position_mult = 1.00

    elif mode == MarketMode.CORRECTION:
        # Defensive: Focus on quality, tighter but not extreme
        wp_weight, rs_weight, dh_weight = 0.45, 0.30, 0.25
        min_score_mult = 1.05      # 75 * 1.05 = 78.75 (slightly tighter)
        max_rank_mult = 0.80       # 15 * 0.80 = 12
        position_mult = 0.85

    else:  # CRISIS
        # Very defensive: Highest quality only
        wp_weight, rs_weight, dh_weight = 0.50, 0.25, 0.25
        min_score_mult = 1.15      # 75 * 1.15 = 86.25
        max_rank_mult = 0.60       # 15 * 0.60 = 9
        position_mult = 0.60

    return MarketModeResult(
        mode=mode,
        breadth=breadth,
        vix_level=vix_level,
        index_return_1m=index_return_1m,
        index_return_3m=index_return_3m,
        index_momentum=index_momentum,
        wp_weight=wp_weight,
        rs_weight=rs_weight,
        dh_weight=dh_weight,
        min_score_mult=min_score_mult,
        max_rank_mult=max_rank_mult,
        position_size_mult=position_mult,
        confidence=confidence,
        signals=signals,
    )


def calculate_breakout_quality(
    prices: pd.Series,
    volumes: pd.Series,
    lookback: int = 63,
) -> float:
    """
    Calculate breakout quality score (0-100).

    A high-quality breakout has:
    - Price at or near 52-week high
    - Volume confirmation (higher than average)
    - Clean price structure (not too volatile)
    - Tight consolidation before breakout

    Args:
        prices: Price series with DateTimeIndex
        volumes: Volume series with DateTimeIndex
        lookback: Days to analyze for breakout pattern

    Returns:
        Quality score 0-100 (higher = better quality breakout)
    """
    if len(prices) < 252 or len(volumes) < 50:
        return 50.0  # Default score

    current_price = prices.iloc[-1]

    # 1. 52-week high proximity (0-30 points)
    high_52w = prices.iloc[-252:].max()
    proximity = current_price / high_52w
    proximity_score = min(30, proximity * 30)

    # 2. Volume confirmation (0-25 points)
    recent_vol = volumes.iloc[-5:].mean()
    avg_vol = volumes.iloc[-50:].mean()

    if avg_vol > 0:
        vol_ratio = recent_vol / avg_vol
        vol_score = min(25, (vol_ratio - 0.5) * 25) if vol_ratio > 0.5 else 0
    else:
        vol_score = 12.5

    # 3. Consolidation tightness (0-25 points)
    # Lower volatility in last 20 days vs prior 40 days = tighter consolidation
    if len(prices) >= 60:
        recent_returns = prices.iloc[-20:].pct_change().dropna()
        prior_returns = prices.iloc[-60:-20].pct_change().dropna()

        recent_vol_pct = recent_returns.std() if len(recent_returns) > 0 else 0.02
        prior_vol_pct = prior_returns.std() if len(prior_returns) > 0 else 0.02

        if prior_vol_pct > 0:
            vol_contraction = 1.0 - (recent_vol_pct / prior_vol_pct)
            consolidation_score = min(25, max(0, vol_contraction * 50))
        else:
            consolidation_score = 12.5
    else:
        consolidation_score = 12.5

    # 4. Price structure (0-20 points)
    # Higher lows in recent period = bullish structure
    if len(prices) >= 20:
        lows = prices.iloc[-20:].rolling(5).min()
        if len(lows) >= 10:
            early_low = lows.iloc[5]
            recent_low = lows.iloc[-1]
            if early_low > 0:
                structure_change = (recent_low - early_low) / early_low
                structure_score = min(20, max(0, structure_change * 200 + 10))
            else:
                structure_score = 10
        else:
            structure_score = 10
    else:
        structure_score = 10

    total_score = proximity_score + vol_score + consolidation_score + structure_score
    return float(np.clip(total_score, 0, 100))


# =============================================================================
# Hybrid Strategy Indicators
# =============================================================================


def calculate_atr_ratio(
    prices: pd.Series,
    highs: Optional[pd.Series] = None,
    lows: Optional[pd.Series] = None,
    current_period: int = 14,
    historical_period: int = 63,
) -> Tuple[float, float, float]:
    """
    Calculate ATR ratio (current ATR / historical average ATR).

    Used for volatility-adjusted stop losses:
    - Ratio > 1.0: Higher than normal volatility -> widen stops
    - Ratio < 1.0: Lower than normal volatility -> tighten stops

    Args:
        prices: Close prices series with DateTimeIndex
        highs: High prices series (if None, estimated from closes)
        lows: Low prices series (if None, estimated from closes)
        current_period: Period for current ATR (default: 14)
        historical_period: Period for historical average ATR (default: 63)

    Returns:
        Tuple of (atr_ratio, current_atr_pct, historical_atr_pct)
    """
    if len(prices) < historical_period:
        return (1.0, 0.0, 0.0)

    # If highs/lows not provided, estimate from closes
    if highs is None:
        daily_range = prices.rolling(20).std() * 0.5
        highs = prices + daily_range.fillna(prices * 0.01)
    if lows is None:
        daily_range = prices.rolling(20).std() * 0.5
        lows = prices - daily_range.fillna(prices * 0.01)

    # Calculate True Range
    high_low = highs - lows
    high_close_prev = (highs - prices.shift(1)).abs()
    low_close_prev = (lows - prices.shift(1)).abs()

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    # Calculate ATR as percentage of price
    current_price = prices.iloc[-1]

    # Current ATR (recent period)
    current_atr = tr.iloc[-current_period:].mean()
    current_atr_pct = current_atr / current_price if current_price > 0 else 0.02

    # Historical average ATR
    historical_atr = tr.iloc[-historical_period:].mean()
    historical_atr_pct = historical_atr / current_price if current_price > 0 else 0.02

    # ATR ratio
    if historical_atr_pct > 0:
        atr_ratio = current_atr_pct / historical_atr_pct
    else:
        atr_ratio = 1.0

    return (float(atr_ratio), float(current_atr_pct), float(historical_atr_pct))


@dataclass
class ADXResult:
    """Result of ADX calculation for trend strength measurement."""

    adx: float           # Average Directional Index (0-100)
    plus_di: float       # Positive Directional Indicator
    minus_di: float      # Negative Directional Indicator
    is_trending: bool    # ADX >= threshold indicates trending market
    is_bullish: bool     # +DI > -DI indicates bullish trend


def calculate_adx(
    prices: pd.Series,
    highs: Optional[pd.Series] = None,
    lows: Optional[pd.Series] = None,
    period: int = 14,
) -> Optional[ADXResult]:
    """
    Calculate ADX (Average Directional Index) and directional indicators.

    ADX measures trend strength regardless of direction:
    - ADX >= 20: Trending market (stronger = more trending)
    - ADX < 20: Weak or no trend

    +DI and -DI show trend direction:
    - +DI > -DI: Bullish trend
    - -DI > +DI: Bearish trend

    Args:
        prices: Close prices series with DateTimeIndex
        highs: High prices series (if None, estimated from closes)
        lows: Low prices series (if None, estimated from closes)
        period: Smoothing period (default: 14)

    Returns:
        ADXResult with ADX, +DI, -DI values, or None if insufficient data
    """
    if len(prices) < period * 2:
        return None

    # If highs/lows not provided, estimate from closes
    if highs is None:
        # Estimate highs as closes + half of typical daily range
        daily_range = prices.rolling(20).std() * 0.5
        highs = prices + daily_range.fillna(prices * 0.01)
    if lows is None:
        daily_range = prices.rolling(20).std() * 0.5
        lows = prices - daily_range.fillna(prices * 0.01)

    # Calculate True Range
    high_low = highs - lows
    high_close_prev = (highs - prices.shift(1)).abs()
    low_close_prev = (lows - prices.shift(1)).abs()

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    # Calculate +DM and -DM (Directional Movement)
    high_diff = highs - highs.shift(1)
    low_diff = lows.shift(1) - lows

    plus_dm = pd.Series(0.0, index=prices.index)
    minus_dm = pd.Series(0.0, index=prices.index)

    # +DM: when high_diff > low_diff and high_diff > 0
    plus_dm_mask = (high_diff > low_diff) & (high_diff > 0)
    plus_dm[plus_dm_mask] = high_diff[plus_dm_mask]

    # -DM: when low_diff > high_diff and low_diff > 0
    minus_dm_mask = (low_diff > high_diff) & (low_diff > 0)
    minus_dm[minus_dm_mask] = low_diff[minus_dm_mask]

    # Smooth with Wilder's smoothing (EMA-like with alpha = 1/period)
    def wilder_smooth(series: pd.Series, n: int) -> pd.Series:
        """Wilder's smoothing method."""
        alpha = 1.0 / n
        return series.ewm(alpha=alpha, adjust=False).mean()

    atr = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)

    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / atr.replace(0, np.nan)).fillna(0)
    minus_di = 100 * (minus_dm_smooth / atr.replace(0, np.nan)).fillna(0)

    # Calculate DX (Directional Index)
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * (di_diff / di_sum.replace(0, np.nan)).fillna(0)

    # Calculate ADX (smoothed DX)
    adx = wilder_smooth(dx, period)

    # Get current values
    current_adx = float(adx.iloc[-1])
    current_plus_di = float(plus_di.iloc[-1])
    current_minus_di = float(minus_di.iloc[-1])

    return ADXResult(
        adx=current_adx,
        plus_di=current_plus_di,
        minus_di=current_minus_di,
        is_trending=current_adx >= 20,
        is_bullish=current_plus_di > current_minus_di,
    )


def calculate_macd_histogram_slope(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    slope_periods: int = 3,
) -> float:
    """
    Calculate MACD histogram slope for momentum quality.

    Positive and increasing slope = strong momentum.
    Negative or decreasing slope = weakening momentum.

    Args:
        prices: Price series with DateTimeIndex
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
        slope_periods: Periods to measure slope (default: 3)

    Returns:
        Slope of MACD histogram (typically -0.5 to 0.5)
    """
    if len(prices) < slow_period + signal_period + slope_periods:
        return 0.0

    # Calculate MACD components
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    # Calculate slope of histogram over recent periods
    if len(histogram) < slope_periods:
        return 0.0

    recent_hist = histogram.iloc[-slope_periods:]
    if len(recent_hist) < 2:
        return 0.0

    # Simple linear regression slope (normalized by price)
    x = np.arange(len(recent_hist))
    y = recent_hist.values
    slope = np.polyfit(x, y, 1)[0]

    # Normalize by current price for comparability
    current_price = prices.iloc[-1]
    if current_price > 0:
        slope = slope / current_price * 100  # As percentage

    return float(slope)


def classify_rsi_regime(rsi: float) -> str:
    """
    Classify RSI into regime categories.

    Args:
        rsi: RSI value (0-100)

    Returns:
        One of: "overbought", "trending_up", "neutral", "trending_down", "oversold"
    """
    if rsi >= 70:
        return "overbought"
    elif rsi >= 55:
        return "trending_up"
    elif rsi >= 45:
        return "neutral"
    elif rsi >= 30:
        return "trending_down"
    else:
        return "oversold"


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index).

    Args:
        prices: Price series with DateTimeIndex
        period: RSI period (default: 14)

    Returns:
        RSI value (0-100)
    """
    if len(prices) < period + 1:
        return 50.0

    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_higher_highs_score(
    highs: pd.Series,
    lows: pd.Series,
    lookback: int = 10,
) -> float:
    """
    Calculate higher highs / higher lows score (0-100).

    Measures the consistency of the bullish price structure.

    Args:
        highs: High prices series
        lows: Low prices series
        lookback: Number of periods to analyze

    Returns:
        Score 0-100 (100 = perfect HH/HL pattern)
    """
    if len(highs) < lookback or len(lows) < lookback:
        return 50.0

    recent_highs = highs.iloc[-lookback:]
    recent_lows = lows.iloc[-lookback:]

    # Count higher highs and higher lows
    hh_count = 0
    hl_count = 0

    for i in range(1, len(recent_highs)):
        if recent_highs.iloc[i] > recent_highs.iloc[i - 1]:
            hh_count += 1
        if recent_lows.iloc[i] > recent_lows.iloc[i - 1]:
            hl_count += 1

    max_possible = lookback - 1
    if max_possible <= 0:
        return 50.0

    # Combined score (HH and HL equally weighted)
    hh_pct = hh_count / max_possible
    hl_pct = hl_count / max_possible

    score = (hh_pct * 50 + hl_pct * 50)
    return float(np.clip(score, 0, 100))


def calculate_weeks_in_trend(
    closes: pd.Series,
    ma_period: int = 20,
) -> int:
    """
    Calculate number of consecutive weeks above moving average.

    Args:
        closes: Close prices (weekly data)
        ma_period: Moving average period

    Returns:
        Number of weeks in current trend
    """
    if len(closes) < ma_period:
        return 0

    ma = closes.rolling(ma_period).mean()
    above_ma = closes > ma

    # Count consecutive True values from end
    weeks = 0
    for val in above_ma.iloc[::-1]:
        if val:
            weeks += 1
        else:
            break

    return weeks


def detect_sideways_market(
    index_prices: pd.Series,
    breadth_above_50d: Optional[float] = None,
    vix_level: float = 15.0,
    atr_threshold: float = 0.015,
    breadth_min: float = 0.40,
    breadth_max: float = 0.60,
    vix_min: float = 12.0,
    vix_max: float = 20.0,
) -> Tuple[bool, Dict[str, float]]:
    """
    Detect if market is in sideways (range-bound) regime.

    Criteria:
    - ATR < 1.5% of price (narrow range)
    - Breadth between 40-60% (neutral)
    - VIX in normal range (12-20)

    Args:
        index_prices: Index price series
        breadth_above_50d: Percentage of stocks above 50-day MA (0-1)
        vix_level: Current VIX value
        atr_threshold: ATR threshold as fraction of price
        breadth_min: Minimum breadth for sideways
        breadth_max: Maximum breadth for sideways
        vix_min: Minimum VIX for sideways
        vix_max: Maximum VIX for sideways

    Returns:
        Tuple of (is_sideways, details_dict)
    """
    if len(index_prices) < 14:
        return (False, {})

    # Calculate ATR as percentage of price
    current_price = index_prices.iloc[-1]
    tr_values = []
    for i in range(1, min(15, len(index_prices))):
        high = index_prices.iloc[-(i + 1):-i + 1].max() if i > 1 else index_prices.iloc[-2:].max()
        low = index_prices.iloc[-(i + 1):-i + 1].min() if i > 1 else index_prices.iloc[-2:].min()
        tr = high - low
        tr_values.append(tr)

    if not tr_values:
        return (False, {})

    atr = sum(tr_values) / len(tr_values)
    atr_pct = atr / current_price if current_price > 0 else 0

    details = {
        "atr_pct": atr_pct,
        "vix_level": vix_level,
        "breadth": breadth_above_50d if breadth_above_50d is not None else 0.5,
    }

    # Check all conditions
    atr_ok = atr_pct < atr_threshold
    vix_ok = vix_min <= vix_level <= vix_max

    if breadth_above_50d is not None:
        breadth_ok = breadth_min <= breadth_above_50d <= breadth_max
    else:
        breadth_ok = True  # If not provided, assume OK

    is_sideways = atr_ok and vix_ok and breadth_ok

    return (is_sideways, details)


@dataclass
class MomentumTypeClassification:
    """Result of momentum type classification."""

    momentum_type: str  # TREND, ROTATION, MEAN_REVERSION, REVERSAL
    confidence: float   # 0-1 confidence in classification
    position_size_mult: float  # Position size multiplier
    max_hold_weeks: int  # Maximum hold period in weeks
    stop_width: float   # Stop loss width (decimal)


def classify_momentum_type(
    prices: pd.Series,
    weekly_prices: Optional[pd.Series] = None,
    ma_40w: Optional[float] = None,
    weeks_in_trend: int = 0,
    hh_score: float = 50.0,
    rs_accelerating: bool = False,
    vol_expanding: bool = False,
    macd_turning_positive: bool = False,
) -> MomentumTypeClassification:
    """
    Classify stock into momentum type categories.

    Types:
    - TREND: >10% above MA40w, 8+ weeks in trend, HH score >70
    - ROTATION: 0-10% from MA40w, 2-8 weeks, RS accelerating
    - MEAN_REVERSION: Below MA or <2 weeks, vol expanding
    - REVERSAL: Monthly MACD turning positive from negative

    Args:
        prices: Daily price series
        weekly_prices: Weekly price series (optional)
        ma_40w: 40-week moving average value
        weeks_in_trend: Consecutive weeks above trend MA
        hh_score: Higher highs/lows score (0-100)
        rs_accelerating: Is relative strength improving?
        vol_expanding: Is volatility expanding?
        macd_turning_positive: Is MACD histogram turning positive?

    Returns:
        MomentumTypeClassification with type and parameters
    """
    current_price = prices.iloc[-1]

    # Calculate distance from MA if not provided
    if ma_40w is None and weekly_prices is not None and len(weekly_prices) >= 40:
        ma_40w = weekly_prices.rolling(40).mean().iloc[-1]
    elif ma_40w is None:
        # Use daily 200 SMA as proxy for 40w MA
        if len(prices) >= 200:
            ma_40w = prices.rolling(200).mean().iloc[-1]
        else:
            ma_40w = current_price  # Default to current price

    dist_from_ma = (current_price - ma_40w) / ma_40w if ma_40w > 0 else 0.0

    # Classification logic
    # Priority: REVERSAL > TREND > ROTATION > MEAN_REVERSION

    # REVERSAL: MACD turning positive (early stage recovery)
    if macd_turning_positive and dist_from_ma < 0.10 and weeks_in_trend < 4:
        return MomentumTypeClassification(
            momentum_type="REVERSAL",
            confidence=0.7 if dist_from_ma > 0 else 0.5,
            position_size_mult=0.80,
            max_hold_weeks=12,
            stop_width=0.18,
        )

    # TREND: Strong established uptrend
    if dist_from_ma > 0.10 and weeks_in_trend >= 8 and hh_score > 70:
        return MomentumTypeClassification(
            momentum_type="TREND",
            confidence=min(0.9, hh_score / 100),
            position_size_mult=1.00,
            max_hold_weeks=20,
            stop_width=0.18,
        )

    # ROTATION: Early/mid stage trend with RS improving
    if 0 <= dist_from_ma <= 0.10 and 2 <= weeks_in_trend <= 8 and rs_accelerating:
        return MomentumTypeClassification(
            momentum_type="ROTATION",
            confidence=0.6 + (weeks_in_trend / 20),
            position_size_mult=0.80,
            max_hold_weeks=8,
            stop_width=0.15,
        )

    # MEAN_REVERSION: Below MA or very early trend
    if dist_from_ma < 0 or weeks_in_trend < 2:
        confidence = 0.4 if vol_expanding else 0.3
        return MomentumTypeClassification(
            momentum_type="MEAN_REVERSION",
            confidence=confidence,
            position_size_mult=0.60,
            max_hold_weeks=4,
            stop_width=0.12,
        )

    # Default to ROTATION for ambiguous cases
    return MomentumTypeClassification(
        momentum_type="ROTATION",
        confidence=0.5,
        position_size_mult=0.80,
        max_hold_weeks=8,
        stop_width=0.15,
    )


# =============================================================================
# Dynamic Rebalancing Indicators
# =============================================================================


@dataclass
class BreadthThrustResult:
    """
    Result of breadth thrust detection.

    Breadth thrust is a powerful bullish signal that occurs when market breadth
    moves from oversold to overbought levels very quickly (within 10 days).

    Source: TheRobustTrader - Market Momentum Breadth Thrust
    """

    is_thrust: bool                    # True if breadth thrust detected
    current_breadth: float             # Current % stocks above 50-day MA
    breadth_10d_ago: float            # Breadth 10 days ago
    breadth_change: float              # Change in breadth over 10 days
    days_to_thrust: int               # Days it took to reach thrust level
    thrust_strength: float            # 0-1 strength score


def detect_breadth_thrust(
    breadth_history: pd.Series,
    thrust_low: float = 0.40,
    thrust_high: float = 0.615,
    max_days: int = 10,
) -> BreadthThrustResult:
    """
    Detect breadth thrust signal from breadth history.

    A breadth thrust occurs when:
    1. Breadth starts below thrust_low (oversold)
    2. Within max_days, breadth rises above thrust_high (overbought)

    This is a powerful bullish signal indicating aggressive buying across
    a wide range of stocks, often marking major market bottoms.

    Args:
        breadth_history: Series of daily breadth values (% stocks above 50-day MA)
        thrust_low: Low breadth level to start measuring (default: 40%)
        thrust_high: High breadth level that confirms thrust (default: 61.5%)
        max_days: Maximum days for the move to qualify as thrust (default: 10)

    Returns:
        BreadthThrustResult with detection results
    """
    if len(breadth_history) < max_days + 1:
        return BreadthThrustResult(
            is_thrust=False,
            current_breadth=breadth_history.iloc[-1] if len(breadth_history) > 0 else 0.5,
            breadth_10d_ago=0.5,
            breadth_change=0.0,
            days_to_thrust=0,
            thrust_strength=0.0,
        )

    current_breadth = breadth_history.iloc[-1]
    breadth_10d_ago = breadth_history.iloc[-(max_days + 1)]
    breadth_change = current_breadth - breadth_10d_ago

    # Check if current breadth is above thrust high
    if current_breadth < thrust_high:
        return BreadthThrustResult(
            is_thrust=False,
            current_breadth=current_breadth,
            breadth_10d_ago=breadth_10d_ago,
            breadth_change=breadth_change,
            days_to_thrust=0,
            thrust_strength=0.0,
        )

    # Look back to find when breadth was below thrust_low
    days_to_thrust = 0
    found_low = False

    for i in range(1, max_days + 1):
        idx = -(i + 1)
        if abs(idx) > len(breadth_history):
            break
        if breadth_history.iloc[idx] <= thrust_low:
            found_low = True
            days_to_thrust = i
            break

    is_thrust = found_low

    # Calculate thrust strength based on speed and magnitude
    if is_thrust:
        speed_factor = (max_days - days_to_thrust + 1) / max_days  # Faster = stronger
        magnitude_factor = min(1.0, (current_breadth - thrust_high) / 0.10 + 0.5)
        thrust_strength = (speed_factor + magnitude_factor) / 2
    else:
        thrust_strength = 0.0

    return BreadthThrustResult(
        is_thrust=is_thrust,
        current_breadth=current_breadth,
        breadth_10d_ago=breadth_10d_ago,
        breadth_change=breadth_change,
        days_to_thrust=days_to_thrust,
        thrust_strength=thrust_strength,
    )


@dataclass
class VIXTermStructureResult:
    """
    Result of VIX term structure analysis.

    VIX term structure provides important market signals:
    - Contango (VIX < VIX3M): Normal market, stability expected
    - Backwardation (VIX > VIX3M): Fear/panic, often marks bottoms

    Source: VIX Regime Detection (Medium - The VIX Code Cracked)
    """

    is_backwardation: bool             # True if VIX > VIX3M (panic)
    is_contango: bool                  # True if VIX < VIX3M (normal)
    vix_level: float                   # Current VIX level
    vix_3m_level: float               # VIX 3-month level (or estimate)
    term_spread: float                 # VIX - VIX3M (negative = contango)
    term_spread_pct: float            # Spread as % of VIX
    signal: str                        # "PANIC", "CAUTION", "NORMAL", "CALM"


def detect_vix_term_structure(
    vix_current: float,
    vix_3m: Optional[float] = None,
    vix_history: Optional[pd.Series] = None,
    backwardation_threshold: float = 0.05,
    contango_threshold: float = -0.05,
) -> VIXTermStructureResult:
    """
    Detect VIX term structure state.

    When VIX > VIX3M (backwardation), it signals near-term fear exceeding
    longer-term expectations - often marks market bottoms and recovery
    opportunities.

    When VIX < VIX3M (contango), it signals normal market conditions with
    stability expected.

    Args:
        vix_current: Current VIX level
        vix_3m: 3-month VIX level (if available)
        vix_history: VIX history for estimating 3M level if not available
        backwardation_threshold: Spread % to confirm backwardation (default: 5%)
        contango_threshold: Spread % to confirm contango (default: -5%)

    Returns:
        VIXTermStructureResult with term structure analysis
    """
    # If VIX3M not available, estimate from history
    if vix_3m is None:
        if vix_history is not None and len(vix_history) >= 63:
            # Use 3-month average as proxy for VIX3M
            vix_3m = vix_history.iloc[-63:].mean()
        else:
            # Default to slightly above current (assume normal contango)
            vix_3m = vix_current * 1.05

    # Calculate spread
    term_spread = vix_current - vix_3m
    term_spread_pct = term_spread / vix_3m if vix_3m > 0 else 0.0

    # Determine term structure state
    is_backwardation = term_spread_pct > backwardation_threshold
    is_contango = term_spread_pct < contango_threshold

    # Determine signal
    if is_backwardation and vix_current > 30:
        signal = "PANIC"
    elif is_backwardation:
        signal = "CAUTION"
    elif vix_current < 15 and is_contango:
        signal = "CALM"
    else:
        signal = "NORMAL"

    return VIXTermStructureResult(
        is_backwardation=is_backwardation,
        is_contango=is_contango,
        vix_level=vix_current,
        vix_3m_level=vix_3m,
        term_spread=term_spread,
        term_spread_pct=term_spread_pct,
        signal=signal,
    )


@dataclass
class VIXRecoveryResult:
    """
    Result of VIX spike recovery detection.

    Detects when VIX is declining from a recent spike, which often
    precedes market recovery opportunities.
    """

    is_recovering: bool                # True if VIX declining from spike
    vix_current: float                 # Current VIX level
    vix_peak_20d: float               # 20-day peak VIX
    decline_from_peak: float          # Percentage decline from peak
    days_since_peak: int              # Days since the peak
    recovery_strength: float          # 0-1 strength of recovery signal


def detect_vix_recovery(
    vix_history: pd.Series,
    spike_threshold: float = 25.0,
    min_decline: float = 0.15,
) -> VIXRecoveryResult:
    """
    Detect VIX recovery from spike.

    A VIX recovery signal occurs when:
    1. VIX spiked above spike_threshold in last 20 days
    2. VIX has declined by at least min_decline from the peak

    This signals that fear is subsiding and market may be ready to recover.

    Args:
        vix_history: Series of VIX values (at least 20 days)
        spike_threshold: VIX level considered a spike (default: 25.0)
        min_decline: Minimum decline from peak to trigger (default: 15%)

    Returns:
        VIXRecoveryResult with recovery detection
    """
    if len(vix_history) < 5:
        return VIXRecoveryResult(
            is_recovering=False,
            vix_current=vix_history.iloc[-1] if len(vix_history) > 0 else 15.0,
            vix_peak_20d=15.0,
            decline_from_peak=0.0,
            days_since_peak=0,
            recovery_strength=0.0,
        )

    # Look at last 20 days (or available)
    lookback = min(20, len(vix_history))
    vix_recent = vix_history.iloc[-lookback:]
    vix_current = vix_history.iloc[-1]

    # Find peak in recent period
    vix_peak = vix_recent.max()
    peak_idx = vix_recent.idxmax()
    days_since_peak = len(vix_recent) - vix_recent.index.get_loc(peak_idx) - 1

    # Calculate decline from peak
    decline_from_peak = (vix_peak - vix_current) / vix_peak if vix_peak > 0 else 0.0

    # Check if this qualifies as recovery
    is_recovering = (
        vix_peak >= spike_threshold and
        decline_from_peak >= min_decline and
        days_since_peak >= 1  # Peak was at least 1 day ago
    )

    # Calculate recovery strength
    if is_recovering:
        # Stronger signal if decline is larger and VIX still elevated
        decline_strength = min(1.0, decline_from_peak / 0.30)  # Max at 30% decline
        elevation_strength = min(1.0, (vix_current - 15) / 15) if vix_current > 15 else 0.0
        recovery_strength = (decline_strength * 0.7 + elevation_strength * 0.3)
    else:
        recovery_strength = 0.0

    return VIXRecoveryResult(
        is_recovering=is_recovering,
        vix_current=vix_current,
        vix_peak_20d=vix_peak,
        decline_from_peak=decline_from_peak,
        days_since_peak=days_since_peak,
        recovery_strength=recovery_strength,
    )


@dataclass
class MomentumCrashSignal:
    """
    Result of momentum crash detection.

    Research shows that after significant market drops, momentum strategies
    suffer "momentum crash" where past losers outperform past winners.
    Switching to contrarian mode can avoid this.

    Source: ScienceDirect - Momentum Crash Research
    """

    is_crash: bool                     # True if momentum crash conditions met
    market_1m_return: float           # Market 1-month return
    market_3m_return: float           # Market 3-month return
    volatility_spike: bool            # True if VIX spiked significantly
    recommendation: str               # "NORMAL", "REDUCE_MOMENTUM", "CONTRARIAN"


def detect_momentum_crash(
    market_prices: pd.Series,
    vix_current: float,
    vix_history: Optional[pd.Series] = None,
    crash_threshold: float = -0.07,
    vix_spike_threshold: float = 30.0,
    early_warning_threshold: float = -0.05,
    early_warning_3m_threshold: float = -0.08,
) -> MomentumCrashSignal:
    """
    Detect momentum crash conditions.

    Two-tier detection (E6):
    Tier 1 (full crash): 1M return <= crash_threshold AND (VIX spike OR 3M <= -15%)
    Tier 2 (early warning): 1M return <= -5% AND 3M return <= -8% (slow grind, no VIX spike needed)

    Args:
        market_prices: Market index prices (e.g., NIFTY 50)
        vix_current: Current VIX level
        vix_history: Optional VIX history for spike detection
        crash_threshold: 1-month return threshold for crash (E6: -0.10→-0.07)
        vix_spike_threshold: VIX level considered a spike (default: 30)
        early_warning_threshold: 1M return for early warning (E6: -0.05)
        early_warning_3m_threshold: 3M return confirmation for early warning (E6: -0.08)

    Returns:
        MomentumCrashSignal with crash detection
    """
    if len(market_prices) < 63:
        return MomentumCrashSignal(
            is_crash=False,
            market_1m_return=0.0,
            market_3m_return=0.0,
            volatility_spike=False,
            recommendation="NORMAL",
        )

    # Calculate returns
    current_price = market_prices.iloc[-1]
    price_1m_ago = market_prices.iloc[-21] if len(market_prices) >= 21 else current_price
    price_3m_ago = market_prices.iloc[-63] if len(market_prices) >= 63 else current_price

    market_1m_return = (current_price - price_1m_ago) / price_1m_ago if price_1m_ago > 0 else 0.0
    market_3m_return = (current_price - price_3m_ago) / price_3m_ago if price_3m_ago > 0 else 0.0

    # Check for VIX spike
    volatility_spike = vix_current >= vix_spike_threshold
    if vix_history is not None and len(vix_history) >= 5:
        vix_5d_ago = vix_history.iloc[-5]
        vix_increase = (vix_current - vix_5d_ago) / vix_5d_ago if vix_5d_ago > 0 else 0.0
        volatility_spike = volatility_spike or vix_increase > 0.50  # 50% VIX increase

    # Tier 1: Full crash conditions (E6: lowered from -0.10 to -0.07)
    is_crash = market_1m_return <= crash_threshold and (volatility_spike or market_3m_return <= -0.15)

    # Determine recommendation
    if is_crash:
        if market_1m_return <= -0.15:
            recommendation = "CONTRARIAN"  # Severe crash - consider buying beaten-down quality
        else:
            recommendation = "REDUCE_MOMENTUM"  # Moderate crash - reduce momentum exposure
    elif market_1m_return <= early_warning_threshold and volatility_spike:
        recommendation = "REDUCE_MOMENTUM"  # Early warning with VIX spike
    elif (market_1m_return <= early_warning_threshold and
          market_3m_return <= early_warning_3m_threshold):
        # E6: Tier 2 — slow grinding decline without VIX spike (catches NBFC-type crises)
        is_crash = True
        recommendation = "REDUCE_MOMENTUM"
    else:
        recommendation = "NORMAL"

    return MomentumCrashSignal(
        is_crash=is_crash,
        market_1m_return=market_1m_return,
        market_3m_return=market_3m_return,
        volatility_spike=volatility_spike,
        recommendation=recommendation,
    )


@dataclass
class RebalanceTrigger:
    """
    Result of dynamic rebalancing trigger evaluation.

    Determines whether to rebalance based on multiple event-driven triggers.
    """

    should_rebalance: bool             # True if any trigger fired
    reason: str                        # Primary reason for rebalance
    days_since_last: int              # Days since last rebalance
    triggers_fired: list               # List of all triggers that fired
    urgency: str                       # "HIGH", "MEDIUM", "LOW"


def should_trigger_rebalance(
    days_since_last: int,
    current_regime: Optional["MarketRegime"],
    previous_regime: Optional["MarketRegime"],
    vix_level: float,
    vix_peak_20d: float,
    portfolio_drawdown: float,
    market_1m_return: float,
    breadth_thrust: bool = False,
    min_days_between: int = 5,
    max_days_between: int = 30,
    vix_recovery_decline: float = 0.15,
    vix_spike_threshold: float = 25.0,
    drawdown_threshold: float = 0.10,
    crash_threshold: float = -0.10,
) -> RebalanceTrigger:
    """
    Evaluate whether to trigger a rebalance based on multiple event-driven signals.

    Triggers:
    1. Regular interval (max_days_between) - baseline
    2. Regime transition - immediate
    3. VIX recovery (>15% drop from spike >25) - opportunity
    4. Portfolio drawdown >10% - defensive
    5. Market crash (1M < -10%) - crash avoidance
    6. Breadth thrust signal - aggressive entry

    Args:
        days_since_last: Trading days since last rebalance
        current_regime: Current market regime
        previous_regime: Previous market regime (for transition detection)
        vix_level: Current VIX level
        vix_peak_20d: Peak VIX in last 20 days
        portfolio_drawdown: Current portfolio drawdown (negative number)
        market_1m_return: Market 1-month return
        breadth_thrust: True if breadth thrust detected
        min_days_between: Minimum days between rebalances
        max_days_between: Maximum days between rebalances
        vix_recovery_decline: VIX decline % to trigger recovery rebalance
        vix_spike_threshold: VIX level considered a spike
        drawdown_threshold: Portfolio drawdown to trigger defensive rebalance
        crash_threshold: Market 1M return to trigger crash avoidance

    Returns:
        RebalanceTrigger with decision and reason
    """
    triggers_fired = []
    urgency = "LOW"

    # Check minimum interval
    if days_since_last < min_days_between:
        return RebalanceTrigger(
            should_rebalance=False,
            reason=f"Too soon ({days_since_last} days < {min_days_between} min)",
            days_since_last=days_since_last,
            triggers_fired=[],
            urgency="LOW",
        )

    # Trigger 1: Regular interval (force after max_days)
    if days_since_last >= max_days_between:
        triggers_fired.append("REGULAR_INTERVAL")
        urgency = "MEDIUM"

    # Trigger 2: Regime transition
    if current_regime is not None and previous_regime is not None:
        if current_regime != previous_regime:
            triggers_fired.append("REGIME_TRANSITION")
            urgency = "HIGH"

    # Trigger 3: VIX recovery from spike
    if vix_peak_20d >= vix_spike_threshold:
        vix_decline = (vix_peak_20d - vix_level) / vix_peak_20d
        if vix_decline >= vix_recovery_decline:
            triggers_fired.append("VIX_RECOVERY")
            urgency = "HIGH" if vix_decline >= 0.25 else "MEDIUM"

    # Trigger 4: Portfolio drawdown
    if portfolio_drawdown <= -drawdown_threshold:
        triggers_fired.append("PORTFOLIO_DRAWDOWN")
        urgency = "HIGH"

    # Trigger 5: Market crash (momentum crash avoidance)
    if market_1m_return <= crash_threshold:
        triggers_fired.append("MARKET_CRASH")
        urgency = "HIGH"

    # Trigger 6: Breadth thrust
    if breadth_thrust:
        triggers_fired.append("BREADTH_THRUST")
        urgency = "HIGH" if urgency != "HIGH" else urgency

    # Determine final decision
    should_rebalance = len(triggers_fired) > 0

    # Determine primary reason
    if not triggers_fired:
        reason = f"No triggers (waiting {max_days_between - days_since_last} more days)"
    elif "MARKET_CRASH" in triggers_fired:
        reason = "Market crash - momentum crash avoidance"
    elif "PORTFOLIO_DRAWDOWN" in triggers_fired:
        reason = f"Portfolio drawdown {portfolio_drawdown:.1%}"
    elif "REGIME_TRANSITION" in triggers_fired:
        reason = f"Regime changed: {previous_regime} → {current_regime}"
    elif "VIX_RECOVERY" in triggers_fired:
        reason = f"VIX recovery from spike ({vix_peak_20d:.1f} → {vix_level:.1f})"
    elif "BREADTH_THRUST" in triggers_fired:
        reason = "Breadth thrust - aggressive entry signal"
    elif "REGULAR_INTERVAL" in triggers_fired:
        reason = f"Regular interval ({days_since_last} days)"
    else:
        reason = "Unknown trigger"

    return RebalanceTrigger(
        should_rebalance=should_rebalance,
        reason=reason,
        days_since_last=days_since_last,
        triggers_fired=triggers_fired,
        urgency=urgency,
    )


def calculate_adaptive_lookback(
    base_lookback_6m: int,
    base_lookback_12m: int,
    portfolio_drawdown: float,
    vix_level: float,
    drawdown_threshold: float = 0.05,
    vix_threshold: float = 30.0,
    recovery_multiplier: float = 0.5,
    volatile_multiplier: float = 1.5,
) -> Tuple[int, int, str]:
    """
    Calculate adaptive lookback periods based on market conditions.

    Shortens lookback during recovery (to capture V-shaped rebounds faster)
    and lengthens during high volatility (to reduce whipsaws).

    Research sources:
    - Dynamic Momentum Learning (arXiv:2106.08420)
    - ReSolve Asset Management (half-life-of-optimal-lookback-horizon)

    Args:
        base_lookback_6m: Base 6-month lookback (default: 126)
        base_lookback_12m: Base 12-month lookback (default: 252)
        portfolio_drawdown: Current portfolio drawdown (negative number)
        vix_level: Current VIX level
        drawdown_threshold: Drawdown level to switch to recovery mode (5%)
        vix_threshold: VIX level to switch to volatile mode (30)
        recovery_multiplier: Multiplier for recovery mode (0.5 = 50% shorter)
        volatile_multiplier: Multiplier for volatile mode (1.5 = 50% longer)

    Returns:
        Tuple of (lookback_6m, lookback_12m, mode)
        Mode is one of: "NORMAL", "RECOVERY", "VOLATILE"
    """
    # Determine mode based on conditions
    if portfolio_drawdown <= -drawdown_threshold:
        # Recovery mode: shorten lookbacks to capture rebounds faster
        mode = "RECOVERY"
        lookback_6m = int(base_lookback_6m * recovery_multiplier)
        lookback_12m = int(base_lookback_12m * recovery_multiplier)
    elif vix_level >= vix_threshold:
        # Volatile mode: lengthen lookbacks to reduce whipsaws
        mode = "VOLATILE"
        lookback_6m = int(base_lookback_6m * volatile_multiplier)
        lookback_12m = int(base_lookback_12m * volatile_multiplier)
    else:
        # Normal mode: use base lookbacks
        mode = "NORMAL"
        lookback_6m = base_lookback_6m
        lookback_12m = base_lookback_12m

    # Ensure minimum and maximum bounds
    lookback_6m = max(42, min(252, lookback_6m))
    lookback_12m = max(63, min(378, lookback_12m))

    return (lookback_6m, lookback_12m, mode)
