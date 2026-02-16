"""
Shared defensive allocation logic for FORTRESS MOMENTUM.

Pure functions used by both backtest and live engines to ensure parity.
Each function takes primitives (no self, no data providers) — callers
handle data fetching and pass pre-computed values.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import MarketRegime, RegimeResult


def should_skip_gold(gold_closes: pd.Series, skip_logic: str) -> Tuple[bool, str]:
    """
    Check if gold ETF should be skipped from defensive allocation.

    Args:
        gold_closes: Series of gold close prices (at least 50 data points)
        skip_logic: "downtrend" or "volatile"

    Returns:
        (should_skip, reason)
    """
    if len(gold_closes) < 50:
        return (False, "")

    if skip_logic == "downtrend":
        current_price = gold_closes.iloc[-1]
        sma_50 = gold_closes.iloc[-50:].mean()
        if current_price < sma_50:
            return (True, f"Gold downtrend: {current_price:.1f} < 50-SMA {sma_50:.1f}")
        return (False, "")
    else:
        # Legacy volatile check
        returns = gold_closes.pct_change().dropna()
        recent_vol = returns.iloc[-10:].std() * (252**0.5)
        avg_vol = returns.std() * (252**0.5)
        if recent_vol > avg_vol * 1.5 and recent_vol > 0.15:
            return (
                True,
                f"Gold volatile: recent_vol {recent_vol:.3f} > 1.5x avg {avg_vol:.3f}",
            )
        return (False, "")


def calculate_gold_exhaustion_scale(
    price: float, sma: float, threshold_low: float, threshold_high: float
) -> float:
    """
    Calculate gold exhaustion scaling factor (GE1).

    Linearly scales gold allocation from 1.0 to 0.0 as gold moves
    from threshold_low to threshold_high above its SMA.

    Args:
        price: Current gold price
        sma: Gold SMA value (e.g. 200-day)
        threshold_low: Deviation below which scale = 1.0
        threshold_high: Deviation above which scale = 0.0

    Returns:
        Float between 0.0 and 1.0
    """
    if sma <= 0:
        return 1.0

    deviation = (price - sma) / sma

    if deviation <= threshold_low:
        return 1.0
    elif deviation >= threshold_high:
        return 0.0
    else:
        span = threshold_high - threshold_low
        return 1.0 - (deviation - threshold_low) / span


def redirect_freed_weight(
    weights: Dict[str, float],
    freed: float,
    is_uptrend: bool,
    gold_symbol: str,
    cash_symbol: str,
) -> None:
    """
    Redirect freed gold weight to equities pro-rata (uptrend) or cash (downtrend).

    Mutates weights dict in-place.

    Args:
        weights: Current weight allocation (mutated)
        freed: Amount of freed weight to redistribute
        is_uptrend: Whether market is in structural uptrend
        gold_symbol: Gold ETF symbol (excluded from equity redistribution)
        cash_symbol: Cash ETF symbol (excluded from equity redistribution)
    """
    if is_uptrend:
        defensive = {gold_symbol, cash_symbol}
        equity_weights = {t: w for t, w in weights.items() if t not in defensive and w > 0}
        total_eq = sum(equity_weights.values())
        if total_eq > 0:
            for t, w in equity_weights.items():
                weights[t] += freed * (w / total_eq)
            return
    weights[cash_symbol] = weights.get(cash_symbol, 0.0) + freed


def calculate_vol_scale(
    recent_returns: List[float], target_vol: float, vol_scale_floor: float
) -> float:
    """
    Calculate portfolio-level volatility scaling factor (E2).

    Scales equity inversely to realized portfolio vol:
    vol_scale = clamp(target_vol / realized_vol, floor, 1.0)

    Args:
        recent_returns: Pre-sliced list of recent daily returns
        target_vol: Target portfolio volatility (annualized)
        vol_scale_floor: Minimum scale factor

    Returns:
        Float between vol_scale_floor and 1.0
    """
    if not recent_returns:
        return 1.0

    realized_vol = np.std(recent_returns) * (252**0.5)

    if realized_vol < 0.01:
        return 1.0

    raw_scale = target_vol / realized_vol
    return min(1.0, max(vol_scale_floor, raw_scale))


def calculate_breadth_scale(
    raw_breadth: float,
    breadth_ema: Optional[float],
    breadth_full: float,
    breadth_low: float,
    breadth_min_scale: float,
) -> Tuple[float, float]:
    """
    Calculate breadth-based exposure scaling factor (E3) with 5-day EMA smoothing.

    Args:
        raw_breadth: Raw market breadth (fraction of stocks above 50-day MA)
        breadth_ema: Previous EMA value (None for first call)
        breadth_full: Breadth threshold for scale = 1.0
        breadth_low: Breadth threshold for scale = min_scale
        breadth_min_scale: Minimum breadth scale factor

    Returns:
        (scale, updated_ema) — caller stores the EMA
    """
    # 5-day EMA smoothing (FIX 8)
    alpha = 2.0 / (5 + 1)
    if breadth_ema is None:
        updated_ema = raw_breadth
    else:
        updated_ema = alpha * raw_breadth + (1 - alpha) * breadth_ema

    breadth = updated_ema

    if breadth >= breadth_full:
        scale = 1.0
    elif breadth <= breadth_low:
        scale = breadth_min_scale
    else:
        t = (breadth - breadth_low) / (breadth_full - breadth_low)
        scale = breadth_min_scale + t * (1.0 - breadth_min_scale)

    return scale, updated_ema


def get_effective_sector_cap(
    regime: Optional[RegimeResult],
    max_sector: float,
    caution_max: float,
    defensive_max: float,
    use_dynamic: bool,
) -> float:
    """
    Get effective sector cap based on market regime (E4).

    Args:
        regime: Current regime result (None → use default)
        max_sector: Default max sector exposure (BULLISH/NORMAL)
        caution_max: Max sector exposure in CAUTION regime
        defensive_max: Max sector exposure in DEFENSIVE regime
        use_dynamic: Whether dynamic sector caps are enabled

    Returns:
        Float representing maximum sector weight
    """
    if not use_dynamic or regime is None:
        return max_sector

    if regime.regime == MarketRegime.DEFENSIVE:
        return defensive_max
    elif regime.regime == MarketRegime.CAUTION:
        return caution_max
    else:
        return max_sector


def apply_iterative_sector_caps(
    weights: Dict[str, float],
    ticker_sectors: Dict[str, str],
    max_sector: float,
) -> Dict[str, float]:
    """
    Iteratively cap overweight sectors and redistribute to uncapped sectors.

    Args:
        weights: Ticker → weight mapping (not mutated)
        ticker_sectors: Ticker → sector mapping
        max_sector: Maximum weight per sector

    Returns:
        New weights dict respecting sector limits
    """
    adjusted = dict(weights)
    capped_sectors: set = set()

    for _ in range(10):
        # Calculate current sector weights
        sector_weights: Dict[str, float] = {}
        for ticker, weight in adjusted.items():
            sector = ticker_sectors.get(ticker, "UNKNOWN")
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        # Find newly overweight sectors
        newly_capped = False
        for sector, sw in sector_weights.items():
            if sector in capped_sectors or sw <= max_sector:
                continue
            capped_sectors.add(sector)
            scale = max_sector / sw
            for ticker in adjusted:
                if ticker_sectors.get(ticker) == sector:
                    adjusted[ticker] *= scale
            newly_capped = True

        if not newly_capped:
            break

        # Redistribute excess to uncapped tickers
        capped_total = sum(
            w for t, w in adjusted.items() if ticker_sectors.get(t) in capped_sectors
        )
        uncapped_total = sum(
            w for t, w in adjusted.items() if ticker_sectors.get(t) not in capped_sectors
        )
        target_uncapped = 1.0 - capped_total
        if uncapped_total > 0 and target_uncapped > 0:
            scale = target_uncapped / uncapped_total
            for ticker in adjusted:
                if ticker_sectors.get(ticker) not in capped_sectors:
                    adjusted[ticker] *= scale

    return adjusted
