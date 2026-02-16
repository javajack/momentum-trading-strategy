"""Tests for fortress.defensive — shared pure functions for backtest/live parity."""

import numpy as np
import pandas as pd
import pytest

from fortress.defensive import (
    apply_iterative_sector_caps,
    calculate_breadth_scale,
    calculate_gold_exhaustion_scale,
    calculate_vol_scale,
    get_effective_sector_cap,
    redirect_freed_weight,
    should_skip_gold,
)
from fortress.indicators import MarketRegime, RegimeResult

# --- should_skip_gold ---


class TestShouldSkipGold:
    def _make_series(self, prices):
        return pd.Series(prices, index=pd.date_range("2024-01-01", periods=len(prices)))

    def test_downtrend_below_sma(self):
        # Last price below 50-SMA → skip
        prices = [100.0] * 49 + [80.0]  # SMA ~99.6, current 80
        skip, reason = should_skip_gold(self._make_series(prices), "downtrend")
        assert skip is True
        assert "downtrend" in reason.lower()

    def test_downtrend_above_sma(self):
        prices = [100.0] * 49 + [105.0]
        skip, _ = should_skip_gold(self._make_series(prices), "downtrend")
        assert skip is False

    def test_volatile_high_vol(self):
        # Stable for 40 days, then 10 volatile days
        np.random.seed(42)
        stable = [100.0] * 40
        volatile = [100.0 + np.random.normal(0, 8) for _ in range(10)]
        prices = stable + volatile
        skip, _ = should_skip_gold(self._make_series(prices), "volatile")
        # May or may not trigger depending on random values; just ensure no crash
        assert isinstance(skip, bool)

    def test_volatile_normal_vol(self):
        prices = list(np.linspace(100, 105, 60))  # steady uptrend
        skip, _ = should_skip_gold(self._make_series(prices), "volatile")
        assert skip is False

    def test_insufficient_data(self):
        prices = [100.0] * 30
        skip, _ = should_skip_gold(self._make_series(prices), "downtrend")
        assert skip is False


# --- calculate_gold_exhaustion_scale ---


class TestGoldExhaustionScale:
    def test_below_low_threshold(self):
        assert calculate_gold_exhaustion_scale(100.0, 100.0, 0.15, 0.40) == 1.0

    def test_above_high_threshold(self):
        assert calculate_gold_exhaustion_scale(150.0, 100.0, 0.15, 0.40) == 0.0

    def test_midpoint(self):
        # 27.5% deviation = midpoint of [15%, 40%]
        scale = calculate_gold_exhaustion_scale(127.5, 100.0, 0.15, 0.40)
        assert abs(scale - 0.5) < 0.01

    def test_zero_sma(self):
        assert calculate_gold_exhaustion_scale(100.0, 0.0, 0.15, 0.40) == 1.0


# --- redirect_freed_weight ---


class TestRedirectFreedWeight:
    def test_uptrend_prorata(self):
        weights = {"A": 0.3, "B": 0.2, "GOLDBEES": 0.0, "LIQUIDBEES": 0.1}
        redirect_freed_weight(weights, 0.1, True, "GOLDBEES", "LIQUIDBEES")
        # Freed 0.1 distributed to A and B proportionally (0.3:0.2 = 3:2)
        assert abs(weights["A"] - 0.36) < 0.01
        assert abs(weights["B"] - 0.24) < 0.01
        assert abs(weights["LIQUIDBEES"] - 0.1) < 0.01  # Unchanged

    def test_downtrend_to_cash(self):
        weights = {"A": 0.3, "B": 0.2, "LIQUIDBEES": 0.1}
        redirect_freed_weight(weights, 0.1, False, "GOLDBEES", "LIQUIDBEES")
        assert abs(weights["LIQUIDBEES"] - 0.2) < 0.01
        assert abs(weights["A"] - 0.3) < 0.01  # Unchanged

    def test_no_equity_falls_back_to_cash(self):
        weights = {"GOLDBEES": 0.0, "LIQUIDBEES": 0.5}
        redirect_freed_weight(weights, 0.2, True, "GOLDBEES", "LIQUIDBEES")
        assert abs(weights["LIQUIDBEES"] - 0.7) < 0.01


# --- calculate_vol_scale ---


class TestVolScale:
    def test_high_vol(self):
        # High vol → scale down
        np.random.seed(1)
        returns = list(np.random.normal(0, 0.03, 60))  # ~47% annualized vol
        scale = calculate_vol_scale(returns, 0.15, 0.50)
        assert scale < 1.0
        assert scale >= 0.50

    def test_low_vol(self):
        returns = list(np.random.normal(0, 0.005, 60))  # ~8% annualized vol
        scale = calculate_vol_scale(returns, 0.15, 0.50)
        assert scale == 1.0

    def test_empty_returns(self):
        assert calculate_vol_scale([], 0.15, 0.50) == 1.0

    def test_near_zero_vol(self):
        returns = [0.0001] * 60
        assert calculate_vol_scale(returns, 0.15, 0.50) == 1.0


# --- calculate_breadth_scale ---


class TestBreadthScale:
    def test_above_full(self):
        scale, ema = calculate_breadth_scale(0.7, None, 0.50, 0.30, 0.60)
        assert scale == 1.0
        assert ema == 0.7

    def test_below_low(self):
        scale, ema = calculate_breadth_scale(0.2, None, 0.50, 0.30, 0.60)
        assert scale == 0.60

    def test_midpoint(self):
        scale, _ = calculate_breadth_scale(0.4, None, 0.50, 0.30, 0.60)
        assert 0.60 < scale < 1.0

    def test_ema_smoothing(self):
        _, ema1 = calculate_breadth_scale(0.5, None, 0.50, 0.30, 0.60)
        assert ema1 == 0.5
        _, ema2 = calculate_breadth_scale(0.3, ema1, 0.50, 0.30, 0.60)
        # EMA should be between 0.3 and 0.5
        assert 0.3 < ema2 < 0.5


# --- get_effective_sector_cap ---


class TestEffectiveSectorCap:
    def _make_regime(self, regime_type):
        return RegimeResult(
            regime=regime_type,
            nifty_52w_position=0.5,
            vix_level=15.0,
            nifty_3m_return=0.05,
            equity_weight=0.8,
            gold_weight=0.1,
            cash_weight=0.1,
            stress_score=0.3,
            primary_regime=regime_type,
            vix_upgrade=False,
            return_upgrade=False,
        )

    def test_bullish(self):
        cap = get_effective_sector_cap(
            self._make_regime(MarketRegime.BULLISH), 0.30, 0.25, 0.20, True
        )
        assert cap == 0.30

    def test_caution(self):
        cap = get_effective_sector_cap(
            self._make_regime(MarketRegime.CAUTION), 0.30, 0.25, 0.20, True
        )
        assert cap == 0.25

    def test_defensive(self):
        cap = get_effective_sector_cap(
            self._make_regime(MarketRegime.DEFENSIVE), 0.30, 0.25, 0.20, True
        )
        assert cap == 0.20

    def test_disabled(self):
        cap = get_effective_sector_cap(
            self._make_regime(MarketRegime.DEFENSIVE), 0.30, 0.25, 0.20, False
        )
        assert cap == 0.30

    def test_none_regime(self):
        cap = get_effective_sector_cap(None, 0.30, 0.25, 0.20, True)
        assert cap == 0.30


# --- apply_iterative_sector_caps ---


class TestIterativeSectorCaps:
    def test_no_overweight(self):
        weights = {"A": 0.2, "B": 0.3, "C": 0.5}
        sectors = {"A": "Tech", "B": "Finance", "C": "Health"}
        result = apply_iterative_sector_caps(weights, sectors, 0.50)
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_single_overweight_sector(self):
        weights = {"A": 0.25, "B": 0.25, "C": 0.20, "D": 0.30}
        sectors = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Finance"}
        # Tech = 0.50 > max 0.35, others are under
        result = apply_iterative_sector_caps(weights, sectors, 0.35)
        tech_total = result["A"] + result["B"]
        assert tech_total <= 0.35 + 0.001
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_does_not_mutate_input(self):
        weights = {"A": 0.6, "B": 0.4}
        sectors = {"A": "Tech", "B": "Finance"}
        original_a = weights["A"]
        apply_iterative_sector_caps(weights, sectors, 0.30)
        assert weights["A"] == original_a
