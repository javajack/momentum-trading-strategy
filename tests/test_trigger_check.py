"""
Tests for dynamic rebalancing trigger evaluation and stale regime handling.

Covers:
- should_trigger_rebalance() core logic
- Stale regime neutralization (previous_regime = None when days > max)
- Regime transition fires only with fresh data
"""

import pytest

from fortress.indicators import MarketRegime, should_trigger_rebalance

# --- Defaults shared across tests ---

DEFAULTS = dict(
    vix_level=15.0,
    vix_peak_20d=15.0,
    portfolio_drawdown=0.0,
    market_1m_return=0.0,
    breadth_thrust=False,
    min_days_between=5,
    max_days_between=30,
)


class TestRegimeTransition:
    """Regime transition trigger fires only when both regimes are non-None and differ."""

    def test_regime_transition_fires_when_changed(self):
        result = should_trigger_rebalance(
            days_since_last=10,
            current_regime=MarketRegime.CAUTION,
            previous_regime=MarketRegime.BULLISH,
            **DEFAULTS,
        )
        assert result.should_rebalance
        assert "REGIME_TRANSITION" in result.triggers_fired

    def test_no_transition_when_same_regime(self):
        result = should_trigger_rebalance(
            days_since_last=10,
            current_regime=MarketRegime.BULLISH,
            previous_regime=MarketRegime.BULLISH,
            **DEFAULTS,
        )
        assert not result.should_rebalance
        assert "REGIME_TRANSITION" not in result.triggers_fired

    def test_no_transition_when_previous_is_none(self):
        """Core of stale regime handling — None previous skips transition check."""
        result = should_trigger_rebalance(
            days_since_last=10,
            current_regime=MarketRegime.CAUTION,
            previous_regime=None,
            **DEFAULTS,
        )
        assert "REGIME_TRANSITION" not in result.triggers_fired

    def test_no_transition_when_current_is_none(self):
        result = should_trigger_rebalance(
            days_since_last=10,
            current_regime=None,
            previous_regime=MarketRegime.BULLISH,
            **DEFAULTS,
        )
        assert "REGIME_TRANSITION" not in result.triggers_fired

    def test_no_transition_when_both_none(self):
        result = should_trigger_rebalance(
            days_since_last=10,
            current_regime=None,
            previous_regime=None,
            **DEFAULTS,
        )
        assert "REGIME_TRANSITION" not in result.triggers_fired


class TestRegularInterval:
    """Regular interval trigger fires when days >= max_days_between."""

    def test_fires_at_max_days(self):
        result = should_trigger_rebalance(
            days_since_last=30,
            current_regime=MarketRegime.BULLISH,
            previous_regime=MarketRegime.BULLISH,
            **DEFAULTS,
        )
        assert result.should_rebalance
        assert "REGULAR_INTERVAL" in result.triggers_fired

    def test_fires_beyond_max_days(self):
        result = should_trigger_rebalance(
            days_since_last=63,
            current_regime=MarketRegime.BULLISH,
            previous_regime=MarketRegime.BULLISH,
            **DEFAULTS,
        )
        assert result.should_rebalance
        assert "REGULAR_INTERVAL" in result.triggers_fired

    def test_does_not_fire_before_max(self):
        result = should_trigger_rebalance(
            days_since_last=20,
            current_regime=MarketRegime.BULLISH,
            previous_regime=MarketRegime.BULLISH,
            **DEFAULTS,
        )
        assert "REGULAR_INTERVAL" not in result.triggers_fired


class TestMinimumInterval:
    """Cooldown: no triggers fire if days < min_days_between."""

    def test_too_soon_blocks_all_triggers(self):
        result = should_trigger_rebalance(
            days_since_last=3,
            current_regime=MarketRegime.DEFENSIVE,
            previous_regime=MarketRegime.BULLISH,
            vix_level=15.0,
            vix_peak_20d=15.0,
            portfolio_drawdown=-0.15,
            market_1m_return=-0.12,
            breadth_thrust=True,
            min_days_between=5,
            max_days_between=30,
        )
        assert not result.should_rebalance
        assert result.triggers_fired == []


class TestStaleRegimeNeutralization:
    """
    Simulate the stale regime pattern from _do_trigger_check():
    when days_since_last > max_days_between, previous_regime is set to None
    before calling should_trigger_rebalance().
    """

    def test_stale_regime_only_fires_regular_interval(self):
        """63 days gap, regime changed: only REGULAR_INTERVAL fires, not REGIME_TRANSITION."""
        days_since_last = 63
        max_days = 30

        # Simulate cli.py stale logic: neutralize previous_regime
        previous_regime = MarketRegime.BULLISH
        if days_since_last > max_days:
            previous_regime = None

        result = should_trigger_rebalance(
            days_since_last=days_since_last,
            current_regime=MarketRegime.CAUTION,
            previous_regime=previous_regime,
            **{**DEFAULTS, "max_days_between": max_days},
        )
        assert result.should_rebalance
        assert "REGULAR_INTERVAL" in result.triggers_fired
        assert "REGIME_TRANSITION" not in result.triggers_fired

    def test_stale_regime_without_neutralization_would_fire_transition(self):
        """Proves the bug: without neutralization, a spurious REGIME_TRANSITION fires."""
        result = should_trigger_rebalance(
            days_since_last=63,
            current_regime=MarketRegime.CAUTION,
            previous_regime=MarketRegime.BULLISH,
            **{**DEFAULTS, "max_days_between": 30},
        )
        assert "REGIME_TRANSITION" in result.triggers_fired
        assert "REGULAR_INTERVAL" in result.triggers_fired

    def test_fresh_regime_transition_still_fires(self):
        """Within max_days, regime transition fires normally."""
        days_since_last = 10
        max_days = 30

        # Not stale — previous_regime preserved
        previous_regime = MarketRegime.BULLISH
        if days_since_last > max_days:
            previous_regime = None

        result = should_trigger_rebalance(
            days_since_last=days_since_last,
            current_regime=MarketRegime.CAUTION,
            previous_regime=previous_regime,
            **{**DEFAULTS, "max_days_between": max_days},
        )
        assert "REGIME_TRANSITION" in result.triggers_fired
        assert "REGULAR_INTERVAL" not in result.triggers_fired

    def test_exactly_at_max_days_not_stale(self):
        """At exactly max_days_between, regime is NOT stale (only > is stale)."""
        days_since_last = 30
        max_days = 30

        previous_regime = MarketRegime.BULLISH
        if days_since_last > max_days:
            previous_regime = None

        result = should_trigger_rebalance(
            days_since_last=days_since_last,
            current_regime=MarketRegime.CAUTION,
            previous_regime=previous_regime,
            **{**DEFAULTS, "max_days_between": max_days},
        )
        # Both fire: REGULAR_INTERVAL because days == max, REGIME_TRANSITION because not stale
        assert "REGULAR_INTERVAL" in result.triggers_fired
        assert "REGIME_TRANSITION" in result.triggers_fired

    def test_one_day_past_max_is_stale(self):
        """At max_days + 1, regime IS stale — only REGULAR_INTERVAL fires."""
        days_since_last = 31
        max_days = 30

        previous_regime = MarketRegime.DEFENSIVE
        if days_since_last > max_days:
            previous_regime = None

        result = should_trigger_rebalance(
            days_since_last=days_since_last,
            current_regime=MarketRegime.BULLISH,
            previous_regime=previous_regime,
            **{**DEFAULTS, "max_days_between": max_days},
        )
        assert "REGULAR_INTERVAL" in result.triggers_fired
        assert "REGIME_TRANSITION" not in result.triggers_fired

    def test_stale_with_no_stored_regime(self):
        """No previous regime stored at all — same outcome, no transition."""
        days_since_last = 45

        result = should_trigger_rebalance(
            days_since_last=days_since_last,
            current_regime=MarketRegime.NORMAL,
            previous_regime=None,
            **{**DEFAULTS, "max_days_between": 30},
        )
        assert result.should_rebalance
        assert "REGULAR_INTERVAL" in result.triggers_fired
        assert "REGIME_TRANSITION" not in result.triggers_fired


class TestStaleRegimeDisplay:
    """Test the display string logic from _do_trigger_check()."""

    @staticmethod
    def _build_regime_display(
        current_regime_enum, previous_regime, last_regime_str, days_since_last, max_days_between
    ):
        """Replicate the display logic from cli.py for testing."""
        regime_stale = days_since_last > max_days_between
        if regime_stale:
            previous_regime = None

        regime_str = current_regime_enum.value.upper() if current_regime_enum else "UNKNOWN"
        prev_str = previous_regime.value.upper() if previous_regime else "—"

        if regime_stale and last_regime_str:
            regime_display = (
                f"{regime_str} (last: {last_regime_str.upper()}, {days_since_last}d ago — stale)"
            )
        elif prev_str != "—":
            regime_display = f"{regime_str} (was {prev_str})"
        else:
            regime_display = regime_str

        return regime_display, regime_stale

    def test_fresh_display_shows_was(self):
        display, stale = self._build_regime_display(
            MarketRegime.CAUTION, MarketRegime.BULLISH, "bullish", 10, 30
        )
        assert display == "CAUTION (was BULLISH)"
        assert not stale

    def test_stale_display_shows_last_with_age(self):
        display, stale = self._build_regime_display(
            MarketRegime.CAUTION, MarketRegime.BULLISH, "bullish", 63, 30
        )
        assert display == "CAUTION (last: BULLISH, 63d ago — stale)"
        assert stale

    def test_no_previous_regime_shows_plain(self):
        display, stale = self._build_regime_display(MarketRegime.NORMAL, None, None, 10, 30)
        assert display == "NORMAL"
        assert not stale

    def test_stale_but_no_stored_regime_shows_plain(self):
        """Stale + no stored regime = just show current regime (no misleading label)."""
        display, stale = self._build_regime_display(MarketRegime.BULLISH, None, None, 45, 30)
        assert display == "BULLISH"
        assert stale

    def test_exactly_at_max_is_fresh(self):
        display, stale = self._build_regime_display(
            MarketRegime.DEFENSIVE, MarketRegime.NORMAL, "normal", 30, 30
        )
        assert display == "DEFENSIVE (was NORMAL)"
        assert not stale


class TestPortfolioMomentumTrigger:
    """Portfolio momentum deterioration trigger (Trigger 7)."""

    def test_fires_when_momentum_below_threshold(self):
        result = should_trigger_rebalance(
            days_since_last=10,
            current_regime=MarketRegime.BULLISH,
            previous_regime=MarketRegime.BULLISH,
            portfolio_momentum_return=-0.06,
            portfolio_momentum_threshold=-0.05,
            **DEFAULTS,
        )
        assert result.should_rebalance
        assert "PORTFOLIO_MOMENTUM" in result.triggers_fired
        assert "momentum" in result.reason.lower()

    def test_does_not_fire_above_threshold(self):
        result = should_trigger_rebalance(
            days_since_last=10,
            current_regime=MarketRegime.BULLISH,
            previous_regime=MarketRegime.BULLISH,
            portfolio_momentum_return=-0.03,
            portfolio_momentum_threshold=-0.05,
            **DEFAULTS,
        )
        assert not result.should_rebalance
        assert "PORTFOLIO_MOMENTUM" not in result.triggers_fired

    def test_does_not_fire_when_none(self):
        """No daily return data available — trigger should not fire."""
        result = should_trigger_rebalance(
            days_since_last=10,
            current_regime=MarketRegime.BULLISH,
            previous_regime=MarketRegime.BULLISH,
            portfolio_momentum_return=None,
            portfolio_momentum_threshold=-0.05,
            **DEFAULTS,
        )
        assert not result.should_rebalance
        assert "PORTFOLIO_MOMENTUM" not in result.triggers_fired

    def test_urgency_is_medium(self):
        result = should_trigger_rebalance(
            days_since_last=10,
            current_regime=MarketRegime.BULLISH,
            previous_regime=MarketRegime.BULLISH,
            portfolio_momentum_return=-0.08,
            portfolio_momentum_threshold=-0.05,
            **DEFAULTS,
        )
        assert result.urgency == "MEDIUM"

    def test_blocked_by_min_days(self):
        """Even with bad momentum, min_days_between should block."""
        result = should_trigger_rebalance(
            days_since_last=3,
            current_regime=MarketRegime.BULLISH,
            previous_regime=MarketRegime.BULLISH,
            portfolio_momentum_return=-0.10,
            portfolio_momentum_threshold=-0.05,
            **{**DEFAULTS, "min_days_between": 5},
        )
        assert not result.should_rebalance

    def test_exactly_at_threshold(self):
        result = should_trigger_rebalance(
            days_since_last=10,
            current_regime=MarketRegime.BULLISH,
            previous_regime=MarketRegime.BULLISH,
            portfolio_momentum_return=-0.05,
            portfolio_momentum_threshold=-0.05,
            **DEFAULTS,
        )
        assert result.should_rebalance
        assert "PORTFOLIO_MOMENTUM" in result.triggers_fired

    def test_does_not_override_high_urgency(self):
        """When crash also fires (HIGH urgency), momentum shouldn't downgrade it."""
        result = should_trigger_rebalance(
            days_since_last=10,
            current_regime=MarketRegime.BULLISH,
            previous_regime=MarketRegime.BULLISH,
            portfolio_momentum_return=-0.08,
            portfolio_momentum_threshold=-0.05,
            **{**DEFAULTS, "market_1m_return": -0.12},
        )
        assert "MARKET_CRASH" in result.triggers_fired
        assert "PORTFOLIO_MOMENTUM" in result.triggers_fired
        assert result.urgency == "HIGH"
