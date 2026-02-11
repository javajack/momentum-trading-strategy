"""
Tests for risk governor module.

Verifies invariants R1-R10 for pure momentum strategy.
"""

import pytest
from datetime import datetime

from fortress.risk_governor import RiskGovernor, RiskCheckResult, StopLossTracker, StopLossEntry
from fortress.config import RiskConfig, PortfolioConfig


@pytest.fixture
def risk_governor():
    """Create risk governor with default settings."""
    return RiskGovernor()


@pytest.fixture
def custom_governor():
    """Create risk governor with custom settings."""
    risk = RiskConfig(
        max_single_position=0.08,
        hard_max_position=0.10,
        max_sector_exposure=0.25,
        hard_max_sector=0.30,
        max_drawdown_warning=0.10,
        max_drawdown_halt=0.20,
        daily_loss_limit=0.03,
        initial_stop_loss=0.18,
        trailing_stop=0.15,
        trailing_activation=0.08,
    )
    portfolio = PortfolioConfig(max_positions=20)
    return RiskGovernor(risk_config=risk, portfolio_config=portfolio)


class TestPositionLimits:
    """Test position size limits (R1)."""

    def test_position_within_soft_limit(self, risk_governor):
        """Positions <= 8% should pass."""
        result = risk_governor.validate_position_size(
            symbol="RELIANCE",
            proposed_value=80000,
            portfolio_value=1000000,
        )
        assert result.passed
        assert result.reason == "OK"

    def test_position_exceeds_soft_limit(self, risk_governor):
        """Positions > 8% but <= 12% should fail with adjustment."""
        result = risk_governor.validate_position_size(
            symbol="RELIANCE",
            proposed_value=100000,  # 10%
            portfolio_value=1000000,
        )
        assert not result.passed
        assert "soft limit" in result.reason.lower()
        assert result.adjusted_value == 80000  # 8% of 1M

    def test_position_exceeds_hard_limit(self, risk_governor):
        """R1: Positions > 12% should fail hard."""
        result = risk_governor.validate_position_size(
            symbol="RELIANCE",
            proposed_value=150000,  # 15%
            portfolio_value=1000000,
        )
        assert not result.passed
        assert "R1" in result.reason
        assert result.adjusted_value == 120000  # 12% of 1M


class TestSectorLimits:
    """Test sector concentration limits (R2)."""

    def test_sector_within_soft_limit(self, risk_governor):
        """Sector exposure <= 35% should pass."""
        result = risk_governor.validate_sector_exposure(
            sector="IT_SERVICES",
            current_exposure=200000,
            proposed_addition=100000,
            portfolio_value=1000000,
        )
        assert result.passed

    def test_sector_exceeds_soft_limit(self, risk_governor):
        """Sector > 35% but <= 45% should fail with adjustment."""
        result = risk_governor.validate_sector_exposure(
            sector="IT_SERVICES",
            current_exposure=300000,
            proposed_addition=100000,  # Would be 40%
            portfolio_value=1000000,
        )
        assert not result.passed
        assert "soft limit" in result.reason.lower()
        assert result.adjusted_value == 50000  # 35% - 30%

    def test_sector_exceeds_hard_limit(self, risk_governor):
        """R2: Sector > 45% should fail hard."""
        result = risk_governor.validate_sector_exposure(
            sector="IT_SERVICES",
            current_exposure=400000,
            proposed_addition=100000,  # Would be 50%
            portfolio_value=1000000,
        )
        assert not result.passed
        assert "R2" in result.reason
        assert result.adjusted_value == 50000  # 45% - 40%


class TestDailyLoss:
    """Test daily loss limit (R3)."""

    def test_daily_loss_within_limit(self, risk_governor):
        """Daily loss < 3% should pass."""
        risk_governor.set_day_start_value(1000000)
        result = risk_governor.check_daily_loss(980000)  # -2%
        assert result.passed

    def test_daily_loss_exceeds_limit(self, risk_governor):
        """R3: Daily loss >= 3% should halt."""
        risk_governor.set_day_start_value(1000000)
        result = risk_governor.check_daily_loss(960000)  # -4%
        assert not result.passed
        assert "R3" in result.reason

    def test_daily_loss_exact_limit(self, risk_governor):
        """Daily loss exactly at 3% should halt."""
        risk_governor.set_day_start_value(1000000)
        result = risk_governor.check_daily_loss(970000)  # -3%
        assert not result.passed


class TestPositionCount:
    """Test position count limit (R7)."""

    def test_within_position_limit(self, risk_governor):
        """Position count <= 20 should pass."""
        result = risk_governor.check_position_count(15, 3)
        assert result.passed

    def test_exceeds_position_limit(self, risk_governor):
        """R7: Position count > 20 should fail."""
        result = risk_governor.check_position_count(18, 5)
        assert not result.passed
        assert "R7" in result.reason


class TestStopLoss:
    """Test stop loss tracking (R9, R10)."""

    def test_initial_stop_loss_triggered(self, custom_governor):
        """R9: Initial stop loss at -18%."""
        custom_governor.register_stop_loss("RELIANCE", 100.0, datetime.now())

        # Check stops at various prices
        triggered = custom_governor.check_stop_losses({"RELIANCE": 85.0})  # -15%, not triggered
        assert len(triggered) == 0

        triggered = custom_governor.check_stop_losses({"RELIANCE": 81.0})  # -19%, triggered
        assert len(triggered) == 1
        assert triggered[0][0] == "RELIANCE"
        assert "Initial stop" in triggered[0][1]

    def test_trailing_stop_activated(self, custom_governor):
        """R10: Trailing stop activated after +8% gain."""
        custom_governor.register_stop_loss("RELIANCE", 100.0, datetime.now())

        # Price rises 10% - should activate trailing stop
        custom_governor.check_stop_losses({"RELIANCE": 110.0})

        entry = custom_governor.get_stop_loss_entry("RELIANCE")
        assert entry.trailing_activated
        assert entry.peak_price == 110.0

    def test_trailing_stop_triggered(self, custom_governor):
        """R10: Trailing stop triggered after -15% from peak."""
        custom_governor.register_stop_loss("RELIANCE", 100.0, datetime.now())

        # Price rises 20% then falls
        custom_governor.check_stop_losses({"RELIANCE": 120.0})  # Peak at 120

        # Falls to 100 - that's 16.7% from peak, should trigger trailing stop
        triggered = custom_governor.check_stop_losses({"RELIANCE": 100.0})
        assert len(triggered) == 1
        assert "Trailing stop" in triggered[0][1]

    def test_stop_loss_removal(self, custom_governor):
        """Stop loss removed when position closed."""
        custom_governor.register_stop_loss("RELIANCE", 100.0, datetime.now())
        custom_governor.remove_stop_loss("RELIANCE")

        triggered = custom_governor.check_stop_losses({"RELIANCE": 50.0})
        assert len(triggered) == 0


class TestCanTrade:
    """Test risk governor veto (R8)."""

    def test_can_trade_normal(self, risk_governor):
        """Can trade in normal conditions."""
        risk_governor.set_day_start_value(1000000)
        can_trade, reason = risk_governor.can_trade(
            current_value=990000,
            current_drawdown=-0.05,
        )
        assert can_trade
        assert reason == "OK"

    def test_cannot_trade_daily_loss(self, risk_governor):
        """R8: Cannot trade after daily loss limit."""
        risk_governor.set_day_start_value(1000000)
        can_trade, reason = risk_governor.can_trade(
            current_value=960000,  # -4%
            current_drawdown=-0.05,
        )
        assert not can_trade
        assert "R3" in reason

    def test_cannot_trade_drawdown_halt(self, risk_governor):
        """R8: Cannot trade when drawdown exceeds halt level."""
        risk_governor.set_day_start_value(1000000)
        can_trade, reason = risk_governor.can_trade(
            current_value=990000,
            current_drawdown=-0.26,  # 26% drawdown
        )
        assert not can_trade
        assert "halted" in reason.lower()


class TestComprehensiveValidation:
    """Test comprehensive order validation."""

    def test_sell_always_allowed(self, risk_governor):
        """Sells should always be allowed (reduce risk)."""
        result = risk_governor.validate_order(
            symbol="RELIANCE",
            sector="OIL_GAS_ENERGY",
            order_value=200000,
            current_position_value=200000,
            current_sector_value=500000,
            portfolio_value=1000000,
            current_positions=20,
            is_buy=False,
        )
        assert result.passed

    def test_buy_validated_fully(self, risk_governor):
        """Buy orders should pass all checks."""
        result = risk_governor.validate_order(
            symbol="TCS",
            sector="IT_SERVICES",
            order_value=50000,
            current_position_value=0,
            current_sector_value=100000,
            portfolio_value=1000000,
            current_positions=10,
            is_buy=True,
        )
        assert result.passed

    def test_buy_blocked_by_position_limit(self, risk_governor):
        """Buy blocked if would exceed position limit."""
        result = risk_governor.validate_order(
            symbol="TCS",
            sector="IT_SERVICES",
            order_value=100000,
            current_position_value=50000,  # Would be 15%
            current_sector_value=100000,
            portfolio_value=1000000,
            current_positions=10,
            is_buy=True,
        )
        assert not result.passed
