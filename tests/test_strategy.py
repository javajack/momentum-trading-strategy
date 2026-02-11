"""
Tests for the pluggable strategy architecture.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from fortress.strategy import (
    StrategyRegistry,
    BaseStrategy,
    AdaptiveDualMomentumStrategy,
    StockScore,
    ExitSignal,
    StopLossConfig,
)
from fortress.config import Config, load_config


class TestStrategyRegistry:
    """Tests for the StrategyRegistry."""

    def test_registry_has_strategies(self):
        """Registry should have the dual_momentum strategy registered."""
        names = StrategyRegistry.get_names()
        assert "dual_momentum" in names

    def test_list_strategies(self):
        """list_strategies should return name/description tuples."""
        strategies = StrategyRegistry.list_strategies()
        assert len(strategies) >= 1
        for name, desc in strategies:
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_get_dual_momentum_strategy(self):
        """Should be able to get dual_momentum strategy."""
        strategy = StrategyRegistry.get("dual_momentum")
        assert strategy is not None
        assert strategy.name == "dual_momentum"
        assert isinstance(strategy, AdaptiveDualMomentumStrategy)

    def test_get_unknown_strategy_raises(self):
        """Getting unknown strategy should raise ValueError."""
        with pytest.raises(ValueError) as exc:
            StrategyRegistry.get("unknown_strategy")
        assert "unknown_strategy" in str(exc.value)
        assert "Available" in str(exc.value)

    def test_is_registered(self):
        """is_registered should work correctly."""
        assert StrategyRegistry.is_registered("dual_momentum")
        assert not StrategyRegistry.is_registered("fake_strategy")


class TestAdaptiveDualMomentumStrategy:
    """Tests for AdaptiveDualMomentumStrategy."""

    @pytest.fixture
    def strategy(self):
        return AdaptiveDualMomentumStrategy()

    def test_name_and_description(self, strategy):
        """Strategy should have correct name and description."""
        assert strategy.name == "dual_momentum"
        assert "momentum" in strategy.description.lower()

    def test_stop_loss_config_is_tiered(self, strategy):
        """Strategy should return tiered stops."""
        config1 = strategy.get_stop_loss_config("TEST", 0.05)  # tier1
        config2 = strategy.get_stop_loss_config("TEST", 0.25)  # tier3
        config3 = strategy.get_stop_loss_config("TEST", 0.60)  # tier4

        # Strategy uses tiered stops
        assert config1.use_tiered
        assert config2.use_tiered
        assert config3.use_tiered

        # Different tiers should have different trailing stops
        assert config1.tiers["current_tier"] == "tier1"
        assert config2.tiers["current_tier"] == "tier3"
        assert config3.tiers["current_tier"] == "tier4"

    def test_exit_signal_stop_loss(self, strategy):
        """Should trigger stop loss exit."""
        signal = strategy.check_exit_triggers(
            ticker="TEST",
            entry_price=100,
            current_price=80,  # -20% loss
            peak_price=100,
            days_held=10,
            stock_score=None,
            nms_percentile=75,
        )
        assert signal.should_exit
        assert signal.exit_type == "stop_loss"

    def test_exit_signal_trailing_stop(self, strategy):
        """Should trigger trailing stop exit."""
        # Trailing stop requires current_gain >= 8% AND drop from peak >= trailing%
        signal = strategy.check_exit_triggers(
            ticker="TEST",
            entry_price=100,
            current_price=109,  # 9% above entry (above 8% activation)
            peak_price=130,     # 30% gain at peak, now down 16% from peak
            days_held=30,
            stock_score=None,
            nms_percentile=75,
        )
        assert signal.should_exit
        assert signal.exit_type == "trailing_stop"

    def test_exit_signal_rs_floor(self, strategy):
        """Should trigger RS floor exit when RS drops below threshold."""
        # Strategy uses RS floor exit
        # This requires stock_score with rs_composite metric
        mock_score = StockScore(
            ticker="TEST",
            sector="TECHNOLOGY",
            sub_sector="Software",
            zerodha_symbol="TEST",
            name="Test Company",
            score=1.0,
            extra_metrics={"rs_composite": 0.90},  # Below 0.95 threshold
        )
        signal = strategy.check_exit_triggers(
            ticker="TEST",
            entry_price=100,
            current_price=110,
            peak_price=110,
            days_held=30,
            stock_score=mock_score,
            nms_percentile=75,
        )
        assert signal.should_exit
        assert signal.exit_type == "rs_floor"

    def test_exit_signal_ok(self, strategy):
        """Should not exit when conditions are good."""
        signal = strategy.check_exit_triggers(
            ticker="TEST",
            entry_price=100,
            current_price=115,
            peak_price=118,
            days_held=20,
            stock_score=None,
            nms_percentile=85,
        )
        assert not signal.should_exit
        assert signal.exit_type == "none"


class TestStockScore:
    """Tests for StockScore dataclass."""

    def test_create_stock_score(self):
        """Should be able to create StockScore."""
        score = StockScore(
            ticker="TEST",
            sector="TECHNOLOGY",
            sub_sector="Software",
            zerodha_symbol="TEST",
            name="Test Company",
            score=1.5,
            rank=1,
            percentile=99.0,
            passes_entry_filters=True,
            filter_reasons=[],
            return_6m=0.20,
            return_12m=0.35,
        )
        assert score.ticker == "TEST"
        assert score.score == 1.5
        assert score.passes_entry_filters

    def test_extra_metrics_default(self):
        """Extra metrics should default to empty dict."""
        score = StockScore(
            ticker="TEST",
            sector="TECHNOLOGY",
            sub_sector="Software",
            zerodha_symbol="TEST",
            name="Test Company",
            score=1.0,
        )
        assert score.extra_metrics == {}


class TestExitSignal:
    """Tests for ExitSignal dataclass."""

    def test_create_exit_signal(self):
        """Should be able to create ExitSignal."""
        signal = ExitSignal(
            should_exit=True,
            reason="Test reason",
            exit_type="stop_loss",
            urgency="immediate",
        )
        assert signal.should_exit
        assert signal.exit_type == "stop_loss"
        assert signal.urgency == "immediate"


class TestStopLossConfig:
    """Tests for StopLossConfig dataclass."""

    def test_create_basic_config(self):
        """Should be able to create basic stop loss config."""
        config = StopLossConfig(
            initial_stop=0.18,
            trailing_stop=0.15,
            trailing_activation=0.08,
        )
        assert config.initial_stop == 0.18
        assert not config.use_tiered

    def test_create_tiered_config(self):
        """Should be able to create tiered stop loss config."""
        config = StopLossConfig(
            initial_stop=0.18,
            trailing_stop=0.25,
            trailing_activation=0.08,
            use_tiered=True,
            tiers={"tier1": 0.15, "tier4": 0.25},
        )
        assert config.use_tiered
        assert config.tiers["tier1"] == 0.15


class TestConfigIntegration:
    """Tests for config integration with strategies."""

    def test_config_has_strategy_fields(self):
        """Config should have strategy-related fields."""
        config = load_config("config.yaml")
        assert hasattr(config, "active_strategy")
        assert hasattr(config, "strategy_dual_momentum")

    def test_strategy_with_config(self):
        """Strategies should work with config."""
        config = load_config("config.yaml")
        dual_momentum = StrategyRegistry.get("dual_momentum", config)

        assert dual_momentum is not None
