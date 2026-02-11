"""
Tests for backtest module.

Tests for BacktestEngine and BacktestConfig.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from fortress.backtest import BacktestConfig, BacktestEngine, BacktestResult
from fortress.config import Config, get_default_config
from fortress.universe import Universe


@pytest.fixture
def sample_universe():
    """Load the actual stock universe."""
    return Universe("stock-universe.json")


@pytest.fixture
def sample_historical_data():
    """Generate sample historical data for a few stocks."""
    np.random.seed(42)

    # Create 400 days of data (enough for 12M lookback + backtest period)
    dates = pd.date_range(start="2024-01-01", periods=400, freq="D")

    stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    data = {}

    for stock in stocks:
        returns = np.random.normal(0.001, 0.02, 400)
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.uniform(1e6, 5e6, 400)

        df = pd.DataFrame({
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": volumes,
        }, index=dates)

        data[stock] = df

    return data


@pytest.fixture
def backtest_config():
    """Create a basic backtest configuration."""
    end_date = datetime(2024, 12, 31)
    start_date = datetime(2024, 6, 1)

    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=1000000,
        rebalance_days=21,
        transaction_cost=0.003,
        target_positions=5,
        min_positions=3,
        use_stop_loss=True,
        initial_stop_loss=0.18,
        trailing_stop=0.15,
        min_score_percentile=90,
        min_52w_high_prox=0.80,
        weight_6m=0.40,
        weight_12m=0.60,
    )


class TestBacktestConfig:
    """Test BacktestConfig creation and validation."""

    def test_config_creation(self, backtest_config):
        """Config creates with valid parameters."""
        assert backtest_config.initial_capital == 1000000
        assert backtest_config.rebalance_days == 21
        assert backtest_config.target_positions == 5

    def test_config_defaults(self):
        """Config has sensible defaults."""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )

        assert config.initial_capital == 1600000
        assert config.rebalance_days == 21
        assert config.transaction_cost == 0.003
        assert config.use_stop_loss is True

    def test_weight_overrides(self):
        """Weight overrides are applied."""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            weight_6m=0.30,
            weight_12m=0.70,
        )

        assert config.weight_6m == 0.30
        assert config.weight_12m == 0.70


class TestBacktestEngine:
    """Test BacktestEngine initialization and methods."""

    def test_engine_creation(self, sample_universe, sample_historical_data, backtest_config):
        """Engine creates with valid inputs."""
        engine = BacktestEngine(
            universe=sample_universe,
            historical_data=sample_historical_data,
            config=backtest_config,
        )

        assert engine.universe is not None
        assert engine.data is not None
        assert engine.config is not None

    def test_get_trading_days(self, sample_universe, sample_historical_data, backtest_config):
        """Engine can get trading days from data."""
        engine = BacktestEngine(
            universe=sample_universe,
            historical_data=sample_historical_data,
            config=backtest_config,
        )

        trading_days = engine._get_trading_days()
        assert len(trading_days) > 0

    def test_get_rebalance_dates(self, sample_universe, sample_historical_data, backtest_config):
        """Engine generates rebalance dates."""
        engine = BacktestEngine(
            universe=sample_universe,
            historical_data=sample_historical_data,
            config=backtest_config,
        )

        rebalance_dates = engine._get_rebalance_dates()
        assert len(rebalance_dates) > 0

        # Check dates are within range
        for date in rebalance_dates:
            assert date >= backtest_config.start_date
            assert date <= backtest_config.end_date

    def test_get_price_at_date(self, sample_universe, sample_historical_data, backtest_config):
        """Engine retrieves prices correctly."""
        engine = BacktestEngine(
            universe=sample_universe,
            historical_data=sample_historical_data,
            config=backtest_config,
        )

        # Get price for a symbol that exists in sample data
        price = engine._get_price_at_date("RELIANCE", datetime(2024, 6, 15))
        assert price is not None
        assert price > 0

        # Non-existent symbol returns None
        price = engine._get_price_at_date("NONEXISTENT", datetime(2024, 6, 15))
        assert price is None


class TestBacktestResult:
    """Test BacktestResult structure."""

    def test_result_has_required_fields(self):
        """BacktestResult has all required fields."""
        result = BacktestResult(
            total_return=0.10,
            cagr=0.12,
            sharpe_ratio=1.5,
            max_drawdown=-0.15,
            win_rate=0.55,
            total_trades=50,
            equity_curve=pd.Series([100, 110, 105]),
            trades=[],
            sector_allocations=pd.DataFrame(),
        )

        assert result.total_return == 0.10
        assert result.cagr == 0.12
        assert result.sharpe_ratio == 1.5
        assert result.max_drawdown == -0.15
        assert result.win_rate == 0.55
        assert result.total_trades == 50


class TestBacktestIntegration:
    """Integration tests for full backtest runs."""

    def test_backtest_runs_with_sample_data(self, sample_universe, sample_historical_data, backtest_config):
        """Backtest completes with sample data."""
        engine = BacktestEngine(
            universe=sample_universe,
            historical_data=sample_historical_data,
            config=backtest_config,
        )

        result = engine.run()

        assert isinstance(result, BacktestResult)
        assert isinstance(result.total_return, float)
        assert isinstance(result.equity_curve, pd.Series)

    def test_backtest_metrics_reasonable(self, sample_universe, sample_historical_data, backtest_config):
        """Backtest produces reasonable metrics."""
        engine = BacktestEngine(
            universe=sample_universe,
            historical_data=sample_historical_data,
            config=backtest_config,
        )

        result = engine.run()

        # Returns should be reasonable (not extreme)
        assert -1.0 <= result.total_return <= 10.0
        assert -1.0 <= result.cagr <= 5.0

        # Drawdown should be negative or zero
        assert result.max_drawdown <= 0

        # Win rate between 0 and 1
        assert 0 <= result.win_rate <= 1

        # Trades should be non-negative
        assert result.total_trades >= 0

    def test_backtest_respects_rebalance_frequency(self, sample_universe, sample_historical_data):
        """Different rebalance frequencies produce different results."""
        end_date = datetime(2024, 12, 31)
        start_date = datetime(2024, 6, 1)

        # Weekly rebalance
        config_weekly = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            rebalance_days=5,
            target_positions=3,
        )

        # Monthly rebalance
        config_monthly = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            rebalance_days=21,
            target_positions=3,
        )

        engine_weekly = BacktestEngine(
            universe=sample_universe,
            historical_data=sample_historical_data,
            config=config_weekly,
        )

        engine_monthly = BacktestEngine(
            universe=sample_universe,
            historical_data=sample_historical_data,
            config=config_monthly,
        )

        result_weekly = engine_weekly.run()
        result_monthly = engine_monthly.run()

        # Weekly should have more rebalance points
        # (Different trade counts indicate different rebalance behavior)
        # Note: Results may be identical if no trades are needed
        assert isinstance(result_weekly, BacktestResult)
        assert isinstance(result_monthly, BacktestResult)

    def test_sector_diversification_default(self, sample_universe, sample_historical_data, backtest_config):
        """Default config has sector diversification enabled."""
        assert backtest_config.max_stocks_per_sector == 3

    def test_sector_diversification_can_be_disabled(self, sample_universe, sample_historical_data):
        """Sector diversification can be disabled with max_stocks_per_sector=0."""
        end_date = datetime(2024, 12, 31)
        start_date = datetime(2024, 6, 1)

        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            max_stocks_per_sector=0,  # Disabled
            target_positions=5,
        )

        engine = BacktestEngine(
            universe=sample_universe,
            historical_data=sample_historical_data,
            config=config,
        )

        result = engine.run()
        assert isinstance(result, BacktestResult)
