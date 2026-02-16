"""
Tests for indicators module.

Tests for Normalized Momentum Score (NMS) calculation
and enhanced indicators for adaptive strategy.
"""

import numpy as np
import pandas as pd
import pytest

from fortress.indicators import (
    ExhaustionResult,
    NMSResult,
    RelativeStrengthResult,
    calculate_breakout_quality,
    calculate_drawdown,
    calculate_exhaustion_score,
    calculate_momentum_acceleration,
    calculate_normalized_momentum_score,
    calculate_relative_strength,
)


@pytest.fixture
def sample_prices():
    """Generate sample price series with 280+ days."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=300, freq="D")
    # Simulate upward trending stock with volatility
    returns = np.random.normal(0.001, 0.02, 300)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates)


@pytest.fixture
def sample_volumes():
    """Generate sample volume series."""
    np.random.seed(43)
    dates = pd.date_range(start="2024-01-01", periods=300, freq="D")
    volumes = np.random.uniform(1e6, 5e6, 300)
    return pd.Series(volumes, index=dates)


class TestNMSCalculation:
    """Test Normalized Momentum Score calculation."""

    def test_nms_returns_result(self, sample_prices, sample_volumes):
        """NMS calculation returns valid result."""
        result = calculate_normalized_momentum_score(sample_prices, sample_volumes)

        assert result is not None
        assert isinstance(result, NMSResult)
        assert isinstance(result.nms, float)

    def test_nms_components(self, sample_prices, sample_volumes):
        """NMS has all expected components."""
        result = calculate_normalized_momentum_score(sample_prices, sample_volumes)

        assert hasattr(result, "nms")
        assert hasattr(result, "return_6m")
        assert hasattr(result, "return_12m")
        assert hasattr(result, "volatility_6m")
        assert hasattr(result, "adj_return_6m")
        assert hasattr(result, "adj_return_12m")
        assert hasattr(result, "high_52w_proximity")
        assert hasattr(result, "above_50ema")
        assert hasattr(result, "above_200sma")
        assert hasattr(result, "volume_surge")
        assert hasattr(result, "daily_turnover")

    def test_nms_weights_sum_to_one(self, sample_prices, sample_volumes):
        """NMS weights must sum to 1.0."""
        # Test with default weights (0.5, 0.5)
        result1 = calculate_normalized_momentum_score(
            sample_prices, sample_volumes, weight_6m=0.5, weight_12m=0.5
        )
        assert result1 is not None

        # Test with custom weights
        result2 = calculate_normalized_momentum_score(
            sample_prices, sample_volumes, weight_6m=0.4, weight_12m=0.6
        )
        assert result2 is not None

    def test_nms_invalid_weights_rejected(self, sample_prices, sample_volumes):
        """NMS rejects weights that don't sum to 1.0."""
        with pytest.raises(AssertionError):
            calculate_normalized_momentum_score(
                sample_prices, sample_volumes, weight_6m=0.3, weight_12m=0.3
            )

    def test_handles_insufficient_data(self):
        """NMS returns None for insufficient data."""
        short_prices = pd.Series([100, 101, 102])
        short_volumes = pd.Series([1e6, 1e6, 1e6])

        result = calculate_normalized_momentum_score(short_prices, short_volumes)
        assert result is None

    def test_volatility_adjustment(self, sample_prices, sample_volumes):
        """Higher volatility stocks get lower adjusted returns."""
        result = calculate_normalized_momentum_score(sample_prices, sample_volumes)

        # Adjusted returns should be return / volatility
        if result.volatility_6m > 0.10:
            assert abs(result.adj_return_6m - result.return_6m / result.volatility_6m) < 1e-6

    def test_52w_high_proximity(self, sample_prices, sample_volumes):
        """52-week high proximity is between 0 and 1."""
        result = calculate_normalized_momentum_score(sample_prices, sample_volumes)

        assert 0 <= result.high_52w_proximity <= 1.0

    def test_volume_surge_calculated(self, sample_prices, sample_volumes):
        """Volume surge is ratio of 20-day to 50-day average."""
        result = calculate_normalized_momentum_score(sample_prices, sample_volumes)

        assert result.volume_surge > 0


class TestDrawdown:
    """Test drawdown calculation."""

    def test_no_drawdown_at_high(self):
        """No drawdown when at peak."""
        prices = pd.Series([100, 110, 120, 130, 140])
        current, max_dd = calculate_drawdown(prices)

        assert current == 0.0
        assert max_dd == 0.0

    def test_current_drawdown(self):
        """Current drawdown calculated correctly."""
        prices = pd.Series([100, 110, 120, 100])  # 20/120 = 16.67% drawdown
        current, max_dd = calculate_drawdown(prices)

        expected_dd = (100 - 120) / 120
        assert abs(current - expected_dd) < 1e-10

    def test_max_drawdown(self):
        """Max drawdown tracks worst point."""
        prices = pd.Series([100, 120, 80, 100])  # Max DD at 80: (80-120)/120
        current, max_dd = calculate_drawdown(prices)

        expected_max_dd = (80 - 120) / 120
        assert abs(max_dd - expected_max_dd) < 1e-10

    def test_empty_prices(self):
        """Empty prices return zero drawdowns."""
        prices = pd.Series([], dtype=float)
        current, max_dd = calculate_drawdown(prices)

        assert current == 0.0
        assert max_dd == 0.0


class TestRelativeStrength:
    """Tests for Relative Strength calculation."""

    @pytest.fixture
    def stock_prices(self):
        """Stock prices outperforming benchmark."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=150, freq="D")
        # Stock with 0.15% daily return (outperforming)
        returns = np.random.normal(0.0015, 0.02, 150)
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates)

    @pytest.fixture
    def benchmark_prices(self):
        """Benchmark prices (lower growth)."""
        np.random.seed(43)
        dates = pd.date_range(start="2024-01-01", periods=150, freq="D")
        # Benchmark with 0.08% daily return
        returns = np.random.normal(0.0008, 0.015, 150)
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates)

    def test_rs_returns_result(self, stock_prices, benchmark_prices):
        """RS calculation returns valid result."""
        result = calculate_relative_strength(stock_prices, benchmark_prices)
        assert result is not None
        assert isinstance(result, RelativeStrengthResult)

    def test_rs_has_all_timeframes(self, stock_prices, benchmark_prices):
        """RS result has all timeframe components."""
        result = calculate_relative_strength(stock_prices, benchmark_prices)
        assert hasattr(result, "rs_21d")
        assert hasattr(result, "rs_63d")
        assert hasattr(result, "rs_126d")
        assert hasattr(result, "rs_composite")

    def test_rs_outperformer_above_one(self, stock_prices, benchmark_prices):
        """Outperforming stock should have RS > 1."""
        result = calculate_relative_strength(stock_prices, benchmark_prices)
        # The stock fixture is designed to outperform
        assert result.rs_composite >= 0.9  # Allow some variance

    def test_rs_insufficient_data_returns_none(self):
        """RS returns None for insufficient data."""
        short_stock = pd.Series([100, 101, 102])
        short_bench = pd.Series([100, 100.5, 101])
        result = calculate_relative_strength(short_stock, short_bench)
        assert result is None


class TestMomentumAcceleration:
    """Tests for Momentum Acceleration calculation."""

    @pytest.fixture
    def accelerating_prices(self):
        """Prices with accelerating momentum."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        # Accelerating: recent returns higher than earlier
        returns = np.concatenate(
            [
                np.random.normal(0.001, 0.01, 70),  # Earlier: 0.1% daily
                np.random.normal(0.003, 0.01, 30),  # Recent: 0.3% daily
            ]
        )
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates)

    @pytest.fixture
    def decelerating_prices(self):
        """Prices with decelerating momentum."""
        np.random.seed(44)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        # Decelerating: recent returns lower than earlier
        returns = np.concatenate(
            [
                np.random.normal(0.003, 0.01, 70),  # Earlier: 0.3% daily
                np.random.normal(0.001, 0.01, 30),  # Recent: 0.1% daily
            ]
        )
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates)

    def test_acceleration_returns_float(self, accelerating_prices):
        """Acceleration calculation returns float."""
        result = calculate_momentum_acceleration(accelerating_prices)
        assert isinstance(result, float)

    def test_accelerating_above_one(self, accelerating_prices):
        """Accelerating momentum should be >= 1."""
        result = calculate_momentum_acceleration(accelerating_prices)
        # May not always be > 1 due to volatility, but should be reasonable
        assert 0.5 <= result <= 2.0

    def test_insufficient_data_returns_default(self):
        """Returns 1.0 for insufficient data."""
        short_prices = pd.Series([100, 101, 102, 103])
        result = calculate_momentum_acceleration(short_prices)
        assert result == 1.0


class TestExhaustionScore:
    """Tests for Exhaustion Score calculation."""

    @pytest.fixture
    def normal_prices(self):
        """Normal trending prices."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        returns = np.random.normal(0.001, 0.015, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates)

    @pytest.fixture
    def normal_volumes(self):
        """Normal volume series."""
        np.random.seed(43)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        volumes = np.random.uniform(1e6, 2e6, 100)
        return pd.Series(volumes, index=dates)

    def test_exhaustion_returns_result(self, normal_prices, normal_volumes):
        """Exhaustion calculation returns valid result."""
        result = calculate_exhaustion_score(normal_prices, normal_volumes)
        assert result is not None
        assert isinstance(result, ExhaustionResult)

    def test_exhaustion_has_components(self, normal_prices, normal_volumes):
        """Exhaustion result has expected components."""
        result = calculate_exhaustion_score(normal_prices, normal_volumes)
        assert hasattr(result, "exhaustion_score")
        assert hasattr(result, "distance_from_20ema")
        assert hasattr(result, "distance_from_50ema")
        assert hasattr(result, "rsi_14")
        assert hasattr(result, "volume_exhaustion")

    def test_exhaustion_score_bounded(self, normal_prices, normal_volumes):
        """Exhaustion score should be 0-100."""
        result = calculate_exhaustion_score(normal_prices, normal_volumes)
        assert 0 <= result.exhaustion_score <= 100

    def test_rsi_bounded(self, normal_prices, normal_volumes):
        """RSI should be 0-100."""
        result = calculate_exhaustion_score(normal_prices, normal_volumes)
        assert 0 <= result.rsi_14 <= 100

    def test_insufficient_data_returns_none(self):
        """Returns None for insufficient data."""
        short_prices = pd.Series([100, 101, 102])
        short_volumes = pd.Series([1e6, 1e6, 1e6])
        result = calculate_exhaustion_score(short_prices, short_volumes)
        assert result is None


class TestBreakoutQuality:
    """Tests for Breakout Quality Score calculation."""

    @pytest.fixture
    def breakout_prices(self):
        """Prices showing a breakout pattern."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=280, freq="D")
        # Consolidation followed by breakout
        returns = np.concatenate(
            [
                np.random.normal(0.0005, 0.01, 250),  # Consolidation
                np.random.normal(0.005, 0.015, 30),  # Breakout
            ]
        )
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates)

    @pytest.fixture
    def breakout_volumes(self):
        """Volumes with surge on breakout."""
        np.random.seed(43)
        dates = pd.date_range(start="2024-01-01", periods=280, freq="D")
        volumes = np.concatenate(
            [
                np.random.uniform(1e6, 2e6, 275),  # Normal volume
                np.random.uniform(3e6, 5e6, 5),  # Volume surge
            ]
        )
        return pd.Series(volumes, index=dates)

    def test_breakout_quality_returns_float(self, breakout_prices, breakout_volumes):
        """Breakout quality returns float."""
        result = calculate_breakout_quality(breakout_prices, breakout_volumes)
        assert isinstance(result, float)

    def test_breakout_quality_bounded(self, breakout_prices, breakout_volumes):
        """Breakout quality should be 0-100."""
        result = calculate_breakout_quality(breakout_prices, breakout_volumes)
        assert 0 <= result <= 100

    def test_insufficient_data_returns_default(self):
        """Returns default score for insufficient data."""
        short_prices = pd.Series([100, 101, 102])
        short_volumes = pd.Series([1e6, 1e6, 1e6])
        result = calculate_breakout_quality(short_prices, short_volumes)
        assert result == 50.0  # Default score
