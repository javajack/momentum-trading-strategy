"""
Strategy package for FORTRESS MOMENTUM.

Provides a pluggable strategy architecture where different momentum strategies
can be used interchangeably. All strategies implement the BaseStrategy protocol
and produce standard outputs that backtest/CLI can consume.

Available strategies:
- AdaptiveDualMomentumStrategy: Dual momentum with regime adaptation and recovery modes
"""

from .base import (
    BaseStrategy,
    StockScore,
    ExitSignal,
    StopLossConfig,
)
from .registry import StrategyRegistry
from .adaptive_dual_momentum import AdaptiveDualMomentumStrategy

__all__ = [
    "BaseStrategy",
    "StockScore",
    "ExitSignal",
    "StopLossConfig",
    "StrategyRegistry",
    "AdaptiveDualMomentumStrategy",
]
