"""
FORTRESS MOMENTUM - Pure Momentum Strategy for Indian Equities

A pure momentum trading system:
- Ranks ALL stocks by Normalized Momentum Score (NMS)
- Volatility-adjusted 6M/12M returns (Nifty 500 Momentum 50 style)
- Entry filters: 52W high proximity, trend, volume, liquidity
- Exit triggers: Stop loss, trailing stop, momentum decay
- Target: 3% monthly returns (36%+ annually)
"""

__version__ = "2.0.0"
__author__ = "Rakesh"

from .config import (
    Config,
    load_config,
    PureMomentumConfig,
    PositionSizingConfig,
    RebalancingConfig,
)
from .indicators import (
    NMSResult,
    calculate_normalized_momentum_score,
    calculate_exit_triggers,
)
from .momentum_engine import (
    MomentumEngine,
    StockMomentum,
    ExitReason,
)
from .risk_governor import (
    RiskGovernor,
    StopLossTracker,
    StopLossEntry,
)

__all__ = [
    "Config",
    "load_config",
    "PureMomentumConfig",
    "PositionSizingConfig",
    "RebalancingConfig",
    "NMSResult",
    "calculate_normalized_momentum_score",
    "calculate_exit_triggers",
    "MomentumEngine",
    "StockMomentum",
    "ExitReason",
    "RiskGovernor",
    "StopLossTracker",
    "StopLossEntry",
]
