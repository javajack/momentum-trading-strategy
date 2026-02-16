"""
Base strategy protocol and data classes for FORTRESS MOMENTUM.

Defines the contract that all momentum strategies must implement.
Strategies produce standard outputs that backtest/CLI can consume.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..config import Config
    from ..market_data import MarketDataProvider
    from ..universe import Universe


@dataclass
class StockScore:
    """
    Universal stock scoring result - all strategies produce this.

    Contains the primary ranking score and all filter information needed
    for portfolio selection decisions.
    """

    ticker: str
    sector: str
    sub_sector: str
    zerodha_symbol: str
    name: str

    # Primary ranking
    score: float  # Primary ranking score (strategy-specific)
    rank: int = 0  # Overall rank (1 = best)
    percentile: float = 0.0  # Score percentile (0-100)

    # Entry filter status
    passes_entry_filters: bool = False
    filter_reasons: List[str] = field(default_factory=list)

    # Common metrics (all strategies populate these)
    return_6m: float = 0.0
    return_12m: float = 0.0
    volatility: float = 0.0
    high_52w_proximity: float = 0.0
    above_50ema: bool = False
    above_200sma: bool = False
    volume_surge: float = 0.0
    daily_turnover: float = 0.0
    current_price: float = 0.0

    # Strategy-specific metrics (optional)
    extra_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExitSignal:
    """
    Universal exit signal - all strategies produce this.

    Contains the exit decision and reason for logging/debugging.
    """

    should_exit: bool
    reason: str
    exit_type: str  # "stop_loss", "trailing", "momentum_decay", etc.
    urgency: str = "normal"  # "immediate", "next_rebalance", "warning"


@dataclass
class StopLossConfig:
    """
    Stop loss configuration returned by strategy.

    Strategies can return different stop loss configurations based on
    current position gain (tiered stops) or other factors.
    """

    initial_stop: float  # Initial stop loss from entry (e.g., 0.18 = 18%)
    trailing_stop: float  # Trailing stop from peak (e.g., 0.15 = 15%)
    trailing_activation: float  # Gain needed to activate trailing (e.g., 0.08)

    # Strategy-specific extensions for tiered stops
    use_tiered: bool = False
    tiers: Optional[Dict[str, float]] = None  # e.g., {"tier2": 0.15, "tier3": 0.18}


class BaseStrategy(ABC):
    """
    Protocol for all momentum strategies.

    Each strategy implements its own:
    - Scoring logic (how to rank stocks)
    - Entry filters (when to buy)
    - Exit triggers (when to sell)
    - Stop loss configuration

    But all produce standard outputs that backtest/CLI can consume.

    Strategies receive a Config object with strategy-specific sections
    (e.g., strategy_dual_momentum) that they can read.
    """

    def __init__(self, config: Optional["Config"] = None):
        """
        Initialize strategy with optional config.

        Args:
            config: Application configuration (strategies read their section)
        """
        self._config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'dual_momentum')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for display in CLI."""
        pass

    @abstractmethod
    def rank_stocks(
        self,
        as_of_date: datetime,
        universe: "Universe",
        market_data: "MarketDataProvider",
        filter_entry: bool = True,
    ) -> List[StockScore]:
        """
        Rank all stocks in universe by strategy's scoring method.

        Args:
            as_of_date: Date for calculation (no look-ahead)
            universe: Stock universe
            market_data: Market data provider
            filter_entry: If True, only return stocks passing entry filters

        Returns:
            Sorted list (best first) with entry filter status
        """
        pass

    @abstractmethod
    def select_portfolio(
        self,
        ranked_stocks: List[StockScore],
        portfolio_value: float,
        current_positions: Dict[str, float],
        max_positions: int,
        max_per_sector: int,
    ) -> Dict[str, float]:
        """
        Select target portfolio from ranked stocks.

        Args:
            ranked_stocks: Ranked stocks from rank_stocks()
            portfolio_value: Total portfolio value
            current_positions: Current holdings (ticker -> value)
            max_positions: Maximum number of positions
            max_per_sector: Maximum stocks per sector

        Returns:
            Dict mapping ticker to target weight (0-1)
        """
        pass

    @abstractmethod
    def calculate_weights(
        self,
        selected_stocks: List[StockScore],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Calculate target weights for selected stocks.

        Args:
            selected_stocks: Stocks to allocate to
            portfolio_value: Total portfolio value

        Returns:
            Dict mapping ticker to target weight (0-1, summing to 1.0)
        """
        pass

    @abstractmethod
    def check_exit_triggers(
        self,
        ticker: str,
        entry_price: float,
        current_price: float,
        peak_price: float,
        days_held: int,
        stock_score: Optional[StockScore],
        nms_percentile: float,
    ) -> ExitSignal:
        """
        Check if position should be exited.

        Args:
            ticker: Stock ticker
            entry_price: Price at entry
            current_price: Current market price
            peak_price: Highest price since entry
            days_held: Days position has been held
            stock_score: Current stock score (if available)
            nms_percentile: Current NMS/score percentile

        Returns:
            ExitSignal with reason and urgency
        """
        pass

    @abstractmethod
    def get_stop_loss_config(
        self,
        ticker: str,
        current_gain: float,
    ) -> StopLossConfig:
        """
        Get stop loss configuration for a position.

        Args:
            ticker: Stock ticker
            current_gain: Current gain from entry (e.g., 0.15 = 15%)

        Returns:
            StopLossConfig (may vary by gain tier for adaptive strategies)
        """
        pass

    def get_config_schema(self) -> Dict:
        """
        Return strategy-specific config parameters.

        Override this to expose strategy-specific configuration options
        for the config.yaml file.

        Returns:
            Dict describing config parameters
        """
        return {}
