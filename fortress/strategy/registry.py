"""
Strategy registry for FORTRESS MOMENTUM.

Central registry for all available strategies. Strategies register themselves
and can be retrieved by name for use in backtest/CLI.
"""

from typing import TYPE_CHECKING, Dict, List, Tuple, Type

from .base import BaseStrategy

if TYPE_CHECKING:
    from ..config import Config


class StrategyRegistry:
    """
    Central registry for all available strategies.

    Usage:
        # Get strategy by name
        strategy = StrategyRegistry.get("dual_momentum", config)

        # List available strategies
        for name, desc in StrategyRegistry.list_strategies():
            print(f"{name}: {desc}")
    """

    _strategies: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a strategy class.

        Args:
            strategy_class: Strategy class to register
        """
        # Instantiate to get the name
        instance = strategy_class()
        cls._strategies[instance.name] = strategy_class

    @classmethod
    def get(cls, name: str, config: "Config" = None) -> BaseStrategy:
        """
        Get strategy instance by name.

        Args:
            name: Strategy identifier (e.g., "simple")
            config: Optional config to pass to strategy

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy name is not found
        """
        if name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(f"Unknown strategy: {name}. Available: {available}")
        return cls._strategies[name](config)

    @classmethod
    def list_strategies(cls) -> List[Tuple[str, str]]:
        """
        List all available strategies.

        Returns:
            List of (name, description) tuples
        """
        result = []
        for name in sorted(cls._strategies.keys()):
            instance = cls._strategies[name]()
            result.append((name, instance.description))
        return result

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a strategy is registered."""
        return name in cls._strategies

    @classmethod
    def get_names(cls) -> List[str]:
        """Get list of registered strategy names."""
        return list(cls._strategies.keys())
