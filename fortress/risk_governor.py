"""
Risk governor for FORTRESS MOMENTUM.

Enforces invariants:
- R1: No position > hard_max_position (10%)
- R2: No sector > hard_max_sector (30%)
- R3: Daily loss triggers halt
- R4: Drawdown > 20% halts all trading
- R7: Position count <= max_positions
- R8: Risk governor has veto power
- R9: Stop loss triggers position exit
- R10: Trailing stop protects gains
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import RiskConfig, PortfolioConfig


@dataclass
class StopLossEntry:
    """Tracks stop loss state for a single position."""

    ticker: str
    entry_price: float
    entry_date: datetime
    peak_price: float              # Highest price since entry
    initial_stop: float            # Initial stop price (entry * (1 - stop_loss_pct))
    trailing_stop: Optional[float] = None  # Activated after trailing_activation gain
    trailing_activated: bool = False
    _trailing_pct: float = 0.15    # Default trailing stop percentage (initialized before update_peak)

    def update_peak(self, current_price: float) -> None:
        """Update peak price and trailing stop if applicable."""
        if current_price > self.peak_price:
            self.peak_price = current_price
            # Trailing stop moves up with price (never down)
            if self.trailing_activated:
                new_trailing = current_price * (1 - self._trailing_pct)
                if self.trailing_stop is None or new_trailing > self.trailing_stop:
                    self.trailing_stop = new_trailing

    def check_stop(
        self,
        current_price: float,
        initial_stop_pct: float = 0.18,
        trailing_stop_pct: float = 0.15,
        trailing_activation_pct: float = 0.08,
    ) -> Tuple[bool, str]:
        """
        Check if any stop loss is triggered.

        Args:
            current_price: Current market price
            initial_stop_pct: Initial stop loss percentage (18%)
            trailing_stop_pct: Trailing stop percentage (15%)
            trailing_activation_pct: Gain to activate trailing (8%)

        Returns:
            Tuple of (triggered, reason)
        """
        self._trailing_pct = trailing_stop_pct

        # Update peak price
        self.update_peak(current_price)

        # Calculate current gain
        current_gain = (current_price - self.entry_price) / self.entry_price

        # Check for trailing stop activation
        if not self.trailing_activated and current_gain >= trailing_activation_pct:
            self.trailing_activated = True
            self.trailing_stop = current_price * (1 - trailing_stop_pct)

        # R9: Check initial stop loss
        initial_stop_price = self.entry_price * (1 - initial_stop_pct)
        if current_price <= initial_stop_price:
            loss_pct = (current_price - self.entry_price) / self.entry_price
            return (True, f"Initial stop: {loss_pct:.1%} loss from entry")

        # R10: Check trailing stop
        if self.trailing_activated and self.trailing_stop:
            if current_price <= self.trailing_stop:
                gain_from_peak = (current_price - self.peak_price) / self.peak_price
                return (True, f"Trailing stop: {gain_from_peak:.1%} from peak")

        return (False, "OK")


@dataclass
class StopLossTracker:
    """Tracks stop losses for all positions."""

    entries: Dict[str, StopLossEntry] = field(default_factory=dict)

    def register(
        self,
        ticker: str,
        entry_price: float,
        entry_date: datetime,
    ) -> None:
        """Register a new position for stop loss tracking."""
        self.entries[ticker] = StopLossEntry(
            ticker=ticker,
            entry_price=entry_price,
            entry_date=entry_date,
            peak_price=entry_price,
            initial_stop=entry_price * 0.82,  # 18% default
        )

    def remove(self, ticker: str) -> None:
        """Remove a position from tracking."""
        if ticker in self.entries:
            del self.entries[ticker]

    def check_all_stops(
        self,
        current_prices: Dict[str, float],
        initial_stop_pct: float = 0.18,
        trailing_stop_pct: float = 0.15,
        trailing_activation_pct: float = 0.08,
    ) -> List[Tuple[str, str]]:
        """
        Check all positions for stop loss triggers.

        Args:
            current_prices: Dict of ticker -> current price
            initial_stop_pct: Initial stop loss percentage
            trailing_stop_pct: Trailing stop percentage
            trailing_activation_pct: Gain to activate trailing

        Returns:
            List of (ticker, reason) for triggered stops
        """
        triggered = []
        for ticker, entry in self.entries.items():
            if ticker not in current_prices:
                continue
            current_price = current_prices[ticker]
            is_triggered, reason = entry.check_stop(
                current_price=current_price,
                initial_stop_pct=initial_stop_pct,
                trailing_stop_pct=trailing_stop_pct,
                trailing_activation_pct=trailing_activation_pct,
            )
            if is_triggered:
                triggered.append((ticker, reason))
        return triggered

    def get_entry(self, ticker: str) -> Optional[StopLossEntry]:
        """Get stop loss entry for a ticker."""
        return self.entries.get(ticker)


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    passed: bool
    reason: str
    adjusted_value: Optional[float] = None


class RiskGovernor:
    """
    Validates all portfolio actions against risk limits.

    Has override authority over all allocation decisions (R8).
    Enhanced with stop loss tracking (R9, R10) for momentum strategy.
    """

    def __init__(
        self,
        risk_config: Optional[RiskConfig] = None,
        portfolio_config: Optional[PortfolioConfig] = None,
    ):
        """
        Initialize risk governor.

        Args:
            risk_config: Risk limit settings
            portfolio_config: Portfolio settings
        """
        self.risk = risk_config or RiskConfig()
        self.portfolio = portfolio_config or PortfolioConfig()
        self._peak_value: float = 0.0
        self._day_start_value: float = 0.0

        # Stop loss tracking for momentum strategy
        self.stop_loss_tracker = StopLossTracker()

    def set_peak_value(self, value: float) -> None:
        """Update peak portfolio value for drawdown calculation."""
        if value > self._peak_value:
            self._peak_value = value

    def set_day_start_value(self, value: float) -> None:
        """Set portfolio value at start of day for daily loss check."""
        self._day_start_value = value

    def validate_position_size(
        self,
        symbol: str,
        proposed_value: float,
        portfolio_value: float,
    ) -> RiskCheckResult:
        """
        Validate proposed position size.

        Enforces R1: No position > hard_max_position (10%).

        Args:
            symbol: Stock symbol
            proposed_value: Proposed position value
            portfolio_value: Total portfolio value

        Returns:
            RiskCheckResult with pass/fail and adjusted value
        """
        if portfolio_value <= 0:
            return RiskCheckResult(
                passed=False,
                reason="Portfolio value must be positive",
            )

        proposed_pct = proposed_value / portfolio_value

        # R1: Hard limit check
        if proposed_pct > self.risk.hard_max_position:
            adjusted = portfolio_value * self.risk.hard_max_position
            return RiskCheckResult(
                passed=False,
                reason=f"R1: Exceeds hard limit {self.risk.hard_max_position:.0%}",
                adjusted_value=adjusted,
            )

        # Soft limit warning
        if proposed_pct > self.risk.max_single_position:
            adjusted = portfolio_value * self.risk.max_single_position
            return RiskCheckResult(
                passed=False,
                reason=f"Exceeds soft limit {self.risk.max_single_position:.0%}",
                adjusted_value=adjusted,
            )

        return RiskCheckResult(passed=True, reason="OK")

    def validate_sector_exposure(
        self,
        sector: str,
        current_exposure: float,
        proposed_addition: float,
        portfolio_value: float,
    ) -> RiskCheckResult:
        """
        Validate sector concentration.

        Enforces R2: No sector > hard_max_sector (30%).

        Args:
            sector: Sector name
            current_exposure: Current sector exposure value
            proposed_addition: Value to add to sector
            portfolio_value: Total portfolio value

        Returns:
            RiskCheckResult with pass/fail and adjusted value
        """
        if portfolio_value <= 0:
            return RiskCheckResult(
                passed=False,
                reason="Portfolio value must be positive",
            )

        total_exposure = (current_exposure + proposed_addition) / portfolio_value

        # R2: Hard limit check
        if total_exposure > self.risk.hard_max_sector:
            max_addition = (
                self.risk.hard_max_sector * portfolio_value
            ) - current_exposure
            return RiskCheckResult(
                passed=False,
                reason=f"R2: Sector hard limit {self.risk.hard_max_sector:.0%}",
                adjusted_value=max(0, max_addition),
            )

        # Soft limit warning
        if total_exposure > self.risk.max_sector_exposure:
            max_addition = (
                self.risk.max_sector_exposure * portfolio_value
            ) - current_exposure
            return RiskCheckResult(
                passed=False,
                reason=f"Sector soft limit {self.risk.max_sector_exposure:.0%}",
                adjusted_value=max(0, max_addition),
            )

        return RiskCheckResult(passed=True, reason="OK")

    def check_daily_loss(
        self,
        current_value: float,
    ) -> RiskCheckResult:
        """
        Check if daily loss limit breached.

        Enforces R3: Daily loss triggers halt.

        Args:
            current_value: Current portfolio value

        Returns:
            RiskCheckResult indicating if trading should halt
        """
        if self._day_start_value <= 0:
            return RiskCheckResult(passed=True, reason="OK")

        daily_return = (
            current_value - self._day_start_value
        ) / self._day_start_value

        # R3: Daily loss limit
        if daily_return <= -self.risk.daily_loss_limit:
            return RiskCheckResult(
                passed=False,
                reason=f"R3: Daily loss {daily_return:.2%} exceeds limit "
                       f"{self.risk.daily_loss_limit:.0%}",
            )

        return RiskCheckResult(passed=True, reason="OK")

    def check_position_count(
        self,
        current_positions: int,
        proposed_additions: int = 0,
    ) -> RiskCheckResult:
        """
        Check if position count is within limits.

        Enforces R7: Position count <= max_positions.

        Args:
            current_positions: Number of current positions
            proposed_additions: Number of new positions to add

        Returns:
            RiskCheckResult
        """
        total = current_positions + proposed_additions

        if total > self.portfolio.max_positions:
            return RiskCheckResult(
                passed=False,
                reason=f"R7: Position count {total} exceeds max "
                       f"{self.portfolio.max_positions}",
            )

        return RiskCheckResult(passed=True, reason="OK")

    def register_stop_loss(
        self,
        ticker: str,
        entry_price: float,
        entry_date: Optional[datetime] = None,
    ) -> None:
        """
        Register a position for stop loss tracking (R9, R10).

        Args:
            ticker: Stock ticker
            entry_price: Entry price
            entry_date: Date of entry (defaults to now)
        """
        self.stop_loss_tracker.register(
            ticker=ticker,
            entry_price=entry_price,
            entry_date=entry_date or datetime.now(),
        )

    def remove_stop_loss(self, ticker: str) -> None:
        """Remove a position from stop loss tracking."""
        self.stop_loss_tracker.remove(ticker)

    def check_stop_losses(
        self,
        current_prices: Dict[str, float],
    ) -> List[Tuple[str, str]]:
        """
        Check all positions for stop loss triggers (R9, R10).

        Args:
            current_prices: Dict of ticker -> current price

        Returns:
            List of (ticker, reason) for triggered stops
        """
        return self.stop_loss_tracker.check_all_stops(
            current_prices=current_prices,
            initial_stop_pct=self.risk.initial_stop_loss,
            trailing_stop_pct=self.risk.trailing_stop,
            trailing_activation_pct=self.risk.trailing_activation,
        )

    def get_stop_loss_entry(self, ticker: str) -> Optional[StopLossEntry]:
        """Get stop loss entry for a ticker."""
        return self.stop_loss_tracker.get_entry(ticker)

    def can_trade(
        self,
        current_value: float,
        current_drawdown: float,
    ) -> Tuple[bool, str]:
        """
        R8: Risk governor veto check - can we trade at all?

        Args:
            current_value: Current portfolio value
            current_drawdown: Current drawdown

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check daily loss
        daily_check = self.check_daily_loss(current_value)
        if not daily_check.passed:
            return (False, daily_check.reason)

        # Check drawdown halt
        if abs(current_drawdown) >= self.risk.max_drawdown_halt:
            return (False, f"Trading halted: drawdown {current_drawdown:.2%}")

        return (True, "OK")

    def validate_order(
        self,
        symbol: str,
        sector: str,
        order_value: float,
        current_position_value: float,
        current_sector_value: float,
        portfolio_value: float,
        current_positions: int,
        is_buy: bool,
    ) -> RiskCheckResult:
        """
        Comprehensive order validation.

        R8: Risk governor has veto power over all orders.

        Args:
            symbol: Stock symbol
            sector: Stock's sector
            order_value: Value of the order
            current_position_value: Current position value in this stock
            current_sector_value: Current total sector exposure
            portfolio_value: Total portfolio value
            current_positions: Number of current positions
            is_buy: True for buy, False for sell

        Returns:
            RiskCheckResult
        """
        # Sells are always allowed (reduces risk)
        if not is_buy:
            return RiskCheckResult(passed=True, reason="OK")

        # Check position size
        new_position_value = current_position_value + order_value
        position_check = self.validate_position_size(
            symbol, new_position_value, portfolio_value
        )
        if not position_check.passed:
            return position_check

        # Check sector exposure
        sector_check = self.validate_sector_exposure(
            sector, current_sector_value, order_value, portfolio_value
        )
        if not sector_check.passed:
            return sector_check

        # Check position count (new position if current is 0)
        if current_position_value == 0:
            count_check = self.check_position_count(current_positions, 1)
            if not count_check.passed:
                return count_check

        return RiskCheckResult(passed=True, reason="OK")
