"""
Portfolio tracking for FORTRESS MOMENTUM.

Enforces invariant:
- R6: Margin check before buy orders
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .universe import Universe


@dataclass
class Position:
    """Represents a single position in the portfolio."""

    symbol: str
    quantity: int
    average_price: float
    sector: str
    current_price: float = 0.0

    @property
    def value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        return self.quantity * self.average_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis


@dataclass
class MergeDiagnostic:
    """Per-symbol diagnostic from holdings/positions merge."""

    holdings_qty: int
    day_bought: int
    day_sold: int
    net_qty: int
    value: float


@dataclass
class PortfolioSnapshot:
    """Point-in-time snapshot of portfolio state."""

    positions: Dict[str, Position]
    cash: float
    total_value: float
    unrealized_pnl: float
    day_start_value: float = 0.0
    merge_diagnostics: Dict[str, MergeDiagnostic] = field(default_factory=dict)

    @property
    def invested_value(self) -> float:
        """Total value invested in positions."""
        return sum(p.value for p in self.positions.values())

    @property
    def position_count(self) -> int:
        """Number of positions."""
        return len(self.positions)

    def get_sector_exposure(self, sector: str) -> float:
        """Get total exposure to a sector."""
        return sum(
            p.value for p in self.positions.values() if p.sector == sector
        )

    def get_sector_weights(self) -> Dict[str, float]:
        """Get sector weights as percentages."""
        if self.total_value == 0:
            return {}

        sector_values: Dict[str, float] = {}
        for p in self.positions.values():
            sector_values[p.sector] = sector_values.get(p.sector, 0) + p.value

        return {s: v / self.total_value for s, v in sector_values.items()}


class Portfolio:
    """
    Manages portfolio state from Zerodha holdings.

    Provides methods to:
    - Load holdings from Kite API
    - Calculate portfolio metrics
    - Check margin availability (R6)
    """

    def __init__(self, kite, universe: Universe):
        """
        Initialize portfolio manager.

        Args:
            kite: Authenticated KiteConnect instance
            universe: Loaded Universe instance
        """
        self.kite = kite
        self.universe = universe
        self._snapshot: Optional[PortfolioSnapshot] = None

    def load_holdings(self) -> PortfolioSnapshot:
        """
        Load current holdings from Zerodha.

        Returns:
            Current PortfolioSnapshot
        """
        holdings = self.kite.holdings()
        positions_dict: Dict[str, Position] = {}

        for h in holdings:
            symbol = h["tradingsymbol"]

            # Include both settled and T1 (unsettled) quantities
            total_qty = h["quantity"] + h.get("t1_quantity", 0)
            if total_qty == 0:
                continue

            # Get sector from universe
            stock = self.universe.get_stock(symbol)
            sector = stock.sector if stock else "UNKNOWN"

            positions_dict[symbol] = Position(
                symbol=symbol,
                quantity=total_qty,
                average_price=h["average_price"],
                sector=sector,
                current_price=h["last_price"],
            )

        # Get margin/cash
        margins = self.kite.margins()
        equity = margins.get("equity", {})
        available = equity.get("available", {})
        cash = available.get("live_balance", 0)

        total_value = cash + sum(p.value for p in positions_dict.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in positions_dict.values())

        self._snapshot = PortfolioSnapshot(
            positions=positions_dict,
            cash=cash,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
        )

        return self._snapshot

    def get_snapshot(self) -> PortfolioSnapshot:
        """
        Get current portfolio snapshot.

        Returns:
            Cached or freshly loaded snapshot
        """
        if self._snapshot is None:
            return self.load_combined_positions()
        return self._snapshot

    def check_margin(self, required_amount: float) -> tuple:
        """
        Check if sufficient margin is available for a buy order.

        Enforces R6: Margin check before buy orders.

        Args:
            required_amount: Amount needed for the buy order

        Returns:
            Tuple of (has_margin, available_margin)
        """
        margins = self.kite.margins()
        equity = margins.get("equity", {})
        available = equity.get("available", {})
        live_balance = available.get("live_balance", 0)

        return (live_balance >= required_amount, live_balance)

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position or None if not held
        """
        snapshot = self.get_snapshot()
        return snapshot.positions.get(symbol)

    def get_position_value(self, symbol: str) -> float:
        """
        Get current value of a position.

        Args:
            symbol: Stock symbol

        Returns:
            Position value or 0 if not held
        """
        position = self.get_position(symbol)
        return position.value if position else 0.0

    def get_positions_by_sector(self, sector: str) -> List[Position]:
        """
        Get all positions in a sector.

        Args:
            sector: Sector name

        Returns:
            List of positions in the sector
        """
        snapshot = self.get_snapshot()
        return [p for p in snapshot.positions.values() if p.sector == sector]

    def refresh(self) -> PortfolioSnapshot:
        """Force refresh of portfolio data (settled + today's trades)."""
        self._snapshot = None
        return self.load_combined_positions()

    def load_combined_positions(self) -> PortfolioSnapshot:
        """
        Load holdings + today's CNC positions for complete picture.

        Holdings = settled (T+1) positions
        Positions = today's trades (unsettled)

        Combines both to get actual current state, making rebalance
        stateless and idempotent.
        """
        # Get delivered holdings
        holdings = self.kite.holdings()

        # Get today's positions (includes intraday CNC trades)
        day_positions = self.kite.positions()
        net_positions = day_positions.get("net", [])

        positions_dict: Dict[str, Position] = {}

        # First, load all holdings (settled + T1 unsettled)
        for h in holdings:
            symbol = h["tradingsymbol"]
            total_qty = h["quantity"] + h.get("t1_quantity", 0)
            if total_qty == 0:
                continue
            stock = self.universe.get_stock(symbol)
            sector = stock.sector if stock else "UNKNOWN"
            positions_dict[symbol] = Position(
                symbol=symbol,
                quantity=total_qty,
                average_price=h["average_price"],
                sector=sector,
                current_price=h["last_price"],
            )

        merge_diagnostics: Dict[str, MergeDiagnostic] = {}

        # Then, overlay today's CNC positions
        # Use day_buy_quantity / day_sell_quantity (unambiguous, no double-counting)
        # because positions "quantity" already includes overnight holdings
        for p in net_positions:
            if p["product"] != "CNC":
                continue
            symbol = p["tradingsymbol"]

            day_bought = p.get("day_buy_quantity", 0)
            day_sold = p.get("day_sell_quantity", 0)
            day_change = day_bought  # day_sold already reflected in holdings quantity

            if symbol in positions_dict:
                # Stock in holdings AND traded today — apply only today's delta
                existing = positions_dict[symbol]
                new_qty = existing.quantity + day_change
                merge_diagnostics[symbol] = MergeDiagnostic(
                    holdings_qty=existing.quantity,
                    day_bought=day_bought,
                    day_sold=day_sold,
                    net_qty=new_qty,
                    value=new_qty * p["last_price"] if new_qty > 0 else 0,
                )
                if new_qty > 0:
                    positions_dict[symbol] = Position(
                        symbol=symbol,
                        quantity=new_qty,
                        average_price=existing.average_price,
                        sector=existing.sector,
                        current_price=p["last_price"],
                    )
                else:
                    del positions_dict[symbol]  # Fully sold today
            elif day_change > 0:
                # New position bought today (not in holdings yet)
                stock = self.universe.get_stock(symbol)
                sector = stock.sector if stock else "UNKNOWN"
                positions_dict[symbol] = Position(
                    symbol=symbol,
                    quantity=day_change,
                    average_price=p.get("average_price", 0),
                    sector=sector,
                    current_price=p["last_price"],
                )
                merge_diagnostics[symbol] = MergeDiagnostic(
                    holdings_qty=0,
                    day_bought=day_bought,
                    day_sold=day_sold,
                    net_qty=day_change,
                    value=day_change * p["last_price"],
                )

        # Add diagnostics for holdings-only symbols (no positions activity today)
        for symbol, pos in positions_dict.items():
            if symbol not in merge_diagnostics:
                merge_diagnostics[symbol] = MergeDiagnostic(
                    holdings_qty=pos.quantity,
                    day_bought=0,
                    day_sold=0,
                    net_qty=pos.quantity,
                    value=pos.value,
                )

        # Get cash
        margins = self.kite.margins()
        equity = margins.get("equity", {})
        available = equity.get("available", {})
        cash = available.get("live_balance", 0)

        total_value = cash + sum(p.value for p in positions_dict.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in positions_dict.values())

        self._snapshot = PortfolioSnapshot(
            positions=positions_dict,
            cash=cash,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            merge_diagnostics=merge_diagnostics,
        )
        return self._snapshot


class BacktestPortfolio:
    """
    Portfolio for backtesting that doesn't use API.
    """

    def __init__(self, initial_capital: float, universe: Universe):
        """
        Initialize backtest portfolio.

        Args:
            initial_capital: Starting capital
            universe: Loaded Universe
        """
        self.universe = universe
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.initial_capital = initial_capital
        self._day_start_value = initial_capital

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update position prices with latest market data."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]

    def get_total_value(self) -> float:
        """Get total portfolio value."""
        return self.cash + sum(p.value for p in self.positions.values())

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        sector: str,
    ) -> bool:
        """
        Execute a buy trade.

        Returns:
            True if successful
        """
        cost = quantity * price
        if cost > self.cash:
            return False

        self.cash -= cost

        if symbol in self.positions:
            # Average up
            pos = self.positions[symbol]
            total_qty = pos.quantity + quantity
            total_cost = pos.cost_basis + cost
            pos.quantity = total_qty
            pos.average_price = total_cost / total_qty
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                sector=sector,
                current_price=price,
            )

        return True

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
    ) -> bool:
        """
        Execute a sell trade.

        Returns:
            True if successful
        """
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]
        if quantity > pos.quantity:
            return False

        proceeds = quantity * price
        self.cash += proceeds

        if quantity == pos.quantity:
            del self.positions[symbol]
        else:
            pos.quantity -= quantity

        return True

    def get_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot."""
        total_value = self.get_total_value()
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())

        return PortfolioSnapshot(
            positions=self.positions.copy(),
            cash=self.cash,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            day_start_value=self._day_start_value,
        )

    def start_new_day(self) -> None:
        """Mark start of new trading day."""
        self._day_start_value = self.get_total_value()
