"""
Portfolio tracking for FORTRESS MOMENTUM.

Enforces invariant:
- R6: Margin check before buy orders
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .universe import Universe

logger = logging.getLogger(__name__)

# Kite's RMS/margins endpoint is unavailable post-April 2026 policy for
# non-whitelisted IPs. Warn once per process, not every refresh.
_MARGINS_WARNED = False


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
    """Point-in-time snapshot of portfolio state.

    `positions` holds only strategy-managed symbols (in universe or a registered
    hedge like GOLDBEES/LIQUIDBEES). `external_positions` holds everything else
    the user happens to own (stray ETFs, stocks outside the universe, etc.).
    The strategy never trades or rebalances external symbols.
    """

    positions: Dict[str, Position]
    cash: float
    total_value: float
    unrealized_pnl: float
    day_start_value: float = 0.0
    merge_diagnostics: Dict[str, MergeDiagnostic] = field(default_factory=dict)
    external_positions: Dict[str, Position] = field(default_factory=dict)

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

    def _safe_margins_cash(self) -> float:
        """Return broker's live cash balance, or 0 if RMS endpoint is unavailable.

        Zerodha's margins/RMS endpoint returns `UNKNOWN_REQUEST` for non-whitelisted
        IPs post-April 2026. The strategy's capital pool is LIQUIDBEES, not demat
        cash, so a 0 fallback is functionally safe — it just means `total_value`
        won't include any uninvested broker balance.
        """
        global _MARGINS_WARNED
        try:
            margins = self.kite.margins()
            equity = margins.get("equity", {})
            available = equity.get("available", {})
            return float(available.get("live_balance", 0) or 0)
        except Exception as e:
            if not _MARGINS_WARNED:
                logger.warning(
                    "Kite margins() unavailable (%s); using cash=0. "
                    "Strategy uses LIQUIDBEES as its capital pool.", str(e)[:120]
                )
                _MARGINS_WARNED = True
            return 0.0

    def _build_position(self, symbol: str, qty: int, avg_price: float, ltp: float) -> Position:
        """Construct a Position with the correct sector label from the universe."""
        stock = self.universe.get_stock(symbol)
        sector = stock.sector if stock else "UNKNOWN"
        return Position(
            symbol=symbol,
            quantity=qty,
            average_price=avg_price,
            sector=sector,
            current_price=ltp,
        )

    def _place_position(
        self,
        symbol: str,
        pos: Position,
        managed: Dict[str, Position],
        external: Dict[str, Position],
    ) -> None:
        """Route a position into the managed or external bucket based on universe membership."""
        if self.universe.is_managed_symbol(symbol):
            managed[symbol] = pos
        else:
            external[symbol] = pos

    def load_holdings(self) -> PortfolioSnapshot:
        """
        Load current holdings from Zerodha, split into managed vs external.

        Returns:
            Current PortfolioSnapshot
        """
        holdings = self.kite.holdings()
        managed: Dict[str, Position] = {}
        external: Dict[str, Position] = {}

        for h in holdings:
            symbol = h["tradingsymbol"]

            # Include both settled and T1 (unsettled) quantities
            total_qty = h["quantity"] + h.get("t1_quantity", 0)
            if total_qty == 0:
                continue

            pos = self._build_position(symbol, total_qty, h["average_price"], h["last_price"])
            self._place_position(symbol, pos, managed, external)

        cash = self._safe_margins_cash()

        invested_total = sum(p.value for p in managed.values()) + sum(
            p.value for p in external.values()
        )
        total_value = cash + invested_total
        unrealized_pnl = sum(p.unrealized_pnl for p in managed.values())

        self._snapshot = PortfolioSnapshot(
            positions=managed,
            cash=cash,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            external_positions=external,
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

        Post-April-2026 Zerodha policy hides RMS margins from non-whitelisted
        IPs. When margins() fails, this degrades to (True, 0.0) so callers
        don't error — the strategy no longer places orders anyway, and any
        surviving callers just receive advisory output.

        Args:
            required_amount: Amount needed for the buy order

        Returns:
            Tuple of (has_margin, available_margin)
        """
        live_balance = self._safe_margins_cash()
        if live_balance == 0:
            return (True, 0.0)
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

        Combines both and then classifies into strategy-managed vs external
        so downstream code never has to re-filter.
        """
        holdings = self.kite.holdings()
        day_positions = self.kite.positions()
        net_positions = day_positions.get("net", [])

        # Single working dict during merge — split into managed/external at the end.
        merged: Dict[str, Position] = {}

        for h in holdings:
            symbol = h["tradingsymbol"]
            total_qty = h["quantity"] + h.get("t1_quantity", 0)
            if total_qty == 0:
                continue
            merged[symbol] = self._build_position(
                symbol, total_qty, h["average_price"], h["last_price"]
            )

        merge_diagnostics: Dict[str, MergeDiagnostic] = {}

        # Overlay today's CNC positions (intraday delta on top of settled holdings).
        for p in net_positions:
            if p["product"] != "CNC":
                continue
            symbol = p["tradingsymbol"]

            day_bought = p.get("day_buy_quantity", 0)
            day_sold = p.get("day_sell_quantity", 0)
            day_change = day_bought  # day_sold already reflected in holdings quantity

            if symbol in merged:
                existing = merged[symbol]
                new_qty = existing.quantity + day_change
                merge_diagnostics[symbol] = MergeDiagnostic(
                    holdings_qty=existing.quantity,
                    day_bought=day_bought,
                    day_sold=day_sold,
                    net_qty=new_qty,
                    value=new_qty * p["last_price"] if new_qty > 0 else 0,
                )
                if new_qty > 0:
                    merged[symbol] = Position(
                        symbol=symbol,
                        quantity=new_qty,
                        average_price=existing.average_price,
                        sector=existing.sector,
                        current_price=p["last_price"],
                    )
                else:
                    del merged[symbol]
            elif day_change > 0:
                merged[symbol] = self._build_position(
                    symbol, day_change, p.get("average_price", 0), p["last_price"]
                )
                merge_diagnostics[symbol] = MergeDiagnostic(
                    holdings_qty=0,
                    day_bought=day_bought,
                    day_sold=day_sold,
                    net_qty=day_change,
                    value=day_change * p["last_price"],
                )

        for symbol, pos in merged.items():
            if symbol not in merge_diagnostics:
                merge_diagnostics[symbol] = MergeDiagnostic(
                    holdings_qty=pos.quantity,
                    day_bought=0,
                    day_sold=0,
                    net_qty=pos.quantity,
                    value=pos.value,
                )

        # Classify: strategy-managed symbols go into `positions`; everything else
        # (stray ETFs, non-universe stocks the user holds) goes into `external_positions`.
        managed: Dict[str, Position] = {}
        external: Dict[str, Position] = {}
        for symbol, pos in merged.items():
            self._place_position(symbol, pos, managed, external)

        cash = self._safe_margins_cash()

        total_value = cash + sum(p.value for p in managed.values()) + sum(
            p.value for p in external.values()
        )
        unrealized_pnl = sum(p.unrealized_pnl for p in managed.values())

        self._snapshot = PortfolioSnapshot(
            positions=managed,
            cash=cash,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            merge_diagnostics=merge_diagnostics,
            external_positions=external,
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
