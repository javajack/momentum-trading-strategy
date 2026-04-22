"""
Rebalance planner — computes trade plans without placing any orders.

Post-April-2026 Zerodha policy requires static-IP whitelisting for order
placement. This codebase no longer calls `kite.place_order`. Instead, the
planner produces a deterministic SELL-first / BUY-after trade list that
humans execute manually from the Kite dashboard (or import as a basket CSV
via fortress.plan_render).

Invariants preserved from the old executor:
- R9: Sells-before-buys (cash-neutral sequencing)
- Sector-aware ordering and position-count tracking
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .config import RiskConfig
from .instruments import InstrumentMapper
from .portfolio import Portfolio, Position
from .utils import calculate_order_quantity, format_currency


class TradeAction(Enum):
    """Type of trade action in rebalance plan."""

    SELL_EXIT = "sell_exit"  # Full position exit
    SELL_REDUCE = "sell_reduce"  # Partial reduction
    BUY_NEW = "buy_new"  # New position
    BUY_INCREASE = "buy_increase"  # Increase existing


@dataclass
class PlannedTrade:
    """A single trade in the rebalance plan."""

    symbol: str
    action: TradeAction
    quantity: int
    price: float
    value: float
    sector: str
    current_qty: int = 0
    target_weight: float = 0.0
    current_weight: float = 0.0
    reason: str = ""
    entry_price: float = 0.0  # For P&L calculation on sells

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell trade."""
        return self.action in (TradeAction.SELL_EXIT, TradeAction.SELL_REDUCE)

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.action in (TradeAction.BUY_NEW, TradeAction.BUY_INCREASE)

    @property
    def pnl_absolute(self) -> float:
        """Absolute P&L for sell trades."""
        if self.is_sell and self.entry_price > 0:
            return (self.price - self.entry_price) * self.quantity
        return 0.0

    @property
    def pnl_percent(self) -> float:
        """Percentage P&L for sell trades."""
        if self.is_sell and self.entry_price > 0:
            return (self.price - self.entry_price) / self.entry_price
        return 0.0


@dataclass
class RebalancePlan:
    """Complete rebalance execution plan."""

    trades: List[PlannedTrade] = field(default_factory=list)
    total_sell_value: float = 0.0
    total_buy_value: float = 0.0
    net_cash_needed: float = 0.0
    available_cash: float = 0.0
    margin_sufficient: bool = True
    warnings: List[str] = field(default_factory=list)

    @property
    def sell_trades(self) -> List[PlannedTrade]:
        """Get sell trades sorted by value (largest first)."""
        return sorted(
            [t for t in self.trades if t.is_sell],
            key=lambda x: x.value,
            reverse=True,
        )

    @property
    def buy_new_trades(self) -> List[PlannedTrade]:
        """Get new position buys sorted by value (largest first)."""
        return sorted(
            [t for t in self.trades if t.action == TradeAction.BUY_NEW],
            key=lambda x: x.value,
            reverse=True,
        )

    @property
    def buy_increase_trades(self) -> List[PlannedTrade]:
        """Get position increase buys sorted by value (largest first)."""
        return sorted(
            [t for t in self.trades if t.action == TradeAction.BUY_INCREASE],
            key=lambda x: x.value,
            reverse=True,
        )


class RebalancePlanner:
    """Builds a RebalancePlan from target weights and current holdings.

    Pure computation — no side effects, no API calls. Every call to
    ``build_plan`` is independent.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        instrument_mapper: InstrumentMapper,
        universe,
        risk_config: RiskConfig = None,
    ):
        """
        Args:
            portfolio: Portfolio (for sector / position lookups)
            instrument_mapper: InstrumentMapper (resolves sector labels for hedges)
            universe: Universe instance
            risk_config: Risk configuration for position limits
        """
        self.portfolio = portfolio
        self.mapper = instrument_mapper
        self.universe = universe
        self.risk_config = risk_config or RiskConfig()

    def build_plan(
        self,
        target_weights: Dict[str, float],
        current_holdings: Dict[str, Position],
        managed_capital: float,
        current_prices: Dict[str, float],
        gold_symbol: str = "",
        cash_symbol: str = "",
    ) -> RebalancePlan:
        """
        Build execution plan from target weights.

        LIQUIDBEES (cash_symbol) is the capital pool — sells fund buys,
        surplus sweeps back to LIQUIDBEES. No demat cash dependency.

        Args:
            target_weights: Dict of symbol -> target weight
            current_holdings: Dict of symbol -> Position
            managed_capital: Total capital being managed (incl LIQUIDBEES)
            current_prices: Dict of symbol -> current price
            gold_symbol: Gold ETF symbol (e.g. GOLDBEES)
            cash_symbol: Cash ETF symbol (e.g. LIQUIDBEES) for surplus sweep

        Returns:
            RebalancePlan with all trades
        """
        plan = RebalancePlan()
        plan.available_cash = self.portfolio.get_snapshot().cash

        from . import rebalance_logic as _rl

        # Input projection — shared logic consumes primitives, not Position objects.
        all_symbols = set(current_holdings.keys()) | set(target_weights.keys())
        current_qtys = {s: current_holdings[s].quantity for s in current_holdings}
        current_values = {s: current_holdings[s].value for s in current_holdings}
        lot_sizes = {s: self.mapper.get_lot_size(s) for s in all_symbols}

        def _sector_for(symbol: str) -> str:
            if symbol in current_holdings:
                return current_holdings[symbol].sector
            stock = self.universe.get_stock(symbol)
            return stock.sector if stock else "UNKNOWN"

        # Phase 1: SELL orders (exits + reductions)
        sell_intents = _rl.compute_sell_intents(
            current_qtys=current_qtys,
            current_values=current_values,
            target_weights=target_weights,
            managed_capital=managed_capital,
            prices=current_prices,
            lot_sizes=lot_sizes,
        )
        for s in sell_intents:
            pos = current_holdings[s.symbol]
            price = current_prices.get(s.symbol, pos.current_price)
            action = TradeAction.SELL_EXIT if s.is_exit else TradeAction.SELL_REDUCE
            current_weight = pos.value / managed_capital if managed_capital > 0 else 0
            plan.trades.append(PlannedTrade(
                symbol=s.symbol,
                action=action,
                quantity=s.quantity,
                price=price,
                value=s.quantity * price,
                sector=pos.sector,
                current_qty=pos.quantity,
                current_weight=current_weight,
                target_weight=s.target_weight,
                reason=s.reason,
                entry_price=pos.average_price,
            ))
            plan.total_sell_value += s.quantity * price

        # Phase 2: BUY orders (new positions + increases)
        buy_intents, buy_warnings = _rl.compute_buy_intents(
            current_qtys=current_qtys,
            current_values=current_values,
            target_weights=target_weights,
            managed_capital=managed_capital,
            prices=current_prices,
            lot_sizes=lot_sizes,
            hard_max_position=self.risk_config.hard_max_position,
        )
        plan.warnings.extend(buy_warnings)
        for b in buy_intents:
            price = current_prices[b.symbol]
            rounded_price = self.mapper.round_to_tick(price, b.symbol)
            sector = _sector_for(b.symbol)
            cq = current_qtys.get(b.symbol, 0)
            cw = (current_values.get(b.symbol, 0.0) / managed_capital) if managed_capital > 0 else 0.0
            plan.trades.append(PlannedTrade(
                symbol=b.symbol,
                action=(TradeAction.BUY_NEW if b.is_new else TradeAction.BUY_INCREASE),
                quantity=b.quantity,
                price=rounded_price,
                value=b.value,
                sector=sector,
                current_qty=cq,
                current_weight=cw,
                target_weight=b.target_weight,
                reason=b.reason,
            ))
            plan.total_buy_value += b.value

        # Self-funding: buys funded entirely from sell proceeds
        available_for_buys = plan.total_sell_value
        scaled_buys, scale_factor = _rl.scale_buys_to_budget(
            buy_intents, available_for_buys, current_prices, lot_sizes,
        )
        scaled = scale_factor is not None
        if scaled:
            plan.warnings.append(
                f"Scaling buys to {scale_factor:.0%} to fit available funds "
                f"({format_currency(available_for_buys)})"
            )
            # Rewrite quantities/values on the PlannedTrade buy rows to match scaled_buys.
            scaled_map = {b.symbol: b for b in scaled_buys}
            kept: List[PlannedTrade] = []
            new_total_buy = 0.0
            for trade in plan.trades:
                if not trade.is_buy:
                    kept.append(trade)
                    continue
                scaled_b = scaled_map.get(trade.symbol)
                if scaled_b is None:
                    continue  # dropped by scaler (qty went to 0)
                trade.quantity = scaled_b.quantity
                trade.value = scaled_b.quantity * trade.price
                kept.append(trade)
                new_total_buy += trade.value
            plan.trades = kept
            plan.total_buy_value = new_total_buy

        # Phase 3: Deploy surplus to keep capital fully allocated
        # Priority: equity top-ups → gold top-up → cash_symbol sweep
        surplus = available_for_buys - plan.total_buy_value
        if surplus > 0 and cash_symbol and managed_capital > 0:
            # Effective values AFTER Phase 1-2 trades — used to compute deficits.
            effective_values: Dict[str, float] = {}
            for sym in (set(current_holdings.keys()) | set(target_weights.keys())):
                val = current_holdings[sym].value if sym in current_holdings else 0.0
                for t in plan.trades:
                    if t.symbol == sym:
                        if t.is_buy:
                            val += t.value
                        elif t.is_sell:
                            val -= t.value
                effective_values[sym] = max(0.0, val)

            phase2_buy_symbols = {t.symbol for t in plan.trades if t.is_buy}
            extras, surplus = _rl.compute_surplus_deploy(
                surplus=surplus,
                managed_capital=managed_capital,
                target_weights=target_weights,
                effective_values=effective_values,
                prices=current_prices,
                lot_sizes=lot_sizes,
                gold_symbol=gold_symbol,
                cash_symbol=cash_symbol,
                exclude_symbols=phase2_buy_symbols,
            )
            for e in extras:
                price = current_prices[e.symbol]
                rounded_price = self.mapper.round_to_tick(price, e.symbol)
                if e.symbol in current_holdings:
                    pos = current_holdings[e.symbol]
                    sector, cq, cw = pos.sector, pos.quantity, pos.value / managed_capital
                elif e.symbol == gold_symbol:
                    sector, cq, cw = "Hedge", 0, 0.0
                elif e.symbol == cash_symbol:
                    sector, cq, cw = "Cash", 0, 0.0
                else:
                    sector = _sector_for(e.symbol)
                    cq, cw = 0, 0.0
                action = (
                    TradeAction.BUY_INCREASE
                    if e.symbol in current_holdings
                    else TradeAction.BUY_NEW
                )
                plan.trades.append(PlannedTrade(
                    symbol=e.symbol,
                    action=action,
                    quantity=e.quantity,
                    price=rounded_price,
                    value=e.value,
                    sector=sector,
                    current_qty=cq,
                    current_weight=cw,
                    target_weight=e.target_weight,
                    reason=e.reason,
                ))
                plan.total_buy_value += e.value

        # Policy: demat cash is OFF-LIMITS to the strategy. To increase
        # exposure, user manually buys LIQUIDBEES — next rebalance then
        # SELL_EXITs LIQUIDBEES and deploys proceeds into stocks. Strategy
        # never touches idle demat cash. See CLAUDE.md Capital Model.

        # Calculate net cash impact
        plan.net_cash_needed = plan.total_buy_value - plan.total_sell_value

        # Self-funding: buys always fit within sell proceeds (already scaled above)
        plan.margin_sufficient = True
        if scaled:
            plan.warnings.append(
                f"Buys scaled to match sell proceeds ({format_currency(plan.total_sell_value)})"
            )

        return plan

    def ordered_trades(self, plan: RebalancePlan) -> List[PlannedTrade]:
        """Return trades in safe execution order (SELL → BUY_NEW → BUY_INCREASE).

        This ordering is cash-neutral: proceeds from sells fund buys, same
        sequencing the old executor used when it placed orders live.
        """
        return plan.sell_trades + plan.buy_new_trades + plan.buy_increase_trades
