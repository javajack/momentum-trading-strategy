"""
Rebalance execution bridge - connects rebalance calculation with order placement.

Enforces invariants:
- O1: Dry-run is default mode
- O2: Live orders require explicit confirmation
- R9: Sells execute before buys
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from .config import RiskConfig
from .order_manager import Order, OrderManager, OrderResult, OrderType
from .portfolio import Portfolio, Position
from .instruments import InstrumentMapper
from .utils import calculate_order_quantity, format_currency


class TradeAction(Enum):
    """Type of trade action in rebalance plan."""

    SELL_EXIT = "sell_exit"       # Full position exit
    SELL_REDUCE = "sell_reduce"   # Partial reduction
    BUY_NEW = "buy_new"           # New position
    BUY_INCREASE = "buy_increase" # Increase existing


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


@dataclass
class ExecutionResult:
    """Results of plan execution."""

    successes: List[OrderResult] = field(default_factory=list)
    failures: List[OrderResult] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        """Check if all orders succeeded."""
        return len(self.failures) == 0


class RebalanceExecutor:
    """Bridges rebalance calculation with order execution."""

    def __init__(
        self,
        kite,
        portfolio: Portfolio,
        instrument_mapper: InstrumentMapper,
        order_manager: OrderManager,
        universe,
        risk_config: RiskConfig = None,
    ):
        """
        Initialize rebalance executor.

        Args:
            kite: Authenticated KiteConnect instance
            portfolio: Portfolio instance
            instrument_mapper: InstrumentMapper instance
            order_manager: OrderManager instance
            universe: Universe instance
            risk_config: Risk configuration for position limits
        """
        self.kite = kite
        self.portfolio = portfolio
        self.mapper = instrument_mapper
        self.order_manager = order_manager
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

        target_symbols = set(target_weights.keys())
        current_symbols = set(current_holdings.keys())

        # Phase 1: SELL orders (exits and reductions)
        for symbol in current_symbols:
            pos = current_holdings[symbol]
            price = current_prices.get(symbol, pos.current_price)
            current_weight = pos.value / managed_capital if managed_capital > 0 else 0

            if symbol not in target_symbols:
                # Full exit
                trade = PlannedTrade(
                    symbol=symbol,
                    action=TradeAction.SELL_EXIT,
                    quantity=pos.quantity,
                    price=price,
                    value=pos.quantity * price,
                    sector=pos.sector,
                    current_qty=pos.quantity,
                    current_weight=current_weight,
                    reason="Exit: Not in target",
                    entry_price=pos.average_price,
                )
                plan.trades.append(trade)
                plan.total_sell_value += trade.value
            else:
                # Check for reduction (10% tolerance)
                target_weight = target_weights[symbol]
                if current_weight > target_weight * 1.10:
                    target_value = managed_capital * target_weight
                    reduce_value = pos.value - target_value
                    lot_size = self.mapper.get_lot_size(symbol)
                    reduce_qty, _ = calculate_order_quantity(reduce_value, price, lot_size)
                    if reduce_qty > 0:
                        trade = PlannedTrade(
                            symbol=symbol,
                            action=TradeAction.SELL_REDUCE,
                            quantity=reduce_qty,
                            price=price,
                            value=reduce_qty * price,
                            sector=pos.sector,
                            current_qty=pos.quantity,
                            current_weight=current_weight,
                            target_weight=target_weight,
                            reason=f"Reduce: {current_weight:.1%}->{target_weight:.1%}",
                            entry_price=pos.average_price,
                        )
                        plan.trades.append(trade)
                        plan.total_sell_value += trade.value

        # Phase 2: BUY orders (new positions and increases)
        for symbol in target_symbols:
            target_weight = target_weights[symbol]
            target_value = managed_capital * target_weight
            price = current_prices.get(symbol)

            if price is None or price <= 0:
                plan.warnings.append(f"No price for {symbol}, skipping")
                continue

            lot_size = self.mapper.get_lot_size(symbol)
            rounded_price = self.mapper.round_to_tick(price, symbol)

            if symbol not in current_symbols:
                # New position
                qty, _ = calculate_order_quantity(target_value, price, lot_size)
                if qty > 0:
                    stock = self.universe.get_stock(symbol)
                    sector = stock.sector if stock else "UNKNOWN"
                    trade = PlannedTrade(
                        symbol=symbol,
                        action=TradeAction.BUY_NEW,
                        quantity=qty,
                        price=rounded_price,
                        value=qty * price,
                        sector=sector,
                        target_weight=target_weight,
                        reason="New position",
                    )
                    plan.trades.append(trade)
                    plan.total_buy_value += trade.value
            else:
                # Check for increase (10% tolerance)
                pos = current_holdings[symbol]
                current_weight = pos.value / managed_capital if managed_capital > 0 else 0

                # Enforce hard limit on target weight
                hard_limit = self.risk_config.hard_max_position
                effective_target_weight = target_weight
                if target_weight > hard_limit:
                    effective_target_weight = hard_limit
                    plan.warnings.append(
                        f"{symbol}: Target {target_weight:.1%} exceeds hard limit "
                        f"{hard_limit:.0%}, capping to {effective_target_weight:.1%}"
                    )

                if effective_target_weight > current_weight * 1.10:
                    effective_target_value = effective_target_weight * managed_capital
                    increase_value = effective_target_value - pos.value
                    qty, _ = calculate_order_quantity(increase_value, price, lot_size)
                    if qty > 0:
                        trade = PlannedTrade(
                            symbol=symbol,
                            action=TradeAction.BUY_INCREASE,
                            quantity=qty,
                            price=rounded_price,
                            value=qty * price,
                            sector=pos.sector,
                            current_qty=pos.quantity,
                            current_weight=current_weight,
                            target_weight=effective_target_weight,
                            reason=f"Increase: {current_weight:.1%}->{effective_target_weight:.1%}",
                        )
                        plan.trades.append(trade)
                        plan.total_buy_value += trade.value

        # Self-funding: buys funded entirely from sell proceeds (LIQUIDBEES + exits + reductions)
        available_for_buys = plan.total_sell_value

        scaled = False
        if plan.total_buy_value > available_for_buys and plan.total_buy_value > 0:
            scaled = True
            scale_factor = available_for_buys / plan.total_buy_value
            plan.warnings.append(
                f"Scaling buys to {scale_factor:.0%} to fit available funds "
                f"({format_currency(available_for_buys)})"
            )

            # Scale down each buy trade
            new_total_buy = 0.0
            for trade in plan.trades:
                if trade.is_buy:
                    # Scale quantity down
                    original_qty = trade.quantity
                    scaled_qty = int(trade.quantity * scale_factor)
                    # Ensure at least 1 share if scale is meaningful (>= 10%)
                    if scaled_qty == 0 and original_qty > 0 and scale_factor >= 0.10:
                        scaled_qty = 1
                    # Round to lot size
                    lot_size = self.mapper.get_lot_size(trade.symbol)
                    scaled_qty = (scaled_qty // lot_size) * lot_size

                    if scaled_qty > 0:
                        trade.quantity = scaled_qty
                        trade.value = scaled_qty * trade.price
                        new_total_buy += trade.value
                    else:
                        # Remove trades with 0 quantity
                        trade.quantity = 0
                        trade.value = 0

            # Remove zero-quantity trades
            plan.trades = [t for t in plan.trades if t.quantity > 0]
            plan.total_buy_value = new_total_buy

        # Phase 3: Deploy surplus to keep capital fully allocated
        # Priority: equity top-ups → gold top-up → cash_symbol sweep
        surplus = available_for_buys - plan.total_buy_value
        if surplus > 0 and cash_symbol and managed_capital > 0:
            # Symbols that already have buy trades from Phase 2
            phase2_buy_symbols = {t.symbol for t in plan.trades if t.is_buy}

            # Compute effective current values after planned Phase 1-2 trades
            def _effective_value(sym: str) -> float:
                val = current_holdings[sym].value if sym in current_holdings else 0.0
                for t in plan.trades:
                    if t.symbol == sym:
                        if t.is_buy:
                            val += t.value
                        elif t.is_sell:
                            val -= t.value
                return max(0.0, val)

            # Step 1: Top up underweight equity positions pro-rata
            equity_deficits = []
            for symbol, tw in target_weights.items():
                if symbol in (gold_symbol, cash_symbol):
                    continue
                if symbol in phase2_buy_symbols:
                    continue  # Phase 2 already handled this
                price = current_prices.get(symbol)
                if not price or price <= 0:
                    continue
                eff_value = _effective_value(symbol)
                target_value = managed_capital * tw
                deficit = target_value - eff_value
                if deficit > 0:
                    equity_deficits.append((symbol, deficit, price, tw))

            if equity_deficits:
                total_deficit = sum(d for _, d, _, _ in equity_deficits)
                # Cap total deployment to surplus (deploy pro-rata by deficit)
                deploy_pool = min(surplus, total_deficit)
                for symbol, deficit, price, tw in equity_deficits:
                    if surplus <= 0:
                        break
                    alloc = deploy_pool * (deficit / total_deficit)
                    lot_size = self.mapper.get_lot_size(symbol)
                    qty, _ = calculate_order_quantity(alloc, price, lot_size)
                    if qty > 0:
                        cost = qty * price
                        rounded_price = self.mapper.round_to_tick(price, symbol)
                        if symbol in current_holdings:
                            pos = current_holdings[symbol]
                            sector = pos.sector
                            cq = pos.quantity
                            cw = pos.value / managed_capital
                        else:
                            stock = self.universe.get_stock(symbol)
                            sector = stock.sector if stock else "UNKNOWN"
                            cq = 0
                            cw = 0.0
                        action = TradeAction.BUY_INCREASE if symbol in current_holdings else TradeAction.BUY_NEW
                        plan.trades.append(PlannedTrade(
                            symbol=symbol, action=action, quantity=qty,
                            price=rounded_price, value=cost, sector=sector,
                            current_qty=cq, current_weight=cw,
                            target_weight=tw, reason="Surplus deploy",
                        ))
                        plan.total_buy_value += cost
                        surplus -= cost

            # Step 2: Top up underweight gold
            if surplus > 0 and gold_symbol and gold_symbol in target_weights:
                if gold_symbol not in phase2_buy_symbols:
                    gold_price = current_prices.get(gold_symbol)
                    if gold_price and gold_price > 0:
                        eff_gold = _effective_value(gold_symbol)
                        gold_target = managed_capital * target_weights[gold_symbol]
                        gold_deficit = gold_target - eff_gold
                        if gold_deficit > 0:
                            alloc = min(surplus, gold_deficit)
                            lot_size = self.mapper.get_lot_size(gold_symbol)
                            qty, _ = calculate_order_quantity(alloc, gold_price, lot_size)
                            if qty > 0:
                                cost = qty * gold_price
                                rounded_price = self.mapper.round_to_tick(gold_price, gold_symbol)
                                if gold_symbol in current_holdings:
                                    pos = current_holdings[gold_symbol]
                                    sector, cq, cw = pos.sector, pos.quantity, pos.value / managed_capital
                                else:
                                    sector, cq, cw = "Hedge", 0, 0.0
                                action = TradeAction.BUY_INCREASE if gold_symbol in current_holdings else TradeAction.BUY_NEW
                                plan.trades.append(PlannedTrade(
                                    symbol=gold_symbol, action=action, quantity=qty,
                                    price=rounded_price, value=cost, sector=sector,
                                    current_qty=cq, current_weight=cw,
                                    target_weight=target_weights[gold_symbol],
                                    reason="Surplus deploy",
                                ))
                                plan.total_buy_value += cost
                                surplus -= cost

            # Step 3: Sweep remainder to cash_symbol (LIQUIDBEES)
            if surplus > 0:
                cash_price = current_prices.get(cash_symbol)
                if cash_price and cash_price > 0:
                    lot_size = self.mapper.get_lot_size(cash_symbol)
                    qty, _ = calculate_order_quantity(surplus, cash_price, lot_size)
                    if qty > 0:
                        cost = qty * cash_price
                        rounded_price = self.mapper.round_to_tick(cash_price, cash_symbol)
                        if cash_symbol in current_holdings:
                            pos = current_holdings[cash_symbol]
                            sector, cq, cw = pos.sector, pos.quantity, pos.value / managed_capital
                        else:
                            sector, cq, cw = "Cash", 0, 0.0
                        action = TradeAction.BUY_INCREASE if cash_symbol in current_holdings else TradeAction.BUY_NEW
                        plan.trades.append(PlannedTrade(
                            symbol=cash_symbol, action=action, quantity=qty,
                            price=rounded_price, value=cost, sector=sector,
                            current_qty=cq, current_weight=cw,
                            target_weight=0.0, reason="Cash sweep",
                        ))
                        plan.total_buy_value += cost
                        surplus -= cost

        # Calculate net cash impact
        plan.net_cash_needed = plan.total_buy_value - plan.total_sell_value

        # Self-funding: buys always fit within sell proceeds (already scaled above)
        plan.margin_sufficient = True
        if scaled:
            plan.warnings.append(
                f"Buys scaled to match sell proceeds ({format_currency(plan.total_sell_value)})"
            )

        return plan

    def execute_plan(
        self,
        plan: RebalancePlan,
        progress_callback: Optional[Callable] = None,
    ) -> ExecutionResult:
        """
        Execute the rebalance plan with proper sequencing.

        R9: Sells execute before buys.

        Args:
            plan: RebalancePlan to execute
            progress_callback: Optional callback(order, result) for progress updates

        Returns:
            ExecutionResult with successes and failures
        """
        result = ExecutionResult()

        # Get current state for validation
        snapshot = self.portfolio.get_snapshot()
        sector_exposures = {}
        position_values = {}

        # External ETFs not managed by strategy - exclude from position count
        external_etfs = {
            "NIFTYBEES", "JUNIORBEES", "MID150BEES", "HDFCSML250",
            "HANGSENGBEES", "HNGSNGBEES", "LIQUIDCASE", "LIQUIDETF",
        }

        # Track managed position count (excluding external ETFs)
        managed_positions = set()
        for symbol, pos in snapshot.positions.items():
            sector_exposures[pos.sector] = sector_exposures.get(pos.sector, 0) + pos.value
            position_values[symbol] = pos.value
            if symbol not in external_etfs:
                managed_positions.add(symbol)

        current_position_count = len(managed_positions)

        # Execute in order: Sells -> New buys -> Increases (R9)
        all_trades = plan.sell_trades + plan.buy_new_trades + plan.buy_increase_trades

        for trade in all_trades:
            order_type = OrderType.SELL if trade.is_sell else OrderType.BUY
            order = self.order_manager.create_order(
                symbol=trade.symbol,
                order_type=order_type,
                quantity=trade.quantity,
                price=trade.price,
                sector=trade.sector,
            )

            order_result = self.order_manager.place_order(
                order=order,
                portfolio_value=snapshot.total_value,
                current_position_value=position_values.get(trade.symbol, 0),
                current_sector_value=sector_exposures.get(trade.sector, 0),
                current_positions=current_position_count,
            )

            if progress_callback:
                progress_callback(order, order_result)

            if order_result.success:
                result.successes.append(order_result)
                # Update tracking for subsequent orders
                if trade.is_sell:
                    sector_exposures[trade.sector] = max(
                        0, sector_exposures.get(trade.sector, 0) - trade.value
                    )
                    old_value = position_values.get(trade.symbol, 0)
                    new_value = max(0, old_value - trade.value)
                    position_values[trade.symbol] = new_value
                    # If position fully sold, decrement count
                    if old_value > 0 and new_value == 0 and trade.symbol in managed_positions:
                        managed_positions.discard(trade.symbol)
                        current_position_count = len(managed_positions)
                else:
                    sector_exposures[trade.sector] = (
                        sector_exposures.get(trade.sector, 0) + trade.value
                    )
                    old_value = position_values.get(trade.symbol, 0)
                    position_values[trade.symbol] = old_value + trade.value
                    # If new position, increment count
                    if old_value == 0 and trade.symbol not in external_etfs:
                        managed_positions.add(trade.symbol)
                        current_position_count = len(managed_positions)
            else:
                result.failures.append(order_result)

        return result
