"""
Pure decision logic for rebalance planning — shared by live (rebalance_planner.py)
and backtest (backtest.py) so both paths compute sells/buys identically.

Design:
- Functions take primitive inputs (dicts of symbol→value, prices, lot_sizes) —
  no dependency on Kite, Portfolio, Universe, or InstrumentMapper.
- Functions return minimal intent records (SellIntent, BuyIntent). Callers
  enrich with sector labels, P&L, and build their respective output objects
  (PlannedTrade for live, Trade for backtest).
- Rounding to lot size happens here via the shared utils.calculate_order_quantity;
  tick-price rounding stays with the caller (backtest doesn't have ticks).

Invariants:
- SELL-first-then-BUY sequencing is the caller's responsibility; the functions
  produce independent sell/buy lists.
- Proportional scaling activates when total_buy_value > available_budget
  (sell proceeds + existing cash budget). Lot-size rounding inside the
  scale step may create tiny residuals — these flow into the surplus deploy.
- Surplus deploy priority: equity top-up → gold top-up → cash sweep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .utils import calculate_order_quantity


@dataclass
class SellIntent:
    symbol: str
    quantity: int
    is_exit: bool  # True = full exit (not in target), False = reduction
    current_qty: int
    current_value: float
    target_weight: float  # 0.0 for exits
    reason: str


@dataclass
class BuyIntent:
    symbol: str
    quantity: int
    value: float  # qty × unrounded_price (for budget accounting)
    is_new: bool  # True = new position, False = increase/top-up
    target_weight: float  # effective target weight (after hard-limit capping)
    current_qty: int = 0
    current_value: float = 0.0
    reason: str = ""


# =============================================================================
# PHASE 1: Sell decisions (exits and reductions)
# =============================================================================
def compute_sell_intents(
    current_qtys: Dict[str, int],
    current_values: Dict[str, float],
    target_weights: Dict[str, float],
    managed_capital: float,
    prices: Dict[str, float],
    lot_sizes: Dict[str, int],
    *,
    reduction_tolerance: float = 0.10,
) -> List[SellIntent]:
    """Compute exits (symbol not in target) + reductions (over-weight).

    Args:
        current_qtys: symbol → current held quantity
        current_values: symbol → current position market value
        target_weights: symbol → target weight (0..1)
        managed_capital: total strategy capital (stocks + LIQUIDBEES etc.)
        prices: symbol → latest mark price
        lot_sizes: symbol → lot size (1 for backtest, real lot size for live)
        reduction_tolerance: how far over target we must be before reducing
            (default 10% — i.e. only reduce if current_weight > target * 1.10)

    Returns:
        List of SellIntent in insertion order (iteration order of current_qtys).
    """
    intents: List[SellIntent] = []
    target_symbols = set(target_weights.keys())

    for symbol, qty in current_qtys.items():
        if qty <= 0:
            continue
        current_value = current_values.get(symbol, 0.0)
        current_weight = current_value / managed_capital if managed_capital > 0 else 0.0

        if symbol not in target_symbols:
            intents.append(SellIntent(
                symbol=symbol,
                quantity=qty,
                is_exit=True,
                current_qty=qty,
                current_value=current_value,
                target_weight=0.0,
                reason="Exit: Not in target",
            ))
            continue

        target_weight = target_weights[symbol]
        if current_weight <= target_weight * (1.0 + reduction_tolerance):
            continue  # within tolerance band — no action

        price = prices.get(symbol, 0.0)
        if price <= 0:
            continue

        target_value = managed_capital * target_weight
        reduce_value = current_value - target_value
        lot_size = lot_sizes.get(symbol, 1)
        reduce_qty, _ = calculate_order_quantity(reduce_value, price, lot_size)
        if reduce_qty <= 0:
            continue

        intents.append(SellIntent(
            symbol=symbol,
            quantity=reduce_qty,
            is_exit=False,
            current_qty=qty,
            current_value=current_value,
            target_weight=target_weight,
            reason=f"Reduce: {current_weight:.1%}->{target_weight:.1%}",
        ))

    return intents


# =============================================================================
# PHASE 2: Buy decisions (new positions and increases)
# =============================================================================
def compute_buy_intents(
    current_qtys: Dict[str, int],
    current_values: Dict[str, float],
    target_weights: Dict[str, float],
    managed_capital: float,
    prices: Dict[str, float],
    lot_sizes: Dict[str, int],
    *,
    hard_max_position: float = 1.0,
    increase_tolerance: float = 0.10,
) -> Tuple[List[BuyIntent], List[str]]:
    """Compute new positions + increases (within hard position cap).

    Args:
        hard_max_position: absolute ceiling per position (e.g. 0.12). Target
            weights above this are clamped and a warning added.
        increase_tolerance: only top up if effective_target > current * (1 + tol)

    Returns:
        (intents, warnings)
    """
    intents: List[BuyIntent] = []
    warnings: List[str] = []

    for symbol, target_weight in target_weights.items():
        price = prices.get(symbol)
        if price is None or price <= 0:
            warnings.append(f"No price for {symbol}, skipping")
            continue

        lot_size = lot_sizes.get(symbol, 1)

        if symbol not in current_qtys or current_qtys[symbol] <= 0:
            # New position
            target_value = managed_capital * target_weight
            qty, _ = calculate_order_quantity(target_value, price, lot_size)
            if qty <= 0:
                continue
            intents.append(BuyIntent(
                symbol=symbol,
                quantity=qty,
                value=qty * price,
                is_new=True,
                target_weight=target_weight,
                reason="New position",
            ))
        else:
            # Existing position — check for increase
            current_qty = current_qtys[symbol]
            current_value = current_values.get(symbol, current_qty * price)
            current_weight = current_value / managed_capital if managed_capital > 0 else 0.0

            effective_target_weight = target_weight
            if target_weight > hard_max_position:
                effective_target_weight = hard_max_position
                warnings.append(
                    f"{symbol}: Target {target_weight:.1%} exceeds hard limit "
                    f"{hard_max_position:.0%}, capping to {effective_target_weight:.1%}"
                )

            if effective_target_weight <= current_weight * (1.0 + increase_tolerance):
                continue

            effective_target_value = effective_target_weight * managed_capital
            increase_value = effective_target_value - current_value
            qty, _ = calculate_order_quantity(increase_value, price, lot_size)
            if qty <= 0:
                continue
            intents.append(BuyIntent(
                symbol=symbol,
                quantity=qty,
                value=qty * price,
                is_new=False,
                target_weight=effective_target_weight,
                current_qty=current_qty,
                current_value=current_value,
                reason=f"Increase: {current_weight:.1%}->{effective_target_weight:.1%}",
            ))

    return intents, warnings


# =============================================================================
# Proportional scaling when buys exceed available budget
# =============================================================================
def scale_buys_to_budget(
    buys: List[BuyIntent],
    available_budget: float,
    prices: Dict[str, float],
    lot_sizes: Dict[str, int],
) -> Tuple[List[BuyIntent], Optional[float]]:
    """Scale buy quantities down proportionally if sum(buy.value) > budget.

    Returns:
        (scaled_buys_nonzero_only, scale_factor_or_None)
        scale_factor is None when no scaling was needed.
    """
    total_buy_value = sum(b.value for b in buys)
    if total_buy_value <= available_budget or total_buy_value <= 0:
        return buys, None

    scale_factor = available_budget / total_buy_value
    scaled: List[BuyIntent] = []

    for b in buys:
        original_qty = b.quantity
        scaled_qty = int(b.quantity * scale_factor)
        # Preserve at least 1 unit when scale ≥ 10% and price allows
        if scaled_qty == 0 and original_qty > 0 and scale_factor >= 0.10:
            scaled_qty = 1
        lot_size = lot_sizes.get(b.symbol, 1)
        scaled_qty = (scaled_qty // lot_size) * lot_size
        if scaled_qty <= 0:
            continue

        price = prices.get(b.symbol, b.value / max(b.quantity, 1))
        scaled.append(BuyIntent(
            symbol=b.symbol,
            quantity=scaled_qty,
            value=scaled_qty * price,
            is_new=b.is_new,
            target_weight=b.target_weight,
            current_qty=b.current_qty,
            current_value=b.current_value,
            reason=b.reason,
        ))

    return scaled, scale_factor


# =============================================================================
# PHASE 3: Deploy surplus (equity top-up → gold top-up → cash sweep)
# =============================================================================
def compute_surplus_deploy(
    surplus: float,
    managed_capital: float,
    target_weights: Dict[str, float],
    effective_values: Dict[str, float],  # position value after Phase 1-2
    prices: Dict[str, float],
    lot_sizes: Dict[str, int],
    *,
    gold_symbol: str = "",
    cash_symbol: str = "",
    exclude_symbols: Optional[Set[str]] = None,
) -> Tuple[List[BuyIntent], float]:
    """Deploy remaining surplus: equity top-up → gold top-up → cash sweep.

    Args:
        surplus: cash left over after Phase 1 sells and Phase 2 buys
        effective_values: symbol → position value AFTER applying planned
            Phase 1-2 trades (used to compute residual deficits)
        exclude_symbols: symbols already handled in Phase 2 (don't top up again)

    Returns:
        (extra_buy_intents, remaining_surplus)
    """
    extras: List[BuyIntent] = []
    exclude = exclude_symbols or set()
    if surplus <= 0 or managed_capital <= 0:
        return extras, surplus

    # Step 1: Top up underweight equity positions pro-rata
    equity_deficits: List[Tuple[str, float, float, float]] = []
    for symbol, tw in target_weights.items():
        if symbol in (gold_symbol, cash_symbol):
            continue
        if symbol in exclude:
            continue
        price = prices.get(symbol)
        if not price or price <= 0:
            continue
        eff_value = effective_values.get(symbol, 0.0)
        target_value = managed_capital * tw
        deficit = target_value - eff_value
        if deficit > 0:
            equity_deficits.append((symbol, deficit, price, tw))

    if equity_deficits:
        total_deficit = sum(d for _, d, _, _ in equity_deficits)
        deploy_pool = min(surplus, total_deficit)
        for symbol, deficit, price, tw in equity_deficits:
            if surplus <= 0:
                break
            alloc = deploy_pool * (deficit / total_deficit)
            lot_size = lot_sizes.get(symbol, 1)
            qty, _ = calculate_order_quantity(alloc, price, lot_size)
            if qty <= 0:
                continue
            cost = qty * price
            extras.append(BuyIntent(
                symbol=symbol,
                quantity=qty,
                value=cost,
                is_new=(effective_values.get(symbol, 0.0) <= 0),
                target_weight=tw,
                current_qty=0,  # caller fills from its own state
                current_value=effective_values.get(symbol, 0.0),
                reason="Surplus deploy",
            ))
            surplus -= cost

    # Step 2: Top up underweight gold
    if surplus > 0 and gold_symbol and gold_symbol in target_weights and gold_symbol not in exclude:
        gold_price = prices.get(gold_symbol)
        if gold_price and gold_price > 0:
            eff_gold = effective_values.get(gold_symbol, 0.0)
            gold_target = managed_capital * target_weights[gold_symbol]
            gold_deficit = gold_target - eff_gold
            if gold_deficit > 0:
                alloc = min(surplus, gold_deficit)
                lot_size = lot_sizes.get(gold_symbol, 1)
                qty, _ = calculate_order_quantity(alloc, gold_price, lot_size)
                if qty > 0:
                    cost = qty * gold_price
                    extras.append(BuyIntent(
                        symbol=gold_symbol,
                        quantity=qty,
                        value=cost,
                        is_new=(eff_gold <= 0),
                        target_weight=target_weights[gold_symbol],
                        current_value=eff_gold,
                        reason="Surplus deploy",
                    ))
                    surplus -= cost

    # Step 3: Sweep remainder to cash_symbol (LIQUIDBEES)
    if surplus > 0 and cash_symbol:
        cash_price = prices.get(cash_symbol)
        if cash_price and cash_price > 0:
            lot_size = lot_sizes.get(cash_symbol, 1)
            qty, _ = calculate_order_quantity(surplus, cash_price, lot_size)
            if qty > 0:
                cost = qty * cash_price
                eff_cash = effective_values.get(cash_symbol, 0.0)
                extras.append(BuyIntent(
                    symbol=cash_symbol,
                    quantity=qty,
                    value=cost,
                    is_new=(eff_cash <= 0),
                    target_weight=0.0,
                    current_value=eff_cash,
                    reason="Cash sweep",
                ))
                surplus -= cost

    return extras, surplus


__all__ = [
    "SellIntent",
    "BuyIntent",
    "compute_sell_intents",
    "compute_buy_intents",
    "scale_buys_to_budget",
    "compute_surplus_deploy",
]
