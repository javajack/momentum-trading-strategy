"""
Rebalance plan rendering — console tables and Kite-compatible basket CSVs.

Separates *what trades to make* (fortress.rebalance_planner) from *how to
show them* (this module). Two outputs:

  1. ``render_console(plan)`` — a Rich table summarizing the plan, intended
     for visual review during a CLI session.

  2. ``write_basket_csv(plan, path)`` — a CSV that can be imported into
     Zerodha Kite's basket-orders UI (or used as a checklist for manual entry).
     Column names follow Kite's `place_order` parameter names so the file
     round-trips through automations that target the same schema.

Post-April-2026 Zerodha policy removed non-whitelisted order placement, so
the planner never talks to the broker — the CSV is the hand-off.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

from rich.console import Console
from rich.table import Table

from .rebalance_planner import PlannedTrade, RebalancePlan
from .utils import format_currency, format_percentage

# Kite basket import columns. Exchange and product default to NSE/CNC for
# the delivery-style rebalance this strategy runs.
_CSV_COLUMNS = [
    "seq",
    "transaction_type",
    "tradingsymbol",
    "exchange",
    "quantity",
    "order_type",
    "product",
    "price",
    "estimated_value",
    "reason",
]


def _iter_trades_in_order(plan: RebalancePlan) -> Iterable[PlannedTrade]:
    """Planned trades in cash-neutral execution order."""
    return plan.sell_trades + plan.buy_new_trades + plan.buy_increase_trades


def render_console(
    plan: RebalancePlan,
    console: Optional[Console] = None,
    title: str = "Rebalance plan",
) -> None:
    """Print the plan as a Rich table: SELLs first, BUYs after."""
    console = console or Console()

    if not plan.trades:
        console.print("[yellow]No trades in plan.[/yellow]")
        return

    table = Table(title=title, show_lines=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Side", justify="center")
    table.add_column("Symbol", style="cyan")
    table.add_column("Sector", style="dim")
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Est. value", justify="right")
    table.add_column("Δ weight", justify="right", style="dim")
    table.add_column("Reason", style="dim")

    seq = 0
    for t in _iter_trades_in_order(plan):
        seq += 1
        side = "SELL" if t.is_sell else "BUY"
        side_fmt = f"[red]{side}[/red]" if t.is_sell else f"[green]{side}[/green]"
        delta = t.target_weight - t.current_weight
        table.add_row(
            str(seq),
            side_fmt,
            t.symbol,
            t.sector,
            str(t.quantity),
            format_currency(t.price),
            format_currency(t.value),
            format_percentage(delta) if delta else "—",
            t.reason or "",
        )

    console.print(table)

    # Footer with aggregates.
    net = plan.total_buy_value - plan.total_sell_value
    console.print(
        f"[dim]Total SELL: {format_currency(plan.total_sell_value)}  |  "
        f"Total BUY: {format_currency(plan.total_buy_value)}  |  "
        f"Net cash needed: {format_currency(net)}[/dim]"
    )
    if plan.warnings:
        for w in plan.warnings:
            console.print(f"[yellow]⚠ {w}[/yellow]")


def write_basket_csv(
    plan: RebalancePlan,
    path: Path | str,
    *,
    exchange: str = "NSE",
    product: str = "CNC",
    order_type: str = "MARKET",
) -> int:
    """Write the plan to a Kite-compatible basket CSV and return the row count.

    Args:
        plan: RebalancePlan to serialize.
        path: Filesystem path for the CSV. Parents are created if missing.
        exchange / product / order_type: Kite-side defaults applied to every row.
            Override for unusual products (e.g. MIS intraday) if needed.

    The CSV has one row per trade, ordered SELL-first then BUY, with a ``seq``
    column so the execution order is preserved after any spreadsheet sort.
    ``price`` is left at 0 for MARKET orders (Kite convention).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()

        seq = 0
        for t in _iter_trades_in_order(plan):
            seq += 1
            writer.writerow(
                {
                    "seq": seq,
                    "transaction_type": "SELL" if t.is_sell else "BUY",
                    "tradingsymbol": t.symbol,
                    "exchange": exchange,
                    "quantity": t.quantity,
                    "order_type": order_type,
                    "product": product,
                    "price": 0 if order_type == "MARKET" else round(t.price, 2),
                    "estimated_value": round(t.value, 2),
                    "reason": t.reason or "",
                }
            )
            written += 1
    return written
