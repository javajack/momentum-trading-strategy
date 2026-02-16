#!/usr/bin/env python3
"""
Reconcile strategy state with actual broker holdings.

Fetches live holdings from Zerodha, identifies strategy-managed positions,
and writes a clean strategy_state.json with accurate managed_symbols,
peak_prices, and uninvested_capital=0.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # env vars set externally or in config

from kiteconnect import KiteConnect

from fortress.config import load_config
from fortress.universe import Universe


def main():
    config = load_config()

    # Authenticate using cached token
    token_cache = Path(".kite_token_cache.json")
    if not token_cache.exists():
        print("ERROR: No token cache found. Login via CLI first.")
        sys.exit(1)

    cache = json.loads(token_cache.read_text())
    if cache.get("date") != datetime.now().strftime("%Y-%m-%d"):
        print("ERROR: Token cache is from a different day. Login via CLI first.")
        sys.exit(1)

    kite = KiteConnect(api_key=cache["api_key"])
    kite.set_access_token(cache["access_token"])

    # Validate token
    try:
        user = kite.profile()
        print(f"Authenticated as: {user['user_name']} ({user['user_id']})")
    except Exception as e:
        print(f"ERROR: Token invalid: {e}")
        sys.exit(1)

    # Load universe
    universe = Universe(config.paths.universe_file)
    universe_symbols = {s.zerodha_symbol for s in universe.get_all_stocks()}
    print(f"Universe stocks: {len(universe_symbols)}")

    # Defensive symbols
    gold_symbol = config.regime.gold_symbol
    cash_symbol = config.regime.cash_symbol
    defensive_symbols = {gold_symbol, cash_symbol}

    external_etfs = {
        "NIFTYBEES",
        "JUNIORBEES",
        "MID150BEES",
        "HDFCSML250",
        "HANGSENGBEES",
        "HNGSNGBEES",
        "LIQUIDCASE",
    }

    # Fetch holdings
    holdings = kite.holdings()
    print(f"\nBroker holdings: {len(holdings)} symbols")

    # Fetch today's positions
    day_positions = kite.positions()
    net_positions = day_positions.get("net", [])

    # Build combined positions (same logic as portfolio.load_combined_positions)
    positions = {}
    for h in holdings:
        symbol = h["tradingsymbol"]
        total_qty = h["quantity"] + h.get("t1_quantity", 0)
        if total_qty == 0:
            continue
        positions[symbol] = {
            "qty": total_qty,
            "avg_price": h["average_price"],
            "last_price": h["last_price"],
            "value": total_qty * h["last_price"],
        }

    # Overlay today's CNC positions
    for p in net_positions:
        if p["product"] != "CNC":
            continue
        symbol = p["tradingsymbol"]
        day_bought = p.get("day_buy_quantity", 0)
        if symbol in positions and day_bought > 0:
            positions[symbol]["qty"] += day_bought
            positions[symbol]["value"] = positions[symbol]["qty"] * p["last_price"]
            positions[symbol]["last_price"] = p["last_price"]
        elif day_bought > 0:
            positions[symbol] = {
                "qty": day_bought,
                "avg_price": p.get("average_price", 0),
                "last_price": p["last_price"],
                "value": day_bought * p["last_price"],
            }

    # Get cash
    margins = kite.margins()
    equity = margins.get("equity", {})
    available = equity.get("available", {})
    cash = available.get("live_balance", 0)

    # Classify holdings
    managed = {}
    external = {}

    for symbol, pos in positions.items():
        if symbol in external_etfs:
            external[symbol] = pos
        elif symbol in defensive_symbols:
            managed[symbol] = pos
        elif symbol in universe_symbols:
            managed[symbol] = pos
        else:
            external[symbol] = pos

    # Display
    managed_value = sum(p["value"] for p in managed.values())
    external_value = sum(p["value"] for p in external.values())

    print(f"\n{'=' * 60}")
    print(f"MANAGED POSITIONS ({len(managed)} symbols)")
    print(f"{'=' * 60}")
    for symbol in sorted(managed.keys()):
        pos = managed[symbol]
        print(
            f"  {symbol:20s}  qty={pos['qty']:>5d}  LTP=₹{pos['last_price']:>10,.2f}  value=₹{pos['value']:>12,.0f}"
        )
    print(f"  {'':20s}  {'':>5s}  {'Total':>14s}  value=₹{managed_value:>12,.0f}")

    if external:
        print(f"\n{'=' * 60}")
        print(f"EXTERNAL POSITIONS ({len(external)} symbols)")
        print(f"{'=' * 60}")
        for symbol in sorted(external.keys()):
            pos = external[symbol]
            print(
                f"  {symbol:20s}  qty={pos['qty']:>5d}  LTP=₹{pos['last_price']:>10,.2f}  value=₹{pos['value']:>12,.0f}"
            )
        print(f"  {'':20s}  {'':>5s}  {'Total':>14s}  value=₹{external_value:>12,.0f}")

    print(f"\nDemat cash: ₹{cash:,.0f}")
    print(f"Total portfolio: ₹{managed_value + external_value + cash:,.0f}")

    # Load existing state
    state_file = Path(config.paths.data_cache) / "strategy_state.json"
    existing = {}
    if state_file.exists():
        existing = json.loads(state_file.read_text())
        print(f"\nExisting state file: {state_file}")
        print(f"  managed_symbols: {existing.get('managed_symbols', [])}")
        print(f"  last_rebalance_date: {existing.get('last_rebalance_date')}")
        print(f"  last_regime: {existing.get('last_regime')}")

    # Build clean state
    managed_symbols = sorted(managed.keys())
    peak_prices = {}
    existing_peaks = existing.get("peak_prices", {})

    for symbol in managed_symbols:
        pos = managed[symbol]
        current = pos["last_price"]
        # Keep existing peak if higher, otherwise use current price
        old_peak = existing_peaks.get(symbol, 0.0)
        peak_prices[symbol] = max(current, old_peak)

    # Compute differences
    old_managed = set(existing.get("managed_symbols", []))
    new_managed = set(managed_symbols)
    added = new_managed - old_managed
    removed = old_managed - new_managed

    print(f"\n{'=' * 60}")
    print("STATE CHANGES")
    print(f"{'=' * 60}")
    if added:
        print(f"  + ADDED to managed:   {sorted(added)}")
    if removed:
        print(f"  - REMOVED from managed: {sorted(removed)}")
    if not added and not removed:
        print(f"  managed_symbols: NO CHANGE")

    # Peak price changes
    for symbol in managed_symbols:
        old = existing_peaks.get(symbol, 0.0)
        new = peak_prices[symbol]
        if abs(old - new) > 0.01:
            print(f"  peak {symbol}: ₹{old:,.2f} → ₹{new:,.2f}")

    new_state = {
        "managed_symbols": managed_symbols,
        "peak_prices": peak_prices,
        "updated": datetime.now().isoformat(),
        "last_rebalance_date": existing.get("last_rebalance_date"),
        "last_regime": existing.get("last_regime"),
    }

    print(f"\n{'=' * 60}")
    print("NEW STATE")
    print(f"{'=' * 60}")
    print(json.dumps(new_state, indent=2))

    # Confirm
    response = input("\nWrite this state? [y/N]: ").strip().lower()
    if response == "y":
        state_file.parent.mkdir(exist_ok=True)
        state_file.write_text(json.dumps(new_state, indent=2))
        print(f"✓ Written to {state_file}")
    else:
        print("Aborted.")


if __name__ == "__main__":
    main()
