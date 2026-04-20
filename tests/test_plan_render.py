"""Tests for fortress.plan_render — console output + Kite basket CSV."""

import csv
from pathlib import Path

import pytest

from fortress.plan_render import render_console, write_basket_csv
from fortress.rebalance_planner import PlannedTrade, RebalancePlan, TradeAction


def _trade(symbol, action, qty=10, price=100.0, reason="") -> PlannedTrade:
    return PlannedTrade(
        symbol=symbol,
        action=action,
        quantity=qty,
        price=price,
        value=qty * price,
        sector="TEST",
        reason=reason,
    )


class TestConsoleRender:
    def test_empty_plan_does_not_crash(self):
        render_console(RebalancePlan())

    def test_populated_plan_renders(self):
        plan = RebalancePlan(trades=[
            _trade("RELIANCE", TradeAction.BUY_NEW, qty=5, price=2500.0),
            _trade("VEDL", TradeAction.SELL_EXIT, qty=100, price=400.0, reason="Score drop"),
        ])
        plan.total_sell_value = 40000.0
        plan.total_buy_value = 12500.0
        # Should complete without error.
        render_console(plan, title="Test plan")


class TestBasketCSV:
    def test_writes_sell_before_buy(self, tmp_path):
        plan = RebalancePlan(trades=[
            _trade("RELIANCE", TradeAction.BUY_NEW, qty=5, price=2500.0),
            _trade("TCS", TradeAction.BUY_INCREASE, qty=3, price=3500.0),
            _trade("VEDL", TradeAction.SELL_EXIT, qty=100, price=400.0),
            _trade("ITC", TradeAction.SELL_REDUCE, qty=50, price=450.0),
        ])

        path = tmp_path / "plan.csv"
        rows = write_basket_csv(plan, path)
        assert rows == 4

        with path.open() as f:
            reader = csv.DictReader(f)
            out = list(reader)

        # SELLs first (seq 1-2), BUYs after (seq 3-4).
        assert [r["transaction_type"] for r in out] == ["SELL", "SELL", "BUY", "BUY"]
        assert [r["seq"] for r in out] == ["1", "2", "3", "4"]

    def test_market_order_price_is_zero(self, tmp_path):
        plan = RebalancePlan(trades=[_trade("RELIANCE", TradeAction.BUY_NEW)])
        path = tmp_path / "plan.csv"
        write_basket_csv(plan, path, order_type="MARKET")

        with path.open() as f:
            row = next(csv.DictReader(f))
        assert row["price"] == "0"
        assert row["order_type"] == "MARKET"

    def test_limit_order_carries_price(self, tmp_path):
        plan = RebalancePlan(trades=[_trade("RELIANCE", TradeAction.BUY_NEW, price=2501.75)])
        path = tmp_path / "plan.csv"
        write_basket_csv(plan, path, order_type="LIMIT")

        with path.open() as f:
            row = next(csv.DictReader(f))
        assert float(row["price"]) == 2501.75
        assert row["order_type"] == "LIMIT"

    def test_kite_required_columns_present(self, tmp_path):
        # Columns align with KiteConnect place_order parameter names so the file
        # can be imported into basket flows without mapping.
        plan = RebalancePlan(trades=[_trade("RELIANCE", TradeAction.BUY_NEW)])
        path = tmp_path / "plan.csv"
        write_basket_csv(plan, path)

        with path.open() as f:
            header = next(csv.reader(f))
        for col in ["transaction_type", "tradingsymbol", "exchange",
                    "quantity", "order_type", "product", "price"]:
            assert col in header, f"Missing Kite column: {col}"

    def test_default_exchange_and_product(self, tmp_path):
        plan = RebalancePlan(trades=[_trade("RELIANCE", TradeAction.BUY_NEW)])
        path = tmp_path / "plan.csv"
        write_basket_csv(plan, path)

        with path.open() as f:
            row = next(csv.DictReader(f))
        assert row["exchange"] == "NSE"
        assert row["product"] == "CNC"

    def test_empty_plan_writes_header_only(self, tmp_path):
        path = tmp_path / "empty.csv"
        rows = write_basket_csv(RebalancePlan(), path)
        assert rows == 0
        assert path.exists()
        with path.open() as f:
            lines = f.readlines()
        assert len(lines) == 1  # header only
