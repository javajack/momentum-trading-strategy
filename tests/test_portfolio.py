"""Tests for the live Portfolio class — managed/external split and margins fallback."""

from pathlib import Path

import pytest

from fortress.portfolio import Portfolio, PortfolioSnapshot
from fortress.universe import Universe


@pytest.fixture
def universe():
    project_root = Path(__file__).parent.parent
    return Universe(str(project_root / "stock-universe.json"))


class _FakeKite:
    """Minimal stand-in for KiteConnect covering the methods Portfolio calls."""

    def __init__(self, holdings=None, net_positions=None, margins=None,
                 margins_raises=None):
        self._holdings = holdings or []
        self._net = net_positions or []
        self._margins = margins
        self._margins_raises = margins_raises

    def holdings(self):
        return list(self._holdings)

    def positions(self):
        return {"net": list(self._net), "day": []}

    def margins(self):
        if self._margins_raises is not None:
            raise self._margins_raises
        return self._margins or {}


def _holding(symbol, qty=10, avg=100.0, ltp=110.0):
    return {
        "tradingsymbol": symbol,
        "quantity": qty,
        "t1_quantity": 0,
        "average_price": avg,
        "last_price": ltp,
    }


class TestManagedExternalSplit:
    def test_universe_stock_goes_managed(self, universe):
        kite = _FakeKite(holdings=[_holding("RELIANCE")], margins={})
        snap = Portfolio(kite, universe).load_combined_positions()
        assert "RELIANCE" in snap.positions
        assert "RELIANCE" not in snap.external_positions

    def test_registered_hedges_go_managed(self, universe):
        kite = _FakeKite(
            holdings=[_holding("GOLDBEES"), _holding("LIQUIDBEES")],
            margins={},
        )
        snap = Portfolio(kite, universe).load_combined_positions()
        assert "GOLDBEES" in snap.positions
        assert "LIQUIDBEES" in snap.positions

    def test_stray_etf_goes_external(self, universe):
        # NIFTYBEES / HANGSENGBEES aren't in the strategy universe and aren't
        # registered hedges — they must land in external_positions so the
        # strategy ignores them in rebalance/exit flows.
        kite = _FakeKite(
            holdings=[_holding("NIFTYBEES"), _holding("HANGSENGBEES")],
            margins={},
        )
        snap = Portfolio(kite, universe).load_combined_positions()
        assert not snap.positions
        assert "NIFTYBEES" in snap.external_positions
        assert "HANGSENGBEES" in snap.external_positions

    def test_mixed_holdings_split_correctly(self, universe):
        kite = _FakeKite(
            holdings=[
                _holding("RELIANCE"),     # managed
                _holding("LIQUIDBEES"),   # managed (hedge)
                _holding("NIFTYBEES"),    # external
                _holding("SOMEJUNK"),     # external (unknown)
            ],
            margins={},
        )
        snap = Portfolio(kite, universe).load_combined_positions()
        assert set(snap.positions) == {"RELIANCE", "LIQUIDBEES"}
        assert set(snap.external_positions) == {"NIFTYBEES", "SOMEJUNK"}

    def test_external_value_counted_in_total_value(self, universe):
        kite = _FakeKite(
            holdings=[_holding("RELIANCE", qty=1, ltp=100),
                      _holding("NIFTYBEES", qty=5, ltp=200)],
            margins={},
        )
        snap = Portfolio(kite, universe).load_combined_positions()
        # 1 × 100 (managed) + 5 × 200 (external) = 1100
        assert snap.total_value == 1100
        # unrealized_pnl reports on managed only
        assert snap.unrealized_pnl == 0  # avg 100, ltp 100 → zero


class TestMarginsGracefulDegrade:
    def test_margins_exception_becomes_zero_cash(self, universe):
        kite = _FakeKite(
            holdings=[_holding("RELIANCE")],
            margins_raises=RuntimeError("Get Rms Limits Entity Response Failed"),
        )
        # Must not raise.
        snap = Portfolio(kite, universe).load_combined_positions()
        assert snap.cash == 0.0
        assert isinstance(snap, PortfolioSnapshot)

    def test_margins_valid_live_balance_used(self, universe):
        kite = _FakeKite(
            holdings=[_holding("RELIANCE")],
            margins={"equity": {"available": {"live_balance": 12345}}},
        )
        snap = Portfolio(kite, universe).load_combined_positions()
        assert snap.cash == 12345

    def test_check_margin_degrades_on_rms_failure(self, universe):
        kite = _FakeKite(margins_raises=RuntimeError("RMS blocked"))
        has_margin, avail = Portfolio(kite, universe).check_margin(50000)
        # Post-April policy: orders gone anyway — return (True, 0) rather than crash.
        assert has_margin is True
        assert avail == 0.0
