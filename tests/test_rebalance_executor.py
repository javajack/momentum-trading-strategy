"""
Tests for RebalanceExecutor self-funding invariant.

The rebalance cycle must NEVER require external cash:
1. Sells generate proceeds
2. Buys consume proceeds (scaled down if insufficient)
3. Surplus sweeps to LIQUIDBEES
4. Demat cash injection is tracked separately

Invariant: plan.total_buy_value <= plan.total_sell_value (after scaling)
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from fortress.portfolio import PortfolioSnapshot, Position
from fortress.rebalance_executor import RebalanceExecutor, RebalancePlan, TradeAction


@dataclass
class _MockUniverse:
    """Minimal universe mock."""

    def get_stock(self, symbol):
        stock = MagicMock()
        stock.sector = "TEST"
        return stock


def _make_executor(cash: float = 0.0) -> RebalanceExecutor:
    """Create executor with minimal mocks."""
    kite = MagicMock()
    portfolio = MagicMock()
    portfolio.get_snapshot.return_value = PortfolioSnapshot(
        positions={}, cash=cash, total_value=cash, unrealized_pnl=0.0
    )
    mapper = MagicMock()
    mapper.get_lot_size.return_value = 1
    mapper.round_to_tick.side_effect = lambda price, sym: price
    order_manager = MagicMock()
    universe = _MockUniverse()

    return RebalanceExecutor(
        kite=kite,
        portfolio=portfolio,
        instrument_mapper=mapper,
        order_manager=order_manager,
        universe=universe,
    )


def _pos(symbol: str, qty: int, price: float, sector: str = "TEST") -> Position:
    """Create a test position."""
    return Position(
        symbol=symbol,
        quantity=qty,
        average_price=price,
        sector=sector,
        current_price=price,
    )


class TestSelfFundingInvariant:
    """The rebalance must always be self-funded from sell proceeds."""

    def test_basic_self_funded(self):
        """Sells cover buys, surplus goes to LIQUIDBEES."""
        executor = _make_executor(cash=0)
        holdings = {
            "EXIT_STOCK": _pos("EXIT_STOCK", 100, 100.0),  # ₹10,000 to sell
        }
        targets = {
            "NEW_STOCK": 0.50,  # Want ₹5,000
        }
        prices = {"EXIT_STOCK": 100.0, "NEW_STOCK": 50.0, "LIQUIDBEES": 1000.0}

        plan = executor.build_plan(
            target_weights=targets,
            current_holdings=holdings,
            managed_capital=10000.0,
            current_prices=prices,
            cash_symbol="LIQUIDBEES",
        )

        assert plan.total_buy_value <= plan.total_sell_value
        assert plan.net_cash_needed <= 0
        assert plan.demat_cash_deployed == 0

    def test_buys_scaled_when_exceeding_sells(self):
        """When buys > sells, buys are scaled down to fit."""
        executor = _make_executor(cash=0)
        holdings = {
            "SMALL_SELL": _pos("SMALL_SELL", 10, 100.0),  # ₹1,000 to sell
        }
        targets = {
            "BIG_BUY": 0.90,  # Want ₹9,000 but only ₹1,000 available
        }
        prices = {"SMALL_SELL": 100.0, "BIG_BUY": 50.0, "LIQUIDBEES": 1000.0}

        plan = executor.build_plan(
            target_weights=targets,
            current_holdings=holdings,
            managed_capital=10000.0,
            current_prices=prices,
            cash_symbol="LIQUIDBEES",
        )

        assert plan.total_buy_value <= plan.total_sell_value
        assert plan.net_cash_needed <= 0
        assert any("Scaling" in w for w in plan.warnings)

    def test_demat_cash_tracked_separately(self):
        """Demat cash → LIQUIDBEES does NOT inflate total_buy_value."""
        executor = _make_executor(cash=50000.0)  # ₹50K demat cash
        holdings = {
            "STOCK_A": _pos("STOCK_A", 100, 100.0),  # ₹10,000 to sell
        }
        targets = {
            "STOCK_B": 0.50,  # Want ₹5,000
        }
        prices = {"STOCK_A": 100.0, "STOCK_B": 50.0, "LIQUIDBEES": 1000.0}

        plan = executor.build_plan(
            target_weights=targets,
            current_holdings=holdings,
            managed_capital=10000.0,
            current_prices=prices,
            cash_symbol="LIQUIDBEES",
        )

        # Rebalance is self-funded
        assert plan.total_buy_value <= plan.total_sell_value
        assert plan.net_cash_needed <= 0

        # Demat cash tracked separately
        assert plan.demat_cash_deployed > 0
        assert plan.demat_cash_deployed <= 50000.0

        # There should be a capital injection trade
        injection_trades = [t for t in plan.trades if "Capital injection" in t.reason]
        assert len(injection_trades) == 1
        assert injection_trades[0].symbol == "LIQUIDBEES"

    def test_no_demat_cash_no_injection(self):
        """When demat cash = 0, no capital injection trade."""
        executor = _make_executor(cash=0)
        holdings = {
            "STOCK_A": _pos("STOCK_A", 100, 100.0),
        }
        targets = {
            "STOCK_B": 0.50,
        }
        prices = {"STOCK_A": 100.0, "STOCK_B": 50.0, "LIQUIDBEES": 1000.0}

        plan = executor.build_plan(
            target_weights=targets,
            current_holdings=holdings,
            managed_capital=10000.0,
            current_prices=prices,
            cash_symbol="LIQUIDBEES",
        )

        assert plan.demat_cash_deployed == 0
        injection_trades = [t for t in plan.trades if "Capital injection" in t.reason]
        assert len(injection_trades) == 0

    def test_liquidbees_sold_to_fund_buys(self):
        """LIQUIDBEES in holdings but not in targets gets sold, proceeds fund buys."""
        executor = _make_executor(cash=0)
        holdings = {
            "LIQUIDBEES": _pos("LIQUIDBEES", 10, 1000.0, sector="Cash"),  # ₹10,000
        }
        targets = {
            "NEW_STOCK": 0.80,  # Want ₹8,000
        }
        prices = {"LIQUIDBEES": 1000.0, "NEW_STOCK": 100.0}

        plan = executor.build_plan(
            target_weights=targets,
            current_holdings=holdings,
            managed_capital=10000.0,
            current_prices=prices,
            cash_symbol="LIQUIDBEES",
        )

        # LIQUIDBEES sold
        sell_trades = [t for t in plan.trades if t.is_sell and t.symbol == "LIQUIDBEES"]
        assert len(sell_trades) == 1

        # Self-funded from LIQUIDBEES sale
        assert plan.total_buy_value <= plan.total_sell_value
        assert plan.net_cash_needed <= 0

    def test_surplus_sweep_included_in_buy_value(self):
        """Phase 3 surplus → LIQUIDBEES IS part of total_buy_value (funded by sells)."""
        executor = _make_executor(cash=0)
        holdings = {
            "BIG_SELL": _pos("BIG_SELL", 100, 200.0),  # ₹20,000 to sell
        }
        targets = {
            "SMALL_BUY": 0.25,  # Want ₹5,000
        }
        prices = {"BIG_SELL": 200.0, "SMALL_BUY": 50.0, "LIQUIDBEES": 1000.0}

        plan = executor.build_plan(
            target_weights=targets,
            current_holdings=holdings,
            managed_capital=20000.0,
            current_prices=prices,
            cash_symbol="LIQUIDBEES",
        )

        # Surplus swept to LIQUIDBEES
        sweep_trades = [
            t for t in plan.trades if t.symbol == "LIQUIDBEES" and "Cash sweep" in t.reason
        ]
        assert len(sweep_trades) == 1

        # Total buy includes the sweep (both funded by sells)
        assert plan.total_buy_value > 0
        assert plan.total_buy_value <= plan.total_sell_value
        assert plan.net_cash_needed <= 0

    def test_full_cycle_invariant(self):
        """Combined: sells, buys, surplus, demat cash — invariant holds."""
        executor = _make_executor(cash=25000.0)  # ₹25K demat
        holdings = {
            "EXIT_A": _pos("EXIT_A", 50, 200.0),  # ₹10,000
            "EXIT_B": _pos("EXIT_B", 100, 150.0),  # ₹15,000
            "GOLDBEES": _pos("GOLDBEES", 100, 130.0, sector="Hedge"),  # ₹13,000
        }
        targets = {
            "NEW_X": 0.30,  # ₹11,400
            "NEW_Y": 0.20,  # ₹7,600
            "GOLDBEES": 0.10,  # ₹3,800
        }
        prices = {
            "EXIT_A": 200.0,
            "EXIT_B": 150.0,
            "GOLDBEES": 130.0,
            "NEW_X": 100.0,
            "NEW_Y": 50.0,
            "LIQUIDBEES": 1000.0,
        }

        plan = executor.build_plan(
            target_weights=targets,
            current_holdings=holdings,
            managed_capital=38000.0,
            current_prices=prices,
            gold_symbol="GOLDBEES",
            cash_symbol="LIQUIDBEES",
        )

        # Core invariant: rebalance is self-funded
        assert plan.total_buy_value <= plan.total_sell_value, (
            f"Self-funding violated: buys={plan.total_buy_value:.0f} > sells={plan.total_sell_value:.0f}"
        )
        assert plan.net_cash_needed <= 0

        # Demat cash is separate
        assert plan.demat_cash_deployed > 0
        # Total executed value = rebalance buys + demat injection
        total_executed = plan.total_buy_value + plan.demat_cash_deployed
        assert total_executed > plan.total_buy_value
