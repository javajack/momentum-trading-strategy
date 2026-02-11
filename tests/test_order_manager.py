"""
Tests for order manager module.

Verifies invariants O1-O8, R9, R10.
"""

from unittest.mock import MagicMock, patch
import pytest

from fortress.order_manager import OrderManager, Order, OrderType, OrderStatus
from fortress.risk_governor import RiskGovernor


@pytest.fixture
def mock_kite():
    """Create mock Kite instance."""
    kite = MagicMock()
    kite.TRANSACTION_TYPE_BUY = "BUY"
    kite.TRANSACTION_TYPE_SELL = "SELL"
    kite.VARIETY_REGULAR = "regular"
    kite.EXCHANGE_NSE = "NSE"
    kite.PRODUCT_CNC = "CNC"
    kite.ORDER_TYPE_MARKET = "MARKET"
    kite.margins.return_value = {
        "equity": {"available": {"live_balance": 1000000}}
    }
    kite.place_order.return_value = "12345678"
    return kite


@pytest.fixture
def risk_governor():
    """Create risk governor."""
    return RiskGovernor()


@pytest.fixture
def order_manager(mock_kite, risk_governor):
    """Create order manager in dry-run mode."""
    return OrderManager(mock_kite, risk_governor, dry_run=True)


@pytest.fixture
def live_order_manager(mock_kite, risk_governor):
    """Create order manager in live mode."""
    return OrderManager(mock_kite, risk_governor, dry_run=False)


class TestDryRunMode:
    """Test dry-run mode (O1, O2)."""

    def test_default_is_dry_run(self, mock_kite, risk_governor):
        """O1: Dry-run is default mode."""
        manager = OrderManager(mock_kite, risk_governor)
        assert manager.dry_run is True

    def test_dry_run_does_not_place_order(self, order_manager, mock_kite):
        """O1: Dry-run doesn't call Kite API."""
        order = order_manager.create_order(
            symbol="RELIANCE",
            order_type=OrderType.BUY,
            quantity=10,
            price=2500,
            sector="OIL_GAS_ENERGY",
        )

        result = order_manager.place_order(order, portfolio_value=1000000)

        assert result.success
        assert "Dry run" in result.message
        mock_kite.place_order.assert_not_called()

    def test_dry_run_order_has_dry_prefix(self, order_manager):
        """Dry-run orders have DRY_ prefix on order_id."""
        order = order_manager.create_order(
            symbol="TCS",
            order_type=OrderType.BUY,
            quantity=5,
            price=3500,
            sector="IT_SERVICES",
        )

        order_manager.place_order(order, portfolio_value=1000000)

        assert order.order_id.startswith("DRY_")

    def test_live_mode_calls_api(self, live_order_manager, mock_kite):
        """O2: Live mode places real orders."""
        order = live_order_manager.create_order(
            symbol="RELIANCE",
            order_type=OrderType.BUY,
            quantity=10,
            price=2500,
            sector="OIL_GAS_ENERGY",
        )

        result = live_order_manager.place_order(order, portfolio_value=1000000)

        assert result.success
        mock_kite.place_order.assert_called_once()


class TestOrderTags:
    """Test order tags (O3)."""

    def test_orders_have_strategy_tag(self, order_manager):
        """O3: All orders have strategy identification tag."""
        order1 = order_manager.create_order(
            symbol="RELIANCE",
            order_type=OrderType.BUY,
            quantity=10,
            price=2500,
        )
        order2 = order_manager.create_order(
            symbol="TCS",
            order_type=OrderType.BUY,
            quantity=10,
            price=2500,
        )

        # All orders from this strategy share the same tag
        assert order1.tag == "RRVSectorMomentum"
        assert order2.tag == "RRVSectorMomentum"

    def test_tag_within_zerodha_limit(self, order_manager):
        """Tags must be <= 20 characters for Zerodha API."""
        order = order_manager.create_order(
            symbol="TCS",
            order_type=OrderType.BUY,
            quantity=5,
            price=3500,
        )

        assert len(order.tag) <= 20


class TestOrderValidation:
    """Test order validation (R10)."""

    def test_invalid_quantity_rejected(self, order_manager):
        """Orders with zero quantity are rejected."""
        order = order_manager.create_order(
            symbol="RELIANCE",
            order_type=OrderType.BUY,
            quantity=0,
            price=2500,
        )

        result = order_manager.place_order(order, portfolio_value=1000000)

        assert not result.success
        assert "quantity" in result.message.lower()

    def test_buy_validated_against_limits(self, order_manager):
        """Buy orders are validated against position limits."""
        order = order_manager.create_order(
            symbol="RELIANCE",
            order_type=OrderType.BUY,
            quantity=1000,  # Large position
            price=2500,
            sector="OIL_GAS_ENERGY",
        )

        result = order_manager.place_order(
            order,
            portfolio_value=1000000,
            current_position_value=100000,  # Would exceed limit
        )

        # Should fail position size check
        assert not result.success

    def test_margin_check_for_buys(self, order_manager, mock_kite):
        """R6: Margin is checked for buy orders."""
        mock_kite.margins.return_value = {
            "equity": {"available": {"live_balance": 10000}}  # Low balance
        }

        # Use smaller quantity to pass position size check but fail margin check
        # 10 * 2500 = 25000 required, but only 10000 available
        order = order_manager.create_order(
            symbol="RELIANCE",
            order_type=OrderType.BUY,
            quantity=10,
            price=2500,  # 25000 required
            sector="OIL_GAS_ENERGY",
        )

        result = order_manager.place_order(order, portfolio_value=1000000)

        assert not result.success
        assert "margin" in result.message.lower()


class TestProductAndExchange:
    """Test CNC product and NSE exchange (O7, O8)."""

    def test_orders_use_cnc_product(self, live_order_manager, mock_kite):
        """O7: CNC product type for all positions."""
        order = live_order_manager.create_order(
            symbol="RELIANCE",
            order_type=OrderType.BUY,
            quantity=10,
            price=2500,
            sector="OIL_GAS_ENERGY",
        )

        live_order_manager.place_order(order, portfolio_value=1000000)

        call_kwargs = mock_kite.place_order.call_args[1]
        assert call_kwargs["product"] == mock_kite.PRODUCT_CNC

    def test_orders_use_nse_exchange(self, live_order_manager, mock_kite):
        """O8: NSE exchange for all orders."""
        order = live_order_manager.create_order(
            symbol="TCS",
            order_type=OrderType.BUY,
            quantity=5,
            price=3500,
            sector="IT_SERVICES",
        )

        live_order_manager.place_order(order, portfolio_value=1000000)

        call_kwargs = mock_kite.place_order.call_args[1]
        assert call_kwargs["exchange"] == mock_kite.EXCHANGE_NSE


class TestSellsBeforeBuys:
    """Test sells execute before buys (R9)."""

    def test_sells_before_buys(self, order_manager):
        """R9: Sells execute before buys."""
        orders = [
            order_manager.create_order(
                symbol="TCS", order_type=OrderType.BUY, quantity=5, price=3500
            ),
            order_manager.create_order(
                symbol="RELIANCE", order_type=OrderType.SELL, quantity=10, price=2500
            ),
            order_manager.create_order(
                symbol="INFY", order_type=OrderType.BUY, quantity=8, price=1500
            ),
            order_manager.create_order(
                symbol="HDFC", order_type=OrderType.SELL, quantity=20, price=1600
            ),
        ]

        results = order_manager.place_orders_batch(
            orders=orders,
            portfolio_value=1000000,
            current_positions=10,
            sector_exposures={},
            position_values={},
        )

        # Extract order types in execution order
        execution_order = [r.order.order_type for r in results]

        # All sells should come before all buys
        first_buy_idx = next(
            i for i, t in enumerate(execution_order) if t == OrderType.BUY
        )
        last_sell_idx = max(
            i for i, t in enumerate(execution_order) if t == OrderType.SELL
        )

        assert last_sell_idx < first_buy_idx


class TestOrderStatus:
    """Test order status tracking (O4)."""

    def test_get_order_status(self, live_order_manager, mock_kite):
        """O4: Order status can be tracked."""
        mock_kite.order_history.return_value = [
            {"status": "OPEN", "order_id": "12345"},
            {"status": "COMPLETE", "order_id": "12345"},
        ]

        status = live_order_manager.get_order_status("12345")

        assert status["status"] == "COMPLETE"

    def test_dry_run_status_is_complete(self, order_manager):
        """Dry-run orders report as COMPLETE."""
        status = order_manager.get_order_status("DRY_TEST_123")

        assert status["status"] == "COMPLETE"
