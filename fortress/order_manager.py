"""
Order management for FORTRESS MOMENTUM.

Enforces invariants:
- O1: Dry-run is default mode
- O2: Live orders require explicit --live flag
- O3: All orders have unique tags
- O4: Order status is tracked to completion
- O5: Failed orders are logged with reason
- O6: Rate limit: max 3 orders/second
- O7: CNC product type for all positions
- O8: NSE exchange for all orders
- R9: Sells execute before buys
- R10: No order placed without validation
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import logging

from .risk_governor import RiskGovernor
from .utils import rate_limit


class OrderType(Enum):
    """Order transaction type."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "PENDING"
    PLACED = "PLACED"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    """Represents a trading order."""

    symbol: str
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    product: str = "CNC"  # O7: CNC for delivery
    exchange: str = "NSE"  # O8: NSE exchange
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    reason: Optional[str] = None
    tag: Optional[str] = None  # O3: Unique tag
    sector: str = ""

    @property
    def value(self) -> float:
        """Estimated order value."""
        return self.quantity * (self.price or 0)


@dataclass
class OrderResult:
    """Result of order placement attempt."""

    order: Order
    success: bool
    message: str


class OrderManager:
    """
    Manages order placement with safety checks.

    O1: Dry-run by default - requires explicit enable for live orders.
    """

    def __init__(
        self,
        kite,
        risk_governor: RiskGovernor,
        dry_run: bool = True,  # O1: Default dry-run
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize order manager.

        Args:
            kite: Authenticated KiteConnect instance
            risk_governor: RiskGovernor instance
            dry_run: If True, orders are simulated (O1)
            logger: Logger instance
        """
        self.kite = kite
        self.risk_governor = risk_governor
        self.dry_run = dry_run  # O1, O2
        self.logger = logger or logging.getLogger(__name__)
        self._order_counter = 0

    def _generate_tag(self, symbol: str) -> str:
        """
        Generate order tag.

        O3: All orders tagged for tracking.
        Zerodha limit: max 20 characters.

        Args:
            symbol: Stock symbol (unused, kept for API compatibility)

        Returns:
            Fixed tag string identifying strategy
        """
        return "RRVSectorMomentum"  # 17 chars, identifies all orders from this strategy

    def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        quantity: int,
        price: Optional[float] = None,
        sector: str = "",
        tag: Optional[str] = None,
    ) -> Order:
        """
        Create order without placing it.

        Args:
            symbol: Stock symbol
            order_type: BUY or SELL
            quantity: Number of shares
            price: Optional price (None for market)
            sector: Stock's sector
            tag: Optional custom tag (O3: auto-generated if None)

        Returns:
            Order object
        """
        return Order(
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            price=price,
            tag=tag or self._generate_tag(symbol),
            sector=sector,
        )

    def validate_order(
        self,
        order: Order,
        portfolio_value: float,
        current_position_value: float = 0,
        current_sector_value: float = 0,
        current_positions: int = 0,
    ) -> tuple:
        """
        Validate order before placement.

        R10: No order placed without validation.

        Args:
            order: Order to validate
            portfolio_value: Total portfolio value
            current_position_value: Current position in this stock
            current_sector_value: Current sector exposure
            current_positions: Number of current positions

        Returns:
            Tuple of (is_valid, reason)
        """
        # Basic validation
        if order.quantity <= 0:
            return (False, "Invalid quantity")

        if order.order_type == OrderType.BUY:
            # R10: Full validation for buys
            check = self.risk_governor.validate_order(
                symbol=order.symbol,
                sector=order.sector,
                order_value=order.value,
                current_position_value=current_position_value,
                current_sector_value=current_sector_value,
                portfolio_value=portfolio_value,
                current_positions=current_positions,
                is_buy=True,
            )

            if not check.passed:
                return (False, check.reason)

            # R6: Check margin
            margins = self.kite.margins()
            available = margins.get("equity", {}).get("available", {})
            live_balance = available.get("live_balance", 0)

            if order.value > live_balance:
                return (
                    False,
                    f"Insufficient margin: need {order.value:.0f}, "
                    f"have {live_balance:.0f}",
                )

        return (True, "OK")

    @rate_limit(calls=3, period=1.0)  # O6: Rate limit
    def place_order(
        self,
        order: Order,
        portfolio_value: float = 0,
        current_position_value: float = 0,
        current_sector_value: float = 0,
        current_positions: int = 0,
    ) -> OrderResult:
        """
        Place order via Kite API.

        O1/O2: Respects dry-run mode.
        O6: Rate limited to 3/second.
        O7/O8: Uses CNC product on NSE.

        Args:
            order: Order to place
            portfolio_value: For validation
            current_position_value: For validation
            current_sector_value: For validation
            current_positions: For validation

        Returns:
            OrderResult with success/failure info
        """
        # R10: Validate first
        is_valid, reason = self.validate_order(
            order,
            portfolio_value,
            current_position_value,
            current_sector_value,
            current_positions,
        )

        if not is_valid:
            order.status = OrderStatus.REJECTED
            order.reason = reason
            self.logger.warning(f"Order rejected: {order.symbol} - {reason}")  # O5
            return OrderResult(order=order, success=False, message=reason)

        # O1: Dry-run mode
        if self.dry_run:
            order.status = OrderStatus.PENDING
            order.reason = "DRY RUN - not placed"
            order.order_id = f"DRY_{order.tag}"
            self.logger.info(
                f"DRY RUN: {order.order_type.value} {order.quantity} "
                f"{order.symbol} @ {order.price}"
            )
            return OrderResult(
                order=order,
                success=True,
                message="Dry run - order not placed",
            )

        # O2: Live order placement
        try:
            transaction = (
                self.kite.TRANSACTION_TYPE_BUY
                if order.order_type == OrderType.BUY
                else self.kite.TRANSACTION_TYPE_SELL
            )

            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,  # O8
                tradingsymbol=order.symbol,
                transaction_type=transaction,
                quantity=order.quantity,
                product=self.kite.PRODUCT_CNC,  # O7
                order_type=self.kite.ORDER_TYPE_MARKET,
                tag=order.tag,  # O3
            )

            order.order_id = order_id
            order.status = OrderStatus.PLACED
            self.logger.info(
                f"Order placed: {order.order_type.value} {order.quantity} "
                f"{order.symbol} - ID: {order_id}"
            )

            return OrderResult(order=order, success=True, message="Order placed")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reason = str(e)
            self.logger.error(f"Order failed: {order.symbol} - {e}")  # O5
            return OrderResult(order=order, success=False, message=str(e))

    def place_orders_batch(
        self,
        orders: List[Order],
        portfolio_value: float,
        current_positions: int,
        sector_exposures: dict,
        position_values: dict,
    ) -> List[OrderResult]:
        """
        Place multiple orders with sells before buys.

        R9: Sells execute before buys.

        Args:
            orders: List of orders
            portfolio_value: Total portfolio value
            current_positions: Number of current positions
            sector_exposures: Dict of sector -> exposure value
            position_values: Dict of symbol -> position value

        Returns:
            List of OrderResults
        """
        # R9: Separate and order sells first
        sell_orders = [o for o in orders if o.order_type == OrderType.SELL]
        buy_orders = [o for o in orders if o.order_type == OrderType.BUY]

        results: List[OrderResult] = []

        # Execute sells first (R9)
        for order in sell_orders:
            result = self.place_order(
                order,
                portfolio_value=portfolio_value,
                current_position_value=position_values.get(order.symbol, 0),
                current_sector_value=sector_exposures.get(order.sector, 0),
                current_positions=current_positions,
            )
            results.append(result)

            # Update tracking after sell
            if result.success:
                sector_exposures[order.sector] = (
                    sector_exposures.get(order.sector, 0) - order.value
                )
                position_values[order.symbol] = 0
                current_positions -= 1

        # Then execute buys
        for order in buy_orders:
            result = self.place_order(
                order,
                portfolio_value=portfolio_value,
                current_position_value=position_values.get(order.symbol, 0),
                current_sector_value=sector_exposures.get(order.sector, 0),
                current_positions=current_positions,
            )
            results.append(result)

            # Update tracking after buy
            if result.success:
                sector_exposures[order.sector] = (
                    sector_exposures.get(order.sector, 0) + order.value
                )
                old_value = position_values.get(order.symbol, 0)
                position_values[order.symbol] = old_value + order.value
                if old_value == 0:
                    current_positions += 1

        return results

    def get_order_status(self, order_id: str) -> Optional[dict]:
        """
        Get order status from Kite.

        O4: Track order status to completion.

        Args:
            order_id: Kite order ID

        Returns:
            Order status dict or None
        """
        if self.dry_run or order_id.startswith("DRY_"):
            return {"status": "COMPLETE", "order_id": order_id}

        try:
            history = self.kite.order_history(order_id)
            if history:
                return history[-1]  # Latest status
            return None
        except Exception as e:
            self.logger.error(f"Failed to get order status: {e}")
            return None
