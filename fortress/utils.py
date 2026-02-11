"""
Utility functions for FORTRESS MOMENTUM.

Enforces invariant O6: Rate limit max 3 API calls/second.
"""

import time
from collections import deque
from functools import wraps
from typing import Callable, Tuple, TypeVar

F = TypeVar("F", bound=Callable)


def rate_limit(calls: int = 3, period: float = 1.0) -> Callable[[F], F]:
    """
    Decorator to rate limit function calls.

    Uses per-instance storage for instance methods, function-level storage
    for standalone functions.

    Args:
        calls: Maximum calls allowed in the period
        period: Time period in seconds

    Returns:
        Decorated function that respects rate limits

    Example:
        @rate_limit(calls=3, period=1.0)
        def api_call():
            pass
    """
    # Fallback timestamps for standalone functions
    _func_timestamps: deque = deque()

    def decorator(func: F) -> F:
        attr_name = f"_rate_limit_{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use per-instance storage if this is an instance method
            if args and hasattr(args[0], "__dict__"):
                instance = args[0]
                if not hasattr(instance, attr_name):
                    setattr(instance, attr_name, deque())
                timestamps = getattr(instance, attr_name)
            else:
                timestamps = _func_timestamps

            now = time.time()

            # Remove timestamps older than period
            while timestamps and timestamps[0] < now - period:
                timestamps.popleft()

            # If at limit, wait until oldest timestamp expires
            if len(timestamps) >= calls:
                sleep_time = timestamps[0] + period - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # Remove the expired timestamp
                while timestamps and timestamps[0] < time.time() - period:
                    timestamps.popleft()

            timestamps.append(time.time())
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def chunks(lst: list, n: int):
    """
    Yield successive n-sized chunks from list.

    Args:
        lst: List to chunk
        n: Chunk size

    Yields:
        Chunks of size n
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def format_currency(value: float, symbol: str = "₹") -> str:
    """
    Format number as Indian currency.

    Args:
        value: Numeric value
        symbol: Currency symbol

    Returns:
        Formatted string like "₹16,00,000"
    """
    if value < 0:
        return f"-{symbol}{format_indian_number(abs(value))}"
    return f"{symbol}{format_indian_number(value)}"


def format_indian_number(num: float) -> str:
    """
    Format number with Indian comma separators.

    Args:
        num: Number to format

    Returns:
        Formatted string like "16,00,000"
    """
    num = int(num)
    s = str(num)
    if len(s) <= 3:
        return s

    # Last 3 digits
    result = s[-3:]
    s = s[:-3]

    # Group remaining in pairs
    while s:
        result = s[-2:] + "," + result
        s = s[:-2]

    return result


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format number as percentage.

    Args:
        value: Decimal value (0.05 = 5%)
        decimals: Decimal places

    Returns:
        Formatted string like "+5.00%"
    """
    pct = value * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.{decimals}f}%"


def trading_days_between(start_date, end_date, holidays: list = None) -> int:
    """
    Count trading days between two dates.

    Args:
        start_date: Start date
        end_date: End date
        holidays: List of holiday dates

    Returns:
        Number of trading days
    """
    import pandas as pd

    holidays = holidays or []
    dates = pd.bdate_range(start=start_date, end=end_date)

    # Filter out holidays
    trading_dates = [d for d in dates if d.date() not in holidays]
    return len(trading_dates)


def is_market_hours() -> bool:
    """
    Check if current time is within market hours.

    Returns:
        True if market is open (9:15 AM - 3:30 PM IST on weekdays)
    """
    from datetime import datetime

    import pytz

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    # Weekend check
    if now.weekday() >= 5:
        return False

    # Market hours: 9:15 AM to 3:30 PM
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    return market_open <= now <= market_close


def validate_market_hours() -> Tuple[bool, str]:
    """
    Check if current time is within NSE market hours.

    Returns:
        Tuple of (is_open, message)
    """
    from datetime import datetime

    import pytz

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    if now.weekday() >= 5:
        return (False, f"Market closed: Weekend ({now.strftime('%A')})")

    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    if now < market_open:
        return (False, "Market not yet open (opens 9:15 AM IST)")
    if now > market_close:
        return (False, "Market closed (closed 3:30 PM IST)")

    return (True, "Market open")


def calculate_order_quantity(
    target_value: float,
    price: float,
    lot_size: int = 1,
) -> Tuple[int, float]:
    """
    Calculate order quantity from target value.

    Args:
        target_value: Target position value in currency
        price: Current price per share
        lot_size: Minimum lot size (default 1 for equity)

    Returns:
        Tuple of (quantity, remainder_value)
    """
    if price <= 0:
        return (0, target_value)

    raw_qty = target_value / price
    qty = int(raw_qty // lot_size) * lot_size  # Round down to lot multiple
    actual_value = qty * price
    remainder = target_value - actual_value

    return (qty, remainder)


def renormalize_with_caps(
    weights: dict,
    max_weight: float,
    min_weight: float = 0.0,
    max_iterations: int = 20,
) -> dict:
    """
    Renormalize weights to sum to 1.0 while respecting position limits.

    Uses iterative capping algorithm:
    1. Renormalize to 100%
    2. Identify stocks exceeding max_weight
    3. Cap those at max_weight
    4. Redistribute excess to uncapped stocks
    5. Repeat until no stock exceeds limit

    This function ensures logic parity between live rebalance
    (momentum_engine.py) and backtest (backtest.py).

    Args:
        weights: Initial weight allocation (may not sum to 1.0)
        max_weight: Maximum allowed weight per position
        min_weight: Minimum allowed weight per position (default 0)
        max_iterations: Safety limit for iterations

    Returns:
        Dict mapping ticker to weight, summing to 1.0,
        with no position exceeding max_weight
    """
    result = weights.copy()

    for _ in range(max_iterations):
        # Renormalize to sum to 1.0
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        # Find stocks exceeding max
        capped = {}
        uncapped = {}
        for ticker, weight in result.items():
            if weight > max_weight:
                capped[ticker] = max_weight
            else:
                uncapped[ticker] = weight

        # If nothing exceeds max, we're done
        if not capped:
            break

        # Calculate excess to redistribute
        excess = sum(result[t] - max_weight for t in capped)

        # If no uncapped stocks to redistribute to, we're stuck
        if not uncapped:
            result = capped
            break

        # Redistribute excess proportionally to uncapped stocks
        uncapped_total = sum(uncapped.values())
        for ticker in uncapped:
            # Add proportional share of excess
            share = uncapped[ticker] / uncapped_total if uncapped_total > 0 else 1.0 / len(uncapped)
            uncapped[ticker] += excess * share
            # Ensure we don't go below minimum
            uncapped[ticker] = max(min_weight, uncapped[ticker])

        # Combine capped and uncapped
        result = {**capped, **uncapped}

    return result
