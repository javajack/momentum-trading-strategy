"""
Structured logging for FORTRESS MOMENTUM.

Enforces invariant P4: All decisions logged with timestamp.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "fortress",
    log_dir: str = "logs",
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Configure structured logging with file and console output.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console_output: Whether to output to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # File handler - daily rotating log file
    today = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(
        log_path / f"fortress_{today}.log",
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler with Rich
    if console_output:
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "fortress") -> logging.Logger:
    """Get existing logger by name."""
    return logging.getLogger(name)


class DecisionLogger:
    """
    Logs trading decisions with full context for audit.

    Enforces P4: Complete decision trace for every action.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger()

    def log_sector_ranking(
        self,
        rankings: list,
        as_of_date: datetime,
    ) -> None:
        """Log sector ranking decision."""
        self.logger.info(
            f"SECTOR_RANKING | date={as_of_date.date()} | "
            f"top_sectors={[r['sector'] for r in rankings[:5]]}"
        )
        for r in rankings:
            self.logger.debug(
                f"  {r['rank']:2d}. {r['sector']:<25} RRV={r['rrv']:.3f} "
                f"ret={r['return_6m']:.2%} vol={r['volatility']:.2%}"
            )

    def log_stock_selection(
        self,
        sector: str,
        stocks: list,
    ) -> None:
        """Log stock selection within sector."""
        self.logger.info(
            f"STOCK_SELECTION | sector={sector} | "
            f"selected={[s['ticker'] for s in stocks]}"
        )

    def log_risk_check(
        self,
        check_type: str,
        passed: bool,
        details: str,
    ) -> None:
        """Log risk check result."""
        status = "PASS" if passed else "FAIL"
        self.logger.info(f"RISK_CHECK | {check_type} | {status} | {details}")

    def log_order(
        self,
        action: str,
        symbol: str,
        quantity: int,
        price: Optional[float],
        order_id: Optional[str],
        dry_run: bool,
    ) -> None:
        """Log order placement."""
        mode = "DRY_RUN" if dry_run else "LIVE"
        self.logger.info(
            f"ORDER | {mode} | {action} | {symbol} | qty={quantity} | "
            f"price={price} | order_id={order_id}"
        )

    def log_rebalance(
        self,
        sells: list,
        buys: list,
        portfolio_value: float,
    ) -> None:
        """Log rebalance decision."""
        self.logger.info(
            f"REBALANCE | portfolio_value={portfolio_value:,.0f} | "
            f"sells={len(sells)} | buys={len(buys)}"
        )

    def log_regime(
        self,
        regime: str,
        vix: float,
        drawdown: float,
    ) -> None:
        """Log risk regime assessment."""
        self.logger.info(
            f"REGIME | {regime} | VIX={vix:.2f} | drawdown={drawdown:.2%}"
        )
