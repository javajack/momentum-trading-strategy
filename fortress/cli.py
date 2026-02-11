"""
Interactive CLI interface for FORTRESS MOMENTUM.

Enforces invariants:
- O1: Dry-run is default mode
- O2: Live orders require explicit confirmation
"""

import json
import sys
import time
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .auth import AuthenticationError, ZerodhaAuth
from .backtest import BacktestConfig, BacktestEngine, BacktestResult
from .cache import CacheManager
from .config import Config, load_config
from .indicators import (
    calculate_drawdown,
    calculate_market_breadth,
    detect_breadth_thrust,
    MarketRegime,
    should_trigger_rebalance,
)
from .instruments import InstrumentMapper
from .market_data import MarketDataProvider
from .momentum_engine import MomentumEngine
from .order_manager import OrderManager, OrderType
from .portfolio import Portfolio, Position
from .rebalance_executor import RebalanceExecutor, RebalancePlan, ExecutionResult, TradeAction
from .risk_governor import RiskGovernor
from .strategy import StrategyRegistry, BaseStrategy
from .universe import Universe
from .utils import format_currency, format_percentage, validate_market_hours

console = Console()
T = TypeVar("T")


def with_retry(
    func: Callable[[], T],
    max_retries: int = 3,
    delay: float = 2.0,
    status_msg: str = "Retrying...",
) -> T:
    """Execute function with retry logic for network errors."""
    last_error = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            is_network_error = any(
                msg in error_str
                for msg in ["timeout", "connection", "network", "reset", "refused"]
            )

            if not is_network_error or attempt == max_retries - 1:
                raise

            console.print(
                f"[yellow]Network issue (attempt {attempt + 1}/{max_retries}): "
                f"{status_msg}[/yellow]"
            )
            time.sleep(delay * (attempt + 1))

    raise last_error


class FortressApp:
    """Interactive menu-driven application for FORTRESS MOMENTUM."""

    MENU_OPTIONS = [
        ("1", "Login", "Authenticate with Zerodha"),
        ("2", "Status", "View portfolio status"),
        ("3", "Scan Stocks", "Rank stocks by momentum"),
        ("4", "Rebalance", "Generate rebalance orders"),
        ("5", "Backtest", "Run historical simulation"),
        ("6", "Strategy", "Select active strategy"),
        ("7", "Triggers", "Check if rebalance is needed"),
        ("9", "Market Phases", "10-year multi-phase backtest analysis"),
        ("0", "Exit", "Exit application"),
    ]

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config: Optional[Config] = None
        self.universe: Optional[Universe] = None  # Profile-filtered universe for strategy
        self._cache_universe: Optional[Universe] = None  # Unfiltered universe for cache
        self.auth: Optional[ZerodhaAuth] = None
        self.kite = None
        self.market_data: Optional[MarketDataProvider] = None
        self.portfolio: Optional[Portfolio] = None
        self.momentum_engine: Optional[MomentumEngine] = None
        self.active_strategy: str = "dual_momentum"  # Default strategy
        self.strategy: Optional[BaseStrategy] = None
        self.cache: Optional[CacheManager] = None
        self.effective_config: Optional[Config] = None  # Config with profile overrides
        self.active_profile_name: str = "primary"  # Active profile key

    def _load_config(self):
        """Load configuration from file."""
        try:
            self.config = load_config(self.config_path)
            console.print(f"[green]Config loaded from {self.config_path}[/green]")

            # Load active strategy from config if available
            if hasattr(self.config, "active_strategy"):
                self.active_strategy = self.config.active_strategy
            self._init_strategy()

        except Exception as e:
            console.print(f"[red]Failed to load config: {e}[/red]")
            sys.exit(1)

    def _init_strategy(self):
        """Initialize the active strategy."""
        cfg = self.effective_config if self.effective_config is not None else self.config
        try:
            self.strategy = StrategyRegistry.get(self.active_strategy, cfg)
            console.print(
                f"[green]Strategy: {self.active_strategy} ({self.strategy.description})[/green]"
            )
        except ValueError as e:
            console.print(f"[yellow]Strategy warning: {e}. Using 'dual_momentum'.[/yellow]")
            self.active_strategy = "dual_momentum"
            self.strategy = StrategyRegistry.get("dual_momentum", cfg)

    def _get_strategy_state_file(self) -> Path:
        """Get path to strategy state file (profile-aware)."""
        cache_dir = Path(self.config.paths.data_cache)
        cache_dir.mkdir(exist_ok=True)
        profile = self.config.get_profile(self.active_profile_name)
        return cache_dir / profile.state_file

    def _load_strategy_state(self) -> Dict:
        """Load strategy state from JSON file.

        Returns dict with:
        - managed_symbols: List of symbols the strategy has bought
        - peak_prices: Dict of symbol -> peak price for trailing stop tracking
        - updated: ISO timestamp of last update
        - last_rebalance_date: ISO date string of last live rebalance (or None)
        - last_regime: String regime name at last rebalance (or None)
        """
        state_file = self._get_strategy_state_file()
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    # Ensure peak_prices exists (backward compatibility)
                    if "peak_prices" not in state:
                        state["peak_prices"] = {}
                    # Ensure trigger fields exist (backward compatibility)
                    if "last_rebalance_date" not in state:
                        state["last_rebalance_date"] = None
                    if "last_regime" not in state:
                        state["last_regime"] = None
                    return state
            except Exception:
                pass
        return {
            "managed_symbols": [], "peak_prices": {}, "updated": None,
            "last_rebalance_date": None, "last_regime": None,
        }

    def _save_strategy_state(
        self,
        managed_symbols: List[str],
        peak_prices: Optional[Dict[str, float]] = None,
        last_rebalance_date: Optional[str] = None,
        last_regime: Optional[str] = None,
    ):
        """Save strategy state to JSON file.

        Args:
            managed_symbols: List of symbols currently managed by the strategy
            peak_prices: Dict of symbol -> peak price for trailing stop tracking
            last_rebalance_date: ISO date string of last live rebalance (preserves existing if None)
            last_regime: String regime name at last rebalance (preserves existing if None)
        """
        state_file = self._get_strategy_state_file()

        # Preserve existing trigger fields if not explicitly set
        existing = self._load_strategy_state()
        state = {
            "managed_symbols": sorted(managed_symbols),
            "peak_prices": peak_prices or {},
            "updated": datetime.now().isoformat(),
            "last_rebalance_date": last_rebalance_date if last_rebalance_date is not None else existing.get("last_rebalance_date"),
            "last_regime": last_regime if last_regime is not None else existing.get("last_regime"),
        }
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save strategy state: {e}[/yellow]")

    def _load_universe(self):
        """Load stock universe and initialize cache manager.

        Creates two Universe instances:
        - self._cache_universe: unfiltered (all stocks) — used for cache so all symbols get cached
        - self.universe: profile-filtered — used for strategy operations (scan, rebalance, backtest)
        """
        try:
            universe_path = self.config.paths.universe_file
            profile = self.config.get_profile(self.active_profile_name)

            # Unfiltered universe for cache (all ~300 stocks)
            self._cache_universe = Universe(universe_path)

            # Profile-filtered universe for strategy operations
            self.universe = Universe(universe_path, filter_universes=profile.universe_filter)

            console.print(
                f"[green]Universe loaded: {len(self.universe.get_all_stocks())} stocks "
                f"(profile: {self.active_profile_name})[/green]"
            )
            # Initialize cache manager with UNFILTERED universe (all symbols cached)
            self.cache = CacheManager(self.config, self._cache_universe)
        except Exception as e:
            console.print(f"[red]Failed to load universe: {e}[/red]")
            sys.exit(1)

    def _get_profile_sizing(self):
        """Get a PositionSizingConfig with profile overrides applied."""
        from .config import PositionSizingConfig
        profile = self.config.get_profile(self.active_profile_name)
        return self.config.position_sizing.model_copy(update={
            "target_positions": profile.target_positions,
            "min_positions": profile.min_positions,
            "max_positions": profile.max_positions,
            "max_single_position": profile.max_single_position,
        })

    def _ensure_auth(self) -> bool:
        """Ensure we have valid authentication."""
        if self.kite is None:
            console.print("[yellow]Not authenticated. Please login first.[/yellow]")
            return False
        return True

    def _display_menu(self):
        """Display the main menu."""
        profile = self.config.get_profile(self.active_profile_name)
        profile_label = self.active_profile_name.upper()
        title = f"FORTRESS MOMENTUM [{profile_label}]"
        # Show active strategy in subtitle
        strategy_desc = self.strategy.description if self.strategy else self.active_strategy
        subtitle = f"Strategy: {self.active_strategy.upper()} - {strategy_desc}"

        console.print(
            Panel(
                f"[bold bright_cyan]{title}[/bold bright_cyan]\n[bright_white]{subtitle}[/bright_white]",
                style="bright_blue",
            )
        )

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="cyan bold", width=3)
        table.add_column("Option", style="white", width=15)
        table.add_column("Description", style="dim")

        for key, option, desc in self.MENU_OPTIONS:
            table.add_row(key, option, desc)

        console.print(table)

    def _select_profile(self):
        """Prompt user to select a portfolio profile at startup."""
        profile_names = self.config.get_profile_names()
        if len(profile_names) <= 1:
            # Single profile (or no profiles configured) — use default
            self.active_profile_name = profile_names[0] if profile_names else "primary"
            return

        console.print(Panel("Select Portfolio Profile", style="bright_blue"))
        for i, name in enumerate(profile_names, 1):
            profile = self.config.get_profile(name)
            universes = " + ".join(profile.universe_filter)
            capital_str = format_currency(profile.initial_capital)
            console.print(
                f"  [bold bright_cyan]{i}[/bold bright_cyan] [dim]─[/dim] "
                f"{name.upper()} [dim]({universes}, {capital_str})[/dim]"
            )

        choice = Prompt.ask(
            "\nSelect profile",
            choices=[str(i) for i in range(1, len(profile_names) + 1)],
            default="1",
        )
        self.active_profile_name = profile_names[int(choice) - 1]
        console.print(f"[green]Profile: {self.active_profile_name.upper()}[/green]")

    def run(self):
        """Run the interactive application."""
        self._load_config()
        self._select_profile()
        # Apply per-profile strategy overrides (smallcap: wider stops, higher vol target, etc.)
        self.effective_config = self.config.with_profile_overrides(self.active_profile_name)
        if self.effective_config is not self.config:
            self.strategy = StrategyRegistry.get(self.active_strategy, self.effective_config)
            console.print("[dim]Strategy overrides applied for profile[/dim]")
        self._load_universe()

        while True:
            console.print()
            self._display_menu()

            choice = Prompt.ask("\nSelect option", default="0")

            if choice == "0":
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif choice == "1":
                self._do_login()
            elif choice == "2":
                self._do_status()
            elif choice == "3":
                self._do_scan()
            elif choice == "4":
                self._do_rebalance_interactive()
            elif choice == "5":
                self._do_backtest()
            elif choice == "6":
                self._do_select_strategy()
            elif choice == "7":
                self._do_trigger_check()
            elif choice == "9":
                self._do_market_phase_analysis()
            else:
                console.print("[red]Invalid option[/red]")

    def _do_login(self):
        """Authenticate with Zerodha."""
        console.print(Panel("Zerodha Authentication", style="bright_blue"))

        try:
            self.auth = ZerodhaAuth(
                self.config.zerodha.api_key, self.config.zerodha.api_secret
            )
            self.kite = self.auth.login_interactive()
            console.print("[green]✓ Authentication successful![/green]")

            # Initialize market data provider (use unfiltered universe for full instrument coverage)
            mapper = InstrumentMapper(self.kite, self._cache_universe)
            with console.status("[bold green]Loading instruments..."):
                mapper.load_instruments()
            self.market_data = MarketDataProvider(self.kite, mapper)

            # Update cache manager with market_data for fetching
            if self.cache:
                self.cache.market_data = self.market_data

            # Initialize portfolio (uses profile-filtered universe for sector mapping)
            self.portfolio = Portfolio(self.kite, self.universe)

            console.print("[green]✓ Ready for trading[/green]")

        except AuthenticationError as e:
            console.print(f"[red]Authentication failed: {e}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    def _do_status(self):
        """Display portfolio status."""
        console.print(Panel("Portfolio Status", style="bright_blue"))

        if not self._ensure_auth():
            return

        try:
            # Always refresh portfolio to get latest data from Zerodha
            snapshot = self.portfolio.refresh()

            # Portfolio summary
            table = Table(title="Portfolio Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")

            table.add_row("Total Value", format_currency(snapshot.total_value))
            table.add_row("Cash", format_currency(snapshot.cash))
            table.add_row("Invested", format_currency(snapshot.invested_value))
            table.add_row("Unrealized P&L", format_currency(snapshot.unrealized_pnl))
            # Calculate P&L percentage
            pnl_pct = snapshot.unrealized_pnl / snapshot.invested_value if snapshot.invested_value > 0 else 0.0
            table.add_row("P&L %", format_percentage(pnl_pct))
            table.add_row("Positions", str(len(snapshot.positions)))

            console.print(table)

            # Position details
            if snapshot.positions:
                pos_table = Table(title="Current Positions")
                pos_table.add_column("Symbol", style="cyan")
                pos_table.add_column("Qty", justify="right")
                pos_table.add_column("Avg Price", justify="right")
                pos_table.add_column("Current", justify="right")
                pos_table.add_column("P&L", justify="right")
                pos_table.add_column("P&L %", justify="right")

                for pos in snapshot.positions.values():
                    pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
                    pos_table.add_row(
                        pos.symbol,
                        str(pos.quantity),
                        format_currency(pos.average_price),
                        format_currency(pos.current_price),
                        f"[{pnl_color}]{format_currency(pos.unrealized_pnl)}[/{pnl_color}]",
                        f"[{pnl_color}]{format_percentage(pos.unrealized_pnl_pct)}[/{pnl_color}]",
                    )

                console.print(pos_table)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    def _do_scan(self):
        """Scan and rank stocks by momentum."""
        strategy_name = self.active_strategy.upper()
        console.print(Panel(f"Momentum Scan ({strategy_name})", style="bright_blue"))

        if not self._ensure_auth():
            return

        try:
            # Use cache manager - load and update if stale
            historical_data = self.cache.load_and_update()
            if not historical_data:
                console.print("[yellow]No cached data available.[/yellow]")
                return

            # Use BacktestDataProvider with cached data (no API calls during scan)
            from .market_data import BacktestDataProvider
            cached_provider = BacktestDataProvider(historical_data)

            # Initialize momentum engine with cached provider and cached data
            # Passing cached_data enables fast parallel NMS calculation
            self.momentum_engine = MomentumEngine(
                universe=self.universe,
                market_data=cached_provider,
                momentum_config=self.effective_config.pure_momentum,
                sizing_config=self._get_profile_sizing(),
                risk_config=self.effective_config.risk,
                strategy=self.strategy,  # Pass strategy for logic parity
                app_config=self.effective_config,
                cached_data=historical_data,  # Fast path for NMS calculation
            )

            # Use T-1 (last completed trading day) to avoid live/incomplete data
            t1_date = self.cache.get_target_date()
            as_of = datetime.combine(t1_date, datetime.max.time())
            console.print(f"[dim]Scan as-of date: {t1_date} (T-1)[/dim]")

            with console.status(f"[bold green]Calculating momentum scores ({strategy_name})..."):
                # Use rank_all_stocks which returns stocks passing entry filters
                rankings = self.momentum_engine.rank_all_stocks(
                    as_of_date=as_of,
                    filter_entry=True,  # Only show stocks passing entry filters
                )

            if not rankings:
                console.print("[yellow]No stocks passed entry filters[/yellow]")
                return

            # Display top stocks
            table = Table(title=f"Top {min(20, len(rankings))} Momentum Stocks ({strategy_name})")
            table.add_column("#", justify="right", width=3)
            table.add_column("Ticker", style="cyan")
            table.add_column("Sector")
            table.add_column("NMS", justify="right")
            table.add_column("6M Ret", justify="right")
            table.add_column("12M Ret", justify="right")
            table.add_column("52W High", justify="right")
            table.add_column("Filters", justify="center")

            for i, stock in enumerate(rankings[:20], 1):
                # Show filter status
                filter_status = "✓" if stock.passes_filters else "✗"
                filter_color = "green" if stock.passes_filters else "red"

                table.add_row(
                    str(i),
                    stock.ticker,
                    stock.sector[:12],
                    f"{stock.nms:.2f}",
                    format_percentage(stock.return_6m),
                    format_percentage(stock.return_12m),
                    format_percentage(stock.high_52w_proximity),
                    f"[{filter_color}]{filter_status}[/{filter_color}]",
                )

            console.print(table)
            console.print(f"\n[dim]Total stocks passing filters: {len(rankings)}[/dim]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    def _do_rebalance_interactive(self):
        """Interactive wrapper for rebalance with execution mode selection."""
        strategy_name = self.active_strategy.upper()
        console.print(Panel(f"Rebalance Portfolio ({strategy_name})", style="bright_blue"))

        # Prompt for execution mode
        console.print("[bold white]Execution Mode:[/bold white]")
        console.print("  [bold bright_cyan]1[/bold bright_cyan] [dim]─[/dim] Dry Run [dim](show plan only, no orders placed)[/dim]")
        console.print("  [bold bright_yellow]2[/bold bright_yellow] [dim]─[/dim] Live [dim](place real orders with confirmation)[/dim]")
        console.print()

        mode_choice = Prompt.ask("Select mode", choices=["1", "2"], default="1")

        if mode_choice == "1":
            console.print("[bright_cyan]► DRY-RUN mode[/bright_cyan]\n")
            self._do_rebalance(dry_run=True)
        else:
            console.print("[bold bright_yellow]► LIVE mode - orders will be placed![/bold bright_yellow]\n")
            self._do_rebalance(dry_run=False)

    def _do_rebalance(self, dry_run: bool = True):
        """Generate rebalance orders using same logic as backtest."""
        strategy_name = self.active_strategy.upper()
        mode_text = "DRY RUN" if dry_run else "LIVE"
        mode_style = "bright_blue" if dry_run else "bold bright_red"
        console.print(Panel(f"Rebalance ({strategy_name}) - {mode_text}", style=mode_style))

        if not self._ensure_auth():
            return

        try:
            # Use cache manager - load and update if stale
            historical_data = self.cache.load_and_update()
            if not historical_data:
                console.print("[red]Failed to load market data[/red]")
                return

            # Use BacktestDataProvider with cached data (no API calls during rebalance)
            from .market_data import BacktestDataProvider
            cached_provider = BacktestDataProvider(historical_data)

            # Initialize momentum engine with cached provider and cached data
            # Passing cached_data enables fast parallel NMS calculation
            self.momentum_engine = MomentumEngine(
                universe=self.universe,
                market_data=cached_provider,
                momentum_config=self.effective_config.pure_momentum,
                sizing_config=self._get_profile_sizing(),
                risk_config=self.effective_config.risk,
                regime_config=self.effective_config.regime,
                strategy=self.strategy,  # Pass strategy for logic parity
                app_config=self.effective_config,
                cached_data=historical_data,  # Fast path for NMS calculation
            )

            # Load combined holdings + today's positions for complete current state
            # This makes rebalance stateless and idempotent
            snapshot = self.portfolio.load_combined_positions()

            # Identify which symbols are managed by the strategy
            universe_symbols = {s.zerodha_symbol for s in self.universe.get_all_stocks()}
            profile = self.config.get_profile(self.active_profile_name)
            if profile.max_gold_allocation is not None and profile.max_gold_allocation == 0.0:
                defensive_symbols = set()
            else:
                defensive_symbols = {self.config.regime.gold_symbol}
            defensive_symbols.add(self.config.regime.cash_symbol)
            strategy_managed_symbols = universe_symbols | defensive_symbols

            # ETFs/funds NOT managed by strategy (user's external holdings)
            external_etfs = {
                "NIFTYBEES", "JUNIORBEES", "MID150BEES", "HDFCSML250",
                "HANGSENGBEES", "HNGSNGBEES", "LIQUIDCASE", "LIQUIDETF",
            }

            # Filter current holdings to only strategy-managed positions
            # LIQUIDBEES (cash_symbol) and GOLDBEES (gold_symbol) are ALWAYS managed
            # when present — they are the strategy's capital pool and hedge instrument
            managed_holdings: Dict[str, Position] = {}
            external_holdings: Dict[str, Position] = {}

            # Load strategy state to identify peak prices
            strategy_state = self._load_strategy_state()
            peak_prices = strategy_state.get("peak_prices", {})

            for symbol, pos in snapshot.positions.items():
                if symbol in external_etfs:
                    external_holdings[symbol] = pos
                elif symbol in defensive_symbols:
                    # Defensive assets (LIQUIDBEES, GOLDBEES) are always managed
                    managed_holdings[symbol] = pos
                elif symbol in universe_symbols:
                    # Equity stocks in universe are strategy-managed
                    managed_holdings[symbol] = pos
                else:
                    # Unknown symbol - treat as external
                    external_holdings[symbol] = pos

            # Calculate current value of equity positions (definitely strategy-managed)
            equity_holdings = {s: p for s, p in managed_holdings.items() if s not in defensive_symbols}
            current_equity_value = sum(pos.value for pos in equity_holdings.values())

            # Calculate defensive holdings value (GOLDBEES that are strategy-managed)
            defensive_value = sum(
                pos.value for s, pos in managed_holdings.items()
                if s in defensive_symbols
            )

            # managed_capital = value of ALL managed positions (equities + GOLDBEES + LIQUIDBEES)
            # LIQUIDBEES is the strategy's capital pool — no demat cash needed
            managed_capital = current_equity_value + defensive_value

            # Cold-start: no managed positions at all
            cash_symbol = self.config.regime.cash_symbol
            if managed_capital == 0:
                console.print(
                    f"\n[bold bright_yellow]► NO MANAGED POSITIONS[/bold bright_yellow]"
                )
                console.print(
                    f"  Target capital: {format_currency(profile.initial_capital)}"
                )
                cash_sym_price = None
                try:
                    ltp = self.kite.ltp([f"NSE:{cash_symbol}"])
                    cash_sym_price = ltp.get(f"NSE:{cash_symbol}", {}).get("last_price")
                except Exception:
                    pass
                if cash_sym_price:
                    units_needed = int(profile.initial_capital / cash_sym_price) + 10
                    console.print(
                        f"  Buy [bold]{units_needed}[/bold] units of {cash_symbol} "
                        f"(~₹{cash_sym_price:.2f} each = {format_currency(units_needed * cash_sym_price)})"
                    )
                else:
                    console.print(
                        f"  Buy {cash_symbol} worth {format_currency(profile.initial_capital)}"
                    )
                console.print(
                    f"  Then run rebalance again — {cash_symbol} will be converted to equity positions."
                )
                return

            # Portfolio summary with capital breakdown
            console.print(f"[bold cyan]Strategy:[/bold cyan] {self.active_strategy.upper()}  |  "
                         f"[bold cyan]Managed Capital:[/bold cyan] {format_currency(managed_capital)}  |  "
                         f"[bold cyan]Equity:[/bold cyan] {format_currency(current_equity_value)}  |  "
                         f"[bold cyan]Defensive:[/bold cyan] {format_currency(defensive_value)}")

            # Diagnostic breakdown: show merge details for managed positions
            if managed_holdings and snapshot.merge_diagnostics:
                ext_count = len(external_holdings)
                console.print(f"\n[dim]Portfolio Breakdown ({len(managed_holdings)} managed, {ext_count} external):[/dim]")
                diag_table = Table(show_header=True, box=None, padding=(0, 1))
                diag_table.add_column("Symbol", style="bold")
                diag_table.add_column("Hld", justify="right", style="dim")
                diag_table.add_column("Day", justify="right")
                diag_table.add_column("Net", justify="right", style="bold")
                diag_table.add_column("Value", justify="right", style="cyan")
                for symbol, pos in sorted(managed_holdings.items(), key=lambda x: x[1].value, reverse=True):
                    diag = snapshot.merge_diagnostics.get(symbol)
                    if diag:
                        day_str = f"+{diag.day_bought}" if diag.day_bought else ""
                        if diag.day_sold:
                            day_str += f"-{diag.day_sold}" if day_str else f"-{diag.day_sold}"
                        if not day_str:
                            day_str = "—"
                        diag_table.add_row(
                            symbol,
                            str(diag.holdings_qty),
                            day_str,
                            str(diag.net_qty),
                            format_currency(pos.value),
                        )
                    else:
                        diag_table.add_row(
                            symbol, str(pos.quantity), "—", str(pos.quantity), format_currency(pos.value),
                        )
                diag_table.add_row("", "", "", "[bold]Total[/bold]", f"[bold]{format_currency(managed_capital)}[/bold]")
                console.print(diag_table)

            # Show external holdings in a table if present
            if external_holdings:
                external_value = sum(pos.value for pos in external_holdings.values())
                console.print(f"\n[dim]External Holdings (not managed): {format_currency(external_value)}[/dim]")
                ext_table = Table(show_header=False, box=None, padding=(0, 2))
                ext_table.add_column("Symbol", style="dim")
                ext_table.add_column("Qty", justify="right", style="dim")
                ext_table.add_column("Value", justify="right", style="dim")
                for symbol, pos in sorted(external_holdings.items(), key=lambda x: x[1].value, reverse=True):
                    ext_table.add_row(symbol, str(pos.quantity), format_currency(pos.value))
                console.print(ext_table)

            # Total account value summary
            external_value = sum(pos.value for pos in external_holdings.values()) if external_holdings else 0
            total_account = managed_capital + external_value
            console.print(f"\n[bold]Total Account:[/bold] {format_currency(total_account)}  "
                         f"[dim](Managed {format_currency(managed_capital)} + "
                         f"External {format_currency(external_value)})[/dim]")

            # Get target portfolio with regime detection
            # Use managed_capital for weight calculations (not total portfolio)
            # Use T-1 (last completed trading day) to avoid live/incomplete data
            t1_date = self.cache.get_target_date()
            as_of = datetime.combine(t1_date, datetime.max.time())
            console.print(f"[dim]Rebalance as-of date: {t1_date} (T-1)[/dim]")

            max_per_sector = 3  # Match backtest default
            with console.status(f"[bold green]Calculating target portfolio ({strategy_name})..."):
                target_weights, regime = self.momentum_engine.select_portfolio_with_regime(
                    as_of_date=as_of,
                    portfolio_value=managed_capital,  # Use calculated capital
                    max_per_sector=max_per_sector,
                    profile_max_gold=profile.max_gold_allocation,
                )

            # Display current regime in compact format
            if regime:
                regime_color = {
                    "bullish": "green",
                    "normal": "cyan",
                    "caution": "yellow",
                    "defensive": "red",
                }.get(regime.regime.value, "white")

                regime_line = (f"[bold]Regime:[/bold] [{regime_color}]{regime.regime.value.upper()}[/{regime_color}]  |  "
                              f"52W: {regime.nifty_52w_position:.0%}  |  "
                              f"VIX: {regime.vix_level:.1f}  |  "
                              f"3M: {regime.nifty_3m_return:+.1%}")

                if regime.equity_weight < 1.0:
                    regime_line += (f"  |  [yellow]Allocation: Eq {regime.equity_weight:.0%} / "
                                   f"Gold {regime.gold_weight:.0%}[/yellow]")

                console.print(f"\n{regime_line}")

            if not target_weights:
                console.print("[yellow]No stocks passed entry filters[/yellow]")
                return

            # Sanity check: verify weights sum to ~100%
            total_weight = sum(target_weights.values())
            if abs(total_weight - 1.0) > 0.01:  # Allow 1% tolerance for rounding
                console.print(f"[yellow]Warning: Target weights sum to {total_weight:.1%} (expected 100%)[/yellow]")

            # Get top stocks for display (need to call select_top_stocks again)
            profile = self.config.get_profile(self.active_profile_name)
            top_stocks = self.momentum_engine.select_top_stocks(
                as_of_date=as_of,
                n=profile.target_positions,
                min_percentile=self.effective_config.pure_momentum.min_score_percentile,
                max_per_sector=max_per_sector,
            )

            # Identify defensive assets in target_weights (only gold, no cash)
            defensive_symbols = {self.config.regime.gold_symbol}
            equity_count = len([t for t in target_weights if t not in defensive_symbols])
            defensive_count = len([t for t in target_weights if t in defensive_symbols])

            console.print(f"\n[bold]Target Portfolio ({equity_count} stocks + {defensive_count} defensive):[/bold]")

            # Build ticker lookup for display
            ticker_lookup = {s.ticker: s for s in top_stocks}

            # Display sector summary (including defensive assets)
            sector_weights: Dict[str, float] = {}
            for ticker, weight in target_weights.items():
                if ticker in defensive_symbols:
                    # Gold hedge gets its own "sector"
                    if ticker == self.config.regime.gold_symbol:
                        sector_weights["GOLD (Hedge)"] = sector_weights.get("GOLD (Hedge)", 0) + weight
                else:
                    stock = ticker_lookup.get(ticker)
                    if stock:
                        sector = stock.sector
                        sector_weights[sector] = sector_weights.get(sector, 0) + weight

            sector_table = Table(title="Sector Allocation", show_header=True, header_style="bold")
            sector_table.add_column("Sector", style="cyan", min_width=25)
            sector_table.add_column("Weight", justify="right", min_width=8)
            sector_table.add_column("#", justify="right", min_width=3)

            sector_counts: Dict[str, int] = {}
            for stock in top_stocks:
                sector_counts[stock.sector] = sector_counts.get(stock.sector, 0) + 1
            # Add gold hedge count (no cash - user manages manually)
            if self.config.regime.gold_symbol in target_weights:
                sector_counts["GOLD (Hedge)"] = 1

            for sector in sorted(sector_weights.keys(), key=lambda s: sector_weights[s], reverse=True):
                weight = sector_weights[sector]
                count = sector_counts.get(sector, 0)
                # Different color scheme for defensive vs equity
                if "Hedge" in sector:
                    color = "yellow"
                else:
                    color = "green" if weight <= 0.30 else "yellow" if weight <= 0.40 else "red"
                sector_table.add_row(
                    sector,
                    f"[{color}]{format_percentage(weight)}[/{color}]",
                    str(count),
                )

            console.print(sector_table)

            # Fetch current prices for quantity calculation
            all_target_symbols = list(target_weights.keys())
            target_prices = {}
            try:
                ltp_data = self.kite.ltp([f"NSE:{s}" for s in all_target_symbols])
                for key, data in ltp_data.items():
                    symbol = key.split(":")[1]
                    target_prices[symbol] = data["last_price"]
            except Exception:
                pass  # Will show "-" for qty if price fetch fails

            # Display position details (equity stocks)
            table = Table(title="Target Positions", show_header=True, header_style="bold")
            table.add_column("Ticker", style="cyan", min_width=12)
            table.add_column("Sector", min_width=20)
            table.add_column("NMS", justify="right")
            table.add_column("52W%", justify="right")
            table.add_column("Wt%", justify="right")
            table.add_column("Qty", justify="right")
            table.add_column("Target", justify="right")

            for stock in top_stocks:
                weight = target_weights.get(stock.ticker, 0)
                target_value = managed_capital * weight
                price = target_prices.get(stock.ticker, 0)
                qty = int(target_value / price) if price > 0 else 0
                qty_str = str(qty) if qty > 0 else "-"
                table.add_row(
                    stock.ticker,
                    stock.sector,
                    f"{stock.nms:.2f}",
                    f"{stock.high_52w_proximity:.0%}",
                    f"{weight:.1%}",
                    qty_str,
                    format_currency(target_value),
                )

            # Add gold hedge to the table (no cash - user manages manually)
            gold_symbol = self.config.regime.gold_symbol
            if gold_symbol in target_weights:
                weight = target_weights[gold_symbol]
                target_value = managed_capital * weight
                price = target_prices.get(gold_symbol, 0)
                qty = int(target_value / price) if price > 0 else 0
                qty_str = str(qty) if qty > 0 else "-"
                table.add_row(
                    f"[yellow]{gold_symbol}[/yellow]",
                    "[yellow]Hedge[/yellow]",
                    "-",
                    "-",
                    f"[yellow]{format_percentage(weight)}[/yellow]",
                    f"[yellow]{qty_str}[/yellow]",
                    f"[yellow]{format_currency(target_value)}[/yellow]",
                )

            console.print(table)

            # Calculate trades needed
            # IMPORTANT: Only consider strategy-managed holdings, not external ETFs
            target_tickers = set(target_weights.keys())

            # Use structured data: (ticker, value, extra_info)
            sells = []  # (ticker, current_value)
            partial_sells = []  # (ticker, reduce_amount, current_weight, target_weight)
            buys = []  # (ticker, target_value)
            increases = []  # (ticker, current_weight, target_weight)

            # Check managed holdings for sells/reductions
            for symbol, pos in managed_holdings.items():
                if symbol not in target_tickers:
                    # Position should be fully exited (only for equity, not defensive)
                    if symbol not in defensive_symbols:
                        sells.append((symbol, pos.value))
                    # For defensive assets not in target (rare), don't suggest selling
                else:
                    # Check if we need to reduce position
                    target_weight = target_weights.get(symbol, 0)
                    target_value = managed_capital * target_weight
                    current_value = pos.value

                    # For defensive assets, DON'T suggest reducing
                    if symbol in defensive_symbols:
                        pass
                    else:
                        # For equity stocks, check weight-based reduction
                        current_weight = current_value / managed_capital if managed_capital > 0 else 0
                        if current_weight > target_weight * 1.1:  # 10% tolerance
                            reduce_amount = current_value - target_value
                            partial_sells.append((symbol, reduce_amount, current_weight, target_weight))

            # Note: Gold skip logic is handled by select_portfolio_with_regime() via
            # _should_skip_gold() using the config-driven gold_skip_logic setting.

            # Build execution plan with quantities
            executor = RebalanceExecutor(
                kite=self.kite,
                portfolio=self.portfolio,
                instrument_mapper=self.market_data.mapper,
                order_manager=OrderManager(
                    self.kite,
                    RiskGovernor(self.config.risk, self.config.portfolio),
                    dry_run=dry_run,
                ),
                universe=self.universe,
                risk_config=self.config.risk,
            )

            # Get current prices for all symbols (include gold + cash symbols for surplus sweep)
            sweep_symbols = {self.config.regime.gold_symbol, self.config.regime.cash_symbol}
            all_symbols = list(target_tickers | set(managed_holdings.keys()) | sweep_symbols)
            current_prices = {}
            try:
                ltp_data = self.kite.ltp([f"NSE:{s}" for s in all_symbols])
                for key, data in ltp_data.items():
                    symbol = key.split(":")[1]
                    current_prices[symbol] = data["last_price"]
            except Exception as e:
                console.print(f"[red]Error fetching prices: {e}[/red]")
                return

            # PARITY WITH BACKTEST: Enforce stop losses BEFORE building execution plan
            # This removes positions that hit stop loss from target_weights
            # Same logic as backtest.py lines 1541-1569
            stop_loss_exits, updated_peak_prices = self._enforce_stop_losses(
                holdings=managed_holdings,
                current_prices=current_prices,
                peak_prices=peak_prices,
                target_weights=target_weights,  # Modified in-place
                defensive_symbols=defensive_symbols,
            )

            # Display enforced stop loss exits prominently
            if stop_loss_exits:
                console.print(f"\n[bold bright_red]🛑 STOP LOSS TRIGGERED - ENFORCED EXITS[/bold bright_red]")
                for symbol, loss_pct, stop_type in stop_loss_exits:
                    console.print(f"  [bright_red]✗ {symbol}[/bright_red] [bold]{loss_pct:+.1%}[/bold] - {stop_type}")
                console.print("[dim]  These positions will be SOLD (same as backtest behavior)[/dim]")

                # Renormalize weights after removing stop-loss exits
                if target_weights:
                    total_weight = sum(target_weights.values())
                    if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
                        scale = 1.0 / total_weight
                        target_weights = {k: v * scale for k, v in target_weights.items()}
                        console.print(f"[dim]  Weights renormalized after exits[/dim]")

            plan = executor.build_plan(
                target_weights=target_weights,
                current_holdings=managed_holdings,
                managed_capital=managed_capital,
                current_prices=current_prices,
                gold_symbol=self.config.regime.gold_symbol,
                cash_symbol=self.config.regime.cash_symbol,
            )

            # Display execution plan
            self._display_execution_plan(plan)

            # Check for positions approaching stop loss (warnings only)
            stop_loss_warnings = self._check_stop_loss_warnings(
                holdings=managed_holdings,
                current_prices=current_prices,
                peak_prices=updated_peak_prices,
                enforced_exits=stop_loss_exits,
                defensive_symbols=defensive_symbols,
            )
            if stop_loss_warnings:
                console.print(f"\n[bold bright_yellow]⚠ APPROACHING STOP LOSS[/bold bright_yellow]")
                for symbol, loss_pct, stop_type in stop_loss_warnings:
                    console.print(f"  [bright_yellow]↓ {symbol}[/bright_yellow] {loss_pct:+.1%} - {stop_type}")
                console.print("[dim]  Monitor these positions closely[/dim]")

            # Show warnings
            for warning in plan.warnings:
                console.print(f"[bright_yellow]⚠ {warning}[/bright_yellow]")

            # Check margin
            if not plan.margin_sufficient:
                console.print(f"\n[bold bright_red]⚠ INSUFFICIENT MARGIN[/bold bright_red]")
                console.print(f"  [dim]Available:[/dim] [white]{format_currency(plan.available_cash)}[/white]")
                console.print(f"  [dim]Need:[/dim]      [bright_red]{format_currency(plan.net_cash_needed)}[/bright_red]")

            # Update strategy state with current target symbols and peak prices
            current_managed = set(managed_holdings.keys())
            new_target = set(target_weights.keys())
            # Only full exits leave managed set; reductions stay managed
            exit_tickers = {t.symbol for t in plan.sell_trades if t.action == TradeAction.SELL_EXIT}
            all_managed = (current_managed | new_target) - exit_tickers

            # Update peak prices for positions still held (for trailing stop tracking)
            # Initialize peak for new positions at their current price
            final_peak_prices = {}
            for symbol in all_managed:
                if symbol in updated_peak_prices:
                    # Keep updated peak from stop loss check
                    final_peak_prices[symbol] = updated_peak_prices[symbol]
                elif symbol in current_prices:
                    # New position - initialize peak at current price
                    final_peak_prices[symbol] = current_prices[symbol]

            self._save_strategy_state(list(all_managed), final_peak_prices)

            # Check market hours for live execution
            if not dry_run:
                market_open, market_msg = validate_market_hours()
                if not market_open:
                    console.print(f"\n[bright_yellow]⚠ {market_msg}[/bright_yellow]")
                    console.print("[dim]Orders cannot be placed outside market hours[/dim]")
                    return

                # Double confirmation flow
                exec_result = self._execute_with_confirmation(executor, plan)

                if exec_result is None:
                    # User cancelled — don't save rebalance date or regime
                    return

                # Reconcile state based on actual execution results
                if exec_result.failures:
                    failed_value = 0.0
                    for order_result in exec_result.failures:
                        o = order_result.order
                        if o.order_type == OrderType.BUY:
                            # Failed buy: remove from managed if it wasn't already held
                            if o.symbol not in current_managed:
                                all_managed.discard(o.symbol)
                                final_peak_prices.pop(o.symbol, None)
                            failed_value += o.value
                        elif o.order_type == OrderType.SELL:
                            # Failed sell: symbol is still held
                            all_managed.add(o.symbol)

                    if failed_value > 0:
                        console.print(
                            f"\n[bold bright_yellow]⚠ {format_currency(failed_value)} "
                            f"from failed buy orders remains as demat cash — "
                            f"buy {self.config.regime.cash_symbol} to redeploy[/bold bright_yellow]"
                        )

                # After live execution, persist rebalance date and regime for trigger checks
                regime_value = regime.regime.value if regime else None
                self._save_strategy_state(
                    list(all_managed), final_peak_prices,
                    last_rebalance_date=date.today().isoformat(),
                    last_regime=regime_value,
                )
            else:
                console.print("\n[bright_cyan]─── Dry-run complete ───[/bright_cyan]")
                console.print("[dim]Select Live mode (option 2) to place orders[/dim]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")

    def _enforce_stop_losses(
        self,
        holdings: Dict[str, Position],
        current_prices: Dict[str, float],
        peak_prices: Dict[str, float],
        target_weights: Dict[str, float],
        defensive_symbols: set,
    ) -> Tuple[List[Tuple[str, float, str]], Dict[str, float]]:
        """
        Enforce stop losses by removing positions from target_weights.

        PARITY WITH BACKTEST: Uses same logic as backtest.py lines 1541-1569.
        Uses strategy's get_stop_loss_config() for tiered/adaptive stops.

        Args:
            holdings: Current managed holdings
            current_prices: Current prices for all symbols
            peak_prices: Peak prices from strategy state (for trailing stops)
            target_weights: Target weights dict (will be modified)
            defensive_symbols: Set of defensive symbols (gold, etc.)

        Returns:
            Tuple of:
            - List of (symbol, loss_percentage, stop_type) for positions exited
            - Updated peak_prices dict
        """
        exits = []
        updated_peaks = dict(peak_prices)  # Copy to update

        for symbol, pos in holdings.items():
            # Skip defensive assets (gold, cash)
            if symbol in defensive_symbols:
                continue

            current_price = current_prices.get(symbol, pos.current_price)
            entry_price = pos.average_price

            if entry_price <= 0 or current_price <= 0:
                continue

            # Calculate gains
            gain_from_entry = (current_price - entry_price) / entry_price

            # Update peak price tracking (same as backtest line 1551)
            old_peak = updated_peaks.get(symbol, entry_price)
            if current_price > old_peak:
                updated_peaks[symbol] = current_price
            peak_price = updated_peaks.get(symbol, entry_price)

            # Get stop loss config from strategy (same as backtest line 1543)
            if self.momentum_engine and hasattr(self.momentum_engine, 'get_stop_loss_config'):
                stop_config = self.momentum_engine.get_stop_loss_config(symbol, gain_from_entry)
                # Handle both StopLossConfig object and dict
                if hasattr(stop_config, 'initial_stop'):
                    initial_stop = stop_config.initial_stop
                    trailing_stop = stop_config.trailing_stop
                    trailing_activation = getattr(stop_config, 'trailing_activation', 0.08)
                else:
                    initial_stop = stop_config.get('initial_stop', 0.18)
                    trailing_stop = stop_config.get('trailing_stop', 0.15)
                    trailing_activation = stop_config.get('trailing_activation', 0.08)
            else:
                # Fallback to config defaults
                initial_stop = self.config.risk.initial_stop_loss
                trailing_stop = self.config.risk.trailing_stop
                trailing_activation = self.config.risk.trailing_activation

            # Check initial stop loss (same as backtest line 1562)
            if gain_from_entry <= -initial_stop:
                # ENFORCE: Remove from target weights
                target_weights.pop(symbol, None)
                exits.append((symbol, gain_from_entry, f"INITIAL STOP ({initial_stop:.0%} loss)"))
                # Clean up peak tracking for exited position
                updated_peaks.pop(symbol, None)
                continue

            # Check trailing stop (same as backtest lines 1567-1568)
            # Trailing activates after position has gained >= trailing_activation
            if peak_price > entry_price * (1 + trailing_activation):
                gain_from_peak = (current_price - peak_price) / peak_price
                if gain_from_peak <= -trailing_stop:
                    # ENFORCE: Remove from target weights
                    target_weights.pop(symbol, None)
                    exits.append((
                        symbol,
                        gain_from_entry,
                        f"TRAILING STOP ({trailing_stop:.0%} from peak, was +{(peak_price/entry_price - 1):.0%})"
                    ))
                    # Clean up peak tracking for exited position
                    updated_peaks.pop(symbol, None)
                    continue

        return sorted(exits, key=lambda x: x[1]), updated_peaks

    def _check_stop_loss_warnings(
        self,
        holdings: Dict[str, Position],
        current_prices: Dict[str, float],
        peak_prices: Dict[str, float],
        enforced_exits: List[Tuple[str, float, str]],
        defensive_symbols: set,
    ) -> List[Tuple[str, float, str]]:
        """
        Check for positions approaching stop loss thresholds (warning only).

        This is for positions that haven't hit stops yet but are close.

        Args:
            holdings: Current managed holdings
            current_prices: Current prices
            peak_prices: Peak prices for trailing calculation
            enforced_exits: Positions already enforced (to exclude)
            defensive_symbols: Set of defensive symbols

        Returns:
            List of (symbol, loss_percentage, warning_type) for positions at risk
        """
        warnings = []
        enforced_symbols = {e[0] for e in enforced_exits}

        for symbol, pos in holdings.items():
            # Skip defensive assets and already-enforced exits
            if symbol in defensive_symbols or symbol in enforced_symbols:
                continue

            current_price = current_prices.get(symbol, pos.current_price)
            entry_price = pos.average_price

            if entry_price <= 0 or current_price <= 0:
                continue

            gain_from_entry = (current_price - entry_price) / entry_price

            # Get stop config
            if self.momentum_engine and hasattr(self.momentum_engine, 'get_stop_loss_config'):
                stop_config = self.momentum_engine.get_stop_loss_config(symbol, gain_from_entry)
                if hasattr(stop_config, 'initial_stop'):
                    initial_stop = stop_config.initial_stop
                    trailing_stop = stop_config.trailing_stop
                else:
                    initial_stop = stop_config.get('initial_stop', 0.18)
                    trailing_stop = stop_config.get('trailing_stop', 0.15)
            else:
                initial_stop = 0.18
                trailing_stop = 0.15

            # Warning if approaching initial stop (within 5% of trigger)
            warning_threshold = -initial_stop * 0.7  # Warn at 70% of stop
            if gain_from_entry <= warning_threshold and gain_from_entry > -initial_stop:
                warnings.append((
                    symbol,
                    gain_from_entry,
                    f"approaching initial stop ({gain_from_entry:.1%} vs {-initial_stop:.0%} trigger)"
                ))

        return sorted(warnings, key=lambda x: x[1])

    def _display_execution_plan(self, plan: RebalancePlan):
        """Display execution plan with quantities and prices."""
        console.print("\n[bold bright_white]═══ EXECUTION PLAN ═══[/bold bright_white]")

        if not plan.trades:
            console.print("[dim]No rebalancing needed - portfolio is on target[/dim]")
            return

        if plan.sell_trades:
            sell_table = Table(
                title="[bold bright_red]SELL Orders[/bold bright_red]",
                show_header=True,
                header_style="bold bright_red",
                border_style="red",
            )
            sell_table.add_column("Symbol", style="bright_red")
            sell_table.add_column("Action", style="red")
            sell_table.add_column("Qty", justify="right", style="white")
            sell_table.add_column("Entry", justify="right", style="dim")
            sell_table.add_column("Now", justify="right", style="white")
            sell_table.add_column("Value", justify="right", style="bright_red")
            sell_table.add_column("P&L", justify="right")
            sell_table.add_column("P&L%", justify="right")

            total_pnl = 0.0
            for trade in plan.sell_trades:
                action = "EXIT" if trade.action == TradeAction.SELL_EXIT else "REDUCE"
                pnl = trade.pnl_absolute
                pnl_pct = trade.pnl_percent
                total_pnl += pnl

                # Color P&L based on profit/loss
                if pnl >= 0:
                    pnl_str = f"[bright_green]+{format_currency(pnl)}[/bright_green]"
                    pnl_pct_str = f"[bright_green]+{pnl_pct:.1%}[/bright_green]"
                else:
                    pnl_str = f"[bright_red]{format_currency(pnl)}[/bright_red]"
                    pnl_pct_str = f"[bright_red]{pnl_pct:.1%}[/bright_red]"

                sell_table.add_row(
                    trade.symbol,
                    action,
                    str(trade.quantity),
                    format_currency(trade.entry_price),
                    format_currency(trade.price),
                    format_currency(trade.value),
                    pnl_str,
                    pnl_pct_str,
                )
            console.print(sell_table)

            # Show total P&L summary for sells
            if total_pnl >= 0:
                console.print(f"  [bold]Total Realized P&L:[/bold] [bright_green]+{format_currency(total_pnl)}[/bright_green]")
            else:
                console.print(f"  [bold]Total Realized P&L:[/bold] [bright_red]{format_currency(total_pnl)}[/bright_red]")

        if plan.buy_new_trades:
            buy_table = Table(
                title="[bold bright_green]BUY Orders (New Positions)[/bold bright_green]",
                show_header=True,
                header_style="bold bright_green",
                border_style="green",
            )
            buy_table.add_column("Symbol", style="bright_green")
            buy_table.add_column("Qty", justify="right", style="white")
            buy_table.add_column("Price", justify="right", style="white")
            buy_table.add_column("Value", justify="right", style="bright_green")
            buy_table.add_column("Target Wt", justify="right", style="dim")

            for trade in plan.buy_new_trades:
                buy_table.add_row(
                    trade.symbol,
                    str(trade.quantity),
                    format_currency(trade.price),
                    format_currency(trade.value),
                    f"{trade.target_weight:.1%}",
                )
            console.print(buy_table)

        if plan.buy_increase_trades:
            inc_table = Table(
                title="[bold bright_cyan]INCREASE Orders (Step-ups)[/bold bright_cyan]",
                show_header=True,
                header_style="bold bright_cyan",
                border_style="cyan",
            )
            inc_table.add_column("Symbol", style="bright_cyan")
            inc_table.add_column("Qty", justify="right", style="white")
            inc_table.add_column("Price", justify="right", style="white")
            inc_table.add_column("Value", justify="right", style="bright_cyan")
            inc_table.add_column("Weight Change", justify="right", style="dim")

            for trade in plan.buy_increase_trades:
                inc_table.add_row(
                    trade.symbol,
                    str(trade.quantity),
                    format_currency(trade.price),
                    format_currency(trade.value),
                    f"{trade.current_weight:.1%} → {trade.target_weight:.1%}",
                )
            console.print(inc_table)

        # Cash flow summary with semantic colors
        console.print(f"\n[bold bright_white]Cash Flow:[/bold bright_white]")
        console.print(f"  [dim]Sell proceeds:[/dim]    [bright_green]+{format_currency(plan.total_sell_value)}[/bright_green]")
        console.print(f"  [dim]Buy cost:[/dim]         [bright_red]-{format_currency(plan.total_buy_value)}[/bright_red]")
        console.print(f"  [dim]Demat cash:[/dim]       [white]{format_currency(plan.available_cash)}[/white] [dim](not used for buys)[/dim]")
        if plan.net_cash_needed > 0:
            console.print(f"  [bold bright_yellow]► Net cash needed: {format_currency(plan.net_cash_needed)}[/bold bright_yellow]")
        else:
            console.print(f"  [bold bright_cyan]► Net cash freed: {format_currency(-plan.net_cash_needed)}[/bold bright_cyan]")

    def _execute_with_confirmation(
        self, executor: RebalanceExecutor, plan: RebalancePlan
    ) -> Optional[ExecutionResult]:
        """Execute plan with double confirmation.

        Returns:
            ExecutionResult if execution happened, None if cancelled or no trades.
        """
        if not plan.trades:
            console.print("[dim]No trades to execute[/dim]")
            return None

        console.print("\n[bold bright_yellow]═══ CONFIRMATION REQUIRED ═══[/bold bright_yellow]")
        console.print(f"[white]Total orders:[/white] [bold]{len(plan.trades)}[/bold]")
        console.print(f"  [bright_red]SELL:[/bright_red]     {len(plan.sell_trades)} orders ({format_currency(plan.total_sell_value)})")
        console.print(f"  [bright_green]BUY NEW:[/bright_green]  {len(plan.buy_new_trades)} orders")
        console.print(f"  [bright_cyan]INCREASE:[/bright_cyan] {len(plan.buy_increase_trades)} orders")
        console.print(f"  [dim]Total buy:[/dim] {format_currency(plan.total_buy_value)}")

        # First confirmation
        if not Confirm.ask("\n[bold bright_white]Proceed with order placement?[/bold bright_white]", default=False):
            console.print("[bright_yellow]Cancelled by user[/bright_yellow]")
            return None

        # Second confirmation - type CONFIRM
        console.print("\n[bold bright_red]⚠ FINAL CONFIRMATION ⚠[/bold bright_red]")
        console.print("[white]This will place REAL orders with your broker.[/white]")
        confirm_text = Prompt.ask("Type [bold bright_white]CONFIRM[/bold bright_white] to execute", default="")

        if confirm_text.strip().upper() != "CONFIRM":
            console.print("[bright_yellow]Cancelled - did not type CONFIRM[/bright_yellow]")
            return None

        # Execute with progress
        console.print("\n[bold bright_green]Executing orders...[/bold bright_green]")

        with Progress(
            SpinnerColumn(style="bright_cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="bright_green", finished_style="bright_green"),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[bright_cyan]Placing orders...", total=len(plan.trades))

            def on_order_complete(order, result):
                status = "[bright_green]✓[/bright_green]" if result.success else "[bright_red]✗[/bright_red]"
                msg = "" if result.success else f" [dim]- {result.message}[/dim]"
                progress.update(
                    task,
                    advance=1,
                    description=f"{status} {order.symbol} {order.order_type.value}{msg}",
                )

            result = executor.execute_plan(plan, progress_callback=on_order_complete)

        # Show results
        self._display_execution_results(result)
        return result

    def _display_execution_results(self, result: ExecutionResult):
        """Display execution results with errors prominently shown."""
        console.print("\n[bold bright_white]═══ EXECUTION RESULTS ═══[/bold bright_white]")

        if result.successes:
            console.print(f"\n[bold bright_green]Successful Orders ({len(result.successes)}):[/bold bright_green]")
            for order_result in result.successes:
                o = order_result.order
                console.print(
                    f"  [bright_green]✓[/bright_green] {o.symbol}: {o.order_type.value} "
                    f"{o.quantity} @ {format_currency(o.price or 0)}"
                )
                if o.order_id:
                    console.print(f"    [dim]Order ID: {o.order_id}[/dim]")

        if result.failures:
            console.print(f"\n[bold bright_red]Failed Orders ({len(result.failures)}):[/bold bright_red]")
            for order_result in result.failures:
                o = order_result.order
                console.print(f"  [bright_red]✗[/bright_red] {o.symbol}: {o.order_type.value} {o.quantity}")
                console.print(f"    [red]Error: {order_result.message}[/red]")

        # Summary
        console.print(f"\n[bold bright_white]Summary:[/bold bright_white]")
        console.print(f"  [dim]Succeeded:[/dim] [bold bright_green]{len(result.successes)}[/bold bright_green]")
        console.print(f"  [dim]Failed:[/dim]    [bold bright_red]{len(result.failures)}[/bold bright_red]")

        if result.failures:
            console.print(
                "\n[bright_yellow]⚠ Some orders failed. Please review errors above "
                "and handle manually if needed.[/bright_yellow]"
            )
        elif result.successes:
            console.print("\n[bold bright_green]✓ All orders placed successfully![/bold bright_green]")

    def _do_trigger_check(self):
        """Check if any dynamic rebalancing triggers have fired.

        Uses cached data only — no login required. Evaluates:
        - Regime transition (NORMAL→CAUTION etc.)
        - VIX recovery from spike
        - Portfolio drawdown
        - Market crash (1M return)
        - Breadth thrust
        - Regular interval (max days between rebalances)
        """
        console.print(Panel("Trigger Check", style="bright_blue"))

        try:
            # Load cached data (no API call)
            historical_data = self.cache.load(silent=True)
            if not historical_data or len(historical_data) < 50:
                console.print("[yellow]Insufficient cached data (need ≥50 symbols). "
                              "Run option 3 (Scan) first to populate cache.[/yellow]")
                return

            # Build lightweight MomentumEngine with cached provider
            from .market_data import BacktestDataProvider
            cached_provider = BacktestDataProvider(historical_data)

            self.momentum_engine = MomentumEngine(
                universe=self.universe,
                market_data=cached_provider,
                momentum_config=self.effective_config.pure_momentum,
                sizing_config=self._get_profile_sizing(),
                risk_config=self.effective_config.risk,
                regime_config=self.effective_config.regime,
                strategy=self.strategy,
                app_config=self.effective_config,
                cached_data=historical_data,
            )

            # T-1 date
            t1_date = self.cache.get_target_date()
            as_of = datetime.combine(t1_date, datetime.max.time())

            # --- Detect current regime ---
            regime = self.momentum_engine.detect_current_regime(as_of)
            current_regime_enum = regime.regime if regime else None
            vix_level = regime.vix_level if regime else 15.0

            # --- Load strategy state ---
            strategy_state = self._load_strategy_state()
            last_rebal_str = strategy_state.get("last_rebalance_date")
            last_regime_str = strategy_state.get("last_regime")

            # Convert last_regime string back to MarketRegime enum
            previous_regime = None
            if last_regime_str:
                try:
                    previous_regime = MarketRegime(last_regime_str.lower())
                except ValueError:
                    pass

            # --- Days since last rebalance ---
            dyn_config = self.effective_config.dynamic_rebalance
            nifty_df = historical_data.get("NIFTY 50")

            if last_rebal_str and nifty_df is not None and not nifty_df.empty:
                last_rebal_date = date.fromisoformat(last_rebal_str)
                # Count trading days between last rebalance and T-1 using Nifty as calendar
                trading_dates = nifty_df.index
                mask = (trading_dates.date > last_rebal_date) & (trading_dates.date <= t1_date)
                days_since_last = int(mask.sum())
            else:
                # No previous rebalance recorded — force trigger
                days_since_last = dyn_config.max_days_between

            # --- Detect stale regime ---
            regime_stale = days_since_last > dyn_config.max_days_between
            if regime_stale:
                # Don't compare against a stale regime — REGULAR_INTERVAL handles long gaps
                previous_regime = None

            # --- VIX peak 20d ---
            vix_df = historical_data.get("INDIA VIX")
            if vix_df is not None and len(vix_df) >= 20:
                vix_peak_20d = float(vix_df["close"].iloc[-20:].max())
            else:
                vix_peak_20d = vix_level

            # --- Portfolio drawdown (approximate from strategy state) ---
            managed_symbols = strategy_state.get("managed_symbols", [])
            peak_prices = strategy_state.get("peak_prices", {})
            portfolio_value = 0.0
            peak_value = 0.0

            for symbol in managed_symbols:
                df = historical_data.get(symbol)
                if df is not None and not df.empty:
                    current_price = float(df["close"].iloc[-1])
                    peak_price = peak_prices.get(symbol, current_price)
                    portfolio_value += current_price
                    peak_value += peak_price

            if peak_value > 0:
                portfolio_drawdown = (portfolio_value - peak_value) / peak_value
            else:
                portfolio_drawdown = 0.0

            # --- Market 1M and 3M returns (Nifty 50) ---
            if nifty_df is not None and len(nifty_df) >= 22:
                market_1m_return = float(
                    nifty_df["close"].iloc[-1] / nifty_df["close"].iloc[-22] - 1
                )
            else:
                market_1m_return = 0.0

            if nifty_df is not None and len(nifty_df) >= 64:
                market_3m_return = float(
                    nifty_df["close"].iloc[-1] / nifty_df["close"].iloc[-64] - 1
                )
            else:
                market_3m_return = 0.0

            # E6: Early warning crash detection (slow grind without VIX spike)
            early_warning_active = (
                market_1m_return <= -0.05 and market_3m_return <= -0.08
            )

            # --- Breadth thrust ---
            # Build daily breadth series (pct_above_50ma) for last 15 days
            breadth_thrust = False
            try:
                breadth_values = []
                for offset in range(15, -1, -1):
                    if offset == 0:
                        day_data = historical_data
                    else:
                        # Slice each symbol's data to end at -offset
                        day_data = {}
                        for sym, df in historical_data.items():
                            if len(df) > offset:
                                day_data[sym] = df.iloc[:-offset]
                            else:
                                day_data[sym] = df

                    breadth = calculate_market_breadth(day_data)
                    breadth_values.append(breadth.pct_above_50ma)

                if len(breadth_values) >= 11:
                    breadth_series = pd.Series(breadth_values)
                    bt_result = detect_breadth_thrust(breadth_series)
                    breadth_thrust = bt_result.is_thrust
            except Exception:
                pass

            # --- Evaluate triggers ---
            trigger = should_trigger_rebalance(
                days_since_last=days_since_last,
                current_regime=current_regime_enum,
                previous_regime=previous_regime,
                vix_level=vix_level,
                vix_peak_20d=vix_peak_20d,
                portfolio_drawdown=portfolio_drawdown,
                market_1m_return=market_1m_return,
                breadth_thrust=breadth_thrust,
                min_days_between=dyn_config.min_days_between,
                max_days_between=dyn_config.max_days_between,
                vix_recovery_decline=dyn_config.vix_recovery_decline,
                vix_spike_threshold=dyn_config.vix_spike_threshold,
                drawdown_threshold=dyn_config.drawdown_threshold,
                crash_threshold=dyn_config.crash_threshold,
            )

            # --- Display results ---
            regime_str = current_regime_enum.value.upper() if current_regime_enum else "UNKNOWN"
            prev_str = previous_regime.value.upper() if previous_regime else "—"
            if regime_stale and last_regime_str:
                regime_display = f"{regime_str} (last: {last_regime_str.upper()}, {days_since_last}d ago — stale)"
            elif prev_str != "—":
                regime_display = f"{regime_str} (was {prev_str})"
            else:
                regime_display = regime_str

            # Stress score and allocation info
            stress_str = f"{regime.stress_score:.2f}" if regime else "—"
            alloc_str = ""
            if regime and regime.stress_score > 0:
                from .indicators import calculate_graduated_allocation
                eq, gd, _ = calculate_graduated_allocation(regime.stress_score, self.effective_config.regime)
                alloc_str = f" → equity {eq:.0%} / gold {gd:.0%}"

            lines = [
                f"  Days since rebalance:  [bold]{days_since_last}[/bold] (min: {dyn_config.min_days_between}, max: {dyn_config.max_days_between})",
                f"  Current regime:        [bold]{regime_display}[/bold]",
                f"  Stress score:          [bold]{stress_str}[/bold]{alloc_str}",
                f"  VIX:                   [bold]{vix_level:.1f}[/bold] (peak 20d: {vix_peak_20d:.1f})",
                f"  Market 1M return:      [bold]{market_1m_return:+.1%}[/bold]",
                f"  Market 3M return:      [bold]{market_3m_return:+.1%}[/bold]",
                f"  Portfolio drawdown:    [bold]{portfolio_drawdown:+.1%}[/bold]",
                f"  Breadth thrust:        [bold]{'Yes' if breadth_thrust else 'No'}[/bold]",
            ]

            if early_warning_active:
                lines.append(f"  [bright_yellow]⚠ EARLY WARNING: 1M ({market_1m_return:+.1%}) ≤ -5% AND 3M ({market_3m_return:+.1%}) ≤ -8% — crash avoidance may activate[/bright_yellow]")

            if regime_stale:
                lines.append(f"  [bright_yellow]⚠ Last rebalance was {days_since_last} days ago — regime history unreliable[/bright_yellow]")

            if trigger.should_rebalance:
                trigger_names = ", ".join(trigger.triggers_fired)
                lines.append("")
                lines.append(f"  [bold bright_green]✓ REBALANCE RECOMMENDED[/bold bright_green]")
                lines.append(f"  Trigger: [bold]{trigger_names}[/bold] ({trigger.urgency} urgency)")
                lines.append(f"  → Run option 4 to generate rebalance plan")
                border_style = "bright_green"
            else:
                days_remaining = dyn_config.max_days_between - days_since_last
                lines.append("")
                lines.append(f"  [dim]✗ No rebalance needed today[/dim]")
                lines.append(f"  [dim]Next forced rebalance in: {days_remaining} trading days[/dim]")
                border_style = "dim"

            panel_content = "\n".join(lines)
            console.print(Panel(
                panel_content,
                title=f"[bold]TRIGGER CHECK — {t1_date}[/bold]",
                border_style=border_style,
            ))

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")

    def _do_backtest(self):
        """Run historical backtest."""
        console.print(Panel("Backtest - Historical Simulation", style="bright_blue"))

        # Strategy selection sub-menu
        console.print("[bold]Select Strategy[/bold]")
        strategies = StrategyRegistry.list_strategies()

        strategy_table = Table(show_header=False, box=None, padding=(0, 2))
        strategy_table.add_column("Num", style="cyan", width=4)
        strategy_table.add_column("Active", width=3)
        strategy_table.add_column("Name", style="bold")
        strategy_table.add_column("Description", style="dim")

        for i, (name, desc) in enumerate(strategies, 1):
            active = "✓" if name == self.active_strategy else " "
            active_cell = f"[green]{active}[/green]" if name == self.active_strategy else active
            strategy_table.add_row(str(i), active_cell, name, desc)

        console.print(strategy_table)

        strategy_input = Prompt.ask(
            "\nSelect strategy (or Enter to keep current)",
            default=""
        ).strip()

        if strategy_input:
            try:
                choice_idx = int(strategy_input) - 1
                if 0 <= choice_idx < len(strategies):
                    new_strategy = strategies[choice_idx][0]
                    self.active_strategy = new_strategy
                    self._init_strategy()
                else:
                    console.print("[red]Invalid selection[/red]")
                    return
            except ValueError:
                # Try matching by name
                for name, _ in strategies:
                    if name.lower() == strategy_input.lower():
                        self.active_strategy = name
                        self._init_strategy()
                        break
                else:
                    console.print("[red]Invalid strategy name[/red]")
                    return

        console.print(f"[green]Using strategy: {self.active_strategy}[/green]\n")

        # Get backtest parameters
        yesterday = datetime.now() - timedelta(days=1)
        while yesterday.weekday() >= 5:
            yesterday -= timedelta(days=1)
        end_date = yesterday

        console.print("[bold]Backtest Duration[/bold]")
        console.print("[dim]Enter months (3, 6, 12, 24, 48) or 'c' for custom[/dim]\n")

        duration_input = Prompt.ask("Duration", default="6").strip().lower()

        if duration_input == "c":
            start_str = Prompt.ask("Start date (YYYY-MM-DD)", default="2024-01-01")
            try:
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
            except ValueError:
                console.print("[red]Invalid date format[/red]")
                return
            end_str = Prompt.ask("End date (YYYY-MM-DD)", default=end_date.strftime("%Y-%m-%d"))
            try:
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
            except ValueError:
                console.print("[red]Invalid date format[/red]")
                return
            if start_date >= end_date:
                console.print("[red]Start date must be before end date[/red]")
                return
        else:
            try:
                months = int(duration_input)
                if months < 1:
                    console.print("[red]Duration must be at least 1 month[/red]")
                    return
                start_date = end_date - relativedelta(months=months)
            except ValueError:
                console.print("[red]Invalid input[/red]")
                return

        rebal_input = Prompt.ask("Rebalance days (5=weekly, 21=monthly)", default=str(self.config.rebalancing.rebalance_days))
        try:
            rebalance_days = int(rebal_input)
            if rebalance_days < 1:
                console.print("[red]Rebalance days must be at least 1[/red]")
                return
        except ValueError:
            console.print("[red]Invalid input[/red]")
            return

        console.print(f"\n[bold]Backtest: {start_date.date()} to {end_date.date()}[/bold]")
        console.print(f"[dim]Rebalance every {rebalance_days} trading days[/dim]\n")

        # Use cache manager - load and update if stale
        historical_data = self.cache.load()

        if len(historical_data) < 50:
            console.print(f"[yellow]Insufficient cached data ({len(historical_data)} symbols).[/yellow]")
            if self.cache.market_data:
                historical_data = self.cache.load_and_update()
            else:
                console.print("[yellow]Login first to fetch data, or run Rebalance to populate cache.[/yellow]")
                return

        if len(historical_data) < 50:
            console.print("[red]Failed to load sufficient data[/red]")
            return

        console.print(f"[green]Using {len(historical_data)} symbols[/green]")

        # Check earliest data point; backfill if needed for full backtest coverage
        non_empty = {s: df for s, df in historical_data.items() if len(df) > 0}
        if non_empty:
            earliest = min(df.index[0] for df in non_empty.values())
            latest = max(df.index[-1] for df in non_empty.values())
            console.print(f"Data range: {earliest.date()} to {latest.date()}")

            # NMS 257-day + 200-SMA + 52-week high warmup ≈ 15 months before bt start
            required_data_start = (start_date - relativedelta(months=15)).date()

            if earliest.date() > required_data_start:
                if self.cache.market_data:
                    console.print(
                        f"[cyan]Data starts {earliest.date()} but need "
                        f"{required_data_start} — backfilling …[/cyan]"
                    )
                    backfilled = self.cache.backfill_history(required_data_start)
                    if backfilled > 0:
                        historical_data = self.cache.data
                        earliest = min(
                            df.index[0] for df in historical_data.values() if len(df) > 0
                        )
                        console.print(f"[green]Data now starts {earliest.date()}[/green]")
                else:
                    console.print(
                        f"[yellow]Data only starts {earliest.date()}; "
                        f"login (Option 1) to auto-fetch older history back to "
                        f"{required_data_start} for better backtest coverage.[/yellow]"
                    )

        console.print()

        # Run backtest with selected strategy
        console.print(f"[cyan]Using strategy: {self.active_strategy}[/cyan]\n")

        profile = self.config.get_profile(self.active_profile_name)
        bt_config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=profile.initial_capital,
            rebalance_days=rebalance_days,
            transaction_cost=self.effective_config.costs.transaction_cost,
            target_positions=profile.target_positions,
            min_score_percentile=self.effective_config.pure_momentum.min_score_percentile,
            min_52w_high_prox=self.effective_config.pure_momentum.min_52w_high_prox,
            initial_stop_loss=self.effective_config.risk.initial_stop_loss,
            trailing_stop=self.effective_config.risk.trailing_stop,
            weight_6m=self.effective_config.pure_momentum.weight_6m,
            weight_12m=self.effective_config.pure_momentum.weight_12m,
            strategy_name=self.active_strategy,
            profile_max_gold=profile.max_gold_allocation,
        )

        engine = BacktestEngine(
            universe=self.universe,
            historical_data=historical_data,
            config=bt_config,
            app_config=self.effective_config,
            strategy_name=self.active_strategy,
        )

        with console.status(f"[bold green]Running backtest ({self.active_strategy})..."):
            result = engine.run()

        self._display_backtest_result(result)

    def _display_rebalance_trail(self, trail: list):
        """Print compact rebalance-by-rebalance trail log."""
        console.print("\n[bold bright_cyan]═══ REBALANCE TRAIL ═══[/bold bright_cyan]\n")

        regime_colors = {
            "bullish": "bright_green",
            "normal": "cyan",
            "caution": "yellow",
            "defensive": "bright_red",
        }

        for rec in trail:
            color = regime_colors.get(rec.regime, "white")
            regime_label = f"[{color}]{rec.regime.upper()}[/{color}]"

            # Header line: #N  DATE  REGIME  VALUE
            console.print(
                f"[bold]#{rec.rebalance_number}[/bold]  "
                f"{rec.date.strftime('%Y-%m-%d')}  "
                f"{regime_label}  "
                f"[bright_white]{format_currency(rec.portfolio_value)}[/bright_white]"
            )

            # Trend / VIX / Breadth signals
            trend_mark = "[green]✓[/green]" if rec.trend_above_sma else "[red]✗[/red]"
            nifty_str = f"NIFTY {rec.nifty_price:,.0f}"
            sma_str = f"SMA {rec.nifty_sma:,.0f}" if rec.nifty_sma > 0 else "SMA n/a"
            vix_color = "green" if rec.vix_value < 18 else "yellow" if rec.vix_value < 25 else "red"
            breadth_color = "green" if rec.breadth_value >= 0.50 else "yellow" if rec.breadth_value >= 0.30 else "red"
            console.print(
                f"  Trend: {nifty_str} vs {sma_str} {trend_mark}"
                f" | VIX: [{vix_color}]{rec.vix_value:.1f}[/{vix_color}]"
                f" | Breadth: [{breadth_color}]{rec.breadth_value:.0%}[/{breadth_color}]"
            )

            # Sleeve weights
            console.print(
                f"  Equity {rec.equity_weight:.0%}"
                f" | Gold {rec.gold_weight:.0%}"
                f" | Liquid {rec.liquid_weight:.0%}"
            )

            # Scaling factors (only if any < 1.0)
            scale_parts = []
            if rec.vol_scale < 0.999:
                scale_parts.append(f"vol={rec.vol_scale:.2f}")
            if rec.breadth_scale < 0.999:
                scale_parts.append(f"breadth={rec.breadth_scale:.2f}")
            if rec.gold_exhaustion_scale < 0.999:
                freed_pct = (1.0 - rec.gold_exhaustion_scale) * rec.gold_weight * 100
                if rec.gold_weight > 0:
                    freed_pct = (1.0 - rec.gold_exhaustion_scale) * 100
                scale_parts.append(f"gold_exhaust={rec.gold_exhaustion_scale:.2f} ({freed_pct:.0f}% freed)")
            if scale_parts:
                console.print(f"  [dim]Scales: {', '.join(scale_parts)}[/dim]")

            # Equity picks (top 8)
            if rec.equity_picks:
                picks_str = "  Picks: "
                for i, (sym, score, weight) in enumerate(rec.equity_picks[:8]):
                    picks_str += f"{sym} {weight:.1%} ({score:+.2f})"
                    if i < min(len(rec.equity_picks), 8) - 1:
                        picks_str += "  "
                if len(rec.equity_picks) > 8:
                    picks_str += f" [dim]+{len(rec.equity_picks) - 8} more[/dim]"
                console.print(picks_str)

            # Gate failures (only if any)
            if rec.gate_failures:
                blocked_str = ", ".join(f"{s} ({sc:+.2f})" for s, sc in rec.gate_failures[:5])
                if len(rec.gate_failures) > 5:
                    blocked_str += f" +{len(rec.gate_failures) - 5} more"
                console.print(f"  [dim]Gate blocked: {blocked_str}[/dim]")

            # Trade count
            console.print(f"  → {rec.trade_count} trades\n")

    def _display_backtest_result(self, result: BacktestResult):
        """Display backtest results."""
        # Show rebalance trail before summary
        if result.rebalance_trail:
            self._display_rebalance_trail(result.rebalance_trail)

        # Show strategy used
        strategy_name = getattr(result, "strategy_name", "dual_momentum")
        strategy_upper = strategy_name.upper()

        # Performance metrics
        table = Table(title=f"Backtest Results ({strategy_upper})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Strategy", strategy_name)

        # Capital amounts
        table.add_row("Starting Capital", format_currency(result.initial_capital))
        profit_color = "bright_green" if result.total_profit > 0 else "bright_red"
        table.add_row("Final Value", f"[{profit_color}]{format_currency(result.final_value)}[/{profit_color}]")
        table.add_row("Total Profit", f"[{profit_color}]{format_currency(result.total_profit)}[/{profit_color}]")
        table.add_row("Peak Value", f"[bright_cyan]{format_currency(result.peak_value)}[/bright_cyan]")

        # Performance metrics
        cagr_color = "bright_green" if result.cagr > 0.20 else "yellow" if result.cagr > 0 else "bright_red"
        dd_color = "bright_green" if result.max_drawdown > -0.15 else "yellow" if result.max_drawdown > -0.25 else "bright_red"

        table.add_row("Total Return", format_percentage(result.total_return))
        table.add_row("CAGR", f"[{cagr_color}]{format_percentage(result.cagr)}[/{cagr_color}]")
        table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        table.add_row("Max Drawdown", f"[{dd_color}]{format_percentage(result.max_drawdown)}[/{dd_color}]")
        table.add_row("Win Rate", format_percentage(result.win_rate))
        table.add_row("Total Trades", str(result.total_trades))

        console.print(table)

        # Benchmark comparison table
        if result.nifty_50_return is not None or result.nifty_midcap_100_return is not None:
            bench_table = Table(title="Benchmark Comparison")
            bench_table.add_column("Strategy/Index", style="cyan")
            bench_table.add_column("Return", justify="right")
            bench_table.add_column("vs NIFTY 50", justify="right")

            # Strategy row
            strategy_vs_nifty = ""
            if result.nifty_50_return is not None:
                diff = result.total_return - result.nifty_50_return
                color = "green" if diff > 0 else "red"
                strategy_vs_nifty = f"[{color}]{diff:+.1%}[/{color}]"

            strategy_color = "green" if result.total_return > 0 else "red"
            strategy_label = f"FORTRESS ({strategy_upper})"
            bench_table.add_row(
                strategy_label,
                f"[{strategy_color}]{format_percentage(result.total_return)}[/{strategy_color}]",
                strategy_vs_nifty,
            )

            # Nifty 50 row
            if result.nifty_50_return is not None:
                n50_color = "green" if result.nifty_50_return > 0 else "red"
                bench_table.add_row(
                    "NIFTY 50",
                    f"[{n50_color}]{format_percentage(result.nifty_50_return)}[/{n50_color}]",
                    "-",
                )

            # Nifty Midcap 100 row
            if result.nifty_midcap_100_return is not None:
                mc_color = "green" if result.nifty_midcap_100_return > 0 else "red"
                mc_vs_nifty = ""
                if result.nifty_50_return is not None:
                    mc_diff = result.nifty_midcap_100_return - result.nifty_50_return
                    mc_diff_color = "green" if mc_diff > 0 else "red"
                    mc_vs_nifty = f"[{mc_diff_color}]{mc_diff:+.1%}[/{mc_diff_color}]"
                bench_table.add_row(
                    "NIFTY MIDCAP 100",
                    f"[{mc_color}]{format_percentage(result.nifty_midcap_100_return)}[/{mc_color}]",
                    mc_vs_nifty,
                )

            console.print(bench_table)

        # Regime analysis table
        if result.time_in_regime and len(result.time_in_regime) > 0:
            regime_table = Table(title="Regime Analysis")
            regime_table.add_column("Metric", style="cyan")
            regime_table.add_column("Value", justify="right")

            regime_table.add_row("Regime Transitions", str(result.regime_transitions))

            # Sort regimes by expected order
            regime_order = ["bullish", "normal", "caution", "defensive"]
            for regime_name in regime_order:
                if regime_name in result.time_in_regime:
                    pct = result.time_in_regime[regime_name]
                    # Color based on regime type
                    if regime_name == "bullish":
                        color = "green"
                    elif regime_name == "normal":
                        color = "cyan"
                    elif regime_name == "caution":
                        color = "yellow"
                    else:  # defensive
                        color = "red"
                    regime_table.add_row(
                        f"Time in {regime_name.upper()}",
                        f"[{color}]{pct:.1%}[/{color}]",
                    )

            console.print(regime_table)

        # Sector distribution summary
        if not result.sector_allocations.empty:
            sector_cols = [c for c in result.sector_allocations.columns if c != "date"]
            if sector_cols:
                avg_weights = result.sector_allocations[sector_cols].mean()
                max_weights = result.sector_allocations[sector_cols].max()

                # Sort by average weight descending
                sorted_sectors = avg_weights.sort_values(ascending=False)

                sector_table = Table(title="Sector Distribution (Avg)")
                sector_table.add_column("Sector", style="cyan")
                sector_table.add_column("Avg Weight", justify="right")
                sector_table.add_column("Max Weight", justify="right")

                for sector in sorted_sectors.index[:8]:  # Top 8 sectors
                    avg = avg_weights[sector]
                    mx = max_weights[sector]
                    if avg > 0.01:  # Skip negligible allocations
                        color = "green" if mx <= 0.30 else "yellow" if mx <= 0.40 else "red"
                        sector_table.add_row(
                            sector[:15],
                            f"{avg:.1%}",
                            f"[{color}]{mx:.1%}[/{color}]"
                        )

                console.print(sector_table)

    # ------------------------------------------------------------------
    # Market Phase Analysis (Menu 9)
    # ------------------------------------------------------------------

    MARKET_PHASES = [
        ("2015 Bull Run", "2015-03-01", "2015-08-24", "Bullish"),
        ("China Scare & Recovery", "2015-08-24", "2016-03-01", "Bearish→Recovery"),
        ("Pre-Demonetization Bull", "2016-03-01", "2016-11-08", "Bullish"),
        ("Demonetization Shock & Recovery", "2016-11-08", "2017-04-01", "Bearish→Recovery"),
        ("2017 Bull Run", "2017-04-01", "2018-01-29", "Bullish"),
        ("NBFC / IL&FS Crisis", "2018-01-29", "2019-03-01", "Bearish"),
        ("2019 Recovery (Corp Tax Cut)", "2019-03-01", "2020-01-20", "Sideways→Bullish"),
        ("COVID Crash", "2020-01-20", "2020-04-01", "Crash"),
        ("Post-COVID Rally", "2020-04-01", "2021-10-18", "Bullish"),
        ("2022 Correction (Ukraine/Rates)", "2021-10-18", "2022-06-17", "Bearish"),
        ("2023-24 Recovery & Bull Run", "2022-06-17", "2024-09-27", "Bullish"),
        ("Late 2024-25 Correction", "2024-09-27", "2026-01-31", "Bearish/Sideways"),
    ]

    # Warmup / data requirements are computed dynamically from the earliest
    # phase so that positions are active from day 1 of Phase 1:
    #   bt_start        = earliest_phase - 12 months  (engine warmup)
    #   required_data   = bt_start - 30 months         (NMS + 52w + SMA lookback)
    _BT_WARMUP_MONTHS = 12   # engine warmup before first phase
    _DATA_LOOKBACK_MONTHS = 30  # historical data needed before bt_start

    def _do_market_phase_analysis(self):
        """Run a continuous 10-year backtest segmented by market phases."""
        console.print(
            Panel(
                f"[bold bright_cyan]Multi-Phase Market Analysis[/bold bright_cyan]\n"
                f"[bright_white]Strategy: {self.active_strategy.upper()}[/bright_white]",
                style="bright_blue",
            )
        )

        # ---- Transparent data loading & backfill ----
        # Dates are computed dynamically from the earliest phase so the
        # system automatically fetches enough history for NMS + filter warmup.
        earliest_phase_dt = datetime.strptime(self.MARKET_PHASES[0][1], "%Y-%m-%d")
        bt_start = earliest_phase_dt - relativedelta(months=self._BT_WARMUP_MONTHS)
        bt_end = datetime.strptime(self.MARKET_PHASES[-1][2], "%Y-%m-%d")

        # Data must cover NMS lookback (252 days), 200-day SMA, 52-week high
        required_data_start = (bt_start - relativedelta(months=self._DATA_LOOKBACK_MONTHS)).date()

        historical_data = self.cache.load()

        # Try update if cache is near-empty
        if len(historical_data) < 50:
            console.print("[yellow]Insufficient cached data.[/yellow]")
            if self.cache.market_data:
                historical_data = self.cache.load_and_update(lookback_days=4500)
            else:
                console.print(
                    "[yellow]Login first to fetch data, or run "
                    "Rebalance to populate cache.[/yellow]"
                )
                return

        if len(historical_data) < 50:
            console.print("[red]Failed to load sufficient data[/red]")
            return

        console.print(f"[green]Loaded {len(historical_data)} symbols[/green]")

        # Check earliest data point; backfill if needed
        earliest = min(df.index[0] for df in historical_data.values() if len(df) > 0)
        latest = max(df.index[-1] for df in historical_data.values() if len(df) > 0)
        console.print(f"Data range: {earliest.date()} to {latest.date()}")

        if earliest.date() > required_data_start:
            if self.cache.market_data:
                console.print(
                    f"[cyan]Data starts {earliest.date()} but need "
                    f"{required_data_start} — backfilling …[/cyan]"
                )
                backfilled = self.cache.backfill_history(required_data_start)
                if backfilled > 0:
                    # Reload from the now-updated session cache
                    historical_data = self.cache.data
                    earliest = min(
                        df.index[0] for df in historical_data.values() if len(df) > 0
                    )
                    console.print(f"[green]Data now starts {earliest.date()}[/green]")
            else:
                console.print(
                    f"[yellow]Data only starts {earliest.date()}; "
                    f"login (Option 1) to auto-fetch older history back to "
                    f"{required_data_start} for better early-phase coverage.[/yellow]"
                )

        # ---- Determine effective phases based on available data ----
        # NMS requires ~252 trading days from data start before positions begin.
        # Compute the earliest date the strategy can realistically take positions.
        data_start = earliest.date()
        # 252 trading days ≈ 365 calendar days; add buffer for filters to stabilise
        nms_ready_date = pd.Timestamp(data_start) + pd.Timedelta(days=420)

        active_phases = []
        skipped_phases = []
        for name, start_str, end_str, phase_type in self.MARKET_PHASES:
            p_end = pd.Timestamp(end_str)
            if p_end <= nms_ready_date:
                skipped_phases.append(name)
            else:
                active_phases.append((name, start_str, end_str, phase_type))

        if skipped_phases:
            console.print(
                f"[yellow]Skipping {len(skipped_phases)} phase(s) before "
                f"NMS warmup completes (~{nms_ready_date.date()}): "
                f"{', '.join(skipped_phases)}[/yellow]"
            )
            console.print(
                "[dim]Tip: login and the system will auto-fetch older "
                "history to cover these phases.[/dim]"
            )

        if not active_phases:
            console.print("[red]No phases have sufficient data coverage.[/red]")
            return

        # ---- Run backtest ----
        phase1_start = active_phases[0][1]
        console.print(
            f"\n[cyan]Running continuous backtest: "
            f"{bt_start.date()} → {bt_end.date()} "
            f"(warmup until {phase1_start}, then {len(active_phases)} phases) "
            f"({self.active_strategy.upper()}) ...[/cyan]"
        )

        profile = self.config.get_profile(self.active_profile_name)
        bt_config = BacktestConfig(
            start_date=bt_start,
            end_date=bt_end,
            initial_capital=profile.initial_capital,
            rebalance_days=5,
            transaction_cost=self.effective_config.costs.transaction_cost,
            target_positions=profile.target_positions,
            min_positions=profile.min_positions,
            min_score_percentile=self.effective_config.pure_momentum.min_score_percentile,
            min_52w_high_prox=self.effective_config.pure_momentum.min_52w_high_prox,
            initial_stop_loss=self.effective_config.risk.initial_stop_loss,
            trailing_stop=self.effective_config.risk.trailing_stop,
            weight_6m=self.effective_config.pure_momentum.weight_6m,
            weight_12m=self.effective_config.pure_momentum.weight_12m,
            strategy_name=self.active_strategy,
            profile_max_gold=profile.max_gold_allocation,
        )

        engine = BacktestEngine(
            universe=self.universe,
            historical_data=historical_data,
            config=bt_config,
            app_config=self.effective_config,
            strategy_name=self.active_strategy,
        )

        with console.status(f"[bold green]Running backtest ({self.active_strategy})..."):
            result = engine.run()

        console.print(
            f"\n[bold green]Backtest complete — "
            f"Return: {result.total_return:+.1%}  "
            f"CAGR: {result.cagr:.1%}  "
            f"Sharpe: {result.sharpe_ratio:.2f}  "
            f"Max DD: {result.max_drawdown:.1%}[/bold green]\n"
        )

        # Segment equity curve by phase
        equity = result.equity_curve
        initial_capital = result.initial_capital
        phase_results: list[dict] = []

        # ---- Refine active phases based on actual first trade ----
        # Strategy entry filters are strict (52w high proximity, volume surge,
        # min turnover, etc.); even after NMS warmup the first position may
        # lag.  Skip phases that end before the strategy actually trades.
        first_buy = next(
            (t for t in result.trades if t.action == "BUY"), None
        )
        if not first_buy:
            console.print("[yellow]No trades executed in entire backtest![/yellow]")
            return

        first_trade_ts = pd.Timestamp(first_buy.date)
        pre_trade = [p for p in active_phases if pd.Timestamp(p[2]) <= first_trade_ts]
        active_phases = [p for p in active_phases if pd.Timestamp(p[2]) > first_trade_ts]

        if not active_phases:
            console.print("[red]No phases overlap with trading activity.[/red]")
            return

        console.print(
            f"[dim]First position entered: {first_trade_ts.date()} "
            f"({first_buy.symbol})[/dim]"
        )
        if pre_trade:
            console.print(
                f"[dim]Skipping {len(pre_trade)} pre-trading phase(s): "
                f"{', '.join(p[0] for p in pre_trade)}[/dim]"
            )

        # Portfolio value at first-active-phase start (after warmup + pre-trade)
        p1_ts = pd.Timestamp(active_phases[0][1])
        eq_at_p1 = equity.loc[equity.index >= p1_ts]
        phase1_value = eq_at_p1.iloc[0] if len(eq_at_p1) > 0 else initial_capital

        # Helper: benchmark return for a date range
        def _bench_return(symbol_keys: list[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp):
            bench = None
            for key in symbol_keys:
                if key in historical_data:
                    bench = historical_data[key]
                    break
            if bench is None:
                return None
            mask = (bench.index >= start_ts) & (bench.index <= end_ts)
            period = bench.loc[mask, "close"]
            if len(period) < 2:
                return None
            return period.iloc[-1] / period.iloc[0] - 1

        def _nifty_return(start_ts: pd.Timestamp, end_ts: pd.Timestamp):
            return _bench_return(["NIFTY 50", "NIFTY50", "NSE:NIFTY50"], start_ts, end_ts)

        def _midcap_return(start_ts: pd.Timestamp, end_ts: pd.Timestamp):
            return _bench_return(["NIFTY MIDCAP 100", "NIFTYMIDCAP100"], start_ts, end_ts)

        # Show warmup summary
        warmup_return = phase1_value / initial_capital - 1
        console.print(
            f"[dim]Warmup period ({bt_start.date()} → {active_phases[0][1]}): "
            f"₹{initial_capital:,.0f} → ₹{phase1_value:,.0f} ({warmup_return:+.1%})[/dim]\n"
        )

        # ---- Phase-by-phase detail table (printed as we go) ----
        console.print(
            Panel(
                "[bold bright_cyan]Phase-by-Phase Breakdown[/bold bright_cyan]",
                style="bright_blue",
                expand=False,
            )
        )

        for idx, (name, start_str, end_str, phase_type) in enumerate(active_phases, 1):
            p_start = pd.Timestamp(start_str)
            p_end = pd.Timestamp(end_str)

            mask = (equity.index >= p_start) & (equity.index <= p_end)
            eq_slice = equity.loc[mask]

            if len(eq_slice) < 2:
                console.print(f"  [yellow]Phase {idx}: {name} — insufficient data, skipping[/yellow]")
                continue

            start_val = eq_slice.iloc[0]
            end_val = eq_slice.iloc[-1]
            phase_return = end_val / start_val - 1

            days = (eq_slice.index[-1] - eq_slice.index[0]).days
            years = max(days / 365.25, 1 / 365.25)
            cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else 0.0

            _, max_dd = calculate_drawdown(eq_slice)

            daily_rets = eq_slice.pct_change().dropna()
            if len(daily_rets) > 1 and daily_rets.std() > 0:
                sharpe = (daily_rets.mean() - 0.06 / 252) / daily_rets.std() * np.sqrt(252)
            else:
                sharpe = 0.0

            buy_trades = sum(
                1 for t in result.trades
                if p_start <= pd.Timestamp(t.date) <= p_end and t.action == "BUY"
            )
            sell_trades = sum(
                1 for t in result.trades
                if p_start <= pd.Timestamp(t.date) <= p_end and t.action == "SELL"
            )

            nifty_ret = _nifty_return(p_start, p_end)
            midcap_ret = _midcap_return(p_start, p_end)
            alpha = phase_return - nifty_ret if nifty_ret is not None else None
            alpha_midcap = phase_return - midcap_ret if midcap_ret is not None else None

            cum_return = end_val / phase1_value - 1

            pr = {
                "name": name, "type": phase_type, "start_val": start_val,
                "end_val": end_val, "return": phase_return, "cagr": cagr,
                "max_dd": max_dd, "sharpe": sharpe, "buy_trades": buy_trades,
                "sell_trades": sell_trades, "nifty_ret": nifty_ret,
                "midcap_ret": midcap_ret, "alpha": alpha,
                "alpha_midcap": alpha_midcap,
                "cum_return": cum_return, "days": days,
                "start_str": start_str, "end_str": end_str,
            }
            phase_results.append(pr)

            # Print detail panel for this phase
            def _cpct(v, invert=False):
                if v is None:
                    return "[dim]N/A[/dim]"
                pos = "red" if invert else "green"
                neg = "green" if invert else "red"
                c = pos if v >= 0 else neg
                return f"[{c}]{v:+.1%}[/{c}]"

            sharpe_c = "green" if sharpe >= 0.5 else ("yellow" if sharpe >= 0 else "red")
            alpha_str = _cpct(alpha) if alpha is not None else "[dim]N/A[/dim]"
            alpha_mc_str = _cpct(alpha_midcap) if alpha_midcap is not None else "[dim]N/A[/dim]"

            phase_table = Table(
                title=f"Phase {idx}: {name}  [{phase_type}]",
                show_header=False,
                title_style="bold",
                min_width=60,
                show_lines=False,
            )
            phase_table.add_column("", style="cyan", min_width=22)
            phase_table.add_column("", justify="right", min_width=18)
            phase_table.add_column("", style="cyan", min_width=22)
            phase_table.add_column("", justify="right", min_width=18)

            phase_table.add_row(
                "Period", f"{start_str} → {end_str}",
                "Trading Days", str(len(eq_slice)),
            )
            phase_table.add_row(
                "Strategy Return", _cpct(phase_return),
                "NIFTY 50 Return", _cpct(nifty_ret),
            )
            phase_table.add_row(
                "MIDCAP 100 Return", _cpct(midcap_ret),
                "Alpha vs NIFTY", alpha_str,
            )
            phase_table.add_row(
                "Alpha vs MIDCAP", alpha_mc_str,
                "Sharpe Ratio", f"[{sharpe_c}]{sharpe:.2f}[/{sharpe_c}]",
            )
            phase_table.add_row(
                "Max Drawdown", _cpct(max_dd, invert=True),
                "CAGR", _cpct(cagr),
            )
            phase_table.add_row(
                "Buy Trades", str(buy_trades),
                "Sell Trades", str(sell_trades),
            )
            phase_table.add_row(
                "Start Value", f"₹{start_val:,.0f}",
                "End Value", f"₹{end_val:,.0f}",
            )
            phase_table.add_row(
                "Cum. Return", _cpct(cum_return),
                "", "",
            )

            console.print(phase_table)
            console.print()

        if not phase_results:
            console.print("[red]No phases had sufficient data.[/red]")
            return

        # ---- Summary table ----
        console.print(
            Panel(
                "[bold bright_cyan]Summary[/bold bright_cyan]",
                style="bright_blue",
                expand=False,
            )
        )

        def _cpct(v, invert=False):
            if v is None:
                return "[dim]N/A[/dim]"
            pos = "red" if invert else "green"
            neg = "green" if invert else "red"
            c = pos if v >= 0 else neg
            return f"[{c}]{v:+.1%}[/{c}]"

        # Compact all-phases table
        has_midcap = any(pr.get("midcap_ret") is not None for pr in phase_results)
        summary_tbl = Table(title="All Phases", show_lines=True, title_style="bold cyan")
        summary_tbl.add_column("#", style="dim", width=3, justify="right")
        summary_tbl.add_column("Phase", style="bold", max_width=28)
        summary_tbl.add_column("Type", style="dim", max_width=16)
        summary_tbl.add_column("Return", justify="right")
        summary_tbl.add_column("Max DD", justify="right")
        summary_tbl.add_column("NIFTY", justify="right")
        summary_tbl.add_column("α N50", justify="right")
        if has_midcap:
            summary_tbl.add_column("MIDCAP", justify="right")
            summary_tbl.add_column("α MC", justify="right")
        summary_tbl.add_column("End Val", justify="right")

        for i, pr in enumerate(phase_results, 1):
            row = [
                str(i), pr["name"], pr["type"],
                _cpct(pr["return"]),
                _cpct(pr["max_dd"], invert=True),
                _cpct(pr["nifty_ret"]),
                _cpct(pr["alpha"]),
            ]
            if has_midcap:
                row.append(_cpct(pr.get("midcap_ret")))
                row.append(_cpct(pr.get("alpha_midcap")))
            row.append(f"₹{pr['end_val']:,.0f}")
            summary_tbl.add_row(*row)

        console.print(summary_tbl)

        # Overall stats — scoped to the phases period (excl. warmup)
        last_phase_end_val = phase_results[-1]["end_val"] if phase_results else result.final_value
        phases_return = last_phase_end_val / phase1_value - 1
        phases_days = (
            pd.Timestamp(active_phases[-1][2]) - pd.Timestamp(active_phases[0][1])
        ).days
        phases_years = max(phases_days / 365.25, 1 / 365.25)
        phases_cagr = (last_phase_end_val / phase1_value) ** (1 / phases_years) - 1

        overall = Table(
            title=f"Overall ({self.active_strategy.upper()})  ·  {active_phases[0][1]} → {active_phases[-1][2]}",
            show_header=False, show_lines=False, title_style="bold cyan",
        )
        overall.add_column("", style="cyan", min_width=25)
        overall.add_column("", justify="right", min_width=20)

        overall.add_row("Capital at Phase 1 Start", f"₹{phase1_value:,.0f}")
        overall.add_row("Final Value", f"₹{last_phase_end_val:,.0f}")
        overall.add_row("Peak Value", f"₹{result.peak_value:,.0f}")
        overall.add_row("Total Return (Phases)", _cpct(phases_return))
        overall.add_row("CAGR (Phases)", _cpct(phases_cagr))
        overall.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        overall.add_row("Max Drawdown", _cpct(result.max_drawdown))
        overall.add_row("Win Rate", f"{result.win_rate:.0%}")
        overall.add_row("Total Trades", str(result.total_trades))

        nifty_total = _nifty_return(
            pd.Timestamp(active_phases[0][1]),
            pd.Timestamp(active_phases[-1][2]),
        )
        if nifty_total is not None:
            nifty_cagr = (1 + nifty_total) ** (1 / phases_years) - 1 if phases_years > 0 else 0
            overall.add_row("NIFTY 50 Total Return", _cpct(nifty_total))
            overall.add_row("NIFTY 50 CAGR", _cpct(nifty_cagr))
            overall.add_row("Alpha vs NIFTY 50", _cpct(phases_return - nifty_total))

        midcap_total = _midcap_return(
            pd.Timestamp(active_phases[0][1]),
            pd.Timestamp(active_phases[-1][2]),
        )
        if midcap_total is not None:
            midcap_cagr = (1 + midcap_total) ** (1 / phases_years) - 1 if phases_years > 0 else 0
            overall.add_row("MIDCAP 100 Total Return", _cpct(midcap_total))
            overall.add_row("MIDCAP 100 CAGR", _cpct(midcap_cagr))
            overall.add_row("Alpha vs MIDCAP 100", _cpct(phases_return - midcap_total))

        console.print(overall)

        # Return contribution — based on phases P&L
        total_pnl = last_phase_end_val - phase1_value

        def _fmt_period(start_s: str, end_s: str) -> str:
            """Format date range as "Mar'15 → Aug'15"."""
            fmt = "%b'%y"
            s = datetime.strptime(start_s, "%Y-%m-%d").strftime(fmt)
            e = datetime.strptime(end_s, "%Y-%m-%d").strftime(fmt)
            return f"{s} → {e}"

        if total_pnl != 0:
            contrib = Table(
                title="Return Contribution by Phase",
                show_lines=False, title_style="bold cyan",
            )
            contrib.add_column("#", style="dim", width=3, justify="right")
            contrib.add_column("Phase", style="bold", no_wrap=True)
            contrib.add_column("Period", style="dim", no_wrap=True)
            contrib.add_column("Start ₹", justify="right", no_wrap=True)
            contrib.add_column("₹ P&L", justify="right", no_wrap=True)
            contrib.add_column("% Total", justify="right", no_wrap=True)

            for i, pr in enumerate(phase_results, 1):
                pnl = pr["end_val"] - pr["start_val"]
                pct = pnl / total_pnl
                pc = "green" if pnl >= 0 else "red"
                contrib.add_row(
                    str(i), pr["name"],
                    _fmt_period(pr["start_str"], pr["end_str"]),
                    f"₹{pr['start_val']:,.0f}",
                    f"[{pc}]₹{pnl:+,.0f}[/{pc}]",
                    f"[{pc}]{pct:+.1%}[/{pc}]",
                )
            console.print(contrib)

        # Key insights
        best = max(phase_results, key=lambda p: p["return"])
        worst = min(phase_results, key=lambda p: p["return"])
        alpha_phases = [p for p in phase_results if p["alpha"] is not None]
        best_alpha = max(alpha_phases, key=lambda p: p["alpha"]) if alpha_phases else None
        worst_alpha = min(alpha_phases, key=lambda p: p["alpha"]) if alpha_phases else None

        alpha_mc_phases = [p for p in phase_results if p.get("alpha_midcap") is not None]
        beats_midcap = [p for p in alpha_mc_phases if p["alpha_midcap"] > 0]

        bull = [p for p in phase_results if "Bull" in p["type"] or p["type"] == "Bullish"]
        bear = [p for p in phase_results if "Bear" in p["type"] or "Crash" in p["type"]]

        insights = Table(
            title="Key Insights", show_header=False, show_lines=False, title_style="bold cyan",
        )
        insights.add_column("", style="cyan", min_width=25)
        insights.add_column("", min_width=50)

        insights.add_row("Best Phase", f"{best['name']} ({best['return']:+.1%})")
        insights.add_row("Worst Phase", f"{worst['name']} ({worst['return']:+.1%})")
        if best_alpha:
            insights.add_row("Highest Alpha (N50)", f"{best_alpha['name']} ({best_alpha['alpha']:+.1%})")
        if worst_alpha:
            insights.add_row("Lowest Alpha (N50)", f"{worst_alpha['name']} ({worst_alpha['alpha']:+.1%})")
        if alpha_mc_phases:
            insights.add_row("Beats MIDCAP 100", f"{len(beats_midcap)}/{len(alpha_mc_phases)} phases")
        if bull:
            avg_bull = np.mean([p['return'] for p in bull])
            bull_mc = [p for p in bull if p.get("midcap_ret") is not None]
            bull_str = f"{avg_bull:+.1%} ({len(bull)} phases)"
            if bull_mc:
                avg_bull_mc = np.mean([p['midcap_ret'] for p in bull_mc])
                bull_str += f" | MIDCAP avg {avg_bull_mc:+.1%}"
            insights.add_row("Avg Bull Return", bull_str)
        if bear:
            avg_bear = np.mean([p['return'] for p in bear])
            bear_mc = [p for p in bear if p.get("midcap_ret") is not None]
            bear_str = f"{avg_bear:+.1%} ({len(bear)} phases)"
            if bear_mc:
                avg_bear_mc = np.mean([p['midcap_ret'] for p in bear_mc])
                bear_str += f" | MIDCAP avg {avg_bear_mc:+.1%}"
            insights.add_row("Avg Bear/Crash Return", bear_str)

        console.print(insights)
        console.print()

    def _do_select_strategy(self):
        """Select which strategy to use."""
        console.print(Panel("Strategy Selection", style="bright_blue"))

        # List available strategies
        strategies = StrategyRegistry.list_strategies()

        table = Table(title="Available Strategies")
        table.add_column("#", style="cyan", justify="right", width=3)
        table.add_column("Name", style="white")
        table.add_column("Description", style="dim")
        table.add_column("Active", justify="center")

        for i, (name, desc) in enumerate(strategies, 1):
            active = "✓" if name == self.active_strategy else ""
            active_style = "green" if name == self.active_strategy else ""
            table.add_row(
                str(i),
                name,
                desc,
                f"[{active_style}]{active}[/{active_style}]" if active else "",
            )

        console.print(table)

        # Prompt for selection
        choice = Prompt.ask(
            "\nSelect strategy number (or Enter to keep current)",
            default="",
        )

        if not choice:
            console.print(f"[dim]Keeping current strategy: {self.active_strategy}[/dim]")
            return

        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(strategies):
                new_strategy_name = strategies[choice_idx][0]
                self.active_strategy = new_strategy_name
                self._init_strategy()
                console.print(
                    f"[green]✓ Strategy changed to: {self.active_strategy}[/green]"
                )
            else:
                console.print("[red]Invalid selection[/red]")
        except ValueError:
            # Try to match by name
            for name, _ in strategies:
                if name.lower() == choice.lower():
                    self.active_strategy = name
                    self._init_strategy()
                    console.print(
                        f"[green]✓ Strategy changed to: {self.active_strategy}[/green]"
                    )
                    return
            console.print("[red]Invalid selection[/red]")


# CLI entry points
@click.command()
@click.option("--config", "-c", default="config.yaml", help="Config file path")
def main(config):
    """FORTRESS MOMENTUM - Interactive CLI"""
    app = FortressApp(config_path=config)
    app.run()


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Config file path")
@click.option("--dry-run/--live", default=True, help="Dry run mode")
@click.pass_context
def cli(ctx, config, dry_run):
    """FORTRESS MOMENTUM - Pure Momentum Strategy CLI"""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["dry_run"] = dry_run


@cli.command()
@click.pass_context
def login(ctx):
    """Authenticate with Zerodha."""
    app = FortressApp(ctx.obj["config_path"])
    app._load_config()
    app._load_universe()
    app._do_login()


@cli.command()
@click.pass_context
def status(ctx):
    """Show portfolio status."""
    app = FortressApp(ctx.obj["config_path"])
    app._load_config()
    app._load_universe()
    # Authenticate using cached token
    if app.config.zerodha.api_key:
        app.auth = ZerodhaAuth(app.config.zerodha.api_key, app.config.zerodha.api_secret)
        try:
            app.kite = app.auth.login_interactive()
            mapper = InstrumentMapper(app.kite, app.universe)
            mapper.load_instruments()
            app.market_data = MarketDataProvider(app.kite, mapper)
            app.portfolio = Portfolio(app.kite, app.universe)
        except Exception as e:
            console.print(f"[red]Authentication failed: {e}[/red]")
            return
    app._do_status()


@cli.command()
@click.pass_context
def scan(ctx):
    """Scan and rank stocks by momentum."""
    app = FortressApp(ctx.obj["config_path"])
    app._load_config()
    app._load_universe()
    # Authenticate using cached token
    if app.config.zerodha.api_key:
        app.auth = ZerodhaAuth(app.config.zerodha.api_key, app.config.zerodha.api_secret)
        try:
            app.kite = app.auth.login_interactive()
            mapper = InstrumentMapper(app.kite, app.universe)
            mapper.load_instruments()
            app.market_data = MarketDataProvider(app.kite, mapper)
            app.portfolio = Portfolio(app.kite, app.universe)
        except Exception as e:
            console.print(f"[red]Authentication failed: {e}[/red]")
            return
    app._do_scan()


@cli.command()
@click.pass_context
def backtest(ctx):
    """Run historical backtest."""
    app = FortressApp(ctx.obj["config_path"])
    app._load_config()
    app._load_universe()
    # Authenticate using cached token (needed if cache is insufficient)
    if app.config.zerodha.api_key:
        app.auth = ZerodhaAuth(app.config.zerodha.api_key, app.config.zerodha.api_secret)
        try:
            app.kite = app.auth.login_interactive()
            mapper = InstrumentMapper(app.kite, app.universe)
            mapper.load_instruments()
            app.market_data = MarketDataProvider(app.kite, mapper)
            app.portfolio = Portfolio(app.kite, app.universe)
        except Exception as e:
            console.print(f"[yellow]Warning: Authentication failed: {e}[/yellow]")
            console.print("[dim]Backtest will use cached data only.[/dim]")
    app._do_backtest()


@cli.command()
@click.option("--confirm", is_flag=True, help="Confirm live orders")
@click.pass_context
def rebalance(ctx, confirm):
    """Execute rebalance."""
    app = FortressApp(ctx.obj["config_path"])
    app._load_config()
    app._load_universe()
    # Authenticate using cached token
    if app.config.zerodha.api_key:
        app.auth = ZerodhaAuth(app.config.zerodha.api_key, app.config.zerodha.api_secret)
        try:
            app.kite = app.auth.login_interactive()
            mapper = InstrumentMapper(app.kite, app.universe)
            mapper.load_instruments()
            app.market_data = MarketDataProvider(app.kite, mapper)
            app.portfolio = Portfolio(app.kite, app.universe)
        except Exception as e:
            console.print(f"[red]Authentication failed: {e}[/red]")
            return
    dry_run = ctx.obj["dry_run"] or not confirm
    app._do_rebalance(dry_run=dry_run)


if __name__ == "__main__":
    main()
