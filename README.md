# FORTRESS MOMENTUM

A momentum-based stock rotation system for Indian equities (NSE). It picks high-momentum stocks, adapts to market conditions, and manages risk automatically -- so you can stay invested without constantly watching the market.

Built on top of the Zerodha Kite Connect API. Works with NIFTY 100 + MIDCAP 100 (200 stocks).

## What Does It Do?

Every few weeks, the system looks at all 200 stocks and asks: *which ones have the strongest momentum right now, and is it safe to be fully invested?*

It ranks stocks by **Normalized Momentum Score (NMS)** -- a volatility-adjusted blend of 6-month and 12-month returns (inspired by the Nifty 500 Momentum 50 index). It then filters for quality (52-week high proximity, trend, volume) and builds a concentrated portfolio of 10-12 high-momentum positions.

The magic is in how it adapts:

- **Reads the room** -- A stress model combining VIX, market breadth, and returns classifies conditions as BULLISH, NORMAL, CAUTION, or DEFENSIVE. The equity/gold/cash mix shifts smoothly along a graduated curve -- no sudden all-in or all-out moves.
- **Rebalances when it matters** -- Instead of fixed schedules, 7 event-driven triggers (regime shifts, VIX recovery, drawdowns, crashes, breadth thrusts, portfolio momentum) fire only when needed. Fewer trades, lower costs, better timing.
- **Protects the downside** -- Five independent defense layers (volatility targeting, breadth scaling, dynamic sector caps, sector momentum filter, crash avoidance) work together to reduce exposure in rough markets.
- **Catches the recovery** -- When markets turn around, the system detects improving breadth and ramps back to full equity faster, avoiding the classic trap of staying defensive too long.

## Does It Work?

### 13-Year Multi-Phase Backtest (Jan 2013 -- Feb 2026)

> Backtest run: 16 Feb 2026. Data: NSE daily OHLCV through 12 Feb 2026. Continuous backtest with compounding -- capital carries forward across phases (starts at 20L).

A continuous backtest across 16 distinct market phases -- bull runs, bear markets, crashes, and everything in between.

**Overall: 22.3% CAGR | 1.02 Sharpe | -20.8% Max DD | 20L grew to 2.81 Cr**

Nifty 50 B&H: 11.8% CAGR | Nifty Midcap 100 B&H: 16.0% CAGR over the same period.

| # | Phase | Period | Type | Return | NIFTY 50 | Alpha | Max DD |
|---|-------|--------|------|--------|----------|-------|--------|
| 1 | 2013 Consolidation | Jan 2013 -- May 2013 | Sideways | -10.0% | +1.7% | -11.7% | -12.1% |
| 2 | Taper Tantrum & Rupee Crisis | May 2013 -- Aug 2013 | Bearish | -2.4% | -11.4% | +9.0% | -8.0% |
| 3 | Pre-Election Rally | Aug 2013 -- May 2014 | Bullish | +36.0% | +33.1% | +2.9% | -4.5% |
| 4 | Modi Election Bull Run | May 2014 -- Mar 2015 | Bullish | +52.3% | +23.5% | +28.8% | -7.8% |
| 5 | 2015 Correction | Mar 2015 -- Aug 2015 | Bearish | -10.8% | -12.5% | +1.7% | -15.6% |
| 6 | China Scare & Recovery | Aug 2015 -- Mar 2016 | Bear/Recovery | -3.9% | -8.4% | +4.4% | -8.2% |
| 7 | Pre-Demonetization Bull | Mar 2016 -- Nov 2016 | Bullish | +21.8% | +15.9% | +5.9% | -6.2% |
| 8 | Demonetization Shock & Recovery | Nov 2016 -- Apr 2017 | Bear/Recovery | +0.3% | +8.8% | -8.5% | -8.7% |
| 9 | 2017 Bull Run | Apr 2017 -- Jan 2018 | Bullish | +29.8% | +20.5% | +9.3% | -5.3% |
| 10 | NBFC / IL&FS Crisis | Jan 2018 -- Mar 2019 | Bearish | -12.0% | -1.7% | -10.3% | -22.3% |
| 11 | 2019 Recovery (Corp Tax Cut) | Mar 2019 -- Jan 2020 | Sideways/Bullish | +14.0% | +11.3% | +2.7% | -11.0% |
| 12 | COVID Crash | Jan 2020 -- Apr 2020 | Crash | -18.7% | -32.0% | +13.3% | -20.5% |
| 13 | Post-COVID Rally | Apr 2020 -- Oct 2021 | Bullish | +231.5% | +128.6% | +102.9% | -6.2% |
| 14 | 2022 Correction (Ukraine/Rates) | Oct 2021 -- Jun 2022 | Bearish | -16.7% | -17.0% | +0.2% | -17.5% |
| 15 | 2023-24 Recovery & Bull Run | Jun 2022 -- Sep 2024 | Bullish | +107.2% | +70.6% | +36.6% | -22.2% |
| 16 | Late 2024-25 Correction | Sep 2024 -- Feb 2026 | Bear/Sideways | -1.7% | +0.6% | -2.2% | -16.2% |

Positive alpha vs Nifty 50 in **10 of 16 phases**. A few highlights:

- **Post-COVID rally** (Phase 13): +231.5% vs NIFTY +128.6% -- momentum selection at its best, +102.9pp alpha
- **Modi bull run** (Phase 4): +52.3% vs NIFTY +23.5% -- concentrated bets in trending stocks
- **COVID crash** (Phase 12): -18.7% vs NIFTY -32.0% -- defensive scaling saved ~13% of capital
- **2023-24 bull run** (Phase 15): +107.2% vs NIFTY +70.6% -- sustained alpha over 2+ years
- **NBFC crisis** (Phase 10): The weakest phase -- strategy lost -12.0% while NIFTY lost only -1.7%

> **Honest note**: The strategy underperforms in early consolidation phases (Phase 1) and prolonged bear markets where large-caps hold up but midcaps don't (Phase 10). Momentum needs trends to work. The edge shows up over full cycles, not every quarter.

## Getting Started

### What You Need

- Python 3.10+
- A Zerodha trading account with [Kite Connect API](https://kite.trade) subscription
- ~15-20L capital recommended (works with less, but position sizing gets tight below 10L)

### Step 1: Clone

```bash
git clone https://github.com/javajack/stock-rotation.git
cd stock-rotation
```

### Step 2: Set Up Credentials

Create a `.env` file with your Zerodha API credentials:

```bash
cat > .env << 'EOF'
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
EOF
```

This file is gitignored -- your credentials stay local.

### Step 3: Configure

```bash
cp config.example.yaml config.yaml
```

Open `config.yaml` and adjust to your setup. The key settings to change:

- `portfolio.initial_capital` -- Your starting capital (default: 20L)
- `position_sizing.target_positions` -- How many stocks to hold (default: 12)

Everything else has sensible defaults. You can always tune later.

### Step 4: Run

```bash
./start.sh
```

On first run, this creates a virtual environment, installs dependencies, loads your credentials, and launches the CLI. Subsequent runs go straight to the menu.

**Or manually:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
source .env
python -m fortress
```

### Step 5: Your First Session

The CLI menu:

```
1. Login          -- Authenticate with Zerodha (cached for the day)
2. Status         -- See your current positions and P&L
3. Scan           -- View top momentum stocks right now
4. Rebalance      -- Generate trades (dry-run first, then live)
5. Backtest       -- Run historical simulations
6. Strategy       -- Select active strategy
7. Triggers       -- Check if rebalance is needed today
8. Market Phases  -- Run the full multi-phase backtest
```

**Recommended first steps:**

1. **Login** (1) -- Authenticate with Zerodha
2. **Scan** (3) -- See which stocks the system likes right now
3. **Backtest** (5) -- Run a 12-month backtest to verify everything works
4. **Triggers** (7) -- Check if today is a good day to start
5. **Rebalance** (4) -- Run in **dry-run mode** first to see the plan, then go live

> **Tip**: The system uses LIQUIDBEES (liquid ETF) as its capital pool. To get started, buy LIQUIDBEES worth your target capital through your broker. The next rebalance will automatically convert it into equity positions.

## How Capital Works

All capital flows through **LIQUIDBEES** (a liquid ETF earning ~6-7% annualized). LIQUIDBEES is the single entry and exit point for capital -- the system never touches your demat cash directly.

### Funding the System

1. **To start**: Buy LIQUIDBEES worth your target capital (e.g., 15-20L) through your broker's normal order flow
2. **The first rebalance** detects LIQUIDBEES in your holdings, sells it, and deploys the proceeds into high-momentum stocks + GOLDBEES
3. **To add capital later**: Buy more LIQUIDBEES -- the next rebalance picks it up automatically
4. **To withdraw**: Sell some LIQUIDBEES -- that capital leaves the strategy

You never need to "tell" the system about capital changes. It discovers LIQUIDBEES in your broker positions and acts accordingly.

### The Self-Funding Rebalance Cycle

Every rebalance is **fully self-funded** -- sells generate the cash for buys:

```
SELL: Exit positions not in target, reduce overweight positions
  │
  ├── Proceeds fund BUY orders (new positions + increases)
  │
  ├── Any surplus deploys to:
  │     1. Underweight equity positions (pro-rata top-up)
  │     2. Underweight GOLDBEES (if below target)
  │     3. LIQUIDBEES (sweep remainder)
  │
  └── Result: ₹0 external cash needed
```

The system will **never ask you for additional cash** during a rebalance. If buys exceed sell proceeds, buy quantities are automatically scaled down to fit. Your capital stays fully allocated at all times -- working in equities, gold, or LIQUIDBEES.

If you have idle demat cash (from dividends, past sells, etc.), the system detects it and sweeps it into LIQUIDBEES as a separate "capital injection" -- but this is shown separately from the rebalance and doesn't affect the self-funding guarantee.

## How Rebalancing Works

The system doesn't rebalance on a fixed schedule. Instead, it checks daily for **trigger conditions** and only acts when there's a reason to.

### Rebalance Triggers

| Trigger | Condition | Urgency | Purpose |
|---------|-----------|---------|---------|
| Regular interval | 15+ days since last rebalance | Medium | Baseline refresh |
| Regime transition | Market regime changed (e.g., NORMAL → CAUTION) | High | Adapt allocation |
| VIX recovery | VIX drops 15%+ from recent spike above 25 | High | Capture recovery |
| Portfolio drawdown | Portfolio down 10%+ from peak | High | Defensive rebalance |
| Market crash | Market 1-month return below -7% | High | Crash avoidance |
| Breadth thrust | Breadth surges from <40% to >61.5% in 10 days | High | Aggressive re-entry |
| Portfolio momentum | Portfolio 20-day return below -7% | Medium | Early reshuffling |

**Guardrails**: Minimum 7 days between rebalances (prevents whipsaw). Maximum 15 days (forces periodic refresh).

### Recommended Rebalance Frequency

In practice, the system triggers a rebalance every **7-15 trading days** (roughly 1-3 weeks). Here's what to expect:

- **Calm bull markets**: Rebalance every ~15 days (max interval). Few changes, low turnover.
- **Volatile periods**: Rebalance every 7-10 days. Regime shifts, VIX spikes, and drawdowns trigger earlier action.
- **Crashes**: Multiple triggers fire simultaneously (crash + drawdown + regime). The system acts quickly but respects the 7-day minimum to avoid panic-selling.

**What you need to do**: Run option 7 (Triggers) periodically to check if a rebalance is needed. When it says yes, run option 4 (Rebalance) in dry-run mode first, review the plan, then execute.

### What a Rebalance Actually Does

A typical rebalance involves 3-8 trades:

1. **Exits** (0-3 stocks): Positions that fell below stops, broke trend, or lost relative strength
2. **New entries** (0-3 stocks): High-NMS stocks that passed all filters
3. **Adjustments** (0-3 trades): Reductions of overweight positions, increases of underweight ones
4. **Sweep** (0-1 trade): Leftover proceeds → LIQUIDBEES

Average annual turnover is ~1700-1800 trades across both directions. Transaction costs are modeled at 0.3% in backtests.

## Day-to-Day Operations Manual

Once you've funded the system and run your first rebalance, here's the routine.

### Daily (2 minutes)

Nothing. Seriously. The system is designed to run every 1-3 weeks, not daily. You don't need to watch it.

If you want to check in:

```
./start.sh
1  → Login
7  → Triggers
```

If Triggers says "No triggers" -- close it and come back tomorrow. If it says a trigger fired (regime change, drawdown, etc.) -- proceed to rebalance.

### When a Rebalance Is Triggered (10-15 minutes)

```
./start.sh
1  → Login (authenticate with Zerodha)
7  → Triggers (confirm rebalance is needed)
4  → Rebalance
     → Mode 1: DRY RUN (always do this first)
```

**Review the dry-run output carefully:**

- Check the SELL orders -- do the exits make sense? (stops hit, trend broke, RS fell)
- Check the BUY orders -- are these high-momentum stocks you're comfortable holding?
- Check "Cash Flow" -- it should say "Fully self-funded"
- If anything looks wrong, just close. No harm done.

**If the plan looks good:**

```
4  → Rebalance
     → Mode 2: LIVE
     → Confirm twice (the system double-checks before placing orders)
```

Sells execute first (R9 invariant), then buys. You'll see order-by-order progress. Failed orders are handled gracefully -- failed buys are removed from tracking, failed sells are kept.

### Weekly Routine (Recommended)

Even if no triggers fire, glance at the system once a week:

| Day | Action | Menu Option |
|-----|--------|-------------|
| Any weekday | Login + check triggers | 1 → 7 |
| If triggered | Dry-run → review → live | 1 → 4 (mode 1, then 2) |
| Monthly (optional) | Run 3-month backtest to verify system health | 1 → 5 → duration: 3 |
| Quarterly (optional) | Run full market phases backtest | 1 → 8 |

### Adding or Withdrawing Capital

| Action | How | When It Takes Effect |
|--------|-----|---------------------|
| **Add capital** | Buy LIQUIDBEES through your broker (normal order, outside this system) | Next rebalance auto-deploys it |
| **Withdraw** | Sell some LIQUIDBEES through your broker | Capital leaves the strategy immediately |
| **Check status** | Menu option 2 (Status) | Shows managed vs external holdings |

You never need to edit config files or tell the system about capital changes. It discovers LIQUIDBEES in your broker positions automatically.

### What NOT to Do

- **Don't manually buy/sell stocks** that the system manages. It tracks positions and will get confused if holdings change outside its knowledge.
- **Don't rebalance more than once per week** unless a HIGH urgency trigger fires. The 7-day minimum exists for a reason.
- **Don't panic-sell during drawdowns**. The system has 5 independent defense layers and will reduce exposure automatically if conditions warrant it.
- **Don't override the dry-run**. If the plan looks wrong, don't execute. Investigate first (run a backtest, check regime, review triggers).

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "Not authenticated" | Zerodha session expired | Login again (option 1). Sessions last one trading day. |
| Stale regime warning | Haven't rebalanced in 30+ days | Run a rebalance. Regime detection needs recent data. |
| Few/no stocks in scan | Market in DEFENSIVE regime | Normal. System is protecting capital. Check regime in Triggers (option 7). |
| Managed capital = 0 | No LIQUIDBEES or managed positions | Buy LIQUIDBEES to seed the system. |
| "Scaling buys to X%" | Sells didn't generate enough proceeds | Normal. Buys are scaled down to fit. No action needed. |

## Key Features

### Adaptive Regime Detection

The system classifies market conditions into four regimes:

| Regime | Equity | Gold | Behavior |
|--------|--------|------|----------|
| BULLISH | 90-95% | 5-10% | Full momentum, tight stops |
| NORMAL | 80-90% | 10-15% | Standard allocation |
| CAUTION | 60-80% | 15-20% | Reduced exposure, wider stops |
| DEFENSIVE | 40-60% | 20%+ | Capital preservation mode |

Transitions happen smoothly along a graduated curve -- no sudden all-in or all-out flips.

### Layered Defenses

Five independent protection layers, all toggleable in config:

1. **Volatility targeting** -- Scales down equity when portfolio vol exceeds 15%
2. **Breadth scaling** -- Reduces exposure when fewer stocks are above their 50-day MA
3. **Dynamic sector caps** -- Tighter sector limits in CAUTION/DEFENSIVE (30%/25%/20%)
4. **Sector momentum filter** -- 15% score penalty for bottom sectors (soft filter, not hard block)
5. **Crash avoidance** -- Early-warning when 1M return <= -5% AND 3M return <= -8%

Plus: gold exhaustion scaling, trend guard (prevents over-de-risking in uptrends), and recovery equity override (faster return to full equity when breadth improves).

### Tiered Stop Loss System

Stops adapt to how much a position has gained:

| Unrealized Gain | Trailing Stop | Philosophy |
|-----------------|--------------|------------|
| < 8% | 18% initial stop | Give new positions room |
| 8-20% | 15% trailing | Protect early gains |
| 20-50% | 15% trailing | Standard protection |
| > 50% | 25% trailing | Let winners run |

Additional exits: trend-break detection, relative strength floor, and a 3-day minimum hold period to prevent whipsaws.

### Live Trading

- **Dry-run mode** -- Preview every trade before committing
- **Self-funding rebalances** -- Sells always generate enough cash for buys. No external capital needed. See [How Capital Works](#how-capital-works)
- **LIQUIDBEES capital pool** -- All capital enters/exits through LIQUIDBEES. Idle cash automatically sweeps in
- **Post-execution reconciliation** -- Handles failed orders gracefully (failed buys removed from tracking, failed sells kept)
- **7 event-driven triggers** -- Only rebalance when the system detects a reason to. See [How Rebalancing Works](#how-rebalancing-works)

## Architecture

```
fortress/
  cli.py                  Interactive CLI (login, scan, rebalance, backtest)
  backtest.py             Backtesting engine with vectorized breadth + asof lookups
  momentum_engine.py      Live-mode stock ranking, filtering, weight calculation
  defensive.py            Shared defensive logic (gold, vol, breadth, sector caps)
  indicators.py           NMS, regime detection, rebalance triggers, breadth
  config.py               Configuration dataclasses (Pydantic)
  strategy/
    adaptive_dual_momentum.py   Main strategy: tiered stops, trend breaks, RS exits
    base.py                     Strategy interface
    registry.py                 Pluggable strategy registry
  portfolio.py            Portfolio tracking for backtests
  rebalance_executor.py   Trade plan generation and order execution
  order_manager.py        Zerodha order placement (dry-run + live)
  risk_governor.py        Position/sector limits, stop loss tracking
  market_data.py          Zerodha historical data provider
  instruments.py          NSE instrument mapping
  cache.py                Parquet-based data cache with incremental updates
  universe.py             Stock universe loader (NIFTY 100 + MIDCAP 100)
  auth.py                 Zerodha authentication (TOTP + request token)
  utils.py                Weight renormalization, rate limiting
tools/
  reconcile_state.py      Reconcile strategy state with live broker holdings
config.example.yaml       Default configuration (copy to config.yaml)
stock-universe.json       200 stocks: NIFTY 100 + MIDCAP 100, with sector mappings
tests/                    201 tests covering indicators, backtest, defensive, strategies, rebalance, risk
```

**Parity guarantee**: `backtest.py` and `momentum_engine.py` share defensive logic via `defensive.py` -- gold allocation, vol targeting, breadth scaling, and sector caps are implemented once as pure functions. Backtests use pre-computed caches for speed; live mode fetches from the API. Both produce equivalent results -- what you backtest is what you trade.

## Configuration Reference

All strategy parameters live in `config.yaml`. Key sections:

| Section | What It Controls |
|---------|-----------------|
| `portfolio` | Initial capital, universe file |
| `pure_momentum` | NMS lookbacks, entry filters, percentile thresholds |
| `position_sizing` | Target/min/max positions, sector caps, weighting |
| `risk` | Stop losses, drawdown limits, position limits |
| `regime` | Stress thresholds, VIX levels, allocation curve, defensive scaling |
| `strategy_dual_momentum` | Stops, recovery detection, crash avoidance, breadth smoothing |
| `dynamic_rebalance` | Trigger conditions, min/max days between rebalances |

See `config.example.yaml` for all available options with inline documentation.

## Research Basis

The strategy draws from well-studied academic and practitioner research:

- **Dual Momentum** (Gary Antonacci) -- Combining absolute and relative momentum for +440 bps annually vs index
- **Nifty 500 Momentum 50** -- NSE's own momentum index methodology, adapted for stock selection
- **Volatility Targeting** (Moskowitz et al.) -- Reduces max drawdown by ~6.6%, can double Sharpe ratio
- **Regime Detection** -- Multi-factor stress scoring for adaptive allocation

## FAQ: "But What About...?"

Momentum strategies have well-known failure modes. Here's how this system addresses each one:

| Concern | How It's Handled | Where |
|---------|-----------------|-------|
| **"Momentum dies in bear markets"** | 4-regime system (BULLISH → NORMAL → CAUTION → DEFENSIVE) shifts allocation from 95% equity down to 40%. Not a binary switch -- a graduated curve. | `indicators.py` regime detection |
| **"How do you decide if it's safe to invest?"** | A multi-factor stress score combining VIX level, market breadth (% stocks above 50-DMA), and 1M/3M returns. No gut feel, no manual override. | `indicators.py` stress model |
| **"No exit logic = silent killer"** | Three-layer exit stack: (1) tiered trailing stops that adapt to gain size, (2) trend-break exits when price breaks below key MAs, (3) relative strength floor exits when a stock falls below RS threshold. Plus a 3-day minimum hold to prevent whipsaws. | `strategy/adaptive_dual_momentum.py` |
| **"All top stocks = same sector = hidden leverage"** | Dynamic sector caps: 30% in BULLISH, 25% in CAUTION, 20% in DEFENSIVE. Plus a soft sector momentum penalty (15% score reduction for bottom sectors) that discourages piling into one theme. | `momentum_engine.py` E4/E5 |
| **"Rebalancing frequency mismatch"** | Not fixed-schedule. 7 trigger types (regime shifts, VIX recovery, drawdowns, crashes, breadth thrusts, portfolio momentum) fire only when needed. Min 7 / max 15 days between rebalances. See [How Rebalancing Works](#how-rebalancing-works). | `indicators.py` dynamic triggers |
| **"Transaction costs eat the edge"** | Configurable transaction cost (default 0.3%) applied in backtests. Minimum hold period (3 days) and minimum days between rebalances (7) reduce churn. Turnover is visible in backtest output. | `config.py`, `backtest.py` |
| **"No drawdown protection"** | Portfolio drawdown > 10% triggers defensive regime. Crash avoidance activates at -5%/-8% (1M/3M). Vol targeting scales down when portfolio volatility exceeds 15%. Three independent circuit breakers. | E2, E3, E6 |
| **"Backtests are too optimistic"** | Backtests include transaction costs, use T-1 data (no lookahead), and the backtest engine runs identical logic to the live engine (parity guarantee). What you backtest is what you trade. | `backtest.py` |

## Disclaimer

This is a personal project shared for educational purposes. It is **not financial advice**.

- Past backtest performance does not guarantee future results
- Momentum strategies can and do underperform, especially in choppy/sideways markets
- Always do your own research before deploying real capital
- Start with dry-run mode and small amounts until you're comfortable with the system

The author uses this system with real capital, but your risk tolerance and circumstances may differ.

## License

MIT
