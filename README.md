# FORTRESS MOMENTUM

**22.3% CAGR over 13 years. 20L grew to 2.81 Cr. Fully automated.**

A momentum-based stock rotation system for Indian equities (NSE). It picks high-momentum stocks from NIFTY 100 + MIDCAP 100 (200 stocks), adapts to market conditions in real-time, and manages risk through five independent defense layers -- so you stay invested without constantly watching the market.

Built on the Zerodha Kite Connect API. ~18,000 lines of Python. 201 tests.

---

## Why This Exists

Most momentum strategies work great in backtests and fall apart in practice. They go all-in during bull markets, get destroyed in crashes, and churn through positions during sideways markets.

FORTRESS MOMENTUM solves this with three ideas:

1. **Regime-aware allocation** -- A stress model reads VIX, breadth, and returns to classify the market as BULLISH, NORMAL, CAUTION, or DEFENSIVE. Equity allocation shifts smoothly from 95% down to 60% along a curve -- no sudden all-in or all-out flips.

2. **Event-driven rebalancing** -- Instead of fixed schedules, 7 triggers (regime shifts, VIX recovery, drawdowns, crashes, breadth thrusts, portfolio momentum deterioration, and regular intervals) fire only when there's a reason to act. Fewer trades, lower costs, better timing.

3. **Self-funding capital model** -- All capital flows through LIQUIDBEES (a liquid ETF). Sells fund buys. Surplus parks in LIQUIDBEES. The system never asks for external cash during a rebalance.

---

## Performance

### 13-Year Multi-Phase Backtest (Jan 2013 -- Feb 2026)

> Run: 16 Feb 2026. Data: NSE daily OHLCV through 12 Feb 2026. Continuous backtest with compounding -- capital carries forward across all phases starting at 20L.

**Overall: 22.3% CAGR | 1.02 Sharpe | -20.8% Max Drawdown | 20L -> 2.81 Cr**

Benchmarks over the same period: Nifty 50 B&H 11.8% CAGR | Nifty Midcap 100 B&H 16.0% CAGR.

| # | Phase | Period | Type | Return | NIFTY 50 | Alpha | Max DD |
|---|-------|--------|------|--------|----------|-------|--------|
| 1 | 2013 Consolidation | Jan 13 -- May 13 | Sideways | -10.0% | +1.7% | -11.7% | -12.1% |
| 2 | Taper Tantrum | May 13 -- Aug 13 | Bearish | -2.4% | -11.4% | +9.0% | -8.0% |
| 3 | Pre-Election Rally | Aug 13 -- May 14 | Bullish | +36.0% | +33.1% | +2.9% | -4.5% |
| 4 | Modi Bull Run | May 14 -- Mar 15 | Bullish | +52.3% | +23.5% | +28.8% | -7.8% |
| 5 | 2015 Correction | Mar 15 -- Aug 15 | Bearish | -10.8% | -12.5% | +1.7% | -15.6% |
| 6 | China Scare | Aug 15 -- Mar 16 | Bear/Recovery | -3.9% | -8.4% | +4.4% | -8.2% |
| 7 | Pre-Demonetization | Mar 16 -- Nov 16 | Bullish | +21.8% | +15.9% | +5.9% | -6.2% |
| 8 | Demonetization Shock | Nov 16 -- Apr 17 | Bear/Recovery | +0.3% | +8.8% | -8.5% | -8.7% |
| 9 | 2017 Bull Run | Apr 17 -- Jan 18 | Bullish | +29.8% | +20.5% | +9.3% | -5.3% |
| 10 | NBFC/IL&FS Crisis | Jan 18 -- Mar 19 | Bearish | -12.0% | -1.7% | -10.3% | -22.3% |
| 11 | 2019 Recovery | Mar 19 -- Jan 20 | Sideways | +14.0% | +11.3% | +2.7% | -11.0% |
| 12 | COVID Crash | Jan 20 -- Apr 20 | Crash | -18.7% | -32.0% | +13.3% | -20.5% |
| 13 | Post-COVID Rally | Apr 20 -- Oct 21 | Bullish | +231.5% | +128.6% | +102.9% | -6.2% |
| 14 | 2022 Correction | Oct 21 -- Jun 22 | Bearish | -16.7% | -17.0% | +0.2% | -17.5% |
| 15 | 2023-24 Bull Run | Jun 22 -- Sep 24 | Bullish | +107.2% | +70.6% | +36.6% | -22.2% |
| 16 | Late 2024-25 | Sep 24 -- Feb 26 | Bear/Sideways | -1.7% | +0.6% | -2.2% | -16.2% |

Positive alpha in **10 of 16 phases**. Highlights:

- **Post-COVID rally** (Phase 13): +231.5% vs NIFTY +128.6% -- momentum at its best, +103pp alpha
- **Modi bull run** (Phase 4): +52.3% vs NIFTY +23.5% -- concentrated bets in trending stocks
- **COVID crash** (Phase 12): -18.7% vs NIFTY -32.0% -- defensive scaling saved ~13% of capital
- **2023-24 bull run** (Phase 15): +107.2% vs NIFTY +70.6% -- sustained alpha over 2+ years

> **Honest note**: The strategy underperforms in early consolidation phases (Phase 1) and prolonged bear markets where large-caps hold up but midcaps don't (Phase 10). Momentum needs trends. The edge shows up over full market cycles, not every quarter.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Zerodha trading account with [Kite Connect API](https://kite.trade) subscription
- 15-20L capital recommended (works with less, but position sizing gets tight below 10L)

### Setup

```bash
git clone https://github.com/javajack/stock-rotation.git
cd stock-rotation

# Credentials (gitignored)
cat > .env << 'EOF'
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
EOF

# Config (copy and adjust initial_capital, target_positions)
cp config.example.yaml config.yaml

# Run
./start.sh
```

`start.sh` creates a virtual environment, installs dependencies, loads credentials, and launches the CLI. Subsequent runs go straight to the menu.

### First Session

```
1  Login       Authenticate with Zerodha (cached for the day)
2  Status      View current positions and P&L
3  Scan        Rank stocks by momentum
4  Rebalance   Generate trades (dry-run first, then live)
5  Backtest    Run historical simulations
6  Strategy    Select active strategy
7  Triggers    Check if rebalance is needed today
8  Phases      Run the full 16-phase backtest
```

**Recommended first steps:**

1. **Login** (1) -- Authenticate with Zerodha
2. **Scan** (3) -- See which stocks the system likes right now
3. **Backtest** (5) -- Run a 12-month backtest to verify everything works
4. **Triggers** (7) -- Check if today warrants a rebalance
5. **Rebalance** (4) -- Dry-run first to see the plan, then go live

---

## How Capital Works

All capital flows through **LIQUIDBEES** (a liquid ETF earning ~6-7% annualized). It is the single entry and exit point -- the system never touches your demat cash for rebalancing.

**To start**: Buy LIQUIDBEES worth your target capital through your broker's normal order flow. The first rebalance detects it, sells it, and deploys proceeds into high-momentum stocks + GOLDBEES.

**To add capital later**: Buy more LIQUIDBEES. The next rebalance picks it up automatically.

**To withdraw**: Sell some LIQUIDBEES. That capital leaves the strategy immediately.

### The Self-Funding Cycle

Every rebalance is fully self-funded from sell proceeds:

```
SELL exits & reduces  ──►  Proceeds fund BUY orders
                             │
                             ├── Any surplus deploys to:
                             │     1. Underweight equity (pro-rata top-up)
                             │     2. Underweight GOLDBEES
                             │     3. LIQUIDBEES (sweep remainder)
                             │
                             └── Result: zero external cash needed
```

If buys exceed sell proceeds, buy quantities scale down automatically. The system **never asks for additional cash** during a rebalance. Idle demat cash (from dividends, past sells) is detected separately and swept into LIQUIDBEES as a "capital injection" -- shown separately from the rebalance.

---

## How Rebalancing Works

The system checks daily for trigger conditions and only acts when there's a reason to.

### 7 Rebalance Triggers

| Trigger | Fires When | Urgency |
|---------|-----------|---------|
| Regime transition | Market shifts (e.g., NORMAL -> CAUTION) | HIGH |
| VIX recovery | VIX drops 15%+ from spike above 25 | HIGH |
| Portfolio drawdown | Portfolio down 10%+ from peak | HIGH |
| Market crash | Nifty 1-month return below -7% | HIGH |
| Breadth thrust | Breadth surges from <40% to >61.5% in 10 days | HIGH |
| Portfolio momentum | Portfolio 20-day return below -7% | MEDIUM |
| Regular interval | 15+ trading days since last rebalance | MEDIUM |

**Guardrails**: Minimum 7 days between rebalances (prevents whipsaw). Maximum 15 days (forces periodic refresh, configurable up to 30).

### What Happens During a Rebalance

A typical rebalance involves 3-8 trades:

1. **Exits** (0-3 stocks) -- Positions below stops, broken trend, or lost relative strength
2. **New entries** (0-3 stocks) -- Highest-NMS stocks that passed all filters
3. **Adjustments** (0-3 trades) -- Reduce overweight, increase underweight
4. **Sweep** (0-1 trade) -- Leftover proceeds -> LIQUIDBEES

### What to Expect

- **Calm bull markets**: Rebalance every ~15 days. Few changes, low turnover.
- **Volatile periods**: Rebalance every 7-10 days. Regime shifts and drawdowns trigger earlier action.
- **Crashes**: Multiple triggers fire simultaneously. System acts quickly but respects the 7-day minimum.

---

## Operations Manual

### Daily Routine (2 minutes)

Most days, nothing to do. The system runs every 1-3 weeks, not daily.

If you want to check in:

```
./start.sh  ->  1 (Login)  ->  7 (Triggers)
```

- **"No triggers"** -- Close and come back tomorrow.
- **Trigger fired** -- Proceed to rebalance (see below).

### When a Rebalance Fires

```
./start.sh
1  Login
7  Triggers          (confirm trigger)
4  Rebalance
   Mode 1: DRY RUN  (always do this first)
```

Review the dry-run:
- **SELL orders** -- Do the exits make sense? (stops hit, trend broke, RS fell)
- **BUY orders** -- High-momentum stocks you're comfortable holding?
- **Cash Flow** -- Should say "Fully self-funded"

If the plan looks good:

```
4  Rebalance
   Mode 2: LIVE
   Confirm twice
```

Sells execute first, then buys. Failed orders are handled gracefully -- failed buys are removed from tracking, failed sells are kept.

### Weekly Schedule

| When | Action | Menu |
|------|--------|------|
| Any weekday | Login + check triggers | 1 -> 7 |
| If triggered | Dry-run -> review -> live | 1 -> 4 (mode 1, then 2) |
| Monthly | Run 3-month backtest to verify system health | 1 -> 5 |
| Quarterly | Run full 16-phase backtest | 1 -> 8 |

### Capital Management

| Action | How | Effect |
|--------|-----|--------|
| **Add capital** | Buy LIQUIDBEES through your broker | Next rebalance auto-deploys |
| **Withdraw** | Sell LIQUIDBEES through your broker | Capital leaves immediately |
| **Check status** | Menu option 2 (Status) | Shows managed vs external holdings |

No config edits needed. The system discovers LIQUIDBEES in your broker positions automatically.

### What NOT to Do

- **Don't manually trade managed stocks.** The system tracks positions and will get confused if holdings change outside its knowledge.
- **Don't rebalance more than once per week** unless a HIGH urgency trigger fires.
- **Don't panic-sell during drawdowns.** Five defense layers reduce exposure automatically when conditions warrant it.
- **Don't override the dry-run.** If the plan looks wrong, don't execute. Investigate first.

---

## How the Strategy Works

### Stock Selection: Normalized Momentum Score (NMS)

Every stock gets ranked by NMS -- a volatility-adjusted blend of 6-month and 12-month returns (inspired by the Nifty 500 Momentum 50 index). Stocks are then filtered for:

- **52-week high proximity** -- Must be within a threshold of its high
- **Trend** -- Price above key moving averages
- **Volume** -- Sufficient liquidity for entry/exit
- **Sector caps** -- No more than 20-30% in any one sector (tightens in defensive regimes)
- **Sector momentum** -- 15% score penalty for bottom sectors (soft filter, not hard block)

The top 10-12 survivors become the portfolio.

### Regime Detection

A composite stress score (VIX 30% + breadth 40% + returns 30%) classifies the market:

| Regime | Stress Score | Equity | Gold | Behavior |
|--------|-------------|--------|------|----------|
| BULLISH | Low (< 0.25) | 90-95% | 5-10% | Full momentum, tight stops |
| NORMAL | Moderate | 80-90% | 10-15% | Standard allocation |
| CAUTION | Elevated | 60-80% | 15-20% | Reduced exposure, wider stops |
| DEFENSIVE | High (> 0.65) | 40-60% | 20%+ | Capital preservation |

Transitions use 3-day confirmation (hysteresis) to prevent whipsaw. The allocation curve is smooth -- no cliff effects.

### Five Defense Layers

| Layer | What It Does | Trigger |
|-------|-------------|---------|
| **Volatility targeting** | Scales down equity when portfolio vol exceeds 15% | Realized vol > target |
| **Breadth scaling** | Reduces exposure when fewer stocks above 50-day MA | Breadth below 50% |
| **Dynamic sector caps** | Tighter limits in bad markets (30% / 25% / 20%) | Regime shift |
| **Sector momentum filter** | 15% score penalty for bottom sectors | Every rebalance |
| **Crash avoidance** | Early warning at 1M <= -5% AND 3M <= -8% | Market returns |

Plus: gold exhaustion scaling (reduces gold allocation when Nifty far above 200-SMA), trend guard (prevents over-de-risking in uptrends), and recovery override (faster return to full equity when breadth improves).

### Tiered Stop Loss System

Stops adapt to how much a position has gained:

| Gain Level | Trailing Stop | Philosophy |
|------------|--------------|------------|
| < 8% | 12% from peak | Give new positions room |
| 8-20% | 14% from peak | Protect early gains |
| 20-50% | 16% from peak | Standard protection |
| > 50% | 22% from peak | Let big winners run |

**Hard stop**: 15% from entry price (always active, catches positions that never gain).

Additional exits: trend-break detection, relative strength floor, and 3-day minimum hold period. Recovery mode widens stops by 25-50% to avoid shaking out positions during bounces.

---

## Architecture

```
fortress/
  cli.py                  Interactive CLI (login, scan, rebalance, backtest)
  backtest.py             Backtesting engine (vectorized breadth, asof lookups)
  momentum_engine.py      Live-mode stock ranking, filtering, weight calculation
  defensive.py            Shared defensive logic (gold, vol, breadth, sector caps)
  indicators.py           NMS, regime detection, rebalance triggers, breadth
  config.py               Configuration dataclasses (Pydantic)
  strategy/
    adaptive_dual_momentum.py   Tiered stops, trend breaks, RS exits, recovery modes
    base.py                     Strategy interface
    registry.py                 Pluggable strategy registry
  portfolio.py            Portfolio tracking for backtests
  rebalance_executor.py   Trade plan builder (self-funding cycle)
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
tests/                    201 tests across 10 files
```

**Parity guarantee**: `backtest.py` and `momentum_engine.py` share all defensive logic via `defensive.py` -- gold allocation, vol targeting, breadth scaling, and sector caps are implemented once as pure functions. What you backtest is what you trade.

## Configuration

All parameters live in `config.yaml`. Key sections:

| Section | Controls |
|---------|----------|
| `portfolio` | Initial capital, universe file |
| `pure_momentum` | NMS lookbacks, entry filters, percentile thresholds |
| `position_sizing` | Target/min/max positions, sector caps, weighting |
| `risk` | Stop losses, drawdown limits, position limits |
| `regime` | Stress thresholds, VIX levels, allocation curve |
| `strategy_dual_momentum` | Stops, recovery, crash avoidance, breadth smoothing |
| `dynamic_rebalance` | Trigger conditions, min/max days between rebalances |

See `config.example.yaml` for all options with inline documentation.

## Research Basis

- **Dual Momentum** (Gary Antonacci) -- Combining absolute and relative momentum
- **Nifty 500 Momentum 50** -- NSE's own momentum index methodology, adapted for stock selection
- **Volatility Targeting** (Moskowitz et al.) -- Reduces max drawdown, can double Sharpe ratio
- **Regime Detection** -- Multi-factor stress scoring for adaptive allocation

## FAQ

| Concern | How It's Handled |
|---------|-----------------|
| **"Momentum dies in bear markets"** | 4-regime system shifts allocation from 95% equity down to 60%. Graduated curve, not binary. |
| **"No exit logic"** | Three-layer exits: tiered trailing stops (12-22%), trend-break detection, relative strength floor. Plus 3-day min hold. |
| **"All top stocks = same sector"** | Dynamic sector caps (30% / 25% / 20% by regime) + 15% score penalty for bottom sectors. |
| **"Rebalancing too often / too rarely"** | 7 event-driven triggers, not fixed schedule. Min 7 / max 15 days between rebalances. |
| **"Transaction costs eat the edge"** | 0.3% cost modeled in backtests. Min hold period (3d) and min rebalance interval (7d) reduce churn. |
| **"No drawdown protection"** | Portfolio drawdown >10% triggers defensive regime. Crash avoidance at -5%/-8%. Vol targeting at 15%. Three independent circuit breakers. |
| **"Backtests are too optimistic"** | Transaction costs included, T-1 data (no lookahead), identical logic in backtest and live engines. |
| **"How do I add/remove capital?"** | Buy/sell LIQUIDBEES through your broker. System detects it automatically. See [How Capital Works](#how-capital-works). |

## Disclaimer

This is a personal project shared for educational purposes. It is **not financial advice**.

- Past backtest performance does not guarantee future results
- Momentum strategies can underperform in choppy/sideways markets
- Always do your own research before deploying real capital
- Start with dry-run mode and small amounts until you're comfortable

The author uses this system with real capital, but your risk tolerance may differ.

## License

MIT
