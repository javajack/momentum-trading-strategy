# FORTRESS MOMENTUM

A momentum-based stock rotation system for Indian equities (NSE). It picks high-momentum stocks, adapts to market conditions, and manages risk automatically -- so you can stay invested without constantly watching the market.

Built on top of the Zerodha Kite Connect API. Works with NIFTY 100 + MIDCAP 100 (200 stocks).

## What Does It Do?

Every few weeks, the system looks at all 200 stocks and asks: *which ones have the strongest momentum right now, and is it safe to be fully invested?*

It ranks stocks by **Normalized Momentum Score (NMS)** -- a volatility-adjusted blend of 6-month and 12-month returns (inspired by the Nifty 500 Momentum 50 index). It then filters for quality (52-week high proximity, trend, volume) and builds a concentrated portfolio of 10-12 high-momentum positions.

The magic is in how it adapts:

- **Reads the room** -- A stress model combining VIX, market breadth, and returns classifies conditions as BULLISH, NORMAL, CAUTION, or DEFENSIVE. The equity/gold/cash mix shifts smoothly along a graduated curve -- no sudden all-in or all-out moves.
- **Rebalances when it matters** -- Instead of fixed schedules, it watches for triggers (regime shifts, VIX spikes, drawdowns, breadth thrusts) and acts only when needed. Fewer trades, lower costs, better timing.
- **Protects the downside** -- Five independent defense layers (volatility targeting, breadth scaling, dynamic sector caps, sector momentum filter, crash avoidance) work together to reduce exposure in rough markets.
- **Catches the recovery** -- When markets turn around, the system detects improving breadth and ramps back to full equity faster, avoiding the classic trap of staying defensive too long.

## Does It Work?

### 13-Year Multi-Phase Backtest (Jan 2013 -- Feb 2026)

> Backtest run: 16 Feb 2026. Data: NSE daily OHLCV through 12 Feb 2026. Continuous backtest with compounding -- capital carries forward across phases (starts at 20L).

A continuous backtest across 16 distinct market phases -- bull runs, bear markets, crashes, and everything in between.

**Overall: 19.8% CAGR | 0.87 Sharpe | -27.3% Max DD | 20L grew to 2.14 Cr**

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

All capital flows through **LIQUIDBEES** (a liquid ETF earning ~6-7% annualized):

- **Add money**: Buy LIQUIDBEES in your demat --> next rebalance deploys it into stocks
- **Withdraw**: Sell some LIQUIDBEES --> that capital stays out
- **Surplus**: After filling all stock positions, leftover cash sweeps into LIQUIDBEES
- **No idle cash**: Your money is always working, even when parked

This means you never need to worry about "cash drag" -- uninvested capital earns returns in LIQUIDBEES while waiting to be deployed.

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
- **Self-funding rebalances** -- Sells generate cash for buys within the same session
- **Post-execution reconciliation** -- Handles failed orders gracefully
- **Daily trigger checks** -- Only rebalance when the system detects a reason to

## Architecture

```
fortress/
  cli.py                  Interactive CLI (login, scan, rebalance, backtest)
  backtest.py             Backtesting engine with vectorized breadth + asof lookups
  momentum_engine.py      Live-mode stock ranking, filtering, weight calculation
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
tests/                    160 tests covering indicators, backtest, strategies, risk
```

**Parity guarantee**: `backtest.py` and `momentum_engine.py` implement the same strategy logic. Backtests use pre-computed caches for speed; live mode fetches from the API. Both produce equivalent results -- what you backtest is what you trade.

~18,500 lines of Python. ~2,400 lines of tests.

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
| **"Rebalancing frequency mismatch"** | Not fixed-schedule. The system checks daily for triggers (regime transitions, VIX spikes, drawdown breaches, breadth thrusts) and only rebalances when there's a reason. Min 7 days between rebalances to avoid whipsaw, max 15 days to stay responsive. | `indicators.py` dynamic triggers |
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
