# FORTRESS MOMENTUM

A momentum-based stock rotation system for Indian equities (NSE), built on top of the Zerodha Kite Connect API. Implements an adaptive dual momentum strategy with regime detection, dynamic rebalancing, and layered defensive scaling.

## How It Works

The system ranks all stocks in its universe by **Normalized Momentum Score (NMS)** -- a volatility-adjusted blend of 6-month and 12-month returns, inspired by the Nifty 500 Momentum 50 index methodology. It then applies entry filters (52-week high proximity, trend, volume, liquidity) and builds a concentrated portfolio of 10-15 high-momentum positions.

What makes it adaptive:

- **Regime Detection** -- A multi-timeframe stress model (VIX, breadth, returns) classifies the market as BULLISH, NORMAL, CAUTION, or DEFENSIVE. The portfolio's equity/gold/cash allocation shifts accordingly via a graduated curve rather than hard thresholds.
- **Dynamic Rebalancing** -- Instead of fixed-interval rebalancing, the system checks daily for trigger conditions (regime transitions, VIX spikes, drawdown breaches, breadth thrusts) and only rebalances when necessary. This reduces trade count while staying responsive.
- **Layered Defenses** -- Five independently toggleable layers protect during downturns: portfolio volatility targeting, breadth-based scaling, dynamic sector caps, sector momentum filtering, and crash avoidance. Each can be enabled/disabled in config.
- **Recovery Modes** -- When markets recover from drawdowns, the system detects improving breadth and relaxes its defensive posture faster, avoiding the common problem of staying too defensive during rebounds.

## Backtest Results

| Period | Return | CAGR | Sharpe | Max DD | vs NIFTY 50 |
|--------|--------|------|--------|--------|-------------|
| 2023-2024 (bull) | +75.25% | +32.46% | 1.44 | -11.56% | +45.3% |
| 6 months | +8.55% | +17.79% | 0.83 | -5.86% | +3.1% |
| 12 months | +7.78% | +7.81% | 0.19 | -8.85% | -4.6% |
| 18 months (incl. correction) | -2.60% | -1.75% | -0.53 | -18.37% | -9.1% |

The 18-month period includes the Oct 2024 - Mar 2025 market correction, where the strategy went defensive (50%+ cash) and underperformed. The 2023-2024 bull run demonstrates the strategy's alpha generation in favorable conditions.

### 10-Year Market Phases (2015-2026)

A continuous backtest across 12 distinct market phases, from the 2015 bull run through the late-2024 correction.

**Overall: +1,216.8% return | 24.2% CAGR | 1.16 Sharpe | -25.8% Max DD | +568% alpha vs NIFTY 50**

| # | Phase | Type | Return | NIFTY | Alpha | Max DD |
|---|-------|------|--------|-------|-------|--------|
| 1 | 2015 Bull Run | Bullish | -11.2% | -12.8% | +1.6% | -13.6% |
| 2 | China Scare & Recovery | Bear/Recovery | +2.4% | -8.4% | +10.8% | -6.1% |
| 3 | Pre-Demonetization Bull | Bullish | +17.6% | +15.9% | +1.7% | -6.1% |
| 4 | Demonetization Shock | Bear/Recovery | +2.3% | +8.8% | -6.5% | -8.7% |
| 5 | 2017 Bull Run | Bullish | +19.3% | +20.5% | -1.2% | -7.8% |
| 6 | NBFC / IL&FS Crisis | Bearish | -15.9% | -1.7% | -14.2% | -21.2% |
| 7 | 2019 Recovery | Sideways/Bull | +6.0% | +11.2% | -5.2% | -10.6% |
| 8 | COVID Crash | Crash | -11.3% | -32.2% | +20.9% | -14.4% |
| 9 | Post-COVID Rally | Bullish | +294.8% | +128.5% | +166.3% | -5.6% |
| 10 | 2022 Correction | Bearish | +0.8% | -17.0% | +17.8% | -12.1% |
| 11 | 2023-24 Bull Run | Bullish | +108.6% | +70.5% | +38.1% | -17.5% |
| 12 | Late 2024-25 Correction | Bear/Sideways | +1.7% | -1.9% | +3.6% | -15.7% |

Key takeaways:
- **Bull phases** average +72.5% return (6 phases) -- captures upside aggressively
- **Bear/crash phases** average -3.3% return (6 phases) -- strong downside protection
- **COVID crash**: -11.3% vs NIFTY -32.2% (+20.9% alpha) -- defensive scaling worked
- **Post-COVID rally**: +294.8% with only -5.6% max DD -- recovery mode captured the rebound
- Capital grew from ~31L to ~2.63 Cr over the full period (20L initial + 12-month warmup)

### Smallcap Profile (NIFTY Smallcap 100)

Separate portfolio with wider stops, higher volatility target, and no gold allocation. Initial capital: 5L.

| Period | Return | CAGR | Sharpe | Max DD | vs NIFTY 50 |
|--------|--------|------|--------|--------|-------------|
| 3 months | -1.67% | -6.54% | -0.70 | -7.28% | -2.6% |
| 6 months | +6.64% | +13.69% | 0.57 | -6.84% | +1.1% |
| 12 months | +1.83% | +1.84% | -0.19 | -14.82% | -10.6% |
| 18 months | +1.63% | +1.08% | -0.19 | -21.79% | -4.9% |
| 2023-2024 (bull) | +191.70% | +70.98% | 2.38 | -19.43% | +161.8% |

The smallcap universe data begins mid-2022, so the market phases backtest only covers 2 phases:

| # | Phase | Type | Return | NIFTY | Alpha | Max DD |
|---|-------|------|--------|-------|-------|--------|
| 1 | 2023-24 Bull Run | Bullish | +166.8% | +70.5% | +96.3% | -18.9% |
| 2 | Late 2024-25 Correction | Bear/Sideways | -8.3% | -1.9% | -6.4% | -21.5% |

**Overall: +143.2% | 27.8% CAGR | -21.5% Max DD | +78.2% alpha vs NIFTY 50** (5L grew to ~12.2L)

Smallcaps show higher beta -- explosive in bull markets (+191.7% in 2023-2024) but deeper drawdowns during corrections (-21.8% vs -18.4% primary in the 18M period).

## Architecture

```
fortress/
  cli.py                  Interactive CLI (login, scan, rebalance, backtest)
  backtest.py             Backtesting engine with pre-computed caches (60x speedup)
  momentum_engine.py      Live-mode stock ranking, filtering, weight calculation
  indicators.py           NMS, regime detection, rebalance triggers, breadth
  config.py               All configuration dataclasses (Pydantic)
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
  universe.py             Stock universe loader with sub-universe filtering
  auth.py                 Zerodha authentication (TOTP + request token)
  utils.py                Weight renormalization, rate limiting
tests/                    160 tests covering indicators, backtest, strategies, risk
```

**Parity guarantee**: `backtest.py` and `momentum_engine.py` implement the same strategy logic in parallel. Backtests use pre-computed caches for speed; live mode fetches from the API. Both produce equivalent results.

~18,000 lines of Python. ~2,400 lines of tests.

## Key Features

### Multi-Profile Support
Run separate portfolios with different universes and parameters:
- **Primary**: NIFTY 100 + MIDCAP 100 (200 stocks, 12 target positions)
- **Smallcap**: NIFTY Smallcap 100 (100 stocks, 13 targets, wider stops)

Each profile has its own state file, initial capital, position sizing, and optional strategy overrides (volatility target, stop multipliers, exit thresholds).

### Defensive Enhancements
All toggleable via config:
- **Volatility targeting** -- Scales equity exposure down when portfolio vol exceeds target (default 15%)
- **Breadth scaling** -- Reduces exposure when fewer stocks are above their 50-day MA
- **Dynamic sector caps** -- Tighter sector limits in CAUTION/DEFENSIVE regimes (30%/25%/20%)
- **Sector momentum filter** -- Soft penalty (not hard exclude) for bottom sectors by momentum
- **Crash avoidance** -- Early-warning system detects slow grinds (1M <= -5% AND 3M <= -8%)
- **Gold exhaustion scaling** -- Reduces gold allocation when gold is overextended above its 200-SMA
- **Trend guard** -- When NIFTY > 200-SMA, limits equity cuts to max 20% (prevents over-de-risking in uptrends)

### Tiered Stop Loss System
Stops adapt based on unrealized gain:
| Gain Tier | Trailing Stop |
|-----------|--------------|
| < 8% | 18% initial stop |
| 8-20% | 15% trailing |
| 20-50% | 15% trailing |
| > 50% | 25% trailing (let winners run) |

Plus trend-break exits, relative strength floor exits, and a minimum hold period (3 days) to avoid whipsaws.

### Live Trading Integration
- Zerodha Kite Connect API for order placement
- Dry-run mode for testing without real orders
- Post-execution reconciliation (handles failed buys/sells gracefully)
- Self-funding rebalances (sells fund buys within same session)

## Setup

### Prerequisites
- Python 3.10+
- Zerodha Kite Connect API credentials ([apply here](https://kite.trade))

### Installation

```bash
git clone https://github.com/javajack/stock-rotation.git
cd stock-rotation

# Option 1: Use the startup script (creates venv, installs deps)
./start.sh

# Option 2: Manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your Zerodha API credentials
```

Or use environment variables:
```bash
export ZERODHA_API_KEY="your_api_key"
export ZERODHA_API_SECRET="your_api_secret"
```

### Running

```bash
# Interactive CLI
./start.sh
# or
.venv/bin/python -m fortress

# Run tests
.venv/bin/python -m pytest tests/ -v
```

The CLI presents a menu:
1. **Login** -- Authenticate with Zerodha
2. **Status** -- View current portfolio
3. **Scan** -- Rank stocks by momentum score
4. **Rebalance** -- Generate and execute trade orders
5. **Backtest** -- Run historical simulation (3M to 48M)
6. **Strategy** -- Select active strategy
7. **Triggers** -- Check if rebalance conditions are met
9. **Market Phases** -- 10-year multi-phase backtest analysis

## Configuration Reference

The system is highly configurable. Key sections in `config.yaml`:

| Section | Controls |
|---------|----------|
| `pure_momentum` | NMS lookbacks, entry filters, percentile thresholds |
| `position_sizing` | Target/min/max positions, sector caps, weighting method |
| `risk` | Stop losses, drawdown limits, position limits |
| `regime` | Stress thresholds, VIX levels, allocation curve, defensive scaling |
| `dynamic_rebalance` | Trigger conditions, min/max days between rebalances |
| `profiles` | Per-profile universes, capital, overrides |

See `config.example.yaml` for all available options with documentation.

## Research Basis

The strategy draws from:
- **Dual Momentum** (Gary Antonacci) -- Combining absolute and relative momentum for +440 bps annually vs index
- **Nifty 500 Momentum 50** -- NSE's own momentum index methodology for stock selection
- **Volatility Targeting** (Moskowitz et al.) -- Reduces max drawdown by ~6.6%, can double Sharpe ratio
- **Regime Detection** -- Multi-factor stress scoring for adaptive allocation

## License

MIT
