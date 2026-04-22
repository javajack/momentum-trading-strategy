# FORTRESS MOMENTUM

## Testing Invariant
- All integration/smoke testing MUST go through `./start.sh` and the interactive CLI menu (options 1-9)
- Do NOT write or run ad-hoc Python scripts for manual testing
- Use `.venv/bin/python -m pytest tests/ -v` for unit tests only
- Backtest validation: use CLI Option 5 (Backtest) or Option 8 (Market Phases)

## Key Rules
- Never change strategy logic without running a full phase backtest (Option 8) before and after
- **Current baseline** (survivorship-free, nse-universe data, 30-day cadence,
  real NIFTY 50 benchmark, `rank_range: [201, 600]`, live-parity scaling,
  **thicker defensive buffer** (max_gold 0.15 + max_cash 0.30 in stress),
  run 2026-04-23 on f_custom):
  **+24.1% CAGR, 1.02 Sharpe, −28.1% MaxDD, +1603% total** (2013-01-01 →
  2026-02-11, 16 phases). vs real NIFTY 50 CAGR 11.83% over same period —
  strategy beats by **+12.3 pp/yr**. ₹20L end value: strategy ₹3.40 Cr,
  NIFTY 50 buy-and-hold ₹86L.
  Earlier baseline +22.1% / 0.90 / −41.0% improved via DD-opt sweep: of 8
  tested levers (threshold tightening, bigger defensive buffer, tighter
  sector caps, position-count reduction, stricter entry filters, DD
  circuit breaker, tighter stops, aggressive crash scale), **only bigger
  defensive buffer** survived empirical validation. Every other lever
  either forced premature exit at local bottoms, killed compounding, or
  concentrated risk — all reverted. Lesson: the strategy benefits from
  MORE defensive allocation at stress peak, not from firing defensive
  mode earlier or more aggressively.
  `rank_range: [1, 200]` large-cap comparison: +14.3% CAGR / 0.55 Sharpe —
  post-ETF-cleanup numbers still strongly favour the [201, 600] mid-small tilt.
  Pre-refactor survivorship-biased baseline (19.8% CAGR / 0.87 Sharpe /
  −27.3% MaxDD) ran today's 200-stock winners against historical prices —
  classic "picked the survivors" trap, ignore for honest comparison.
- `config.yaml` is tracked (see config.example.yaml for the reference template).
- Only one strategy exists: `dual_momentum` — no profiles, no multi-strategy
- Orders are not placed by code (Zerodha IP-whitelist policy, April 2026).
  Option 4 (Rebalance) and Option 9 (Exit All) write a Kite basket CSV to
  `plans/`; humans execute it on the Kite dashboard. No `--live` / `--confirm`
  flags anywhere any more.

## Architecture
- **Universe**: point-in-time from nse-universe. Default `rank_range: [201, 600]`
  picked by sweep — small/mid-cap spread by 6mo median turnover. Filter-pipeline
  has ~400 candidates to pick 15 from, no cap-tier concentration.
  100-rank slice sweep (2026-04-23, with current strategy incl. DD-opt #2 and
  live-parity scaling) confirms [201, 600] is optimal and **beats every
  narrower sub-slice** on all three metrics (CAGR / Sharpe / MaxDD):
    `[1, 100]`   large-cap:   8.9% / 0.26 / −35.4% (momentum-free zone)
    `[101, 200]` upper-mid:  12.5% / 0.47 / −40.9% (crowded, correlated)
    `[201, 300]` mid-small:  19.5% / 0.86 / −27.7% (shallowest DD of any slice)
    `[301, 400]` small top:  20.5% / 0.91 / −34.2% (best single-slice CAGR)
    `[401, 500]` small mid:  21.5% / 0.96 / −34.4% (best single-slice Sharpe)
    `[501, 600]` small tail: 17.6% / 0.79 / −32.1% (deep-tail noise on its own)
    `[201, 500]` (drop tail): 22.3% / 0.95 / −34.8% — worse DD than full band
    **`[201, 600]` combined: 24.1% / 1.02 / −28.1%**
  Why combined beats any slice: (a) 400-symbol pool gives the 85th-percentile
  filter more to pick from, higher picks on average, (b) cap-tier leadership
  rotates across phases (mid-cap-heavy '14-'17, deep-small '20-'21, both
  '23-'24), (c) [501, 600] is noisy alone but adds uncorrelated diversification
  when combined with [201, 400] core.
  Also falsified as too-wide: `[201, 750]` (−42.3% DD), `[201, 1000]` (deep-tail
  dilution, CAGR 22.1%). Also falsified as too-narrow: `[1, 200]` large-cap
  (14.3% CAGR).
- **Sector classification**: `stock-sectors.json` built offline by
  `tools/build_sectors.py` — covers all 4,166 NSE EQ symbols. Primary source:
  `nse_universe.sectors` (NSE authoritative, 754 symbols, 100% of the
  `[201, 600]` window). Fallback: hand-curated LLM map + heuristic rules for
  deep-tail / delisted names NSE's sectoral CSVs don't cover.
- **Market metadata**: `market-metadata.json` carries benchmark / sectoral
  indices / VIX / hedges (GOLDBEES, LIQUIDBEES).
- **Capital model**: LIQUIDBEES is the ONLY gateway for capital in/out of
  the strategy. User deposits → manually buys LIQUIDBEES on Kite → strategy
  SELL_EXITs LIQUIDBEES on next rebalance → deploys into stocks. Demat cash
  sitting outside LIQUIDBEES is invisible — strategy never auto-sweeps it.
  Stock-to-stock rotation is funded by sell proceeds (stocks + LIQUIDBEES
  exits); if buys exceed that budget, planner scales buys proportionally
  and warns. To reduce exposure, user manually sells LIQUIDBEES after a
  rebalance surplus-sweep.
- **Data sources**: Kite historical for live signals (adjusted, 12-month cap),
  nse-universe parquet for backtest (20-year coverage, split-adjusted via
  `compute_adj_factor`). Set in `fortress/nse_data_loader.py` and `fortress/cache.py`.
- **Unmanaged holdings**: Portfolio splits holdings into managed (in universe
  or registered hedge) vs external (stray ETFs, non-universe stocks user owns).
  Rebalance / exit-all ignore external.

## Known caveats
- **Sectoral-index momentum filter (E5)** is no-op in backtest: sectoral
  indices (NIFTY BANK, NIFTY IT, etc.) aren't in the parquet. Filter silently
  returns empty. Live mode uses Kite historical for these indices normally.
  A synthetic (equal-weighted) injection was tried and proved too noisy —
  regressed CAGR by 2.5pp. The function lives in nse_data_loader.py for
  future use with proper cap-weighted reconstruction.
- **Filter toggles available but default off**: `falling_knife_6m_cutoff`
  and `require_above_12m_sma` in strategy_dual_momentum config. Both were
  measured to regress CAGR in the 13-year backtest (over-constrain the
  strategy). Opt in only for specific regime experiments.

## Run Commands
- **CLI**: `.venv/bin/python -m fortress` or `./start.sh`
- **Tests**: `.venv/bin/python -m pytest tests/ -q` (222 tests)
- **Single test**: `.venv/bin/python -m pytest tests/test_backtest.py -v -k test_name`
- **Rebuild sector map** (quarterly or when adding ranked IPOs): `.venv/bin/python tools/build_sectors.py`
