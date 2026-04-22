# FORTRESS MOMENTUM

## Testing Invariant
- All integration/smoke testing MUST go through `./start.sh` and the interactive CLI menu (options 1-9)
- Do NOT write or run ad-hoc Python scripts for manual testing
- Use `.venv/bin/python -m pytest tests/ -v` for unit tests only
- Backtest validation: use CLI Option 5 (Backtest) or Option 8 (Market Phases)

## Key Rules
- Never change strategy logic without running a full phase backtest (Option 8) before and after
- **Current baseline** (survivorship-free, nse-universe data, 30-day cadence,
  real NIFTY 50 benchmark, `rank_range: [201, 600]` with NSE-authoritative
  sector classification + ETF filter + live-parity scaling via
  `rebalance_logic.py`, run 2026-04-23 on f_custom):
  **+22.1% CAGR, 0.90 Sharpe, −41.0% MaxDD, +1272% total** (2013-01-01 →
  2026-02-11, 16 phases). vs real NIFTY 50 CAGR 11.83% over same period —
  strategy beats by **+10.3 pp/yr**. ₹20L end value: strategy ₹2.73 Cr,
  NIFTY 50 buy-and-hold ₹86L.
  Earlier baseline +21.3% / 0.87 / −36.1% (pre-parity) was under-deploying
  capital because backtest silently dropped unaffordable buys due to tx
  costs. Post-parity adopts the same proportional-scaling logic as the live
  planner — higher CAGR because no trade silently skipped, wider DD because
  the book now holds more marginal positions through crashes. The +22.1%
  figure is closer to what live execution would actually produce.
  Prior inflated baseline of +26.8% CAGR was ETF-contamination artifact
  (pre-ETF-cleanup commit 87cb165); ignore for comparison.
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
  has ~400 candidates to pick 15 from, no cap-tier concentration. Other tested
  bands: `[1, 200]` large-cap (14.3% CAGR), `[101, 250]` mid (18.7%), `[251, 500]`
  small (20.6%), `[201, 1000]` too-wide (22.1% — deep tail dilutes).
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
