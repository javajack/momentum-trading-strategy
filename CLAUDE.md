# FORTRESS MOMENTUM

## Testing Invariant
- All integration/smoke testing MUST go through `./start.sh` and the interactive CLI menu (options 1-9)
- Do NOT write or run ad-hoc Python scripts for manual testing
- Use `.venv/bin/python -m pytest tests/ -v` for unit tests only
- Backtest validation: use CLI Option 5 (Backtest) or Option 8 (Market Phases)

## Key Rules
- Never change strategy logic without running a full phase backtest (Option 8) before and after
- **Current baseline** (survivorship-free, nse-universe data, 30-day cadence,
  2013-01-01 → 2026-02-11, re-baselined after rebalance-interval sweep
  2026-04-21): **+9.7% CAGR, 0.30 Sharpe, −33.4% MaxDD, +238.4% total.**
  Improvement over the 15-day baseline came mostly from MaxDD reduction
  (−45.6% → −33.4%); lower turnover dampens whipsaw in crisis phases.
  Old pre-refactor baseline (19.8% CAGR / 0.87 Sharpe / −27.3% MaxDD) was
  survivorship-biased — it ran today's 200-stock winners against historical
  prices, which is the classic "picked the survivors" trap.
- `config.yaml` is tracked (see config.example.yaml for the reference template).
- Only one strategy exists: `dual_momentum` — no profiles, no multi-strategy
- Orders are not placed by code (Zerodha IP-whitelist policy, April 2026).
  Option 4 (Rebalance) and Option 9 (Exit All) write a Kite basket CSV to
  `plans/`; humans execute it on the Kite dashboard. No `--live` / `--confirm`
  flags anywhere any more.

## Architecture
- **Universe**: point-in-time from nse-universe. Default `rank_range: (1, 200)` =
  top-200 by 6-month median turnover (nifty_200 equivalent). Configurable
  via `universe.rank_range` in `config.yaml` — use `(101, 250)` for mid-cap-focused.
- **Sector classification**: `stock-sectors.json` built offline by
  `tools/build_sectors.py` — covers all 4,166 NSE EQ symbols. 100% of current
  top-200 classified, 97% of historical top-200 union.
- **Market metadata**: `market-metadata.json` carries benchmark / sectoral
  indices / VIX / hedges (GOLDBEES/LIQUIDBEES/LIQUIDCASE).
- **Capital model**: LIQUIDBEES is the single source/sink of strategy capital.
  No demat cash dependency (Kite RMS/margins endpoint is blocked for
  non-whitelisted IPs post-April 2026; we handle this with a graceful degrade).
- **Data sources**: Kite historical for live signals (adjusted, 12-month cap),
  nse-universe parquet for backtest (20-year coverage, split-adjusted via
  `compute_adj_factor`). Set in `fortress/nse_data_loader.py` and `fortress/cache.py`.
- **Unmanaged holdings**: Portfolio splits holdings into managed (in universe
  or registered hedge) vs external (stray ETFs, non-universe stocks user owns).
  Rebalance / exit-all ignore external.

## Known caveats
- **Benchmark alpha is understated**: the backtest uses NIFTYBEES as a proxy
  for NIFTY 50 in relative-strength calculations. NIFTYBEES raw bhavcopy
  misses dividend payouts (~1.5% p.a.), so the printed "NIFTY 50 CAGR" and
  "Alpha vs NIFTY" columns are cosmetic. Strategy's own CAGR is unaffected.
- **Sectoral-index momentum filter (E5)** is no-op in backtest: sectoral
  indices (NIFTY BANK, NIFTY IT, etc.) aren't in the parquet. Filter silently
  returns empty. Live mode uses Kite historical for these indices normally.

## Run Commands
- **CLI**: `.venv/bin/python -m fortress` or `./start.sh`
- **Tests**: `.venv/bin/python -m pytest tests/ -q` (222 tests)
- **Single test**: `.venv/bin/python -m pytest tests/test_backtest.py -v -k test_name`
- **Rebuild sector map** (quarterly or when adding ranked IPOs): `.venv/bin/python tools/build_sectors.py`
