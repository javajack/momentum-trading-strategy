# FORTRESS MOMENTUM

## Testing Invariant
- All integration/smoke testing MUST go through `./start.sh` and the interactive CLI menu (options 1-9)
- Do NOT write or run ad-hoc Python scripts for manual testing
- Use `.venv/bin/python -m pytest tests/ -v` for unit tests only
- Backtest validation: use CLI Option 5 (Backtest) or Option 8 (Market Phases)

## Key Rules
- Never change strategy logic without running a full phase backtest (Option 8) before and after
- **Current baseline** (survivorship-free, nse-universe data, 30-day cadence,
  real NIFTY 50 benchmark, run 2026-04-22 on f_custom):
  **+14.3% CAGR, 0.55 Sharpe, −32.5% MaxDD, +476.7% total** (2013-01-01 →
  2026-02-11, 16 phases). vs real NIFTY 50 CAGR 11.83% over same period —
  strategy beats by **+2.47 pp/yr**. ₹20L end value: strategy ₹1.15 Cr,
  NIFTY 50 buy-and-hold ₹86L.
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
