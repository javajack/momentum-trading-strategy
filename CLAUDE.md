# FORTRESS MOMENTUM

## Testing Invariant
- All integration/smoke testing MUST go through `./start.sh` and the interactive CLI menu (options 1-8)
- Do NOT write or run ad-hoc Python scripts for manual testing
- Use `.venv/bin/python -m pytest tests/ -v` for unit tests only
- Backtest validation: use CLI Option 5 (Backtest) or Option 8 (Market Phases)

## Key Rules
- Never change strategy logic without running a full phase backtest (Option 8) before and after
- Definitive backtest baseline: 19.8% CAGR, 0.87 Sharpe, -27.3% MaxDD (run 16 Feb 2026, data through 12 Feb 2026)
- `config.yaml` is gitignored — users copy from `config.example.yaml`
- Only one strategy exists: `dual_momentum` — no profiles, no multi-strategy

## Architecture
- **Universe**: NIFTY 100 + MIDCAP 100 (200 stocks), defined in `stock-universe.json`
- **Capital model**: LIQUIDBEES is the single source/sink of capital. No demat cash dependency.
- **Backtest parity**: `backtest.py` and `momentum_engine.py` implement identical strategy logic
- **Config**: Strategy runs on hardcoded defaults in `_get_config_values()`. YAML `strategy_dual_momentum:` section can override.

## Run Commands
- **CLI**: `.venv/bin/python -m fortress` or `./start.sh`
- **Tests**: `.venv/bin/python -m pytest tests/ -v` (159 tests)
- **Single test**: `.venv/bin/python -m pytest tests/test_backtest.py -v -k test_name`
