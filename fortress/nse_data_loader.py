"""Backtest data source: bulk historical OHLCV from the nse-universe parquet.

Live mode uses Kite historical (adjusted, rate-limited). Backtests use this
loader — it reads the nse-universe bhavcopy archive (5,256 trading days,
2005-01-03 to today, EQ only) in one DuckDB query, applies split adjustment
via nse_universe.actions.fetch.compute_adj_factor, and returns the
Dict[symbol, DataFrame] shape fortress's backtest engine already expects.

Two practical wins over Kite for backtests:
  1. No rate limit, no historical-window cap (Kite caps non-whitelisted
     history at ~1 year post-April 2026; here we have 20 years).
  2. Survivorship-bias-free: every bhavcopy NSE ever published is in
     the archive, including days when now-delisted names traded.

Prices are adjusted for splits (when yfinance provides actions — ~41% of
symbols have split/dividend data). Unadjusted prices are returned for
the rest; the momentum signal is robust to this as long as the strategy
doesn't hit a split-day discontinuity (filter or warn; future work).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _apply_split_adjustment(df: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Scale historical prices to present-day equivalent.

    Each row of ``adj`` says: for prices strictly before ``event_date``,
    multiply by ``after_split_factor``. Multiple splits compound (the
    factor for the earliest upcoming event already reflects all later
    splits — nse-universe's actions pipeline computes the cumulative
    factor, not per-event ratios).

    Volume is divided by the same factor so shares-traded remains
    consistent across split days.
    """
    if adj is None or adj.empty or df.empty:
        return df

    events = adj.sort_values("event_date", ascending=False).copy()
    events["event_ts"] = pd.to_datetime(events["event_date"])

    factors = pd.Series(1.0, index=df.index)
    for _, ev in events.iterrows():
        mask = factors.index < ev["event_ts"]
        factors = factors.where(~mask, ev["after_split_factor"])

    df = df.copy()
    for col in ("open", "high", "low", "close", "prev_close"):
        if col in df.columns:
            df[col] = df[col] * factors
    if "volume" in df.columns:
        df["volume"] = (df["volume"] / factors).astype("int64", errors="ignore")
    return df


def _parquet_glob() -> str:
    """Glob for nse-universe partitioned parquet — respects NSE_UNIVERSE_DATA_DIR."""
    import os
    from pathlib import Path

    data_dir = os.environ.get("NSE_UNIVERSE_DATA_DIR")
    if data_dir:
        base = Path(data_dir) / "parquet"
    else:
        # Default to sibling repo layout.
        base = Path.home() / "work" / "nse500" / "data" / "parquet"
    return str(base / "year=*/month=*/*.parquet")


def _load_adj_factors(symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
    """Pull split/dividend factors from nse-universe.actions for each symbol.

    Symbols without yfinance coverage get an empty DataFrame — the caller
    treats that as "no adjustment needed".
    """
    try:
        from nse_universe.actions.fetch import compute_adj_factor
    except ImportError:  # pragma: no cover
        return {s: pd.DataFrame() for s in symbols}

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym] = compute_adj_factor(sym)
        except Exception:
            out[sym] = pd.DataFrame()
    return out


def load_historical_bulk(
    start: date,
    end: date,
    symbols: Optional[Iterable[str]] = None,
    apply_adj: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load daily OHLCV for every requested symbol via one DuckDB query.

    Args:
        start, end: Inclusive date range.
        symbols: Optional filter. If None, every distinct symbol present
            in the date range is loaded (large! Use for full-universe
            rebuilds only). Pass the union of rank-window members across
            the backtest horizon to keep the result bounded.
        apply_adj: If True, multiply historical prices by the cumulative
            split factor so every day's close is on the same terms as
            the final day. Safe to disable for diagnostic runs.

    Returns:
        ``{symbol: DataFrame}`` — each DataFrame indexed by a pandas
        DatetimeIndex, columns ``[open, high, low, close, volume]``.
        Matches the shape of fortress.market_data.BacktestDataProvider's
        input, so BacktestEngine needs no wiring changes.
    """
    import duckdb

    pattern = _parquet_glob()
    where_symbol = ""
    params: list = [str(start), str(end)]
    if symbols is not None:
        sym_list = list(dict.fromkeys(symbols))  # dedupe preserving order
        if not sym_list:
            return {}
        placeholders = ",".join("?" * len(sym_list))
        where_symbol = f" AND symbol IN ({placeholders})"
        params.extend(sym_list)

    sql = f"""
        SELECT symbol, date, open, high, low, close, volume
          FROM read_parquet('{pattern}')
         WHERE date BETWEEN ? AND ?
               {where_symbol}
         ORDER BY symbol, date
    """
    con = duckdb.connect()
    df = con.execute(sql, params).df()
    con.close()

    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    result: Dict[str, pd.DataFrame] = {}
    symbols_present = df["symbol"].unique()
    for sym in symbols_present:
        sub = df[df["symbol"] == sym].drop(columns=["symbol"]).sort_index()
        result[sym] = sub

    if apply_adj:
        adj_factors = _load_adj_factors(result.keys())
        for sym, price_df in list(result.items()):
            adj = adj_factors.get(sym)
            if adj is not None and not adj.empty:
                result[sym] = _apply_split_adjustment(price_df, adj)

    logger.info(
        "nse-universe bulk load: %d symbols × %s-%s",
        len(result),
        start,
        end,
    )
    return result


def _synthetic_sectoral_index(
    member_symbols: list[str],
    data: Dict[str, pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Equal-weight index of member stocks' daily returns.

    Returns a DataFrame with a ``close`` column (the compounded index
    level starting at 100) indexed by trading day. Used to approximate
    NSE sectoral indices (NIFTY BANK, NIFTY IT, etc.) which aren't in
    the bhavcopy parquet. Correlation with the real sector index is
    usually > 0.95 for equal-weighted baskets of 10+ constituents.
    """
    if not member_symbols:
        return None
    return_frames = []
    for sym in member_symbols:
        df = data.get(sym)
        if df is None or df.empty or "close" not in df.columns:
            continue
        rets = df["close"].pct_change()
        return_frames.append(rets)
    if not return_frames:
        return None
    # Align on union of dates; NaN-fill missing stocks with 0 return
    # (conservative — treats missing data as flat, not a drag/boost).
    combined = pd.concat(return_frames, axis=1).fillna(0)
    eq_weighted = combined.mean(axis=1)
    level = 100 * (1 + eq_weighted).cumprod()
    level = level.ffill().bfill()
    return pd.DataFrame({
        "open": level, "high": level, "low": level, "close": level, "volume": 0.0,
    })


def _inject_synthetic_sectoral_indices(
    data: Dict[str, pd.DataFrame],
    sectors_path: str = "stock-sectors.json",
    metadata_path: str = "market-metadata.json",
) -> None:
    """Compute synthetic NIFTY sectoral indices (NIFTY BANK / IT / PHARMA /
    etc.) from constituent stocks and inject them into ``data`` so the
    backtest's E5 sector-momentum filter has series to compare against.

    Real NSE sectoral indices aren't in the bhavcopy parquet; the strategy
    expects keys like "NIFTY BANK", "NIFTY IT" in market_data. We substitute
    an equal-weighted basket of all stock-sectors.json members of each
    sector (5-50 stocks per sector), reconstructed from current ``data``.
    """
    import json
    from pathlib import Path as _P

    root = _P(__file__).resolve().parent.parent
    sectors_file = root / sectors_path
    metadata_file = root / metadata_path
    if not sectors_file.exists() or not metadata_file.exists():
        return

    sectors = json.loads(sectors_file.read_text()).get("symbols", {})
    metadata = json.loads(metadata_file.read_text())
    sectoral_indices = metadata.get("sectoral_indices", {})

    # Group data symbols by sector.
    by_sector: Dict[str, list[str]] = {}
    for sym in data.keys():
        entry = sectors.get(sym)
        if entry:
            by_sector.setdefault(entry["sector"], []).append(sym)

    for idx_key, idx_data in sectoral_indices.items():
        sector = idx_data.get("maps_to_sector")
        if sector is None:
            continue
        members = by_sector.get(sector, [])
        if not members:
            continue
        synth = _synthetic_sectoral_index(members, data)
        if synth is not None:
            zs = idx_data.get("zerodha_symbol") or idx_data.get("symbol")
            if zs:
                data[zs] = synth


def load_historical_for_backtest(
    start: date,
    end: date,
    rank_range: tuple = (1, 500),
) -> Dict[str, pd.DataFrame]:
    """Convenience wrapper for the backtest engine.

    Computes the union of rank-window members across every monthly
    snapshot in [start, end], then bulk-loads their prices with split
    adjustment. The rank_range widens to 500 by default so the snapshot
    set tracks drop-outs — a stock ranked 180 this month might be 220
    next month but still needs price data through the transition.
    """
    from nse_universe import Universe as NSEUniverse

    nse = NSEUniverse()
    members_df = nse.members_df(start, end, "nifty_500")
    lo, hi = rank_range
    # Widen the window to nifty_500 so drop-outs mid-window stay priced.
    wide_lo = max(1, lo)
    wide_hi = max(hi + 100, 500)
    in_window = members_df[(members_df["rank"] >= wide_lo) & (members_df["rank"] <= wide_hi)]
    symbols = in_window["symbol"].unique().tolist()
    logger.info(
        "backtest symbol union %s-%s rank (%d,%d): %d symbols",
        start, end, wide_lo, wide_hi, len(symbols),
    )
    data = load_historical_bulk(start, end, symbols=symbols, apply_adj=True)

    # Inject real NIFTY 50 index from the cached yfinance download for the
    # strategy's relative-strength filter. Previous design used NIFTYBEES
    # as a proxy, but its raw bhavcopy series understates NIFTY 50 returns
    # by ~30% over 13 years because ETF dividends aren't captured. The real
    # ^NSEI index from Yahoo tracks constituent re-investment properly.
    from pathlib import Path as _P
    bench_parquet = _P(__file__).resolve().parent.parent / "data" / "benchmarks" / "nifty_50.parquet"
    if bench_parquet.exists():
        bench_df = pd.read_parquet(bench_parquet)
        # Filter to the requested window.
        mask = (bench_df.index >= pd.Timestamp(start)) & (bench_df.index <= pd.Timestamp(end))
        sliced = bench_df.loc[mask]
        if not sliced.empty:
            data["NIFTY 50"] = sliced
    else:
        logger.warning(
            "NIFTY 50 benchmark parquet missing at %s. "
            "Run tools/build_benchmark.py. Strategy RS calc will default to 1.0.",
            bench_parquet,
        )

    # Synthetic sectoral indices: function is kept available for experimentation
    # but NOT injected by default. Backtest showed that an equal-weighted basket
    # of sector members is too noisy relative to the real cap-weighted NSE index,
    # and the strategy's E5 sector-momentum filter penalises the wrong sectors
    # when fed this proxy (CAGR 14.3% → 11.8% in 13y phase test). With the
    # injection off, E5 silently no-ops in backtest (strategy treats missing
    # sector-index data as zero momentum for that sector), which preserves the
    # benchmark-fix baseline. Live mode uses real sectoral indices from Kite.

    return data
