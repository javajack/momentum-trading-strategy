#!/usr/bin/env python
"""Fetch real NIFTY 50 index history from Yahoo Finance and cache to parquet.

The backtest uses this series for:
  1. Relative-strength calculations (RS = stock_return / NIFTY_return)
  2. Reported alpha vs benchmark

Previously the code used NIFTYBEES as a NIFTY 50 proxy, but NIFTYBEES's raw
bhavcopy series misses cumulative dividend payouts (~1.5% p.a.) so the
"NIFTY 50" proxy under-stated real index returns by ~30% over 13 years.
The real ^NSEI index on Yahoo already accounts for dividends implicitly
(price-only; but dividends feed back through the index constituent NAVs).

Run once at setup, or quarterly to keep the benchmark current:

    .venv/bin/python tools/build_benchmark.py

Writes data/benchmarks/nifty_50.parquet (~200 KB, 20 years of daily OHLCV).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "nifty_50.parquet"


def fetch_nifty_50(start: str = "2005-01-01") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "yfinance not installed — pull via `pip install yfinance` or "
            "rely on the nse-universe dep which brings it in."
        ) from e

    ticker = yf.Ticker("^NSEI")
    df = ticker.history(start=start, end=str(date.today()), auto_adjust=False)
    if df.empty:
        raise RuntimeError("yfinance returned empty for ^NSEI — retry later")

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df = df.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    return df


def build() -> None:
    df = fetch_nifty_50()
    first, last = df.iloc[0]["close"], df.iloc[-1]["close"]
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    total = last / first - 1
    cagr = (last / first) ** (1 / yrs) - 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, compression="snappy")

    print(f"NIFTY 50 benchmark → {OUTPUT}")
    print(f"  rows:         {len(df):,}")
    print(f"  date range:   {df.index[0].date()} → {df.index[-1].date()} ({yrs:.1f} yrs)")
    print(f"  first close:  {first:,.2f}")
    print(f"  last close:   {last:,.2f}")
    print(f"  total return: {total:+.1%}")
    print(f"  CAGR:         {cagr:+.2%}")


if __name__ == "__main__":
    build()
