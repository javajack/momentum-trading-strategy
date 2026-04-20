"""Tests for fortress.nse_data_loader — the backtest data source."""

from datetime import date

import pandas as pd
import pytest

from fortress.nse_data_loader import (
    _apply_split_adjustment,
    load_historical_bulk,
)


class TestSplitAdjustment:
    def test_no_events_returns_unchanged(self):
        df = pd.DataFrame(
            {"close": [100, 101, 102], "volume": [1000, 1000, 1000]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        )
        out = _apply_split_adjustment(df, pd.DataFrame())
        pd.testing.assert_frame_equal(out, df)

    def test_single_future_split_halves_prices_before_event(self):
        # Price pre-split 1000; a 2:1 split on 2020-06-01 means pre-split
        # prices should be multiplied by 0.5 so they're comparable to post.
        df = pd.DataFrame(
            {"close": [1000.0, 1000.0, 500.0], "volume": [100, 100, 200]},
            index=pd.to_datetime(["2020-01-01", "2020-05-31", "2020-06-01"]),
        )
        adj = pd.DataFrame(
            [{"event_date": "2020-06-01", "after_split_factor": 0.5}]
        )
        out = _apply_split_adjustment(df, adj)
        assert out.loc["2020-01-01", "close"] == 500.0
        assert out.loc["2020-05-31", "close"] == 500.0
        # Event date itself is NOT adjusted (price is already post-split).
        assert out.loc["2020-06-01", "close"] == 500.0
        # Volume inverts — more shares equivalent-traded pre-split.
        assert out.loc["2020-01-01", "volume"] == 200

    def test_compounded_splits_use_earliest_upcoming_factor(self):
        # Two splits: 2020 (cum factor 0.25 — i.e. 4 shares per historic 1)
        # and 2022 (cum factor 0.5 — 2 shares per historic 1).
        # nse-universe already encodes cumulative factors; the function
        # picks the earliest upcoming event for any given date.
        df = pd.DataFrame(
            {"close": [1000.0, 1000.0, 1000.0, 1000.0]},
            index=pd.to_datetime(["2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]),
        )
        adj = pd.DataFrame([
            {"event_date": "2020-06-01", "after_split_factor": 0.25},
            {"event_date": "2022-06-01", "after_split_factor": 0.5},
        ])
        out = _apply_split_adjustment(df, adj)
        assert out.loc["2019-01-01", "close"] == 250.0  # × 0.25 (both splits ahead)
        assert out.loc["2021-01-01", "close"] == 500.0  # × 0.5 (only 2022 ahead)
        assert out.loc["2023-01-01", "close"] == 1000.0  # no events ahead
        assert out.loc["2025-01-01", "close"] == 1000.0


class TestBulkLoader:
    """Full-stack tests — require ~/work/nse500 data to be present."""

    def test_tiny_range_returns_expected_shape(self):
        data = load_historical_bulk(
            start=date(2024, 1, 1),
            end=date(2024, 1, 15),
            symbols=["RELIANCE", "TCS"],
            apply_adj=False,
        )
        assert set(data.keys()) == {"RELIANCE", "TCS"}
        for df in data.values():
            # 10-11 trading days in the window.
            assert 8 <= len(df) <= 12
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]
            assert df.index.is_monotonic_increasing
            assert df.index.dtype.kind == "M"  # datetime

    def test_split_adjustment_moves_pre_split_prices(self):
        # RELIANCE had a 1:1 bonus on 2024-10-28 (after_split_factor=0.5).
        # A close from 2024-01 should be ~half the raw bhavcopy value.
        unadj = load_historical_bulk(
            date(2024, 1, 1), date(2024, 1, 5),
            symbols=["RELIANCE"], apply_adj=False,
        )["RELIANCE"]
        adj = load_historical_bulk(
            date(2024, 1, 1), date(2024, 1, 5),
            symbols=["RELIANCE"], apply_adj=True,
        )["RELIANCE"]
        # Adjusted should be strictly smaller than unadjusted (split-halved).
        assert (adj["close"] < unadj["close"]).all()
        # Factor ratio ≈ 0.5.
        ratio = (adj["close"] / unadj["close"]).mean()
        assert 0.4 < ratio < 0.6, f"expected ~0.5, got {ratio}"

    def test_empty_symbol_filter_returns_empty(self):
        data = load_historical_bulk(date(2024, 1, 1), date(2024, 1, 5), symbols=[])
        assert data == {}
