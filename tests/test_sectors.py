"""Integrity tests for stock-sectors.json — the offline classification artifact.

stock-sectors.json is generated once by tools/build_sectors.py and read at
runtime by both live and backtest paths. These tests guard its shape,
vocabulary, and regression invariants (every symbol currently in
stock-universe.json must keep the same sector in the generated file).
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SECTORS_PATH = PROJECT_ROOT / "stock-sectors.json"
UNIVERSE_PATH = PROJECT_ROOT / "stock-universe.json"

# Sector vocabulary mirrored from tools/build_sectors.py. Kept here so the
# check is independent of the builder (defense in depth).
VALID_SECTORS = {
    "FINANCIALS", "INFORMATION_TECHNOLOGY", "HEALTHCARE",
    "CONSUMER_DISCRETIONARY", "CONSUMER_STAPLES", "INDUSTRIALS",
    "INFRASTRUCTURE", "AUTOMOBILES", "ENERGY", "UTILITIES",
    "MATERIALS", "METALS_MINING", "REAL_ESTATE", "TELECOM",
    "MEDIA", "COMMODITIES", "DEBT", "INTERNATIONAL", "DEFENSIVE",
    "UNCLASSIFIED",
}


@pytest.fixture(scope="module")
def sectors():
    """Loaded stock-sectors.json."""
    if not SECTORS_PATH.exists():
        pytest.skip("stock-sectors.json not built yet — run tools/build_sectors.py")
    return json.loads(SECTORS_PATH.read_text())


class TestShape:
    def test_top_level_keys(self, sectors):
        for k in ("generated_at", "total_symbols", "source_counts", "symbols"):
            assert k in sectors, f"missing top-level key: {k}"

    def test_total_matches_symbols(self, sectors):
        assert sectors["total_symbols"] == len(sectors["symbols"])

    def test_each_entry_has_sector_and_source(self, sectors):
        for sym, entry in sectors["symbols"].items():
            assert "sector" in entry, f"{sym} missing sector"
            assert "sub_sector" in entry, f"{sym} missing sub_sector"
            assert "source" in entry, f"{sym} missing source"
            assert entry["source"] in (
                "universe_json", "llm_map", "heuristic", "unclassified"
            ), f"{sym} bad source: {entry['source']}"

    def test_every_sector_in_vocabulary(self, sectors):
        bad = [
            (sym, entry["sector"])
            for sym, entry in sectors["symbols"].items()
            if entry["sector"] not in VALID_SECTORS
        ]
        assert not bad, f"out-of-vocab sectors: {bad[:10]}"


class TestUniverseJsonRegression:
    """Every symbol in stock-universe.json must have the same sector in
    stock-sectors.json — otherwise we silently broke sector caps for live."""

    def test_no_regression_on_universe_stocks(self, sectors):
        universe = json.loads(UNIVERSE_PATH.read_text())
        by_symbol = sectors["symbols"]
        mismatches = []
        for uni_name, uni in universe.get("universes", {}).items():
            for s in uni.get("stocks", []):
                ticker = s["ticker"]
                want = s["sector"]
                got = by_symbol.get(ticker, {}).get("sector")
                if got != want:
                    mismatches.append(f"{ticker}: universe={want} sectors={got}")
        assert not mismatches, f"sector regression: {mismatches[:10]}"

    def test_universe_stocks_marked_authoritative(self, sectors):
        """Every symbol from stock-universe.json should have source=universe_json."""
        universe = json.loads(UNIVERSE_PATH.read_text())
        by_symbol = sectors["symbols"]
        non_authoritative = []
        for uni_name, uni in universe.get("universes", {}).items():
            for s in uni.get("stocks", []):
                ticker = s["ticker"]
                entry = by_symbol.get(ticker, {})
                if entry.get("source") != "universe_json":
                    non_authoritative.append(f"{ticker}: source={entry.get('source')}")
        assert not non_authoritative, (
            f"universe.json symbols should be marked universe_json: {non_authoritative[:5]}"
        )


class TestCoverage:
    def test_current_nifty_200_full_coverage(self, sectors):
        """The strategy's default window is top-200; every current member must
        have a non-UNCLASSIFIED sector so the sector cap works correctly."""
        from datetime import date

        nse_universe = pytest.importorskip("nse_universe")
        u = nse_universe.Universe()
        members = u.members(date.today(), "nifty_200")
        by_symbol = sectors["symbols"]
        unclassified = [
            m for m in members
            if by_symbol.get(m, {}).get("sector") == "UNCLASSIFIED"
        ]
        # Some newly-added IPOs may be unclassified; keep the guard loose but
        # flag if it exceeds 10% — at that point rebuild is overdue.
        assert len(unclassified) <= max(20, int(0.10 * len(members))), (
            f"{len(unclassified)} of {len(members)} current nifty_200 unclassified. "
            f"Rebuild sectors: .venv/bin/python tools/build_sectors.py"
        )

    def test_hedges_classified_as_defensive(self, sectors):
        """Registered hedges must land under COMMODITIES / DEBT (or similar),
        never in an equity sector where sector caps would pollute them."""
        by_symbol = sectors["symbols"]
        defensive_sectors = {"COMMODITIES", "DEBT", "DEFENSIVE", "INTERNATIONAL"}
        for sym in ("GOLDBEES", "LIQUIDBEES"):
            if sym in by_symbol:  # guard — not every env has the same universe
                assert by_symbol[sym]["sector"] in defensive_sectors, (
                    f"{sym} classified as equity sector: {by_symbol[sym]}"
                )
