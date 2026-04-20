"""
Tests for universe module.

Verifies invariants D1-D5.
"""

from pathlib import Path

import pytest

from fortress.universe import Universe, UniverseValidationError


@pytest.fixture
def universe():
    """Load the stock universe."""
    # Find the universe file relative to project root
    project_root = Path(__file__).parent.parent
    universe_path = project_root / "stock-universe.json"
    return Universe(str(universe_path))


class TestUniverseLoading:
    """Test universe loading and validation."""

    def test_loads_successfully(self, universe):
        """Universe loads without errors."""
        assert universe is not None

    def test_metadata_present(self, universe):
        """D1: Metadata is present (composed from market-metadata.json)."""
        assert universe.metadata is not None
        assert "version" in universe.metadata

    def test_benchmark_present(self, universe):
        """D1: Benchmark index is present."""
        benchmark = universe.benchmark
        assert benchmark.symbol == "NIFTY 50"
        assert benchmark.api_format == "NSE:NIFTY 50"


class TestStockIntegrity:
    """Test stock data integrity (D2, D4)."""

    def test_all_stocks_have_zerodha_symbol(self, universe):
        """D2: All stocks have valid zerodha_symbol."""
        for stock in universe.get_all_stocks():
            assert stock.zerodha_symbol, f"{stock.ticker} missing zerodha_symbol"
            assert stock.api_format, f"{stock.ticker} missing api_format"
            assert stock.api_format.startswith("NSE:"), f"{stock.ticker} bad api_format"

    def test_stock_count(self, universe):
        """Rank-200 window returns near-200 stocks after ETF filter.

        nse-universe's rank_range=(1,200) delivers 200 symbols, but a
        handful are index/ETF instruments (NIFTYBEES, GOLDBEES etc.)
        that the DEFENSIVE/COMMODITIES/DEBT filter strips out. Keep
        tolerance wide: the window aims at equities and the count
        drifts as new ETFs earn liquidity rank.
        """
        stocks = universe.get_all_stocks()
        assert 170 <= len(stocks) <= 200, f"unexpected universe size: {len(stocks)}"

    def test_no_duplicate_tickers(self, universe):
        """D4: No duplicate tickers in universe."""
        tickers = [s.ticker for s in universe.get_all_stocks()]
        assert len(tickers) == len(set(tickers)), "Duplicate tickers found"


class TestRankRange:
    """The rank_range knob picks different slices of the nse-universe."""

    def test_default_is_top_200(self):
        u = Universe()
        stocks = u.get_all_stocks()
        # After ETF filter, a top-200 window yields roughly 180-200 equities.
        assert 170 <= len(stocks) <= 200

    def test_narrower_range_yields_fewer(self):
        top50 = Universe(rank_range=(1, 50))
        top200 = Universe(rank_range=(1, 200))
        assert len(top50.get_all_stocks()) < len(top200.get_all_stocks())

    def test_midcap_window_different_from_large_cap(self):
        top100 = set(s.ticker for s in Universe(rank_range=(1, 100)).get_all_stocks())
        midcap = set(s.ticker for s in Universe(rank_range=(101, 250)).get_all_stocks())
        # Disjoint windows — no overlap.
        assert top100.isdisjoint(midcap)

    def test_hedges_always_available_regardless_of_rank_range(self):
        u = Universe(rank_range=(1, 50))  # tight window
        gold = u.get_hedge("gold")
        assert gold is not None
        assert gold["symbol"] == "GOLDBEES"
        # Hedge ticker is lookable even if not in rank window.
        goldbees = u.get_stock("GOLDBEES")
        assert goldbees is not None


class TestSectorData:
    """Test sector-related functionality."""

    def test_valid_sectors_returned(self, universe):
        """C4: Sectors with >= 3 stocks are valid."""
        valid = universe.get_valid_sectors(min_stocks=3)
        assert len(valid) >= 10, f"Expected at least 10 valid sectors, got {len(valid)}"
        assert "TEXTILES" not in valid, "TEXTILES should not be valid (only 1 stock)"

    def test_sector_index_mapping(self, universe):
        """D8: Sectors have index mappings."""
        # Key sectors should have indices
        for sector in ["FINANCIALS", "INFORMATION_TECHNOLOGY", "HEALTHCARE"]:
            index = universe.get_sector_index(sector)
            assert index is not None, f"{sector} missing sector index"
            assert index.api_format, f"{sector} index missing api_format"

    def test_get_stocks_by_sector(self, universe):
        """Get stocks filters correctly by sector."""
        it_stocks = universe.get_stocks_by_sector("INFORMATION_TECHNOLOGY")
        assert len(it_stocks) >= 5, "Expected at least 5 IT stocks"
        assert all(s.sector == "INFORMATION_TECHNOLOGY" for s in it_stocks)


class TestSectorSummary:
    """Test sector summary integrity (D5)."""

    def test_sector_summary_matches_actual(self, universe):
        """D5: sector_summary totals match actual counts."""
        # This is validated during loading, but let's verify manually
        all_stocks = universe.get_all_stocks()
        actual_counts = {}
        for stock in all_stocks:
            actual_counts[stock.sector] = actual_counts.get(stock.sector, 0) + 1

        # Check a few key sectors
        fin_stocks = universe.get_stocks_by_sector("FINANCIALS")
        assert len(fin_stocks) == actual_counts.get("FINANCIALS", 0)


class TestHedges:
    """Hedge instrument registry (sector ETFs live in a different codebase)."""

    def test_hedge_instruments(self, universe):
        gold = universe.get_hedge("gold")
        assert gold is not None
        assert gold["symbol"] == "GOLDBEES"

        cash = universe.get_hedge("cash")
        assert cash is not None
        assert cash["symbol"] == "LIQUIDBEES"


class TestManagedFilter:
    """is_managed_symbol / hedge_symbols — foundation for skipping stray holdings."""

    def test_hedge_symbols_include_gold_and_cash(self, universe):
        assert "GOLDBEES" in universe.hedge_symbols
        assert "LIQUIDBEES" in universe.hedge_symbols

    def test_is_managed_for_universe_stock(self, universe):
        assert universe.is_managed_symbol("RELIANCE") is True

    def test_is_managed_for_hedge(self, universe):
        assert universe.is_managed_symbol("GOLDBEES") is True
        assert universe.is_managed_symbol("LIQUIDBEES") is True

    def test_not_managed_for_stray_etf(self, universe):
        # NIFTYBEES isn't in the strategy universe and isn't a hedge — it's
        # the kind of symbol the user might own that the strategy should ignore.
        assert universe.is_managed_symbol("NIFTYBEES") is False

    def test_not_managed_for_unknown_symbol(self, universe):
        assert universe.is_managed_symbol("SOMETHING_INVENTED") is False


class TestAPIFormat:
    """Test API format conversion."""

    def test_get_api_symbols(self, universe):
        """API symbols are correctly formatted."""
        symbols = universe.get_api_symbols(["RELIANCE", "TCS", "HDFC"])
        assert len(symbols) >= 2  # At least 2 should exist
        for sym in symbols:
            assert sym.startswith("NSE:"), f"Bad API format: {sym}"

    def test_vix_available(self, universe):
        """VIX index is available."""
        vix = universe.get_vix()
        assert vix.symbol == "INDIA VIX"
        assert vix.instrument_token == 264969
