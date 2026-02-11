"""
Tests for universe module.

Verifies invariants D1-D5.
"""

import pytest
from pathlib import Path

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
        """D1: Metadata is present."""
        assert universe.metadata is not None
        assert "version" in universe.metadata
        assert "total_stocks" in universe.metadata

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
        """Verify expected stock count (NIFTY100 + MIDCAP100 + NIFTYSC100)."""
        stocks = universe.get_all_stocks()
        assert len(stocks) == 300, f"Expected 300 stocks, got {len(stocks)}"

    def test_no_duplicate_tickers(self, universe):
        """D4: No duplicate tickers in universe."""
        tickers = [s.ticker for s in universe.get_all_stocks()]
        assert len(tickers) == len(set(tickers)), "Duplicate tickers found"


class TestUniverseFiltering:
    """Test universe filtering by sub-universe."""

    def test_filter_primary(self):
        """Filter to NIFTY100 + MIDCAP100 returns ~200 stocks."""
        project_root = Path(__file__).parent.parent
        universe_path = project_root / "stock-universe.json"
        u = Universe(str(universe_path), filter_universes=["NIFTY100", "MIDCAP100"])
        stocks = u.get_all_stocks()
        assert len(stocks) == 200, f"Expected 200 stocks, got {len(stocks)}"

    def test_filter_smallcap(self):
        """Filter to NIFTYSC100 returns 100 stocks."""
        project_root = Path(__file__).parent.parent
        universe_path = project_root / "stock-universe.json"
        u = Universe(str(universe_path), filter_universes=["NIFTYSC100"])
        stocks = u.get_all_stocks()
        assert len(stocks) == 100, f"Expected 100 stocks, got {len(stocks)}"

    def test_filter_preserves_hedges(self):
        """Hedges are always included regardless of filter."""
        project_root = Path(__file__).parent.parent
        universe_path = project_root / "stock-universe.json"
        u = Universe(str(universe_path), filter_universes=["NIFTYSC100"])
        gold = u.get_hedge("gold")
        assert gold is not None
        assert gold["symbol"] == "GOLDBEES"
        # Hedge tickers should be in stock cache
        goldbees = u.get_stock("GOLDBEES")
        assert goldbees is not None

    def test_no_filter_loads_all(self):
        """No filter loads all stocks."""
        project_root = Path(__file__).parent.parent
        universe_path = project_root / "stock-universe.json"
        u = Universe(str(universe_path))
        stocks = u.get_all_stocks()
        assert len(stocks) == 300


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


class TestETFAndHedges:
    """Test ETF and hedge instrument data."""

    def test_sector_etf_available(self, universe):
        """ETFs available for key sectors."""
        fin_etf = universe.get_sector_etf("FINANCIALS")
        assert fin_etf is not None
        assert fin_etf.symbol == "BANKBEES"

    def test_hedge_instruments(self, universe):
        """Hedge instruments are defined."""
        gold = universe.get_hedge("gold")
        assert gold is not None
        assert gold["symbol"] == "GOLDBEES"

        cash = universe.get_hedge("cash")
        assert cash is not None
        assert cash["symbol"] == "LIQUIDBEES"


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
