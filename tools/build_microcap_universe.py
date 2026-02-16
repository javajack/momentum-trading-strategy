#!/usr/bin/env python3
"""
Build MICROCAP150 universe from Nifty Smallcap 250 minus existing universes.

Usage:
    python tools/build_microcap_universe.py

Reads: ind_niftysmallcap250list.csv (in project root), stock-universe.json
Outputs: MICROCAP150 JSON block to stdout (paste into stock-universe.json)
"""

import csv
import json
import sys
from pathlib import Path

# NSE Industry -> our sector taxonomy (same as build_smallcap_universe.py)
INDUSTRY_TO_SECTOR = {
    "Financial Services": "FINANCIALS",
    "Capital Goods": "INDUSTRIALS",
    "Chemicals": "MATERIALS",
    "Construction": "INDUSTRIALS",
    "Construction Materials": "MATERIALS",
    "Consumer Durables": "CONSUMER_DISCRETIONARY",
    "Consumer Services": "CONSUMER_DISCRETIONARY",
    "Fast Moving Consumer Goods": "CONSUMER_STAPLES",
    "Healthcare": "HEALTHCARE",
    "Information Technology": "INFORMATION_TECHNOLOGY",
    "Metals & Mining": "METALS_MINING",
    "Oil Gas & Consumable Fuels": "ENERGY",
    "Power": "UTILITIES",
    "Realty": "REAL_ESTATE",
    "Telecommunication": "TELECOM",
    "Textiles": "MATERIALS",
    "Forest Materials": "REAL_ESTATE",
    "Automobile and Auto Components": "AUTOMOBILES",
    "Media Entertainment & Publication": "CONSUMER_DISCRETIONARY",
    "Diversified": "INDUSTRIALS",
}

# Per-company overrides for "Services" industry
SERVICES_OVERRIDES = {
    "BLUEDART": "INDUSTRIALS",
    "ECLERX": "INFORMATION_TECHNOLOGY",
    "INTELLECT": "INFORMATION_TECHNOLOGY",
    "LATENTVIEW": "INFORMATION_TECHNOLOGY",
    "TBOTEK": "INFORMATION_TECHNOLOGY",
}

# Industry -> sub_sector mapping
INDUSTRY_TO_SUB_SECTOR = {
    "Financial Services": "FINANCIAL_SERVICES",
    "Capital Goods": "CAPITAL_GOODS",
    "Chemicals": "SPECIALTY_CHEMICALS",
    "Construction": "CONSTRUCTION",
    "Construction Materials": "BUILDING_MATERIALS",
    "Consumer Durables": "CONSUMER_DURABLES",
    "Consumer Services": "CONSUMER_SERVICES",
    "Fast Moving Consumer Goods": "FMCG",
    "Healthcare": "PHARMACEUTICALS",
    "Information Technology": "IT_SERVICES",
    "Metals & Mining": "METALS_MINING",
    "Oil Gas & Consumable Fuels": "OIL_GAS",
    "Power": "POWER_GENERATION",
    "Realty": "REAL_ESTATE",
    "Telecommunication": "TELECOM_SERVICES",
    "Textiles": "TEXTILES",
    "Forest Materials": "REAL_ESTATE",
    "Automobile and Auto Components": "AUTO_COMPONENTS",
    "Media Entertainment & Publication": "MEDIA_ENTERTAINMENT",
    "Diversified": "DIVERSIFIED",
}

# Per-company sub_sector overrides
SUB_SECTOR_OVERRIDES = {
    "BLUEDART": "LOGISTICS",
    "ECLERX": "IT_SERVICES",
    "INTELLECT": "FINTECH",
    "LATENTVIEW": "IT_SERVICES",
    "TBOTEK": "IT_SERVICES",
}


def main():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "ind_niftysmallcap250list.csv"
    universe_path = project_root / "stock-universe.json"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        print(
            "Download from: https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    if not universe_path.exists():
        print(f"ERROR: {universe_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load existing universe tickers to exclude
    with open(universe_path) as f:
        uni = json.load(f)

    existing_tickers = set()
    for key in ["NIFTY100", "MIDCAP100", "NIFTYSC100"]:
        if key in uni.get("universes", {}):
            for s in uni["universes"][key]["stocks"]:
                existing_tickers.add(s["ticker"])

    print(f"Existing universe: {len(existing_tickers)} stocks to exclude", file=sys.stderr)

    stocks = []
    skipped = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row["Symbol"].strip()
            industry = row["Industry"].strip()
            isin = row["ISIN Code"].strip()
            series = row["Series"].strip()
            name = row["Company Name"].strip()

            # Skip if already in existing universes
            if symbol in existing_tickers:
                skipped.append(symbol)
                continue

            # Determine sector
            if industry == "Services":
                sector = SERVICES_OVERRIDES.get(symbol, "INDUSTRIALS")
            else:
                sector = INDUSTRY_TO_SECTOR.get(industry)
                if sector is None:
                    print(f"WARNING: Unknown industry '{industry}' for {symbol}", file=sys.stderr)
                    sector = "OTHER"

            # Determine sub_sector
            if symbol in SUB_SECTOR_OVERRIDES:
                sub_sector = SUB_SECTOR_OVERRIDES[symbol]
            elif industry == "Services":
                sub_sector = "SERVICES"
            else:
                sub_sector = INDUSTRY_TO_SUB_SECTOR.get(industry, sector)

            stocks.append(
                {
                    "ticker": symbol,
                    "name": name,
                    "isin": isin,
                    "industry": industry,
                    "sector": sector,
                    "series": series,
                    "zerodha_symbol": symbol,
                    "api_format": f"NSE:{symbol}",
                    "sub_sector": sub_sector,
                }
            )

    # Sort alphabetically by ticker
    stocks.sort(key=lambda s: s["ticker"])

    universe_block = {
        "MICROCAP150": {
            "count": len(stocks),
            "index_symbol": "NIFTY SMLCAP 250",
            "index_api_format": "NSE:NIFTY SMLCAP 250",
            "stocks": stocks,
        }
    }

    print(json.dumps(universe_block, indent=2))

    # Summary
    print(
        f"\n// Total: {len(stocks)} stocks (excluded {len(skipped)} overlapping)", file=sys.stderr
    )
    print(
        f"// Skipped: {', '.join(sorted(skipped)[:20])}{'...' if len(skipped) > 20 else ''}",
        file=sys.stderr,
    )
    sector_counts = {}
    for s in stocks:
        sector_counts[s["sector"]] = sector_counts.get(s["sector"], 0) + 1
    print("// Sector breakdown:", file=sys.stderr)
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"//   {sector}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
