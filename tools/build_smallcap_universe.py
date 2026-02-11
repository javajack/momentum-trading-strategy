#!/usr/bin/env python3
"""
Convert ind_niftysmallcap100list.csv to JSON block for stock-universe.json.

Usage:
    python tools/build_smallcap_universe.py

Reads: ind_niftysmallcap100list.csv (in project root)
Outputs: NIFTYSC100 JSON block to stdout (paste into stock-universe.json)
"""

import csv
import json
import sys
from pathlib import Path

# NSE Industry -> our sector taxonomy
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
}

# Per-company overrides for "Services" industry
SERVICES_OVERRIDES = {
    "DELHIVERY": "INDUSTRIALS",
    "FSL": "INFORMATION_TECHNOLOGY",
    "GESHIP": "INDUSTRIALS",
    "IGIL": "CONSUMER_DISCRETIONARY",
    "REDINGTON": "INFORMATION_TECHNOLOGY",
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
}

# Per-company sub_sector overrides
SUB_SECTOR_OVERRIDES = {
    "DELHIVERY": "LOGISTICS",
    "FSL": "IT_SERVICES",
    "GESHIP": "SHIPPING",
    "IGIL": "LUXURY_GEMS",
    "REDINGTON": "IT_DISTRIBUTION",
}


def main():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "ind_niftysmallcap100list.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    stocks = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row["Symbol"].strip()
            industry = row["Industry"].strip()
            isin = row["ISIN Code"].strip()
            series = row["Series"].strip()
            name = row["Company Name"].strip()

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

            stocks.append({
                "ticker": symbol,
                "name": name,
                "isin": isin,
                "industry": industry,
                "sector": sector,
                "series": series,
                "zerodha_symbol": symbol,
                "api_format": f"NSE:{symbol}",
                "sub_sector": sub_sector,
            })

    # Sort alphabetically by ticker
    stocks.sort(key=lambda s: s["ticker"])

    universe_block = {
        "NIFTYSC100": {
            "count": len(stocks),
            "index_symbol": "NIFTY SMLCAP 100",
            "index_api_format": "NSE:NIFTY SMLCAP 100",
            "stocks": stocks,
        }
    }

    print(json.dumps(universe_block, indent=2))

    # Summary
    sector_counts = {}
    for s in stocks:
        sector_counts[s["sector"]] = sector_counts.get(s["sector"], 0) + 1
    print(f"\n// Total: {len(stocks)} stocks", file=sys.stderr)
    print("// Sector breakdown:", file=sys.stderr)
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"//   {sector}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
