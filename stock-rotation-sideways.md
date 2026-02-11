# FORTRESS SIDEWAYS: Complete Implementation Specification
## Sector Rotation Strategy for Indian Equities

**Version:** 3.0  
**Date:** 2026-01-29  
**Platform:** Zerodha Kite Connect API + NSE  
**Capital:** ₹16,00,000 (Configurable)  
**Target:** Claude Code / LLM Implementation

---

# TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Universe Configuration](#3-universe-configuration)
4. [Data Infrastructure](#4-data-infrastructure)
5. [Sector Rotation Engine](#5-sector-rotation-engine)
6. [Risk Management](#6-risk-management)
7. [Order Management](#7-order-management)
8. [Zerodha Integration](#8-zerodha-integration)
9. [CLI Interface](#9-cli-interface)
10. [Backtest Framework](#10-backtest-framework)
11. [Configuration Schema](#11-configuration-schema)
12. [File Structure](#12-file-structure)
13. [Invariants Checklist](#13-invariants-checklist)

---

# 1. EXECUTIVE SUMMARY

## 1.1 Strategy Overview

FORTRESS SIDEWAYS is a **sector rotation momentum strategy** that:
- Ranks 17 sectors by Risk-Adjusted Relative Velocity (RRV)
- Allocates capital to top 3-5 sectors
- Selects top 5 stocks per chosen sector using momentum scoring
- Rebalances weekly with incremental position changes
- Uses sectoral indices for regime detection and ETFs for hedging

## 1.2 Key Design Principles

1. **Stateless Operation** - Every run recomputes state from holdings + market data
2. **Safety > Speed > Cleverness** - Conservative defaults, explicit overrides
3. **Human-in-the-Loop** - Dry-run by default, explicit confirmation for live orders
4. **Fail Closed** - Any ambiguity halts trading, never assumes
5. **Audit Everything** - Complete decision trace for every action

## 1.3 Performance Targets

| Metric | Target | Hard Limit |
|--------|--------|------------|
| CAGR | 25-40% | - |
| Max Drawdown | <20% | 25% (halt) |
| Sharpe Ratio | >1.2 | - |
| Win Rate | >55% | - |
| Max Single Position | 8% | 12% |
| Max Sector Exposure | 35% | 45% |

---

# 2. SYSTEM ARCHITECTURE

## 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI Interface                                │
│  [status] [scan] [rebalance] [backtest] [holdings] [config]         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      Session Manager                                 │
│  - Kite Connect Auth (request_token from user)                      │
│  - Token refresh handling                                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Universe Loader │ │  Market Data    │ │ Portfolio State │
│ (universe.json) │ │  (Kite API)     │ │ (Holdings/Pos)  │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Sector Rotation Engine                            │
│  1. Sector RRV Ranking                                              │
│  2. Stock Momentum Scoring                                          │
│  3. Target Portfolio Construction                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      Risk Governor                                   │
│  - Drawdown checks                                                  │
│  - Position limits                                                  │
│  - Sector concentration                                             │
│  - VIX regime detection                                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    Rebalance Engine                                  │
│  - Delta calculation (current vs target)                            │
│  - Order generation (sells first, then buys)                        │
│  - Incremental sizing (max 20% position change per week)            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                     Order Manager                                    │
│  - Dry-run validation                                               │
│  - Margin checks                                                    │
│  - Order placement (CNC)                                            │
│  - Status tracking                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## 2.2 Module Dependencies

```
fortress/
├── __init__.py
├── cli.py              # Click-based CLI entry point
├── config.py           # Pydantic config models
├── auth.py             # Zerodha authentication
├── universe.py         # Universe parser (universe.json)
├── instruments.py      # Instrument token mapper
├── market_data.py      # OHLC fetching, caching
├── indicators.py       # Momentum, volatility calculations
├── sector_engine.py    # Sector ranking, stock selection
├── risk_governor.py    # Risk checks, position limits
├── portfolio.py        # Holdings, positions, P&L
├── rebalancer.py       # Delta calculation, order generation
├── order_manager.py    # Order placement, tracking
├── backtest.py         # Historical simulation
├── logger.py           # Structured logging, decision trace
└── utils.py            # Rate limiting, helpers
```

---

# 3. UNIVERSE CONFIGURATION

## 3.1 Universe Schema (universe.json)

The universe file contains all tradeable instruments with Zerodha-compatible symbols.

### 3.1.1 Top-Level Structure

```json
{
  "metadata": {
    "version": "2.0",
    "schema_version": "2025-01",
    "source": "NSE Index Constituents",
    "created_at": "2026-01-29",
    "total_stocks": 201,
    "notes": {
      "zerodha_api_usage": "Use exchange:tradingsymbol format",
      "index_segment": "Indices use segment='INDICES'",
      "etf_segment": "ETFs use segment='EQ'"
    }
  },
  "benchmark": { ... },
  "broad_market_indices": { ... },
  "sectoral_indices": { ... },
  "hedges": { ... },
  "sectoral_etfs": { ... },
  "broad_market_etfs": { ... },
  "sector_mapping": { ... },
  "sectors": { ... },
  "universes": {
    "NIFTY100": { "count": 101, "stocks": [...] },
    "MIDCAP100": { "count": 100, "stocks": [...] }
  },
  "sector_summary": { ... }
}
```

### 3.1.2 Benchmark Definition

```json
"benchmark": {
  "symbol": "NIFTY 50",
  "zerodha_symbol": "NIFTY 50",
  "exchange": "NSE",
  "instrument_type": "INDEX",
  "segment": "INDICES",
  "instrument_token": 256265,
  "api_format": "NSE:NIFTY 50",
  "description": "NSE NIFTY 50 Index"
}
```

### 3.1.3 Sectoral Indices (Pre-Resolved Tokens)

```json
"sectoral_indices": {
  "NIFTY_BANK": {
    "symbol": "NIFTY BANK",
    "zerodha_symbol": "NIFTY BANK",
    "exchange": "NSE",
    "segment": "INDICES",
    "instrument_token": 260105,
    "api_format": "NSE:NIFTY BANK",
    "maps_to_sector": "FINANCIAL_SERVICES",
    "description": "Banking sector index"
  },
  "NIFTY_IT": {
    "symbol": "NIFTY IT",
    "instrument_token": 259849,
    "api_format": "NSE:NIFTY IT",
    "maps_to_sector": "IT_SERVICES"
  },
  "NIFTY_PHARMA": {
    "symbol": "NIFTY PHARMA",
    "instrument_token": 262409,
    "api_format": "NSE:NIFTY PHARMA",
    "maps_to_sector": "PHARMA_HEALTHCARE"
  },
  "NIFTY_AUTO": {
    "symbol": "NIFTY AUTO",
    "instrument_token": 263433,
    "api_format": "NSE:NIFTY AUTO",
    "maps_to_sector": "AUTOMOBILE"
  },
  "NIFTY_METAL": {
    "symbol": "NIFTY METAL",
    "instrument_token": 263689,
    "api_format": "NSE:NIFTY METAL",
    "maps_to_sector": "METALS_MINING"
  },
  "NIFTY_ENERGY": {
    "symbol": "NIFTY ENERGY",
    "instrument_token": 261641,
    "api_format": "NSE:NIFTY ENERGY",
    "maps_to_sector": "OIL_GAS_ENERGY"
  },
  "NIFTY_FMCG": {
    "symbol": "NIFTY FMCG",
    "instrument_token": 261897,
    "api_format": "NSE:NIFTY FMCG",
    "maps_to_sector": "CONSUMER_GOODS"
  },
  "NIFTY_REALTY": {
    "symbol": "NIFTY REALTY",
    "instrument_token": 261129,
    "api_format": "NSE:NIFTY REALTY",
    "maps_to_sector": "REALTY"
  },
  "NIFTY_INFRA": {
    "symbol": "NIFTY INFRA",
    "api_format": "NSE:NIFTY INFRA",
    "maps_to_sector": "CEMENT_CONSTRUCTION"
  },
  "NIFTY_MEDIA": {
    "symbol": "NIFTY MEDIA",
    "instrument_token": 263945,
    "api_format": "NSE:NIFTY MEDIA",
    "maps_to_sector": "TELECOM_MEDIA"
  },
  "NIFTY_CONSUMPTION": {
    "symbol": "NIFTY CONSUMPTION",
    "instrument_token": 257545,
    "api_format": "NSE:NIFTY CONSUMPTION",
    "maps_to_sector": "CONSUMER_SERVICES"
  },
  "NIFTY_PSU_BANK": {
    "symbol": "NIFTY PSU BANK",
    "instrument_token": 262921,
    "api_format": "NSE:NIFTY PSU BANK",
    "maps_to_sector": "FINANCIAL_SERVICES"
  },
  "NIFTY_FIN_SERVICE": {
    "symbol": "NIFTY FIN SERVICE",
    "api_format": "NSE:NIFTY FIN SERVICE",
    "maps_to_sector": "FINANCIAL_SERVICES"
  },
  "NIFTY_SERV_SECTOR": {
    "symbol": "NIFTY SERV SECTOR",
    "instrument_token": 263177,
    "api_format": "NSE:NIFTY SERV SECTOR",
    "maps_to_sector": "SERVICES"
  },
  "NIFTY_COMMODITIES": {
    "symbol": "NIFTY COMMODITIES",
    "instrument_token": 257289,
    "api_format": "NSE:NIFTY COMMODITIES",
    "maps_to_sector": "METALS_MINING"
  },
  "NIFTY_PSE": {
    "symbol": "NIFTY PSE",
    "instrument_token": 262665,
    "api_format": "NSE:NIFTY PSE"
  }
}
```

### 3.1.4 Hedge Instruments

```json
"hedges": {
  "gold": {
    "symbol": "GOLDBEES",
    "api_format": "NSE:GOLDBEES",
    "fund_house": "Nippon India",
    "description": "Gold ETF"
  },
  "international": {
    "symbol": "MON100",
    "api_format": "NSE:MON100",
    "description": "NASDAQ 100 ETF",
    "alternate_symbols": ["N100"]
  },
  "cash": {
    "symbol": "LIQUIDBEES",
    "api_format": "NSE:LIQUIDBEES",
    "description": "Liquid ETF - cash equivalent for margin"
  },
  "silver": {
    "symbol": "SILVERBEES",
    "api_format": "NSE:SILVERBEES",
    "description": "Silver ETF"
  }
}
```

### 3.1.5 Sectoral ETFs

```json
"sectoral_etfs": {
  "FINANCIAL_SERVICES": {
    "primary_etf": {
      "symbol": "BANKBEES",
      "api_format": "NSE:BANKBEES",
      "tracks_index": "NIFTY BANK"
    },
    "alternate_etfs": [
      {"symbol": "PSUBNKBEES", "tracks_index": "NIFTY PSU BANK"},
      {"symbol": "KOTAKBKETF", "tracks_index": "NIFTY BANK"},
      {"symbol": "SETFNIFBK", "tracks_index": "NIFTY BANK"}
    ]
  },
  "IT_SERVICES": {
    "primary_etf": {
      "symbol": "NETFIT",
      "api_format": "NSE:NETFIT",
      "tracks_index": "NIFTY IT"
    }
  },
  "PHARMA_HEALTHCARE": {
    "primary_etf": {
      "symbol": "PHARMABEES",
      "api_format": "NSE:PHARMABEES",
      "tracks_index": "NIFTY PHARMA"
    }
  },
  "CAPITAL_GOODS": {
    "primary_etf": {
      "symbol": "INFRABEES",
      "api_format": "NSE:INFRABEES",
      "tracks_index": "NIFTY INFRA"
    }
  },
  "CONSUMER_GOODS": {
    "primary_etf": {
      "symbol": "NETFCONSUM",
      "api_format": "NSE:NETFCONSUM",
      "tracks_index": "NIFTY CONSUMPTION"
    }
  }
}
```

### 3.1.6 Stock Entry Schema

```json
{
  "ticker": "RELIANCE",
  "name": "Reliance Industries Ltd.",
  "isin": "INE002A01018",
  "industry": "Oil Gas & Consumable Fuels",
  "sector": "OIL_GAS_ENERGY",
  "series": "EQ",
  "zerodha_symbol": "RELIANCE",
  "api_format": "NSE:RELIANCE"
}
```

### 3.1.7 Sector Distribution

| Sector | NIFTY100 | MIDCAP100 | Total | Valid for Ranking |
|--------|----------|-----------|-------|-------------------|
| FINANCIAL_SERVICES | 24 | 24 | 48 | ✓ |
| CAPITAL_GOODS | 8 | 17 | 25 | ✓ |
| AUTOMOBILE | 10 | 7 | 17 | ✓ |
| IT_SERVICES | 7 | 9 | 16 | ✓ |
| PHARMA_HEALTHCARE | 8 | 8 | 16 | ✓ |
| CONSUMER_GOODS | 9 | 6 | 15 | ✓ |
| CONSUMER_SERVICES | 5 | 7 | 12 | ✓ |
| POWER | 7 | 4 | 11 | ✓ |
| METALS_MINING | 7 | 4 | 11 | ✓ |
| OIL_GAS_ENERGY | 6 | 4 | 10 | ✓ |
| CEMENT_CONSTRUCTION | 5 | 3 | 8 | ✓ |
| CONSUMER_DURABLES | 3 | 5 | 8 | ✓ |
| CHEMICALS | 2 | 5 | 7 | ✓ |
| REALTY | 2 | 4 | 6 | ✓ |
| TELECOM_MEDIA | 1 | 4 | 5 | ✓ |
| SERVICES | 2 | 2 | 4 | ✓ |
| TEXTILES | 0 | 1 | 1 | ✗ (min 3) |

---

# 4. DATA INFRASTRUCTURE

## 4.1 Universe Parser

```python
# fortress/universe.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import json

@dataclass
class Stock:
    ticker: str
    name: str
    isin: str
    industry: str
    sector: str
    series: str
    zerodha_symbol: str
    api_format: str
    
@dataclass
class IndexInfo:
    symbol: str
    zerodha_symbol: str
    exchange: str
    segment: str
    instrument_token: Optional[int]
    api_format: str
    maps_to_sector: Optional[str]
    description: str

@dataclass
class ETFInfo:
    symbol: str
    api_format: str
    fund_house: Optional[str]
    tracks_index: str
    description: Optional[str]

@dataclass
class SectorInfo:
    description: str
    index: Optional[str]
    index_api_format: Optional[str]
    etf: Optional[str]

class Universe:
    """Parser for enhanced universe.json"""
    
    def __init__(self, filepath: str = "universe.json"):
        with open(filepath) as f:
            self._data = json.load(f)
        self._stocks_cache: Dict[str, Stock] = {}
        self._build_stock_cache()
    
    def _build_stock_cache(self):
        for universe in self._data.get("universes", {}).values():
            for s in universe.get("stocks", []):
                stock = Stock(**s)
                self._stocks_cache[stock.ticker] = stock
    
    @property
    def metadata(self) -> dict:
        return self._data.get("metadata", {})
    
    @property
    def benchmark(self) -> IndexInfo:
        b = self._data["benchmark"]
        return IndexInfo(
            symbol=b["symbol"],
            zerodha_symbol=b["zerodha_symbol"],
            exchange=b["exchange"],
            segment=b["segment"],
            instrument_token=b.get("instrument_token"),
            api_format=b["api_format"],
            maps_to_sector=None,
            description=b.get("description", "")
        )
    
    def get_all_stocks(self) -> List[Stock]:
        """Returns all 201 stocks from both universes"""
        return list(self._stocks_cache.values())
    
    def get_stocks_by_sector(self, sector: str) -> List[Stock]:
        """Returns stocks belonging to a specific sector"""
        return [s for s in self._stocks_cache.values() if s.sector == sector]
    
    def get_valid_sectors(self, min_stocks: int = 3) -> List[str]:
        """Returns sectors with at least min_stocks for ranking"""
        summary = self._data.get("sector_summary", {})
        return [
            sector for sector, counts in summary.items()
            if counts.get("total", 0) >= min_stocks
        ]
    
    def get_sector_index(self, sector: str) -> Optional[IndexInfo]:
        """Get the sectoral index for a sector"""
        sectors = self._data.get("sectors", {})
        if sector not in sectors:
            return None
        sector_info = sectors[sector]
        if not sector_info.get("index"):
            return None
        
        # Find in sectoral_indices
        for idx_key, idx_data in self._data.get("sectoral_indices", {}).items():
            if idx_data.get("maps_to_sector") == sector:
                return IndexInfo(
                    symbol=idx_data["symbol"],
                    zerodha_symbol=idx_data.get("zerodha_symbol", idx_data["symbol"]),
                    exchange=idx_data.get("exchange", "NSE"),
                    segment=idx_data.get("segment", "INDICES"),
                    instrument_token=idx_data.get("instrument_token"),
                    api_format=idx_data["api_format"],
                    maps_to_sector=sector,
                    description=idx_data.get("description", "")
                )
        return None
    
    def get_sector_etf(self, sector: str) -> Optional[ETFInfo]:
        """Get the primary ETF for a sector"""
        etfs = self._data.get("sectoral_etfs", {})
        if sector not in etfs:
            return None
        etf_data = etfs[sector].get("primary_etf")
        if not etf_data:
            return None
        return ETFInfo(
            symbol=etf_data["symbol"],
            api_format=etf_data["api_format"],
            fund_house=etf_data.get("fund_house"),
            tracks_index=etf_data.get("tracks_index", ""),
            description=etf_data.get("description")
        )
    
    def get_hedge(self, hedge_type: str) -> Optional[dict]:
        """Get hedge instrument (gold, cash, international, silver)"""
        return self._data.get("hedges", {}).get(hedge_type)
    
    def get_api_symbols(self, tickers: List[str]) -> List[str]:
        """Convert tickers to Zerodha API format"""
        return [
            self._stocks_cache[t].api_format 
            for t in tickers 
            if t in self._stocks_cache
        ]
    
    def get_stock(self, ticker: str) -> Optional[Stock]:
        return self._stocks_cache.get(ticker)
```

## 4.2 Instrument Token Mapper

```python
# fortress/instruments.py

from typing import Dict, Optional, List
import pandas as pd
from functools import lru_cache
from .universe import Universe

class InstrumentMapper:
    """
    Maps symbols to Zerodha instrument tokens.
    Pre-loads index tokens from universe.json to avoid API lookups.
    """
    
    # Pre-resolved tokens from universe.json (never change)
    INDEX_TOKENS = {
        "NIFTY 50": 256265,
        "NIFTY 100": 260617,
        "NIFTY JUNIOR": 260361,
        "NIFTY MIDCAP 100": 256777,
        "NIFTY MIDCAP 50": 260873,
        "INDIA VIX": 264969,
        "NIFTY BANK": 260105,
        "NIFTY IT": 259849,
        "NIFTY PHARMA": 262409,
        "NIFTY AUTO": 263433,
        "NIFTY METAL": 263689,
        "NIFTY ENERGY": 261641,
        "NIFTY FMCG": 261897,
        "NIFTY REALTY": 261129,
        "NIFTY MEDIA": 263945,
        "NIFTY CONSUMPTION": 257545,
        "NIFTY PSU BANK": 262921,
        "NIFTY COMMODITIES": 257289,
        "NIFTY SERV SECTOR": 263177,
        "NIFTY PSE": 262665,
    }
    
    def __init__(self, kite, universe: Universe):
        self.kite = kite
        self.universe = universe
        self._instrument_df: Optional[pd.DataFrame] = None
        self._token_cache: Dict[str, int] = dict(self.INDEX_TOKENS)
    
    def load_instruments(self):
        """Load instrument dump from Kite - call once daily"""
        instruments = self.kite.instruments("NSE")
        self._instrument_df = pd.DataFrame(instruments)
        
        # Pre-populate cache for all universe stocks
        for stock in self.universe.get_all_stocks():
            match = self._instrument_df[
                (self._instrument_df["tradingsymbol"] == stock.zerodha_symbol) &
                (self._instrument_df["segment"] == "NSE")
            ]
            if not match.empty:
                self._token_cache[stock.zerodha_symbol] = int(match.iloc[0]["instrument_token"])
    
    def get_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for a symbol"""
        # Check cache first (includes pre-resolved indices)
        if symbol in self._token_cache:
            return self._token_cache[symbol]
        
        # Fallback to lookup
        if self._instrument_df is None:
            self.load_instruments()
        
        match = self._instrument_df[
            self._instrument_df["tradingsymbol"] == symbol
        ]
        if not match.empty:
            token = int(match.iloc[0]["instrument_token"])
            self._token_cache[symbol] = token
            return token
        return None
    
    def get_tokens_bulk(self, symbols: List[str]) -> Dict[str, int]:
        """Get tokens for multiple symbols efficiently"""
        result = {}
        missing = []
        
        for symbol in symbols:
            if symbol in self._token_cache:
                result[symbol] = self._token_cache[symbol]
            else:
                missing.append(symbol)
        
        if missing and self._instrument_df is not None:
            for symbol in missing:
                token = self.get_token(symbol)
                if token:
                    result[symbol] = token
        
        return result
    
    def get_api_format(self, symbol: str, exchange: str = "NSE") -> str:
        """Convert symbol to API format (e.g., NSE:RELIANCE)"""
        return f"{exchange}:{symbol}"
```

## 4.3 Market Data Layer

```python
# fortress/market_data.py

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from functools import lru_cache

class MarketDataProvider:
    """
    Fetches and caches OHLC data from Zerodha Kite API.
    Rate limited: max 3 requests/second.
    """
    
    def __init__(self, kite, instrument_mapper):
        self.kite = kite
        self.mapper = instrument_mapper
        self._cache: Dict[str, pd.DataFrame] = {}
    
    @rate_limit(calls=3, period=1)  # 3 calls per second
    def get_historical(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = "day"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data.
        
        Args:
            symbol: Trading symbol (e.g., "RELIANCE" or "NIFTY 50")
            from_date: Start date
            to_date: End date  
            interval: "minute", "3minute", "5minute", "day", "week", "month"
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        cache_key = f"{symbol}_{from_date}_{to_date}_{interval}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        token = self.mapper.get_token(symbol)
        if token is None:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        data = self.kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
        
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        self._cache[cache_key] = df
        return df
    
    def get_ltp(self, symbols: List[str]) -> Dict[str, float]:
        """Get last traded prices for multiple symbols"""
        api_symbols = [self.mapper.get_api_format(s) for s in symbols]
        quotes = self.kite.ltp(api_symbols)
        
        return {
            s: quotes[self.mapper.get_api_format(s)]["last_price"]
            for s in symbols
            if self.mapper.get_api_format(s) in quotes
        }
    
    def get_ohlc(self, symbols: List[str]) -> Dict[str, dict]:
        """Get OHLC snapshot for multiple symbols"""
        api_symbols = [self.mapper.get_api_format(s) for s in symbols]
        return self.kite.ohlc(api_symbols)
```

---

# 5. SECTOR ROTATION ENGINE

## 5.1 Risk-Adjusted Relative Velocity (RRV)

The core ranking metric for sectors:

```
RRV = (Annualized Return) / (Annualized Volatility)
```

Where:
- **Annualized Return** = `(ln(P_t) - ln(P_{t-126})) * (252/126)` (6-month log return, annualized)
- **Annualized Volatility** = `std(daily_log_returns, window=63) * sqrt(252)`

### 5.1.1 Sector RRV Calculation

```python
# fortress/sector_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class SectorScore:
    sector: str
    rrv: float
    return_6m: float
    volatility: float
    index_symbol: str
    etf_symbol: Optional[str]
    stock_count: int
    rank: int

class SectorRotationEngine:
    """
    Ranks sectors by RRV and selects top stocks within chosen sectors.
    """
    
    LOOKBACK_RETURN = 126      # 6 months in trading days
    LOOKBACK_VOLATILITY = 63   # 3 months for volatility
    TRADING_DAYS_YEAR = 252
    MIN_STOCKS_PER_SECTOR = 3
    
    def __init__(self, universe: Universe, market_data: MarketDataProvider):
        self.universe = universe
        self.market_data = market_data
    
    def calculate_rrv(self, prices: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate RRV for a price series.
        
        Returns: (rrv, annualized_return, annualized_volatility)
        """
        if len(prices) < self.LOOKBACK_RETURN:
            return (0.0, 0.0, 0.0)
        
        # Log returns
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # 6-month return (annualized)
        period_return = np.log(prices.iloc[-1] / prices.iloc[-self.LOOKBACK_RETURN])
        annualized_return = period_return * (self.TRADING_DAYS_YEAR / self.LOOKBACK_RETURN)
        
        # Volatility (last 3 months, annualized)
        recent_returns = log_returns.iloc[-self.LOOKBACK_VOLATILITY:]
        volatility = recent_returns.std() * np.sqrt(self.TRADING_DAYS_YEAR)
        
        # RRV
        rrv = annualized_return / volatility if volatility > 0 else 0.0
        
        return (rrv, annualized_return, volatility)
    
    def rank_sectors(self, as_of_date: datetime) -> List[SectorScore]:
        """
        Rank all valid sectors by RRV.
        
        Returns: List of SectorScore sorted by RRV descending
        """
        valid_sectors = self.universe.get_valid_sectors(self.MIN_STOCKS_PER_SECTOR)
        scores = []
        
        for sector in valid_sectors:
            index_info = self.universe.get_sector_index(sector)
            if not index_info:
                continue
            
            # Fetch index price history
            from_date = as_of_date - timedelta(days=self.LOOKBACK_RETURN * 2)
            prices = self.market_data.get_historical(
                symbol=index_info.symbol,
                from_date=from_date,
                to_date=as_of_date,
                interval="day"
            )["close"]
            
            rrv, ret, vol = self.calculate_rrv(prices)
            
            etf_info = self.universe.get_sector_etf(sector)
            stocks = self.universe.get_stocks_by_sector(sector)
            
            scores.append(SectorScore(
                sector=sector,
                rrv=rrv,
                return_6m=ret,
                volatility=vol,
                index_symbol=index_info.symbol,
                etf_symbol=etf_info.symbol if etf_info else None,
                stock_count=len(stocks),
                rank=0  # Will be set after sorting
            ))
        
        # Sort by RRV descending
        scores.sort(key=lambda x: x.rrv, reverse=True)
        
        # Assign ranks
        for i, score in enumerate(scores):
            score.rank = i + 1
        
        return scores
    
    def select_top_sectors(
        self, 
        scores: List[SectorScore], 
        n: int = 3,
        min_rrv: float = 0.5
    ) -> List[SectorScore]:
        """
        Select top N sectors for allocation.
        
        Args:
            scores: Ranked sector scores
            n: Number of sectors to select
            min_rrv: Minimum RRV threshold
        
        Returns: Top sectors meeting criteria
        """
        qualified = [s for s in scores if s.rrv >= min_rrv]
        return qualified[:n]
```

## 5.2 Stock Momentum Scoring

```python
@dataclass
class StockScore:
    ticker: str
    name: str
    sector: str
    momentum_score: float
    return_3m: float
    return_6m: float
    volatility: float
    relative_strength: float  # vs sector index
    rank_in_sector: int

class StockSelector:
    """
    Selects top stocks within each chosen sector based on momentum.
    """
    
    LOOKBACK_3M = 63
    LOOKBACK_6M = 126
    TRADING_DAYS = 252
    
    def __init__(self, universe: Universe, market_data: MarketDataProvider):
        self.universe = universe
        self.market_data = market_data
    
    def calculate_momentum_score(
        self, 
        stock_prices: pd.Series,
        sector_prices: pd.Series
    ) -> dict:
        """
        Calculate composite momentum score for a stock.
        
        Score = 0.4 * (3m_return) + 0.4 * (6m_return) + 0.2 * (relative_strength)
        """
        if len(stock_prices) < self.LOOKBACK_6M:
            return None
        
        # Returns
        ret_3m = (stock_prices.iloc[-1] / stock_prices.iloc[-self.LOOKBACK_3M]) - 1
        ret_6m = (stock_prices.iloc[-1] / stock_prices.iloc[-self.LOOKBACK_6M]) - 1
        
        # Volatility
        log_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()
        volatility = log_returns.std() * np.sqrt(self.TRADING_DAYS)
        
        # Relative strength vs sector
        sector_ret_6m = (sector_prices.iloc[-1] / sector_prices.iloc[-self.LOOKBACK_6M]) - 1
        relative_strength = ret_6m - sector_ret_6m
        
        # Composite score
        momentum_score = (0.4 * ret_3m) + (0.4 * ret_6m) + (0.2 * relative_strength)
        
        return {
            "momentum_score": momentum_score,
            "return_3m": ret_3m,
            "return_6m": ret_6m,
            "volatility": volatility,
            "relative_strength": relative_strength
        }
    
    def select_stocks_for_sector(
        self,
        sector: str,
        as_of_date: datetime,
        top_n: int = 5
    ) -> List[StockScore]:
        """
        Select top N stocks from a sector by momentum score.
        """
        stocks = self.universe.get_stocks_by_sector(sector)
        index_info = self.universe.get_sector_index(sector)
        
        # Get sector index prices
        from_date = as_of_date - timedelta(days=self.LOOKBACK_6M * 2)
        sector_prices = self.market_data.get_historical(
            symbol=index_info.symbol,
            from_date=from_date,
            to_date=as_of_date,
            interval="day"
        )["close"]
        
        scores = []
        for stock in stocks:
            try:
                prices = self.market_data.get_historical(
                    symbol=stock.zerodha_symbol,
                    from_date=from_date,
                    to_date=as_of_date,
                    interval="day"
                )["close"]
                
                metrics = self.calculate_momentum_score(prices, sector_prices)
                if metrics:
                    scores.append(StockScore(
                        ticker=stock.ticker,
                        name=stock.name,
                        sector=sector,
                        rank_in_sector=0,
                        **metrics
                    ))
            except Exception as e:
                # Log and skip stocks with data issues
                continue
        
        # Sort by momentum score
        scores.sort(key=lambda x: x.momentum_score, reverse=True)
        
        # Assign ranks
        for i, score in enumerate(scores):
            score.rank_in_sector = i + 1
        
        # Handle sectors with < top_n stocks
        return scores[:min(top_n, len(scores))]
```

---

# 6. RISK MANAGEMENT

## 6.1 Risk Governor

```python
# fortress/risk_governor.py

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class RiskRegime(Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    DEFENSIVE = "defensive"
    HALT = "halt"

@dataclass
class RiskLimits:
    max_single_position: float = 0.08      # 8% of capital
    hard_max_position: float = 0.12        # 12% absolute max
    max_sector_exposure: float = 0.35      # 35% in one sector
    hard_max_sector: float = 0.45          # 45% absolute max
    max_drawdown_warning: float = 0.15     # 15% triggers caution
    max_drawdown_halt: float = 0.25        # 25% halts trading
    vix_caution_threshold: float = 20.0    # VIX > 20 = caution
    vix_defensive_threshold: float = 25.0  # VIX > 25 = defensive
    daily_loss_limit: float = 0.03         # 3% daily loss = stop
    max_positions: int = 20                # Maximum 20 holdings

class RiskGovernor:
    """
    Validates all portfolio actions against risk limits.
    Has override authority over all allocation decisions.
    """
    
    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self._peak_value: float = 0.0
        self._day_start_value: float = 0.0
    
    def assess_regime(
        self,
        current_drawdown: float,
        vix: float
    ) -> RiskRegime:
        """
        Determine current risk regime based on drawdown and VIX.
        """
        if current_drawdown >= self.limits.max_drawdown_halt:
            return RiskRegime.HALT
        
        if vix >= self.limits.vix_defensive_threshold:
            return RiskRegime.DEFENSIVE
        
        if (current_drawdown >= self.limits.max_drawdown_warning or 
            vix >= self.limits.vix_caution_threshold):
            return RiskRegime.CAUTION
        
        return RiskRegime.NORMAL
    
    def validate_position_size(
        self,
        symbol: str,
        proposed_value: float,
        portfolio_value: float
    ) -> Tuple[bool, str, float]:
        """
        Validate proposed position size.
        
        Returns: (is_valid, reason, adjusted_value)
        """
        proposed_pct = proposed_value / portfolio_value
        
        if proposed_pct > self.limits.hard_max_position:
            adjusted = portfolio_value * self.limits.hard_max_position
            return (False, f"Exceeds hard limit {self.limits.hard_max_position:.0%}", adjusted)
        
        if proposed_pct > self.limits.max_single_position:
            adjusted = portfolio_value * self.limits.max_single_position
            return (False, f"Exceeds soft limit {self.limits.max_single_position:.0%}", adjusted)
        
        return (True, "OK", proposed_value)
    
    def validate_sector_exposure(
        self,
        sector: str,
        current_exposure: float,
        proposed_addition: float,
        portfolio_value: float
    ) -> Tuple[bool, str, float]:
        """
        Validate sector concentration.
        """
        total_exposure = (current_exposure + proposed_addition) / portfolio_value
        
        if total_exposure > self.limits.hard_max_sector:
            max_addition = (self.limits.hard_max_sector * portfolio_value) - current_exposure
            return (False, f"Sector hard limit {self.limits.hard_max_sector:.0%}", max(0, max_addition))
        
        if total_exposure > self.limits.max_sector_exposure:
            max_addition = (self.limits.max_sector_exposure * portfolio_value) - current_exposure
            return (False, f"Sector soft limit {self.limits.max_sector_exposure:.0%}", max(0, max_addition))
        
        return (True, "OK", proposed_addition)
    
    def check_daily_loss(
        self,
        current_value: float,
        day_start_value: float
    ) -> Tuple[bool, str]:
        """
        Check if daily loss limit breached.
        """
        if day_start_value <= 0:
            return (True, "OK")
        
        daily_return = (current_value - day_start_value) / day_start_value
        
        if daily_return <= -self.limits.daily_loss_limit:
            return (False, f"Daily loss {daily_return:.2%} exceeds limit")
        
        return (True, "OK")
    
    def get_position_multiplier(self, regime: RiskRegime) -> float:
        """
        Get position size multiplier based on regime.
        """
        multipliers = {
            RiskRegime.NORMAL: 1.0,
            RiskRegime.CAUTION: 0.75,
            RiskRegime.DEFENSIVE: 0.50,
            RiskRegime.HALT: 0.0
        }
        return multipliers.get(regime, 1.0)
```

---

# 7. ORDER MANAGEMENT

## 7.1 Order Manager

```python
# fortress/order_manager.py

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import time

class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    PLACED = "PLACED"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

@dataclass
class Order:
    symbol: str
    order_type: OrderType
    quantity: int
    price: Optional[float]
    product: str = "CNC"  # Delivery
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    reason: Optional[str] = None
    tag: Optional[str] = None  # For idempotency

class OrderManager:
    """
    Manages order placement with safety checks.
    Dry-run by default - requires explicit enable for live orders.
    """
    
    def __init__(self, kite, risk_governor: RiskGovernor, dry_run: bool = True):
        self.kite = kite
        self.risk_governor = risk_governor
        self.dry_run = dry_run
        self._pending_orders: List[Order] = []
    
    def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        quantity: int,
        price: Optional[float] = None,
        tag: Optional[str] = None
    ) -> Order:
        """Create order without placing it."""
        return Order(
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            price=price,
            tag=tag or f"{symbol}_{int(time.time())}"
        )
    
    def validate_order(self, order: Order) -> Tuple[bool, str]:
        """Validate order before placement."""
        
        # Check quantity
        if order.quantity <= 0:
            return (False, "Invalid quantity")
        
        # Check margin for buy orders
        if order.order_type == OrderType.BUY:
            margins = self.kite.margins()
            available = margins["equity"]["available"]["live_balance"]
            required = order.quantity * (order.price or 0)
            if required > available:
                return (False, f"Insufficient margin: need {required}, have {available}")
        
        return (True, "OK")
    
    def place_order(self, order: Order) -> Order:
        """
        Place order via Kite API.
        Returns updated order with order_id and status.
        """
        # Validate first
        is_valid, reason = self.validate_order(order)
        if not is_valid:
            order.status = OrderStatus.REJECTED
            order.reason = reason
            return order
        
        if self.dry_run:
            order.status = OrderStatus.PENDING
            order.reason = "DRY RUN - not placed"
            order.order_id = f"DRY_{order.tag}"
            return order
        
        try:
            transaction = (
                self.kite.TRANSACTION_TYPE_BUY 
                if order.order_type == OrderType.BUY 
                else self.kite.TRANSACTION_TYPE_SELL
            )
            
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=order.symbol,
                transaction_type=transaction,
                quantity=order.quantity,
                product=self.kite.PRODUCT_CNC,
                order_type=self.kite.ORDER_TYPE_MARKET,
                tag=order.tag
            )
            
            order.order_id = order_id
            order.status = OrderStatus.PLACED
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reason = str(e)
        
        return order
    
    def place_orders_batch(
        self,
        orders: List[Order],
        sells_first: bool = True
    ) -> List[Order]:
        """
        Place multiple orders with sells before buys.
        """
        if sells_first:
            sell_orders = [o for o in orders if o.order_type == OrderType.SELL]
            buy_orders = [o for o in orders if o.order_type == OrderType.BUY]
            ordered = sell_orders + buy_orders
        else:
            ordered = orders
        
        results = []
        for order in ordered:
            result = self.place_order(order)
            results.append(result)
            time.sleep(0.5)  # Rate limiting
        
        return results
```

---

# 8. ZERODHA INTEGRATION

## 8.1 Authentication Flow

```python
# fortress/auth.py

from kiteconnect import KiteConnect
import webbrowser
from pathlib import Path
import json

class ZerodhaAuth:
    """
    Handles Zerodha Kite Connect authentication.
    
    Flow:
    1. User provides API key and secret in config
    2. System opens login URL in browser
    3. User logs in and gets redirected with request_token
    4. User pastes request_token in console
    5. System exchanges for access_token
    6. Access token cached for the day
    """
    
    TOKEN_CACHE_FILE = ".kite_token_cache.json"
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        self._access_token: Optional[str] = None
    
    def get_login_url(self) -> str:
        """Get the Kite login URL."""
        return self.kite.login_url()
    
    def authenticate(self, request_token: str) -> str:
        """
        Exchange request_token for access_token.
        
        Args:
            request_token: Token from redirect URL after login
        
        Returns:
            access_token
        """
        data = self.kite.generate_session(
            request_token=request_token,
            api_secret=self.api_secret
        )
        self._access_token = data["access_token"]
        self.kite.set_access_token(self._access_token)
        
        # Cache token
        self._save_token_cache()
        
        return self._access_token
    
    def _save_token_cache(self):
        """Save access token to cache file."""
        from datetime import datetime
        cache = {
            "access_token": self._access_token,
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        Path(self.TOKEN_CACHE_FILE).write_text(json.dumps(cache))
    
    def _load_token_cache(self) -> Optional[str]:
        """Load access token from cache if valid for today."""
        from datetime import datetime
        try:
            cache = json.loads(Path(self.TOKEN_CACHE_FILE).read_text())
            if cache.get("date") == datetime.now().strftime("%Y-%m-%d"):
                return cache.get("access_token")
        except:
            pass
        return None
    
    def login_interactive(self) -> KiteConnect:
        """
        Interactive login flow for CLI.
        
        Returns:
            Authenticated KiteConnect instance
        """
        # Check cache first
        cached_token = self._load_token_cache()
        if cached_token:
            print("Using cached access token from today")
            self.kite.set_access_token(cached_token)
            self._access_token = cached_token
            return self.kite
        
        # Fresh login
        login_url = self.get_login_url()
        print(f"\nOpening Zerodha login in browser...")
        print(f"URL: {login_url}\n")
        webbrowser.open(login_url)
        
        print("After logging in, you will be redirected to your redirect URL.")
        print("Copy the 'request_token' parameter from the URL.\n")
        
        request_token = input("Paste request_token here: ").strip()
        
        if not request_token:
            raise ValueError("No request_token provided")
        
        self.authenticate(request_token)
        print("Authentication successful!\n")
        
        return self.kite
```

## 8.2 Rate Limiting Decorator

```python
# fortress/utils.py

import time
from functools import wraps
from collections import deque

def rate_limit(calls: int, period: float):
    """
    Decorator to rate limit function calls.
    
    Args:
        calls: Maximum calls allowed
        period: Time period in seconds
    """
    timestamps = deque()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old timestamps
            while timestamps and timestamps[0] < now - period:
                timestamps.popleft()
            
            # Check limit
            if len(timestamps) >= calls:
                sleep_time = timestamps[0] + period - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            timestamps.append(time.time())
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
```

---

# 9. CLI INTERFACE

## 9.1 Main CLI Structure

```python
# fortress/cli.py

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

@click.group()
@click.option("--config", "-c", default="config.yaml", help="Config file path")
@click.option("--dry-run/--live", default=True, help="Dry run mode (default: dry-run)")
@click.pass_context
def cli(ctx, config, dry_run):
    """FORTRESS SIDEWAYS - Sector Rotation Strategy CLI"""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["dry_run"] = dry_run

@cli.command()
@click.pass_context
def login(ctx):
    """Authenticate with Zerodha"""
    config = load_config(ctx.obj["config_path"])
    auth = ZerodhaAuth(config.api_key, config.api_secret)
    kite = auth.login_interactive()
    console.print("[green]✓ Logged in successfully[/green]")

@cli.command()
@click.pass_context
def status(ctx):
    """Show current portfolio status and regime"""
    # ... implementation

@cli.command()
@click.pass_context
def scan(ctx):
    """Scan and rank sectors by RRV"""
    # ... implementation

@cli.command()
@click.option("--top-n", default=3, help="Number of sectors to allocate")
@click.option("--confirm", is_flag=True, help="Confirm and execute orders")
@click.pass_context
def rebalance(ctx, top_n, confirm):
    """Calculate and execute rebalance"""
    # ... implementation

@cli.command()
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--capital", default=1600000, help="Initial capital")
@click.pass_context
def backtest(ctx, start, end, capital):
    """Run historical backtest"""
    # ... implementation

@cli.command()
@click.pass_context
def holdings(ctx):
    """Show current holdings"""
    # ... implementation

if __name__ == "__main__":
    cli()
```

## 9.2 CLI Output Examples

```
$ fortress status

╭─────────────────── FORTRESS SIDEWAYS ───────────────────╮
│                                                          │
│  Portfolio Value:  ₹16,42,350  (+2.65%)                 │
│  Cash Available:   ₹1,23,450                            │
│  Risk Regime:      NORMAL                               │
│  VIX:              18.45                                │
│  Drawdown:         -3.2%                                │
│                                                          │
╰──────────────────────────────────────────────────────────╯

$ fortress scan

                    Sector Rankings (RRV)                    
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Rank  ┃ Sector             ┃ RRV    ┃ Return   ┃ Vol      ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ 1     │ IT_SERVICES        │ 2.34   │ +28.5%   │ 12.2%    │
│ 2     │ PHARMA_HEALTHCARE  │ 1.98   │ +22.1%   │ 11.2%    │
│ 3     │ FINANCIAL_SERVICES │ 1.87   │ +24.3%   │ 13.0%    │
│ 4     │ AUTOMOBILE         │ 1.65   │ +19.8%   │ 12.0%    │
│ 5     │ CONSUMER_GOODS     │ 1.42   │ +15.2%   │ 10.7%    │
│ ...   │ ...                │ ...    │ ...      │ ...      │
└───────┴────────────────────┴────────┴──────────┴──────────┘
```

---

# 10. BACKTEST FRAMEWORK

## 10.1 Backtest Engine

```python
# fortress/backtest.py

from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np

@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    initial_capital: float
    rebalance_frequency: str = "weekly"  # "weekly" or "monthly"
    top_sectors: int = 3
    stocks_per_sector: int = 5
    transaction_cost: float = 0.003  # 0.3% (STT + brokerage + slippage)

@dataclass
class BacktestResult:
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: pd.Series
    trades: pd.DataFrame

class BacktestEngine:
    """
    Historical simulation with realistic constraints.
    
    Key features:
    - Look-ahead bias prevention (uses only data available at decision time)
    - Realistic transaction costs
    - Slippage modeling
    - Weekly/monthly rebalance options
    """
    
    def __init__(
        self,
        universe: Universe,
        config: BacktestConfig
    ):
        self.universe = universe
        self.config = config
    
    def run(self) -> BacktestResult:
        """
        Execute backtest simulation.
        """
        # Implementation details...
        pass
    
    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame
    ) -> BacktestResult:
        """Calculate performance metrics."""
        
        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # CAGR
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1
        
        # Sharpe ratio
        daily_returns = equity_curve.pct_change().dropna()
        sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        
        # Max drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = len(trades[trades["pnl"] > 0])
        win_rate = winning_trades / len(trades) if len(trades) > 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            equity_curve=equity_curve,
            trades=trades
        )
```

---

# 11. CONFIGURATION SCHEMA

## 11.1 config.yaml

```yaml
# FORTRESS SIDEWAYS Configuration
# All values have safe defaults - customize as needed

# Zerodha API credentials
zerodha:
  api_key: "your_api_key_here"
  api_secret: "your_api_secret_here"

# Portfolio settings
portfolio:
  initial_capital: 1600000
  max_positions: 20
  
# Sector rotation settings
rotation:
  top_sectors: 3              # Number of sectors to hold
  stocks_per_sector: 5        # Max stocks per sector
  min_sector_stocks: 3        # Min stocks required for sector eligibility
  min_rrv_threshold: 0.5      # Minimum RRV to consider sector
  rebalance_day: "friday"     # Day of week for rebalance
  
# Momentum calculation
momentum:
  lookback_return: 126        # 6 months in trading days
  lookback_volatility: 63     # 3 months for volatility
  trading_days_year: 252

# Risk limits
risk:
  max_single_position: 0.08   # 8% of capital
  hard_max_position: 0.12     # 12% absolute max
  max_sector_exposure: 0.35   # 35% in one sector
  hard_max_sector: 0.45       # 45% absolute max
  max_drawdown_warning: 0.15  # 15% triggers caution
  max_drawdown_halt: 0.25     # 25% halts trading
  daily_loss_limit: 0.03      # 3% daily loss = stop
  vix_caution: 20.0           # VIX > 20 = reduce exposure
  vix_defensive: 25.0         # VIX > 25 = defensive mode

# Transaction costs (for backtest)
costs:
  transaction_cost: 0.003     # 0.3% round trip

# Paths
paths:
  universe_file: "universe.json"
  log_dir: "logs"
  data_cache: ".cache"
```

---

# 12. FILE STRUCTURE

```
fortress-sideways/
├── README.md
├── pyproject.toml
├── config.yaml
├── universe.json              # Enhanced universe with Zerodha symbols
│
├── fortress/
│   ├── __init__.py
│   ├── cli.py                # Click CLI entry point
│   ├── config.py             # Pydantic config models
│   ├── auth.py               # Zerodha authentication
│   ├── universe.py           # Universe parser
│   ├── instruments.py        # Instrument token mapper
│   ├── market_data.py        # OHLC fetching
│   ├── indicators.py         # Momentum calculations
│   ├── sector_engine.py      # Sector ranking
│   ├── stock_selector.py     # Stock selection
│   ├── risk_governor.py      # Risk management
│   ├── portfolio.py          # Holdings tracking
│   ├── rebalancer.py         # Delta calculation
│   ├── order_manager.py      # Order placement
│   ├── backtest.py           # Backtest engine
│   ├── logger.py             # Structured logging
│   └── utils.py              # Helpers, rate limiting
│
├── tests/
│   ├── test_universe.py
│   ├── test_sector_engine.py
│   ├── test_risk_governor.py
│   ├── test_order_manager.py
│   └── test_backtest.py
│
├── scripts/
│   └── update_universe.py    # Refresh universe from NSE
│
└── docs/
    ├── SPEC.md               # This document
    └── CHANGELOG.md
```

---

# 13. INVARIANTS CHECKLIST

## 13.1 Data Integrity (8 invariants)

| # | Invariant | Check |
|---|-----------|-------|
| D1 | Universe JSON validates against schema | Startup |
| D2 | All stocks have valid zerodha_symbol | Startup |
| D3 | All sectoral indices have instrument_token | Startup |
| D4 | No duplicate tickers in universe | Startup |
| D5 | sector_summary totals match actual counts | Startup |
| D6 | Historical data has no gaps > 5 days | Before calculation |
| D7 | Price data is adjusted for corporate actions | Data fetch |
| D8 | All required sectors have index mapping | Before ranking |

## 13.2 Calculation Integrity (7 invariants)

| # | Invariant | Check |
|---|-----------|-------|
| C1 | RRV uses log returns, not simple returns | Unit test |
| C2 | Volatility is annualized (√252) | Unit test |
| C3 | Lookback periods match config | Runtime |
| C4 | Sectors with < min_stocks excluded | Before ranking |
| C5 | Momentum score weights sum to 1.0 | Unit test |
| C6 | Relative strength calculated vs sector, not benchmark | Unit test |
| C7 | No look-ahead bias in backtest | Backtest audit |

## 13.3 Risk Management (10 invariants)

| # | Invariant | Check |
|---|-----------|-------|
| R1 | No position > hard_max_position | Before order |
| R2 | No sector > hard_max_sector | Before order |
| R3 | Daily loss triggers halt | Real-time |
| R4 | Drawdown > 25% halts all trading | Before any action |
| R5 | VIX regime affects position sizing | Before rebalance |
| R6 | Margin check before buy orders | Before order |
| R7 | Position count ≤ max_positions | Before order |
| R8 | Risk governor has veto power | All decisions |
| R9 | Sells execute before buys | Order sequence |
| R10 | No order placed without validation | Order manager |

## 13.4 Order Management (8 invariants)

| # | Invariant | Check |
|---|-----------|-------|
| O1 | Dry-run is default mode | Config |
| O2 | Live orders require explicit --live flag | CLI |
| O3 | All orders have unique tags | Order creation |
| O4 | Order status is tracked to completion | Post-order |
| O5 | Failed orders are logged with reason | Exception handler |
| O6 | Rate limit: max 3 orders/second | Order manager |
| O7 | CNC product type for all positions | Order creation |
| O8 | NSE exchange for all orders | Order creation |

## 13.5 Operational (7 invariants)

| # | Invariant | Check |
|---|-----------|-------|
| P1 | Authentication required before any API call | Session check |
| P2 | Token cache expires daily | Token load |
| P3 | Holiday calendar checked before trading | Before rebalance |
| P4 | All decisions logged with timestamp | Logger |
| P5 | Config changes require restart | Runtime |
| P6 | Backtest and live use same calculation code | Code structure |
| P7 | No hardcoded secrets in code | Code review |

---

# END OF SPECIFICATION

**Total Invariants: 40**

This specification is designed to be:
1. **Complete** - All components specified with interfaces
2. **Unambiguous** - Explicit formulas, thresholds, and logic
3. **Testable** - 40 invariants provide verification criteria
4. **Implementable** - Ready for Claude Code or developer execution

**Next Steps:**
1. Implement modules in order: universe → instruments → market_data → indicators → sector_engine → risk_governor → order_manager → cli
2. Write tests for each invariant
3. Run backtest to validate strategy
4. Paper trade for 2-4 weeks before live deployment
