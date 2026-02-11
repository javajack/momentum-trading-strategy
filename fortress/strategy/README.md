# FORTRESS Momentum Strategies

This document describes the momentum strategies available in FORTRESS. Each strategy implements the `BaseStrategy` protocol and can be used interchangeably with the backtest engine and CLI.

---

## Table of Contents

1. [Strategy Architecture](#strategy-architecture)
2. [Adaptive Dual Momentum (`dual_momentum`)](#adaptive-dual-momentum)
3. [Adding New Strategies](#adding-new-strategies)

---

## Strategy Architecture

All strategies implement the `BaseStrategy` abstract class with these core methods:

| Method | Purpose |
|--------|---------|
| `rank_stocks()` | Score and rank all stocks in universe |
| `select_portfolio()` | Select target portfolio from ranked stocks |
| `calculate_weights()` | Calculate position weights |
| `check_exit_triggers()` | Check if positions should be exited |
| `get_stop_loss_config()` | Get stop loss parameters for a position |

Strategies are registered via `StrategyRegistry` and accessed by name:

```python
from fortress.strategy import StrategyRegistry

strategy = StrategyRegistry.get("dual_momentum", config)
```

---

## Adaptive Dual Momentum

**Registry Name:** `dual_momentum`
**Class:** `AdaptiveDualMomentumStrategy`
**File:** `adaptive_dual_momentum.py`

### Overview

Research-backed momentum strategy combining:
- **Dual Momentum** (Antonacci): Absolute + Relative momentum
- **Regime Adaptation**: Adjusts parameters based on market stress
- **Recovery Modes**: Captures V-shaped rebounds after corrections
- **Volatility Targeting**: Sizes positions inversely to volatility
- **Tiered Stops**: Lets winners run with progressively wider stops

### Research Foundation

| Concept | Source | Benefit |
|---------|--------|---------|
| Dual Momentum | Antonacci (2014) | +440 bps annually vs index |
| Quality + Momentum | AQR Research | 93% outperformance rate |
| Volatility Targeting | Moreira & Muir | Reduces max DD by 6.6%, can double Sharpe |
| Momentum Crash Avoidance | Daniel & Moskowitz | Protects during market reversals |

---

### Entry Rules

A stock must pass ALL of these filters to be eligible for entry:

| Filter | Default | Description |
|--------|---------|-------------|
| **Absolute Momentum** | NMS > 0 | Stock has positive momentum |
| **Relative Momentum** | RS > 1.05 | Stock beats benchmark by 5% |
| **Trend Filter** | Price > 50 EMA | Uptrend confirmation |
| **Liquidity** | Turnover > Rs 1 Cr | Sufficient trading volume |
| **52W High Proximity** | > 85% | Near recent highs (adaptive) |
| **Above 200 SMA** | Yes | Long-term trend (can be skipped in crash recovery) |

**Regime Adaptation:**
- Bullish (stress < 0.3): Filters relaxed by ~10-15%
- Defensive (stress > 0.6): Filters tightened by ~5%
- Recovery modes: Additional relaxation for capturing rebounds

---

### Scoring Formula

```
Score = NMS × (1 + rs_weight × (RS - 1))
```

Where:
- **NMS**: Normalized Momentum Score (6M/12M returns adjusted for volatility)
- **RS**: Relative Strength vs benchmark (composite of 21d/63d/126d)
- **rs_weight**: Default 0.25 (configurable)

**Effect:**
- RS = 1.10 (10% outperformance) → Score boosted by 2.5%
- RS = 0.90 (10% underperformance) → Score penalized by 2.5%

---

### Exit Rules

Positions are exited when ANY of these triggers fire:

#### 1. Hard Stop Loss
- **Trigger:** Loss exceeds initial stop (default -15%)
- **Urgency:** Immediate
- **Adaptive:** Wider in bullish (×1.25), tighter in defensive (×0.85)

#### 2. Tiered Trailing Stop
Let winners run with progressively wider stops:

| Tier | Gain Range | Trailing Stop |
|------|------------|---------------|
| Tier 1 | 0% - 8% | 12% from peak |
| Tier 2 | 8% - 20% | 14% from peak |
| Tier 3 | 20% - 50% | 16% from peak |
| Tier 4 | > 50% | 22% from peak |

**Activation:** Trailing stop only activates after 8% gain from entry.

#### 3. Trend Break
- **Trigger:** Price falls below 50 EMA with buffer
- **Buffer:** 3% default, adaptive (5% in bullish, 0% in defensive)
- **Confirmation:** 2-3 days below buffer required
- **Urgency:** Next rebalance

#### 4. RS Floor
- **Trigger:** Relative Strength drops below 0.95
- **Meaning:** Stock underperforming benchmark
- **Urgency:** Next rebalance

---

### Position Sizing

**Volatility Targeting:**
```
Position Weight = Base Weight × (Target Volatility / Stock Volatility)
```

- High volatility stocks → Smaller positions → Same risk exposure
- Low volatility stocks → Larger positions → Capture more upside

**Constraints:**
- Max single position: 12% (configurable)
- Min single position: 3% (configurable)
- Max sector exposure: 30% (configurable)

---

### Recovery Modes

The strategy detects three types of recovery conditions:

#### 1. Bull Recovery Mode
**Trigger:** VIX declining from elevated levels (>20) + positive momentum
**Duration:** 60 days
**Effect:** Filter relaxation of 25% for capturing V-shaped rebounds

#### 2. General Recovery Mode
**Trigger:** Portfolio drawdown exceeds -7%
**Duration:** 60 days
**Effect:** Filter relaxation of 25% for re-entering after corrections

#### 3. Crash Recovery Mode
**Trigger:** VIX spike above 50 (extreme fear)
**Duration:** 90 days
**Effect:** Most aggressive relaxation
- 52W high requirement: 85% → 64%
- Allow entries 15% below 50 EMA
- Skip 200 SMA filter entirely

---

### Regime Detection

The strategy uses a stress score (0-1) derived from:

| Factor | Weight | Source |
|--------|--------|--------|
| VIX level | 25% | Current VIX vs thresholds |
| Nifty position | 30% | Multi-timeframe (21d/63d/126d) |
| Market returns | 20% | 1M and 3M returns |
| Momentum signals | 25% | Position momentum + VIX recovery |

**Stress Score Interpretation:**
- 0.0 - 0.3: Bullish (relaxed filters, wider stops)
- 0.3 - 0.6: Neutral (base parameters)
- 0.6 - 1.0: Defensive (strict filters, tighter stops)

---

### Configuration

All parameters are configurable via `config.yaml`:

```yaml
strategy_dual_momentum:
  # Dual Momentum
  min_rs_threshold: 1.05      # RS required for entry
  rs_weight: 0.25             # RS impact on score
  rs_exit_threshold: 0.95     # RS floor for exit

  # Regime Thresholds
  vix_bullish_threshold: 18.0
  vix_defensive_threshold: 25.0

  # Feature Toggles
  use_adaptive_parameters: true
  use_recovery_modes: true
  use_tiered_stops: true

  # Stop Loss Tiers
  tier1_trailing: 0.12
  tier2_trailing: 0.14
  tier3_trailing: 0.16
  tier4_trailing: 0.22

  # Recovery Settings
  recovery_drawdown_trigger: -0.07
  recovery_duration_days: 60
  crash_recovery_vix_trigger: 50.0
```

---

### Backtest Usage

```python
from fortress.backtest import BacktestEngine, BacktestConfig
from fortress.universe import Universe

config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
    strategy_name="dual_momentum",  # Use this strategy
    target_positions=12,
    max_stocks_per_sector=3,
)

engine = BacktestEngine(universe, historical_data, config)
result = engine.run()
```

---

## Adding New Strategies

To add a new strategy:

### 1. Create the Strategy File

```python
# fortress/strategy/my_strategy.py

from .base import BaseStrategy, StockScore, ExitSignal, StopLossConfig
from .registry import StrategyRegistry

class MyStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "my_strategy"

    @property
    def description(self) -> str:
        return "Description for CLI display"

    def rank_stocks(self, ...):
        # Implement scoring logic
        pass

    def select_portfolio(self, ...):
        # Implement selection logic
        pass

    # ... implement other required methods

# Auto-register
StrategyRegistry.register(MyStrategy)
```

### 2. Add to Package Exports

```python
# fortress/strategy/__init__.py

from .my_strategy import MyStrategy

__all__ = [
    # ... existing exports
    "MyStrategy",
]
```

### 3. Add Configuration (Optional)

```python
# fortress/config.py

class MyStrategyConfig(BaseModel):
    """Configuration for MyStrategy."""
    param1: float = Field(default=1.0, description="...")
    param2: int = Field(default=10, description="...")

class Config(BaseModel):
    # ...
    strategy_my_strategy: MyStrategyConfig = Field(
        default_factory=MyStrategyConfig
    )
```

### 4. Document the Strategy

Add a section to this README following the same structure:
- Overview
- Entry Rules
- Scoring Formula
- Exit Rules
- Position Sizing
- Configuration

---

## Strategy Comparison Template

When adding multiple strategies, use this template for comparison:

| Aspect | Strategy A | Strategy B |
|--------|------------|------------|
| **Core Approach** | ... | ... |
| **Entry Filters** | ... | ... |
| **Scoring** | ... | ... |
| **Exit Rules** | ... | ... |
| **Position Sizing** | ... | ... |
| **Regime Adaptation** | ... | ... |
| **Best Market Conditions** | ... | ... |
| **Weaknesses** | ... | ... |
