# FORTRESS MOMENTUM — Improvement Plan

## Problem Statement

The 16-phase backtest (2013-2026) shows +1,207% return / 21.7% CAGR / 0.99 Sharpe overall, but two specific market regimes cause significant alpha destruction:

| Problem Phase | Strategy | NIFTY 50 | Alpha | Root Cause |
|---|---|---|---|---|
| Phase 1: 2013 Consolidation (Sideways) | -11.7% | +1.7% | **-13.4%** | Churn + multiplicative defensive drag |
| Phase 10: NBFC Crisis 2018 (Bearish) | -16.6% | -1.7% | **-14.9%** | Soft sector penalty + stale momentum |
| Phase 8: Demonetization (Recovery) | +1.4% | +8.8% | **-7.4%** | Recovery mode requires portfolio DD |
| Phase 11: 2019 Recovery (Gradual) | +6.6% | +11.3% | **-4.7%** | Extreme signals needed + sector filter active |
| Phase 5: 2015 Correction | -10.1% | -12.5% | +2.4% | Gold cap at 10% limits hedging |
| Phase 12: COVID Crash | -16.4% | -32.2% | +15.7% | Daily stops miss trend-break gap |

Bull phases are excellent (Phase 4: +64.6%, Phase 13: +268.4%, Phase 15: +115.2%). The goal is to improve weak phases WITHOUT hurting the strong ones.

---

## Root Cause Analysis

### Problem 1: Multiplicative Defensive Layer Compounding

The system has 5+ defensive layers that multiply against each other:

```
effective_equity = regime_allocation × vol_scale × breadth_scale × crash_scale
```

In a sideways market where NIFTY oscillates around its 200-SMA with breadth at 40-45%:
- Regime allocation: 0.85 (NORMAL)
- Vol targeting: 0.75 (21-day vol spikes from chop, target 15%, realized 20%)
- Breadth scaling: 0.75 (breadth at 40%, linear interpolation between 30%-50%)
- Combined: 0.85 × 0.75 × 0.75 = 0.478

The combined floor (0.50 downtrend, 0.80 uptrend) catches the worst case, but:
- If NIFTY is below 200-SMA (common in sideways): floor is only 0.50
- The strategy runs at 50-60% equity in a FLAT market
- Freed weight goes to LIQUIDBEES (~6-7% returns) while the market does 0-5%
- Every scaling change triggers trades → transaction costs compound the drag

**Code location**: `backtest.py:1575` — `combined_scale = vol_scale * breadth_scale`

### Problem 2: Stop Loss Whipsaw in Choppy Markets

The tiered stop system fires repeatedly in sideways conditions:
- Stock bought on momentum, rallies +5-8%
- Tier 1 trailing stop activates at +8% gain (12% trailing from peak)
- Normal 2-4 week retracement pulls stock back, triggering stop
- Strategy sells, pays 0.3% cost, buys replacement → repeat
- Each cycle costs ~0.6% round-trip

The `min_hold_days: 3` provides no protection against 2-4 week oscillations.
The `trend_break_buffer: 0.035` (3.5% × regime multiplier) in defensive/caution = 1.05-1.4% buffer — too thin for choppy markets.

**Code location**: `adaptive_dual_momentum.py:1394-1470` (check_exit_triggers)

### Problem 3: Recovery Detection Requires Portfolio Pain

Recovery mode (`_check_recovery_mode`) only triggers after -7% portfolio drawdown. But in demonetization (Phase 8) or 2019 tax cut (Phase 11):
- Defensive layers were already active, so portfolio didn't draw down 7%
- Market recovered from external shock → portfolio missed the bounce
- Bull recovery detection requires VIX > 20 declining + momentum > 0.003/day (extreme)
- Gradual recoveries don't produce these extreme signals

**Code location**: `adaptive_dual_momentum.py:190-220` (_check_recovery_mode), `indicators.py:1620-1700` (calculate_bull_recovery_signals)

### Problem 4: Stale Momentum in Sector Crashes

NMS uses 126/252 day lookbacks. When a sector crashes (NBFC 2018), its stocks retain high NMS from the prior rally for months. The 15% sector penalty is insufficient — a stock with NMS 2.5 after a 15% penalty is still 2.125, easily making the top 12.

**Code location**: `adaptive_dual_momentum.py:580-615` (rank_stocks, sector penalty application)

### Problem 5: Unused Indicators

Several existing functions in `indicators.py` could help but aren't wired into the dual momentum strategy:

| Function | Lines | What It Does | Why It Would Help |
|---|---|---|---|
| `calculate_momentum_acceleration` | 1445-1507 | Detects if momentum is speeding up or slowing | Filter out decelerating stocks before entry |
| `calculate_rs_trend` | 548-600 | Rising/falling relative strength slope | Boost accelerating outperformers |
| `calculate_exhaustion_score` | 1508-1600 | RSI extremes + distance from EMAs + volume climax | Avoid buying at exhaustion peaks |
| `calculate_macd_histogram_slope` | 2257+ | MACD divergence detection | Spot momentum waning before price confirms |

---

## Improvement Proposals

### I1: Sideways Market Detection — Reduce Churn, Not Exposure

**Rationale**: Previous session 11 tested this and showed +0.3pp CAGR, -49 trades. Was removed during code consolidation but the data supports re-implementation.

**Detection**: Composite signal from:
- Low ADX (< 20) on NIFTY 50 — indicates no strong trend
- Bollinger Band Width contraction (below 20-day median) — price range narrowing
- ATR ratio: 10-day ATR / 50-day ATR < 0.9 — volatility not expanding

Require 2 of 3 signals for sideways confirmation.

**Effect when sideways detected**:
- Widen `min_hold_days` from 3 → 7 (hold longer through chop)
- Widen `trend_break_buffer` by 50% (from 3.5% to 5.25% base)
- Increase `min_days_between` from 7 → 12 (rebalance less frequently)
- Do NOT change equity allocation — the problem is churn, not exposure

**What this does NOT do**: Reduce positions, change stops, or alter regime detection. It only slows down trading when the market isn't trending.

**Expected impact**: Directly targets Phase 1 and Phase 10 churn. If sideways is misdetected, worst case is slightly longer holds — minimal downside.

**Implementation**:
- Add `detect_sideways_market()` to `indicators.py` (new function)
- Add `sideways_state` to strategy state, check at each rebalance
- In `adaptive_dual_momentum.py`, apply wider parameters when sideways=True
- Mirror in `backtest.py` for parity

---

### I2: Additive Defensive Scaling Instead of Multiplicative

**Rationale**: Vol spike and breadth decline are correlated (~0.7). When VIX rises, breadth drops simultaneously. Multiplying double-counts the same risk event.

**Current code** (`backtest.py:1575`):
```python
combined_scale = vol_scale * breadth_scale
```

**Proposed**:
```python
combined_scale = min(vol_scale, breadth_scale)
```

Take the worst signal, don't multiply them.

**Impact analysis**:

| Scenario | Vol Scale | Breadth Scale | Current (multiply) | Proposed (min) |
|---|---|---|---|---|
| Mild stress | 0.85 | 0.80 | 0.68 | 0.80 |
| Moderate stress | 0.75 | 0.75 | 0.56 | 0.75 |
| Severe stress | 0.60 | 0.50 | 0.30 | 0.50 |
| One signal only | 0.70 | 1.00 | 0.70 | 0.70 |

In mild/moderate stress (typical sideways): effective equity goes from 56-68% to 75-80%. In severe stress: floor catches both at 0.50 anyway.

This is the single highest-impact change — one line, affects every defensive scaling event.

**Implementation**: Change one line in `backtest.py` and one in `momentum_engine.py`. Both must match (parity).

---

### I3: Momentum Deceleration Filter for Entry

**Rationale**: In sideways markets, many stocks show strong 6M returns (from prior rally) but decelerating recent momentum. Buying these is a trap — you're buying stale momentum.

**Using existing code**: `calculate_momentum_acceleration()` in `indicators.py:1445` already computes this as the ratio of short-term (63-day) to long-term (126-day) momentum.

**Proposal**: Soft penalty (similar to sector penalty approach):
```python
accel = calculate_momentum_acceleration(prices, short=63, long=126)
if accel < 0.85:  # momentum decelerating
    nms_score *= (1.0 - deceleration_penalty)  # e.g., 0.12 = 12% penalty
```

Not a hard filter — just pushes decelerating stocks lower in the ranking so accelerating stocks get priority.

**Config fields**:
- `use_momentum_deceleration_filter: true`
- `deceleration_threshold: 0.85`
- `deceleration_penalty: 0.12`

**Expected impact**: Fewer entries into stocks that are about to reverse. Specifically targets Phase 1 (sideways) and Phase 10 (NBFC) where the strategy bought stocks with strong historical momentum that was already fading.

**Implementation**:
- Wire `calculate_momentum_acceleration` into `rank_stocks()` in `adaptive_dual_momentum.py`
- Apply penalty after NMS calculation, before sector penalty
- Mirror in `backtest.py` for parity

---

### I4: Raise Gold Cap to 15% in Defensive Regimes

**Rationale**: The current 10% hard cap means even at maximum stress, gold is only 10% of portfolio. Research on trend-following portfolios suggests 15-20% gold in defensive environments improves risk-adjusted returns significantly.

**Current**: `max_gold_allocation: 0.10` (all regimes)

**Proposed**: Dynamic gold cap by regime:
- BULLISH/NORMAL: 10% (unchanged)
- CAUTION: 12%
- DEFENSIVE: 15%

The gold exhaustion scaling already handles the case where gold is overextended (above 40% of 200-SMA → zero gold), so this won't chase gold at tops.

**Impact**: In Phase 5 (2015 correction) and Phase 12 (COVID), the strategy was CAUTION/DEFENSIVE but gold was capped at 10%. An extra 5% in gold (which rallied during both periods) would have offset some equity losses.

**Implementation**:
- Add `max_gold_caution: 0.12` and `max_gold_defensive: 0.15` to regime config
- In `_calculate_defensive_allocation` (both engines), select gold cap based on current regime
- Straightforward config + 5-line logic change

---

### I5: Market-Based Recovery Trigger

**Rationale**: Current recovery mode requires -7% portfolio drawdown. In Phase 8 (demonetization) and Phase 11 (2019), defensive layers prevented the portfolio from reaching -7%, so recovery mode never activated. The market recovered but the strategy stayed defensive.

**Proposed**: Add a market-based trigger alongside the portfolio-based one:
```
IF nifty_price > nifty_3_month_low * 1.05   (5% above recent trough)
AND breadth_10d_ema is rising for 5+ days
AND current_regime in (CAUTION, DEFENSIVE)
THEN activate recovery mode
```

This catches external-shock recoveries (demonetization, tax cuts, policy changes) where the market turns before the portfolio suffers enough.

**Safeguards**:
- Only activates if currently in CAUTION or DEFENSIVE (no effect during BULLISH/NORMAL)
- Requires breadth confirmation (not just price bounce — need broad participation)
- 5-day EMA rising requirement prevents false triggers from dead-cat bounces

**Implementation**:
- Add `_check_market_recovery()` to `adaptive_dual_momentum.py`
- Called alongside `_check_recovery_mode()` at each rebalance
- Add equivalent logic to `backtest.py`
- New config fields: `market_recovery_bounce_pct: 0.05`, `market_recovery_breadth_days: 5`

---

### I6: Softer Sector Filter During Recovery

**Rationale**: Currently the sector filter is binary — fully active (bull recovery) or fully off (drawdown/crash recovery). In gradual recoveries (Phase 11), the previously-worst sectors often lead the rebound. The 15% penalty on these sectors means the strategy systematically underweights the recovery leaders.

**Current code** (`adaptive_dual_momentum.py:557`):
```python
if self._recovery_state.is_active or self._crash_recovery_state.is_active:
    return set()  # No sector filter during recovery
```

**Proposed**: During any recovery mode, reduce sector penalty from 15% to 7% (half strength):
```python
if in_recovery:
    penalty = base_penalty * 0.5  # Half strength during recovery
else:
    penalty = base_penalty
```

**Expected impact**: In Phase 11 (2019), recovery mode (if triggered by I5 above) would allow recovering sectors to rank higher, capturing more of the rebound.

**Implementation**: 3-line change in `rank_stocks()`, add `recovery_sector_penalty_mult: 0.50` to config.

---

### I7: Regime-Aware NMS Lookback

**Rationale**: NMS uses fixed 126/252 day lookbacks. In a bear market, 6 months of history includes the prior bull run, keeping scores artificially high for declining stocks. The strategy buys into "still looks strong on paper" stocks that are actually rolling over.

**Proposed**:
- BULLISH/NORMAL: Keep 126/252 day lookbacks (ride trends longer)
- CAUTION/DEFENSIVE: Shorten to 63/126 day lookbacks (respond to recent price action)

**Risk**: Shorter lookbacks increase turnover. This MUST be backtested carefully.

**Mitigation**: Combine with I1 (sideways detection) — in sideways + defensive, use shorter lookbacks but trade less frequently. The two improvements work together.

**Implementation**:
- Add `regime_adaptive_lookback: true` to config
- In `rank_stocks()`, check current regime and adjust lookback
- Recompute NMS with shorter windows when in CAUTION/DEFENSIVE
- Both engines must match

---

### I8: RS Trend Boost for Entry

**Rationale**: `calculate_rs_trend()` exists in `indicators.py:548` but is disabled (`use_rs_trend_boost: false`). An improving RS trend means a stock is accelerating its outperformance vs the benchmark — the strongest momentum signal.

**Proposed**: Give a +10% NMS boost to stocks with improving RS trend:
```python
rs_trend = calculate_rs_trend(prices, benchmark_prices, window=10)
if rs_trend > 0:  # Improving relative strength
    nms_score *= 1.10  # 10% boost
```

This is the mirror of I3 (deceleration penalty). Together they create a momentum quality filter:
- Accelerating + strong RS → boosted (best candidates)
- Decelerating + weak RS → penalized (avoid)
- Mixed signals → no adjustment

**Implementation**: Wire existing function into `rank_stocks()`, add `rs_trend_boost: 0.10` to config.

---

### I9: Daily Trend-Break Check

**Rationale**: Currently daily stop checks only evaluate hard stop (-15%) and trailing stop. Trend-break exits (price < 50-EMA) are only checked at rebalance (every 7-15 days).

This creates a gap: a stock at +5% gain (below trailing activation at +8%) can fall to -14.9% (just above hard stop) — a **20% decline from peak** that neither stop catches.

In the COVID crash (Phase 12), this gap contributed to the -16.4% loss.

**Proposed**: Add simplified daily trend-break check:
```python
# During daily stop check loop
ema_50 = prices.ewm(span=50).mean().iloc[-1]
if current_price < ema_50 * (1 - trend_break_buffer):
    if consecutive_days_below >= 2:
        trigger_exit()
```

The 50-EMA is computable from cached price data (already available via `_close_prices` dict). No additional data fetches needed.

**Risk**: Increased daily exits = more turnover. Need to backtest with and without.

**Mitigation**: Only apply daily trend-break during CAUTION/DEFENSIVE regimes. In BULLISH/NORMAL, keep the current rebalance-only check.

**Implementation**:
- Add 50-EMA computation to daily stop check loop in `backtest.py:1870-1920`
- Add `daily_trend_break_check: true` and `daily_trend_break_regimes: ["caution", "defensive"]` to config
- No change needed in live engine (already checks on every trigger)

---

### I10: Position Replacement Threshold

**Rationale**: When rankings shuffle by tiny amounts, the system sells position #12 and buys position #11 for a marginal NMS improvement. Each swap costs ~0.6% round-trip. In sideways markets, rankings shuffle constantly → high churn with minimal alpha benefit.

**Proposed**: Only replace an existing position if the replacement's NMS exceeds the current position's NMS by at least 25%:
```python
for existing_pos in current_holdings:
    existing_nms = get_nms(existing_pos)
    for candidate in new_rankings:
        if candidate.nms > existing_nms * 1.25:
            # Worth replacing
            swap(existing_pos, candidate)
        else:
            # Keep existing, not worth the cost
            keep(existing_pos)
```

**Impact**: In a sideways market, most rank shuffles are < 25% NMS difference. This change would eliminate most unnecessary swaps while still allowing strong new candidates to enter.

**Config field**: `position_replacement_threshold: 0.25`

**Risk**: In a trending market where new momentum leaders emerge quickly, 25% might be too high a bar. Backtest will reveal the right threshold — could be 15-30%.

**Implementation**:
- Add threshold check in `_build_target_portfolio()` (both engines)
- Compare candidate NMS vs existing holding NMS before deciding to swap
- ~15 lines of new logic

---

## Implementation Order

Ranked by expected impact × confidence × ease of implementation:

| Priority | Improvement | Expected Impact | Risk | Effort |
|---|---|---|---|---|
| 1 | **I2**: Min-based scaling | High (Phase 1, 10) | Very low (one-line) | 10 min |
| 2 | **I10**: Replacement threshold | Medium (all phases) | Low | 30 min |
| 3 | **I3**: Deceleration filter | Medium (Phase 1, 10) | Low (soft penalty) | 1 hr |
| 4 | **I1**: Sideways detection | Medium-High (Phase 1, 10) | Medium | 2 hr |
| 5 | **I5**: Market recovery trigger | Medium (Phase 8, 11) | Medium | 1 hr |
| 6 | **I4**: Gold cap 15% | Low-Medium (Phase 5, 12) | Low | 15 min |
| 7 | **I6**: Recovery sector penalty | Low-Medium (Phase 11) | Low | 15 min |
| 8 | **I8**: RS trend boost | Low-Medium (all phases) | Low | 30 min |
| 9 | **I9**: Daily trend-break | Medium (Phase 12) | Medium-High | 1 hr |
| 10 | **I7**: Regime-aware lookback | Medium (Phase 10) | High (turnover) | 1 hr |

**Approach**: Implement and backtest one at a time. Run 16-phase backtest after each change. Keep only what improves overall + doesn't hurt bull phases. Discard anything that doesn't measurably help.

---

## What We Will NOT Do (Documented Failed Experiments)

Based on session memory — these have been tested and proven harmful:

- **Faster crash detection** — -3%/-5%/-7% weekly thresholds all hurt returns
- **More defensive layers** — the problem is layer compounding, not layer count
- **Wider stops globally** — wider stops only help in detected sideways, hurt in trends
- **Bear-phase-specific logic** — sessions 6-7 proved comprehensively that being "smarter about bears" costs more than it saves
- **Weekly drawdown sentinel** — small drawdowns are routine noise, not signals

---

## Success Criteria

After implementing all retained improvements, the 16-phase backtest should show:

1. **Phase 1 alpha**: Improve from -13.4% to > -8% (reduce sideways drag)
2. **Phase 10 alpha**: Improve from -14.9% to > -10% (less stale momentum, less churn)
3. **Phase 8 alpha**: Improve from -7.4% to > -3% (faster recovery detection)
4. **Phase 11 alpha**: Improve from -4.7% to > -2% (softer sector filter in recovery)
5. **Overall CAGR**: Maintain or improve from 21.7%
6. **Bull phase returns**: No degradation in Phase 4, 13, 15 (must stay within 5% of current)
7. **Total trades**: Reduce by 10-20% (from reduced churn)
8. **Max DD**: Maintain or improve from -29.6%
