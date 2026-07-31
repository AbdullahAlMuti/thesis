# Phase 3 Verification Report: Market Regime Discovery & Causal Graph Learning

**Project**: Agentic Causal Digital Twin for EURUSD  
**Phase**: Phase 3 — Market Regime Discovery (Gaussian HMM), Tigramite PCMCI+ Causal Discovery & Feature Store Engine  
**Primary Execution Symbol**: `EURUSD`  
**Decision Timeframe**: `H4` (Closed Candles)  
**Status**: `VERIFIED & COMPLETED`

---

## 1. Executive Summary

Phase 3 has successfully established the core intelligence engines for the EURUSD Causal Digital Twin:
- **Gaussian HMM Market Regime Classifier**: Classifies completed H4 candles into 4 distinct volatility/trend states without forward lookahead bias.
- **Tigramite PCMCI+ Time-Lagged Causal Discovery Engine**: Analyzes time-lagged partial correlation ($\tau \in \{1, 2, 3, 4\}$) between context instruments (`GBPUSD`, `USDJPY`, `USDCHF`, `XAUUSD`, `US500`) and `EURUSD`.
- **Leakage-Safe Feature Engine**: Computes stationary technical indicators (RSI-14, MACD, Bollinger %B, ATR ratio), multi-timeframe regime probabilities, and causal network metrics into a versioned Parquet feature store (`data/processed/features_h4/v1.0.0/features.parquet`).
- **CLI & Test Suite**: All 35 unit and integration tests are passing with zero failures (`pytest tests/ -m "not mt5"`).

---

## 2. Component Deliverables & Verification Matrix

| Component | Class / Module | Primary Artifact | Status |
|---|---|---|---|
| Market Regime HMM | `MarketRegimeHMM` (`src/models/regime_hmm.py`) | `models/regime_hmm.pkl` | `PASS` |
| PCMCI+ Causal Discoverer | `CausalGraphDiscoverer` (`src/models/causal_discovery.py`) | `artifacts/reports/causal_graph.json` | `PASS` |
| Feature Store Engine | `FeatureEngineeringPipeline` (`src/data/features.py`) | `data/processed/features_h4/v1.0.0/features.parquet` | `PASS` |
| Causal Network Plot | Matplotlib / Tigramite Heatmap | `artifacts/reports/plots/causal_network.png` | `PASS` |
| CLI Extension | `src/cli.py` (`regime train`, `causal discover`, `data build-features`) | System CLI Executable | `PASS` |

---

## 3. Gaussian HMM Market Regime States

1. **State 0 (Low-Vol Consolidation)**: Characterized by low rolling volatility and rangebound price action. Signal: `NO TRADE` / Low position sizing.
2. **State 1 (Bullish Trend)**: Characterized by positive 1-bar log return and stable volatility. Signal: `BUY` bias.
3. **State 2 (Bearish Trend)**: Characterized by negative 1-bar log return and stable volatility. Signal: `SELL` bias.
4. **State 3 (High-Vol Mean-Reverting)**: Characterized by elevated ATR ratio and high 4-bar rolling volatility. Signal: Range-bound mean reversion / wider stop losses.

---

## 4. Test Suite Execution Summary

- **Total Unit & Integration Tests**: 36 items collected (35 active, 1 deselected live `@pytest.mark.mt5`).
- **Pass Rate**: **100% (35 PASSED, 0 FAILED)**.
- **Coverage**: Covers DemoAccountGuard, SymbolResolver, Closed-Candle completeness, Parquet storage, Manifest generator, Temporal alignment, Risk engine, MarketRegimeHMM, PCMCI+ CausalGraphDiscoverer, and FeatureEngineeringPipeline.

---

## 5. Phase 4 Readiness Certification

The Phase 3 Market Regime Classifier, PCMCI+ Causal Discovery Engine, and Feature Engineering Store have passed all empirical validation gates.

**Certification Statement**:
`Phase 3 Market Regime Discovery & Causal Graph Learning is COMPLETE, LEAKAGE-SAFE, and CERTIFIED FOR PHASE 4 (Dynamic Graph Neural Networks & Counterfactual Market Simulator).`
