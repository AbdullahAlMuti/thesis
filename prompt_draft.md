# Teamwork Project Prompt — Draft

> Status: Step 9 — Ready for launch (Scaffold Phase Complete)
> Goal: Execute Agentic Causal Digital Twin for EURUSD

Agentic Causal Digital Twin for EURUSD: Regime-Aware Counterfactual Graph Learning and Multi-Objective Reinforcement Learning for MetaTrader 5 Decision Support.

Working directory: d:\eBay Software\AIUB Thesis
Integrity mode: development

## Requirements

### R1. Mandated Primary Target Execution Focus
- Primary Execution Symbol is strictly `EURUSD`.
- Context variables (GBPUSD, USDJPY, XAUUSD, US500, etc.) provide cross-sectional explanatory signals and graph nodes but are not directly traded.

### R2. Leakage-Safe Timeframe & Data Processing
- Primary decision timeframe is completed `H4` candles.
- Start index is strictly $1$ (excluding position 0 forming candle).
- Multithreaded/multi-timeframe features must be strictly aligned with no lookahead bias.

### R3. MetaTrader 5 Connection & Safety Guards
- Direct interface via official `MetaTrader5` Python package.
- Mandatory `DemoAccountGuard` enforcing `ACCOUNT_TRADE_MODE_DEMO` / `ACCOUNT_TRADE_MODE_CONTEST`.
- Hard stop / exception raised if a real/live account is detected.

### R4. Dynamic Broker Symbol Discovery
- Dynamic resolution of broker symbol variations (`EURUSD`, `EURUSD.`, `EURUSD#`, `EURUSDm`, `EURUSDmicro`).
- Fallback to largest valid common dataset when contextual symbols are missing.

### R5. Risk Management & Position Sizing Engine
- Action space: continuous scalar $action_t \in [-1, 1]$.
- Recommendation classes: BUY ($action \ge +0.15$), SELL ($action \le -0.15$), HOLD / MANAGE ($-0.15 < action < +0.15$ with open position), NO TRADE (otherwise or when risk/spread constraints are violated).
- Risk limits: max 0.5% equity per trade, ATR-based volatility protective stops, minimum 1.5 R:R.

## Acceptance Criteria

### Scaffold & Safety Criteria
- [x] Primary target strictly confirmed as EURUSD.
- [x] Resource registry created documenting all repositories, licenses, and adoption status (`docs/resource_registry.md`).
- [x] Architecture Decision Record 001 created (`docs/adr/001_initial_dependency_stack.md`).
- [x] MetaTrader 5 installation detected (`C:\Program Files\XM Global MT5\terminal64.exe`).
- [x] Safety guards implemented preventing execution on real accounts (`src/safety/demo_guard.py`).
- [x] Dynamic broker symbol resolver implemented (`src/data/symbol_discovery.py`).
- [x] Offline provider with leakage-safe slicing implemented (`src/data/offline_provider.py`).
- [x] Risk engine converting continuous actions to XM lot sizes implemented (`src/safety/risk_engine.py`).
- [x] Unit test suite passing for demo guard, offline provider, symbol resolver, and risk engine.
