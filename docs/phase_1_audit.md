# Phase 1 Codebase & Baseline Audit Report

## 1. Executive Summary
This document records the empirical audit of the existing codebase as required by Phase 2. All files, dependencies, tests, CLI entry points, and mock vs. real implementations have been directly inspected.

---

## 2. Environment Details
- **Operating System**: Windows 11 (64-bit)
- **Python Version**: Python 3.13.4 (`MSC v.1943 64 bit AMD64`)
- **Virtual Environment Path**: `d:\eBay Software\AIUB Thesis\.venv`
- **MetaTrader 5 Executable**: `C:\Program Files\XM Global MT5\terminal64.exe`

---

## 3. Installed Package Versions (`pip freeze`)
```text
MetaTrader5==5.0.6070
torch==2.13.0
torch_geometric==2.8.0.post1
tigramite==5.2.10.1
hmmlearn==0.3.3
gymnasium==1.3.0
stable-baselines3==2.9.0
mo-gymnasium==1.4.0
morl-baselines==1.0.0
xgboost==3.3.0
networkx==3.6.1
mlflow==3.15.0
streamlit==1.60.0
pandas==3.0.5
numpy==2.5.1
scipy==1.18.0
scikit-learn==1.9.0
matplotlib==3.11.1
pytest==9.1.1
pyarrow==24.0.0
pyyaml==6.0.3
ruff==0.16.1
mypy==2.3.0
statsmodels==0.14.6
```

---

## 4. Repository Tree
```text
d:\eBay Software\AIUB Thesis\
├── docs/
│   ├── resource_registry.md
│   ├── adr/
│   │   └── 001_initial_dependency_stack.md
│   └── phase_1_audit.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── symbol_discovery.py
│   │   ├── offline_provider.py
│   │   └── mt5_provider.py
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── demo_guard.py
│   │   └── risk_engine.py
│   ├── features/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_demo_guard.py
│   ├── test_offline_provider.py
│   ├── test_symbol_discovery.py
│   └── test_risk_engine.py
├── requirements.txt
├── prompt_draft.md
└── README.md
```

---

## 5. Existing Modules & Functionality
| Module Path | Primary Class / Function | Purpose | Implementation Status |
|---|---|---|---|
| `src/config.py` | `PRIMARY_EXECUTION_SYMBOL`, `RiskPolicy` | Configuration settings for EURUSD, H4 timeframe, broker suffixes | Complete |
| `src/safety/demo_guard.py` | `DemoAccountGuard` | Validates `ACCOUNT_TRADE_MODE_DEMO` / `CONTEST` mode | Basic implementation (Needs Trade Mode strengthen audit) |
| `src/data/symbol_discovery.py` | `SymbolResolver` | Resolves symbol variations (`EURUSD`, `EURUSD.`, `EURUSD#`, etc.) | Complete for basic candidates |
| `src/data/offline_provider.py` | `OfflineDataProvider` | Synthetic/cached H4 OHLCV data generator with lookahead-safe slice | Complete |
| `src/data/mt5_provider.py` | `MT5DataProvider` | Fetches MT5 candles starting at index 1 (`copy_rates_from_pos`) | Basic implementation |
| `src/safety/risk_engine.py` | `RiskEngine` | Position sizing based on 0.5% max risk per trade, ATR stops, spread check | Complete |
| `src/main.py` | `run_scaffold_demo()` | Demonstration script | Functional (Uses synthetic mock data fallback) |

---

## 6. Real vs. Mocked Implementation Status
- `src/main.py`: Currently runs in **offline demonstration mode** using synthetic dataset generation (`OfflineDataProvider`). MT5 live initialization is implemented in `src/data/mt5_provider.py` but requires MT5 terminal connection.
- `DemoAccountGuard`: Audits `trade_mode` dictionary field, but requires explicit `account_info.trade_mode == 0` check, contest mode rejection, and login allowlist filtering.

---

## 7. Existing Test Suite (`pytest -q`)
Passed 11/11 tests in 0.89s:
- `test_demo_guard.py` (3 tests)
- `test_offline_provider.py` (2 tests)
- `test_symbol_discovery.py` (2 tests)
- `test_risk_engine.py` (4 tests)

---

## 8. Diagnostic Output Records
- `python --version`: `Python 3.13.4`
- `pytest -q`: `11 passed in 0.89s`
