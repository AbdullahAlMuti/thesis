"""
Global Configuration for Agentic Causal Digital Twin for EURUSD
"""
from dataclasses import dataclass, field
from typing import List

PRIMARY_EXECUTION_SYMBOL: str = "EURUSD"

# Timeframe definitions
PRIMARY_TIMEFRAME: str = "H4"
SUPPORTING_TIMEFRAMES: List[str] = field(default_factory=lambda: ["H1", "D1"])

# Context symbols to search on XM MT5
FOREX_CONTEXT_SYMBOLS: List[str] = [
    "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "EURGBP", "EURJPY", "EURCHF"
]

RISK_CONTEXT_SYMBOLS: List[str] = [
    "XAUUSD", "US500", "US100", "US30", "GER40", "UK100", "OIL"
]

MACRO_CONTEXT_SYMBOLS: List[str] = [
    "DXY", "VIX"
]

# Broker Suffix Variations to test dynamically
SYMBOL_SUFFIX_CANDIDATES: List[str] = ["", ".", "#", "m", "micro", ".a", ".b"]

@dataclass
class RiskPolicy:
    max_risk_per_trade_pct: float = 0.005  # 0.5% max risk per trade
    max_total_open_risk_pct: float = 0.01   # 1.0% max total risk
    max_simultaneous_positions: int = 1     # Net 1 position on EURUSD
    max_daily_loss_pct: float = 0.02         # 2.0% daily drawdown limit
    max_weekly_drawdown_pct: float = 0.05    # 5.0% weekly drawdown limit
    min_reward_to_risk: float = 1.5          # Minimum 1.5 R:R ratio
    require_demo_account: bool = True       # MANDATORY DEMO GUARD
