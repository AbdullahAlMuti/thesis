"""
Dynamic Symbol Discovery Engine for MetaTrader 5 (XM Broker Compatible).
Resolves EURUSD target symbol and context universe symbols dynamically.
"""
import os
import yaml
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any

from src.config import PRIMARY_EXECUTION_SYMBOL, FOREX_CONTEXT_SYMBOLS, RISK_CONTEXT_SYMBOLS, SYMBOL_SUFFIX_CANDIDATES

logger = logging.getLogger(__name__)

@dataclass
class SymbolMetadata:
    generic_symbol: str
    broker_symbol: str
    confidence: float
    source: str
    enabled: bool
    role: str
    description: str = ""
    currency_base: str = ""
    currency_profit: str = ""
    digits: int = 5
    point: float = 0.00001
    tick_size: float = 0.00001
    tick_value: float = 1.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    stops_level: int = 10
    spread: float = 15.0
    contract_size: float = 100000.0


class SymbolResolver:
    """
    Dynamic broker symbol resolver supporting XM MT5 terminal symbol mapping.
    """

    DEFAULT_KNOWN_EQUIVALENTS = {
        "XAUUSD": ["GOLD", "XAUUSD", "GOLD.m", "XAUUSD.m"],
        "US500": ["US500Cash", "US500", "SPX500", "US500.Cash"],
        "US100": ["US100Cash", "US100", "NAS100", "US100.Cash"],
        "US30": ["US30Cash", "US30", "DJ30", "US30.Cash"],
        "GER40": ["GER40Cash", "GER40", "DAX40", "GER40.Cash"],
        "UK100": ["UK100Cash", "UK100", "FTSE100", "UK100.Cash"],
        "OIL": ["OILCash", "OIL", "WTI", "USOIL", "OIL.Cash"]
    }

    def __init__(self, mt5_interface: Optional[Any] = None):
        self.mt5 = mt5_interface
        self.resolved_symbols: Dict[str, SymbolMetadata] = {}

    def discover_symbol(self, base_symbol: str, candidate_suffixes: List[str], role: str) -> Optional[SymbolMetadata]:
        """
        Discovers broker symbol for a generic base symbol using candidate search hierarchy.
        """
        # If offline or MT5 interface not active, return standard mock resolution
        if self.mt5 is None:
            return SymbolMetadata(
                generic_symbol=base_symbol,
                broker_symbol=base_symbol,
                confidence=1.0,
                source="offline_default",
                enabled=True,
                role=role,
                description=f"Offline default for {base_symbol}"
            )

        # Build candidate list
        candidates = []
        if base_symbol in self.DEFAULT_KNOWN_EQUIVALENTS:
            candidates.extend(self.DEFAULT_KNOWN_EQUIVALENTS[base_symbol])
        
        candidates.append(base_symbol)
        for suf in candidate_suffixes:
            candidates.append(f"{base_symbol}{suf}")

        for cand in candidates:
            info = self.mt5.symbol_info(cand)
            if info is not None:
                tick_value = getattr(info, "trade_tick_value", 1.0)
                tick_size = getattr(info, "trade_tick_size", info.point)

                meta = SymbolMetadata(
                    generic_symbol=base_symbol,
                    broker_symbol=info.name,
                    confidence=1.0 if info.name == base_symbol else 0.95,
                    source="exact_match" if info.name == base_symbol else "equivalent_match",
                    enabled=True,
                    role=role,
                    description=getattr(info, "description", ""),
                    currency_base=getattr(info, "currency_base", ""),
                    currency_profit=getattr(info, "currency_profit", ""),
                    digits=getattr(info, "digits", 5),
                    point=getattr(info, "point", 0.00001),
                    tick_size=tick_size,
                    tick_value=tick_value,
                    volume_min=getattr(info, "volume_min", 0.01),
                    volume_max=getattr(info, "volume_max", 100.0),
                    volume_step=getattr(info, "volume_step", 0.01),
                    stops_level=getattr(info, "stops_level", 10),
                    spread=float(getattr(info, "spread", 15)),
                    contract_size=getattr(info, "trade_contract_size", 100000.0)
                )
                logger.info(f"Discovered {base_symbol} -> XM Broker Symbol: '{info.name}' (Conf: {meta.confidence})")
                return meta

        logger.warning(f"Could not discover broker symbol for base: '{base_symbol}'")
        return None

    def resolve_dataset_universe(
        self,
        primary_symbol: str = PRIMARY_EXECUTION_SYMBOL,
        forex_context: List[str] = FOREX_CONTEXT_SYMBOLS,
        risk_context: List[str] = RISK_CONTEXT_SYMBOLS,
        candidate_suffixes: List[str] = SYMBOL_SUFFIX_CANDIDATES
    ) -> Dict[str, Any]:
        """
        Resolves primary target and context symbol universe.
        """
        self.resolved_symbols.clear()

        # Primary execution target
        primary_meta = self.discover_symbol(primary_symbol, candidate_suffixes, role="target")
        if primary_meta:
            self.resolved_symbols[primary_symbol] = primary_meta

        # Forex context symbols
        for sym in forex_context:
            meta = self.discover_symbol(sym, candidate_suffixes, role="context")
            if meta:
                self.resolved_symbols[sym] = meta

        # Risk context symbols
        for sym in risk_context:
            meta = self.discover_symbol(sym, candidate_suffixes, role="context")
            if meta:
                self.resolved_symbols[sym] = meta

        return {
            "primary_target": primary_symbol,
            "total_resolved": len(self.resolved_symbols),
            "resolved_symbols": {k: asdict(v) for k, v in self.resolved_symbols.items()}
        }

    def export_mapping_yaml(self, filepath: str = "configs/data/xm_symbol_mapping.yaml"):
        """Exports YAML symbol mapping."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {k: asdict(v) for k, v in self.resolved_symbols.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.info(f"Exported XM symbol mapping to '{filepath}'")

    def export_report_json(self, filepath: str = "artifacts/reports/symbol_discovery.json"):
        """Exports JSON discovery report."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "primary_target": PRIMARY_EXECUTION_SYMBOL,
            "total_resolved": len(self.resolved_symbols),
            "symbols": {k: asdict(v) for k, v in self.resolved_symbols.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported symbol discovery report to '{filepath}'")
