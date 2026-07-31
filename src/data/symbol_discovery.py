"""
Dynamic Broker Symbol Discovery for XM MetaTrader 5 Terminal.
Resolves and ranks symbol name variations (e.g. EURUSD, EURUSD., EURUSD#, EURUSDm) dynamically.
Exports YAML configuration to configs/data/xm_symbol_mapping.yaml and JSON report to artifacts/reports/symbol_discovery.json.
"""
import os
import json
import logging
import yaml
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

@dataclass
class SymbolMappingRecord:
    generic_symbol: str
    broker_symbol: str
    confidence: float
    source: str
    enabled: bool
    role: str  # target or context
    digits: int
    point: float
    tick_size: float
    tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level: int
    spread: int
    currency_base: str
    currency_profit: str
    trade_contract_size: float


class SymbolResolver:
    """
    Resolves broker-specific symbol names dynamically via MT5 API or mock symbol dictionary.
    """

    def __init__(self, mt5_interface: Optional[Any] = None):
        self.mt5 = mt5_interface
        self.resolved_records: Dict[str, SymbolMappingRecord] = {}
        self.unavailable_symbols: List[str] = []

    def discover_symbol(
        self,
        base_symbol: str,
        candidate_suffixes: List[str],
        role: str = "context"
    ) -> Optional[SymbolMappingRecord]:
        """
        Tries exact base symbol, case-insensitive match, and candidate suffixes against MT5.
        """
        if self.mt5 is None:
            # Offline mode mock record
            record = SymbolMappingRecord(
                generic_symbol=base_symbol,
                broker_symbol=base_symbol,
                confidence=1.0 if role == "target" else 0.95,
                source="offline_default",
                enabled=True,
                role=role,
                digits=5 if "JPY" not in base_symbol else 3,
                point=0.00001 if "JPY" not in base_symbol else 0.001,
                tick_size=0.00001 if "JPY" not in base_symbol else 0.001,
                tick_value=1.0,
                volume_min=0.01,
                volume_max=100.0,
                volume_step=0.01,
                stops_level=10,
                spread=15,
                currency_base=base_symbol[:3] if len(base_symbol) >= 6 else "USD",
                currency_profit=base_symbol[3:6] if len(base_symbol) >= 6 else "USD",
                trade_contract_size=100000.0 if len(base_symbol) == 6 else 100.0
            )
            self.resolved_records[base_symbol] = record
            return record

        # Live/Connected MT5 resolution
        candidates = [
            (base_symbol, 1.0, "exact_match"),
            (base_symbol.lower(), 0.98, "case_insensitive_match")
        ]
        for suf in candidate_suffixes:
            if suf:
                candidates.append((f"{base_symbol}{suf}", 0.92, f"suffix_match_{suf}"))

        for cand, conf, src in candidates:
            info = self.mt5.symbol_info(cand)
            if info is not None:
                if not info.select:
                    self.mt5.symbol_select(cand, True)

                _tick_info = self.mt5.symbol_info_tick(cand)
                tick_value = getattr(info, "trade_tick_value", 1.0)
                tick_size = getattr(info, "trade_tick_size", info.point)

                record = SymbolMappingRecord(
                    generic_symbol=base_symbol,
                    broker_symbol=cand,
                    confidence=conf,
                    source=src,
                    enabled=True,
                    role=role,
                    digits=info.digits,
                    point=info.point,
                    tick_size=tick_size,
                    tick_value=tick_value,
                    volume_min=info.volume_min,
                    volume_max=info.volume_max,
                    volume_step=info.volume_step,
                    stops_level=getattr(info, "trade_stops_level", 0),
                    spread=info.spread,
                    currency_base=info.currency_base,
                    currency_profit=info.currency_profit,
                    trade_contract_size=info.trade_contract_size
                )
                self.resolved_records[base_symbol] = record
                logger.info(f"Discovered {base_symbol} -> XM Broker Symbol: '{cand}' (Conf: {conf})")
                return record

        logger.warning(f"Could not discover broker symbol for base: '{base_symbol}'")
        self.unavailable_symbols.append(base_symbol)
        return None

    def resolve_dataset_universe(
        self,
        primary_symbol: str,
        forex_context: List[str],
        risk_context: List[str],
        candidate_suffixes: List[str]
    ) -> Dict[str, Any]:
        """
        Scans all primary and contextual symbols and exports YAML & JSON artifacts.
        """
        self.resolved_records.clear()
        self.unavailable_symbols.clear()

        # 1. Target symbol EURUSD first
        self.discover_symbol(primary_symbol, candidate_suffixes, role="target")

        # 2. Forex context
        for fx in forex_context:
            self.discover_symbol(fx, candidate_suffixes, role="context")

        # 3. Risk context
        for rk in risk_context:
            self.discover_symbol(rk, candidate_suffixes, role="context")

        # Export YAML config
        os.makedirs("configs/data", exist_ok=True)
        yaml_path = "configs/data/xm_symbol_mapping.yaml"
        yaml_dict = {}
        for gen_sym, rec in self.resolved_records.items():
            yaml_dict[gen_sym] = {
                "broker_symbol": rec.broker_symbol,
                "confidence": rec.confidence,
                "source": rec.source,
                "enabled": rec.enabled,
                "role": rec.role
            }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False)

        # Export JSON report
        os.makedirs("artifacts/reports", exist_ok=True)
        json_path = "artifacts/reports/symbol_discovery.json"
        json_dict = {
            "primary_target": primary_symbol,
            "total_resolved": len(self.resolved_records),
            "total_unavailable": len(self.unavailable_symbols),
            "unavailable_symbols": self.unavailable_symbols,
            "resolved_symbols": {k: asdict(v) for k, v in self.resolved_records.items()}
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=2)

        logger.info(
            f"Symbol Discovery Complete. Total Resolved: {len(self.resolved_records)}, "
            f"Unavailable: {len(self.unavailable_symbols)}"
        )
        return json_dict
