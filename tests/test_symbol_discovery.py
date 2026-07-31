"""
Unit tests for SymbolResolver and mapping exporter.
"""
import os
from src.data.symbol_discovery import SymbolResolver

def test_offline_symbol_discovery():
    resolver = SymbolResolver(mt5_interface=None)
    rec = resolver.discover_symbol("EURUSD", ["", ".", "#", "m"], role="target")
    assert rec is not None
    assert rec.broker_symbol == "EURUSD"
    assert rec.role == "target"
    assert rec.confidence == 1.0

def test_resolve_dataset_universe_exports():
    resolver = SymbolResolver(mt5_interface=None)
    universe = resolver.resolve_dataset_universe(
        primary_symbol="EURUSD",
        forex_context=["GBPUSD", "USDJPY"],
        risk_context=["XAUUSD", "US500"],
        candidate_suffixes=["", ".", "#"]
    )
    assert universe["primary_target"] == "EURUSD"
    assert universe["total_resolved"] == 5
    assert os.path.exists("configs/data/xm_symbol_mapping.yaml")
    assert os.path.exists("artifacts/reports/symbol_discovery.json")
