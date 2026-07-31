"""
Live MT5 Symbol Discovery and Resolution Script.
Queries active MetaTrader 5 terminal for XM broker symbols and exports mapping.
"""
import logging
import MetaTrader5 as mt5
from src.safety.demo_guard import DemoAccountGuard
from src.data.symbol_discovery import SymbolResolver
from src.config import PRIMARY_EXECUTION_SYMBOL, FOREX_CONTEXT_SYMBOLS, RISK_CONTEXT_SYMBOLS, SYMBOL_SUFFIX_CANDIDATES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Live_Symbol_Discovery")

def run_live_discovery():
    if not mt5.initialize():
        logger.error(f"MT5 initialization failed: {mt5.last_error()}")
        return
    
    acc_info = mt5.account_info()._asdict()
    guard = DemoAccountGuard(enforce_demo=True)
    guard.verify_account_info(acc_info)
    
    logger.info(f"Connected to Live MT5 Terminal: Account {acc_info['login']} ({acc_info['server']})")
    
    resolver = SymbolResolver(mt5_interface=mt5)
    universe = resolver.resolve_dataset_universe(
        primary_symbol=PRIMARY_EXECUTION_SYMBOL,
        forex_context=FOREX_CONTEXT_SYMBOLS,
        risk_context=RISK_CONTEXT_SYMBOLS,
        candidate_suffixes=SYMBOL_SUFFIX_CANDIDATES
    )
    
    logger.info("Exporting live symbol mapping to configs/data/xm_symbol_mapping.yaml")
    resolver.export_mapping_yaml("configs/data/xm_symbol_mapping.yaml")
    
    logger.info("Exporting live symbol report to artifacts/reports/symbol_discovery.json")
    resolver.export_report_json("artifacts/reports/symbol_discovery.json")
    
    mt5.shutdown()
    logger.info("Live MT5 Symbol Discovery Completed Successfully.")

if __name__ == "__main__":
    run_live_discovery()
