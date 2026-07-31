"""
Live MT5 Historical H4 Data Downloader Script.
Downloads 1000 completed H4 candles for all XM symbols from connected MT5 terminal.
Stores immutable Parquet files, validates quality, aligns timeline, and generates dataset manifest.
"""
import sys
import os
import logging
import MetaTrader5 as mt5
import pandas as pd

from src.safety.demo_guard import DemoAccountGuard
from src.data.symbol_discovery import SymbolResolver
from src.data.pipeline import DataPipelineManager
from src.data.quality import DataQualityValidator
from src.config import PRIMARY_EXECUTION_SYMBOL, FOREX_CONTEXT_SYMBOLS, RISK_CONTEXT_SYMBOLS, SYMBOL_SUFFIX_CANDIDATES

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Live_Data_Downloader")

def download_live_history(num_bars: int = 1000, dataset_version: str = "v1.0.0"):
    print("Starting Live Data Download...", flush=True)
    
    mt5_path = r"C:\Program Files\XM Global MT5\terminal64.exe"
    if not mt5.initialize(path=mt5_path):
        print(f"MT5 initialization failed: {mt5.last_error()}", flush=True)
        return

    acc_info = mt5.account_info()._asdict()
    guard = DemoAccountGuard(enforce_demo=True)
    guard.verify_account_info(acc_info)

    print(f"Connected to Live MT5 Terminal: Account {acc_info['login']} ({acc_info['server']})", flush=True)

    # 1. Resolve symbols against active XM MT5 terminal
    resolver = SymbolResolver(mt5_interface=mt5)
    universe = resolver.resolve_dataset_universe(
        primary_symbol=PRIMARY_EXECUTION_SYMBOL,
        forex_context=FOREX_CONTEXT_SYMBOLS,
        risk_context=RISK_CONTEXT_SYMBOLS,
        candidate_suffixes=SYMBOL_SUFFIX_CANDIDATES
    )

    pipeline = DataPipelineManager(download_version=dataset_version)
    symbol_files = {}

    # 2. Download closed H4 candles for resolved symbols
    for gen_sym, sym_meta in universe["resolved_symbols"].items():
        broker_sym = sym_meta["broker_symbol"]
        print(f"Fetching {num_bars} completed H4 bars for '{gen_sym}' (XM Symbol: '{broker_sym}')...", flush=True)

        mt5.symbol_select(broker_sym, True)
        rates = mt5.copy_rates_from_pos(broker_sym, mt5.TIMEFRAME_H4, 1, num_bars)
        
        if rates is None or len(rates) == 0:
            print(f"Could not fetch rates for symbol '{broker_sym}': {mt5.last_error()}", flush=True)
            continue

        raw_df = pd.DataFrame(rates)
        raw_df["time"] = pd.to_datetime(raw_df["time"], unit="s", utc=True)

        canonical = pipeline.build_canonical_dataframe(
            raw_df, symbol=broker_sym, generic_symbol=gen_sym, timeframe="H4", source="mt5_live"
        )
        
        completed_df, meta = DataQualityValidator.verify_closed_candles(canonical)
        fpath = pipeline.save_raw_parquet(completed_df, broker_symbol=gen_sym)
        symbol_files[gen_sym] = fpath

        print(f"Downloaded & Stored {len(completed_df)} completed H4 bars for '{gen_sym}' at '{fpath}'", flush=True)

    # 3. Align Dataset on EURUSD Anchor Timeline
    if PRIMARY_EXECUTION_SYMBOL in symbol_files:
        aligned_df, aligned_path = pipeline.create_aligned_dataset(symbol_files, primary_symbol=PRIMARY_EXECUTION_SYMBOL)
        manifest = pipeline.generate_dataset_manifest(symbol_files, primary_target=PRIMARY_EXECUTION_SYMBOL)
        print(f"Successfully aligned dataset universe ({len(aligned_df)} rows) at '{aligned_path}'", flush=True)

    mt5.shutdown()
    print("Live MT5 Historical Data Download Complete.", flush=True)

if __name__ == "__main__":
    download_live_history()
