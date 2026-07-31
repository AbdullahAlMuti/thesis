"""
Historical Data Pipeline, Parquet Storage, Manifest Generator, and Temporal Alignment Engine.
"""
import os
import sys
import json
import logging
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any

from src.data.quality import DataQualityValidator, CANONICAL_SCHEMA_COLUMNS, DataValidationError

logger = logging.getLogger(__name__)

def get_git_commit_hash() -> str:
    """Gets current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "uncommitted_scaffold"


class DataPipelineManager:
    """
    Manages historical MT5 data downloading, Parquet storage, manifest generation, and temporal alignment.
    """

    def __init__(self, download_version: str = "v1.0.0", mt5_provider: Optional[Any] = None):
        self.download_version = download_version
        self.mt5_provider = mt5_provider
        self.validator = DataQualityValidator()

    def build_canonical_dataframe(
        self,
        raw_df: pd.DataFrame,
        symbol: str,
        generic_symbol: str,
        timeframe: str = "H4",
        source: str = "offline"
    ) -> pd.DataFrame:
        """
        Converts raw OHLCV DataFrame to canonical schema.
        """
        df = raw_df.copy()

        # Reset index if timestamp is index
        if "time" not in df.columns and "timestamp_utc" not in df.columns:
            df = df.reset_index()

        time_col = "timestamp_utc" if "timestamp_utc" in df.columns else ("time" if "time" in df.columns else df.columns[0])

        df["timestamp_utc"] = pd.to_datetime(df[time_col], utc=True)
        df["symbol"] = symbol
        df["generic_symbol"] = generic_symbol
        df["timeframe"] = timeframe
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["tick_volume"] = df["tick_volume"].astype(int) if "tick_volume" in df.columns else 0
        df["spread_points"] = df["spread"].astype(int) if "spread" in df.columns else (df["spread_points"].astype(int) if "spread_points" in df.columns else 15)
        df["real_volume"] = df["real_volume"].astype(int) if "real_volume" in df.columns else 0
        df["is_complete"] = True
        df["source"] = source
        df["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat()

        # Reorder canonical columns
        canonical = df[CANONICAL_SCHEMA_COLUMNS].copy()

        # Sort ascending & drop duplicates by unique key: timestamp_utc + symbol + timeframe
        canonical = canonical.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc", "symbol", "timeframe"])
        return canonical

    def save_raw_parquet(self, canonical_df: pd.DataFrame, broker_symbol: str) -> str:
        """
        Saves raw canonical DataFrame immutably as Parquet.
        Path: data/raw/mt5/h4/<broker_symbol>/<download_version>/bars.parquet
        """
        dir_path = os.path.join("data", "raw", "mt5", "h4", broker_symbol, self.download_version)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, "bars.parquet")

        canonical_df.to_parquet(filepath, index=False)
        logger.info(f"Saved immutable Parquet for '{broker_symbol}' ({len(canonical_df)} rows) to '{filepath}'")
        return filepath

    def generate_dataset_manifest(
        self,
        symbol_files: Dict[str, str],
        primary_target: str = "EURUSD"
    ) -> Dict[str, Any]:
        """
        Generates dataset manifest JSON.
        Path: data/manifests/<download_version>.json
        """
        os.makedirs(os.path.join("data", "manifests"), exist_ok=True)
        manifest_path = os.path.join("data", "manifests", f"{self.download_version}.json")

        summary = {}
        total_rows = 0
        file_hashes = {}

        for gen_sym, fpath in symbol_files.items():
            df = pd.read_parquet(fpath)
            row_cnt = len(df)
            total_rows += row_cnt
            sha256 = self.validator.calculate_file_sha256(fpath)
            file_hashes[gen_sym] = sha256

            summary[gen_sym] = {
                "file_path": fpath,
                "row_count": row_cnt,
                "first_timestamp_utc": df["timestamp_utc"].iloc[0].isoformat() if row_cnt > 0 else None,
                "last_timestamp_utc": df["timestamp_utc"].iloc[-1].isoformat() if row_cnt > 0 else None,
                "sha256": sha256
            }

        manifest = {
            "dataset_version": self.download_version,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "primary_target": primary_target,
            "symbol_count": len(symbol_files),
            "total_rows": total_rows,
            "git_commit_hash": get_git_commit_hash(),
            "python_version": sys.version.split()[0],
            "symbols_summary": summary
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Generated dataset manifest at '{manifest_path}'")
        return manifest

    def create_aligned_dataset(
        self,
        symbol_files: Dict[str, str],
        primary_symbol: str = "EURUSD"
    ) -> Tuple[pd.DataFrame, str]:
        """
        Anchors strictly on EURUSD completed H4 timestamps.
        Attaches context symbol log returns and availability masks without lookahead bias.
        Path: data/interim/aligned_h4/<download_version>/aligned.parquet
        """
        if primary_symbol not in symbol_files:
            raise DataValidationError(f"Primary symbol '{primary_symbol}' missing from symbol files.")

        # Load primary anchor DataFrame
        target_df = pd.read_parquet(symbol_files[primary_symbol])
        target_df = target_df.sort_values("timestamp_utc").set_index("timestamp_utc")

        aligned = pd.DataFrame(index=target_df.index)
        aligned["eurusd_open"] = target_df["open"]
        aligned["eurusd_high"] = target_df["high"]
        aligned["eurusd_low"] = target_df["low"]
        aligned["eurusd_close"] = target_df["close"]
        aligned["eurusd_volume"] = target_df["tick_volume"]
        aligned["eurusd_spread"] = target_df["spread_points"]
        aligned["eurusd_return_1b"] = np.log(target_df["close"] / target_df["close"].shift(1))
        aligned["eurusd_spread_ratio"] = (target_df["spread_points"] * 0.00001) / target_df["close"]

        # Attach context symbols
        for gen_sym, fpath in symbol_files.items():
            if gen_sym == primary_symbol:
                continue

            ctx_df = pd.read_parquet(fpath).sort_values("timestamp_utc").set_index("timestamp_utc")
            ctx_ret = np.log(ctx_df["close"] / ctx_df["close"].shift(1))

            # Reindex to target EURUSD timeline strictly up to timestamp (no forward lookahead)
            aligned[f"{gen_sym}_close"] = ctx_df["close"].reindex(aligned.index)
            aligned[f"{gen_sym}_return_1b"] = ctx_ret.reindex(aligned.index)
            aligned[f"avail_{gen_sym}"] = (~aligned[f"{gen_sym}_close"].isna()).astype(int)

        out_dir = os.path.join("data", "interim", "aligned_h4", self.download_version)
        os.makedirs(out_dir, exist_ok=True)
        aligned_path = os.path.join(out_dir, "aligned.parquet")

        aligned_reset = aligned.reset_index()
        aligned_reset.to_parquet(aligned_path, index=False)

        logger.info(f"Created aligned dataset ({len(aligned_reset)} rows, {len(aligned_reset.columns)} cols) at '{aligned_path}'")
        return aligned_reset, aligned_path
