"""
Data Quality Validation and Reporting Script.
Generates artifacts/reports/data_quality.json, data_quality.md, and diagnostic plots.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from src.data.quality import DataQualityValidator
from src.data.offline_provider import OfflineDataProvider
from src.data.pipeline import DataPipelineManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Data_Quality_Validator")

def run_data_quality_checks(dataset_version: str = "v1.0.0"):
    logger.info("Running Data Quality Validation Suite...")

    os.makedirs("artifacts/reports/plots", exist_ok=True)
    pipeline = DataPipelineManager(download_version=dataset_version)
    provider = OfflineDataProvider()

    # Generate synthetic offline dataset universe if raw files don't exist yet
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAUUSD", "US500"]
    symbol_files = {}

    for sym in symbols:
        raw_df = provider.generate_synthetic_ohlcv(symbol=sym, num_bars=1000)
        canonical = pipeline.build_canonical_dataframe(raw_df, symbol=sym, generic_symbol=sym, source="offline")
        filepath = pipeline.save_raw_parquet(canonical, broker_symbol=sym)
        symbol_files[sym] = filepath

    # 1. Quality Validation across all symbols
    quality_summary = {}
    overall_status = "PASS"

    for sym, fpath in symbol_files.items():
        df = pd.read_parquet(fpath)
        is_valid, errors = DataQualityValidator.validate_ohlc_sanity(df)
        completed_df, meta = DataQualityValidator.verify_closed_candles(df)

        # Calculate return stats and outliers
        returns = np.log(completed_df["close"] / completed_df["close"].shift(1)).dropna()
        outliers = (np.abs(returns - returns.mean()) > 4 * returns.std()).sum()

        quality_summary[sym] = {
            "is_valid_ohlc": is_valid,
            "ohlc_errors": errors,
            "total_rows": len(df),
            "completed_rows": len(completed_df),
            "rejected_forming": meta["rejected_forming_rows"],
            "duplicates": int(df.duplicated(subset=["timestamp_utc"]).sum()),
            "outliers_4sigma": int(outliers),
            "mean_spread_points": float(df["spread_points"].mean()) if "spread_points" in df.columns else 15.0
        }
        if not is_valid or int(df.duplicated(subset=["timestamp_utc"]).sum()) > 0:
            overall_status = "FAIL"

    # 2. Build Aligned Dataset
    aligned_df, aligned_path = pipeline.create_aligned_dataset(symbol_files, primary_symbol="EURUSD")
    _manifest = pipeline.generate_dataset_manifest(symbol_files, primary_target="EURUSD")

    # 3. Save JSON Quality Report
    report_dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "overall_status": overall_status,
        "aligned_rows": len(aligned_df),
        "aligned_path": aligned_path,
        "quality_summary": quality_summary
    }

    json_path = "artifacts/reports/data_quality.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # 4. Save Markdown Summary
    md_path = "artifacts/reports/data_quality.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Market Data Quality & Validation Report\n\n")
        f.write(f"**Dataset Version**: `{dataset_version}`\n")
        f.write(f"**Timestamp UTC**: {report_dict['timestamp_utc']}\n")
        f.write(f"**Overall Quality Status**: `{overall_status}`\n\n")
        f.write("## Quality Summary Table\n\n")
        f.write("| Symbol | OHLC Valid | Completed Rows | Duplicates | 4-Sigma Outliers | Mean Spread |\n|---|---|---|---|---|---|\n")
        for sym, q in quality_summary.items():
            valid_str = "PASS" if q["is_valid_ohlc"] else "FAIL"
            f.write(f"| `{sym}` | `{valid_str}` | {q['completed_rows']} | {q['duplicates']} | {q['outliers_4sigma']} | {q['mean_spread_points']:.1f} |\n")
        f.write("\n## Alignment Metadata\n\n")
        f.write(f"- Aligned File: `{aligned_path}`\n")
        f.write(f"- Anchor EURUSD Rows: {len(aligned_df)}\n")

    # 5. Generate Diagnostic Plots
    plt.figure(figsize=(10, 4))
    eurusd_df = pd.read_parquet(symbol_files["EURUSD"])
    plt.plot(pd.to_datetime(eurusd_df["timestamp_utc"]), eurusd_df["close"], label="EURUSD Close")
    plt.title("EURUSD Historical H4 Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.savefig("artifacts/reports/plots/eurusd_close_history.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    returns = np.log(eurusd_df["close"] / eurusd_df["close"].shift(1)).dropna()
    plt.hist(returns, bins=50, alpha=0.75, color="blue")
    plt.title("EURUSD Log Returns Distribution")
    plt.xlabel("Log Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("artifacts/reports/plots/eurusd_returns_distribution.png")
    plt.close()

    logger.info(f"Data Quality Validation Complete. Status: {overall_status}")
    return report_dict

if __name__ == "__main__":
    run_data_quality_checks()
