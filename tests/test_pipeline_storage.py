"""
Unit tests for Parquet storage, manifests, and file hashes.
"""
import os
from src.data.pipeline import DataPipelineManager
from src.data.offline_provider import OfflineDataProvider

def test_parquet_saving_and_manifest_generation(tmp_path):
    provider = OfflineDataProvider()
    pipeline = DataPipelineManager(download_version="test_v1")

    raw = provider.generate_synthetic_ohlcv(symbol="EURUSD", num_bars=50)
    canonical = pipeline.build_canonical_dataframe(raw, symbol="EURUSD", generic_symbol="EURUSD")

    fpath = pipeline.save_raw_parquet(canonical, broker_symbol="EURUSD")
    assert os.path.exists(fpath)

    manifest = pipeline.generate_dataset_manifest({"EURUSD": fpath}, primary_target="EURUSD")
    assert manifest["dataset_version"] == "test_v1"
    assert manifest["total_rows"] == 50
    assert "EURUSD" in manifest["symbols_summary"]
