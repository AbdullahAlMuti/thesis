"""
Unit tests for multi-symbol temporal alignment.
"""
from src.data.pipeline import DataPipelineManager
from src.data.offline_provider import OfflineDataProvider

def test_temporal_alignment_anchored_on_eurusd():
    provider = OfflineDataProvider()
    pipeline = DataPipelineManager(download_version="align_v1")

    eur_raw = provider.generate_synthetic_ohlcv("EURUSD", num_bars=100)
    gbp_raw = provider.generate_synthetic_ohlcv("GBPUSD", num_bars=100)

    eur_can = pipeline.build_canonical_dataframe(eur_raw, "EURUSD", "EURUSD")
    gbp_can = pipeline.build_canonical_dataframe(gbp_raw, "GBPUSD", "GBPUSD")

    f_eur = pipeline.save_raw_parquet(eur_can, "EURUSD")
    f_gbp = pipeline.save_raw_parquet(gbp_can, "GBPUSD")

    aligned_df, aligned_path = pipeline.create_aligned_dataset(
        {"EURUSD": f_eur, "GBPUSD": f_gbp}, primary_symbol="EURUSD"
    )

    assert len(aligned_df) == 100
    assert "eurusd_close" in aligned_df.columns
    assert "GBPUSD_close" in aligned_df.columns
    assert "avail_GBPUSD" in aligned_df.columns
    assert (aligned_df["avail_GBPUSD"] == 1).all()
