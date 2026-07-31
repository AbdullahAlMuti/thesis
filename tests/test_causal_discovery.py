"""
Unit tests for CausalGraphDiscoverer (Tigramite PCMCI+).
"""
import os
import pytest
import pandas as pd
from src.models.causal_discovery import CausalGraphDiscoverer
from src.data.pipeline import DataPipelineManager
from src.data.offline_provider import OfflineDataProvider

def test_pcmci_causal_discovery(tmp_path):
    provider = OfflineDataProvider()
    pipeline = DataPipelineManager(download_version="causal_test")
    
    eur_raw = provider.generate_synthetic_ohlcv("EURUSD", num_bars=200)
    gbp_raw = provider.generate_synthetic_ohlcv("GBPUSD", num_bars=200)
    
    eur_f = pipeline.save_raw_parquet(pipeline.build_canonical_dataframe(eur_raw, "EURUSD", "EURUSD"), "EURUSD")
    gbp_f = pipeline.save_raw_parquet(pipeline.build_canonical_dataframe(gbp_raw, "GBPUSD", "GBPUSD"), "GBPUSD")
    aligned_df, _ = pipeline.create_aligned_dataset({"EURUSD": eur_f, "GBPUSD": gbp_f}, primary_symbol="EURUSD")
    
    discoverer = CausalGraphDiscoverer(tau_max=2, alpha_level=0.1)
    results = discoverer.run_pcmci(aligned_df)
    
    assert "var_names" in results
    assert "significant_links" in results
    assert "EURUSD" in results["var_names"]
    
    json_p = os.path.join(tmp_path, "causal_graph.json")
    plot_p = os.path.join(tmp_path, "causal_network.png")
    discoverer.export_causal_reports(results, json_path=json_p, plot_path=plot_p)
    
    assert os.path.exists(json_p)
