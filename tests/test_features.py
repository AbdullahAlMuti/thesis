"""
Unit tests for FeatureEngineeringPipeline.
"""
import os
import pytest
import pandas as pd
from src.data.features import FeatureEngineeringPipeline
from src.data.offline_provider import OfflineDataProvider

def test_feature_matrix_building(tmp_path):
    provider = OfflineDataProvider()
    df = provider.generate_synthetic_ohlcv(symbol="EURUSD", num_bars=200)
    
    pipeline = FeatureEngineeringPipeline(dataset_version="feat_test")
    feats = pipeline.build_feature_matrix(df, fit_hmm=True)
    
    assert "log_return_1b" in feats.columns
    assert "rsi_14" in feats.columns
    assert "bb_percent_b" in feats.columns
    assert "macd_diff" in feats.columns
    assert "prob_regime_0" in feats.columns
    assert "dominant_regime" in feats.columns
    assert not feats.isna().any().any()
    
    fpath = pipeline.save_feature_store(feats)
    assert os.path.exists(fpath)
