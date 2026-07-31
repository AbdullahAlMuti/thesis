"""
Unit tests for MarketRegimeHMM.
"""
import os
import pytest
import pandas as pd
from src.models.regime_hmm import MarketRegimeHMM
from src.data.offline_provider import OfflineDataProvider

def test_hmm_fit_and_prediction():
    provider = OfflineDataProvider()
    df = provider.generate_synthetic_ohlcv(symbol="EURUSD", num_bars=300)
    
    hmm = MarketRegimeHMM(n_components=4, random_state=42)
    hmm.fit(df)
    assert hmm.is_fitted is True
    
    prob_df = hmm.predict_proba(df)
    assert "prob_regime_0" in prob_df.columns
    assert "prob_regime_1" in prob_df.columns
    assert "prob_regime_2" in prob_df.columns
    assert "prob_regime_3" in prob_df.columns
    assert "dominant_regime" in prob_df.columns
    assert "dominant_regime_label" in prob_df.columns
    
    # Check row probabilities sum to ~1.0
    sum_probs = prob_df[["prob_regime_0", "prob_regime_1", "prob_regime_2", "prob_regime_3"]].sum(axis=1)
    assert ((sum_probs - 1.0).abs() < 1e-3).all()

def test_hmm_model_serialization(tmp_path):
    provider = OfflineDataProvider()
    df = provider.generate_synthetic_ohlcv(symbol="EURUSD", num_bars=200)
    
    hmm = MarketRegimeHMM(n_components=4)
    hmm.fit(df)
    
    model_path = os.path.join(tmp_path, "regime_hmm.pkl")
    hmm.save_model(model_path)
    assert os.path.exists(model_path)
    
    loaded_hmm = MarketRegimeHMM(n_components=4)
    loaded_hmm.load_model(model_path)
    assert loaded_hmm.is_fitted is True
