"""
Unit tests for OfflineDataProvider and leakage prevention.
"""
from src.data.offline_provider import OfflineDataProvider

def test_generate_synthetic_ohlcv():
    provider = OfflineDataProvider()
    df = provider.load_dataset("EURUSD")
    assert len(df) == 1000
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "tick_volume" in df.columns

def test_leakage_safe_slice():
    provider = OfflineDataProvider()
    df = provider.load_dataset("EURUSD")
    cutoff_time = df.index[100]

    sliced = provider.get_slice_up_to_time("EURUSD", cutoff_time)
    assert len(sliced) == 101
    assert sliced.index[-1] == cutoff_time
    assert (sliced.index > cutoff_time).sum() == 0
