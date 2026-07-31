"""
Unit tests for canonical schema and OHLC sanity checks.
"""
import pandas as pd
from src.data.quality import DataQualityValidator
from src.data.pipeline import DataPipelineManager

def test_valid_ohlc_sanity():
    df = pd.DataFrame({
        "open": [1.0850], "high": [1.0900], "low": [1.0800], "close": [1.0870],
        "tick_volume": [1000], "spread_points": [15]
    })
    is_valid, errors = DataQualityValidator.validate_ohlc_sanity(df)
    assert is_valid is True
    assert len(errors) == 0

def test_invalid_high_less_than_low():
    df = pd.DataFrame({
        "open": [1.0850], "high": [1.0700], "low": [1.0800], "close": [1.0870]
    })
    is_valid, errors = DataQualityValidator.validate_ohlc_sanity(df)
    assert is_valid is False
    assert any("high < low" in e for e in errors)

def test_invalid_negative_spread():
    df = pd.DataFrame({
        "open": [1.0850], "high": [1.0900], "low": [1.0800], "close": [1.0870],
        "spread_points": [-5]
    })
    is_valid, errors = DataQualityValidator.validate_ohlc_sanity(df)
    assert is_valid is False
    assert any("negative spread_points" in e for e in errors)

def test_canonical_dataframe_building():
    manager = DataPipelineManager()
    raw = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=2, freq="4h"),
        "open": [1.08, 1.085], "high": [1.09, 1.095], "low": [1.075, 1.08], "close": [1.085, 1.09],
        "tick_volume": [500, 600], "spread": [12, 14]
    })
    canonical = manager.build_canonical_dataframe(raw, symbol="EURUSD", generic_symbol="EURUSD")
    assert "timestamp_utc" in canonical.columns
    assert canonical["symbol"].iloc[0] == "EURUSD"
    assert canonical["timeframe"].iloc[0] == "H4"
    assert len(canonical) == 2
