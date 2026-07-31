"""
Unit tests for DecisionAlertEngine.
"""
import os
import pytest
from src.dashboard.alert_engine import DecisionAlertEngine

def test_alert_generation_and_logging(tmp_path):
    log_p = os.path.join(tmp_path, "live_alerts.json")
    engine = DecisionAlertEngine(log_filepath=log_p)

    alert = engine.evaluate_live_candle(
        current_price=1.0850,
        action_val=0.6,
        regime_id=1,
        regime_prob=0.82,
        equity_usd=10000.0,
        spread_points=12.0
    )

    assert alert["target_symbol"] == "EURUSD"
    assert alert["recommendation"] == "BUY"
    assert alert["lot_size"] > 0.0
    assert alert["regime"] == "Bullish Trend"
    assert os.path.exists(log_p)
