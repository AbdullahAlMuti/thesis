"""
Unit tests for closed-candle completeness verification.
"""
import pandas as pd
from datetime import datetime, timezone
from src.data.quality import DataQualityValidator

def test_closed_candle_filtering():
    trusted_now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)

    # Bar 1: Started 2026-08-01 04:00, Closed at 08:00 (Completed)
    # Bar 2: Started 2026-08-01 08:00, Closed at 12:00 (Completed)
    # Bar 3: Started 2026-08-01 12:00, Closed at 16:00 (Forming/Incomplete)
    timestamps = [
        datetime(2026, 8, 1, 4, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 1, 8, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    ]
    df = pd.DataFrame({"open": [1.08, 1.085, 1.09]}, index=timestamps)

    completed_df, metrics = DataQualityValidator.verify_closed_candles(
        df, timeframe_hours=4, trusted_now_utc=trusted_now
    )

    assert metrics["total_rows"] == 3
    assert metrics["completed_rows"] == 2
    assert metrics["rejected_forming_rows"] == 1
    assert len(completed_df) == 2
    assert completed_df.index[-1] == datetime(2026, 8, 1, 8, 0, tzinfo=timezone.utc)

def test_future_timestamp_rejected():
    trusted_now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    future_time = datetime(2026, 8, 2, 0, 0, tzinfo=timezone.utc)
    df = pd.DataFrame({"open": [1.08]}, index=[future_time])

    completed_df, metrics = DataQualityValidator.verify_closed_candles(
        df, timeframe_hours=4, trusted_now_utc=trusted_now
    )
    assert len(completed_df) == 0
    assert metrics["rejected_forming_rows"] == 1
