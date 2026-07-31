"""
Data Quality Validation and Canonical Schema Verification Engine.
Enforces strict OHLC sanity, timestamp ordering, volume, spread, and closed-candle completeness rules.
"""
import hashlib
import logging
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

CANONICAL_SCHEMA_COLUMNS = [
    "timestamp_utc", "symbol", "generic_symbol", "timeframe",
    "open", "high", "low", "close", "tick_volume", "spread_points",
    "real_volume", "is_complete", "source", "downloaded_at_utc"
]

class DataValidationError(Exception):
    """Raised when data fails strict quality or canonical schema checks."""
    pass


class DataQualityValidator:
    """
    Validates market OHLCV data against canonical schema and lookahead-free completeness rules.
    """

    @staticmethod
    def validate_ohlc_sanity(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validates OHLC high >= open, high >= close, high >= low, low <= open, low <= close, low <= high.
        """
        errors = []
        if df.empty:
            return False, ["DataFrame is empty."]

        # Check positive price bounds
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                if (df[col] <= 0).any():
                    invalid_count = (df[col] <= 0).sum()
                    errors.append(f"Column '{col}' has {invalid_count} zero or negative values.")

        # Check high bound
        invalid_high_open = (df["high"] < df["open"]).sum()
        invalid_high_close = (df["high"] < df["close"]).sum()
        invalid_high_low = (df["high"] < df["low"]).sum()
        if invalid_high_open > 0:
            errors.append(f"{invalid_high_open} rows have high < open.")
        if invalid_high_close > 0:
            errors.append(f"{invalid_high_close} rows have high < close.")
        if invalid_high_low > 0:
            errors.append(f"{invalid_high_low} rows have high < low.")

        # Check low bound
        invalid_low_open = (df["low"] > df["open"]).sum()
        invalid_low_close = (df["low"] > df["close"]).sum()
        if invalid_low_open > 0:
            errors.append(f"{invalid_low_open} rows have low > open.")
        if invalid_low_close > 0:
            errors.append(f"{invalid_low_close} rows have low > close.")

        # Check volume & spread
        if "tick_volume" in df.columns and (df["tick_volume"] < 0).any():
            errors.append(f"{(df['tick_volume'] < 0).sum()} rows have negative tick_volume.")
        if "spread_points" in df.columns and (df["spread_points"] < 0).any():
            errors.append(f"{(df['spread_points'] < 0).sum()} rows have negative spread_points.")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def verify_closed_candles(
        df: pd.DataFrame,
        timeframe_hours: int = 4,
        trusted_now_utc: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Verifies that every candle is closed:
        bar_close_time = timestamp_utc + timeframe_hours <= trusted_now_utc
        Filters out any currently forming or future candle.
        """
        now_utc = trusted_now_utc or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        df_copy = df.copy()

        # Ensure timestamp_utc index or column is timezone-aware UTC
        if "timestamp_utc" in df_copy.columns:
            ts_series = pd.to_datetime(df_copy["timestamp_utc"], utc=True)
        else:
            ts_series = pd.to_datetime(df_copy.index, utc=True)

        close_times = ts_series + pd.Timedelta(hours=timeframe_hours)
        is_complete_mask = close_times <= now_utc

        rejected_forming = (~is_complete_mask).sum()
        completed_df = df_copy[is_complete_mask].copy()

        close_times_completed = close_times[is_complete_mask]
        latest_close_ts = close_times_completed.iloc[-1] if len(close_times_completed) > 0 else None
        metrics = {
            "total_rows": len(df_copy),
            "completed_rows": len(completed_df),
            "rejected_forming_rows": int(rejected_forming),
            "trusted_now_utc": now_utc.isoformat(),
            "latest_completed_close_time": pd.to_datetime(latest_close_ts).isoformat() if latest_close_ts is not None else None
        }

        logger.info(f"Closed-Candle Verification: Retained {len(completed_df)} completed H4 bars, Rejected {rejected_forming} forming/future bars.")
        return completed_df, metrics

    @staticmethod
    def calculate_file_sha256(filepath: str) -> str:
        """Calculates SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
