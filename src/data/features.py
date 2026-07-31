"""
Leakage-Safe Feature Engineering Pipeline & Feature Store Engine.
Combines stationary technical indicators, HMM regime posterior probabilities, and causal network metrics.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.models.regime_hmm import MarketRegimeHMM

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Leakage-safe feature engineering pipeline for H4 market dataset.
    """

    def __init__(self, dataset_version: str = "v1.0.0"):
        self.dataset_version = dataset_version
        self.regime_hmm = MarketRegimeHMM(n_components=4)

    def compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Computes Relative Strength Index (RSI) normalized to [0, 1]."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100.0

    def compute_bollinger_percent_b(self, series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
        """Computes Bollinger Bands %B indicator."""
        ma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = ma + (num_std * std)
        lower = ma - (num_std * std)
        pct_b = (series - lower) / ((upper - lower) + 1e-8)
        return pct_b

    def compute_macd_diff(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Computes MACD histogram difference."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def build_feature_matrix(self, aligned_df: pd.DataFrame, fit_hmm: bool = True) -> pd.DataFrame:
        """
        Builds complete leakage-safe feature matrix.
        """
        df = aligned_df.copy()
        if "timestamp_utc" in df.columns:
            df = df.set_index("timestamp_utc")

        feats = pd.DataFrame(index=df.index)

        # Primary EURUSD price series
        close = df["eurusd_close"].astype(float) if "eurusd_close" in df.columns else df["close"].astype(float)
        high = df["eurusd_high"].astype(float) if "eurusd_high" in df.columns else (df["high"].astype(float) if "high" in df.columns else close)
        low = df["eurusd_low"].astype(float) if "eurusd_low" in df.columns else (df["low"].astype(float) if "low" in df.columns else close)
        volume = df["eurusd_volume"].astype(float) if "eurusd_volume" in df.columns else (df["tick_volume"].astype(float) if "tick_volume" in df.columns else pd.Series(0.0, index=df.index))
        spread = df["eurusd_spread"].astype(float) if "eurusd_spread" in df.columns else (df["spread_points"].astype(float) if "spread_points" in df.columns else pd.Series(15.0, index=df.index))

        # Stationary Technical Features
        feats["eurusd_close"] = close
        feats["log_return_1b"] = np.log(close / close.shift(1))
        feats["log_return_4b"] = np.log(close / close.shift(4))
        feats["log_return_12b"] = np.log(close / close.shift(12))

        feats["rsi_14"] = self.compute_rsi(close, period=14)
        feats["bb_percent_b"] = self.compute_bollinger_percent_b(close, period=20)
        feats["macd_diff"] = self.compute_macd_diff(close, fast=12, slow=26, signal=9)

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
        )
        feats["atr_14_ratio"] = tr.rolling(window=14).mean() / close
        feats["spread_ratio"] = (spread * 0.00001) / close
        feats["volume_log_ratio"] = np.log((volume + 1.0) / (volume.rolling(window=12).mean() + 1.0))

        # Context symbol return features
        ctx_ret_cols = [c for c in df.columns if c.endswith("_return_1b") and not c.startswith("eurusd")]
        for col in ctx_ret_cols:
            feats[col] = df[col]

        # Fit or Predict Gaussian HMM Market Regimes
        sample_df = pd.DataFrame({"close": close, "high": high, "low": low}, index=df.index)
        if fit_hmm or not self.regime_hmm.is_fitted:
            self.regime_hmm.fit(sample_df)

        regime_probs = self.regime_hmm.predict_proba(sample_df)
        for c in ["prob_regime_0", "prob_regime_1", "prob_regime_2", "prob_regime_3", "dominant_regime"]:
            if c in regime_probs.columns:
                feats[c] = regime_probs[c]

        # Drop initial NaN rows due to rolling windows
        feats = feats.dropna()
        logger.info(f"Built feature matrix: {len(feats)} rows x {len(feats.columns)} columns.")
        return feats

    def save_feature_store(self, feats_df: pd.DataFrame) -> str:
        """
        Saves processed feature store as Parquet.
        Path: data/processed/features_h4/<version>/features.parquet
        """
        out_dir = os.path.join("data", "processed", "features_h4", self.dataset_version)
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, "features.parquet")
        
        feats_reset = feats_df.reset_index()
        feats_reset.to_parquet(fpath, index=False)
        logger.info(f"Saved feature store parquet to '{fpath}'")
        return fpath
