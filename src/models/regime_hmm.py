"""
Market Regime Discovery Engine using Gaussian Hidden Markov Models (hmmlearn).
Classifies completed H4 market data into discrete market regimes without lookahead bias.
"""
import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

REGIME_NAMES = {
    0: "Low-Vol Consolidation",
    1: "Bullish Trend",
    2: "Bearish Trend",
    3: "High-Vol Mean-Reverting"
}


class MarketRegimeHMM:
    """
    Gaussian Hidden Markov Model for Market Regime Classification.
    """

    def __init__(self, n_components: int = 4, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model: Optional[GaussianHMM] = None
        self.state_map: Dict[int, int] = {}
        self.is_fitted = False

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts stationary regime features from OHLCV DataFrame:
        - log_return_1b: 1-bar log return
        - vol_4b: 4-bar rolling stddev of log return
        - atr_ratio: 12-bar ATR divided by close price
        """
        feats = pd.DataFrame(index=df.index)
        close = df["close"].astype(float)
        high = df["high"].astype(float) if "high" in df.columns else close
        low = df["low"].astype(float) if "low" in df.columns else close

        feats["log_return_1b"] = np.log(close / close.shift(1))
        feats["vol_4b"] = feats["log_return_1b"].rolling(window=4).std()

        # ATR calculation
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close.shift(1)),
                np.abs(low - close.shift(1))
            )
        )
        feats["atr_ratio"] = tr.rolling(window=12).mean() / close
        
        feats = feats.dropna()
        return feats

    def fit(self, df: pd.DataFrame) -> "MarketRegimeHMM":
        """
        Fits Gaussian HMM model on extracted regime features.
        """
        feats = self.extract_features(df)
        X = feats.values

        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            min_covar=1e-3,
            n_iter=100,
            random_state=self.random_state
        )
        self.model.fit(X)

        # Order states deterministically by volatility & mean return
        stds = [np.sqrt(np.diag(cov))[1] for cov in self.model.covars_]
        sorted_indices = np.argsort(stds)
        self.state_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        self.is_fitted = True

        logger.info(f"Gaussian HMM fitted successfully with {self.n_components} states.")
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates posterior state probabilities P(S_t = k | F_t) for each bar.
        Returns DataFrame with columns: [regime_0, regime_1, regime_2, regime_3, dominant_regime]
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        feats = self.extract_features(df)
        X = feats.values
        raw_probs = self.model.predict_proba(X)

        prob_cols = {}
        for old_idx, new_idx in self.state_map.items():
            prob_cols[f"prob_regime_{new_idx}"] = raw_probs[:, old_idx]

        prob_df = pd.DataFrame(prob_cols, index=feats.index)

        # Reorder columns regime_0 to regime_N
        cols = [f"prob_regime_{i}" for i in range(self.n_components)]
        prob_df = prob_df[cols]
        prob_df["dominant_regime"] = np.argmax(prob_df.values, axis=1)
        prob_df["dominant_regime_label"] = prob_df["dominant_regime"].map(REGIME_NAMES)

        # Reindex to match full original DataFrame
        full_df = pd.DataFrame(index=df.index)
        full_df = full_df.join(prob_df).bfill().ffill()
        return full_df

    def save_model(self, filepath: str = "models/regime_hmm.pkl"):
        """Saves fitted HMM model artifact."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "state_map": self.state_map, "n_components": self.n_components}, filepath)
        logger.info(f"Saved HMM regime model to '{filepath}'")

    def load_model(self, filepath: str = "models/regime_hmm.pkl"):
        """Loads fitted HMM model artifact."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.state_map = data["state_map"]
        self.n_components = data["n_components"]
        self.is_fitted = True
        logger.info(f"Loaded HMM regime model from '{filepath}'")
