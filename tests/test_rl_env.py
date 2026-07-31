"""
Unit tests for EURUSDTradingEnv (Gymnasium).
"""
import pytest
import numpy as np
from src.models.rl_env import EURUSDTradingEnv
from src.data.features import FeatureEngineeringPipeline
from src.data.offline_provider import OfflineDataProvider

def test_rl_env_reset_and_step():
    provider = OfflineDataProvider()
    df = provider.generate_synthetic_ohlcv(symbol="EURUSD", num_bars=200)
    pipe = FeatureEngineeringPipeline()
    feats = pipe.build_feature_matrix(df, fit_hmm=True)

    env = EURUSDTradingEnv(df_features=feats, initial_balance=10000.0)
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert info["balance"] == 10000.0

    action = np.array([0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "equity" in next_info
