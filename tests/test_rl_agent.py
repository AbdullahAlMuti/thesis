"""
Unit tests for PPOTradingAgent (Stable-Baselines3).
"""
import os
import pytest
from src.models.rl_env import EURUSDTradingEnv
from src.models.rl_agent import PPOTradingAgent
from src.data.features import FeatureEngineeringPipeline
from src.data.offline_provider import OfflineDataProvider

def test_ppo_agent_train_eval_save_load(tmp_path):
    provider = OfflineDataProvider()
    df = provider.generate_synthetic_ohlcv(symbol="EURUSD", num_bars=200)
    pipe = FeatureEngineeringPipeline()
    feats = pipe.build_feature_matrix(df, fit_hmm=True)

    env = EURUSDTradingEnv(df_features=feats)
    agent = PPOTradingAgent(env=env, seed=42)

    # Train for short timesteps
    agent.train(total_timesteps=256)
    assert agent.model is not None

    metrics = agent.evaluate()
    assert "total_reward" in metrics
    assert "final_equity" in metrics
    assert "total_return_pct" in metrics

    save_path = os.path.join(tmp_path, "ppo_eurusd.zip")
    agent.save_agent(save_path)
    assert os.path.exists(save_path)

    loaded_agent = PPOTradingAgent(env=env, seed=42)
    loaded_agent.load_agent(save_path)
    assert loaded_agent.model is not None
