"""
PPO Reinforcement Learning Agent for EURUSD Continuous Action Decision Support.
Wraps Stable-Baselines3 PPO with MlpPolicy.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.models.rl_env import EURUSDTradingEnv

logger = logging.getLogger(__name__)


class PPOTradingAgent:
    """
    PPO RL Agent for Trading Decision Support.
    """

    def __init__(self, env: EURUSDTradingEnv, seed: int = 42):
        self.env = env
        self.vec_env = DummyVecEnv([lambda: env])
        self.seed = seed
        self.model: Optional[PPO] = None

    def train(self, total_timesteps: int = 10000) -> PPO:
        """
        Trains PPO policy on EURUSDTradingEnv.
        """
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=0.0003,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            seed=self.seed
        )
        self.model.learn(total_timesteps=total_timesteps)
        logger.info(f"PPO Agent training completed ({total_timesteps} timesteps).")
        return self.model

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates trained PPO policy over one full episode.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        obs, _ = self.env.reset()
        done = False
        total_reward = 0.0
        actions = []

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            actions.append(float(action[0]))

        final_equity = info.get("equity", self.env.initial_balance)
        total_ret_pct = ((final_equity - self.env.initial_balance) / self.env.initial_balance) * 100.0

        metrics = {
            "total_reward": round(float(total_reward), 4),
            "final_equity": round(float(final_equity), 2),
            "total_return_pct": round(float(total_ret_pct), 2),
            "mean_action": round(float(np.mean(actions)), 4),
            "std_action": round(float(np.std(actions)), 4)
        }
        logger.info(f"PPO Evaluation Metrics: Return={total_ret_pct:.2f}%, Final Equity=${final_equity:.2f}")
        return metrics

    def save_agent(self, filepath: str = "models/ppo_eurusd.zip"):
        """Saves trained PPO model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if self.model is not None:
            self.model.save(filepath)
            logger.info(f"Saved PPO agent model to '{filepath}'")

    def load_agent(self, filepath: str = "models/ppo_eurusd.zip"):
        """Loads trained PPO model."""
        self.model = PPO.load(filepath, env=self.vec_env)
        logger.info(f"Loaded PPO agent model from '{filepath}'")
