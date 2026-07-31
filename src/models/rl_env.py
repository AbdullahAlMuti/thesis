"""
Gymnasium Trading Environment for EURUSD Agentic Digital Twin.
Continuous Action Space a_t in [-1.0, 1.0], Vectorized Reward Function balancing Profit, Sharpe, Drawdown, and Transaction Costs.
"""
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

from src.safety.risk_engine import RiskEngine

logger = logging.getLogger(__name__)


class EURUSDTradingEnv(gym.Env):
    """
    Gymnasium Trading Environment for EURUSD Decision Support.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_features: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost_pct: float = 0.0001,
        risk_fraction: float = 0.005
    ):
        super(EURUSDTradingEnv, self).__init__()

        self.df = df_features.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.risk_fraction = risk_fraction

        # Action Space: Continuous [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space: Feature vector excluding timestamp/labels
        feat_cols = [c for c in self.df.columns if c not in ("timestamp_utc", "dominant_regime_label")]
        self.feature_cols = feat_cols
        self.num_features = len(feat_cols)

        # Observation space Box bounds
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32
        )

        self.risk_engine = RiskEngine()
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.max_steps = len(self.df) - 1

        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.current_position = 0.0  # Current exposure in [-1.0, 1.0]
        self.returns_history = []

        obs = self._get_observation()
        info = {"balance": self.balance, "equity": self.equity}
        return obs, info

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_step][self.feature_cols].values
        return np.nan_to_num(row.astype(np.float32))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        clamped_action = float(np.clip(action[0], -1.0, 1.0))
        
        row = self.df.iloc[self.current_step]
        next_row = self.df.iloc[self.current_step + 1]

        price = float(row["eurusd_close"]) if "eurusd_close" in row else float(row["close"])
        next_price = float(next_row["eurusd_close"]) if "eurusd_close" in next_row else float(next_row["close"])
        
        price_ret = (next_price - price) / price

        # Position change & Transaction cost penalty
        pos_change = abs(clamped_action - self.current_position)
        cost_penalty = pos_change * self.transaction_cost_pct

        # Portfolio return
        step_return = (clamped_action * price_ret) - cost_penalty
        self.equity *= (1.0 + step_return)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.returns_history.append(step_return)

        # Multi-Objective Vector Reward Components
        r_profit = step_return
        
        # Sharpe component
        if len(self.returns_history) > 5:
            std = np.std(self.returns_history) + 1e-8
            r_sharpe = np.mean(self.returns_history) / std
        else:
            r_sharpe = 0.0

        # Drawdown penalty component
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        r_drawdown = -drawdown

        # Transaction cost penalty
        r_cost = -cost_penalty

        # Total Weighted Multi-Objective Reward
        reward = float((1.0 * r_profit) + (0.1 * r_sharpe) + (0.5 * r_drawdown) + (0.2 * r_cost))

        self.current_position = clamped_action
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = self.equity <= (self.initial_balance * 0.5)  # 50% max drawdown termination threshold

        obs = self._get_observation() if not terminated else np.zeros((self.num_features,), dtype=np.float32)
        info = {
            "balance": self.balance,
            "equity": self.equity,
            "drawdown": drawdown,
            "position": clamped_action
        }

        return obs, reward, terminated, truncated, info
