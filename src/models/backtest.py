"""
Walk-Forward Backtesting and Out-of-Sample Validation Engine for EURUSD Digital Twin.
Evaluates Sharpe Ratio, Sortino Ratio, Max Drawdown %, Profit Factor, and Win Rate % across rolling train/test folds.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

from src.models.rl_env import EURUSDTradingEnv
from src.models.rl_agent import PPOTradingAgent
from src.data.features import FeatureEngineeringPipeline

logger = logging.getLogger(__name__)


def calculate_performance_metrics(returns: np.ndarray, equity_curve: np.ndarray) -> Dict[str, float]:
    """
    Calculates quantitative metrics from returns array and equity curve.
    """
    if len(returns) == 0:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "win_rate_pct": 0.0,
            "total_return_pct": 0.0
        }

    ann_factor = np.sqrt(252 * 6)  # 6 H4 bars per day, 252 trading days per year
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) + 1e-8

    # Sharpe Ratio
    sharpe = float((mean_ret / std_ret) * ann_factor)

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) + 1e-8 if len(downside_returns) > 0 else 1e-8
    sortino = float((mean_ret / downside_std) * ann_factor)

    # Max Drawdown
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / peaks
    max_dd_pct = float(np.max(drawdowns) * 100.0)

    # Profit Factor
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    sum_gains = np.sum(gains) if len(gains) > 0 else 0.0
    sum_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-8
    profit_factor = float(sum_gains / sum_losses)

    # Win Rate
    win_rate = float((len(gains) / len(returns)) * 100.0) if len(returns) > 0 else 0.0

    # Total Return
    total_ret_pct = float(((equity_curve[-1] - equity_curve[0]) / equity_curve[0]) * 100.0)

    return {
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "profit_factor": round(profit_factor, 2),
        "win_rate_pct": round(win_rate, 2),
        "total_return_pct": round(total_ret_pct, 2)
    }


class WalkForwardBacktester:
    """
    Rolling Walk-Forward Backtester for Out-of-Sample Performance Validation.
    """

    def __init__(self, df_features: pd.DataFrame, n_folds: int = 4, initial_balance: float = 10000.0):
        self.df = df_features.reset_index(drop=True)
        self.n_folds = n_folds
        self.initial_balance = initial_balance

    def run_walk_forward(self) -> Dict[str, Any]:
        """
        Executes rolling walk-forward train/test evaluation.
        """
        total_len = len(self.df)
        fold_size = total_len // (self.n_folds + 1)
        
        fold_results = []
        all_agent_returns = []
        all_bnh_returns = []
        
        agent_equity = [self.initial_balance]
        bnh_equity = [self.initial_balance]

        for fold in range(self.n_folds):
            train_end = (fold + 1) * fold_size
            test_end = min(train_end + fold_size, total_len)

            train_df = self.df.iloc[:train_end].copy()
            test_df = self.df.iloc[train_end:test_end].copy()

            if len(test_df) < 10:
                continue

            # Train Agent on train split
            train_env = EURUSDTradingEnv(df_features=train_df, initial_balance=agent_equity[-1])
            agent = PPOTradingAgent(env=train_env, seed=42 + fold)
            agent.train(total_timesteps=512)

            # Evaluate Out-of-Sample on test split
            test_env = EURUSDTradingEnv(df_features=test_df, initial_balance=agent_equity[-1])
            metrics = agent.evaluate()

            fold_agent_equity = test_env.equity
            agent_equity.append(fold_agent_equity)

            # Buy & Hold benchmark on test split
            close_start = float(test_df["eurusd_close"].iloc[0]) if "eurusd_close" in test_df.columns else float(test_df["close"].iloc[0])
            close_end = float(test_df["eurusd_close"].iloc[-1]) if "eurusd_close" in test_df.columns else float(test_df["close"].iloc[-1])
            bnh_ret_pct = ((close_end - close_start) / close_start) * 100.0
            fold_bnh_equity = bnh_equity[-1] * (1.0 + (bnh_ret_pct / 100.0))
            bnh_equity.append(fold_bnh_equity)

            fold_results.append({
                "fold": fold + 1,
                "train_bars": len(train_df),
                "test_bars": len(test_df),
                "agent_return_pct": metrics["total_return_pct"],
                "bnh_return_pct": round(bnh_ret_pct, 2),
                "agent_final_equity": metrics["final_equity"]
            })

        # Calculate global performance metrics
        agent_equity_arr = np.array(agent_equity)
        bnh_equity_arr = np.array(bnh_equity)

        agent_rets = np.diff(agent_equity_arr) / agent_equity_arr[:-1]
        bnh_rets = np.diff(bnh_equity_arr) / bnh_equity_arr[:-1]

        agent_perf = calculate_performance_metrics(agent_rets, agent_equity_arr)
        bnh_perf = calculate_performance_metrics(bnh_rets, bnh_equity_arr)

        logger.info(f"Walk-Forward Backtest Complete across {len(fold_results)} folds. Agent Sharpe: {agent_perf['sharpe_ratio']}, B&H Sharpe: {bnh_perf['sharpe_ratio']}")

        return {
            "n_folds": len(fold_results),
            "agent_performance": agent_perf,
            "buy_and_hold_performance": bnh_perf,
            "fold_details": fold_results,
            "agent_equity_curve": agent_equity,
            "bnh_equity_curve": bnh_equity
        }

    def export_backtest_reports(
        self,
        results: Dict[str, Any],
        json_path: str = "artifacts/reports/backtest_results.json",
        plot_path: str = "artifacts/reports/plots/equity_curves.png"
    ):
        """Exports backtest JSON report and equity curve visualization."""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        plt.figure(figsize=(10, 4))
        plt.plot(results["agent_equity_curve"], label="Causal Digital Twin PPO", color="green", linewidth=2)
        plt.plot(results["bnh_equity_curve"], label="Buy & Hold EURUSD", color="gray", linestyle="--", linewidth=1.5)
        plt.title("Walk-Forward Out-of-Sample Equity Curves ($10,000 Initial)")
        plt.xlabel("Walk-Forward Test Fold")
        plt.ylabel("Portfolio Equity ($)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Exported backtest reports to '{json_path}' and equity plot to '{plot_path}'")
