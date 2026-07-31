"""
Unit tests for WalkForwardBacktester and performance metrics.
"""
import os
import pytest
import numpy as np
import pandas as pd
from src.models.backtest import WalkForwardBacktester, calculate_performance_metrics
from src.data.features import FeatureEngineeringPipeline
from src.data.offline_provider import OfflineDataProvider

def test_calculate_performance_metrics():
    returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
    equity = np.array([10000, 10100, 10049.5, 10250.49, 10404.25, 10300.21])

    metrics = calculate_performance_metrics(returns, equity)
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown_pct" in metrics
    assert "profit_factor" in metrics
    assert "win_rate_pct" in metrics
    assert "total_return_pct" in metrics
    assert metrics["win_rate_pct"] == 60.0

def test_walk_forward_backtester(tmp_path):
    provider = OfflineDataProvider()
    df = provider.generate_synthetic_ohlcv(symbol="EURUSD", num_bars=400)
    pipe = FeatureEngineeringPipeline()
    feats = pipe.build_feature_matrix(df, fit_hmm=True)

    tester = WalkForwardBacktester(df_features=feats, n_folds=3, initial_balance=10000.0)
    results = tester.run_walk_forward()

    assert "n_folds" in results
    assert "agent_performance" in results
    assert "buy_and_hold_performance" in results
    assert len(results["fold_details"]) == 3

    json_p = os.path.join(tmp_path, "backtest_results.json")
    plot_p = os.path.join(tmp_path, "equity_curves.png")
    tester.export_backtest_reports(results, json_path=json_p, plot_path=plot_p)

    assert os.path.exists(json_p)
    assert os.path.exists(plot_p)
