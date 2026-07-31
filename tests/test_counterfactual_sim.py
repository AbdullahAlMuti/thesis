"""
Unit tests for CounterfactualMarketSimulator.
"""
import pytest
from src.models.counterfactual_sim import CounterfactualMarketSimulator

def test_counterfactual_intervention():
    sim = CounterfactualMarketSimulator()
    base_returns = {"EURUSD": 0.0010, "USDJPY": 0.0000, "XAUUSD": 0.0000}
    interventions = {"USDJPY": 0.0100}  # +1% USDJPY spike

    res = sim.simulate_intervention(base_returns, interventions)
    assert "counterfactual_eurusd_return" in res
    assert "total_causal_impact" in res
    # USDJPY weight is -0.35, so +0.0100 spike causes -0.0035 EURUSD impact
    assert round(res["total_causal_impact"], 4) == -0.0035

def test_stress_test_grid():
    sim = CounterfactualMarketSimulator()
    base_returns = {"EURUSD": 0.0005, "USDJPY": 0.0010, "XAUUSD": -0.0005}
    grid_df = sim.run_stress_test_grid(base_returns)

    assert len(grid_df) == 6
    assert "scenario_name" in grid_df.columns
    assert "total_impact" in grid_df.columns
