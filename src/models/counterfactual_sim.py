"""
Counterfactual Market Simulator using Structural Causal Equations.
Simulates What-If market interventions do(X_j = x_j') and calculates counterfactual EURUSD impacts.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class CounterfactualMarketSimulator:
    """
    Simulates counterfactual market scenarios using causal structural equations.
    """

    def __init__(self, causal_coefficients: Optional[Dict[str, float]] = None):
        # Default structural causal weights if none provided (derived from PCMCI discovery)
        self.causal_weights = causal_coefficients or {
            "GBPUSD": 0.45,
            "USDJPY": -0.35,
            "USDCHF": -0.40,
            "XAUUSD": 0.25,
            "US500": -0.20
        }

    def simulate_intervention(
        self,
        base_returns: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Simulates counterfactual intervention: do(X_j = delta_j).
        Calculates expected change in EURUSD return.
        """
        base_eurusd_ret = base_returns.get("EURUSD", 0.0)
        cf_returns = base_returns.copy()
        
        # Apply interventions do(X_j = val)
        for sym, val in interventions.items():
            cf_returns[sym] = val

        # Structural Causal Equation for EURUSD:
        # \Delta Y = \sum w_j * (do(X_j) - X_j_base)
        causal_impact = 0.0
        details = {}

        for sym, w in self.causal_weights.items():
            if sym in interventions:
                delta_x = interventions[sym] - base_returns.get(sym, 0.0)
                impact_j = w * delta_x
                causal_impact += impact_j
                details[sym] = {
                    "base_return": base_returns.get(sym, 0.0),
                    "intervened_return": interventions[sym],
                    "weight": w,
                    "impact": round(impact_j, 6)
                }

        counterfactual_eurusd_ret = base_eurusd_ret + causal_impact

        logger.info(f"Counterfactual Simulation: Base EURUSD={base_eurusd_ret:.5f} -> CF EURUSD={counterfactual_eurusd_ret:.5f} (Impact: {causal_impact:+.5f})")

        return {
            "base_eurusd_return": round(base_eurusd_ret, 6),
            "counterfactual_eurusd_return": round(counterfactual_eurusd_ret, 6),
            "total_causal_impact": round(causal_impact, 6),
            "interventions": interventions,
            "impact_breakdown": details
        }

    def run_stress_test_grid(self, base_returns: Dict[str, float]) -> pd.DataFrame:
        """
        Runs a grid of stress-test counterfactual scenarios (e.g. Gold shock, S&P 500 shock, USDJPY shock).
        """
        scenarios = [
            {"name": "USDJPY Spike (+1.0%)", "interventions": {"USDJPY": 0.010}},
            {"name": "USDJPY Drop (-1.0%)", "interventions": {"USDJPY": -0.010}},
            {"name": "Gold Rally (+2.0%)", "interventions": {"XAUUSD": 0.020}},
            {"name": "Gold Crash (-2.0%)", "interventions": {"XAUUSD": -0.020}},
            {"name": "S&P500 Selloff (-3.0%)", "interventions": {"US500": -0.030}},
            {"name": "Combined Risk-Off", "interventions": {"USDJPY": 0.015, "US500": -0.025, "XAUUSD": 0.015}}
        ]

        results = []
        for sc in scenarios:
            res = self.simulate_intervention(base_returns, sc["interventions"])
            results.append({
                "scenario_name": sc["name"],
                "base_eurusd_return": res["base_eurusd_return"],
                "cf_eurusd_return": res["counterfactual_eurusd_return"],
                "total_impact": res["total_causal_impact"]
            })

        return pd.DataFrame(results)
