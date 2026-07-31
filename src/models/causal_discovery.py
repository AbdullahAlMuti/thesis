"""
Causal Graph Discovery Engine using Tigramite PCMCI+ algorithm with Partial Correlation (ParCorr).
Discovers time-lagged causal links between context instruments and target EURUSD.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

logger = logging.getLogger(__name__)


class CausalGraphDiscoverer:
    """
    Time-Lagged Causal Discovery Engine using Tigramite PCMCI+.
    """

    def __init__(self, tau_max: int = 4, alpha_level: float = 0.05):
        self.tau_max = tau_max
        self.alpha_level = alpha_level
        self.val_matrix: Optional[np.ndarray] = None
        self.p_matrix: Optional[np.ndarray] = None
        self.var_names: List[str] = []

    def prepare_causal_dataframe(self, aligned_df: pd.DataFrame) -> Tuple[pp.DataFrame, List[str]]:
        """
        Extracts 1-bar log return series for target EURUSD and context symbols.
        """
        df = aligned_df.copy()
        if "timestamp_utc" in df.columns:
            df = df.set_index("timestamp_utc")

        target_col = "eurusd_return_1b" if "eurusd_return_1b" in df.columns else None
        if target_col is None:
            # Fallback to computing eurusd_return_1b
            close_col = "eurusd_close" if "eurusd_close" in df.columns else df.columns[0]
            df["eurusd_return_1b"] = np.log(df[close_col] / df[close_col].shift(1))
            target_col = "eurusd_return_1b"

        # Collect context return columns
        ret_cols = [c for c in df.columns if c.endswith("_return_1b")]
        if target_col not in ret_cols:
            ret_cols.insert(0, target_col)
        else:
            # Ensure target is first
            ret_cols.remove(target_col)
            ret_cols.insert(0, target_col)

        sub_df = df[ret_cols].dropna()
        var_names = [c.replace("_return_1b", "").upper() for c in ret_cols]

        dataframe = pp.DataFrame(sub_df.values, datatime=np.arange(len(sub_df)), var_names=var_names)
        self.var_names = var_names
        return dataframe, var_names

    def run_pcmci(self, aligned_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes PCMCI causal discovery with ParCorr partial correlation test.
        """
        dataframe, var_names = self.prepare_causal_dataframe(aligned_df)
        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

        results = pcmci.run_pcmci(tau_max=self.tau_max, pc_alpha=0.1)
        self.val_matrix = results["val_matrix"]
        self.p_matrix = results["p_matrix"]

        # Parse significant links targeting EURUSD (var index 0)
        target_idx = 0
        significant_links = []
        num_vars = len(var_names)

        for src_idx in range(num_vars):
            for tau in range(1, self.tau_max + 1):
                val = float(self.val_matrix[src_idx, target_idx, tau])
                pval = float(self.p_matrix[src_idx, target_idx, tau])
                
                if pval <= self.alpha_level and abs(val) > 1e-4:
                    significant_links.append({
                        "source": var_names[src_idx],
                        "target": var_names[target_idx],
                        "lag": tau,
                        "correlation": round(val, 4),
                        "p_value": round(pval, 6)
                    })

        logger.info(f"PCMCI Causal Discovery Complete. Discovered {len(significant_links)} significant links targeting {var_names[0]}.")
        return {
            "var_names": var_names,
            "tau_max": self.tau_max,
            "alpha_level": self.alpha_level,
            "total_links_found": len(significant_links),
            "significant_links": significant_links
        }

    def export_causal_reports(self, results: Dict[str, Any], json_path: str = "artifacts/reports/causal_graph.json", plot_path: str = "artifacts/reports/plots/causal_network.png"):
        """Exports JSON report and network visualization plot."""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Plot causal link strength heatmap
        plt.figure(figsize=(8, 5))
        links = results["significant_links"]
        if links:
            sources = [f"{l['source']} (lag {l['lag']})" for l in links]
            corrs = [l['correlation'] for l in links]
            colors = ["green" if c > 0 else "red" for c in corrs]
            plt.barh(sources, corrs, color=colors)
            plt.axvline(0, color="gray", linestyle="--")
            plt.title("Causal Impact Strengths on EURUSD (PCMCI+ Lags 1-4)")
            plt.xlabel("Partial Correlation Link Strength")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        logger.info(f"Exported causal graph reports to '{json_path}' and plot to '{plot_path}'")
