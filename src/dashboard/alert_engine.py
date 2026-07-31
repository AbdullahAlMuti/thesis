"""
Real-Time Decision Support Alert Engine for EURUSD Agentic Causal Digital Twin.
Evaluates newly closed H4 candles and emits structured trading decision support proposals.
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from src.safety.demo_guard import DemoAccountGuard
from src.models.regime_hmm import REGIME_NAMES
from src.safety.risk_engine import RiskEngine, RiskProposal

logger = logging.getLogger(__name__)


class DecisionAlertEngine:
    """
    Real-Time Decision Support Alert Engine.
    """

    def __init__(self, log_filepath: str = "artifacts/reports/live_alerts.json"):
        self.log_filepath = log_filepath
        self.risk_engine = RiskEngine()
        self.demo_guard = DemoAccountGuard(enforce_demo=True)
        self.alerts_history: List[Dict[str, Any]] = []

    def evaluate_live_candle(
        self,
        current_price: float,
        action_val: float,
        regime_id: int,
        regime_prob: float,
        equity_usd: float = 10000.0,
        spread_points: float = 14.0,
        atr_price: float = 0.0020
    ) -> Dict[str, Any]:
        """
        Evaluates newly closed H4 candle and generates structured decision proposal.
        """
        # Evaluate XM Risk Proposal
        proposal: RiskProposal = self.risk_engine.evaluate_proposal(
            action=action_val,
            current_price=current_price,
            atr_price=atr_price,
            equity_usd=equity_usd,
            current_spread_points=spread_points
        )

        regime_label = REGIME_NAMES.get(regime_id, "Low-Vol Consolidation")
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        alert = {
            "timestamp_utc": timestamp_utc,
            "target_symbol": "EURUSD",
            "timeframe": "H4",
            "current_price": round(current_price, 5),
            "recommendation": proposal.recommendation,
            "target_exposure": round(proposal.target_exposure, 2),
            "lot_size": proposal.lot_size,
            "stop_loss_price": proposal.stop_loss_price,
            "take_profit_price": proposal.take_profit_price,
            "risk_amount_usd": proposal.risk_amount_usd,
            "risk_pct_equity": proposal.risk_pct_equity,
            "regime": regime_label,
            "regime_probability": round(regime_prob, 4),
            "is_valid": proposal.is_valid,
            "rejection_reason": proposal.rejection_reason,
            "safety_guard": "DemoAccountGuard ACTIVE (ALLOW_ORDER_SEND=false)"
        }

        self.alerts_history.append(alert)
        self._append_to_file(alert)

        logger.info(f"Generated Live Alert [{timestamp_utc}]: {alert['recommendation']} EURUSD ({proposal.lot_size} lots) | Regime: {regime_label}")
        return alert

    def _append_to_file(self, alert: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
        data = []
        if os.path.exists(self.log_filepath):
            try:
                with open(self.log_filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []

        data.append(alert)
        with open(self.log_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
