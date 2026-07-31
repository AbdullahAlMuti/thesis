"""
Risk Management Engine for EURUSD Position Sizing and Order Filtering.
Converts scalar continuous action in [-1.0, 1.0] into XM-compliant lot size.
Supports broker API cross-checks via order_calc_profit() and order_calc_margin().
"""
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

from src.config import RiskPolicy

logger = logging.getLogger(__name__)

@dataclass
class RiskProposal:
    action: float
    recommendation: str  # BUY, SELL, HOLD, NO TRADE
    target_exposure: float
    lot_size: float
    stop_loss_price: float
    take_profit_price: float
    stop_distance_pips: float
    risk_amount_usd: float
    risk_pct_equity: float
    required_margin_usd: float
    is_valid: bool
    rejection_reason: str = ""


class RiskEngine:
    """
    Risk control engine enforcing conservative risk limits and converting model target exposure
    into XM MetaTrader 5 lot sizes.
    """

    def __init__(self, policy: Optional[RiskPolicy] = None, mt5_interface: Optional[Any] = None):
        self.policy = policy or RiskPolicy()
        self.mt5 = mt5_interface

    def calculate_lot_loss(
        self,
        symbol: str,
        order_type: int,  # 0 for BUY, 1 for SELL
        entry_price: float,
        stop_price: float,
        contract_size: float = 100000.0
    ) -> float:
        """
        Calculates loss in USD for 1.0 lot position from entry to stop price.
        Tries MT5 order_calc_profit first; falls back to analytical formula if MT5 is None or returns None.
        """
        if self.mt5 is not None:
            profit = self.mt5.order_calc_profit(order_type, symbol, 1.0, entry_price, stop_price)
            if profit is not None:
                return abs(float(profit))

        # Analytical fallback for EURUSD standard lot:
        # Loss = Lot * Contract_Size * |entry - stop|
        dist = abs(entry_price - stop_price)
        return float(contract_size * dist)

    def calculate_lot_margin(
        self,
        symbol: str,
        order_type: int,
        entry_price: float,
        lot_size: float,
        leverage: float = 100.0,
        contract_size: float = 100000.0
    ) -> float:
        """
        Calculates required margin in USD for given lot size.
        """
        if self.mt5 is not None and lot_size > 0:
            margin = self.mt5.order_calc_margin(order_type, symbol, lot_size, entry_price)
            if margin is not None:
                return float(margin)

        # Analytical fallback for EURUSD:
        # Margin = Lot * Contract_Size * entry_price / leverage
        if leverage <= 0:
            leverage = 100.0
        return float(lot_size * contract_size * entry_price / leverage)

    def evaluate_proposal(
        self,
        action: float,
        current_price: float,
        atr_price: float,
        equity_usd: float,
        free_margin_usd: Optional[float] = None,
        current_spread_points: float = 15.0,
        existing_position_lot: float = 0.0,
        pip_size: float = 0.0001,
        contract_size: float = 100000.0,
        volume_min: float = 0.01,
        volume_max: float = 100.0,
        volume_step: float = 0.01,
        stops_level_points: int = 10,
        max_allowed_spread_points: float = 35.0,
        custom_stop_price: Optional[float] = None
    ) -> RiskProposal:
        """
        Evaluates continuous action in [-1.0, 1.0] and returns structured RiskProposal.
        """
        clamped_action = max(-1.0, min(1.0, float(action)))
        free_margin = free_margin_usd if free_margin_usd is not None else equity_usd

        # Recommendation Classification
        if clamped_action >= 0.15:
            recommendation = "BUY"
            direction = 1.0
            order_type = 0  # ORDER_TYPE_BUY
        elif clamped_action <= -0.15:
            recommendation = "SELL"
            direction = -1.0
            order_type = 1  # ORDER_TYPE_SELL
        else:
            if abs(existing_position_lot) > 1e-5:
                recommendation = "HOLD"
            else:
                recommendation = "NO TRADE"
            direction = 0.0
            order_type = -1

        if recommendation in ("HOLD", "NO TRADE"):
            return RiskProposal(
                action=clamped_action,
                recommendation=recommendation,
                target_exposure=clamped_action,
                lot_size=0.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                stop_distance_pips=0.0,
                risk_amount_usd=0.0,
                risk_pct_equity=0.0,
                required_margin_usd=0.0,
                is_valid=True,
                rejection_reason=""
            )

        # Check 1: Spread threshold
        if current_spread_points > max_allowed_spread_points:
            return RiskProposal(
                action=clamped_action,
                recommendation="NO TRADE",
                target_exposure=clamped_action,
                lot_size=0.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                stop_distance_pips=0.0,
                risk_amount_usd=0.0,
                risk_pct_equity=0.0,
                required_margin_usd=0.0,
                is_valid=False,
                rejection_reason=f"Spread ({current_spread_points} points) exceeds max threshold ({max_allowed_spread_points} points)"
            )

        # Calculate Stop Loss and Take Profit Prices
        if custom_stop_price is not None:
            stop_loss_price = custom_stop_price
            stop_dist_price = abs(current_price - stop_loss_price)
        else:
            stop_dist_price = max(atr_price * 1.5, 0.0015)
            if direction > 0:
                stop_loss_price = current_price - stop_dist_price
            else:
                stop_loss_price = current_price + stop_dist_price

        # Check 2: Invalid Stop Direction
        if direction > 0 and stop_loss_price >= current_price:
            return RiskProposal(
                action=clamped_action, recommendation="NO TRADE", target_exposure=clamped_action,
                lot_size=0.0, stop_loss_price=stop_loss_price, take_profit_price=0.0, stop_distance_pips=0.0,
                risk_amount_usd=0.0, risk_pct_equity=0.0, required_margin_usd=0.0, is_valid=False,
                rejection_reason="BUY stop-loss price must be strictly below entry price."
            )
        if direction < 0 and stop_loss_price <= current_price:
            return RiskProposal(
                action=clamped_action, recommendation="NO TRADE", target_exposure=clamped_action,
                lot_size=0.0, stop_loss_price=stop_loss_price, take_profit_price=0.0, stop_distance_pips=0.0,
                risk_amount_usd=0.0, risk_pct_equity=0.0, required_margin_usd=0.0, is_valid=False,
                rejection_reason="SELL stop-loss price must be strictly above entry price."
            )

        stop_distance_pips = stop_dist_price / pip_size
        min_stops_pips = (stops_level_points * (pip_size * 0.1)) / pip_size
        if stop_distance_pips < min_stops_pips:
            return RiskProposal(
                action=clamped_action, recommendation="NO TRADE", target_exposure=clamped_action,
                lot_size=0.0, stop_loss_price=stop_loss_price, take_profit_price=0.0, stop_distance_pips=stop_distance_pips,
                risk_amount_usd=0.0, risk_pct_equity=0.0, required_margin_usd=0.0, is_valid=False,
                rejection_reason=f"Stop distance ({stop_distance_pips:.1f} pips) is smaller than broker stops level ({min_stops_pips:.1f} pips)."
            )

        if direction > 0:
            take_profit_price = current_price + (stop_dist_price * self.policy.min_reward_to_risk)
        else:
            take_profit_price = current_price - (stop_dist_price * self.policy.min_reward_to_risk)

        # Risk budget calculation
        desired_risk_usd = equity_usd * self.policy.max_risk_per_trade_pct * abs(clamped_action)
        loss_per_lot = self.calculate_lot_loss("EURUSD", order_type, current_price, stop_loss_price, contract_size)

        if loss_per_lot <= 0:
            raw_lot = 0.0
        else:
            raw_lot = desired_risk_usd / loss_per_lot

        # Round DOWN to volume step
        steps = math.floor(raw_lot / volume_step)
        quantized_lot = steps * volume_step
        quantized_lot = round(quantized_lot, 2)

        if quantized_lot < volume_min:
            return RiskProposal(
                action=clamped_action, recommendation="NO TRADE", target_exposure=clamped_action,
                lot_size=0.0, stop_loss_price=round(stop_loss_price, 5), take_profit_price=round(take_profit_price, 5),
                stop_distance_pips=round(stop_distance_pips, 1), risk_amount_usd=0.0, risk_pct_equity=0.0,
                required_margin_usd=0.0, is_valid=False,
                rejection_reason=f"Calculated position size ({quantized_lot} lots) is below broker minimum ({volume_min} lots)"
            )

        final_lot = min(quantized_lot, volume_max)
        actual_risk_usd = final_lot * loss_per_lot
        actual_risk_pct = actual_risk_usd / equity_usd

        # Margin cross-check
        req_margin = self.calculate_lot_margin("EURUSD", order_type, current_price, final_lot, leverage=100.0, contract_size=contract_size)
        if req_margin > free_margin:
            return RiskProposal(
                action=clamped_action, recommendation="NO TRADE", target_exposure=clamped_action,
                lot_size=0.0, stop_loss_price=round(stop_loss_price, 5), take_profit_price=round(take_profit_price, 5),
                stop_distance_pips=round(stop_distance_pips, 1), risk_amount_usd=0.0, risk_pct_equity=0.0,
                required_margin_usd=req_margin, is_valid=False,
                rejection_reason=f"Required margin (${req_margin:.2f}) exceeds free margin (${free_margin:.2f})."
            )

        return RiskProposal(
            action=clamped_action,
            recommendation=recommendation,
            target_exposure=clamped_action,
            lot_size=final_lot,
            stop_loss_price=round(stop_loss_price, 5),
            take_profit_price=round(take_profit_price, 5),
            stop_distance_pips=round(stop_distance_pips, 1),
            risk_amount_usd=round(actual_risk_usd, 2),
            risk_pct_equity=round(actual_risk_pct, 4),
            required_margin_usd=round(req_margin, 2),
            is_valid=True,
            rejection_reason=""
        )
