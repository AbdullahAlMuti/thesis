"""
MetaTrader 5 Demo Order Execution Engine.
STRICT SAFETY ENFORCEMENT:
- Operates EXCLUSIVELY on Demo Accounts (trade_mode == 0).
- Rejects any real-money account (trade_mode == 2) or unallowed login.
- Enforces ALLOW_ORDER_SEND=true environment gate.
"""
import os
import logging
from typing import Dict, Any, Optional

from src.safety.demo_guard import DemoAccountGuard, SecurityViolationError
from src.safety.risk_engine import RiskProposal
from src.config import PRIMARY_EXECUTION_SYMBOL

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


class MT5OrderExecutor:
    """
    Executes trades on MetaTrader 5 terminal for DEMO accounts only.
    """

    def __init__(self, demo_guard: Optional[DemoAccountGuard] = None):
        self.demo_guard = demo_guard or DemoAccountGuard(enforce_demo=True)

    def execute_demo_order(
        self,
        account_info_dict: Dict[str, Any],
        proposal: RiskProposal,
        symbol: str = PRIMARY_EXECUTION_SYMBOL,
        magic_number: int = 130188
    ) -> Dict[str, Any]:
        """
        Executes a demo order on MT5 based on a validated RiskProposal.
        """
        # 1. Pre-flight Demo Safety Guard check
        self.demo_guard.assert_order_allowed(account_info_dict, is_demo_order=True)

        if not proposal.is_valid or proposal.lot_size <= 0:
            raise ValueError(f"Cannot execute invalid proposal: {proposal.rejection_reason}")

        if symbol != PRIMARY_EXECUTION_SYMBOL:
            raise SecurityViolationError(f"Symbol '{symbol}' is not allowed for execution. Locked to '{PRIMARY_EXECUTION_SYMBOL}'.")

        if not MT5_AVAILABLE or mt5 is None:
            logger.warning("MT5 module unavailable. Operating in mock execution mode.")
            return {
                "status": "MOCK_SUCCESS",
                "symbol": symbol,
                "recommendation": proposal.recommendation,
                "lot_size": proposal.lot_size,
                "sl": proposal.stop_loss_price,
                "tp": proposal.take_profit_price,
                "message": "Mock demo order executed successfully."
            }

        # 2. Fetch market tick from MT5 terminal (with mock fallback if terminal disconnected)
        tick = None
        try:
            tick = mt5.symbol_info_tick(symbol)
        except Exception as e:
            logger.warning(f"Error calling symbol_info_tick: {e}")

        if tick is None:
            logger.warning(f"MT5 tick for '{symbol}' unavailable. Returning mock demo execution response.")
            return {
                "status": "MOCK_SUCCESS",
                "symbol": symbol,
                "recommendation": proposal.recommendation,
                "lot_size": proposal.lot_size,
                "sl": proposal.stop_loss_price,
                "tp": proposal.take_profit_price,
                "message": "Mock demo order executed (MT5 terminal tick offline)."
            }

        if proposal.recommendation == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif proposal.recommendation == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            raise ValueError(f"Unsupported recommendation for order execution: {proposal.recommendation}")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": proposal.lot_size,
            "type": order_type,
            "price": price,
            "sl": proposal.stop_loss_price,
            "tp": proposal.take_profit_price,
            "deviation": 20,
            "magic": magic_number,
            "comment": "CausalDigitalTwin_Demo",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logger.info(f"Submitting MT5 DEMO Order Request: {proposal.recommendation} {proposal.lot_size} lots on {symbol} at {price:.5f}")
        result = mt5.order_send(request)

        if result is None:
            err = mt5.last_error()
            raise RuntimeError(f"MT5 order_send returned None. Error: {err}")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"MT5 Order Execution Failed with code {result.retcode}: {result.comment}")

        logger.info(f"MT5 DEMO Order Executed Successfully! Order Ticket #{result.order}")
        return {
            "status": "SUCCESS",
            "order_ticket": result.order,
            "deal_ticket": result.deal,
            "volume": result.volume,
            "price": result.price,
            "bid": result.bid,
            "ask": result.ask,
            "comment": result.comment
        }
