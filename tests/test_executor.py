"""
Unit tests for MT5OrderExecutor and Demo Account Safety Guard.
"""
import os
import pytest
from src.safety.executor import MT5OrderExecutor
from src.safety.demo_guard import DemoAccountGuard, SecurityViolationError
from src.safety.risk_engine import RiskProposal

def test_executor_rejects_allow_order_send_false(monkeypatch):
    monkeypatch.setenv("ALLOW_ORDER_SEND", "false")
    guard = DemoAccountGuard(enforce_demo=True)
    executor = MT5OrderExecutor(demo_guard=guard)

    demo_account = {"login": 12345678, "server": "XMGlobal-Demo", "company": "XM", "trade_mode": 0}
    proposal = RiskProposal(
        action=0.5, recommendation="BUY", target_exposure=0.5, lot_size=0.05,
        stop_loss_price=1.0800, take_profit_price=1.0900, stop_distance_pips=50.0,
        risk_amount_usd=25.0, risk_pct_equity=0.0025, required_margin_usd=50.0, is_valid=True
    )

    with pytest.raises(SecurityViolationError, match="ALLOW_ORDER_SEND=false"):
        executor.execute_demo_order(demo_account, proposal)

def test_executor_rejects_real_account(monkeypatch):
    monkeypatch.setenv("ALLOW_ORDER_SEND", "true")
    guard = DemoAccountGuard(enforce_demo=True)
    executor = MT5OrderExecutor(demo_guard=guard)

    real_account = {"login": 99999999, "server": "XMGlobal-Real", "company": "XM", "trade_mode": 2}
    proposal = RiskProposal(
        action=0.5, recommendation="BUY", target_exposure=0.5, lot_size=0.05,
        stop_loss_price=1.0800, take_profit_price=1.0900, stop_distance_pips=50.0,
        risk_amount_usd=25.0, risk_pct_equity=0.0025, required_margin_usd=50.0, is_valid=True
    )

    with pytest.raises(SecurityViolationError, match="REAL \(LIVE\) ACCOUNT"):
        executor.execute_demo_order(real_account, proposal)

def test_executor_mock_demo_success(monkeypatch):
    monkeypatch.setenv("ALLOW_ORDER_SEND", "true")
    guard = DemoAccountGuard(enforce_demo=True)
    executor = MT5OrderExecutor(demo_guard=guard)

    demo_account = {"login": 1301884615, "server": "XMGlobal-MT5 6", "company": "XM", "trade_mode": 0}
    proposal = RiskProposal(
        action=0.5, recommendation="BUY", target_exposure=0.5, lot_size=0.05,
        stop_loss_price=1.0800, take_profit_price=1.0900, stop_distance_pips=50.0,
        risk_amount_usd=25.0, risk_pct_equity=0.0025, required_margin_usd=50.0, is_valid=True
    )

    res = executor.execute_demo_order(demo_account, proposal)
    assert res["status"] in ("SUCCESS", "MOCK_SUCCESS")
    assert res["recommendation"] == "BUY"
    assert res["lot_size"] == 0.05
