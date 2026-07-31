"""
Comprehensive unit tests for DemoAccountGuard and Read-Only enforcement.
"""
import pytest
from unittest.mock import MagicMock
from src.safety.demo_guard import DemoAccountGuard, SecurityViolationError, UnsupportedAccountTypeError

def test_demo_account_accepted():
    guard = DemoAccountGuard(enforce_demo=True)
    valid_demo = {
        "login": 12345678,
        "server": "XMGlobal-Demo3",
        "company": "Trading Point of Financial Instruments Ltd",
        "trade_mode": 0,  # DEMO
        "trade_allowed": True
    }
    assert guard.verify_account_info(valid_demo) is True

def test_real_account_rejected():
    guard = DemoAccountGuard(enforce_demo=True)
    real_info = {
        "login": 87654321,
        "server": "XMGlobal-Real",
        "company": "XM Global",
        "trade_mode": 2,  # REAL
        "trade_allowed": True
    }
    with pytest.raises(SecurityViolationError) as exc_info:
        guard.verify_account_info(real_info)
    assert "REAL (LIVE) ACCOUNT" in str(exc_info.value)

def test_contest_account_rejected():
    guard = DemoAccountGuard(enforce_demo=True)
    contest_info = {
        "login": 55554444,
        "server": "XMGlobal-Contest",
        "company": "XM Global",
        "trade_mode": 1,  # CONTEST
        "trade_allowed": True
    }
    with pytest.raises(UnsupportedAccountTypeError):
        guard.verify_account_info(contest_info)

def test_unknown_mode_rejected():
    guard = DemoAccountGuard(enforce_demo=True)
    unknown_info = {
        "login": 99998888,
        "server": "XMGlobal-Demo",
        "company": "XM Global",
        "trade_mode": 99,  # UNKNOWN
        "trade_allowed": True
    }
    with pytest.raises(SecurityViolationError):
        guard.verify_account_info(unknown_info)

def test_missing_account_info_rejected():
    guard = DemoAccountGuard(enforce_demo=True)
    with pytest.raises(SecurityViolationError):
        guard.verify_account_info({})

def test_server_containing_demo_but_real_mode_rejected():
    guard = DemoAccountGuard(enforce_demo=True)
    fake_demo_server = {
        "login": 12341234,
        "server": "XMGlobal-DemoServerForLive",  # Server name tricks
        "company": "XM Global",
        "trade_mode": 2,  # REAL MODE!
        "trade_allowed": True
    }
    with pytest.raises(SecurityViolationError) as exc:
        guard.verify_account_info(fake_demo_server)
    assert "REAL (LIVE) ACCOUNT" in str(exc.value)

def test_allowlist_acceptance_and_rejection():
    guard = DemoAccountGuard(enforce_demo=True, allowed_logins=[12345678, 88888888])

    valid_demo = {
        "login": 12345678,
        "server": "XMGlobal-Demo",
        "company": "XM Global",
        "trade_mode": 0
    }
    assert guard.verify_account_info(valid_demo) is True

    unallowed_demo = {
        "login": 99999999,
        "server": "XMGlobal-Demo",
        "company": "XM Global",
        "trade_mode": 0
    }
    with pytest.raises(SecurityViolationError) as exc:
        guard.verify_account_info(unallowed_demo)
    assert "allowlist" in str(exc.value)

def test_read_only_order_send_disabled(monkeypatch):
    monkeypatch.setenv("ALLOW_ORDER_SEND", "false")
    guard = DemoAccountGuard()
    demo_info = {
        "login": 12345678,
        "server": "XMGlobal-Demo",
        "company": "XM Global",
        "trade_mode": 0
    }
    with pytest.raises(SecurityViolationError) as exc:
        guard.assert_order_allowed(demo_info, is_demo_order=True)
    assert "ALLOW_ORDER_SEND=false" in str(exc.value)

def test_order_send_never_called_in_phase2(monkeypatch):
    import MetaTrader5 as mt5
    mock_order_send = MagicMock()
    monkeypatch.setattr(mt5, "order_send", mock_order_send, raising=False)

    # Import and run Phase 2 main scaffold
    from src.main import run_scaffold_demo
    run_scaffold_demo()

    # Assert MetaTrader5.order_send was never invoked
    mock_order_send.assert_not_called()
