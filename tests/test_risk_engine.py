"""
Comprehensive unit tests for RiskEngine.
"""
from unittest.mock import MagicMock
from src.safety.risk_engine import RiskEngine

def test_buy_proposal_calculation():
    engine = RiskEngine()
    proposal = engine.evaluate_proposal(
        action=0.5, current_price=1.0850, atr_price=0.0020, equity_usd=10000.0, current_spread_points=12.0
    )
    assert proposal.recommendation == "BUY"
    assert proposal.is_valid is True
    assert proposal.lot_size > 0.0
    assert proposal.stop_loss_price < 1.0850
    assert proposal.take_profit_price > 1.0850
    assert proposal.risk_pct_equity <= 0.005

def test_sell_proposal_calculation():
    engine = RiskEngine()
    proposal = engine.evaluate_proposal(
        action=-0.8, current_price=1.0850, atr_price=0.0020, equity_usd=10000.0, current_spread_points=12.0
    )
    assert proposal.recommendation == "SELL"
    assert proposal.is_valid is True
    assert proposal.stop_loss_price > 1.0850
    assert proposal.take_profit_price < 1.0850

def test_high_spread_rejection():
    engine = RiskEngine()
    proposal = engine.evaluate_proposal(
        action=0.5, current_price=1.0850, atr_price=0.0020, equity_usd=10000.0, current_spread_points=50.0
    )
    assert proposal.recommendation == "NO TRADE"
    assert proposal.is_valid is False
    assert "exceeds max threshold" in proposal.rejection_reason

def test_hold_proposal():
    engine = RiskEngine()
    proposal = engine.evaluate_proposal(
        action=0.05, current_price=1.0850, atr_price=0.0020, equity_usd=10000.0, existing_position_lot=0.1
    )
    assert proposal.recommendation == "HOLD"
    assert proposal.lot_size == 0.0

def test_buy_stop_above_entry_rejected():
    engine = RiskEngine()
    proposal = engine.evaluate_proposal(
        action=0.5, current_price=1.0850, atr_price=0.0020, equity_usd=10000.0, custom_stop_price=1.0900
    )
    assert proposal.recommendation == "NO TRADE"
    assert proposal.is_valid is False
    assert "strictly below" in proposal.rejection_reason

def test_sell_stop_below_entry_rejected():
    engine = RiskEngine()
    proposal = engine.evaluate_proposal(
        action=-0.5, current_price=1.0850, atr_price=0.0020, equity_usd=10000.0, custom_stop_price=1.0800
    )
    assert proposal.recommendation == "NO TRADE"
    assert proposal.is_valid is False
    assert "strictly above" in proposal.rejection_reason

def test_volume_smaller_than_min_rejected():
    engine = RiskEngine()
    proposal = engine.evaluate_proposal(
        action=0.16, current_price=1.0850, atr_price=0.0500, equity_usd=100.0, volume_min=0.1
    )
    assert proposal.recommendation == "NO TRADE"
    assert proposal.is_valid is False
    assert "below broker minimum" in proposal.rejection_reason

def test_volume_step_rounding():
    engine = RiskEngine()
    proposal = engine.evaluate_proposal(
        action=1.0, current_price=1.0850, atr_price=0.0020, equity_usd=10000.0, volume_step=0.01
    )
    # Check that lot_size is exact multiple of 0.01
    assert round(proposal.lot_size % 0.01, 4) == 0.0

def test_margin_exceeding_free_margin_rejected():
    engine = RiskEngine()
    proposal = engine.evaluate_proposal(
        action=1.0, current_price=1.0850, atr_price=0.0015, equity_usd=100000.0, free_margin_usd=10.0
    )
    assert proposal.recommendation == "NO TRADE"
    assert proposal.is_valid is False
    assert "exceeds free margin" in proposal.rejection_reason

def test_mock_mt5_order_calc_profit_fallback():
    mock_mt5 = MagicMock()
    mock_mt5.order_calc_profit.return_value = 300.0  # $300 loss per 1.0 lot
    mock_mt5.order_calc_margin.return_value = 1000.0

    engine = RiskEngine(mt5_interface=mock_mt5)
    proposal = engine.evaluate_proposal(
        action=0.5, current_price=1.0850, atr_price=0.0020, equity_usd=10000.0
    )
    assert proposal.is_valid is True
    mock_mt5.order_calc_profit.assert_called()
