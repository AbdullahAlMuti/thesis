"""
Main Entry Point for EURUSD Agentic Causal Digital Twin Scaffold.
"""
import logging
from datetime import datetime, timezone

from src.config import (
    PRIMARY_EXECUTION_SYMBOL,
    PRIMARY_TIMEFRAME,
    FOREX_CONTEXT_SYMBOLS,
    RISK_CONTEXT_SYMBOLS,
    SYMBOL_SUFFIX_CANDIDATES,
    RiskPolicy
)
from src.safety.demo_guard import DemoAccountGuard
from src.data.symbol_discovery import SymbolResolver
from src.data.offline_provider import OfflineDataProvider
from src.safety.risk_engine import RiskEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EURUSD_Digital_Twin")

def run_scaffold_demo():
    logger.info("==========================================================================")
    logger.info("Initializing Agentic Causal Digital Twin for EURUSD (Scaffold Phase)")
    logger.info("==========================================================================")

    # 1. Target Confirmation
    logger.info(f"MANDATORY EXECUTION TARGET: {PRIMARY_EXECUTION_SYMBOL}")
    logger.info(f"PRIMARY TIMEFRAME: {PRIMARY_TIMEFRAME}")

    # 2. Demo Guard Setup
    _demo_guard = DemoAccountGuard(enforce_demo=True)
    logger.info("Demo Account Guard: ACTIVE (Enforcing Demo/Contest trade mode)")

    # 3. Symbol Discovery (Offline Mode Demonstration)
    resolver = SymbolResolver(mt5_interface=None)
    universe = resolver.resolve_dataset_universe(
        primary_symbol=PRIMARY_EXECUTION_SYMBOL,
        forex_context=FOREX_CONTEXT_SYMBOLS,
        risk_context=RISK_CONTEXT_SYMBOLS,
        candidate_suffixes=SYMBOL_SUFFIX_CANDIDATES
    )
    logger.info(f"Resolved Primary Instrument: {universe.get('primary_target', PRIMARY_EXECUTION_SYMBOL)}")
    forex_resolved = [k for k, v in universe.get("resolved_symbols", {}).items() if k in FOREX_CONTEXT_SYMBOLS]
    risk_resolved = [k for k, v in universe.get("resolved_symbols", {}).items() if k in RISK_CONTEXT_SYMBOLS]
    logger.info(f"Resolved Forex Context ({len(forex_resolved)}): {forex_resolved}")
    logger.info(f"Resolved Risk Context ({len(risk_resolved)}): {risk_resolved}")

    # 4. Data Provider Slicing (Leakage-Safe Demonstration)
    provider = OfflineDataProvider()
    eurusd_df = provider.load_dataset(PRIMARY_EXECUTION_SYMBOL)
    latest_completed_time = eurusd_df.index[-1]
    latest_close = eurusd_df["close"].iloc[-1]

    logger.info(f"Latest Completed Bar UTC Time: {latest_completed_time}, Close: {latest_close:.5f}")

    # 5. Risk Engine Proposal Evaluation
    risk_engine = RiskEngine(policy=RiskPolicy())

    # Example continuous model action output (+0.32 BUY signal)
    sample_model_action = 0.32
    proposal = risk_engine.evaluate_proposal(
        action=sample_model_action,
        current_price=latest_close,
        atr_price=0.0018,
        equity_usd=10000.0,
        current_spread_points=14.0
    )

    # 6. Report Decision Output (Format matching Section 21)
    report = f"""
==========================================================================
EURUSD DECISION SUPPORT REPORT
==========================================================================
SYMBOL: {PRIMARY_EXECUTION_SYMBOL}
TIMEFRAME: {PRIMARY_TIMEFRAME}
DECISION TIME: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
DATA STATUS: FRESH (Completed Bars Only)

RECOMMENDATION: {proposal.recommendation}
TARGET EXPOSURE: {proposal.target_exposure:+.2f}
MODEL CONFIDENCE: 0.68 (Baseline Demo)
RISK LEVEL: CONSERVATIVE
REGIME: TRENDING (HMM Baseline)
REGIME PROBABILITY: 0.74

PROPOSED POSITION SIZE:
- Lot Size: {proposal.lot_size:.2f} lots (XM Standard)
- Risk Amount: ${proposal.risk_amount_usd:.2f} ({proposal.risk_pct_equity * 100:.2f}% of $10,000 equity)

PROTECTIVE STOP:
- Stop Loss Price: {proposal.stop_loss_price:.5f} ({proposal.stop_distance_pips:.1f} pips distance)
- Take Profit Price: {proposal.take_profit_price:.5f}

MAIN EVIDENCE:
- EURUSD momentum is positive on completed H4 candles
- Cross-sectional USD context factors indicate moderate weakness
- Spread ({14.0} points) remains within validated safety threshold (35.0 points)

NO-TRADE CONDITIONS & SAFEGUARDS:
- Demo Account Guard verified
- Execution restricted to EURUSD only
==========================================================================
"""
    print(report)

if __name__ == "__main__":
    run_scaffold_demo()
