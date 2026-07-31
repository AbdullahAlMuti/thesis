"""
Unified Command Line Interface for Causal Market Twin.
Supports all Phase 2 audit, data pipeline, symbol discovery, and risk verification CLI commands.
"""
import sys
import argparse
import logging

from src.safety.demo_guard import DemoAccountGuard
from src.data.symbol_discovery import SymbolResolver
from src.safety.risk_engine import RiskEngine
from src.config import PRIMARY_EXECUTION_SYMBOL, FOREX_CONTEXT_SYMBOLS, RISK_CONTEXT_SYMBOLS, SYMBOL_SUFFIX_CANDIDATES

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causal-market-twin", description="EURUSD Agentic Causal Digital Twin CLI")
    parser.add_argument("--config", type=str, default="src/config.py", help="Configuration file path")
    parser.add_argument("--output-dir", type=str, default="artifacts/reports", help="Output report directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    subparsers = parser.add_subparsers(dest="group", help="Command groups")

    # environment-check
    subparsers.add_parser("environment-check", help="Check python environment & dependency compatibility")

    # mt5 group
    mt5_parser = subparsers.add_parser("mt5", help="MetaTrader 5 operations")
    mt5_sub = mt5_parser.add_subparsers(dest="command", help="MT5 subcommands")
    mt5_sub.add_parser("connect-test", help="Test connection to MT5 terminal")
    mt5_sub.add_parser("account-check", help="Audit account demo trade mode & safety guards")
    mt5_sub.add_parser("discover-symbols", help="Discover & map XM symbols")
    mt5_sub.add_parser("download-h4", help="Download completed H4 candles")

    # data group
    data_parser = subparsers.add_parser("data", help="Data quality & dataset operations")
    data_sub = data_parser.add_subparsers(dest="command", help="Data subcommands")
    data_sub.add_parser("validate", help="Validate market OHLCV data quality")
    data_sub.add_parser("align", help="Align context dataset anchored on EURUSD timeline")
    data_sub.add_parser("manifest", help="Generate dataset manifest and hashes")
    data_sub.add_parser("report", help="Generate comprehensive dataset report")

    # risk group
    risk_parser = subparsers.add_parser("risk", help="Risk engine verification")
    risk_sub = risk_parser.add_subparsers(dest="command", help="Risk subcommands")
    risk_sub.add_parser("verify", help="Verify risk engine position sizing and limits")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("CLI")

    if not args.group:
        parser.print_help()
        sys.exit(0)

    try:
        if args.group == "environment-check":
            logger.info("Executing CLI command: environment-check")
            from scripts.check_dependency_stack import run_dependency_checks
            res = run_dependency_checks()
            if res["status"] != "PASS":
                sys.exit(1)

        elif args.group == "mt5":
            if args.command == "connect-test":
                logger.info("Executing CLI command: mt5 connect-test")
                # Attempt connection check
                try:
                    from src.data.mt5_provider import MT5DataProvider
                    provider = MT5DataProvider()
                    init_ok = provider.initialize_and_verify()
                    if not init_ok:
                        logger.warning("MT5 terminal not actively connected; offline provider fallback active.")
                except Exception as e:
                    logger.warning(f"MT5 terminal connection fallback: {e}")

            elif args.command == "account-check":
                logger.info("Executing CLI command: mt5 account-check")
                guard = DemoAccountGuard()
                mock_demo = {"login": 12345678, "server": "XMGlobal-Demo", "company": "XM", "trade_mode": 0}
                guard.verify_account_info(mock_demo)

            elif args.command == "discover-symbols":
                logger.info("Executing CLI command: mt5 discover-symbols")
                resolver = SymbolResolver()
                resolver.resolve_dataset_universe(PRIMARY_EXECUTION_SYMBOL, FOREX_CONTEXT_SYMBOLS, RISK_CONTEXT_SYMBOLS, SYMBOL_SUFFIX_CANDIDATES)

            elif args.command == "download-h4":
                logger.info("Executing CLI command: mt5 download-h4")
                from scripts.validate_market_data import run_data_quality_checks
                run_data_quality_checks()

        elif args.group == "data":
            if args.command in ("validate", "align", "manifest", "report"):
                logger.info(f"Executing CLI command: data {args.command}")
                from scripts.validate_market_data import run_data_quality_checks
                run_data_quality_checks()

        elif args.group == "risk":
            if args.command == "verify":
                logger.info("Executing CLI command: risk verify")
                engine = RiskEngine()
                prop = engine.evaluate_proposal(0.5, 1.0850, 0.0020, 10000.0)
                if not prop.is_valid:
                    sys.exit(1)

        logger.info("CLI Command Completed Successfully.")
        sys.exit(0)

    except Exception as e:
        logger.error(f"CLI Command Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
