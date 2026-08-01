"""
Strengthened Demo Account Safety Guard for MetaTrader 5 Execution.
MANDATORY DEMO-ONLY ACCOUNT RESTRICTION GUARD & READ-ONLY ENFORCEMENT.
"""
import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class SecurityViolationError(Exception):
    """Raised when an operation violates trading safety rules (e.g. attempting live execution or unallowed login)."""
    pass

class UnsupportedAccountTypeError(Exception):
    """Raised when account type is unsupported (e.g. contest mode)."""
    pass


def redact_sensitive_str(text: str) -> str:
    """Redacts sensitive information like logins, passwords, tokens, and account names."""
    if not text:
        return ""
    # Redact login digits except last 4
    if text.isdigit() and len(text) > 4:
        return "*" * (len(text) - 4) + text[-4:]
    return text[:2] + "***" if len(text) > 2 else "***"


class DemoAccountGuard:
    """
    Demo Account Safety Guard.
    Strictly verifies MT5 account_info.trade_mode == 0 (DEMO).
    """

    TRADE_MODE_DEMO = 0
    TRADE_MODE_CONTEST = 1
    TRADE_MODE_REAL = 2

    def __init__(self, enforce_demo: bool = True, allowed_logins: Optional[List[int]] = None):
        self.enforce_demo = enforce_demo

        # Load allowed logins from argument or environment variable ALLOWED_DEMO_LOGINS
        env_logins = os.getenv("ALLOWED_DEMO_LOGINS", "")
        parsed_logins = []
        if env_logins:
            for item in env_logins.split(","):
                item_str = item.strip()
                if item_str.isdigit():
                    parsed_logins.append(int(item_str))

        self.allowed_logins = allowed_logins if allowed_logins is not None else parsed_logins

    def verify_account_info(self, account_info_dict: Dict[str, Any]) -> bool:
        """
        Verifies account metadata dictionary from MT5 account_info().

        Mandatory Conditions:
        1. account_info is not None and not empty.
        2. account_info.trade_mode == 0 (DEMO).
        3. account_info.login is valid integer > 0.
        4. account_info.server is non-empty string.
        5. account_info.company is non-empty string.
        """
        if not account_info_dict:
            raise SecurityViolationError("Account info is empty or None.")

        login = account_info_dict.get("login")
        server = account_info_dict.get("server", "")
        company = account_info_dict.get("company", "")
        trade_mode = account_info_dict.get("trade_mode")

        redacted_login = redact_sensitive_str(str(login))
        logger.info(f"Auditing MT5 Account: Login={redacted_login}, Server={server}, Company={company}, TradeMode={trade_mode}")

        # Check required non-empty fields
        if not login or not isinstance(login, int) or login <= 0:
            raise SecurityViolationError("Invalid or missing account login number.")
        if not server or not isinstance(server, str) or not server.strip():
            raise SecurityViolationError("Missing or invalid server name in account info.")
        if not company or not isinstance(company, str) or not company.strip():
            raise SecurityViolationError("Missing or invalid company name in account info.")

        # Trade mode strict check
        if trade_mode == self.TRADE_MODE_REAL:
            msg = (
                f"CRITICAL SECURITY VIOLATION: Account {redacted_login} on server {server} is a REAL (LIVE) ACCOUNT! "
                "Execution on live real-money accounts is strictly prohibited."
            )
            logger.critical(msg)
            raise SecurityViolationError(msg)

        if trade_mode == self.TRADE_MODE_CONTEST:
            msg = f"Account {redacted_login} is a CONTEST account (Trade Mode {trade_mode}). Contest accounts are unsupported."
            logger.error(msg)
            raise UnsupportedAccountTypeError(msg)

        if trade_mode != self.TRADE_MODE_DEMO:
            msg = f"Account {redacted_login} trade mode '{trade_mode}' is unknown. Failing closed."
            logger.error(msg)
            raise SecurityViolationError(msg)

        # Allowlist check if populated
        if self.allowed_logins:
            if login not in self.allowed_logins:
                msg = f"Account login {redacted_login} is not in the ALLOWED_DEMO_LOGINS allowlist."
                logger.error(msg)
                raise SecurityViolationError(msg)

        logger.info(f"Demo Account Guard: Account {redacted_login} verified successfully as DEMO mode.")
        return True

    def assert_order_allowed(self, account_info_dict: Dict[str, Any], is_demo_order: bool = True, allow_override: bool = False) -> None:
        """
        Pre-flight safety check before order submission.
        """
        allow_order_send = os.getenv("ALLOW_ORDER_SEND", "false").lower() == "true" or allow_override
        if not allow_order_send:
            raise SecurityViolationError("Order send is permanently disabled (ALLOW_ORDER_SEND=false). Set $env:ALLOW_ORDER_SEND='true' to enable demo order execution.")

        self.verify_account_info(account_info_dict)
        if not is_demo_order:
            raise SecurityViolationError("Non-demo order request rejected by DemoAccountGuard.")
