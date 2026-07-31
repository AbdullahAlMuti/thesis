"""
MetaTrader 5 Live/Demo Connection and Data Fetcher.
STRICT LEAKAGE SAFETY: Fetches completed bars only (excluding index 0 forming candle).
"""
import logging
from typing import Optional, Dict, Any
import pandas as pd

from src.safety.demo_guard import DemoAccountGuard

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not installed. Live MT5 provider will operate in offline/mock state.")


class MT5DataProvider:
    """
    Interface for MetaTrader 5 terminal connection and bar history retrieval.
    Enforces Demo Account Guard on login.
    """

    def __init__(self, demo_guard: Optional[DemoAccountGuard] = None):
        self.demo_guard = demo_guard or DemoAccountGuard(enforce_demo=True)
        self.is_connected = False
        self.account_info_dict: Dict[str, Any] = {}

    def initialize_and_verify(self, path: Optional[str] = None) -> bool:
        """
        Initializes MT5 terminal connection and audits account safety.
        """
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 Python module is not installed.")

        init_kwargs = {}
        if path:
            init_kwargs["path"] = path

        if not mt5.initialize(**init_kwargs):
            err = mt5.last_error()
            logger.error(f"MT5 initialize failed: {err}")
            return False

        account = mt5.account_info()
        if account is None:
            logger.error("Failed to retrieve MT5 account info.")
            mt5.shutdown()
            return False

        self.account_info_dict = account._asdict()

        # AUDIT DEMO SAFETY GUARD
        self.demo_guard.verify_account_info(self.account_info_dict)

        self.is_connected = True
        logger.info(f"MT5 Connected & Verified DEMO Account: Login={self.account_info_dict.get('login')}")
        return True

    def fetch_completed_h4_rates(self, symbol: str, num_bars: int = 1000) -> pd.DataFrame:
        """
        Fetches completed H4 candles from MT5.

        CRITICAL LEAKAGE-PREVENTION RULE:
        Position start is set to 1 (NOT 0).
        Position 0 is the currently forming candle and must NEVER be used for signals.
        Position 1 is the most recently completed H4 candle.
        """
        if not self.is_connected:
            raise RuntimeError("MT5DataProvider is not initialized or connected.")

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 1, num_bars)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to fetch H4 rates for symbol '{symbol}': {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        logger.info(f"Fetched {len(df)} completed H4 bars for '{symbol}' ending at {df.index[-1]}")
        return df

    def shutdown(self) -> None:
        """Disconnects MT5 terminal connection."""
        if MT5_AVAILABLE and self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            logger.info("MT5 connection shut down successfully.")
