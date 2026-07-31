"""
Offline Data Provider and Synthetic Generator for EURUSD and Contextual Series.
Enforces leakage-safe time indexing and completed candle boundaries.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class OfflineDataProvider:
    """
    Provides historical or synthetic market data for offline research and unit testing.
    Guarantees lookahead-safe indexing.
    """

    def __init__(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None):
        """
        data_dict: Dictionary mapping symbol name (e.g. 'EURUSD') to OHLCV DataFrame.
        """
        self.data_dict: Dict[str, pd.DataFrame] = data_dict or {}

    @staticmethod
    def generate_synthetic_ohlcv(
        symbol: str = "EURUSD",
        start_date: str = "2023-01-01",
        num_bars: int = 1000,
        freq: str = "4h",
        start_price: float = 1.0850,
        volatility: float = 0.0015,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generates realistic synthetic OHLCV data for H4 timeframe.
        """
        np.random.seed(seed)
        timestamps = pd.date_range(start=start_date, periods=num_bars, freq=freq)

        returns = np.random.normal(loc=0.00005, scale=volatility, size=num_bars)
        price_paths = start_price * np.exp(np.cumsum(returns))

        df_list = []
        for i in range(num_bars):
            close = price_paths[i]
            high_extra = np.abs(np.random.normal(0, volatility * 0.5)) * close
            low_extra = np.abs(np.random.normal(0, volatility * 0.5)) * close

            high = max(close, close + high_extra)
            low = min(close, close - low_extra)
            open_p = low + np.random.uniform(0, 1) * (high - low)
            tick_vol = int(np.random.uniform(500, 5000))
            spread = int(np.random.uniform(10, 25))

            df_list.append({
                "time": timestamps[i],
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "tick_volume": tick_vol,
                "spread": spread
            })

        df = pd.DataFrame(df_list)
        df.set_index("time", inplace=True)
        logger.info(f"Generated synthetic dataset for '{symbol}' with {len(df)} bars.")
        return df

    def load_dataset(self, symbol: str) -> pd.DataFrame:
        """
        Loads dataset for a symbol, generating synthetic fallback if missing.
        """
        if symbol not in self.data_dict:
            logger.info(f"Symbol '{symbol}' not found in data_dict. Generating synthetic dataset.")
            self.data_dict[symbol] = self.generate_synthetic_ohlcv(symbol=symbol)
        return self.data_dict[symbol]

    def get_slice_up_to_time(self, symbol: str, current_time: pd.Timestamp) -> pd.DataFrame:
        """
        LEAKAGE-SAFE: Returns only completed bars with timestamp <= current_time.
        """
        df = self.load_dataset(symbol)
        sliced = df[df.index <= current_time].copy()
        return sliced
