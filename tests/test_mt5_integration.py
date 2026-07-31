"""
Integration test for real MetaTrader 5 terminal connectivity.
Marked with @pytest.mark.mt5 to exclude from offline CI runs.
"""
import pytest

@pytest.mark.mt5
def test_real_mt5_terminal_connection():
    try:
        import MetaTrader5 as mt5
        init_ok = mt5.initialize()
        if not init_ok:
            pytest.skip("MT5 terminal is not open/connected on local machine.")

        acc = mt5.account_info()
        assert acc is not None
        assert hasattr(acc, "login")
        mt5.shutdown()
    except ImportError:
        pytest.skip("MetaTrader5 package is not available.")
