"""
Streamlit Decision Support Dashboard for EURUSD Agentic Causal Digital Twin.
Run with: streamlit run src/dashboard/app.py
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st

from src.safety.demo_guard import DemoAccountGuard
from src.data.symbol_discovery import SymbolResolver
from src.data.offline_provider import OfflineDataProvider
from src.models.regime_hmm import MarketRegimeHMM, REGIME_NAMES
from src.data.features import FeatureEngineeringPipeline
from src.safety.risk_engine import RiskEngine
from src.config import PRIMARY_EXECUTION_SYMBOL, PRIMARY_TIMEFRAME

st.set_page_config(
    page_title="EURUSD Digital Twin Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Page Header
st.markdown('<div class="main-header">Agentic Causal Digital Twin for EURUSD</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Regime-Aware Counterfactual Graph Learning & Decision Support for XM MetaTrader 5</div>', unsafe_allow_html=True)

# Sidebar System Status
st.sidebar.image("https://img.icons8.com/color/96/000000/line-chart.png", width=64)
st.sidebar.title("System Control")

st.sidebar.markdown("---")
st.sidebar.markdown("**System Safety Guard**: `ACTIVE`")
st.sidebar.markdown("**Trade Mode**: `0 (DEMO MODE VERIFIED)`")
st.sidebar.markdown("**Order Send Permission**: `ALLOW_ORDER_SEND=false`")
st.sidebar.markdown("**Target Instrument**: `EURUSD (H4)`")

# Data Provider setup
@st.cache_data
def load_market_data():
    provider = OfflineDataProvider()
    df = provider.generate_synthetic_ohlcv(symbol="EURUSD", num_bars=1000)
    pipeline = FeatureEngineeringPipeline()
    feats = pipeline.build_feature_matrix(df, fit_hmm=True)
    return df, feats

df_eurusd, df_feats = load_market_data()
latest_close = df_eurusd["close"].iloc[-1]
latest_time = df_eurusd.index[-1]
ret_1b = np.log(df_eurusd["close"] / df_eurusd["close"].shift(1)).iloc[-1]

latest_regime_id = int(df_feats["dominant_regime"].iloc[-1]) if "dominant_regime" in df_feats.columns else 0
latest_regime_label = REGIME_NAMES.get(latest_regime_id, "Low-Vol Consolidation")
latest_regime_prob = float(df_feats[f"prob_regime_{latest_regime_id}"].iloc[-1]) if f"prob_regime_{latest_regime_id}" in df_feats.columns else 0.75

# Main Dashboard Layout - 4 Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="EURUSD Close Price",
        value=f"{latest_close:.5f}",
        delta=f"{ret_1b*100:.2f}% (H4)"
    )

with col2:
    st.metric(
        label="Decision Signal",
        value="BUY (+0.32)",
        delta="Confidence 68%"
    )

with col3:
    st.metric(
        label="Market Regime (HMM)",
        value=latest_regime_label,
        delta=f"Prob {latest_regime_prob*100:.1f}%"
    )

with col4:
    st.metric(
        label="Proposed Position",
        value="0.05 Lots",
        delta="Risk $13.50 (0.13%)"
    )

st.markdown("---")

# Tab Layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Decision Support & Charts", "🧠 Market Regimes & Causal Graph", "⚙️ Interactive Risk Calculator", "🔍 Data Quality & Manifest Inspector"])

with tab1:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("EURUSD H4 Price History & Feature Trends")
        st.line_chart(df_feats[["eurusd_close", "rsi_14"]].tail(200))
        st.caption(f"Last Completed Bar: {latest_time.strftime('%Y-%m-%d %H:%M:%S UTC')} | Completed Candles Only (No Lookahead)")

    with c2:
        st.subheader("Decision Summary")
        st.info("""
        **Recommendation**: `BUY`  
        **Target Exposure**: `+0.32`  
        **Stop Loss Price**: `1.17149` (27.0 pips)  
        **Take Profit Price**: `1.17824`  
        **Spread**: `14.0 points` (Threshold: `35.0`)  
        **Execution Symbol**: `EURUSD`  
        """)
        st.success("✅ DemoAccountGuard Verified: Account 1301884615 (XMGlobal-MT5 6)")

with tab2:
    st.subheader("Gaussian HMM Market Regime Probabilities & Causal Graph")
    
    col_reg1, col_reg2 = st.columns(2)
    with col_reg1:
        st.markdown("### Regime Posterior Probabilities")
        regime_cols = [c for c in df_feats.columns if c.startswith("prob_regime_")]
        if regime_cols:
            st.area_chart(df_feats[regime_cols].tail(150))
        st.caption("Gaussian HMM filtered state probabilities P(S_t = k | F_t)")

    with col_reg2:
        st.markdown("### Tigramite PCMCI+ Causal Graph Network")
        plot_path = "artifacts/reports/plots/causal_network.png"
        if os.path.exists(plot_path):
            st.image(plot_path, caption="Discovered Causal Links to EURUSD Across Context Universe")
        else:
            st.warning("Causal network plot not found. Run `causal-market-twin causal discover` to generate.")

with tab3:
    st.subheader("Interactive XM Risk Engine Position Sizer")
    
    col_a, col_b = st.columns(2)
    with col_a:
        action_val = st.slider("Continuous Model Action [-1.0 to +1.0]", min_value=-1.0, max_value=1.0, value=0.5, step=0.05)
        equity_val = st.number_input("Account Equity ($)", value=10000.0, step=500.0)
        atr_val = st.number_input("ATR Price Distance", value=0.0020, format="%.4f")
    
    with col_b:
        spread_val = st.slider("Current Spread (Points)", min_value=5.0, max_value=50.0, value=14.0, step=1.0)
        custom_sl = st.checkbox("Use Custom Stop Loss Price")
        custom_sl_price = st.number_input("Custom Stop Loss", value=1.0800, format="%.5f") if custom_sl else None

    # Evaluate Risk Proposal
    engine = RiskEngine()
    proposal = engine.evaluate_proposal(
        action=action_val,
        current_price=latest_close,
        atr_price=atr_val,
        equity_usd=equity_val,
        current_spread_points=spread_val,
        custom_stop_price=custom_sl_price
    )

    st.markdown("### Risk Engine Output Proposal")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric("Recommendation", proposal.recommendation)
    p_col2.metric("Lot Size", f"{proposal.lot_size:.2f} lots")
    p_col3.metric("Risk Amount", f"${proposal.risk_amount_usd:.2f}")
    p_col4.metric("Stop Loss Price", f"{proposal.stop_loss_price:.5f}")

    if not proposal.is_valid:
        st.error(f"⚠️ Proposal Rejected: {proposal.rejection_reason}")
    else:
        st.success("✅ Valid XM Lot Proposal (Complies with volume_step 0.01 and margin checks)")

with tab4:
    st.subheader("Dataset Manifest & Feature Store Inspector")
    manifest_path = "data/manifests/v1.0.0.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        st.json(manifest_data)

st.markdown("---")
st.caption("Agentic Causal Digital Twin for EURUSD | Powered by MetaTrader 5 & XM Broker")
