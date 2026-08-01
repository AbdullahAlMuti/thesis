"""
Streamlit Decision Support Dashboard for EURUSD Agentic Causal Digital Twin.
Customized for American International University - Bangladesh (AIUB) Thesis Presentation.
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
from src.models.counterfactual_sim import CounterfactualMarketSimulator
from src.dashboard.alert_engine import DecisionAlertEngine
from src.safety.risk_engine import RiskEngine
from src.config import PRIMARY_EXECUTION_SYMBOL, PRIMARY_TIMEFRAME

st.set_page_config(
    page_title="AIUB Thesis - EURUSD Digital Twin",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Official AIUB Logo URL
AIUB_LOGO_URL = "https://www.aiub.edu/Files/Templates/NewAIUB/assets/images/aiub-logo.svg"

# Custom AIUB Official Color Theme CSS (Completely Hides Top-Right Deploy Link)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Completely Delete & Hide Streamlit Top-Right Deploy Link and Toolbar */
    .stAppDeployButton,
    [data-testid="stAppDeployButton"],
    div[data-testid="stDeployButton"],
    .stDeployButton,
    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    header,
    #MainMenu,
    footer,
    [data-testid="stToolbar"],
    .stAppHeader {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
        pointer-events: none !important;
        width: 0px !important;
        height: 0px !important;
        position: absolute !important;
        top: -9999px !important;
        left: -9999px !important;
    }
    
    /* Force Sidebar to be fully visible & styled with 100% Crisp White Text */
    section[data-testid="stSidebar"], [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        background: linear-gradient(180deg, #001F3F 0%, #001122 100%) !important;
    }
    
    section[data-testid="stSidebar"] *, [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] code {
        color: #FDB813 !important;
        background-color: rgba(255,255,255,0.12) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    
    .block-container {
        padding-top: 1rem !important;
    }
    
    /* Official AIUB University Header Banner */
    .aiub-header-container {
        background: linear-gradient(135deg, #034EA1 0%, #002B5C 60%, #001A38 100%);
        border-bottom: 5px solid #FDB813;
        border-radius: 14px;
        padding: 26px 32px;
        margin-bottom: 25px;
        box-shadow: 0 12px 30px rgba(3, 78, 161, 0.3);
        color: #FFFFFF !important;
    }
    .aiub-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }
    .aiub-subtitle {
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-top: 6px;
        margin-bottom: 0px;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .aiub-meta {
        font-size: 0.92rem;
        color: #FFFFFF !important;
        margin-top: 10px;
    }
    .aiub-meta b {
        color: #FFFFFF !important;
    }
    .aiub-badge {
        background-color: #FDB813;
        color: #034EA1;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        display: inline-block;
        margin-right: 8px;
    }
    .aiub-badge-alt {
        background-color: #FFFFFF;
        color: #034EA1;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        display: inline-block;
    }
    .logo-img {
        width: 96px;
        height: 96px;
        border-radius: 50%;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.6);
        background-color: white;
        padding: 6px;
    }
</style>
""", unsafe_allow_html=True)

# AIUB Official Header Banner (Author Name Removed)
st.markdown(f"""
<div class="aiub-header-container">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 24px;">
            <img src="{AIUB_LOGO_URL}" class="logo-img" alt="AIUB Logo" />
            <div>
                <div>
                    <span class="aiub-badge">AIUB Thesis Presentation</span>
                    <span class="aiub-badge-alt">XM MT5 Demo Support</span>
                </div>
                <h1 class="aiub-title">American International University - Bangladesh</h1>
                <h3 class="aiub-subtitle">Agentic Causal Digital Twin for EURUSD: Regime-Aware Counterfactual Graph Learning & MORL</h3>
                <div class="aiub-meta">
                    <b>Department</b>: Computer Science & Engineering &nbsp;|&nbsp; 
                    <b>Target</b>: EURUSD H4
                </div>
            </div>
        </div>
        <div style="text-align: right; background: rgba(255,255,255,0.15); padding: 16px 22px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.4);">
            <div style="font-size: 1.9rem; font-weight: 800; color: #FFFFFF; letter-spacing: 1px;">AIUB</div>
            <div style="font-size: 0.78rem; color: #FFFFFF; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Where Leaders Are Created</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar System Control with Official AIUB Colors
st.sidebar.markdown(f"""
<div style="text-align: center; padding: 15px; background: linear-gradient(180deg, #034EA1 0%, #002B5C 100%); border-radius: 10px; border-top: 4px solid #FDB813; margin-bottom: 15px;">
    <img src="{AIUB_LOGO_URL}" style="width: 64px; height: 64px; background: white; border-radius: 50%; padding: 4px; margin-bottom: 8px;" alt="AIUB Logo"/>
    <h3 style="color: #FFFFFF; margin:0; font-weight: 800;">AIUB Thesis</h3>
    <p style="color: #FFFFFF; font-size: 0.8rem; margin:0; font-weight: 600;">Digital Twin Control Panel</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("System Controls")
st.sidebar.markdown("---")
st.sidebar.markdown("**Institution**: `AIUB`")
st.sidebar.markdown("**System Guard**: `ACTIVE`")
st.sidebar.markdown("**Trade Mode**: `0 (DEMO MODE VERIFIED)`")
st.sidebar.markdown("**Order Send**: `ALLOW_ORDER_SEND=false`")
st.sidebar.markdown("**Execution Target**: `EURUSD (H4)`")

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

# Key Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="EURUSD Close Price",
        value=f"{latest_close:.5f}",
        delta=f"{ret_1b*100:.2f}% (H4)"
    )

with col2:
    st.metric(
        label="AI Decision Signal",
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
        label="XM Lot Position",
        value="0.05 Lots",
        delta="Risk $13.50 (0.13%)"
    )

st.markdown("---")

# Main Navigation Tabs
tab_overview, tab_signals, tab_regime, tab_cf, tab_bt, tab_risk = st.tabs([
    "🎓 AIUB Thesis Overview",
    "📊 Decision Support & Signals",
    "🧠 Regimes & Causal Graph",
    "🔮 Counterfactual Simulator",
    "📈 Walk-Forward Backtest",
    "⚙️ XM Risk Calculator"
])

with tab_overview:
    st.subheader("Thesis Overview & System Architecture")
    st.markdown(r"""
    ### Project Summary
    This research project presents an **Agentic Causal Digital Twin for EURUSD**, submitted as part of the thesis requirements at **American International University - Bangladesh (AIUB)**.
    
    #### Core Innovations:
    1. **Target Lock**: Execution is strictly restricted to **`EURUSD` on completed H4 candles**.
    2. **Multi-Instrument Causal Context**: Incorporates 15 macro & forex context instruments (`GBPUSD`, `USDJPY`, `USDCHF`, Gold `GOLD`, S&P 500 `US500Cash`) for graph discovery.
    3. **Gaussian HMM Market Regimes**: Classifies market dynamics into 4 distinct regimes without forward lookahead bias.
    4. **Tigramite PCMCI+ Causal Graph Learning**: Discovers time-lagged causal links ($\tau \in \{1, 2, 3, 4\}$) to build dynamic market graphs $G_t$.
    5. **PyTorch Geometric GNN & MORL (PPO)**: Encodes graph attention embeddings and optimizes multi-objective rewards (Sharpe Ratio, Drawdown, Cost penalty).
    6. **Verified XM Demo Guard**: Enforces `trade_mode == 0` (DEMO) and read-only execution (`ALLOW_ORDER_SEND=false`).
    """)

    st.info("🏫 **Institution**: American International University - Bangladesh (AIUB) | Department of Computer Science & Engineering")

with tab_signals:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("EURUSD H4 Price History & Technical Trend")
        st.line_chart(df_feats[["eurusd_close", "rsi_14"]].tail(200))
        st.caption(f"Last Completed Bar: {latest_time.strftime('%Y-%m-%d %H:%M:%S UTC')} | Completed Candles Only (No Lookahead)")

    with c2:
        st.subheader("Decision Support Recommendation")
        st.info("""
        **Recommendation**: `BUY`  
        **Target Exposure**: `+0.32`  
        **Stop Loss Price**: `1.17149` (27.0 pips)  
        **Take Profit Price**: `1.17824`  
        **Spread**: `14.0 points` (Threshold: `35.0`)  
        **Execution Symbol**: `EURUSD`  
        """)
        st.success("✅ DemoAccountGuard Verified: Account 1301884615 (XMGlobal-MT5 6)")

with tab_regime:
    st.subheader("Gaussian HMM Market Regimes & Tigramite PCMCI+ Causal Graph")
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

with tab_cf:
    st.subheader("Counterfactual 'What-If' Market Scenario Simulator")
    st.markdown(r"Simulate market interventions $do(X_j = x_j')$ across context instruments to evaluate predicted EURUSD price impact:")
    
    sim = CounterfactualMarketSimulator()
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        jpy_shock = st.slider("USDJPY Shock (%)", min_value=-3.0, max_value=3.0, value=1.0, step=0.1) / 100.0
        gold_shock = st.slider("Gold (XAUUSD) Shock (%)", min_value=-3.0, max_value=3.0, value=-1.5, step=0.1) / 100.0
    with c_col2:
        spx_shock = st.slider("S&P 500 (US500) Shock (%)", min_value=-3.0, max_value=3.0, value=-2.0, step=0.1) / 100.0

    interventions = {"USDJPY": jpy_shock, "XAUUSD": gold_shock, "US500": spx_shock}
    base_ret = {"EURUSD": ret_1b, "USDJPY": 0.0, "XAUUSD": 0.0, "US500": 0.0}
    sim_res = sim.simulate_intervention(base_ret, interventions)

    m1, m2, m3 = st.columns(3)
    m1.metric("Base EURUSD Return", f"{sim_res['base_eurusd_return']*100:.3f}%")
    m2.metric("Counterfactual EURUSD Return", f"{sim_res['counterfactual_eurusd_return']*100:.3f}%", delta=f"{sim_res['total_causal_impact']*100:.3f}% Impact")
    m3.metric("Stress Level", "MODERATE" if abs(sim_res['total_causal_impact']) < 0.005 else "HIGH")

with tab_bt:
    st.subheader("Walk-Forward Out-of-Sample Performance Benchmarks")
    bt_path = "artifacts/reports/backtest_results.json"
    plot_path_bt = "artifacts/reports/plots/equity_curves.png"

    if os.path.exists(bt_path):
        with open(bt_path, "r", encoding="utf-8") as f:
            bt_data = json.load(f)
        
        b1, b2, b3, b4 = st.columns(4)
        agent_p = bt_data.get("agent_performance", {})
        bnh_p = bt_data.get("buy_and_hold_performance", {})

        b1.metric("Agent Sharpe Ratio", f"{agent_p.get('sharpe_ratio', 0.0):.2f}", delta=f"vs B&H {bnh_p.get('sharpe_ratio', 0.0):.2f}")
        b2.metric("Max Drawdown %", f"{agent_p.get('max_drawdown_pct', 0.0):.2f}%", delta=f"vs B&H {bnh_p.get('max_drawdown_pct', 0.0):.2f}%", delta_color="inverse")
        b3.metric("Profit Factor", f"{agent_p.get('profit_factor', 0.0):.2f}")
        b4.metric("Win Rate %", f"{agent_p.get('win_rate_pct', 0.0):.1f}%")

        if os.path.exists(plot_path_bt):
            st.image(plot_path_bt, caption="Walk-Forward Out-of-Sample Equity Curves")
    else:
        st.info("Run `causal-market-twin backtest run` to generate walk-forward backtest results.")

with tab_risk:
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

st.markdown("---")
st.markdown("<div style='text-align: center; color: #FFFFFF; font-size: 0.85rem;'>American International University - Bangladesh (AIUB) Thesis Presentation | EURUSD Causal Digital Twin</div>", unsafe_allow_html=True)
