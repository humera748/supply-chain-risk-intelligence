"""
app.py - Explainable Supply Chain Risk Intelligence & Adaptive Route Ranking
=============================================================================
B.Tech Mini Project Prototype
Focus: International Sea-Freight Long-Haul Delay Risk & Multi-Criteria Route Ranking

Run:
    streamlit run app.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Risk Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek Maritime & Executive Corporate CSS
st.markdown("""
<style>
    /* Global Font & Header Aesthetics */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Gradient Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hero-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0px;
        line-height: 1.4;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-sub {
        font-size: 12px;
        color: #0284c7;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Info & Callout Banners */
    .disclaimer-card {
        background: #f8fafc;
        border-left: 4px solid #0284c7;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 16px 0;
        font-size: 13px;
        color: #334155;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }

    .shipment-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 14px 0 20px 0;
    }
    .shipment-card-high {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 14px 0 20px 0;
    }

    /* Tag badges */
    .badge-high {
        background: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-ok {
        background: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Table customization */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Loading & Real Output Bridge
# ---------------------------------------------------------------------------
PORTS = [
    "Port of Los Angeles", "Port of New York & New Jersey", "Port of Savannah",
    "Port of San Juan", "Port of Houston", "Port of Rotterdam", "Port of Hamburg",
    "Port of Le Havre", "Port of Shanghai", "Port of Singapore", "Port of Sydney",
    "Port of Jawaharlal Nehru (Nhava Sheva/Mumbai)", "Port of Santos", "Port of Manzanillo"
]

CARRIER_BASE_RELIABILITY = {
    "Standard Class": 0.66,   # ~34% historical delay rate in DataCo
    "Second Class": 0.27,     # ~73% historical delay rate
    "First Class": 0.06,      # ~94% historical delay rate (tight 1-day deadline)
}

CARRIER_BASE_COST = {
    "Standard Class": 0.35,   # Low freight index
    "Second Class": 0.65,     # Medium freight index
    "First Class": 1.00,      # Premium expedited freight index
}


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Loads real model output from scored_shipments.csv or data/scored_shipments.csv.
    Falls back to synthetic proxy data only if neither file is found.
    """
    candidates = [
        Path("scored_shipments.csv"),
        Path("data/scored_shipments.csv"),
        Path(__file__).parent / "scored_shipments.csv",
        Path(__file__).parent / "data" / "scored_shipments.csv",
    ]
    
    for f in candidates:
        if f.exists():
            df = pd.read_csv(f, parse_dates=["booking_date"])
            return df

    # Fallback placeholder demo dataset if run without training
    rng = np.random.default_rng(42)
    n = 150
    placeholder_carriers = list(CARRIER_BASE_RELIABILITY.keys())
    events = ["None", "None", "None", "Port strike", "Storm warning", "Port congestion"]
    
    df = pd.DataFrame({
        "shipment_id": [f"SH-{10000 + i}" for i in range(n)],
        "booking_date": pd.Timestamp("2026-09-01") + pd.to_timedelta(rng.integers(0, 45, n), unit="D"),
        "origin_port": rng.choice(PORTS[:5], n),
        "carrier": rng.choice(placeholder_carriers, n),
        "scheduled_days": rng.choice([1, 2, 3, 4, 5], n),
        "active_event": rng.choice(events, n),
    })
    df["dest_port"] = [rng.choice(PORTS[5:]) for _ in range(n)]

    # Simulated risk
    c_risk = df.carrier.map({"Standard Class": 0.34, "Second Class": 0.72, "First Class": 0.93})
    event_risk = np.where(df.active_event == "None", 0, 0.15)
    noise = rng.normal(0, 0.05, n)
    df["risk"] = np.clip(c_risk + event_risk + noise, 0.02, 0.98).round(3)

    parts = pd.DataFrame({
        "Carrier mode delay history": c_risk,
        "Active event at port": event_risk,
        "Tight scheduled transit time": (4 - df.scheduled_days) * 0.1
    })
    df["top_factors"] = parts.apply(
        lambda r: " + ".join(r.sort_values(ascending=False).head(2).index), axis=1
    )
    return df


df = load_data()


# ---------------------------------------------------------------------------
# Route Alternatives Ranking Engine
# ---------------------------------------------------------------------------
def calculate_route_alternatives(row: pd.Series, w_risk: float, w_cost: float, w_rel: float) -> pd.DataFrame:
    """
    Ranks available carrier / shipping service options for a trade lane
    using a multi-criteria decision function:
        Score = w1 * Risk + w2 * Cost + w3 * (1 - Reliability)
    Lower score is better.
    """
    current_carrier = row["carrier"]
    current_risk = float(row["risk"])
    
    rows = []
    # Evaluate each available shipping mode/carrier alternative
    all_carriers = ["Standard Class", "Second Class", "First Class"]
    
    for c in all_carriers:
        is_cur = (c == current_carrier)
        
        # Risk estimation for alternative mode on this specific lane
        if is_cur:
            risk_val = current_risk
        else:
            # Shift risk relative to carrier baseline historical gap
            cur_base_delay = 1.0 - CARRIER_BASE_RELIABILITY.get(current_carrier, 0.5)
            alt_base_delay = 1.0 - CARRIER_BASE_RELIABILITY.get(c, 0.5)
            # Lane-adjusted delta
            delta = (alt_base_delay - cur_base_delay) * 0.8
            risk_val = float(np.clip(current_risk + delta, 0.03, 0.97))
            
        rel_val = CARRIER_BASE_RELIABILITY.get(c, 0.50)
        cost_val = CARRIER_BASE_COST.get(c, 0.50)

        rows.append({
            "Carrier": c,
            "Service Mode": "Ocean Economy" if c == "Standard Class" else ("Ocean Semi-Express" if c == "Second Class" else "Ocean Priority Express"),
            "Delay Risk": round(risk_val, 3),
            "Cost Index": round(cost_val, 2),
            "Reliability": round(rel_val, 2),
            "Current": is_cur
        })

    alt_df = pd.DataFrame(rows)

    # Normalize weights so sum = 1.0
    w_sum = w_risk + w_cost + w_rel
    if w_sum > 0:
        w1_n, w2_n, w3_n = w_risk / w_sum, w_cost / w_sum, w_rel / w_sum
    else:
        w1_n, w2_n, w3_n = 0.33, 0.33, 0.33

    # Transparent Multi-Criteria Ranking Formula
    alt_df["Score (lower = better)"] = (
        w1_n * alt_df["Delay Risk"] +
        w2_n * alt_df["Cost Index"] +
        w3_n * (1.0 - alt_df["Reliability"])
    ).round(3)

    alt_df = alt_df.sort_values("Score (lower = better)").reset_index(drop=True)
    alt_df["Rank"] = [f"#{i+1}" for i in range(len(alt_df))]
    return alt_df


# ---------------------------------------------------------------------------
# Sidebar Navigation & Global Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚢 Navigation & Filters")
    page = st.radio("Select View:", ["Live Risk Monitor", "Route Alternatives"], index=0)
    
    st.markdown("---")
    st.markdown("#### ⚙️ Risk Threshold")
    threshold = st.slider("High-risk threshold (Delay Probability)", 0.20, 0.90, 0.60, 0.05)
    st.caption("Shipments with predicted risk ≥ threshold are highlighted in red and queued for adaptive route ranking.")

    # Status column computation
    df["Status"] = np.where(df["risk"] >= threshold, "🔴 High risk", "🟢 OK")
    flagged = df[df["risk"] >= threshold].sort_values("risk", ascending=False)

    st.markdown("---")
    st.markdown("#### ℹ️ Project Context")
    st.caption(
        "**Explainable Supply Chain Risk Intelligence**\n\n"
        "B.Tech Mini Project prototype using time-split XGBoost modeling and SHAP explainability on sea-freight corridors."
    )


# ---------------------------------------------------------------------------
# PAGE 1: LIVE RISK MONITOR
# ---------------------------------------------------------------------------
if page == "Live Risk Monitor":
    # Top Hero Header
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">
            <span>🚢</span>
            <span>Live Supply Chain Delay Risk Monitor</span>
        </div>
        <p class="hero-subtitle">
            Real-time machine learning inference evaluated at booking time with strictly zero temporal or post-delivery leakage.
            Monitoring global container shipments across major maritime corridors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Shipments</div>
            <div class="metric-val">{len(df):,}</div>
            <div class="metric-sub">Active evaluated bookings</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        flagged_pct = (len(flagged) / len(df)) * 100 if len(df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Flagged High-Risk</div>
            <div class="metric-val" style="color: #dc2626;">{len(flagged):,}</div>
            <div class="metric-sub" style="color: #ef4444;">{flagged_pct:.1f}% exceed threshold ({threshold:.0%})</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        avg_risk = df["risk"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fleet Delay Risk</div>
            <div class="metric-val" style="color: #0284c7;">{avg_risk:.1%}</div>
            <div class="metric-sub">Expected average lateness</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        active_events = int((df["active_event"] != "None").sum())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Port Advisories</div>
            <div class="metric-val" style="color: #f59e0b;">{active_events}</div>
            <div class="metric-sub">Active weather/congestion alerts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Filter Bar
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 1.5])
    with f_col1:
        sel_carrier = st.multiselect("Carrier / Service Mode", sorted(df["carrier"].unique()))
    with f_col2:
        sel_origin = st.multiselect("Origin Gateway Port", sorted(df["origin_port"].unique()))
    with f_col3:
        sel_dest = st.multiselect("Destination Port", sorted(df["dest_port"].unique()))
    with f_col4:
        risk_filter = st.selectbox("Status Filter", ["All Shipments", "Flagged High Risk Only", "OK Only"])

    # Apply filters
    view = df.copy()
    if sel_carrier:
        view = view[view["carrier"].isin(sel_carrier)]
    if sel_origin:
        view = view[view["origin_port"].isin(sel_origin)]
    if sel_dest:
        view = view[view["dest_port"].isin(sel_dest)]
    if risk_filter == "Flagged High Risk Only":
        view = view[view["risk"] >= threshold]
    elif risk_filter == "OK Only":
        view = view[view["risk"] < threshold]

    view = view.sort_values("risk", ascending=False)

    st.subheader(f"📋 Live Scored Shipments ({len(view):,} matching)")
    st.caption("Model predictions computed strictly from booking-time features. Top factors derived from tree SHAP explainability.")

    # Dataframe with rich formatting
    display_cols = [
        "shipment_id", "Status", "risk", "carrier", "scheduled_days",
        "origin_port", "dest_port", "top_factors", "active_event"
    ]
    st.dataframe(
        view[display_cols],
        column_config={
            "shipment_id": st.column_config.TextColumn("Shipment ID", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "risk": st.column_config.ProgressColumn(
                "Delay Risk Probability",
                min_value=0.0,
                max_value=1.0,
                format="%.1f%%",
                width="medium"
            ),
            "carrier": st.column_config.TextColumn("Carrier (Shipping Mode)", width="medium"),
            "scheduled_days": st.column_config.NumberColumn("Scheduled Days", format="%d d"),
            "origin_port": st.column_config.TextColumn("Origin Gateway"),
            "dest_port": st.column_config.TextColumn("Destination Port"),
            "top_factors": st.column_config.TextColumn("Primary Delay Risk Drivers (SHAP Explanations)", width="large"),
            "active_event": st.column_config.TextColumn("Port Event", width="small"),
        },
        hide_index=True,
        width="stretch",
        height=400,
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Risk Intelligence Analytics Charts
    st.markdown("### 📊 Fleet Risk Analytics")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("#### Average Delay Risk by Carrier")
        carrier_risk = df.groupby("carrier")["risk"].mean().sort_values(ascending=False).reset_index()
        carrier_risk.columns = ["Carrier", "Avg Delay Risk"]
        st.bar_chart(carrier_risk.set_index("Carrier"), color="#0284c7")

    with ch2:
        st.markdown("#### Top 7 Origin Gateways by Delay Risk")
        origin_risk = df.groupby("origin_port")["risk"].mean().sort_values(ascending=False).head(7).reset_index()
        origin_risk.columns = ["Origin Port", "Avg Delay Risk"]
        st.bar_chart(origin_risk.set_index("Origin Port"), color="#3b82f6")

    # Expandable Academic Model Summary
    with st.expander("🔬 Academic Model Benchmark & Leakage Ablation Results (src/train.py)"):
        st.markdown("""
        **Experimental Benchmark on Cleaned Sea-Freight Dataset (156,197 Orders):**
        """)
        benchmark_table = pd.DataFrame([
            {"Split Protocol": "Chronological Time-based", "Model Architecture": "Logistic Regression Baseline (Class-Weighted)", "Accuracy": "70.35%", "Precision": "83.04%", "Recall": "57.69%", "F1": "0.6808", "ROC-AUC": "0.7361"},
            {"Split Protocol": "Chronological Time-based", "Model Architecture": "XGBoost Classifier (Class-Weighted)", "Accuracy": "70.24%", "Precision": "82.86%", "Recall": "57.62%", "F1": "0.6798", "ROC-AUC": "0.7325"},
            {"Split Protocol": "Chronological Time-based", "Model Architecture": "XGBoost + SMOTE Oversampling Experiment", "Accuracy": "56.98%", "Precision": "91.55%", "Recall": "23.71%", "F1": "0.3767", "ROC-AUC": "0.7050"},
            {"Split Protocol": "Random 80/20 Split", "Model Architecture": "XGBoost (Prior Literature Benchmark)", "Accuracy": "70.36%", "Precision": "84.17%", "Recall": "57.18%", "F1": "0.6810", "ROC-AUC": "0.7507"},
            {"Split Protocol": "Chronological Time-based", "Model Architecture": "XGBoost WITH Leaky Columns (ABLATION STUDY)", "Accuracy": "100.0%", "Precision": "100.0%", "Recall": "100.0%", "F1": "1.0000", "ROC-AUC": "1.0000"},
        ])
        st.table(benchmark_table)
        st.info(
            "💡 **Key Finding & Leakage Ablation Proof:** When leaky post-delivery columns (`Days for shipping (real)`, `Delivery Status`) "
            "are included, models score an artificial 100% ROC-AUC. Dropping them reveals the true real-world predictive frontier (~73.3% ROC-AUC). "
            "SMOTE degrades performance because class proportions (~45% / 55%) are naturally balanced, and synthetic samples introduce boundary distortion."
        )


# ---------------------------------------------------------------------------
# PAGE 2: ROUTE ALTERNATIVES
# ---------------------------------------------------------------------------
else:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">
            <span>🧭</span>
            <span>Adaptive Route Alternatives & Ranking</span>
        </div>
        <p class="hero-subtitle">
            Multi-criteria decision support for at-risk shipments. Transparently ranks service alternatives using
            a composite weighted scoring framework balancing Delay Risk, Cost, and Historical Reliability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if flagged.empty:
        st.success("🎉 No shipments currently exceed the high-risk threshold. All bookings are within acceptable operating parameters.")
        st.stop()

    # Shipment Selection
    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        shipment_options = flagged["shipment_id"].tolist()
        pick_id = st.selectbox(
            "Select Flagged High-Risk Shipment to Evaluate:",
            shipment_options,
            format_func=lambda s: f"{s} · {flagged.set_index('shipment_id').loc[s, 'carrier']} · {flagged.set_index('shipment_id').loc[s, 'risk']:.1%} delay risk"
        )
    with s_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.caption(f"Showing 1 of {len(flagged):,} flagged shipments")

    # Selected Shipment Inspection Card
    sel_row = df[df["shipment_id"] == pick_id].iloc[0]
    
    card_class = "shipment-card-high" if sel_row["risk"] >= threshold else "shipment-card"
    st.markdown(f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div style="font-size: 18px; font-weight: 700; color: #0f172a;">
                📦 {sel_row['shipment_id']} &nbsp;|&nbsp; 
                <span style="color: #0284c7;">{sel_row['origin_port']}</span> ➔ 
                <span style="color: #0f766e;">{sel_row['dest_port']}</span>
            </div>
            <span class="badge-high">Delay Risk: {sel_row['risk']:.1%}</span>
        </div>
        <div style="font-size: 14px; color: #334155; line-height: 1.6;">
            <strong>Current Carrier / Mode:</strong> {sel_row['carrier']} &nbsp;|&nbsp;
            <strong>Scheduled Transit:</strong> {sel_row['scheduled_days']} days &nbsp;|&nbsp;
            <strong>Booking Date:</strong> {sel_row['booking_date'].strftime('%Y-%m-%d')} &nbsp;|&nbsp;
            <strong>Active Event:</strong> {sel_row['active_event']}
        </div>
        <div style="font-size: 13px; color: #475569; margin-top: 8px; background: rgba(255,255,255,0.7); padding: 8px 12px; border-radius: 6px;">
            🔍 <strong>Explainable Drivers (SHAP):</strong> {sel_row['top_factors']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Decision Matrix Equation Banner
    st.markdown("### ⚖️ Multi-Criteria Decision Framework")
    st.info("📐 **Formula: Score = (w₁ × Delay Risk) + (w₂ × Cost Index) + (w₃ × (1 − Reliability))**")

    # Preset Profiles
    preset_cols = st.columns([1, 1, 1, 3])
    preset_mode = preset_cols[0].button("🛡️ Risk-Averse")
    preset_bal = preset_cols[1].button("⚖️ Balanced")
    preset_cost = preset_cols[2].button("💰 Cost Priority")

    # Weights initialization based on preset buttons
    if preset_mode:
        default_w1, default_w2, default_w3 = 0.70, 0.15, 0.15
    elif preset_bal:
        default_w1, default_w2, default_w3 = 0.40, 0.30, 0.30
    elif preset_cost:
        default_w1, default_w2, default_w3 = 0.20, 0.60, 0.20
    else:
        default_w1, default_w2, default_w3 = 0.50, 0.25, 0.25

    # Interactive Sliders
    sc1, sc2, sc3 = st.columns(3)
    w1 = sc1.slider("w₁ Delay Risk Weight", 0.0, 1.0, default_w1, 0.05)
    w2 = sc2.slider("w₂ Freight Cost Weight", 0.0, 1.0, default_w2, 0.05)
    w3 = sc3.slider("w₃ Historical Reliability Weight", 0.0, 1.0, default_w3, 0.05)

    # Compute Ranking Alternatives
    ranked_alts = calculate_route_alternatives(sel_row, w1, w2, w3)
    best_opt = ranked_alts.iloc[0]
    curr_opt = ranked_alts[ranked_alts["Current"]].iloc[0]

    # Actionable Recommendation Banner
    st.markdown("### 🏆 Ranked Carrier & Service Alternatives")
    if best_opt["Current"]:
        st.warning(
            f"⚠️ **Current Option ({best_opt['Carrier']}) is Already Optimal** under your chosen weight configuration "
            f"(Score: **{best_opt['Score (lower = better)']:.3f}**)."
        )
    else:
        risk_diff = curr_opt['Delay Risk'] - best_opt['Delay Risk']
        st.success(
            f"✅ **Recommended Switch to {best_opt['Carrier']} ({best_opt['Service Mode']})**: "
            f"Reduces delay risk from **{curr_opt['Delay Risk']:.1%}** to **{best_opt['Delay Risk']:.1%}** "
            f"(-{risk_diff:.1%} delay reduction) with a superior composite penalty score of **{best_opt['Score (lower = better)']:.3f}** "
            f"(vs {curr_opt['Score (lower = better)']:.3f} currently)."
        )

    # Render Ranked Table (NO MAP)
    display_ranked = ranked_alts.copy()
    display_ranked["Current Status"] = np.where(display_ranked["Current"], "📌 Current Choice", "Available Option")

    st.dataframe(
        display_ranked[[
            "Rank", "Current Status", "Carrier", "Service Mode",
            "Delay Risk", "Cost Index", "Reliability", "Score (lower = better)"
        ]],
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Current Status": st.column_config.TextColumn("Status", width="medium"),
            "Carrier": st.column_config.TextColumn("Carrier / Service Mode", width="medium"),
            "Service Mode": st.column_config.TextColumn("Service Category"),
            "Delay Risk": st.column_config.ProgressColumn(
                "Predicted Risk", min_value=0.0, max_value=1.0, format="%.1f%%", width="medium"
            ),
            "Cost Index": st.column_config.NumberColumn("Relative Cost Index", format="%.2f"),
            "Reliability": st.column_config.ProgressColumn(
                "Historical Reliability", min_value=0.0, max_value=1.0, format="%.0f%%"
            ),
            "Score (lower = better)": st.column_config.NumberColumn(
                "Composite Score (Lower = Better)", format="%.3f"
            ),
        },
        hide_index=True,
        width="stretch",
    )

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # About & Academic Limitations Box (Strict User Instruction)
    st.markdown("""
    <div class="disclaimer-card">
        <strong>📌 About & Project Limitations:</strong>
        <ul style="margin-top: 6px; margin-bottom: 0px; padding-left: 20px;">
            <li><strong>DataCo as a Sea Freight Proxy:</strong> DataCo Supply Chain records represent global cross-region commercial shipments. Sea freight transit times and container trade corridors are simulated using empirical order-to-destination geographic features.</li>
            <li><strong>Approximate Port Mapping:</strong> Because DataCo does not log maritime bill-of-lading terminal codes, seaports are approximately mapped via regional centroids to primary world commercial container ports (e.g. Los Angeles, San Juan, Le Havre, Rotterdam, Shanghai).</li>
            <li><strong>Carrier Equivalence:</strong> Carrier entities correspond directly to the empirical shipping service modes in DataCo (<code>Standard Class</code>, <code>Second Class</code>, <code>First Class</code>).</li>
            <li><strong>Risk-Aware Ranking:</strong> The decision engine performs <em>risk-aware route ranking</em> across discrete maritime carrier options rather than physical route optimization or navigation waypoint steering.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
