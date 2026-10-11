"""
Explainable Supply Chain Risk Intelligence & Adaptive Route Ranking (prototype dashboard)
Run:  streamlit run app_v2.py      (needs scored_shipments.csv in the same folder)
Required columns: shipment_id, origin_port, dest_port, carrier (= shipping mode), risk (0-1)
Optional columns: top_factors, scheduled_days, booking_date
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Supply Chain Risk", page_icon="🚢", layout="wide")

REQUIRED = ["shipment_id", "origin_port", "dest_port", "carrier", "risk"]
COST = {"Standard Class": 1.0, "Second Class": 1.15, "First Class": 1.35, "Same Day": 1.6}  # assumed
PRESETS = {"Safety first": (0.8, 0.05, 0.15), "Balanced": (0.5, 0.25, 0.25), "Lowest cost": (0.15, 0.7, 0.15)}
EXPLAIN = [
    ("scheduled|transit", "The promised delivery time is short, so there is little room for any delay."),
    ("lane|route", "Earlier shipments on this route were often delayed."),
    ("mode|service|class", "This shipping service has often been late in past orders."),
    ("event|strike|storm|congestion|weather", "A strike, storm or port congestion raises the chance of delay."),
    ("region|country|market", "Orders to this region were often delayed in the past."),
]


def explain(text):
    out = []
    for part in [p.strip() for p in str(text).split("+") if p.strip() and p.strip() != "-"]:
        sentence = next((s for k, s in EXPLAIN if re.search(k, part, re.I)), "This factor pushed the risk up.")
        out.append((part, sentence))
    return out


@st.cache_data
def load():
    f = Path("scored_shipments.csv")
    return pd.read_csv(f) if f.exists() else None


df = load()
if df is None:
    st.error("scored_shipments.csv not found. Put it in the same folder as this file.")
    st.stop()
if [c for c in REQUIRED if c not in df.columns]:
    st.error(f"Missing columns: {[c for c in REQUIRED if c not in df.columns]}")
    st.write("Columns found:", list(df.columns))
    st.stop()

df = df.copy()
df["risk"] = df["risk"].astype(float)
if df["risk"].max() > 1:
    df["risk"] /= 100
if "top_factors" not in df.columns:
    df["top_factors"] = "-"
df["carrier"] = df["carrier"].astype(str)
df["Route"] = df["origin_port"].astype(str) + " → " + df["dest_port"].astype(str)

st.sidebar.title("🚢 Shipment Risk")
page = st.sidebar.radio("Go to", ["Overview", "Shipment details", "Route alternatives", "How it works"])
thr = st.sidebar.slider("High-risk line", 0.3, 0.95, 0.75, 0.05, help="A shipment above this delay chance is called high risk.")
st.sidebar.caption("Risk = predicted chance of delay, using only information known at booking time.")


def level(r):
    return "🔴 High" if r >= thr else "🟡 Medium" if r >= thr - 0.2 else "🟢 Low"


df["Level"] = df["risk"].apply(level)
flagged = df[df["risk"] >= thr].sort_values("risk", ascending=False)

# ---------------- Overview ----------------
if page == "Overview":
    st.title("Which shipments are at risk?")
    st.write("Every shipment gets a delay risk at booking time, so problems can be handled before the ship sails.")
    a, b, c = st.columns(3)
    a.metric("Shipments checked", f"{len(df):,}")
    b.metric("High-risk shipments", f"{len(flagged):,}", f"{len(flagged) / len(df):.0%} of all", delta_color="off")
    c.metric("Average delay risk", f"{df['risk'].mean():.0%}")

    t1, t2, t3, t4 = st.tabs(["Riskiest shipments", "By service", "By route", "Risk spread"])
    with t1:
        mode = st.selectbox("Show only this service", ["All"] + sorted(df["carrier"].unique()))
        v = flagged if mode == "All" else flagged[flagged["carrier"] == mode]
        st.dataframe(
            v.head(25)[["shipment_id", "Level", "risk", "Route", "carrier", "top_factors"]].rename(
                columns={"shipment_id": "Shipment", "risk": "Delay risk", "carrier": "Service", "top_factors": "Main reasons"}),
            column_config={"Delay risk": st.column_config.ProgressColumn("Delay risk", min_value=0, max_value=1, format="percent")},
            hide_index=True, width="stretch")
        st.caption("25 riskiest shown. Open 'Shipment details' to see the reasons in plain words.")
    with t2:
        s = df.groupby("carrier")["risk"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        st.bar_chart(s["mean"])
        st.write(f"**{s.index[0]}** has the highest average delay risk ({s['mean'].iloc[0]:.0%}); "
                 f"**{s.index[-1]}** has the lowest ({s['mean'].iloc[-1]:.0%}).")
    with t3:
        r = df.groupby("Route")["risk"].agg(avg_risk="mean", shipments="count")
        r = r[r["shipments"] >= 10].sort_values("avg_risk", ascending=False).head(10).reset_index()
        st.dataframe(r.rename(columns={"avg_risk": "Average delay risk", "shipments": "Shipments"}),
                     column_config={"Average delay risk": st.column_config.ProgressColumn(
                         "Average delay risk", min_value=0, max_value=1, format="percent")},
                     hide_index=True, width="stretch")
        st.caption("Top 10 routes (with at least 10 shipments) by average delay risk.")
    with t4:
        counts, _ = np.histogram(df["risk"], bins=10, range=(0, 1))
        st.bar_chart(pd.Series(counts, index=[f"{i * 10}-{i * 10 + 10}%" for i in range(10)]))
        st.caption("How many shipments fall in each delay-risk band.")

# ---------------- Shipment details ----------------
elif page == "Shipment details":
    st.title("Why is this shipment risky?")
    if flagged.empty:
        st.success("No shipments above the high-risk line. Lower it in the sidebar.")
        st.stop()
    top = flagged.head(300).set_index("shipment_id")
    pick = st.selectbox("Choose a high-risk shipment", top.index,
                        format_func=lambda s: f"{s} | {top.loc[s, 'Route']} | {top.loc[s, 'risk']:.0%}")
    row = top.loc[pick]
    st.subheader(f"{row['Level']} risk: {row['risk']:.0%} chance of delay")
    st.progress(float(row["risk"]))
    x, y, z = st.columns(3)
    x.metric("From", row["origin_port"])
    y.metric("To", row["dest_port"])
    z.metric("Service", row["carrier"])

    st.markdown("### Why is it risky?")
    reasons = explain(row["top_factors"])
    if reasons:
        for name, sentence in reasons:
            st.markdown(f"- **{name}**: {sentence}")
    else:
        st.write("No reason text available in the data.")

    st.markdown("### How does it compare?")
    same_route = df[df["Route"] == row["Route"]]["risk"].mean()
    same_svc = df[df["carrier"] == row["carrier"]]["risk"].mean()
    p, q, w = st.columns(3)
    p.metric("This shipment", f"{row['risk']:.0%}")
    q.metric("Same route average", f"{same_route:.0%}", f"{row['risk'] - same_route:+.0%} vs this", delta_color="off")
    w.metric("Same service average", f"{same_svc:.0%}", f"{row['risk'] - same_svc:+.0%} vs this", delta_color="off")
    st.info("Next step: open 'Route alternatives' to see safer service options for this shipment.")

# ---------------- Route alternatives ----------------
elif page == "Route alternatives":
    st.title("What can we do about it?")
    st.write("We compare the available services for a risky shipment and rank them with a simple score.")
    if flagged.empty or df["carrier"].nunique() < 2:
        st.warning("Need at least one high-risk shipment and two service options.")
        st.stop()
    top = flagged.head(300).set_index("shipment_id")
    pick = st.selectbox("Choose a shipment", top.index,
                        format_func=lambda s: f"{s} | {top.loc[s, 'Route']} | {top.loc[s, 'risk']:.0%}")
    row = top.loc[pick]
    st.write(f"**{row['Route']}** · current service **{row['carrier']}** · delay risk **{row['risk']:.0%}**")

    lane = df[df["Route"] == row["Route"]]
    rows = []
    for svc in sorted(df["carrier"].unique()):
        on_lane = lane[lane["carrier"] == svc]["risk"]
        risk = float(row["risk"]) if svc == row["carrier"] else float(on_lane.mean() if len(on_lane) >= 5 else df[df["carrier"] == svc]["risk"].mean())
        rows.append({"Service": svc, "Delay risk": risk, "Relative cost*": COST.get(svc, 1.0),
                     "Past reliability": 1 - float(df[df["carrier"] == svc]["risk"].mean())})
    base = pd.DataFrame(rows)
    span = base["Relative cost*"].max() - base["Relative cost*"].min()
    cn = (base["Relative cost*"] - base["Relative cost*"].min()) / (span or 1)


    def rank(goal):
        w1, w2, w3 = PRESETS[goal]
        d = base.copy()
        d["Score"] = (w1 * d["Delay risk"] + w2 * cn + w3 * (1 - d["Past reliability"])).round(3)
        d = d.sort_values("Score").reset_index(drop=True)
        d.insert(0, "Rank", d.index + 1)
        return d


    goal = st.radio("What matters most?", list(PRESETS), horizontal=True, index=1)
    d = rank(goal)
    best = d.iloc[0]
    why = []
    if best["Delay risk"] <= d["Delay risk"].min() + 1e-9: why.append("the lowest delay risk")
    if best["Relative cost*"] <= d["Relative cost*"].min(): why.append("the lowest cost")
    if best["Past reliability"] >= d["Past reliability"].max() - 1e-9: why.append("the best past reliability")
    if best["Service"] == row["carrier"]:
        st.warning(f"For '{goal}', the current service ({best['Service']}) is already the best choice.")
    else:
        st.success(f"For '{goal}', switch to **{best['Service']}**: delay risk {row['risk']:.0%} → {best['Delay risk']:.0%}."
                   + (f" It wins because it has {' and '.join(why)}." if why else ""))
    st.dataframe(d, column_config={"Delay risk": st.column_config.ProgressColumn("Delay risk", min_value=0, max_value=1, format="percent"),
                                   "Past reliability": st.column_config.NumberColumn(format="%.2f")},
                 hide_index=True, width="stretch")

    st.markdown("#### Best choice for each goal")
    st.dataframe(pd.DataFrame([{"Goal": g, "Best service": rank(g).iloc[0]["Service"]} for g in PRESETS]),
                 hide_index=True, width="stretch")
    st.caption("Score = (w1 × delay risk) + (w2 × cost) + (w3 × unreliability); lower is better. Delay risk for other services = "
               "their average past risk on this route. *Relative cost is an assumed level, not from the data. "
               "This is risk-aware route ranking, not route optimization.")

# ---------------- How it works ----------------
else:
    st.title("How this project works")
    st.markdown("""
| Step | What it does | Status |
|---|---|---|
| 1. Data cleaning | Cleans DataCo orders and removes columns that would leak the answer | Built |
| 2. Risk model | Logistic Regression baseline vs XGBoost, time-based split | Built |
| 3. Explanation | Shows the top reasons behind each risk score | Built |
| 4. Route ranking | Ranks service options by risk, cost and reliability | Built (cost assumed) |
| 5. Real-world events | Strikes, storms and congestion from GDELT, linked to ports | Next |
""")
    st.markdown("#### Why booking-time prediction?")
    st.write("A ship cannot be rerouted like a car. The real choices (service, port, date) are made at booking, "
             "so the system scores risk then, and can re-score as new events arrive.")
    st.markdown("#### Limitations (honest)")
    st.markdown("""
- DataCo is general shipment data, used here as a proxy for sea freight. Ports are an approximate mapping.
- Risk depends strongly on the shipping service, a pattern in this dataset and not a real-world rule.
- Cost is an assumed level, since the data has no real shipping cost.
- Event data (strikes, storms) is not connected yet.
""")
