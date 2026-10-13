"""
NAVI: Explainable Supply Chain Risk Intelligence & Decision Support System
Scope: OCEAN shipments only. Road and rail are planned, not implemented.

Run:  streamlit run navi_app.py        (needs scored_shipments.csv in the same folder)

Required columns : shipment_id, origin_port, dest_port, carrier (= shipping service), risk (0-1)
Optional columns : top_factors   ("factor A + factor B + factor C")
                   scheduled_days
                   shap_json     JSON list [{"feature": "...", "shap": 0.41, "value": "..."}]
                                 -> enables the increase/decrease chart (log-odds scale)
                   cost column   one of: est_cost, cost, freight_cost, shipping_cost
                   outcome column (1 = late) one of: late, is_late, delayed, actual_late, late_delivery_risk
                                 -> enables a genuine reliability figure
Nothing is invented: optional figures appear only when the column exists.
"""
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PRODUCT = "NAVI"
SUBTITLE = "Explainable Supply Chain Risk Intelligence & Decision Support System"
DESCRIPTION = ("Predict shipment delay risk at booking time, explain the main risk factors in plain English, "
               "and help managers compare available service options.")
SCOPE = "Current scope: ocean shipments. Road and rail are planned, not implemented."

st.set_page_config(page_title="NAVI | Supply Chain Risk Intelligence", page_icon="🚢", layout="wide")

REQUIRED = ["shipment_id", "origin_port", "dest_port", "carrier", "risk"]
COST_COLS = ["est_cost", "cost", "freight_cost", "shipping_cost"]
OUTCOME_COLS = ["late", "is_late", "delayed", "actual_late", "late_delivery_risk"]
MED_GAP = 0.20      # Medium band = from (threshold - 20 points) up to the threshold (display rule)
MIN_LANE = 5        # minimum lane records for a lane-specific service average

P_OVER, P_DET, P_RANK = "Overview", "Shipment details", "Service ranking"
P_EVT, P_RR, P_HOW = "Real-world events (coming soon)", "Road & rail (planned)", "How it works"
PAGES = [P_OVER, P_DET, P_RANK, P_EVT, P_RR, P_HOW]

# (regex on feature name, readable label, plain-English meaning, suggested check)
FACTORS = [
    ("scheduled|transit", "Scheduled transit time",
     "The promised transit time matters to the model for this shipment. Tight schedules leave little room for delay.",
     "Add a schedule buffer or book an earlier sailing."),
    ("lane|route", "Route history",
     "The model links this route with higher delay risk, a pattern learned from earlier shipments on similar routes.",
     "Review alternative ports and confirm the port slot early."),
    ("mode|service|class|carrier", "Service level",
     "The model links this service level with higher delay risk, a pattern learned from past orders.",
     "Compare other service levels on the Service ranking page."),
    ("event|strike|storm|congestion|weather", "Disruption signal",
     "A disruption-related signal matters to the model for this shipment.",
     "Check port advisories and consider an alternate port."),
    ("region|country|market", "Destination region",
     "The model links this destination region with higher delay risk, a pattern learned from past orders.",
     "Warn the receiver and plan extra stock."),
]
LEVEL_CLASS = {"High": "hi", "Medium": "mid", "Low": "lo"}
ICON = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}


# ---------------------------------------------------------------- helpers
def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def esc(x):
    return html.escape(str(x))


def prettify(name):
    name = re.sub(r"^(num|cat|remainder)__", "", str(name))
    return re.sub(r"[_\-]+", " ", name).strip().capitalize() or "Unnamed factor"


def parse_factors(text):
    return [p.strip() for p in str(text).split("+") if p.strip() not in ("", "-", "nan")]


def describe(name, row, df):
    """Readable label, meaning, suggested check and (only if supported by data) the actual value."""
    hit = next((f for f in FACTORS if re.search(f[0], name, re.I)), None)
    label = hit[1] if hit else prettify(name)
    meaning = hit[2] if hit else "The model lists this as one of the strongest factors for this shipment."
    action = hit[3] if hit else "Review this shipment with the carrier."
    value = "Not stored in the data"
    if label == "Scheduled transit time" and "scheduled_days" in df.columns and pd.notna(row.get("scheduled_days")):
        med = df["scheduled_days"].median()
        value = f"{row['scheduled_days']:g} days (median across analysed shipments: {med:g} days)"
    elif label == "Route history":
        value = str(row["Route"])
    elif label == "Service level":
        value = str(row["carrier"])
    return {"feature": name, "label": label, "meaning": meaning, "action": action, "value": value}


def parse_shap(row):
    raw = row.get("shap_json") if hasattr(row, "get") else None
    if raw is None or pd.isna(raw) or not str(raw).strip():
        return None
    try:
        items = json.loads(raw)
        out = [(str(i["feature"]), float(i["shap"])) for i in items]
    except Exception:
        return None
    return sorted(out, key=lambda t: abs(t[1]), reverse=True) or None


def recommend(level, desc):
    first = desc[0]["action"] if desc else "Review this shipment with the carrier before it sails."
    if level == "High":
        return "warning", f"Act before the ship sails. First check: {first} Then compare service options for this shipment."
    if level == "Medium":
        return "info", f"Monitor. Confirm schedule buffer and port slot with the carrier. First check: {first}"
    return "success", "No action needed beyond normal monitoring."


def get_row(df, sid):
    m = df[df["shipment_id"] == str(sid)]
    return m.iloc[0] if len(m) else None


def find_csv():
    """Look next to this script first, then in the folder Streamlit was launched from."""
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        if (base / "scored_shipments.csv").exists():
            return base / "scored_shipments.csv"
    return None


@st.cache_data
def prepare():
    """Load and validate scored_shipments.csv. Returns (df, notes, missing_columns, found_columns)."""
    f = find_csv()
    if f is None:
        return None, [], None, []
    raw = pd.read_csv(f)
    miss = [c for c in REQUIRED if c not in raw.columns]
    if miss:
        return None, [], miss, list(raw.columns)
    d, notes = raw.copy(), []
    d["risk"] = pd.to_numeric(d["risk"], errors="coerce")
    n_bad = int(d["risk"].isna().sum())
    if n_bad:
        d = d[d["risk"].notna()]
        notes.append(f"{n_bad:,} rows with a missing or non-numeric risk score were excluded.")
    if len(d) and d["risk"].max() > 1:
        d["risk"] = d["risk"] / 100
        notes.append("Risk values above 1 were read as percentages and divided by 100.")
    out_of_range = int(((d["risk"] < 0) | (d["risk"] > 1)).sum())
    if out_of_range:
        d = d[(d["risk"] >= 0) & (d["risk"] <= 1)]
        notes.append(f"{out_of_range:,} rows with risk outside 0-1 were excluded.")
    d["shipment_id"] = d["shipment_id"].astype(str).str.strip()
    dups = int(d["shipment_id"].duplicated().sum())
    if dups:
        d = d.drop_duplicates("shipment_id", keep="first")
        notes.append(f"{dups:,} duplicate shipment IDs were removed (first record kept).")
    for c in ("origin_port", "dest_port", "carrier"):
        d[c] = d[c].fillna("Unknown").astype(str).str.strip()
    if "top_factors" not in d.columns:
        d["top_factors"] = "-"
    d["top_factors"] = d["top_factors"].fillna("-").astype(str)
    d["Route"] = d["origin_port"] + " → " + d["dest_port"]
    d = d.reset_index(drop=True)
    return (d if len(d) else None), notes, [] if len(d) else ["(no valid rows)"], list(raw.columns)


def add_levels(d, thr):
    d["Level"] = np.select([d["risk"] >= thr, d["risk"] >= thr - MED_GAP], ["High", "Medium"], "Low")
    return d


def band_counts(d, thr):
    lv = np.select([d["risk"] >= thr, d["risk"] >= thr - MED_GAP], ["High", "Medium"], "Low")
    return [int((lv == k).sum()) for k in ("High", "Medium", "Low")]


def first_col(d, names):
    return next((c for c in names if c in d.columns), None)


def service_table(d, row, basis):
    """Typical predicted risk per service. Returns (table, services_without_lane_data)."""
    cost_col, out_col = first_col(d, COST_COLS), first_col(d, OUTCOME_COLS)
    lane = d[(d["Route"] == row["Route"]) & (d["shipment_id"] != row["shipment_id"])]
    rows, skipped = [], []
    for svc in sorted(d["carrier"].unique()):
        on_lane = lane[lane["carrier"] == svc]
        use_lane = basis == "lane" and len(on_lane) >= MIN_LANE
        if basis == "lane" and not use_lane and svc != row["carrier"]:
            skipped.append(svc)
            continue
        pool = on_lane if use_lane else d[d["carrier"] == svc]
        rel = np.nan
        if out_col:
            m = pd.to_numeric(pool[out_col], errors="coerce").mean()
            rel = 1 - m if 0 <= m <= 1 else np.nan
        rows.append({
            "Service": svc, "is_current": svc == row["carrier"], "Risk": float(pool["risk"].mean()), "n": len(pool),
            "Basis": "this lane" if use_lane else "all lanes", "Cost": pool[cost_col].mean() if cost_col else np.nan,
            "Reliability": rel})
    t = pd.DataFrame(rows).sort_values(["Risk", "Service"]).reset_index(drop=True)
    t.insert(0, "Rank", t.index + 1)
    if len(t):
        best = t.iloc[0]

        def why(r):
            tail = " (current service)" if r["is_current"] else ""
            if r["Rank"] == 1:
                return f"Lowest typical predicted risk ({pct(r['Risk'])}){tail}"
            gap = (r["Risk"] - best["Risk"]) * 100
            if gap < 0.05:
                return f"Practically tied with #1 {best['Service']}{tail}"
            return f"Typical risk {gap:.1f} points higher than #1 {best['Service']}{tail}"
        t["Reason"] = t.apply(why, axis=1)
    return t, skipped


def alt_routes(d, row):
    """Other origin ports shipping to the same destination, same service, ranked by typical predicted risk."""
    pool = d[(d["dest_port"] == row["dest_port"]) & (d["carrier"] == row["carrier"]) & (d["shipment_id"] != row["shipment_id"])]
    g = pool.groupby("origin_port")["risk"].agg(Risk="mean", n="count").reset_index()
    g = g[(g["n"] >= MIN_LANE) | (g["origin_port"] == row["origin_port"])].copy()
    g["is_current"] = g["origin_port"] == row["origin_port"]
    g = g.sort_values(["Risk", "origin_port"]).reset_index(drop=True)
    g.insert(0, "Rank", g.index + 1)
    if len(g):
        best = g.iloc[0]

        def why(r):
            tail = " (current origin)" if r["is_current"] else ""
            if r["Rank"] == 1:
                return f"Lowest typical predicted risk ({pct(r['Risk'])}){tail}"
            gap = (r["Risk"] - best["Risk"]) * 100
            return (f"Practically tied with #1{tail}" if gap < 0.05 else
                    f"Typical risk {gap:.1f} points higher than #1 {best['origin_port']}{tail}")
        g["Reason"] = g.apply(why, axis=1)
    return g


def next_step_text(d, flagged, thr):
    n, nh = len(d), len(flagged)
    if nh == 0:
        return (f"No shipment reaches the High threshold of {pct(thr, 0)}. No escalation needed at this setting; "
                "lower the threshold in the sidebar to review borderline shipments.")
    top = flagged.iloc[0]
    svc = flagged["carrier"].value_counts()
    lane = flagged["Route"].value_counts()
    return (f"{nh:,} of {n:,} shipments ({pct(nh / n)}) are High risk at the {pct(thr, 0)} threshold. "
            f"Start with {top['shipment_id']} ({pct(top['risk'])}, {top['Route']}). "
            f"{svc.index[0]} accounts for {pct(svc.iloc[0] / nh, 0)} of High-risk shipments, so review that service first. "
            f"The lane with most High-risk shipments is {lane.index[0]} ({int(lane.iloc[0])}).")


# ---------------------------------------------------------------- styles
st.markdown("""<style>
.hero{padding:28px 34px;border-radius:18px;background:linear-gradient(120deg,#0b1f3a 0%,#0e4d75 60%,#12829b 100%);color:#fff;margin-bottom:18px}
.hero h1{margin:0;font-size:3rem;letter-spacing:.14em;color:#fff}
.hero .sub{font-size:1.2rem;font-weight:600;opacity:.95;margin:.1rem 0 .6rem}
.hero p{margin:.2rem 0 0;opacity:.95;max-width:920px}
.hero .scope{font-size:.85rem;opacity:.8;border-top:1px solid rgba(255,255,255,.25);padding-top:.5rem;margin-top:.7rem}
.card{border:1px solid rgba(127,127,127,.28);background:rgba(127,127,127,.09);border-radius:14px;padding:14px 16px;min-height:150px}
.card h4{margin:.1rem 0 .35rem;font-size:1rem}.card p{margin:0;font-size:.86rem;opacity:.85}
.pill{display:inline-block;font-size:.7rem;padding:2px 9px;border-radius:99px;margin-bottom:6px;font-weight:600}
.built{background:#14532d;color:#bbf7d0}.plan{background:#78350f;color:#fde68a}
.bar{display:flex;height:30px;border-radius:9px;overflow:hidden;margin:.4rem 0 .3rem}
.bar div{display:flex;align-items:center;justify-content:center;font-size:.8rem;color:#fff;font-weight:600}
.note{border-left:4px solid #12829b;padding:.5rem .9rem;background:rgba(18,130,155,.12);border-radius:6px;font-size:.9rem}
.soon{border-left:4px solid #d97706;padding:.6rem .9rem;background:rgba(217,119,6,.13);border-radius:6px}
.topcard{border:1px solid rgba(127,127,127,.3);border-radius:14px;padding:16px 20px;background:rgba(127,127,127,.07)}
.topcard .id{font-size:.85rem;opacity:.7}.topcard .route{font-size:1.35rem;font-weight:700;line-height:1.35;word-break:break-word}
.topcard .meta{font-size:.95rem;opacity:.85;margin:.2rem 0 .6rem}.topcard .big{font-size:2.2rem;font-weight:800}
.chip{display:inline-block;padding:2px 12px;border-radius:99px;font-weight:700;font-size:.85rem;margin-left:8px;color:#fff}
.chip.hi{background:#b91c1c}.chip.mid{background:#b45309}.chip.lo{background:#15803d}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- data + navigation state
df, notes, missing, found = prepare()
if df is None and missing is None:
    st.error("scored_shipments.csv not found.")
    st.write(f"NAVI looked in: `{Path(__file__).resolve().parent}` and `{Path.cwd()}`.")
    st.write("This app does not read the raw DataCo file. It reads `scored_shipments.csv`, which is created by "
             "`python src/clean.py` then `python src/train.py` (needs DataCoSupplyChainDataset.csv), or can be "
             "downloaded from the GitHub repo. Put it next to `app_v4.py`.")
    st.stop()
if df is None:
    st.error(f"Cannot use scored_shipments.csv. Problem: {missing}")
    st.write("Columns found:", found)
    st.stop()

st.session_state.setdefault("page", P_OVER)
st.session_state.setdefault("panel", None)
st.session_state.setdefault("sel", df.sort_values("risk", ascending=False).iloc[0]["shipment_id"])


def go(page, ship=None):
    st.session_state.page = page
    st.session_state.panel = None
    if ship is not None:
        st.session_state.sel = str(ship)


def open_panel(name):
    st.session_state.panel = name


def open_lookup(key):
    q = str(st.session_state.get("lookup_id", "")).strip()
    data, _, _, _ = prepare()
    if data is not None and q in set(data["shipment_id"]):
        st.session_state.sel = q
        st.session_state[key] = q
        st.session_state.lookup_msg = ""
    else:
        st.session_state.lookup_msg = f"Shipment ID '{q}' was not found."


st.sidebar.markdown(f"## 🚢 {PRODUCT}")
st.sidebar.caption(SUBTITLE)
st.sidebar.success("Build: navi_app.py (v4) · 6 separate pages")
page = st.sidebar.radio("Go to", PAGES, key="page")
thr = st.sidebar.slider("High-risk threshold", 30, 95, 75, 5, format="%d%%",
                        help="A shipment at or above this predicted delay risk is called High.") / 100
st.sidebar.caption("Predicted delay risk = the model's estimated chance a shipment is late, "
                   "using booking-time information. Scope: ocean shipments.")

df = add_levels(df, thr)
flagged = df[df["Level"] == "High"].sort_values("risk", ascending=False)
N = len(df)


def hero():
    st.markdown(f'<div class="hero"><h1>{PRODUCT}</h1><div class="sub">{SUBTITLE}</div><p>{DESCRIPTION}</p>'
                f'<div class="scope">{SCOPE}</div></div>', unsafe_allow_html=True)
    for n in notes:
        st.warning("Data note: " + n)


def ship_selector(key):
    top = df.sort_values("risk", ascending=False).head(300)
    ids = list(top["shipment_id"])
    cur = st.session_state["sel"]
    if cur not in ids:
        ids = [cur] + ids
    labels = {}
    for sid in ids:
        r = get_row(df, sid)
        labels[sid] = f"{sid} | {r['origin_port']} → {r['dest_port']} | {pct(r['risk'])} | {r['carrier']}"

    def on_pick():
        st.session_state.sel = st.session_state[key]
    st.selectbox("Choose a shipment (the 300 highest-risk shipments are listed)", ids, index=ids.index(cur),
                 key=key, format_func=lambda s: labels[s], on_change=on_pick)
    with st.expander("Look up any shipment by ID"):
        st.text_input("Shipment ID", key="lookup_id")
        st.button("Open shipment", key=f"lk_{key}", on_click=open_lookup, args=(key,))
        if st.session_state.get("lookup_msg"):
            st.warning(st.session_state.lookup_msg)
    return get_row(df, st.session_state["sel"])


# ================================================================ OVERVIEW
if page == P_OVER:
    hero()
    st.markdown("### What NAVI does")
    top_id = df.sort_values("risk", ascending=False).iloc[0]["shipment_id"]
    cards = [
        ("built", "Working", "1. Booking-time prediction",
         "Browse the stored booking-time delay prediction for any shipment.", "Open shipment predictions",
         "c1", (go, (P_DET,))),
        ("built", "Working", "2. Plain-English reasons",
         "See why a real shipment is flagged, with a suggested check.", "Explain the riskiest shipment",
         "c2", (go, (P_DET, top_id))),
        ("built", "Working", "3. Adjustable risk bands",
         "See how the threshold changes High, Medium and Low counts.", "Show risk-band impact",
         "c3", (open_panel, ("bands",))),
        ("built", "Working", "4. Risk-aware service & route ranking",
         "Compare service options and alternative origin ports by predicted risk.", "Open service ranking",
         "c4", (go, (P_RANK,))),
        ("plan", "Planned", "5. Real-world events",
         "News-based disruption context. Not connected yet.", "See planned feature",
         "c5", (go, (P_EVT,))),
    ]
    for col, (cls, tag, head, body, btn, key, (fn, args)) in zip(st.columns(5), cards):
        with col:
            st.markdown(f'<div class="card"><span class="pill {cls}">{tag}</span><h4>{head}</h4><p>{body}</p></div>',
                        unsafe_allow_html=True)
            st.button(btn, key=key, on_click=fn, args=args)

    if st.session_state.panel == "bands":
        st.markdown("#### How the risk threshold changes the categories")
        st.write(f"Current setting (sidebar slider): **High ≥ {pct(thr, 0)}**, Medium {pct(thr - MED_GAP, 0)} to under "
                 f"{pct(thr, 0)}, Low under {pct(thr - MED_GAP, 0)}. The Medium band is always {int(MED_GAP * 100)} points "
                 "wide (a display rule, not statistically derived). Counts below are computed from your data.")
        rows = []
        for t in sorted(set([0.5, 0.6, 0.7, 0.75, 0.8, 0.9, thr])):
            h, m, l = band_counts(df, t)
            rows.append({"High threshold": pct(t, 0) + (" (current)" if abs(t - thr) < 1e-9 else ""),
                         "High": h, "High %": pct(h / N), "Medium": m, "Low": l})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Risk values are the model's predicted probabilities. Their calibration against real late rates "
                   "has not been checked in this app.")
        st.button("Close", key="close_panel", on_click=open_panel, args=(None,))

    st.markdown("### Risk snapshot")
    nh, nm, nl = [int((df["Level"] == k).sum()) for k in ("High", "Medium", "Low")]
    a, b, c, d = st.columns(4)
    a.metric("Total shipments analysed", f"{N:,}", help="Number of valid rows in scored_shipments.csv.")
    b.metric("High-risk shipments", f"{nh:,}", f"{pct(nh / N)} of all", delta_color="off",
             help=f"Shipments with predicted delay risk ≥ {pct(thr, 0)} (sidebar threshold).")
    c.metric("Medium-risk shipments", f"{nm:,}", f"{pct(nm / N)} of all", delta_color="off")
    d.metric("Average predicted delay risk", pct(df["risk"].mean()), "not affected by threshold", delta_color="off",
             help="Mean of the model's predicted delay probability over all analysed shipments. "
                  "Not model accuracy and not the share of shipments that were actually late.")
    st.markdown(f'<div class="bar"><div style="width:{max(nl / N * 100, 0.1)}%;background:#15803d">Low {pct(nl / N, 0)}</div>'
                f'<div style="width:{max(nm / N * 100, 0.1)}%;background:#b45309">Medium {pct(nm / N, 0)}</div>'
                f'<div style="width:{max(nh / N * 100, 0.1)}%;background:#b91c1c">High {pct(nh / N, 0)}</div></div>',
                unsafe_allow_html=True)
    st.caption("Average predicted delay risk is the model's estimated chance of delay, averaged over all shipments.")

    st.markdown("### Recommended next step")
    st.info(next_step_text(df, flagged, thr))

    st.markdown("### Shipments requiring attention")
    scope = st.radio("Display scope", ["High-risk only", "All shipments"], horizontal=True, key="ov_scope")
    f1, f2 = st.columns(2)
    mode = f1.selectbox("Service", ["All"] + sorted(df["carrier"].unique()), key="ov_service")
    nshow = f2.selectbox("Rows to show", [10, 25, 50], key="ov_rows")
    base = flagged if scope == "High-risk only" else df.sort_values("risk", ascending=False)
    v = base if mode == "All" else base[base["carrier"] == mode]
    if v.empty:
        st.success("No shipments for this selection at the current threshold.")
    else:
        w = [1.0, 2.2, 2.2, 1.3, 0.9, 1.0, 1.9, 1.1]
        for col, h in zip(st.columns(w), ["Shipment ID", "Origin", "Destination", "Service", "Predicted risk",
                                          "Risk category", "Main risk driver", "Action"]):
            col.markdown(f"**{h}**")
        for r in v.head(nshow).to_dict("records"):
            fs = parse_factors(r["top_factors"])
            driver = describe(fs[0], r, df)["label"] if fs else "Not recorded"
            cs = st.columns(w)
            cs[0].write(r["shipment_id"])
            cs[1].write(r["origin_port"])
            cs[2].write(r["dest_port"])
            cs[3].write(r["carrier"])
            cs[4].write(pct(r["risk"]))
            cs[5].write(ICON[r["Level"]])
            cs[6].write(driver)
            cs[7].button("View details", key=f"vd_{r['shipment_id']}", on_click=go, args=(P_DET, r["shipment_id"]))
        st.caption(f"Showing {min(nshow, len(v))} of {len(v):,} shipments, highest risk first. "
                   "Main risk driver = first factor listed for the shipment in the data file.")

    t1, t2 = st.tabs(["Average risk by service", "Average risk by route"])
    with t1:
        s = df.groupby("carrier")["risk"].mean().sort_values(ascending=False)
        st.bar_chart(s)
        st.write(f"Average predicted risk is highest for **{s.index[0]}** ({pct(s.iloc[0])}) and lowest for "
                 f"**{s.index[-1]}** ({pct(s.iloc[-1])}).")
    with t2:
        r = df.groupby("Route")["risk"].agg(avg="mean", n="count")
        r = r[r["n"] >= 10].sort_values("avg", ascending=False).head(10).reset_index()
        if r.empty:
            st.write("No route has at least 10 shipments.")
        else:
            st.dataframe(r.rename(columns={"avg": "Average predicted risk", "n": "Shipments"}),
                         column_config={"Average predicted risk": st.column_config.ProgressColumn(
                             "Average predicted risk", min_value=0, max_value=1, format="percent")},
                         hide_index=True, width="stretch")
        st.caption("Top 10 routes with at least 10 shipments.")

# ================================================================ SHIPMENT DETAILS
elif page == P_DET:
    hero()
    st.subheader("Shipment details")
    row = ship_selector("pick_det")
    shap_items = parse_shap(row)
    names = [n for n, _ in shap_items[:3]] if shap_items else parse_factors(row["top_factors"])[:3]
    desc = [describe(n, row, df) for n in names]
    level = row["Level"]

    st.markdown(
        f'<div class="topcard"><div class="id">Shipment {esc(row["shipment_id"])}</div>'
        f'<div class="route">{esc(row["origin_port"])} → {esc(row["dest_port"])}</div>'
        f'<div class="meta">Service: {esc(row["carrier"])}</div>'
        f'<div><span class="big">{pct(row["risk"])}</span> predicted delay risk'
        f'<span class="chip {LEVEL_CLASS[level]}">{level}</span></div></div>', unsafe_allow_html=True)
    st.progress(float(row["risk"]))

    st.markdown("#### Recommended action")
    kind, text = recommend(level, desc)
    getattr(st, kind)(text)
    st.caption("Rule-based suggestion from the risk category and the first factor. It is not a model output.")

    st.markdown("#### Most important risk factors")
    if desc:
        for i, dsc in enumerate(desc, 1):
            st.markdown(f"**{i}. {dsc['label']}**: {dsc['meaning']}")
    else:
        st.write("No factor information is available for this shipment in the data file.")
    st.button("Compare service options for this shipment", key="to_rank", on_click=go, args=(P_RANK,))

    with st.expander("Supporting evidence and technical explanation", expanded=True):
        if shap_items:
            st.markdown("**Model explanation: what raises or lowers this shipment's predicted risk**")
            lab = [describe(n, row, df)["label"] for n, _ in shap_items[:8]]
            lab = [l if lab.count(l) == 1 else f"{l} [{n}]" for l, (n, _) in zip(lab, shap_items[:8])]
            ch = pd.DataFrame({"Raises risk": [max(v, 0) for _, v in shap_items[:8]],
                               "Lowers risk": [min(v, 0) for _, v in shap_items[:8]]}, index=lab)
            st.bar_chart(ch, color=["#b91c1c", "#15803d"])
            st.caption("Bar length = SHAP value on the model's log-odds scale (not percentage points). "
                       "Red pushes the prediction toward higher risk, green toward lower risk.")
        else:
            st.info("This data file stores factor names only. Direction and size of each factor's push are not "
                    "available, so no increase/decrease chart is shown. Export per-shipment SHAP values as "
                    "'shap_json' to enable it (see How it works).")
        if names:
            st.markdown("**Factor details**")
            all_desc = [describe(n, row, df) for n in (n for n, _ in shap_items[:8])] if shap_items else desc
            st.dataframe(pd.DataFrame([{"Model feature": x["feature"], "Readable label": x["label"],
                                        "Value for this shipment": x["value"], "What it may mean": x["meaning"],
                                        "Suggested check": x["action"]} for x in all_desc]),
                         hide_index=True, width="stretch")
        route_df = df[(df["Route"] == row["Route"]) & (df["shipment_id"] != row["shipment_id"])]
        svc_df = df[(df["carrier"] == row["carrier"]) & (df["shipment_id"] != row["shipment_id"])]
        p, q, w = st.columns(3)
        p.metric("This shipment", pct(row["risk"]))
        if len(route_df):
            q.metric(f"Same-route average ({len(route_df):,} others)", pct(route_df["risk"].mean()),
                     f"this shipment is {(row['risk'] - route_df['risk'].mean()) * 100:+.1f} points vs average",
                     delta_color="off")
        else:
            q.metric("Same-route average", "n/a", "no other shipments on this route", delta_color="off")
        if len(svc_df):
            w.metric(f"Same-service average ({len(svc_df):,} others)", pct(svc_df["risk"].mean()),
                     f"this shipment is {(row['risk'] - svc_df['risk'].mean()) * 100:+.1f} points vs average",
                     delta_color="off")
        st.markdown("**Real-world events:** not connected yet (planned GDELT integration). "
                    "No event information is used in this prediction.")
    st.caption("Limitation: these are model-based explanations of a statistical prediction. They show what the model "
               "associates with delay, not proof that a factor caused a delay.")

# ================================================================ SERVICE RANKING
elif page == P_RANK:
    hero()
    st.subheader("Service ranking")
    row = ship_selector("pick_rank")
    st.write(f"**{row['origin_port']} → {row['dest_port']}** · current service **{row['carrier']}** · "
             f"this shipment's predicted delay risk **{pct(row['risk'])}**")
    basis_lbl = st.radio("Compare services using", ["This lane only (more specific)", "All lanes (broader)"],
                         horizontal=True, key="rank_basis")
    basis = "lane" if basis_lbl.startswith("This") else "all"
    t, skipped = service_table(df, row, basis)

    if len(t) < 2:
        st.warning(f"The data cannot support a valid comparison on this lane: fewer than two services have at least "
                   f"{MIN_LANE} other shipments on it. Choose 'All lanes' for a broader (less specific) comparison.")
    else:
        best = t.iloc[0]
        if best["is_current"]:
            st.success(f"The current service ({best['Service']}) already has the lowest typical predicted risk "
                       f"({pct(best['Risk'])}) in this comparison.")
        else:
            cur = t[t["is_current"]].iloc[0]
            st.success(f"{best['Service']} has the lowest typical predicted risk ({pct(best['Risk'])}), "
                       f"{(cur['Risk'] - best['Risk']) * 100:.1f} points below the current service's typical risk "
                       f"({pct(cur['Risk'])}). Check availability and cost before switching.")
    if len(t):
        has_cost, has_rel = t["Cost"].notna().any(), t["Reliability"].notna().any()
        show = pd.DataFrame({
            "Rank": t["Rank"],
            "Service": [s + (" (current)" if c else "") for s, c in zip(t["Service"], t["is_current"])],
            "Typical predicted risk": t["Risk"],
            "Based on": [f"{b}, {n:,} shipments" for b, n in zip(t["Basis"], t["n"])],
            "Avg. cost (from data)": t["Cost"] if has_cost else "Not available in data",
            "Reliability (from data)": t["Reliability"] if has_rel else "Not available in data",
            "Ranking score (lower is better)": t["Risk"],
            "Why": t["Reason"]})
        cfg = {"Typical predicted risk": st.column_config.ProgressColumn("Typical predicted risk", min_value=0,
                                                                         max_value=1, format="percent"),
               "Ranking score (lower is better)": st.column_config.NumberColumn(format="%.3f"),
               "Why": st.column_config.TextColumn("Why", width="large")}
        if has_rel:
            cfg["Reliability (from data)"] = st.column_config.NumberColumn(format="percent")
        st.dataframe(show, column_config=cfg, hide_index=True, width="stretch")
    if skipped:
        st.caption("Not compared (fewer than "
                   f"{MIN_LANE} other shipments on this lane): {', '.join(skipped)}.")
    st.markdown("#### Alternative routes: other origin ports to the same destination")
    ar = alt_routes(df, row)
    if len(ar) < 2:
        st.info(f"No valid route comparison: fewer than two origin ports have at least {MIN_LANE} other "
                f"{row['carrier']} shipments to {row['dest_port']}.")
    else:
        st.dataframe(pd.DataFrame({
            "Rank": ar["Rank"],
            "Origin port": [o + (" (current)" if c else "") for o, c in zip(ar["origin_port"], ar["is_current"])],
            "Destination": row["dest_port"], "Service": row["carrier"],
            "Typical predicted risk": ar["Risk"], "Based on": [f"{n:,} shipments" for n in ar["n"]],
            "Why": ar["Reason"]}),
            column_config={"Typical predicted risk": st.column_config.ProgressColumn(
                "Typical predicted risk", min_value=0, max_value=1, format="percent"),
                "Why": st.column_config.TextColumn("Why", width="large")},
            hide_index=True, width="stretch")
        st.caption("Historical average predicted risk for the same service and destination, by origin port. "
                   "Port feasibility, cost and transit time are not in the data and are not checked; the port "
                   "mapping in DataCo is approximate. Decision support only.")
    st.markdown("#### How this ranking is calculated")
    st.markdown(
        "- **Ranking score = typical predicted delay risk** of the service, from the model's scores for other "
        "shipments (the selected shipment is excluded). Lower is better.\n"
        "- **Lane view** averages shipments on the same origin → destination (at least "
        f"{MIN_LANE} per service). **All-lanes view** averages every shipment on that service.\n"
        "- **Cost and reliability are shown only if the data file contains genuine cost or late-outcome columns.** "
        "This one does not, so no cost or reliability figures are shown and none are invented.\n"
        "- These are averages of other shipments, **not a re-prediction of this shipment under a different service**, "
        "and the data does not confirm that each service is offered on this lane. Treat this as decision support.")

# ================================================================ REAL-WORLD EVENTS (planned)
elif page == P_EVT:
    hero()
    st.subheader("Real-world events")
    st.markdown('<div class="soon"><b>Coming Soon — GDELT Integration.</b> NAVI will soon integrate GDELT news and event '
                'data to identify potential real-world disruptions affecting ocean shipments. Not implemented yet: '
                'no news, event counts, impact scores or alerts are shown or used anywhere in NAVI today.</div>',
                unsafe_allow_html=True)
    st.markdown("#### Planned event panel (intentionally empty)")
    st.dataframe(pd.DataFrame(columns=["Headline", "Source", "Published (UTC)", "Location", "Event category",
                                       "Affected port / trade lane", "Relevance to selected shipment",
                                       "Evidence status"]), hide_index=True, width="stretch")
    st.caption("Each future row will link to the original news source with its publication time so a manager can verify it.")
    a, b = st.columns(2)
    a.markdown("#### Planned workflow")
    a.markdown("1. Retrieve recent events from GDELT.\n2. Filter by location and event type (strikes, storms, "
               "congestion, closures).\n3. Match events to ports and trade lanes.\n4. Link an event to a shipment only "
               "where the match is defensible (right place, right time window).\n5. Show the source and timestamp next "
               "to every event.")
    b.markdown("#### Rules the integration will follow")
    b.markdown("- A **reported event** is not a **confirmed port disruption**.\n- **Potential impact** is not "
               "**measured impact**.\n- No claim that an event caused a delay without supporting evidence.\n"
               "- The ML prediction will **not** use news data until the integration and model are properly tested.\n"
               "- First release covers **ocean shipments only**; road and rail stay future work.")
    st.button("Back to Overview", key="evt_back", on_click=go, args=(P_OVER,))

# ================================================================ ROAD & RAIL (planned)
elif page == P_RR:
    hero()
    st.subheader("Road & Rail")
    st.markdown('<div class="soon"><b>Planned — not yet implemented.</b> No road or rail data, predictions or rankings '
                'exist in NAVI. Ocean routes are never compared with road, rail or air.</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    a.markdown('<div class="card"><span class="pill plan">Planned</span><h4>🚚 Road / truck</h4>'
               '<p>Would need road shipment data with corridors, cities and border checkpoints, scored and ranked '
               'separately from ocean.</p></div>', unsafe_allow_html=True)
    b.markdown('<div class="card"><span class="pill plan">Planned</span><h4>🚆 Rail / train</h4>'
               '<p>Would need rail shipment data with terminals and freight corridors, scored and ranked separately '
               'from ocean.</p></div>', unsafe_allow_html=True)
    st.caption("Each mode needs its own mode-specific shipment data before it can be built. None is included in this version.")

# ================================================================ HOW IT WORKS
else:
    hero()
    st.subheader("How NAVI works")
    st.markdown("""
| Step | What it does | Status |
|---|---|---|
| 1. Data preparation | Cleans the DataCo orders; removes columns only known after delivery and personal columns | Completed |
| 2. Model training | Logistic Regression baseline vs XGBoost, evaluated on a time-based split (metrics are in the training output) | Completed |
| 3. Prediction | Each shipment gets a delay probability from booking-time information | Completed |
| 4. Explanation | Top factors per shipment in plain English. Increase/decrease chart appears when SHAP values are exported | Completed (chart needs `shap_json`) |
| 5. Service ranking | Ranks service options by typical predicted risk, with cost and reliability only if the data has them | Completed (risk only) |
| 6. Real-world events (GDELT) | News-based disruption context linked to ports and lanes | **Planned**, not implemented |
| 7. Road and rail | Separate analysis for each mode | **Planned**, not implemented |
""")
    st.markdown("#### In simple language")
    st.markdown("1. **Prepare data**: keep only what is known when the booking is made.\n"
                "2. **Train the model**: learn from past shipments which ones ran late, testing on later dates than it learned from.\n"
                "3. **Predict**: give each shipment a delay probability.\n"
                "4. **Explain**: show the main factors behind each prediction.\n"
                "5. **Rank services**: compare available service options by typical predicted risk.")
    st.markdown("#### How the dashboard numbers are calculated")
    st.markdown("- **Total shipments analysed**: valid rows in the data file.\n"
                "- **High risk**: predicted delay risk at or above the sidebar threshold. Medium: the 20 points below it. Low: the rest.\n"
                "- **Average predicted delay risk**: mean predicted probability over all shipments. It is not accuracy and not the actual late rate.")
    st.markdown("#### Limitations")
    st.markdown("- DataCo is general shipment data used as a proxy for ocean freight; the port mapping is approximate.\n"
                "- Predicted risk depends strongly on the shipping service, a pattern in this dataset and not a real-world rule.\n"
                "- The data has no genuine shipping cost or late-outcome column, so cost and reliability are not shown.\n"
                "- Explanations show what the model associates with delay, not proven causes.\n"
                "- Probabilities have not been checked for calibration in this app.\n"
                "- GDELT event data is planned and not connected. Road and rail are planned and not built.")
