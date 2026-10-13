"""
src/export_shap.py - add REAL per-shipment SHAP values to scored_shipments.csv
==============================================================================
Run from the project root, AFTER src/clean.py and src/train.py have been run:

    python src/export_shap.py

What it does (no retraining):
 1. Loads the saved XGBoost model + feature list written by train.py
    (data/models/xgb_time_model.joblib, feature_names.joblib).
 2. Rebuilds the test-period feature matrix with the SAME functions train.py used
    (compute_historical_rates + extract_features, same 80/20 time split, same dedupe),
    so the features match what produced the 'risk' column.
 3. SAFETY CHECK 1: recomputed probabilities must match the 'risk' already in
    scored_shipments.csv (it was rounded to 3 decimals). If not, nothing is written.
 4. Computes Tree SHAP on the model's raw (log-odds) output.
 5. SAFETY CHECK 2: base value + sum(SHAP) must equal the model's raw output for every shipment.
 6. Writes two new columns: shap_json (top 6 factors by |SHAP|, with sign) and
    actual_late (historical outcome, used by the dashboard only for service reliability).
 The original CSV is saved once as scored_shipments.backup.csv.
"""
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import compute_historical_rates, extract_features  # noqa: E402
from src.explain import FEATURE_DICTIONARY  # noqa: E402

TOP_K = 6
PROB_TOLERANCE = 0.0015     # 'risk' in the CSV is rounded to 3 decimals
ADDITIVITY_TOLERANCE = 0.01  # log-odds


def readable(feature, value):
    """Manager-friendly label and value text for a model feature."""
    if feature in FEATURE_DICTIONARY:
        meta = FEATURE_DICTIONARY[feature]
        try:
            return meta["label"], meta["format"](value)
        except Exception:
            return meta["label"], f"{value:.3g}"
    if "_" in feature and feature.split("_", 1)[0] in ("Shipping Mode", "Market"):
        group, cat = feature.split("_", 1)
        return f"{group}: {cat}", "Yes" if float(value) >= 0.5 else "No"
    return feature.replace("_", " ").title(), f"{float(value):.3g}"


def main():
    clean_path = PROJECT_ROOT / "data" / "clean.csv"
    model_dir = PROJECT_ROOT / "data" / "models"
    scored_path = PROJECT_ROOT / "scored_shipments.csv"
    for p in (clean_path, model_dir / "xgb_time_model.joblib", model_dir / "feature_names.joblib", scored_path):
        if not p.exists():
            sys.exit(f"Missing {p}. Run src/clean.py and src/train.py first.")

    model = joblib.load(model_dir / "xgb_time_model.joblib")
    feature_names = joblib.load(model_dir / "feature_names.joblib")
    scored = pd.read_csv(scored_path)

    # --- rebuild exactly what train.py did ---------------------------------
    df = pd.read_csv(clean_path, low_memory=False)
    df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"])
    df = df.sort_values("order date (DateOrders)").reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    train_df, test_df = compute_historical_rates(df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy())
    test_eval = test_df.drop_duplicates(subset=["Order Id"]).copy()
    X = extract_features(test_eval).reindex(columns=feature_names, fill_value=0.0)
    ids = [f"SH-{int(o)}" for o in test_eval["Order Id"]]

    # --- check 1: same shipments, same probabilities ------------------------
    if set(ids) != set(scored["shipment_id"]):
        sys.exit("STOP: shipments rebuilt from clean.csv do not match scored_shipments.csv. "
                 "Re-run src/clean.py and src/train.py, then try again. Nothing was changed.")
    probs = pd.Series(model.predict_proba(X)[:, 1], index=ids)
    ref = scored.set_index("shipment_id")["risk"]
    diff = (probs.reindex(ref.index) - ref).abs().max()
    print(f"Check 1 - max |recomputed probability - CSV risk| = {diff:.5f} (limit {PROB_TOLERANCE})")
    if diff > PROB_TOLERANCE:
        sys.exit("STOP: recomputed probabilities do not match the CSV. SHAP would not describe the shown risk. "
                 "Nothing was changed.")

    # --- SHAP on the model's raw output (log-odds) ---------------------------
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    base = float(np.ravel(explainer.expected_value)[-1])
    margin = np.asarray(model.predict(X, output_margin=True))
    gap = np.abs(base + sv.sum(axis=1) - margin).max()
    print(f"Check 2 - max |base + sum(SHAP) - model raw output| = {gap:.5f} (limit {ADDITIVITY_TOLERANCE})")
    if gap > ADDITIVITY_TOLERANCE:
        sys.exit("STOP: SHAP values do not add up to the model output. Nothing was changed.")

    # --- build shap_json ------------------------------------------------------
    records = {}
    for i, sid in enumerate(ids):
        order = np.argsort(-np.abs(sv[i]))[:TOP_K]
        items = []
        for j in order:
            label, val_txt = readable(feature_names[j], X.iloc[i, j])
            items.append({"feature": feature_names[j], "label": label, "value": val_txt, "shap": round(float(sv[i, j]), 4)})
        records[sid] = json.dumps(items)

    out = scored.copy()
    out["shap_json"] = out["shipment_id"].map(records)
    late = pd.Series(test_eval["Late_delivery_risk"].values, index=ids)
    out["actual_late"] = out["shipment_id"].map(late).astype(int)

    backup = PROJECT_ROOT / "scored_shipments.backup.csv"
    if not backup.exists():
        shutil.copy(scored_path, backup)
    out.to_csv(scored_path, index=False)
    out.to_csv(PROJECT_ROOT / "data" / "scored_shipments.csv", index=False)
    print(f"Done. Wrote shap_json and actual_late for {len(out):,} shipments to {scored_path}")
    print(f"Original saved as {backup}")
    print("Note: SHAP values are on the log-odds scale (not percentage points).")


if __name__ == "__main__":
    main()
