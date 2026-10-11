"""
src/clean.py - Data Cleaning Pipeline for Sea-Freight Delay Risk Intelligence
=============================================================================
Cleans the raw DataCoSupplyChainDataset.csv with strict zero-leakage principles:
1. Parses booking and shipment timestamps as datetimes.
2. Drops leakage columns not known at booking time (Days for shipping (real), Delivery Status, Order Status).
3. Drops PII and uninformative metadata (Customer Email, Password, Fname, Lname, Street, Zipcode, Product Image/Description).
4. Enforces sea-freight international focus:
   - Drops 'Same Day' shipping mode (short-haul urban parcel express, incompatible with maritime shipping).
   - Retains international/cross-region orders (maritime container corridors from US/Puerto Rico gateways to overseas markets).
5. Handles duplicates and missing values.
6. Maps proxy ports & lanes via src.ports.
7. Prints a before/after summary table and target class ratios.
8. Exports cleaned dataset to data/clean.csv.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path to allow importing src modules
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ports import assign_ports_and_lanes


# Explicit column drop lists with academic rationale
LEAKAGE_COLUMNS = {
    "Days for shipping (real)": "Actual transit days elapsed; strictly post-delivery outcome.",
    "Delivery Status": "Final delivery status ('Late delivery', 'Advance', etc.); directly leaks target.",
    "Order Status": "Fulfillment status ('COMPLETE', 'CLOSED', etc.); known only after processing/delivery."
}

PII_USELESS_COLUMNS = {
    "Customer Email": "PII - Personally Identifiable Information.",
    "Customer Password": "PII - Sensitive security credential.",
    "Customer Fname": "PII - Customer first name.",
    "Customer Lname": "PII - Customer last name.",
    "Customer Street": "PII - Granular street address.",
    "Customer Zipcode": "PII - Highly granular local postal code.",
    "Order Zipcode": "Extremely sparse metadata (>86% missing values).",
    "Product Image": "Static web resource URL string.",
    "Product Description": "Unused metadata (100% missing values in DataCo)."
}


def clean_dataset(raw_csv_path: str = "DataCoSupplyChainDataset.csv",
                  output_csv_path: str = "data/clean.csv") -> pd.DataFrame:
    """
    Executes the end-to-end cleaning and preprocessing pipeline.
    """
    raw_path = Path(raw_csv_path)
    if not raw_path.exists():
        # Look in PROJECT_ROOT if relative path didn't resolve
        raw_path = PROJECT_ROOT / raw_csv_path

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")

    print("=" * 80)
    print("STEP 1: LOADING RAW DATASET")
    print("=" * 80)
    print(f"Reading: {raw_path} with encoding='latin-1'...")
    df_raw = pd.read_csv(raw_path, encoding="latin-1", low_memory=False)
    
    initial_rows, initial_cols = df_raw.shape
    print(f"Raw Dataset loaded successfully: {initial_rows:,} rows, {initial_cols} columns.")

    # -----------------------------------------------------------------------
    # Step 1: Parse Datetimes
    # -----------------------------------------------------------------------
    print("\nParsing order and shipping timestamps...")
    df_raw["order date (DateOrders)"] = pd.to_datetime(
        df_raw["order date (DateOrders)"], errors="coerce"
    )
    df_raw["shipping date (DateOrders)"] = pd.to_datetime(
        df_raw["shipping date (DateOrders)"], errors="coerce"
    )

    # -----------------------------------------------------------------------
    # Step 2: Drop Target Leakage and PII Columns
    # -----------------------------------------------------------------------
    columns_to_drop = list(LEAKAGE_COLUMNS.keys()) + list(PII_USELESS_COLUMNS.keys())
    existing_cols_to_drop = [c for c in columns_to_drop if c in df_raw.columns]
    
    df_clean = df_raw.drop(columns=existing_cols_to_drop)

    # -----------------------------------------------------------------------
    # Step 3: Sea-Freight Focus & Cross-Region Filtering
    # -----------------------------------------------------------------------
    # SEA-FREIGHT FILTERING RULE EXPLANATION:
    # 1. Drop 'Same Day' shipping: In container logistics, same-day dispatch
    #    represents immediate local ground delivery, not deep-sea container voyages.
    # 2. Exclude purely domestic US contiguous overland freight:
    #    Orders where Customer Country is 'EE. UU.' (USA) and Order Country is
    #    'Estados Unidos' (USA) represent domestic continental highway/rail trucking.
    # 3. Retain International and Cross-Region corridors:
    #    - Inter-continental long-haul lanes to Europe, Pacific Asia, LATAM, Africa, etc.
    #    - Shipments originating from Puerto Rico (island territory requiring ocean vessel transit).
    print("\nApplying Sea-Freight Domain Filter:")
    pre_filter_len = len(df_clean)
    
    # 3a. Drop 'Same Day' mode
    df_clean = df_clean[df_clean["Shipping Mode"] != "Same Day"].copy()
    same_day_dropped = pre_filter_len - len(df_clean)
    print(f" - Dropped {same_day_dropped:,} 'Same Day' orders (local parcel mode).")

    # 3b. Keep international & island maritime lanes
    is_domestic_us = (
        (df_clean["Customer Country"] == "EE. UU.") &
        (df_clean["Order Country"].astype(str).str.strip().str.lower() == "estados unidos")
    )
    df_clean = df_clean[~is_domestic_us].copy()
    domestic_dropped = pre_filter_len - same_day_dropped - len(df_clean)
    print(f" - Dropped {domestic_dropped:,} domestic US-to-US overland trucking orders.")
    print(f" - Retained {len(df_clean):,} international & cross-region maritime orders.")

    # -----------------------------------------------------------------------
    # Step 4: Deduplication and Missing Value Imputation
    # -----------------------------------------------------------------------
    pre_dedup_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = pre_dedup_len - len(df_clean)
    print(f"\nDuplicates removed: {duplicates_removed:,}")

    # Check remaining nulls
    null_counts = df_clean.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print(f"Handling missing values in: {null_cols.to_dict()}")
        for col in null_cols.index:
            if df_clean[col].dtype in ["float64", "int64"]:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna("Unknown")
    else:
        print("No missing values found in remaining operational columns.")

    # -----------------------------------------------------------------------
    # Step 5: Assign Proxy Seaports & Shipping Lanes
    # -----------------------------------------------------------------------
    print("\nAssigning proxy origin/destination seaports and maritime trade lanes...")
    df_clean = assign_ports_and_lanes(df_clean)

    # -----------------------------------------------------------------------
    # Step 6: Save Output
    # -----------------------------------------------------------------------
    out_path = Path(output_csv_path)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / output_csv_path
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving cleaned dataset to: {out_path} ...")
    df_clean.to_csv(out_path, index=False)
    print("Cleaned dataset successfully saved!")

    # -----------------------------------------------------------------------
    # Step 7: Print Summary Report & Class Ratios
    # -----------------------------------------------------------------------
    final_rows, final_cols = df_clean.shape
    late_counts = df_clean["Late_delivery_risk"].value_counts()
    late_ratio = df_clean["Late_delivery_risk"].value_counts(normalize=True)

    print("\n" + "=" * 80)
    print("DATASET CLEANING SUMMARY TABLE")
    print("=" * 80)
    summary_data = [
        ["Initial Raw Shape", f"{initial_rows:,} rows, {initial_cols} columns"],
        ["Final Clean Shape", f"{final_rows:,} rows, {final_cols} columns"],
        ["Total Rows Filtered", f"{initial_rows - final_rows:,} ({(initial_rows - final_rows)/initial_rows:.1%})"],
        ["Columns Removed (Leakage + PII)", f"{len(existing_cols_to_drop)} columns"],
        ["Columns Added (Proxy Ports)", "3 columns (origin_port, dest_port, lane)"],
    ]
    summary_df = pd.DataFrame(summary_data, columns=["Pipeline Metric", "Value"])
    print(summary_df.to_string(index=False))

    print("\n" + "-" * 80)
    print("COLUMNS REMOVED AND ACADEMIC RATIONALE")
    print("-" * 80)
    rationale_rows = []
    for col, rat in {**LEAKAGE_COLUMNS, **PII_USELESS_COLUMNS}.items():
        if col in df_raw.columns:
            rationale_rows.append([col, "Leakage" if col in LEAKAGE_COLUMNS else "PII/Useless", rat])
    rat_df = pd.DataFrame(rationale_rows, columns=["Column Name", "Category", "Rationale"])
    print(rat_df.to_string(index=False))

    print("\n" + "-" * 80)
    print("TARGET CLASS DISTRIBUTION (Late_delivery_risk)")
    print("-" * 80)
    class_df = pd.DataFrame({
        "Class": ["On-Time / Early (0)", "Delayed / Late Risk (1)"],
        "Count": [late_counts.get(0, 0), late_counts.get(1, 0)],
        "Proportion": [f"{late_ratio.get(0, 0):.2%}", f"{late_ratio.get(1, 0):.2%}"]
    })
    print(class_df.to_string(index=False))
    print("=" * 80 + "\n")

    return df_clean


if __name__ == "__main__":
    clean_dataset()
