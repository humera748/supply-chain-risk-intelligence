# Explainable Supply Chain Risk Intelligence & Adaptive Route Ranking

A production-grade, zero-leakage prototype for predictive supply chain delay risk intelligence and multi-criteria maritime route ranking, built for a B.Tech mini-project.

---

## 📌 Executive Summary & Focus
- **Primary Domain:** International deep-sea long-haul container logistics.
- **Problem:** Conventional supply chain delay prediction papers report artificial ~99–100% accuracy due to severe post-delivery data leakage. Furthermore, logistics managers lack plain-English explainability when evaluating at-risk cargo bookings.
- **Our Solution:**
  1. **Strict Booking-Time Feature Engineering:** Eliminates all post-order leakage (`Days for shipping (real)`, `Delivery Status`, `Order Status`).
  2. **Zero-Leakage Target Encoding:** Historical delay rates per shipping mode and maritime lane are computed **strictly from earlier orders** within a chronological time-based split.
  3. **Empirical Sea-Freight Filtering:** Drops local express modes (`Same Day`) and focuses on international ocean freight corridors from coastal/island centroids.
  4. **Explainable AI (XAI):** Tree SHAP computes exact marginal contributions, translating complex feature interactions into top-2 plain-English risk drivers.
  5. **Risk-Aware Route Ranking:** A transparent, multi-criteria decision framework:
     $$\text{Score} = w_1 \times \text{Risk} + w_2 \times \text{Cost} + w_3 \times (1 - \text{Reliability})$$
     ranking available maritime service tiers and carrier options.

---

## 🗂️ Project Directory Structure

```
supply_chain_risk_intelligence/
│
├── DataCoSupplyChainDataset.csv       # Raw DataCo supply chain dataset (180,519 rows)
├── requirements.txt                   # Production package dependencies
├── README.md                          # Comprehensive project documentation
├── app.py                             # Sleek Streamlit decision dashboard
├── scored_shipments.csv               # Model-scored test shipments (15,676 real shipments)
│
├── data/
│   ├── clean.csv                      # Cleaned zero-leakage dataset (156,197 rows)
│   ├── scored_shipments.csv           # Model-scored test shipments (data copy)
│   └── models/
│       ├── xgb_time_model.joblib      # Trained XGBoost classifier
│       └── feature_names.joblib       # Feature metadata
│
└── src/
    ├── __init__.py
    ├── clean.py                       # Data cleaning, leakage removal & before/after summary
    ├── ports.py                       # Approximate proxy seaport & maritime lane mapper
    └── train.py                       # Time-split training, baseline, SMOTE, ablation & SHAP
```

---

## 🔬 Zero-Leakage Pipeline Architecture

### 1. Cleaning & Domain Filtering (`src/clean.py`)
- **Leakage Columns Removed:**
  - `Days for shipping (real)`: Elapsed calendar transit days; strictly unknown until arrival at the destination port terminal.
  - `Delivery Status`: Categorical delivery outcome (`Late delivery`, `Advance shipping`, `Shipping on time`, `Shipping canceled`); directly discloses the target label.
  - `Order Status`: Post-checkout fulfillment status (`COMPLETE`, `CLOSED`, etc.).
- **PII / Uninformative Columns Removed:**
  - `Customer Email`, `Customer Password`, `Customer Fname`, `Customer Lname`, `Customer Street`, `Customer Zipcode`, `Order Zipcode`, `Product Image`, `Product Description`.
- **Sea-Freight Domain Rule:**
  - `Same Day` shipping mode is dropped (represents local urban courier dispatch, incompatible with ocean transit times).
  - Domestic US-to-US contiguous shipments (`Customer Country == 'EE. UU.'` and `Order Country == 'Estados Unidos'`) are excluded as domestic overland trucking.
  - Cross-border international corridors and island maritime routes originating from Puerto Rico are retained.

### 2. Approximate Proxy Port & Maritime Lane Mapping (`src/ports.py`)
Because DataCo records commercial order locations rather than maritime container terminal bills of lading, `src/ports.py` maps regional centroids to premier international commercial seaports:
- **Origin Gateways:** `Port of Los Angeles` (US West Coast), `Port of New York & New Jersey` (US North Atlantic), `Port of Savannah` (US Southeast), `Port of Houston` (US Gulf Coast), `Port of San Juan` (Caribbean Island Hub).
- **Destination Gateways:** Premier container hubs worldwide (e.g., `Port of Le Havre`, `Port of Hamburg`, `Port of Rotterdam`, `Port of Shanghai`, `Port of Singapore`, `Port of Santos`, `Port of Jawaharlal Nehru (Mumbai)`).
- Every order is converted into a trade corridor: `origin_port -> dest_port`.

### 3. Chronological Time-Based Evaluation & Zero-Leakage Target Encoding (`src/train.py`)
- **Time Split Protocol:** Orders are sorted chronologically by `order date (DateOrders)`. The first 80% (2015-01-01 to 2017-05-09; 124,957 orders) are used for training; the subsequent 20% (2017-05-09 to 2018-01-31; 31,240 orders) are reserved for out-of-time testing.
- **Historical Delay Rates:**
  $$\text{Mode Delay Rate} = \mathbb{E}[\text{Late} \mid \text{Shipping Mode}]_{\text{train}}$$
  $$\text{Lane Delay Rate} = \frac{N_{\text{lane}} \cdot \bar{y}_{\text{lane}} + m \cdot y_{\text{prior}}}{N_{\text{lane}} + m}$$
  Computed **exclusively on the training partition**. Test instances look up these historical rates without leaking future performance.

---

## 📊 Experimental Evaluation & Ablation Results

| Evaluation Protocol | Model Architecture | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Chronological Time-based** | **Logistic Regression Baseline** (Class-Weighted) | 70.35% | 83.04% | 57.69% | 0.6808 | 0.7361 |
| **Chronological Time-based** | **XGBoost Classifier** (Class-Weighted) | 70.24% | 82.86% | 57.62% | 0.6798 | 0.7325 |
| **Chronological Time-based** | **XGBoost + SMOTE Oversampling** | 56.98% | 91.55% | 23.71% | 0.3767 | 0.7050 |
| **Random 80/20 Split** | **XGBoost Classifier** (Literature Benchmark) | 70.36% | 84.17% | 57.18% | 0.6810 | 0.7507 |
| **Chronological Time-based** | **XGBoost WITH Leaky Columns** (Ablation Proof) | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |

### Key Academic Findings
1. **The Leakage Proof (Ablation Study):** Including post-delivery columns (`Days for shipping (real)`, `Delivery Status`) results in an artificial **1.0000 ROC-AUC / 100% Accuracy**. This empirically proves that prior academic studies claiming ~99–100% accuracy on DataCo are fundamentally flawed due to target leakage. Our booking-time model establishes the true, realistic operational predictive frontier (~73.3% ROC-AUC).
2. **SMOTE Behavior:** Synthetic minority oversampling degrades performance (F1 drops from 0.68 to 0.38). In balanced distributions (~45% on-time vs ~55% delayed), synthetic generation introduces boundary noise rather than discriminative separation.
3. **Time-Based vs Random Split:** The random split yields slightly higher ROC-AUC (0.7507 vs 0.7325) because customer patterns overlap across time; the chronological split is the authentic metric for real-world deployment.

---

## 🧠 Explainability (Tree SHAP)

For every evaluated test shipment, `shap.TreeExplainer` computes exact Shapley attributions across all features. The top 2 positive drivers pushing the shipment towards delay risk are translated into plain-English explanations:
- `"Carrier mode delay history + Tight scheduled transit time"`
- `"Maritime trade lane delay rate + Destination terminal clearance backlog"`
- `"Weekend dispatch schedule backlog + High-volume cargo batch complexity"`

These explanations are stored in `scored_shipments.csv` and rendered directly in the Streamlit UI.

---

## 🚢 Multi-Criteria Risk-Aware Route Ranking

Rather than claiming "optimization" (which implies dynamic vessel routing or continuous physics-based solvers), the prototype performs **risk-aware route ranking**:

$$\text{Score} = w_1 \times \text{Risk} + w_2 \times \text{Cost} + w_3 \times (1 - \text{Reliability})$$

- **Lower Score is Better.**
- **$w_1$ (Delay Risk Weight):** Penalizes predicted probability of lateness.
- **$w_2$ (Cost Weight):** Penalizes relative freight expense.
- **$w_3$ (Reliability Weight):** Penalizes historical carrier unreliability $(1 - \text{Reliability})$.
- **Interactive UI:** Logistics managers can use preset profiles (`Risk-Averse`, `Balanced`, `Cost Priority`) or custom sliders to dynamically rank service alternatives on that trade corridor.

---

## 🚀 Step-by-Step Execution Guide

### Prerequisites
Ensure Python 3.10+ is installed.

```bash
pip install -r requirements.txt
```

### 1. Run Data Cleaning
Executes the zero-leakage filter, drops PII, maps proxy ports, and prints the before/after summary table:
```bash
python src/clean.py
```
*Output: `data/clean.csv`*

### 2. Run Port & Lane Mapper Test
Verifies the approximate proxy seaport lookup tables:
```bash
python src/ports.py
```

### 3. Run Model Training & Explainability Export
Executes time-split training, baseline, SMOTE, ablation study, computes SHAP values, and exports scored shipments:
```bash
python src/train.py
```
*Outputs: `scored_shipments.csv`, `data/scored_shipments.csv`, `data/models/xgb_time_model.joblib`*

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

- **Page 1 (Live Risk Monitor):** View real-time KPIs, filter by carrier or port, inspect delay probabilities, and review global SHAP fleet analytics.
- **Page 2 (Route Alternatives):** Select any high-risk shipment, tune multi-criteria weights, and view transparent ranked carrier alternatives.

---

## 📌 Project Limitations & Academic Assumptions
1. **DataCo as a Sea-Freight Proxy:** DataCo Supply Chain records represent global cross-region commercial retail shipments. Sea freight transit times and container trade corridors are simulated using empirical order-to-destination geographic features.
2. **Approximate Port Mapping:** Because DataCo does not log maritime bill-of-lading terminal codes, seaports are approximately mapped via regional centroids to primary world commercial container ports.
3. **Carrier Equivalence:** Carrier entities correspond directly to the empirical shipping service modes in DataCo (`Standard Class`, `Second Class`, `First Class`).
4. **Ranking vs Optimization:** The decision engine performs *risk-aware route ranking* across discrete maritime carrier options rather than physical vessel route optimization or maritime navigation steering.
