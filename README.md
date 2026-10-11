# Explainable Supply Chain Risk Intelligence and Adaptive Route Ranking

B.Tech mini project. Predicts delay risk for international shipments at booking time,
explains the reasons, and ranks safer service options. Focus: long-haul sea freight
(DataCo is used as a proxy dataset).

## What it does
1. Cleans the DataCo Smart Supply Chain dataset (removes columns only known after delivery, and personal columns).
2. Trains Logistic Regression (baseline) and XGBoost on a time-based split; reports Precision, Recall, F1, ROC-AUC.
3. Explains each risk score with its top reasons.
4. Ranks service options with a transparent score: w1*Risk + w2*Cost + w3*(1-Reliability). This is risk-aware ranking, not route optimization.
5. Streamlit dashboard: Overview, Shipment details, Route alternatives, How it works.

## Run
pip install -r requirements.txt
python src/clean.py
python src/train.py
streamlit run app_v2.py

Dataset (not included, too large): https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
Put DataCoSupplyChainDataset.csv in the project folder.

## Limitations
- DataCo is general shipment data, used as a proxy for sea freight; port mapping is approximate.
- Cost is an assumed level, not from the data.
- Real-world event data (GDELT: strikes, storms) is the next step and is not connected yet.
