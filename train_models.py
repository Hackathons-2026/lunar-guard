"""
Step 2: Train Scikit-Learn ML Models on Lunar South Pole Terrain & Horizon Data
Run: python train_models.py
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)

def train():
    csv_path = "data/south_pole_dem.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Run fetch_nasa_data.py first to generate data/south_pole_dem.csv")

    df = pd.read_csv(csv_path)
    horizon_cols = [f"h_{deg}" for deg in range(0, 360, 15)]

    # 1. Random Forest Regressor: Predicts 24-sector 360° horizon obstruction for ANY clicked (lat, lon)
    X_coords = df[["lat", "lon"]]
    y_horizon = df[horizon_cols]

    rf_horizon = RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42)
    rf_horizon.fit(X_coords, y_horizon)
    joblib.dump(rf_horizon, "models/horizon_rf.pkl")
    print("Trained & saved models/horizon_rf.pkl")

    # 2. Random Forest Regressor: Predicts terrain slope risk score for ANY clicked (lat, lon)
    rf_risk = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42)
    rf_risk.fit(X_coords, df["risk"])
    joblib.dump(rf_risk, "models/risk_rf.pkl")
    print("Trained & saved models/risk_rf.pkl")

    # 3. KNN Site Similarity & Multi-Objective Ranker
    df["mean_horizon"] = df[horizon_cols].mean(axis=1)
    rank_features = ["mean_horizon", "risk", "lat"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[rank_features])

    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(X_scaled)
    joblib.dump({"knn": knn, "scaler": scaler, "candidates": df.iloc[:5].to_dict(orient="records")}, "models/site_knn.pkl")
    print("Trained & saved models/site_knn.pkl")

if __name__ == "__main__":
    train()
