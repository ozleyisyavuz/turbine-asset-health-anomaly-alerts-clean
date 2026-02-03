from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from turbine_asset_health.utils.config import (
    PROCESSED_SCADA_CSV,
    MODELS_DIR,
    EXPECTED_MODEL_PATH,
    ANOMALY_MODEL_PATH,
    THRESHOLD_PATH,
)

@dataclass
class TrainConfig:
    contamination: float = 0.04
    random_state: int = 42

FEATURES = [
    "wind_speed_mps",
    "wind_dir_deg",
    "ambient_temp_c",
    "nacelle_temp_c",
    "rotor_rpm",
    "pitch_deg",
]

def train(cfg: TrainConfig = TrainConfig()) -> None:
    if not PROCESSED_SCADA_CSV.exists():
        raise FileNotFoundError("Data missing. First run: python -m turbine_asset_health.data.make_dataset")

    df = pd.read_csv(PROCESSED_SCADA_CSV).dropna()
    X = df[FEATURES]
    y = df["active_power_kw"].astype(float)

    expected_model = RandomForestRegressor(
        n_estimators=300,
        random_state=cfg.random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    expected_model.fit(X, y)
    yhat = expected_model.predict(X)
    mae = mean_absolute_error(y, yhat)

    residual = y - yhat
    ratio = np.where(yhat > 50, y / yhat, 1.0)

    anomaly_X = np.column_stack(
        [
            residual,
            np.clip(ratio, 0, 3),
            X["wind_speed_mps"].to_numpy(),
            X["rotor_rpm"].to_numpy(),
            X["pitch_deg"].to_numpy(),
        ]
    )

    anomaly_model = IsolationForest(
        n_estimators=300,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
    )
    anomaly_model.fit(anomaly_X)

    decision = anomaly_model.decision_function(anomaly_X)  # yüksek=normal
    anomaly_score = (-decision).astype(float)
    thr = float(np.quantile(anomaly_score, 1 - cfg.contamination))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(expected_model, EXPECTED_MODEL_PATH)
    joblib.dump(anomaly_model, ANOMALY_MODEL_PATH)
    THRESHOLD_PATH.write_text(json.dumps({"anomaly_score_threshold": thr, "mae_kw": mae}, indent=2))

    print(f"Saved: {EXPECTED_MODEL_PATH}")
    print(f"Saved: {ANOMALY_MODEL_PATH}")
    print(f"Saved: {THRESHOLD_PATH}")
    print(f"Train MAE (expected power): {mae:.2f} kW")
    print(f"Anomaly score threshold: {thr:.4f}")

def main() -> None:
    train()

if __name__ == "__main__":
    main()
