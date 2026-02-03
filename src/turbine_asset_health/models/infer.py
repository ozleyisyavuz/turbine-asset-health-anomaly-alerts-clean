from __future__ import annotations

import json
import joblib
import numpy as np

from turbine_asset_health.utils.config import EXPECTED_MODEL_PATH, ANOMALY_MODEL_PATH, THRESHOLD_PATH
from turbine_asset_health.models.train import FEATURES

def _load_threshold() -> float:
    obj = json.loads(THRESHOLD_PATH.read_text())
    return float(obj["anomaly_score_threshold"])

def score_one(features: dict) -> dict:
    if not EXPECTED_MODEL_PATH.exists() or not ANOMALY_MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        raise FileNotFoundError("Models missing. First run: python -m turbine_asset_health.models.train")

    expected_model = joblib.load(EXPECTED_MODEL_PATH)
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
    thr = _load_threshold()

    x = np.array([[float(features[k]) for k in FEATURES]], dtype=float)
    expected_kw = float(expected_model.predict(x)[0])

    actual_kw = float(features["active_power_kw"])
    residual = actual_kw - expected_kw
    ratio = (actual_kw / expected_kw) if expected_kw > 50 else 1.0

    anomaly_x = np.array(
        [[residual, np.clip(ratio, 0, 3), float(features["wind_speed_mps"]), float(features["rotor_rpm"]), float(features["pitch_deg"])]],
        dtype=float,
    )
    decision = float(anomaly_model.decision_function(anomaly_x)[0])
    anomaly_score = float(-decision)

    return {
        "expected_power_kw": expected_kw,
        "residual_kw": float(residual),
        "anomaly_score": anomaly_score,
        "threshold": thr,
        "is_anomaly": bool(anomaly_score >= thr),
    }
