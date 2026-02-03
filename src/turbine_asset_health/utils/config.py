from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_SCADA_CSV = PROCESSED_DIR / "scada_demo.csv"
EXPECTED_MODEL_PATH = MODELS_DIR / "expected_power_model.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
