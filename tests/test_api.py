from fastapi.testclient import TestClient

from turbine_asset_health.app.main import app
from turbine_asset_health.data.make_dataset import make_dataset
from turbine_asset_health.models.train import train

client = TestClient(app)

def _ensure_artifacts():
    make_dataset()
    train()

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_anomaly_endpoint():
    _ensure_artifacts()

    payload = {
        "timestamp": "2026-02-02T12:00:00Z",
        "wind_speed_mps": 8.2,
        "wind_dir_deg": 190,
        "ambient_temp_c": 12,
        "nacelle_temp_c": 18,
        "rotor_rpm": 17,
        "pitch_deg": 3,
        "active_power_kw": 1200,
    }

    r = client.post("/anomaly", json=payload)
    assert r.status_code == 200, r.text

    data = r.json()
    assert "expected_power_kw" in data
    assert "anomaly_score" in data
    assert "is_anomaly" in data
    assert isinstance(data["is_anomaly"], bool)
