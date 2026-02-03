from __future__ import annotations
from pydantic import BaseModel, Field

class AnomalyRequest(BaseModel):
    timestamp: str = Field(..., examples=["2026-02-02T12:00:00Z"])
    wind_speed_mps: float
    wind_dir_deg: float
    ambient_temp_c: float
    nacelle_temp_c: float
    rotor_rpm: float
    pitch_deg: float
    active_power_kw: float

class AnomalyResponse(BaseModel):
    expected_power_kw: float
    residual_kw: float
    anomaly_score: float
    threshold: float
    is_anomaly: bool

class BatchAnomalyRequest(BaseModel):
    rows: list[AnomalyRequest]
