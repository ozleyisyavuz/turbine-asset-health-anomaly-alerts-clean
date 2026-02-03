from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from turbine_asset_health.utils.config import PROCESSED_DIR, PROCESSED_SCADA_CSV


@dataclass
class DatasetConfig:
    days: int = 30
    freq_minutes: int = 10
    rated_power_kw: float = 3000.0
    seed: int = 42
    anomaly_rate: float = 0.04


def _power_curve_kw(ws: np.ndarray, rated_power_kw: float) -> np.ndarray:
    """Basit güç eğrisi (demo): cut-in=3, rated=12, cut-out=25 m/s"""
    ws = np.asarray(ws)
    out = np.zeros_like(ws, dtype=float)

    cut_in, rated, cut_out = 3.0, 12.0, 25.0

    mid = (ws >= cut_in) & (ws < rated)
    x = (ws[mid] - cut_in) / (rated - cut_in)
    out[mid] = rated_power_kw * (1 / (1 + np.exp(-8 * (x - 0.5))))

    hi = (ws >= rated) & (ws <= cut_out)
    out[hi] = rated_power_kw

    out[ws > cut_out] = 0.0
    return out


def make_dataset(cfg: DatasetConfig = DatasetConfig(), out_path: Path = PROCESSED_SCADA_CSV) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    periods = int((cfg.days * 24 * 60) / cfg.freq_minutes)
    ts = pd.date_range(end=pd.Timestamp.utcnow().floor("min"), periods=periods, freq=f"{cfg.freq_minutes}min")

    base = 7.5 + 2.0 * np.sin(np.linspace(0, 10 * math.pi, periods))
    ws = np.clip(base + rng.normal(0, 1.2, size=periods), 0, 30)

    wind_dir = (180 + 60 * np.sin(np.linspace(0, 2 * math.pi, periods)) + rng.normal(0, 15, size=periods)) % 360
    ambient_temp = 12 + 8 * np.sin(np.linspace(0, 4 * math.pi, periods)) + rng.normal(0, 1.5, size=periods)
    nacelle_temp = ambient_temp + 6 + rng.normal(0, 1.0, size=periods)

    rotor_rpm = np.clip(8 + 1.2 * ws + rng.normal(0, 1.0, size=periods), 0, 25)
    pitch_deg = np.clip(2 + 0.3 * np.maximum(ws - 10, 0) + rng.normal(0, 0.5, size=periods), 0, 20)

    expected_kw = _power_curve_kw(ws, cfg.rated_power_kw)
    power_kw = np.clip(expected_kw + rng.normal(0, 70, size=periods), 0, cfg.rated_power_kw * 1.05)

    
    is_anom = rng.random(periods) < cfg.anomaly_rate
    loss_type = rng.choice([0, 1], size=periods) 
    loss = np.where(loss_type == 0, rng.uniform(0.40, 0.70, size=periods), rng.uniform(0.20, 0.40, size=periods))
    power_kw = np.where(is_anom, power_kw * (1 - loss), power_kw)

    df = pd.DataFrame(
        {
            "timestamp": ts.astype("datetime64[ns]"),
            "wind_speed_mps": ws,
            "wind_dir_deg": wind_dir,
            "ambient_temp_c": ambient_temp,
            "nacelle_temp_c": nacelle_temp,
            "rotor_rpm": rotor_rpm,
            "pitch_deg": pitch_deg,
            "active_power_kw": power_kw,
            "is_anomaly_true": is_anom.astype(int),
        }
    )

    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    path = make_dataset()
    print(f"Saved dataset: {path}")


if __name__ == "__main__":
    main()
