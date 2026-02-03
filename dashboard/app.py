from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Wind Turbine Asset Health", layout="wide")
st.title("Wind Turbine Asset Health — Anomaly Dashboard")

api_url = st.text_input("API URL", "http://127.0.0.1:8000")
data_path = st.text_input("Dataset path", "data/processed/scada_demo.csv")

if not Path(data_path).exists():
    st.warning("Dataset not found. Run: python -m turbine_asset_health.data.make_dataset")
    st.stop()

df = pd.read_csv(data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna().sort_values("timestamp").tail(500)

st.subheader("Recent SCADA rows (demo)")
st.dataframe(df.tail(10), use_container_width=True)

rows = []
for _, r in df.tail(100).iterrows():
    rows.append(
        {
            "timestamp": r["timestamp"].isoformat(),
            "wind_speed_mps": float(r["wind_speed_mps"]),
            "wind_dir_deg": float(r["wind_dir_deg"]),
            "ambient_temp_c": float(r["ambient_temp_c"]),
            "nacelle_temp_c": float(r["nacelle_temp_c"]),
            "rotor_rpm": float(r["rotor_rpm"]),
            "pitch_deg": float(r["pitch_deg"]),
            "active_power_kw": float(r["active_power_kw"]),
        }
    )

resp = requests.post(f"{api_url}/batch_anomaly", json={"rows": rows}, timeout=30)
if resp.status_code != 200:
    st.error(resp.text)
    st.stop()

scores = pd.DataFrame(resp.json()["results"])
plot_df = df.tail(100).reset_index(drop=True).join(scores)

c1, c2 = st.columns(2)

with c1:
    st.plotly_chart(px.line(plot_df, x="timestamp", y="active_power_kw", title="Active Power (kW)"), use_container_width=True)

with c2:
    st.plotly_chart(px.line(plot_df, x="timestamp", y="anomaly_score", title="Anomaly Score"), use_container_width=True)

anom_points = plot_df[plot_df["is_anomaly"] == True]
st.write(f"Anomaly count (last 100): **{len(anom_points)}**")
st.dataframe(
    anom_points[["timestamp", "active_power_kw", "expected_power_kw", "residual_kw", "anomaly_score", "threshold", "is_anomaly"]],
    use_container_width=True,
)
