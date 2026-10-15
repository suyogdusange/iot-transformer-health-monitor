import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.anomaly_detector import rolling_anomaly_detector
from src.data_processing import load_and_clean
from src.health_index import calculate_health_index
from src.threshold_detector import threshold_detector


st.set_page_config(page_title="Transformer Health Monitor", layout="wide")
st.title("IoT-Based Distribution Transformer Health Monitoring")
st.caption(
    "Demonstration using synthetic transformer telemetry. "
    "Not a field-validated diagnostic system."
)

uploaded = st.file_uploader("Upload transformer CSV", type=["csv"])
if uploaded:
    df = load_and_clean(uploaded)
else:
    try:
        df = load_and_clean("data/synthetic_transformer_data.csv")
    except FileNotFoundError:
        st.error("Run src/data_generator.py first to generate the dataset.")
        st.stop()

df = threshold_detector(df)
df = rolling_anomaly_detector(df)
df = calculate_health_index(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Load", f"{df.load_current_a.mean():.2f} A")
col2.metric("Max Temp Rise", f"{df.temperature_rise_c.max():.2f} °C")
col3.metric("Threshold Alerts", int((df.threshold_alert != "Normal").sum()))
col4.metric("Anomaly Points", int((df.anomaly_alert == "Anomaly").sum()))

st.subheader("Transformer Operating Parameters")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    df.timestamp,
    df.load_current_a,
    label="Load Current (A)",
    color="#1f77b4",
)
ax.plot(
    df.timestamp,
    df.temperature_rise_c,
    label="Temperature Rise (°C)",
    color="#ff7f0e",
)
ax.set_xlabel("Time")
ax.set_ylabel("Measurement")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Health Index")
fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(df.timestamp, df.health_index, label="Health Index", color="green")
ax2.axhline(80, color="orange", linestyle="--", label="Warning (80)")
ax2.axhline(60, color="red", linestyle="--", label="Critical (60)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Health Index")
ax2.set_ylim(0, 105)
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Recent Alerts")
alerts = df.loc[
    (df["threshold_alert"] != "Normal")
    | (df["anomaly_alert"] == "Anomaly")
].tail(50)

st.dataframe(
    alerts[
        [
            "timestamp",
            "load_current_a",
            "temperature_rise_c",
            "voltage_v",
            "threshold_alert",
            "anomaly_alert",
            "health_status",
        ]
    ],
    width="stretch",
)

st.download_button(
    label="Download alert log",
    data=alerts.to_csv(index=False).encode("utf-8"),
    file_name="transformer_alerts.csv",
    mime="text/csv",
)
