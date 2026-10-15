"""Rolling-window anomaly detection for transformer telemetry."""

from pathlib import Path

import numpy as np
import pandas as pd


def rolling_anomaly_detector(
    df: pd.DataFrame,
    window: int = 60,
    z_threshold: float = 3.5,
) -> pd.DataFrame:
    """Return telemetry with rolling MAD anomaly flags and a combined alert."""
    result = df.copy()
    parameters = [
        "load_current_a",
        "temperature_rise_c",
        "voltage_v",
    ]

    for column in parameters:
        rolling = result[column].rolling(
            window=window,
            center=True,
            min_periods=20,
        )
        rolling_median = rolling.median()
        mad = rolling.apply(
            lambda values: np.median(
                np.abs(values - np.median(values))
            ),
            raw=True,
        )
        modified_z = (
            0.6745
            * (result[column] - rolling_median)
            / (mad + 1e-6)
        )
        result[f"{column}_anomaly"] = modified_z.abs() > z_threshold

    anomaly_columns = [f"{column}_anomaly" for column in parameters]
    result["anomaly_alert"] = np.where(
        result[anomaly_columns].any(axis=1),
        "Anomaly",
        "Normal",
    )
    return result


if __name__ == "__main__":
    from src.data_processing import load_and_clean

    df = load_and_clean(Path("data/synthetic_transformer_data.csv"))
    df = rolling_anomaly_detector(df)
    print(df["anomaly_alert"].value_counts())

    anomaly_columns = [
        "load_current_a_anomaly",
        "temperature_rise_c_anomaly",
        "voltage_v_anomaly",
    ]
    print("\nAnomaly flag counts by parameter:")
    for column in anomaly_columns:
        print(f"\n{column}:")
        print(df[column].value_counts())

    overheating_anomalies = df.loc[
        df["timestamp"].between(
            "2026-01-20 12:00",
            "2026-01-20 18:00",
            inclusive="left",
        )
        & (df["anomaly_alert"] == "Anomaly")
    ]
    voltage_sag_anomalies = df.loc[
        df["timestamp"].between(
            "2026-01-25 15:00",
            "2026-01-25 15:45",
            inclusive="left",
        )
        & (df["anomaly_alert"] == "Anomaly")
    ]
    print("\nFirst 5 anomalies during the overheating event:")
    print(overheating_anomalies.head(5))
    print("\nFirst 5 anomalies during the voltage sag:")
    print(voltage_sag_anomalies.head(5))
