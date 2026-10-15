"""Baseline threshold alerts for transformer telemetry."""

from pathlib import Path

import pandas as pd


def threshold_detector(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of telemetry with Normal, Warning, and Critical alerts."""
    result = df.copy()
    result["threshold_alert"] = "Normal"

    warning = (
        (result["load_current_a"] > 95)
        | (result["temperature_rise_c"] > 35)
    )
    result.loc[warning, "threshold_alert"] = "Warning"

    critical = (
        (result["load_current_a"] > 110)
        | (result["temperature_rise_c"] > 45)
        | (result["voltage_v"] < 207)
        | (result["voltage_v"] > 253)
    )
    result.loc[critical, "threshold_alert"] = "Critical"
    return result


if __name__ == "__main__":
    from src.data_processing import load_and_clean

    df = load_and_clean(Path("data/synthetic_transformer_data.csv"))
    df = threshold_detector(df)
    print(df["threshold_alert"].value_counts())

    overheating_period = df.loc[
        df["timestamp"].between(
            "2026-01-20 12:00",
            "2026-01-20 18:00",
            inclusive="left",
        )
        & (df["threshold_alert"] != "Normal")
    ]
    voltage_sag_period = df.loc[
        df["timestamp"].between(
            "2026-01-25 15:00",
            "2026-01-25 15:45",
            inclusive="left",
        )
        & (df["threshold_alert"] != "Normal")
    ]
    print("\nFirst 5 alerts during the overheating period:")
    print(overheating_period.head(5))
    print("\nFirst 5 alerts during the voltage sag period:")
    print(voltage_sag_period.head(5))
