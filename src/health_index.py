"""Calculate a thermal-load health index for transformer telemetry."""

from pathlib import Path

import numpy as np
import pandas as pd


def calculate_health_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of telemetry with a health score and status category."""
    result = df.copy()

    load_penalty = np.clip((result["load_current_a"] - 90) / 40, 0, 1)
    temperature_penalty = np.clip(
        (result["temperature_rise_c"] - 30) / 25,
        0,
        1,
    )
    voltage_penalty = np.clip(
        (np.abs(result["voltage_v"] - 230) - 10) / 25,
        0,
        1,
    )
    result["health_index"] = 100 * (
        1
        - (
            0.4 * load_penalty
            + 0.4 * temperature_penalty
            + 0.2 * voltage_penalty
        )
    )
    result["health_status"] = pd.cut(
        result["health_index"],
        bins=[-np.inf, 60, 80, np.inf],
        labels=["Critical", "Warning", "Normal"],
    )
    return result


if __name__ == "__main__":
    from src.data_processing import load_and_clean

    df = load_and_clean(Path("data/synthetic_transformer_data.csv"))
    df = calculate_health_index(df)

    print("Health index summary:")
    print(df["health_index"].agg(["mean", "min", "max", "std"]))
    print("\nHealth status counts:")
    print(df["health_status"].value_counts())

    normal_operation = ~(
        df["timestamp"].between(
            "2026-01-20 12:00",
            "2026-01-20 18:00",
            inclusive="left",
        )
        | df["timestamp"].between(
            "2026-01-25 15:00",
            "2026-01-25 15:45",
            inclusive="left",
        )
    )
    overheating_event = df["timestamp"].between(
        "2026-01-20 12:00",
        "2026-01-20 18:00",
        inclusive="left",
    )
    voltage_sag_event = df["timestamp"].between(
        "2026-01-25 15:00",
        "2026-01-25 15:45",
        inclusive="left",
    )

    print("\nMean health index during normal operation:")
    print(df.loc[normal_operation, "health_index"].mean())
    print("\nMean health index during the overheating event:")
    print(df.loc[overheating_event, "health_index"].mean())
    print("\nMean health index during the voltage sag:")
    print(df.loc[voltage_sag_event, "health_index"].mean())
