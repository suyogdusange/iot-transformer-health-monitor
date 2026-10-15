"""Clean synthetic transformer telemetry and engineer thermal features."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "load_current_a",
    "voltage_v",
    "oil_temperature_c",
    "ambient_temperature_c",
]


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Load telemetry, clean measurements, and add thermal feature columns."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    df["load_current_a"] = df["load_current_a"].clip(lower=0)
    df["voltage_v"] = df["voltage_v"].clip(lower=0)
    df = df.set_index("timestamp")
    df["oil_temperature_c"] = df["oil_temperature_c"].interpolate(
        method="time",
        limit=10,
    )
    df = df.reset_index()
    df["temperature_rise_c"] = (
        df["oil_temperature_c"] - df["ambient_temperature_c"]
    )
    df["data_quality_flag"] = df["oil_temperature_c"].isna().map(
        {True: "missing_temperature", False: "ok"}
    )
    return df


if __name__ == "__main__":
    df = load_and_clean(Path("data/synthetic_transformer_data.csv"))
    df.info()
    print(df["temperature_rise_c"].describe())
    print(df["data_quality_flag"].value_counts())
