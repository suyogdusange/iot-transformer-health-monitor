"""Generate synthetic distribution-transformer telemetry."""

from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_COUNT = 30 * 24 * 60
RANDOM_SEED = 42
DEFAULT_MISSING_RATE = 200 / SAMPLE_COUNT


def generate_transformer_data(
    noise_multiplier: float = 1.0,
    missing_rate: float = DEFAULT_MISSING_RATE,
) -> pd.DataFrame:
    """Create synthetic readings with configurable sensor noise and missingness."""
    if noise_multiplier < 0:
        raise ValueError("noise_multiplier must be non-negative")
    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be between 0 and 1")

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=SAMPLE_COUNT,
        freq="min",
    )
    rng = np.random.default_rng(RANDOM_SEED)

    hours = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60
    daily_phase = 2 * np.pi * hours / 24

    ambient_temperature = (
        24 + 5 * np.sin(daily_phase - 2 * np.pi * 9 / 24)
        + rng.normal(0, 0.8 * noise_multiplier, SAMPLE_COUNT)
    )
    load_current = (
        75 + 12 * np.sin(daily_phase - 2 * np.pi * 8 / 24)
        + rng.normal(0, 3 * noise_multiplier, SAMPLE_COUNT)
    )
    voltage = 230 - 0.03 * (load_current - 75) + rng.normal(
        0, 0.8 * noise_multiplier, SAMPLE_COUNT
    )
    oil_temperature = (
        ambient_temperature
        + 17
        + 0.18 * (load_current - 60)
        + rng.normal(0, 0.5 * noise_multiplier, SAMPLE_COUNT)
    )

    overheating = (
        (timestamps >= "2026-01-20 12:00")
        & (timestamps < "2026-01-20 18:00")
    )
    load_current[overheating] += 25
    oil_temperature[overheating] += 12

    voltage_sag = (
        (timestamps >= "2026-01-25 15:00")
        & (timestamps < "2026-01-25 15:45")
    )
    voltage[voltage_sag] -= 25

    missing_count = round(SAMPLE_COUNT * missing_rate)
    missing_indices = rng.choice(
        SAMPLE_COUNT,
        size=missing_count,
        replace=False,
    )
    spike_pool = np.setdiff1d(np.arange(SAMPLE_COUNT), missing_indices)
    spike_indices = rng.choice(spike_pool, size=30, replace=False)
    spike_magnitudes = rng.uniform(8, 15, size=30) * noise_multiplier
    spike_directions = rng.choice(np.array([-1.0, 1.0]), size=30)
    oil_temperature[spike_indices] += spike_magnitudes * spike_directions
    oil_temperature[missing_indices] = np.nan

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_current_a": load_current,
            "voltage_v": voltage,
            "oil_temperature_c": oil_temperature,
            "ambient_temperature_c": ambient_temperature,
        }
    )


def generate_telemetry_data() -> pd.DataFrame:
    """Create the default synthetic telemetry dataset."""
    return generate_transformer_data()


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent.parent / "data" / (
        "synthetic_transformer_data.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_transformer_data().to_csv(output_path, index=False)
    print(f"Saved synthetic transformer data to {output_path}")
