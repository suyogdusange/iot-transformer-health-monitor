"""Generate publication-style figures for the transformer monitoring study."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.anomaly_detector import rolling_anomaly_detector
from src.data_processing import load_and_clean
from src.health_index import calculate_health_index
from src.threshold_detector import threshold_detector


FIGURES_DIRECTORY = Path("results/figures")
FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def _save_figure(figure: plt.Figure, filename: str) -> Path:
    """Save a figure using the publication output settings."""
    output_path = FIGURES_DIRECTORY / filename
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _load_detector_evaluation_data() -> pd.DataFrame:
    """Load telemetry, calculate detector outputs, and label injected events."""
    df = load_and_clean("data/synthetic_transformer_data.csv")
    df = threshold_detector(df)
    df = rolling_anomaly_detector(df)
    overload = df["timestamp"].between(
        "2026-01-20 12:00",
        "2026-01-20 18:00",
        inclusive="both",
    )
    voltage_sag = df["timestamp"].between(
        "2026-01-25 15:00",
        "2026-01-25 15:45",
        inclusive="both",
    )
    df["ground_truth"] = (overload | voltage_sag).astype(int)
    df["pred_threshold"] = (df["threshold_alert"] != "Normal").astype(int)
    df["pred_anomaly"] = (df["anomaly_alert"] == "Anomaly").astype(int)
    return df


def plot_telemetry_and_health() -> Path:
    """Plot operating telemetry and health index with event windows shaded."""
    df = load_and_clean("data/synthetic_transformer_data.csv")
    df = calculate_health_index(df)

    figure, (telemetry_axis, health_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    telemetry_axis.plot(
        df["timestamp"],
        df["load_current_a"],
        label="Load Current (A)",
        color="tab:blue",
    )
    temperature_axis = telemetry_axis.twinx()
    temperature_axis.plot(
        df["timestamp"],
        df["temperature_rise_c"],
        label="Temperature Rise (°C)",
        color="tab:orange",
    )
    telemetry_axis.set_ylabel("Load Current (A)")
    temperature_axis.set_ylabel("Temperature Rise (°C)")

    event_windows = [
        (
            pd.Timestamp("2026-01-20 12:00"),
            pd.Timestamp("2026-01-20 18:00"),
            "Overload",
            "tab:red",
        ),
        (
            pd.Timestamp("2026-01-25 15:00"),
            pd.Timestamp("2026-01-25 15:45"),
            "Voltage sag",
            "tab:purple",
        ),
    ]
    for start, end, label, color in event_windows:
        telemetry_axis.axvspan(
            start,
            end,
            color=color,
            alpha=0.12,
            label=label,
        )
        health_axis.axvspan(start, end, color=color, alpha=0.12)

    telemetry_lines, telemetry_labels = telemetry_axis.get_legend_handles_labels()
    temperature_lines, temperature_labels = (
        temperature_axis.get_legend_handles_labels()
    )
    telemetry_axis.legend(
        telemetry_lines + temperature_lines,
        telemetry_labels + temperature_labels,
        loc="upper left",
        ncol=2,
    )
    telemetry_axis.set_title("Transformer Operating Telemetry")

    health_axis.plot(
        df["timestamp"],
        df["health_index"],
        label="Health Index",
        color="tab:green",
    )
    health_axis.axhline(
        80,
        color="orange",
        linestyle="--",
        label="Warning (80)",
    )
    health_axis.axhline(
        60,
        color="red",
        linestyle="--",
        label="Critical (60)",
    )
    health_axis.set_ylabel("Health Index")
    health_axis.set_xlabel("Time")
    health_axis.set_ylim(0, 105)
    health_axis.legend(loc="lower left")
    figure.tight_layout()
    return _save_figure(figure, "fig1_telemetry_events.png")


def plot_confusion_matrices() -> Path:
    """Plot annotated confusion matrices for both detectors."""
    df = _load_detector_evaluation_data()
    methods = [
        ("Fixed Threshold", "pred_threshold"),
        ("Rolling Anomaly Detector", "pred_anomaly"),
    ]
    labels = ["Normal (0)", "Abnormal (1)"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for axis, (method, prediction_column) in zip(axes, methods):
        matrix = confusion_matrix(
            df["ground_truth"],
            df[prediction_column],
            labels=[0, 1],
        )
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(method)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(2), labels=labels)
        axis.set_yticks(range(2), labels=labels)
        threshold = matrix.max() / 2
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.tight_layout()
    return _save_figure(figure, "fig2_confusion_matrices.png")


def plot_robustness_curves() -> Path:
    """Plot derived F1 scores against noise at each missing-data rate."""
    results = pd.read_csv("results/robustness.csv")
    required_columns = {
        "Noise_Multiplier",
        "Missing_Rate",
        "Method",
        "Precision",
        "Recall",
    }
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(
            f"Robustness results are missing columns: {sorted(missing_columns)}"
        )

    denominator = results["Precision"] + results["Recall"]
    results["F1_Score"] = np.divide(
        2 * results["Precision"] * results["Recall"],
        denominator,
        out=np.zeros(len(results), dtype=float),
        where=denominator.to_numpy() != 0,
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    methods = results["Method"].drop_duplicates().tolist()
    missing_rates = sorted(results["Missing_Rate"].unique())
    colors = plt.get_cmap("tab10")
    method_markers = {
        "Fixed Threshold": "o",
        "Rolling Anomaly": "s",
    }
    for rate_index, missing_rate in enumerate(missing_rates):
        rate_data = results[results["Missing_Rate"] == missing_rate]
        for method_index, method in enumerate(methods):
            method_data = rate_data[rate_data["Method"] == method].sort_values(
                "Noise_Multiplier"
            )
            axis.plot(
                method_data["Noise_Multiplier"],
                method_data["F1_Score"],
                marker=method_markers.get(method, "o"),
                linestyle="-" if method_index == 0 else "--",
                color=colors(rate_index),
                label=f"{method}, missing {missing_rate:.0%}",
            )

    axis.set_title("Detector F1 Score Across Noise and Missingness")
    axis.set_xlabel("Noise Multiplier")
    axis.set_ylabel("F1 Score")
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    return _save_figure(figure, "fig3_robustness_curves.png")


if __name__ == "__main__":
    for plotting_function in (
        plot_telemetry_and_health,
        plot_confusion_matrices,
        plot_robustness_curves,
    ):
        saved_path = plotting_function()
        print(f"Saved figure: {saved_path}")
