"""Evaluate threshold and rolling-anomaly detectors on synthetic events."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.anomaly_detector import rolling_anomaly_detector
from src.data_processing import load_and_clean
from src.threshold_detector import threshold_detector


METHOD_PREDICTIONS = {
    "Fixed Threshold": "pred_threshold",
    "Rolling Anomaly": "pred_anomaly",
}


def _prepare_evaluation_data(csv_path: str | Path) -> pd.DataFrame:
    """Load telemetry, run detectors, and label the injected event periods."""
    df = load_and_clean(csv_path)
    df = threshold_detector(df)
    df = rolling_anomaly_detector(df)

    df["ground_truth"] = 0
    overload = df["timestamp"].between(
        "2026-01-20 12:00",
        "2026-01-20 18:00",
        inclusive="both",
    )
    voltage = df["timestamp"].between(
        "2026-01-25 15:00",
        "2026-01-25 15:45",
        inclusive="both",
    )
    df.loc[overload | voltage, "ground_truth"] = 1

    df["pred_threshold"] = (df["threshold_alert"] != "Normal").astype(int)
    df["pred_anomaly"] = (df["anomaly_alert"] == "Anomaly").astype(int)
    return df


def _calculate_metrics(
    ground_truth: pd.Series,
    predictions: pd.Series,
) -> tuple[dict[str, float], list[list[int]]]:
    """Calculate classification metrics and confusion matrix counts."""
    matrix = confusion_matrix(ground_truth, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    false_alarm_denominator = false_positive + true_negative
    missed_detection_denominator = true_positive + false_negative

    return (
        {
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score(
                ground_truth,
                predictions,
                zero_division=0,
            ),
            "False_Alarm_Rate": (
                false_positive / false_alarm_denominator
                if false_alarm_denominator
                else 0.0
            ),
            "Missed_Detection_Rate": (
                false_negative / missed_detection_denominator
                if missed_detection_denominator
                else 0.0
            ),
        },
        matrix.tolist(),
    )


def evaluate_detectors(
    csv_path: str | Path = "data/synthetic_transformer_data.csv",
) -> pd.DataFrame:
    """Evaluate both detectors and save their metric comparison to results/."""
    df = _prepare_evaluation_data(csv_path)
    rows = []
    for method, prediction_column in METHOD_PREDICTIONS.items():
        metrics, _ = _calculate_metrics(
            df["ground_truth"],
            df[prediction_column],
        )
        rows.append({"Method": method, **metrics})

    comparison = pd.DataFrame(
        rows,
        columns=[
            "Method",
            "Precision",
            "Recall",
            "F1_Score",
            "False_Alarm_Rate",
            "Missed_Detection_Rate",
        ],
    )
    output_path = Path("results/metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return comparison


if __name__ == "__main__":
    metrics_table = evaluate_detectors()
    print("Detector evaluation metrics:")
    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    evaluation_data = _prepare_evaluation_data(
        "data/synthetic_transformer_data.csv"
    )
    for method, prediction_column in METHOD_PREDICTIONS.items():
        _, matrix = _calculate_metrics(
            evaluation_data["ground_truth"],
            evaluation_data[prediction_column],
        )
        print(f"\nConfusion matrix for {method} (rows=true, columns=predicted):")
        print(pd.DataFrame(matrix, index=[0, 1], columns=[0, 1]))
