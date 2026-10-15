"""Evaluate detector performance across sensor noise and missingness levels."""

from pathlib import Path
import tempfile

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from src.anomaly_detector import rolling_anomaly_detector
from src.data_generator import generate_transformer_data
from src.data_processing import load_and_clean
from src.threshold_detector import threshold_detector


NOISE_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]
MISSING_RATES = [0.0, 0.01, 0.05, 0.10]
RESULT_COLUMNS = [
    "Noise_Multiplier",
    "Missing_Rate",
    "Method",
    "Precision",
    "Recall",
    "False_Alarm_Rate",
]
METHOD_PREDICTIONS = {
    "Fixed Threshold": "pred_threshold",
    "Rolling Anomaly": "pred_anomaly",
}


def _add_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Label only the configured synthetic overload and voltage-sag periods."""
    result = df.copy()
    overload = result["timestamp"].between(
        "2026-01-20 12:00",
        "2026-01-20 18:00",
        inclusive="both",
    )
    voltage_sag = result["timestamp"].between(
        "2026-01-25 15:00",
        "2026-01-25 15:45",
        inclusive="both",
    )
    result["ground_truth"] = (overload | voltage_sag).astype(int)
    return result


def evaluate_robustness() -> pd.DataFrame:
    """Compare detectors across every configured noise and missing-data level."""
    results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "robustness_telemetry.csv"
        for noise_multiplier in NOISE_MULTIPLIERS:
            for missing_rate in MISSING_RATES:
                generated = generate_transformer_data(
                    noise_multiplier=noise_multiplier,
                    missing_rate=missing_rate,
                )
                generated.to_csv(csv_path, index=False)
                df = load_and_clean(csv_path)
                df = threshold_detector(df)
                df = rolling_anomaly_detector(df)
                df = _add_ground_truth(df)
                df["pred_threshold"] = (
                    df["threshold_alert"] != "Normal"
                ).astype(int)
                df["pred_anomaly"] = (
                    df["anomaly_alert"] == "Anomaly"
                ).astype(int)

                for method, prediction_column in METHOD_PREDICTIONS.items():
                    ground_truth = df["ground_truth"]
                    predictions = df[prediction_column]
                    tn, fp, _, _ = confusion_matrix(
                        ground_truth,
                        predictions,
                        labels=[0, 1],
                    ).ravel()
                    false_alarm_denominator = tn + fp
                    results.append(
                        {
                            "Noise_Multiplier": noise_multiplier,
                            "Missing_Rate": missing_rate,
                            "Method": method,
                            "Precision": precision_score(
                                ground_truth,
                                predictions,
                                zero_division=0,
                            ),
                            "Recall": recall_score(
                                ground_truth,
                                predictions,
                                zero_division=0,
                            ),
                            "False_Alarm_Rate": (
                                fp / false_alarm_denominator
                                if false_alarm_denominator
                                else 0.0
                            ),
                        }
                    )

    comparison = pd.DataFrame(results, columns=RESULT_COLUMNS)
    output_path = Path("results/robustness.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return comparison


if __name__ == "__main__":
    robustness_metrics = evaluate_robustness()
    print("Detector robustness metrics:")
    print(
        robustness_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
