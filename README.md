# IoT-Based Distribution Transformer Thermal-Load Health Monitoring System

## Project scope

This project is a software-based IoT prototype that uses synthetic
distribution-transformer telemetry to compare fixed threshold alarms with a
rolling-window anomaly detector. It demonstrates a data-processing,
monitoring, and evaluation workflow; it is not a deployed or field-validated
transformer diagnostic system.

**Limitations:** The system does not diagnose internal transformer faults and
does not use live field measurements. All telemetry and event labels used in
the experiments are synthetic. Results should not be interpreted as evidence
of field performance or as operational protection settings.

## Telemetry and architecture

The synthetic dataset contains 30 days of one-minute readings, with
representative diurnal cycles and injected overload and voltage-sag events.
Measured and derived variables are:

- Load current (A)
- Voltage (V)
- Top-oil temperature (°C), stored as `oil_temperature_c`
- Ambient temperature (°C)
- Temperature rise (°C), derived as top-oil temperature minus ambient
  temperature

The main pipeline is:

```text
data_generator
    -> data_processing
    -> threshold_detector
    -> anomaly_detector
    -> health_index
    -> dashboard
```

The evaluation and publication workflow adds `evaluate` and `robustness` for
detector metrics, and `generate_figures` for publication-ready plots. The
interactive dashboard can use the generated local CSV or an uploaded CSV with
the required telemetry columns.

## Setup and reproducible run instructions

From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Generate synthetic telemetry and run the processing and monitoring modules:

```powershell
python -m src.data_generator
python -m src.threshold_detector
python -m src.anomaly_detector
python -m src.health_index
```

Run the experimental evaluation, robustness analysis, and figure generation:

```powershell
python -m src.evaluate
python -m src.robustness
python -m src.generate_figures
```

Launch the interactive Streamlit dashboard:

```powershell
streamlit run app/dashboard.py
```

The evaluation writes `results/metrics.csv`; robustness results are written to
`results/robustness.csv`; publication figures are saved under `results/figures/`.

## Experimental summary

### Detector comparison

The saved results in [`results/metrics.csv`](results/metrics.csv) compare
detector outputs with the synthetic labels for the injected overload and
voltage-sag windows:

| Method | Precision | Recall | F1 score | False-alarm rate | Missed-detection rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed Threshold | 0.9689 | 0.9951 | 0.9818 | 0.0003 | 0.0049 |
| Rolling Anomaly | 0.0204 | 0.0123 | 0.0153 | 0.0056 | 0.9877 |

For these synthetic labels, fixed thresholds achieved **0.9951 recall** on the
sustained multi-hour level shifts. The rolling-window MAD detector instead
flags transient deviations, such as abrupt transitions and onset edges; as a
sustained shift becomes part of its rolling baseline, it is less likely to
remain flagged. The low rolling-detector recall in this experiment reflects
that difference between transient-anomaly detection and persistent event
labeling. The two methods therefore capture different behaviors and should
not be treated as interchangeable.

### Noise and missing-data robustness

[`results/robustness.csv`](results/robustness.csv) reports both methods for all
combinations of noise multipliers `0.5`, `1.0`, `1.5`, and `2.0`, and missing
oil-temperature rates `0%`, `1%`, `5%`, and `10%`. Across these 16 settings,
the fixed-threshold detector's mean recall was approximately **0.992**; its
mean precision decreased from **1.000** at noise multiplier `0.5` to **0.327**
at multiplier `2.0`, while its mean false-alarm rate increased from **0.0000**
to **0.0193**. The rolling anomaly detector's recall remained low across these
settings (approximately **0.007–0.015**), consistent with its focus on local,
transient deviations rather than sustained event windows.

These results describe only this synthetic setup. Noise multipliers scale
simulated sensor noise, missing rates apply to oil-temperature readings, and
the experiment does not establish performance on real transformers.
