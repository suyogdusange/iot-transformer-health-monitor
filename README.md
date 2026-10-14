# IoT-Based Transformer Health Monitoring System

A software-based IoT prototype for distribution transformer thermal-load
monitoring using simulated telemetry.

The system monitors:
- Load current
- Voltage
- Oil temperature
- Ambient temperature
- Temperature rise

It compares fixed threshold alarms with a rolling-window anomaly detector.

## Important limitation

This project uses synthetic transformer data. It does not diagnose internal
transformer faults and is not a field-validated diagnostic system.

## Tech stack

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Streamlit

## How to run

```bash
pip install -r requirements.txt
python src/data_generator.py
streamlit run app/dashboard.py
```
