# LMD-SOC Sentinel

AI-powered lightweight early lateral movement detection dashboard for Windows Sysmon-style telemetry.

The system combines:

- MITRE ATT&CK-aligned rule detection
- Isolation Forest anomaly scoring
- A hybrid decision layer
- A local Flask SOC dashboard for CSV upload analysis and live event monitoring

## Run the Dashboard

On Windows, double-click:

```bat
RUN_WEBSITE.bat
```

Then open:

```text
http://localhost:5000
```

Manual setup:

```bash
pip install -r requirements.txt
python app.py
```

## Included Files

- `app.py` - Flask API server and dashboard backend
- `index.html` - frontend dashboard
- `src/` - import-compatible detection modules
- `outputs/` - trained model, threshold, and Mission 4 summary artifacts
- `isolation_forest_model.pkl` and `isolation_forest_threshold.pkl` - model artifacts also kept at root for flat-layout compatibility
- `family2_test_upload_unlabeled.csv`, `random_soc_test_unlabeled.csv`, `lmd-upload-test.json` - sample upload files
- `tools/Start-LocalSysmonCollector.ps1` - optional local live-event collector

## Main Pipeline Scripts

- `Mission1_run.py` - data acquisition and preprocessing
- `Mission2_run.py` - rule-based detection engine
- `Mission3_run.py` - Isolation Forest model training and evaluation
- `Mission4_run.py` - final hybrid detection pipeline

## Notes

- The dashboard binds to `127.0.0.1` by default, so it is available only on the local machine.
- Live monitoring keeps a small in-memory event buffer. Restarting the server clears live history.
- The large original LMD-2023 dataset is not included.
