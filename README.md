# 🎯 AI Lateral Hunter

AI Lateral Hunter is an advanced, hybrid Threat Hunting & Security Information and Event Management (SIEM) platform designed to detect **Lateral Movement** and **Credential Dumping** within enterprise Windows environments using raw Sysmon telemetry.

By combining deterministic **Static Forensic Rules** (mapped to MITRE ATT&CK) with an unsupervised **Machine Learning (Isolation Forest)** model, this platform provides High-Fidelity "Hybrid" alerts with an near-zero false-positive rate.

---

## ✨ Key Features

### 1. Hybrid Detection Engine
The backend pipeline fuses two parallel analytical engines:
- **Rule-Based Engine:** Scans raw logs line-by-line using precise IOCs to detect Credential Dumping (LSASS access), Suspicious Network Connections, Named Pipe usage (e.g., PsExec), and Remote Process Creation.
- **Machine Learning Engine:** An unsupervised Isolation Forest model dynamically baselines user behavior in 10-minute windows. It flags structural and volumetric anomalies (e.g., logging into 3+ distinct hosts simultaneously).
- **Hybrid Correlation:** When a static rule fires during a mathematically proven ML anomalous window, the event is automatically upgraded to **Hybrid Confirmation** (Ultra-High Confidence).

### 2. Premium Forensic Dashboard
A gorgeous, glassmorphic UI built for SOC analysts, featuring:
- **Real-Time Metrics:** Counters explicitly tracking Rule-based, ML Anomalies, Hybrid Confirmed, and Actionable Events.
- **Interactive Security Events Table:** A clean grid displaying compromised users, targeted hosts, malicious processes, and source/destination IP telemetry.
- **Forensic Inspection Drawer:** Clicking any alert opens a right-side drawer containing detailed MITRE ATT&CK mappings, detection reasoning, and severity justifications.
- **Raw Telemetry Viewer:** A dedicated tab to view all ingested Sysmon events before they are processed by the analytical engine.

### 3. Active Threat Response (Tier 2)
A dedicated workspace for SOC Tier 2 analysts and Threat Hunters:
- Automatically groups critical and high-severity alerts by the compromised entity.
- Provides immediate intervention actions (UI placeholders) for containment: **Isolate Host**, **Disable Account**, and **Trigger EDR Scan**.

---

## 📂 Project Architecture

```text
AI-Lateral-Hunter/
├── app.py                           # Flask backend serving the dashboard API
├── run_pipeline.py                  # Orchestrates the end-to-end analytical pipeline
├── generate_realistic_sysmon.py     # Generates customized, highly realistic test datasets
├── sysmon_850normal_15abnormal.csv  # Pre-generated test dataset (850 normal, 15 malicious logs)
├── requirements.txt                 # Python dependencies
├── dashboard/                       # Frontend UI (HTML, CSS, JS)
│   ├── index.html                   # Glassmorphic UI layout
│   ├── index.css                    # Styling and animations
│   └── script.js                    # UI logic, chart rendering, and data loading
├── src/                             # Core Python Backend
│   ├── ingestion/                   # Normalizes and cleans raw Sysmon CSVs
│   ├── ml/                          # Feature Engineering and baseline aggregation
│   └── detection/                   # Rule Engine, ML Training, and Alert Merging
└── data/                            # Local storage for the pipeline
    ├── raw/                         # Drop raw Sysmon CSVs here
    ├── processed/                   # Cleaned and normalized telemetry
    ├── features/                    # Fused feature tables for the ML model
    └── alerts/                      # Final output of the detection engines
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Pipeline
You can run the end-to-end analytical pipeline on the provided test dataset (`sysmon_850normal_15abnormal.csv`):
```bash
# First, copy the test file into the data/raw folder
copy sysmon_850normal_15abnormal.csv data\raw\

# Run the analytical engines
python run_pipeline.py
```
This will ingest the logs, engineer features, train the dynamic model, execute the static rules, and merge the final alerts.

### 3. Launching the Dashboard
To view the results in the forensic dashboard, start the Flask server:
```bash
python app.py
```
Navigate to `http://127.0.0.1:8085` in your web browser.

---

## 🛡️ Detected MITRE ATT&CK Techniques
- **T1078** - Valid Accounts
- **T1021** - Remote Services
- **T1003** - OS Credential Dumping (LSASS Memory)
- **T1570** - Lateral Tool Transfer (Named Pipes)
- **T1059** - Command and Scripting Interpreter
- **T1055** - Process Injection
- **T1047** - Windows Management Instrumentation

---

## 🛠️ Custom Dataset Generation
If you want to test the ML engine against different volumes of data, you can generate a new synthetic Sysmon file using the built-in generator:
```bash
python generate_realistic_sysmon.py
```
This script ensures that normal users obey realistic workstation baselines, allowing the ML engine to accurately identify genuine lateral movement anomalies without generating false positives.
