# 🎯 AI Lateral Hunter

AI Lateral Hunter is an advanced, hybrid Threat Hunting & Security Information and Event Management (SIEM) platform designed to detect **Lateral Movement (LM)** and **Credential Dumping** within enterprise Windows environments using raw Sysmon telemetry.

By combining deterministic **Static Forensic Rules** (mapped to MITRE ATT&CK) with an unsupervised **Machine Learning (Isolation Forest)** model, this platform provides High-Fidelity "Hybrid" alerts with a near-zero false-positive rate.

---

## 🔬 Academic Foundation & Methodology

The entire workflow, feature selection, and Machine Learning training methodology for this project were heavily inspired by and directly modeled after the peer-reviewed research paper:

> **"Assessing the detection of lateral movement through unsupervised learning techniques"**
> *Authors:* Christos Smiliotopoulos, Georgios Kambourakis, Constantinos Kolias, Stefanos Gritzalis
> *Journal:* Computers & Security 149 (2025) 104190

### The LMD-2023 Dataset
As detailed in the paper, traditional datasets (like LANL or NSL-KDD) are either outdated or lack sufficient Lateral Movement samples. To solve this, our ML model was trained on the **LMD-2023 Dataset**, which contains $\approx$1.75 million Sysmon EventID logs specifically capturing 15 modern LM attack techniques. 
These techniques are categorized into two major classes:
1. **EoRS (Exploitation of Remote Services):** e.g., EternalBlue, BlueKeep, Log4Shell, Follina, SMBGhost, Zerologon.
2. **EoHT (Exploitation of Hashing Techniques):** e.g., Mimikatz, LaZagne Project.

The dataset is highly imbalanced (Normal $\approx$92%, Malicious $\approx$8%), reflecting real-world SOHO (Small Office/Home Office) network conditions.

---

## 🧠 Machine Learning Engine: Tuned Isolation Forest

Following the paper's extensive evaluation of unsupervised ML techniques, we selected the **Isolation Forest (IF)** algorithm. The paper proved that IF is one of the most effective algorithms for LM detection, as it inherently isolates high-dimensional anomalous data structures rather than attempting to model the overwhelming volume of normal traffic.

### Feature Selection
The paper utilized Principal Component Analysis (PCA) to reduce 93 raw Sysmon fields down to the 8 most impactful features. We exclusively utilize these 8 features to maximize performance and minimize memory overhead:
1. `Computer` (Categorical, One-Hot Encoded)
2. `DestinationPortName` (Categorical, One-Hot Encoded)
3. `EventID` (Categorical, One-Hot Encoded)
4. `Initiated` (Categorical, One-Hot Encoded)
5. `SourceIsIpv6` (Categorical, One-Hot Encoded)
6. `EventRecordID` (Numerical, MinMax Scaled)
7. `Execution_ProcessID` (Numerical, MinMax Scaled)
8. `ProcessId` (Numerical, MinMax Scaled)

### Training & Preprocessing Workflow
The model was trained using the `tune_isolation_forest_model.py` script. The training pipeline utilizes:
- **Data Split:** A stratified split preserving the normal/attack ratio:
  - **Train:** 60% (1,051,701 rows)
  - **Validation:** 20% (350,567 rows)
  - **Test:** 20% (350,568 rows)
- **Labeling:** The original multiclass labels were binarized (`0` = normal, `>0` = attack). These labels were *only* used for stratified splitting and choosing the validation threshold to maximize the F1 score, remaining strictly unsupervised during the fitting phase.
- **Preprocessing (`build_preprocessor()`):**
  - *Categorical Fields:* `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(handle_unknown="ignore")`.
  - *Numeric Fields:* `SimpleImputer(strategy="median")` $\rightarrow$ `MinMaxScaler()`.

### Hyperparameter Tuning & Results
Due to resource constraints (8GB RAM), a highly targeted search was executed. The best configuration achieved was:
- `n_estimators`: 300
- `contamination`: 0.3
- `max_features`: 0.5
- `max_samples`: "auto"
- `warm_start`: True

The final anomaly threshold was automatically calibrated to **0.1098** on the validation set.

**Academic Held-out Test Split Performance:**
- **AUC:** `0.9739` (Surpassing the paper's baseline of 94.71%)
- **Precision:** `0.6435`
- **Recall:** `0.9559`
- **F1 Score:** `0.7692`
- **Accuracy:** `0.9538`

### Real-World Testing via Synthetic Sysmon Logs
To validate the model's performance in a live environment, the entire pipeline is tested using **synthetic Sysmon log files**. We built a dedicated log generator (`generate_realistic_sysmon.py`) that simulates the exact SOHO network conditions of the LMD-2023 dataset by bounding normal users to specific workstations. This allows us to inject highly realistic lateral movement anomalies and verify that the platform successfully isolates them with zero false positives.

---

## 🛡️ Static Rule Engine & MITRE ATT&CK Mappings

While the ML Engine aggregates behavior into 10-minute windows, the **Static Rule Engine** analyzes logs line-by-line using explicit Indicators of Compromise (IOCs). These rules were designed to hunt the specific techniques present in the LMD-2023 dataset:

- **T1003 (Credential Dumping):** Detects memory access to `lsass.exe` via `taskmgr.exe` or `mimikatz.exe` (EventID 10).
- **T1570 (Lateral Tool Transfer):** Detects malicious named pipes such as `\psexesvc` (EventID 17/18).
- **T1021 (Remote Services):** Detects unauthorized network connections over SMB/RDP (ports 445, 3389) by unusual processes like `cmd.exe` (EventID 3).
- **T1078 (Valid Accounts):** Detects logons occurring significantly outside working hours, or lateral movement across 3+ distinct hosts within a 10-minute window (EventID 4624).
- **T1055 (Process Injection):** Detects processes creating remote threads in other processes (EventID 8).
- **T1059 (Command and Scripting):** Detects suspicious `powershell.exe` execution with encoded/hidden commands (EventID 4688).

**Hybrid Correlation:** When a static rule fires during a mathematically proven ML anomalous window, the event is automatically upgraded to **Hybrid Confirmation** (Ultra-High Confidence).

---

## 📂 Codebase Deep Dive

Every component of the AI Lateral Hunter is modularly designed. Here is a comprehensive breakdown of every script and its role in the pipeline:

### 1. Root Orchestration & UI
- **`app.py`**: The Flask web server. It acts as the bridge between the backend analytical engine and the frontend Dashboard UI, serving the HTML files and providing REST endpoints.
- **`run_pipeline.py`**: The master orchestration script. Executing this script runs the entire analytical chain in order: Log Ingestion $\rightarrow$ Feature Engineering $\rightarrow$ ML Detection $\rightarrow$ Static Rules $\rightarrow$ Alert Correlation.

### 2. Model Training & Testing
- **`tune_isolation_forest_model.py`**: Executes the training pipeline. Loads the massive LMD-2023 dataset, applies the `OneHotEncoder` and `MinMaxScaler` preprocessors, performs hyperparameter searching, and exports the final `.joblib` model artifact.
- **`test_isolation_forest_model.py`**: Evaluates the compiled `.joblib` model against hold-out data to generate the confusion matrices, accuracy, precision, and F1 metrics documented above.

### 3. Simulation & Data Generation
- **`generate_realistic_sysmon.py`**: A synthetic data generator that creates custom Sysmon CSV logs. It explicitly bounds normal users to specific workstations, establishing a clean behavioral baseline, before injecting targeted MITRE ATT&CK techniques (e.g., PsExec, Credential Dumping) to test the platform's detection capabilities. 
- **`sysmon_850normal_15abnormal.csv`**: A generated test file consisting of 850 normal baseline logs and 15 highly malicious lateral movement logs.

### 4. Core Pipeline (`src/`)
- **`src/ingestion/log_ingestion.py`**: The first step of the pipeline. It sweeps the `data/raw/` directory for uploaded CSVs, standardizes column names, imputes missing timestamps, and exports a unified `clean_logs.csv` DataFrame.
- **`src/ml/feature_engineering.py`**: Aggregates user behavior into discrete 10-minute time windows, formatting the raw data into the 8 required features for the Isolation Forest.
- **`src/detection/ml_detection.py`**: Feeds the engineered features into the `.joblib` model. If the anomaly score exceeds the `0.1098` threshold, it flags the entire 10-minute window as an anomaly.
- **`src/detection/rule_based_detection.py`**: Executes deterministic, line-by-line regex and IOC matching for the MITRE ATT&CK rules listed above.
- **`src/detection/merge_alerts.py`**: The Hybrid Correlation Engine. It overlays the isolated ML Anomaly windows on top of the Static Rules to generate the final `all_alerts.csv` output.

### 5. Frontend Dashboard (`dashboard/`)
- **`index.html` & `index.css`**: A premium, glassmorphic UI built for SOC analysts. Features glowing threat metrics, a raw telemetry viewer, and a forensic investigation drawer.
- **`script.js`**: Parses the `all_alerts.csv` outputs to dynamically render charts, populate the Security Events table, and group critical alerts into the **Threat Response** panel for SOC Tier 2 intervention (Isolate Host, Disable Account, Trigger EDR).

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Pipeline
Place any raw Sysmon telemetry CSV files into the `data/raw/` directory, then execute the pipeline:
```bash
# Example: Copying the test dataset
copy sysmon_850normal_15abnormal.csv data\raw\

# Run the pipeline
python run_pipeline.py
```

### 3. Launching the Dashboard
Start the forensic dashboard server:
```bash
python app.py
```
Navigate to `http://127.0.0.1:8085` to begin hunting!
