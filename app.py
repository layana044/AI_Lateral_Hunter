"""
dashboard/app.py â€” Flask API Server for the SOC Dashboard
==========================================================

Supports:
  - Serving the full 10-section SOC dashboard HTML
  - REST API for pre-computed metrics (fallback / Mission4 output)
  - CSV file upload endpoint that runs the hybrid detection pipeline live
    (Rule Engine + Isolation Forest) on any compatible Sysmon log CSV,
    then returns enriched per-row results and aggregate stats instantly.

Usage (run from the LM_System project root):
    python dashboard/app.py

Then open your browser at:
    http://localhost:5000
"""

import os
import sys
import io
import json
import uuid
import gc
import time
import traceback
import pickle
import threading
from datetime import datetime
from types import ModuleType

import numpy as np
import pandas as pd
from flask import Flask, send_file, jsonify, request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# â”€â”€ Path setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
APP_DIR = os.path.dirname(os.path.abspath(__file__))
NESTED_PROJECT_ROOT = os.path.dirname(APP_DIR)
FLAT_LAYOUT = os.path.exists(os.path.join(APP_DIR, "isolation_forest.py"))
PROJECT_ROOT = APP_DIR if FLAT_LAYOUT else NESTED_PROJECT_ROOT
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "mission4_hybrid_summary.json")
if not os.path.exists(SUMMARY_PATH):
    SUMMARY_PATH = os.path.join(PROJECT_ROOT, "mission4_hybrid_summary.json")
HTML_PATH    = os.path.join(APP_DIR, "index.html")
MODEL_PATH   = os.path.join(OUTPUTS_DIR, "isolation_forest_model.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "isolation_forest_model.pkl")
THRESH_PATH  = os.path.join(OUTPUTS_DIR, "isolation_forest_threshold.pkl")
if not os.path.exists(THRESH_PATH):
    THRESH_PATH = os.path.join(PROJECT_ROOT, "isolation_forest_threshold.pkl")

# Add the project root to sys.path so src.* imports work
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if FLAT_LAYOUT and "src" not in sys.modules:
    src_pkg = ModuleType("src")
    src_pkg.__path__ = [PROJECT_ROOT]
    sys.modules["src"] = src_pkg

import src.isolation_forest as isolation_forest_module
from src.isolation_forest import (
    encode_features,
    load_model,
    CATEGORICAL_OHE_FEATURES,
    NUMERIC_MINMAX_FEATURES,
)
from src.rule_engine import ALL_RULES, apply_rules

isolation_forest_module.MODEL_SAVE_PATH = MODEL_PATH
isolation_forest_module.THRESHOLD_SAVE_PATH = THRESH_PATH

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB cap per upload

# In-memory live monitoring store used by the dashboard. The collector posts
# events to /api/live-event; the frontend polls the read endpoints below.
live_events = []
live_alerts = []
live_events_seen = 0
MAX_LIVE_EVENTS = 500
MAX_LIVE_ALERTS = 100
live_state_lock = threading.RLock()

# â”€â”€ Load trained model once at startup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_model     = None
_threshold = None
_preprocessor = None
model_load_lock = threading.Lock()

HIGH_CONFIDENCE_EVENTIDS = frozenset({7, 10, 23, 17})
THRESHOLD_DELTA = 0.02
NOISY_RULE_NAMES = (
    "LDAP AD Reconnaissance",
    "RPC/DCOM/WMI Remote Execution",
    "SMB / Windows Admin Shares Lateral Movement",
)
RULE_MITRE_BY_NAME = {rule.name: rule.mitre_id for rule in ALL_RULES}


def _mitre_ids_for_matched_rules(matched_rules: str) -> str:
    if not matched_rules or matched_rules == "none":
        return ""
    ids = []
    for name in [part.strip() for part in matched_rules.split(",")]:
        mitre_id = RULE_MITRE_BY_NAME.get(name)
        if mitre_id and mitre_id not in ids:
            ids.append(mitre_id)
    return ", ".join(ids)

def _get_model():
    global _model, _threshold, _preprocessor
    if _model is None and os.path.exists(MODEL_PATH) and os.path.exists(THRESH_PATH):
        with model_load_lock:
            if _model is None:
                _model, _threshold = load_model()
                try:
                    with open(MODEL_PATH, "rb") as f:
                        pkg = pickle.load(f)
                    if isinstance(pkg, dict):
                        _preprocessor = pkg.get("preprocessor")
                except Exception:
                    _preprocessor = None
    return _model, _threshold


def _get_final_if_threshold(base_threshold):
    """Use the latest Mission4 operating threshold for live dashboard uploads."""
    try:
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("threshold_strict") is not None:
            return float(summary["threshold_strict"])
    except Exception:
        pass
    return float(base_threshold) - THRESHOLD_DELTA if base_threshold is not None else None


# â”€â”€ Detection rule masks (vectorized, no external import needed) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _apply_rules_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the MITRE ATT&CK detection rules using the canonical rule engine.
    Works on any DataFrame that has the expected Sysmon column names.
    Returns the DataFrame with added columns:
      rule_alert (bool), matched_rules (str), severity (str)
    """
    return apply_rules(df)


# â”€â”€ Exact feature encoding â€” matches isolation_forest.py encode_features() â”€â”€
#
# The model was trained with:
#   - LabelEncoder on DestinationPortName and Computer (NOT OHE)
#   - SourceIp classified into 4 types (0=none, 1=internal, 2=external, 3=loopback)
#   - Initiated â†’ encoded as 0/1/2
#   - 4 numeric cols: EventID, EventRecordID, Execution_ProcessID, ProcessId
#   - 5 engineered features: is_high_risk_event, event_rarity_score,
#                            same_process_ids, is_lm_port, is_internal_network_event
#   - NO MinMax scaling was applied to the training data
#
# This encoding is completely self-contained and works on ANY raw Sysmon CSV.

BENIGN_COUNT_PER_EVENTID = {
    1:  19530, 2:    234, 3: 1380345, 4:    439, 5:  172473,
    6:     24, 7:      2, 8:    330,  10:     1, 11:   9212,
    12:   661, 13: 17751, 15:   337,  16:   339, 17:     1,
    18:     1, 22:   9929, 23:     1,
}
TOTAL_BENIGN     = 1_611_619
HIGH_RISK_AIDS   = {7, 8, 10, 17, 18, 23}
LM_PORTS         = {"ldap","kerberos","epmap","ms-wbt-server","microsoft-ds","smb","rdp"}

EXPECTED_FEATURES = [
    "EventID", "EventRecordID", "Execution_ProcessID", "ProcessId",
    "DestinationPortName", "Initiated", "SourceIp_type", "Computer",
    "is_high_risk_event", "event_rarity_score", "same_process_ids",
    "is_lm_port", "is_internal_network_event",
]


def _encode_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the exact encode_features() logic from src/isolation_forest.py.
    Produces the same 13-column numeric matrix the model was trained on.
    Works on raw (unpreprocessed) Sysmon CSV files with no scaler needed.
    """
    enc = pd.DataFrame(index=df.index)

    # â”€â”€ 1. Numeric columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for col in ["EventID", "EventRecordID", "Execution_ProcessID", "ProcessId"]:
        enc[col] = pd.to_numeric(df.get(col, pd.Series(0, index=df.index)),
                                 errors="coerce").fillna(0)

    # â”€â”€ 2. DestinationPortName â€” label-encode to integer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    port_raw = df.get("DestinationPortName",
                      pd.Series("unknown", index=df.index)).astype(str).str.lower().str.strip()
    # Map each unique string to a stable integer
    port_cats = {p: i for i, p in enumerate(sorted(port_raw.unique()))}
    enc["DestinationPortName"] = port_raw.map(port_cats).fillna(0).astype(int)

    # â”€â”€ 3. Initiated â†’ 0 (inbound) / 1 (outbound) / 2 (local) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lower = df.get("Initiated",
                   pd.Series("0", index=df.index)).astype(str).str.lower().str.strip()
    init_enc = pd.Series(2, index=df.index, dtype=int)
    init_enc[lower.isin(["true", "1", "yes"])] = 1
    init_enc[lower == "false"]                 = 0
    enc["Initiated"] = init_enc

    # â”€â”€ 4. SourceIp â†’ classify into 4 IP types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    src = df.get("SourceIp", pd.Series("0", index=df.index)).astype(str).str.strip()
    ip_type = pd.Series(2, index=df.index, dtype=int)   # default: external
    ip_type[src == "0"]               = 0               # no IP (local event)
    ip_type[src == "127.0.0.1"]       = 3               # loopback
    internal_mask = (
        src.str.startswith("192.168.") | src.str.startswith("10.")
        | src.str.startswith("172.")   | src.str.startswith("fe80:")
        | src.str.startswith("0:0:0:0:0:0:0:1")
    )
    ip_type[internal_mask] = 1
    enc["SourceIp_type"] = ip_type

    # â”€â”€ 5. Computer â€” label-encode hostname â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    comp_raw = df.get("Computer", pd.Series("unknown", index=df.index)).astype(str).str.strip()
    comp_cats = {c: i for i, c in enumerate(sorted(comp_raw.unique()))}
    enc["Computer"] = comp_raw.map(comp_cats).fillna(0).astype(int)

    # â”€â”€ 6. Engineered: is_high_risk_event â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    enc["is_high_risk_event"] = enc["EventID"].isin(HIGH_RISK_AIDS).astype(int)

    # â”€â”€ 7. Engineered: event_rarity_score â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    benign_frac = enc["EventID"].map(
        {eid: cnt / TOTAL_BENIGN for eid, cnt in BENIGN_COUNT_PER_EVENTID.items()}
    ).fillna(1e-7)
    enc["event_rarity_score"] = -np.log10(benign_frac + 1e-7)

    # â”€â”€ 8. Engineered: same_process_ids â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    enc["same_process_ids"] = (
        (enc["Execution_ProcessID"] != 0)
        & (enc["ProcessId"] != 0)
        & (enc["Execution_ProcessID"] == enc["ProcessId"])
    ).astype(int)

    # â”€â”€ 9. Engineered: is_lm_port â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    enc["is_lm_port"] = port_raw.isin(LM_PORTS).astype(int)

    # â”€â”€ 10. Engineered: is_internal_network_event â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    enc["is_internal_network_event"] = (
        (enc["SourceIp_type"] == 1) & (enc["EventID"] == 3)
    ).astype(int)

    return enc.fillna(0).astype(np.float32)


def _apply_ml_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the trained Isolation Forest on uploaded data.
    Uses the exact same feature encoding as the training pipeline.
    Works on raw (unpreprocessed) Sysmon CSV â€” no scaler needed.
    Adds ml_score (float) and ml_alert (bool) columns.
    """
    try:
        model, threshold = _get_model()
    except Exception as e:
        print(f"  [ML] Model load failed: {e}")
        df["ml_score"] = 0.0
        df["ml_alert"] = False
        df["ml_alert_strict"] = False
        df.attrs["model_used"] = False
        return df

    if model is None:
        df["ml_score"] = 0.0
        df["ml_alert"] = False
        df["ml_alert_strict"] = False
        df.attrs["model_used"] = False
        return df

    try:
        for col in CATEGORICAL_OHE_FEATURES:
            if col not in df.columns:
                df[col] = "missing"
        for col in NUMERIC_MINMAX_FEATURES:
            if col not in df.columns:
                df[col] = 0

        if _preprocessor is not None:
            work = df[CATEGORICAL_OHE_FEATURES + NUMERIC_MINMAX_FEATURES].copy()
            for col in CATEGORICAL_OHE_FEATURES:
                work[col] = work[col].astype(str).fillna("missing")
            for col in NUMERIC_MINMAX_FEATURES:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
            feat = _preprocessor.transform(work)
        else:
            feat = encode_features(df)

        if isinstance(feat, pd.DataFrame):
            try:
                expected = list(model.feature_names_in_)
            except AttributeError:
                expected = EXPECTED_FEATURES
            for col in expected:
                if col not in feat.columns:
                    feat[col] = 0.0
            feat = feat[expected]
        scores = model.decision_function(feat)
        df["ml_score"] = scores

        if threshold is not None:
            base_threshold = float(threshold)
            strict_threshold = _get_final_if_threshold(base_threshold)
            df["ml_alert"] = scores < base_threshold
            df["ml_alert_strict"] = scores < strict_threshold
        else:
            fallback_threshold = np.percentile(scores, 10)
            df["ml_alert"] = scores < fallback_threshold
            df["ml_alert_strict"] = df["ml_alert"]
        df.attrs["model_used"] = True
    except Exception as e:
        print(f"  [ML] Scoring failed: {e}")
        df["ml_score"] = 0.0
        df["ml_alert"] = False
        df["ml_alert_strict"] = False
        df.attrs["model_used"] = False

    return df



# â”€â”€ Fallback data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FALLBACK_DATA = {
    "_fallback": True,
    "meta": {
        "title":             "LMD-2023 Hybrid Lateral Movement Detection System",
        "dataset":           "Smiliotopoulos et al. (2025): LMD-2023",
        "generated_at":      None,
        "total_records":     1752836,
        "malicious_records": 141217,
        "benign_records":    1611619,
        "attack_rate_pct":   8.06,
        "execution_time_s":  None,
    },
    "rules": {
        "TP": 103933, "FP": 2446, "TN": 1609173, "FN": 37284,
        "detection_rate_pct":      73.5981,
        "false_positive_rate_pct":  0.1518,
        "precision_pct":           97.7007,
        "f1_pct":                  83.9537,
        "accuracy_pct":            97.7334,
    },
    "ml": {
        "TP": 136159, "FP": 47092, "TN": 1564527, "FN": 5058,
        "detection_rate_pct":      96.4183,
        "false_positive_rate_pct":  2.9220,
        "precision_pct":           74.3019,
        "f1_pct":                  83.9275,
        "accuracy_pct":            97.0180,
    },
    "hybrid": {
        "TP": 136159, "FP": 47092, "TN": 1564527, "FN": 5058,
        "detection_rate_pct":      96.4183,
        "false_positive_rate_pct":  2.9220,
        "precision_pct":           74.3019,
        "f1_pct":                  83.9275,
        "accuracy_pct":            97.0180,
    },
    "fn_recovered_by_ml": 32226,
    "per_rule": [
        {"name": "Malicious DLL / Module Load",        "mitre_id": "T1574.001", "severity": "HIGH",
         "alerts_fired": 46869, "tp": 46867, "fp": 2,    "precision_pct": 100.0, "recall_pct": 33.2,
         "description": "Detects Sysmon Event 7 (ImageLoad). 99.99% of Event 7 records are attack events."},
        {"name": "LSASS / Cross-Process Handle Access","mitre_id": "T1003.001", "severity": "HIGH",
         "alerts_fired": 32176, "tp": 32175, "fp": 1,    "precision_pct": 100.0, "recall_pct": 22.8,
         "description": "Detects Sysmon Event 10 (ProcessAccess). Used by Mimikatz and credential dumping tools."},
        {"name": "File Deletion / Evidence Removal",   "mitre_id": "T1070.004", "severity": "HIGH",
         "alerts_fired": 22764, "tp": 22763, "fp": 1,    "precision_pct": 100.0, "recall_pct": 16.1,
         "description": "Detects Sysmon Event 23 (FileDelete). Attackers delete tools and logs post-exploitation."},
        {"name": "LDAP AD Reconnaissance",             "mitre_id": "T1018",     "severity": "MEDIUM",
         "alerts_fired": 3256,  "tp":  1560, "fp": 1696, "precision_pct":  47.9, "recall_pct":  1.1,
         "description": "Detects Event 3 to LDAP port from internal hosts. Indicates BloodHound/PowerView enumeration."},
        {"name": "RPC/DCOM/WMI Remote Execution",      "mitre_id": "T1021.003", "severity": "HIGH",
         "alerts_fired": 1245,  "tp":   505, "fp":  740, "precision_pct":  40.6, "recall_pct":  0.4,
         "description": "Detects Event 3 to RPC Endpoint Mapper (port 135). Used by PsExec and WMI lateral movement."},
        {"name": "RDP Lateral Movement",               "mitre_id": "T1021.001", "severity": "HIGH",
         "alerts_fired":   42,  "tp":    37, "fp":    5, "precision_pct":  88.1, "recall_pct":  0.0,
         "description": "Detects outbound RDP (port 3389) from internal hosts. Most direct lateral movement technique."},
        {"name": "Kerberos Ticket Abuse",              "mitre_id": "T1558",     "severity": "HIGH",
         "alerts_fired":   12,  "tp":    12, "fp":    0, "precision_pct": 100.0, "recall_pct":  0.0,
         "description": "Detects Kerberos port (88) activity. Indicates Pass-the-Ticket or Kerberoasting attacks."},
        {"name": "Named Pipe Creation (C2 Channel)",   "mitre_id": "T1559.001", "severity": "HIGH",
         "alerts_fired":   15,  "tp":    14, "fp":    1, "precision_pct":  93.3, "recall_pct":  0.0,
         "description": "Detects Sysmon Event 17 (Pipe Created). Used by Cobalt Strike SMB Beacon as C2 channel."},
    ],
}


def load_summary() -> dict:
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return FALLBACK_DATA


# â”€â”€ Routes â€” static / pre-computed data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/")
def index():
    return send_file(HTML_PATH)


@app.route("/api/metrics")
def api_metrics():
    data = load_summary()
    return jsonify({
        "meta":               data.get("meta", {}),
        "rules":              data.get("rules", {}),
        "ml":                 data.get("ml", {}),
        "hybrid":             data.get("hybrid", {}),
        "fn_recovered_by_ml": data.get("fn_recovered_by_ml", 0),
        "is_fallback":        data.get("_fallback", False),
    })


@app.route("/api/rules")
def api_rules():
    data = load_summary()
    return jsonify(data.get("per_rule", []))


@app.route("/api/status")
def api_status():
    summary_exists = os.path.exists(SUMMARY_PATH)
    model_exists   = os.path.exists(MODEL_PATH)
    data           = load_summary()
    return jsonify({
        "mission4_complete": summary_exists,
        "model_trained":     model_exists,
        "using_fallback":    data.get("_fallback", False),
        "generated_at":      data.get("meta", {}).get("generated_at"),
        "server_time":       datetime.now().isoformat(),
    })


# â”€â”€ Upload endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Accept one or more CSV files, run the hybrid detection pipeline,
    and return aggregated results + per-row alert data.

    Expects multipart/form-data with field name 'files[]' (or 'file').
    Returns JSON:
      {
        "success": true,
        "files": [...],          # per-file summaries
        "combined": {...},       # aggregate stats across all files
        "alerts": [...],         # top-N alert rows (capped for response size)
        "ioc_summary": {...},
        "timeline": [...],
        "per_rule": [...],
      }
    """
    t0 = time.time()

    # Accept either 'files[]' (batch) or 'file' (single)
    uploaded = request.files.getlist("files[]") or request.files.getlist("file")
    if not uploaded:
        return jsonify({"success": False, "error": "No files received."}), 400

    all_dfs      = []
    file_summaries = []
    parse_errors   = []

    for f in uploaded:
        if not f or not f.filename:
            continue
        filename = f.filename
        try:
            raw_bytes = f.read()
            # Try comma first, then semicolon/tab
            for sep in [",", ";", "\t"]:
                try:
                    chunk = pd.read_csv(
                        io.BytesIO(raw_bytes), sep=sep,
                        low_memory=False, nrows=500_000  # safety cap per file
                    )
                    if len(chunk.columns) > 3:
                        break
                except Exception:
                    continue
            else:
                raise ValueError("Cannot parse CSV â€” tried , ; and tab separators.")

            n_rows = len(chunk)
            file_summaries.append({"filename": filename, "rows": n_rows, "status": "ok"})
            all_dfs.append(chunk)
        except Exception as e:
            parse_errors.append({"filename": filename, "error": str(e)})

    if not all_dfs:
        return jsonify({
            "success": False,
            "error": "No valid CSV data found.",
            "parse_errors": parse_errors,
        }), 422

    # Merge all uploaded files
    df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    n_total = len(df)

    # â”€â”€ Run rule engine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df = _apply_rules_fast(df)

    # â”€â”€ Run ML model (if available) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df = _apply_ml_fast(df)
    model_available = bool(df.attrs.get("model_used", False))

    # â”€â”€ Hybrid decision: same precision-preserving fusion as Mission 4 â”€â”€â”€
    event_ids = pd.to_numeric(df.get("EventID", pd.Series(0, index=df.index)),
                              errors="coerce").fillna(0).astype(int)
    rule_alert = df["rule_alert"].astype(bool)
    ml_strict = df.get("ml_alert_strict", df["ml_alert"]).astype(bool)
    matched_rules = df.get("matched_rules", pd.Series("", index=df.index)).fillna("")
    noisy_rule_alert = pd.Series(False, index=df.index)
    for rule_name in NOISY_RULE_NAMES:
        noisy_rule_alert |= matched_rules.str.contains(rule_name, regex=False)
    trusted_rule_alert = rule_alert & ~noisy_rule_alert
    rule_blind_spot = ~rule_alert & ~event_ids.isin(HIGH_CONFIDENCE_EVENTIDS)
    ml_candidate_zone = rule_blind_spot | noisy_rule_alert
    df["hybrid_alert"] = trusted_rule_alert | (ml_strict & ml_candidate_zone)

    # â”€â”€ Compute metrics (with label if present) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    has_label      = "Label" in df.columns
    label_col      = None
    if has_label:
        # Normalise label: 1/True/"Attack"/"1"/"attack" â†’ True
        raw_label = df["Label"].astype(str).str.lower().str.strip()
        label_col = (
            raw_label.isin(["1", "attack", "true", "malicious", "malware", "yes"])
        ).astype(bool)

    def _metrics(pred: pd.Series, truth=label_col) -> dict:
        if truth is None:
            return {"has_ground_truth": False}
        tp = int((pred  & truth).sum())
        fp = int((pred  & ~truth).sum())
        tn = int((~pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        total_att = tp + fn
        total_ben = fp + tn
        dr  = tp / total_att * 100 if total_att > 0 else 0
        fpr = fp / total_ben * 100 if total_ben > 0 else 0
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        f1   = (2 * prec * dr) / (prec + dr) if (prec + dr) > 0 else 0
        acc  = (tp + tn) / (tp + fp + tn + fn) * 100
        return {
            "has_ground_truth": True,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "detection_rate_pct":      round(dr,  4),
            "false_positive_rate_pct": round(fpr, 4),
            "precision_pct":           round(prec,4),
            "f1_pct":                  round(f1,  4),
            "accuracy_pct":            round(acc, 4),
            "total_malicious":         total_att,
            "total_benign":            total_ben,
        }

    rule_metrics   = _metrics(df["rule_alert"])
    ml_metrics     = _metrics(df["ml_alert"])
    hybrid_metrics = _metrics(df["hybrid_alert"])

    # alert counts (when no ground truth)
    n_rule_alerts   = int(df["rule_alert"].sum())
    n_ml_alerts     = int(df["ml_alert"].sum())
    n_hybrid_alerts = int(df["hybrid_alert"].sum())
    n_rule_only_final = int((df["hybrid_alert"] & df["rule_alert"] & ~df["ml_alert_strict"]).sum())
    n_ml_only_final   = int((df["hybrid_alert"] & ~df["rule_alert"] & df["ml_alert_strict"]).sum())
    n_both_final      = int((df["hybrid_alert"] & df["rule_alert"] & df["ml_alert_strict"]).sum())

    # â”€â”€ Per-rule breakdown â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    RULE_DEF = [
        ("Malicious DLL / Module Load",        "T1574.001", "HIGH"),
        ("LSASS / Cross-Process Handle Access","T1003.001", "HIGH"),
        ("File Deletion / Evidence Removal",   "T1070.004", "HIGH"),
        ("LDAP AD Reconnaissance",             "T1018",     "MEDIUM"),
        ("RPC/DCOM/WMI Remote Execution",      "T1021.003", "HIGH"),
        ("RDP Lateral Movement",               "T1021.001", "HIGH"),
        ("Kerberos Ticket Abuse",              "T1558",     "HIGH"),
        ("Named Pipe Creation (C2 Channel)",   "T1559.001", "HIGH"),
    ]
    per_rule_results = []
    for (name, mid, sev) in RULE_DEF:
        fired = df["matched_rules"].str.contains(name, regex=False, na=False)
        alerts_fired = int(fired.sum())
        if has_label and label_col is not None:
            tp = int((fired & label_col).sum())
            fp = int((fired & ~label_col).sum())
        else:
            tp, fp = alerts_fired, 0
        prec_  = tp / alerts_fired * 100 if alerts_fired > 0 else 0
        per_rule_results.append({
            "name": name, "mitre_id": mid, "severity": sev,
            "alerts_fired": alerts_fired,
            "tp": tp, "fp": fp,
            "precision_pct": round(prec_, 2),
            "recall_pct": 0,
        })

    # â”€â”€ IOC summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ioc_summary = {}
    if "SourceIp" in df.columns:
        alerted = df[df["hybrid_alert"]]
        top_ips = alerted["SourceIp"].value_counts().head(20).to_dict()
        ioc_summary["top_source_ips"] = [{"ip": k, "count": v} for k, v in top_ips.items()]
    else:
        ioc_summary["top_source_ips"] = []

    if "DestinationPortName" in df.columns:
        top_ports = df[df["hybrid_alert"]]["DestinationPortName"].value_counts().head(10).to_dict()
        ioc_summary["top_ports"] = [{"port": k, "count": v} for k, v in top_ports.items()]
    else:
        ioc_summary["top_ports"] = []

    if "Computer" in df.columns:
        top_hosts = df[df["hybrid_alert"]]["Computer"].value_counts().head(10).to_dict()
        ioc_summary["top_hosts"] = [{"host": k, "count": v} for k, v in top_hosts.items()]
    else:
        ioc_summary["top_hosts"] = []

    # â”€â”€ Timeline: event counts by EventID â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    timeline = []
    if "EventID" in df.columns:
        eid_counts = df["EventID"].value_counts().head(20).to_dict()
        timeline = [{"event_id": k, "total": v,
                     "alerted": int(df[(df["EventID"] == k) & df["hybrid_alert"]].shape[0])}
                    for k, v in eid_counts.items()]

    # â”€â”€ Alert rows (top 200 for dashboard table) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    alert_df = df[df["hybrid_alert"]].copy()
    # select presentable columns
    keep_cols = [c for c in [
        "EventID", "Computer", "SourceIp", "DestinationPortName",
        "Initiated", "ProcessId", "Label",
        "rule_alert", "ml_alert", "ml_alert_strict", "hybrid_alert",
        "matched_rules", "severity", "ml_score",
    ] if c in alert_df.columns]
    alert_rows = alert_df[keep_cols].head(200).replace({np.nan: None}).to_dict(orient="records")

    # â”€â”€ Lateral movement graph (sourceâ†’dest pairs) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lm_edges = []
    if "SourceIp" in df.columns and "Computer" in df.columns:
        edge_counts = (
            df[df["hybrid_alert"]]
            .groupby(["SourceIp", "Computer"])
            .size()
            .reset_index(name="weight")
            .sort_values("weight", ascending=False)
            .head(50)
        )
        lm_edges = edge_counts.to_dict(orient="records")

    exec_time = round(time.time() - t0, 2)

    combined = {
        "total_records":    n_total,
        "rule_alerts":      n_rule_alerts,
        "ml_alerts":        n_ml_alerts,
        "hybrid_alerts":    n_hybrid_alerts,
        "rule_only_final":  n_rule_only_final,
        "ml_only_final":    n_ml_only_final,
        "both_final":       n_both_final,
        "model_used":       model_available,
        "has_ground_truth": has_label,
        "rule_metrics":     rule_metrics,
        "ml_metrics":       ml_metrics,
        "hybrid_metrics":   hybrid_metrics,
        "execution_time_s": exec_time,
        "processed_at":     datetime.now().isoformat(),
    }

    return jsonify({
        "success":     True,
        "files":       file_summaries,
        "parse_errors":parse_errors,
        "combined":    combined,
        "alerts":      alert_rows,
        "ioc_summary": ioc_summary,
        "timeline":    timeline,
        "per_rule":    per_rule_results,
        "lm_edges":    lm_edges,
    })


# â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/api/live-event", methods=["POST"])
def api_live_event():
    """
    Receive one Sysmon event from a collector, apply the existing rule engine,
    and keep a small in-memory buffer for the dashboard's live monitoring tab.
    """
    global live_events_seen

    event = request.get_json(silent=True)
    if not isinstance(event, dict) or not event:
        return jsonify({"success": False, "error": "No event received."}), 400

    try:
        live_event = {
            "EventID": event.get("EventID", event.get("event_id", 0)),
            "Computer": event.get("Computer", event.get("computer", "unknown")),
            "User": event.get("User", event.get("user", "")),
            "SourceIp": event.get("SourceIp", event.get("source_ip", "0")),
            "DestinationIp": event.get("DestinationIp", event.get("destination_ip", "")),
            "DestinationPortName": event.get("DestinationPortName", event.get("destination_port", event.get("DestinationPort", "0"))),
            "DestinationPort": event.get("DestinationPort", event.get("destination_port", event.get("DestinationPortName", "0"))),
            "Initiated": event.get("Initiated", event.get("initiated", False)),
            "ProcessId": event.get("ProcessId", event.get("process_id", 0)),
            "EventRecordID": event.get("EventRecordID", event.get("record_id", 0)),
        }
        live_event.update(event)

        # Evaluate the new event with recent live context. Several real rules
        # are frequency-based, so a single isolated event cannot trigger them.
        with live_state_lock:
            context_rows = list(reversed(live_events[:MAX_LIVE_EVENTS])) + [live_event]
        df = _apply_rules_fast(pd.DataFrame(context_rows))
        row = df.iloc[-1]
        rule_alert = bool(row.get("rule_alert", False))
        ml_alert = bool(row.get("ml_alert", False))
        hybrid_alert = bool(row.get("hybrid_alert", rule_alert or ml_alert))
        matched_rules = str(row.get("matched_rules", "none") or "none")
        mitre_id = _mitre_ids_for_matched_rules(matched_rules)
        severity = str(row.get("severity", "INFO") or "INFO").upper()
        received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def clean(value):
            if isinstance(value, np.generic):
                return value.item()
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            return value

        result = {
            "time": str(event.get("UtcTime") or event.get("TimeCreated") or event.get("time") or received_at),
            "received_at": received_at,
            "severity": severity if (rule_alert or hybrid_alert) else "INFO",
            "EventRecordID": clean(row.get("EventRecordID", event.get("EventRecordID"))),
            "EventID": clean(row.get("EventID", event.get("EventID"))),
            "event_id": clean(row.get("EventID", event.get("EventID"))),
            "Computer": clean(row.get("Computer", event.get("Computer"))),
            "computer": clean(row.get("Computer", event.get("Computer"))),
            "User": clean(row.get("User", event.get("User"))),
            "SourceIp": clean(row.get("SourceIp", event.get("SourceIp"))),
            "source_ip": clean(row.get("SourceIp", event.get("SourceIp"))),
            "DestinationIp": clean(row.get("DestinationIp", event.get("DestinationIp"))),
            "destination_ip": clean(row.get("DestinationIp", event.get("DestinationIp"))),
            "DestinationPortName": clean(row.get("DestinationPortName", event.get("DestinationPortName"))),
            "DestinationPort": clean(row.get("DestinationPort", event.get("DestinationPort"))),
            "destination_port": clean(row.get("DestinationPortName", event.get("DestinationPortName", event.get("DestinationPort")))),
            "Initiated": clean(row.get("Initiated", event.get("Initiated"))),
            "ProcessId": clean(row.get("ProcessId", event.get("ProcessId"))),
            "Image": clean(row.get("Image", event.get("Image"))),
            "CommandLine": clean(row.get("CommandLine", event.get("CommandLine"))),
            "rule_alert": rule_alert,
            "ml_alert": ml_alert,
            "hybrid_alert": hybrid_alert,
            "matched_rules": matched_rules,
            "mitre_id": mitre_id or clean(row.get("mitre_id", event.get("mitre_id"))),
            "ml_score": clean(row.get("ml_score", event.get("ml_score"))),
            "reason": str(row.get("reason", "")) or ("Matched rule: " + matched_rules if matched_rules != "none" else ""),
        }

        with live_state_lock:
            live_events_seen += 1
            live_events.insert(0, result)
            del live_events[MAX_LIVE_EVENTS:]

            if hybrid_alert or rule_alert or ml_alert:
                live_alerts.insert(0, result)
                del live_alerts[MAX_LIVE_ALERTS:]

        return jsonify({"success": True, **result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/live-events", methods=["GET"])
def api_live_events():
    with live_state_lock:
        events = list(live_events)
        total_seen = live_events_seen
    return jsonify({
        "success": True,
        "count": len(events),
        "total_live_events_seen": total_seen,
        "events": events,
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/live-alerts", methods=["GET"])
def api_live_alerts():
    with live_state_lock:
        alerts = list(live_alerts)
        total_seen = live_events_seen
    return jsonify({
        "success": True,
        "count": len(alerts),
        "total_live_events_seen": total_seen,
        "alerts": alerts,
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/live-alerts/clear", methods=["POST"])
def api_clear_live_alerts():
    global live_events_seen
    with live_state_lock:
        live_events.clear()
        live_alerts.clear()
        live_events_seen = 0
    return jsonify({"success": True, "message": "Live data cleared.", "count": 0})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LMD-SOC SENTINEL - Dashboard Server")
    print("=" * 60)
    print(f"  Mission 4 data : {'FOUND [OK]' if os.path.exists(SUMMARY_PATH) else 'NOT FOUND  (using fallback data)'}")
    print(f"  ML Model       : {'LOADED [OK]' if os.path.exists(MODEL_PATH) else 'NOT FOUND  (rule-only mode)'}")
    print(f"  Dashboard URL  : http://localhost:5000")
    print("  Network bind   : 127.0.0.1 (local machine only)")
    print("=" * 60 + "\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
