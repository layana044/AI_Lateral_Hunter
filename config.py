"""
config.py — Central configuration for LM_System.
All paths, constants, and experiment parameters live here.
To switch between dev-mode and full-scale, edit SAMPLE_SIZES or
set DEV_MODE = True to run only the 100k slice quickly.
"""

import os

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────

# Absolute path to the raw LMD-2023 CSV (1.75 M records, 93 features)
RAW_DATA_PATH = r"C:\Users\USER\Desktop\LM_Project\LMD-2023 [1.75M Elements][Labelled]checked.csv"

# Project root (auto-resolved so the script works from any cwd)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")

# ─────────────────────────────────────────────
#  Dataset expectations (for integrity checks)
# ─────────────────────────────────────────────

EXPECTED_MIN_ROWS = 1_700_000   # allow slight tolerance around 1.75 M
EXPECTED_COLS     = 93          # raw feature count per the paper

# ─────────────────────────────────────────────
#  8 Core Features — Table 3, Smiliotopoulos 2025
#  Format:  paper_name  →  actual CSV column name
#  (data_loader prints the real header so you can fix any mismatch)
# ─────────────────────────────────────────────

FEATURE_MAP = {
    "CompSTA"     : "Computer",
    "DstPortName" : "DestinationPortName",
    "EventID"     : "EventID",
    "EventRecID"  : "EventRecordID",
    "ExecProcID"  : "Execution_ProcessID",
    "Init"        : "Initiated",
    "ProcessId"   : "ProcessId",
    "SrcIpv6"     : "SourceIsIpv6",
}

# Convenience list of CSV column names to keep
CORE_FEATURES = list(FEATURE_MAP.values())

# Label column name in the raw CSV
LABEL_COLUMN = "Label"

# ─────────────────────────────────────────────
#  Categorical vs Numeric split (for OHE / MinMax)
# ─────────────────────────────────────────────

CATEGORICAL_FEATURES = [
    "Computer",
    "DestinationPortName",
    "Initiated",
    "SourceIp",
]

NUMERIC_FEATURES = [
    "EventID",
    "EventRecordID",
    "Execution_ProcessID",
    "ProcessId",
]

# ─────────────────────────────────────────────
#  Incremental data scales
#  None → load all rows (full 1.75 M — paper-comparable)
# ─────────────────────────────────────────────

SAMPLE_SIZES = {
    "100k" : 100_000,
    "500k" : 500_000,
    "1M"   : 1_000_000,
    "full" : None,
}

# ─────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────

RANDOM_STATE = 42
