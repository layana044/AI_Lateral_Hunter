"""
Mission2_run.py — Entry Point for Mission 2: Rule-Based Detection Engine
=========================================================================

Run from the LM_System project root:
    python Mission2_run.py

What it does (3 steps):
  Step 1 — Load raw data    : Read the 8 core features + label from the raw
                              LMD-2023 CSV. Loading only 9 columns (not all 93)
                              keeps RAM usage under 200 MB regardless of row count.

  Step 2 — Apply rules      : The rule engine evaluates 7 MITRE ATT&CK-aligned
                              detection rules against every record. Each rule
                              targets a specific Sysmon event pattern that
                              corresponds to a known lateral movement technique.

  Step 3 — Evaluate + Save  : Compare rule predictions against the ground-truth
                              Label column. Compute DR, FPR, Precision, F1.
                              Print a detailed report. Save results for Mission 3.

Output files produced:
  outputs/mission2_predictions.csv    — Per-record predictions (for Mission 3/4)
  outputs/mission2_eval_report.txt    — Plain-text evaluation report (for thesis)

Why load raw data and NOT the preprocessed CSV?
  The preprocessed CSV (Mission 1 output) has scaled numeric values and OHE
  categorical columns. Rules need the ORIGINAL human-readable values:
    EventID = 7, not EventID = 0.2857...
    DestinationPortName = "ms-wbt-server", not a one-hot binary column.
  Only 9 of the 93 raw columns are loaded — RAM footprint ~200 MB.
"""

import sys
import os
import time
import gc

# Force UTF-8 output so coloured characters render correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Make sure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from colorama import Fore, Style, init as colorama_init

from src.config          import RAW_DATA_PATH, CORE_FEATURES, LABEL_COLUMN
from src.rule_engine     import apply_rules, ALL_RULES
from src.rule_evaluator  import evaluate_and_report

colorama_init(autoreset=True)

BANNER = """
+--------------------------------------------------------------+
|   Lightweight Early Lateral Movement Detection System        |
|   Mission 2 — Rule-Based Detection Engine                    |
|   Based on: Smiliotopoulos et al. (2025), C&S Vol. 149      |
+--------------------------------------------------------------+
"""

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 HELPER — Efficient raw data loader
#
#  We load ONLY the 9 required columns (8 features + label) from the raw CSV.
#  This avoids loading the other 84 irrelevant columns, keeping RAM well
#  below 300 MB even for the full 1.75 M row dataset.
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_features() -> pd.DataFrame:
    """
    Load the 8 core features plus the label column from the raw LMD-2023 CSV.

    Strategy:
      - Use pd.read_csv(usecols=...) to skip all irrelevant columns.
      - Read in 200k-row chunks so peak RAM stays flat during loading.
      - Concatenate chunks once for uniform downstream processing.

    Returns
    -------
    pd.DataFrame with columns: CORE_FEATURES + [LABEL_COLUMN]
    RAM footprint: ~200 MB for 1.75 M rows (9 cols, mixed types).
    """
    required_cols = CORE_FEATURES + [LABEL_COLUMN]

    print(Fore.CYAN + "\n" + "="*60)
    print("  STEP 1 — Loading Raw Feature Columns")
    print("="*60 + Style.RESET_ALL)
    print(f"  [-->]  File   : {RAW_DATA_PATH}")
    print(f"  [-->]  Cols   : {required_cols}")
    print(f"  [-->]  Reading in 200,000-row chunks to conserve RAM...")
    print()

    chunks     = []
    chunk_size = 200_000
    reader     = pd.read_csv(
        RAW_DATA_PATH,
        usecols    = required_cols,
        chunksize  = chunk_size,
        low_memory = False,
    )

    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        loaded = (i + 1) * chunk_size
        print(f"    Loaded {loaded:>10,} rows...", end="\r")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    print(f"    Loaded {len(df):>10,} rows — done.         ")

    # Summary
    n_mal = (df[LABEL_COLUMN] != 0).sum()
    n_ben = (df[LABEL_COLUMN] == 0).sum()
    ram   = df.memory_usage(deep=True).sum() / 1e6

    print(f"\n  [OK]   Rows loaded    : {len(df):,}")
    print(f"  [OK]   Malicious rows : {n_mal:,}  ({n_mal/len(df)*100:.2f}%)")
    print(f"  [OK]   Benign rows    : {n_ben:,}  ({n_ben/len(df)*100:.2f}%)")
    print(f"  [OK]   RAM used       : {ram:.1f} MB")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(Fore.CYAN + BANNER + Style.RESET_ALL)
    t_start = time.time()

    # ── Step 1: Load raw features ─────────────────────────────────────────────
    df = load_raw_features()

    # ── Step 2: Apply all detection rules ─────────────────────────────────────
    # The rule engine adds three new columns to the DataFrame:
    #   'rule_alert'   : bool  — True if any rule fired
    #   'matched_rules': str   — names of all rules that fired
    #   'severity'     : str   — highest severity among matched rules
    df = apply_rules(df)

    # ── Step 3: Evaluate and save ─────────────────────────────────────────────
    results = evaluate_and_report(df)

    # ── Final Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    m       = results["metrics"]
    cm      = results["confusion_matrix"]

    print(Fore.CYAN + "="*60)
    print("  MISSION 2 COMPLETE")
    print("="*60 + Style.RESET_ALL)
    print(f"\n  Time elapsed       : {elapsed:.1f} seconds")
    print(f"  Rules applied      : {len(ALL_RULES)}")
    print(f"  Records evaluated  : {cm['total']:,}")
    print()
    print(f"  Detection Rate     : {m['Detection_Rate']*100:.2f}%")
    print(f"  False Positive Rate: {m['False_Positive_Rate']*100:.2f}%")
    print(f"  F1 Score           : {m['F1_Score']*100:.2f}%")
    print()
    print(f"  False Negatives (attacks missed by rules): {cm['FN']:,}")
    print(f"  → These will be handled by Mission 3 (Isolation Forest ML)")
    print()
    print(f"  Outputs:")
    print(f"     {results['predictions_path']}")
    print(f"     {results['report_path']}")
    print(f"\n  Next: Mission 3 — Isolation Forest Anomaly Detection\n")


if __name__ == "__main__":
    main()
