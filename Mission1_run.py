"""
Mission1_run.py -- Entry point for Mission 1: Data Acquisition & Preprocessing.

Run from the LM_System project root:
    python Mission1_run.py

What it does:
  1. Loads and verifies the raw LMD-2023 dataset (1.75 M records, 93 features)
  2. Applies OHE + MinMax preprocessing on the 8 core features (Table 3)
  3. Saves 4 pre-sliced CSV files to outputs/:
       preprocessed_data_100k.csv  -- dev / debug
       preprocessed_data_500k.csv  -- validation run
       preprocessed_data_1M.csv    -- near-full run
       preprocessed_data_full.csv  -- 1.75 M (paper comparison)
"""

import sys
import os
import time

# Force UTF-8 output so any Unicode characters render on all Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from colorama import Fore, Style, init as colorama_init

# Ensure the project root is on the path so `src.*` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader   import load_and_verify
from src.preprocessing import run_pipeline

colorama_init(autoreset=True)

BANNER = """
+--------------------------------------------------------------+
|   Lightweight Early Lateral Movement Detection System        |
|   Mission 1 -- Data Acquisition & Preprocessing             |
|   Based on: Smiliotopoulos et al. (2025), C&S Vol. 149      |
+--------------------------------------------------------------+
"""


def main() -> None:
    print(Fore.CYAN + BANNER + Style.RESET_ALL)
    t_start = time.time()

    # -- Step 1: Load & verify raw data -----------------------
    df = load_and_verify()

    # -- Step 2: Preprocess + save all 4 slices ---------------
    saved_paths = run_pipeline(df)

    # -- Final summary ----------------------------------------
    elapsed = time.time() - t_start
    print(Fore.CYAN + "="*60)
    print("  MISSION 1 COMPLETE")
    print("="*60 + Style.RESET_ALL)
    print(f"\n  Time elapsed  : {elapsed:.1f} seconds")
    print(f"  Raw rows      : {len(df):,}")
    print(f"\n  Output files:")
    for name, path in saved_paths.items():
        size_mb = os.path.getsize(path) / (1024 ** 2)
        label = {
            "100k": "dev / debug",
            "500k": "validation",
            "1M"  : "near-full run",
            "full": "PAPER COMPARISON -- use this for final results",
        }.get(name, "")
        print(f"     [{name:>5s}]  {os.path.basename(path):<40s}  {size_mb:6.1f} MB  -- {label}")

    print(f"\n  [OK] All 4 slices ready. Mission 1 complete.")
    print(f"  Next: Mission 2 -- Rule-Based Engine\n")


if __name__ == "__main__":
    main()
