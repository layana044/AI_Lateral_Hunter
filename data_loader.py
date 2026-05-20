"""
data_loader.py -- Loads the raw LMD-2023 CSV and runs integrity checks.

Steps:
  1. Peek at the CSV header -> print all column names so mismatches are visible.
  2. Load the full dataset with pandas (chunked for memory safety on 1 GB file).
  3. Assert row count, column count, and label presence.
  4. Print a summary report.
"""

import os
import sys
import pandas as pd
from colorama import Fore, Style, init as colorama_init

from src.config import (
    RAW_DATA_PATH,
    EXPECTED_MIN_ROWS,
    EXPECTED_COLS,
    CORE_FEATURES,
    FEATURE_MAP,
    LABEL_COLUMN,
)

colorama_init(autoreset=True)


# ----------------------------------------------------------------
#  Internal helpers
# ----------------------------------------------------------------

def _ok(msg: str)   -> None: print(Fore.GREEN  + "  [OK]   " + Style.RESET_ALL + msg)
def _warn(msg: str) -> None: print(Fore.YELLOW + "  [WARN] " + Style.RESET_ALL + msg)
def _err(msg: str)  -> None: print(Fore.RED    + "  [ERR]  " + Style.RESET_ALL + msg)


def _check_file_exists(path: str) -> None:
    """Abort early if the raw CSV is missing."""
    if not os.path.isfile(path):
        _err(f"Raw data file not found:\n    {path}")
        _err("Update RAW_DATA_PATH in src/config.py and try again.")
        sys.exit(1)


def _peek_header(path: str) -> list:
    """Read only the first row to get column names -- very fast even for 1 GB files."""
    header_df = pd.read_csv(path, nrows=0)
    return list(header_df.columns)


def _validate_feature_map(all_columns: list) -> None:
    """
    Warn if any expected CSV column from FEATURE_MAP is missing in the header.
    Prints all CSV columns so the user can fix the mapping in config.py.
    """
    print("\n" + Fore.CYAN + "  CSV header columns detected:" + Style.RESET_ALL)
    for col in all_columns:
        print(f"       {col}")

    print()
    missing = [csv_col for csv_col in CORE_FEATURES if csv_col not in all_columns]
    if missing:
        _warn("The following expected column names were NOT found in the CSV:")
        for col in missing:
            paper_name = [k for k, v in FEATURE_MAP.items() if v == col][0]
            print(f"       {Fore.YELLOW}{paper_name}{Style.RESET_ALL} -> expected '{col}'")
        _warn("Update FEATURE_MAP in src/config.py to match the actual column names above.")
    else:
        _ok("All 8 core feature columns found in the CSV header.")


def _load_csv(path: str) -> pd.DataFrame:
    """
    Load the full CSV using chunked reading so it does not spike RAM.
    Chunk size of 200k rows approx 200 MB peak memory per chunk on a 93-col dataset.
    """
    print(f"\n  Loading CSV (this may take 30-60 s for 1.75 M rows)...")
    chunks = []
    chunk_size = 200_000
    reader = pd.read_csv(path, chunksize=chunk_size, low_memory=False)

    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        loaded = (i + 1) * chunk_size
        print(f"    -> {loaded:,} rows loaded...", end="\r")

    df = pd.concat(chunks, ignore_index=True)
    print()  # newline after \r
    return df


# ----------------------------------------------------------------
#  Public API
# ----------------------------------------------------------------

def load_and_verify() -> pd.DataFrame:
    """
    Main entry point for data loading.

    Returns
    -------
    pd.DataFrame
        The raw (unprocessed) dataset with all 93 columns.
    """
    print(Fore.CYAN + "\n" + "="*60)
    print("  STEP 1 -- Data Loading & Integrity Check")
    print("="*60 + Style.RESET_ALL)

    # 1. File existence
    _check_file_exists(RAW_DATA_PATH)
    _ok(f"File found: {RAW_DATA_PATH}")

    # 2. Header peek + feature map validation
    all_columns = _peek_header(RAW_DATA_PATH)
    _validate_feature_map(all_columns)

    # 3. Full load
    df = _load_csv(RAW_DATA_PATH)

    # 4. Integrity assertions
    print(f"\n  Running integrity checks...")
    n_rows, n_cols = df.shape

    # Row count
    if n_rows >= EXPECTED_MIN_ROWS:
        _ok(f"Row count: {n_rows:,}  (>= {EXPECTED_MIN_ROWS:,} expected)")
    else:
        _err(f"Row count too low: {n_rows:,}  (expected >= {EXPECTED_MIN_ROWS:,})")

    # Column count
    if n_cols == EXPECTED_COLS:
        _ok(f"Column count: {n_cols}  (matches expected {EXPECTED_COLS})")
    else:
        _warn(f"Column count: {n_cols}  (expected {EXPECTED_COLS} -- may still be OK)")

    # Label column
    if LABEL_COLUMN in df.columns:
        label_counts = df[LABEL_COLUMN].value_counts()
        _ok(f"Label column '{LABEL_COLUMN}' found. Class distribution:")
        for cls, cnt in label_counts.items():
            pct = cnt / n_rows * 100
            print(f"       {cls}: {cnt:,}  ({pct:.2f}%)")
    else:
        _warn(f"Label column '{LABEL_COLUMN}' NOT found. Update LABEL_COLUMN in config.py.")

    # Null check on core features
    available_core = [c for c in CORE_FEATURES if c in df.columns]
    null_counts = df[available_core].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls == 0:
        _ok("No null values in the 8 core feature columns.")
    else:
        _warn(f"{total_nulls:,} null values detected in core features:")
        for col, cnt in null_counts[null_counts > 0].items():
            print(f"       {col}: {cnt:,} nulls")

    print(Fore.CYAN + "\n  Integrity check complete.\n" + Style.RESET_ALL)
    return df
