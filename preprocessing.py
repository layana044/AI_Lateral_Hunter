"""
preprocessing.py -- Feature selection, OHE, MinMax scaling, and multi-scale slicing.

Pipeline (exactly as Smiliotopoulos 2025, Section 3):
  1. Select 8 core features + label column
  2. One-Hot Encode categorical features
  3. MinMax Scale numeric features  (fit on full dataset -> transform all slices)
  4. Re-attach the label column (never scaled)
  5. Save 4 pre-sliced CSV files: 100k / 500k / 1M / full
     (large slices written in chunks to stay within RAM limits)
"""

import os
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from colorama import Fore, Style, init as colorama_init

from src.config import (
    CORE_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    LABEL_COLUMN,
    OUTPUT_DIR,
    SAMPLE_SIZES,
    RANDOM_STATE,
)

colorama_init(autoreset=True)

# Rows written per chunk to disk (keeps peak RAM low during CSV export)
WRITE_CHUNK_SIZE = 50_000


# ----------------------------------------------------------------
#  Internal helpers
# ----------------------------------------------------------------

def _ok(msg: str)   -> None: print(Fore.GREEN  + "  [OK]   " + Style.RESET_ALL + msg)
def _info(msg: str) -> None: print(Fore.CYAN   + "  [-->]  " + Style.RESET_ALL + msg)
def _warn(msg: str) -> None: print(Fore.YELLOW + "  [WARN] " + Style.RESET_ALL + msg)


def _select_features(df: pd.DataFrame):
    """
    Keep only the 8 core features and the label column.
    Returns (features_df, label_series).
    """
    available = [c for c in CORE_FEATURES if c in df.columns]
    missing   = [c for c in CORE_FEATURES if c not in df.columns]

    if missing:
        _warn(f"Skipping {len(missing)} missing core feature(s): {missing}")
    else:
        _ok(f"All 8 core feature columns selected.")

    features = df[available].copy()
    label    = df[LABEL_COLUMN].copy() if LABEL_COLUMN in df.columns else None

    if label is None:
        _warn(f"Label column '{LABEL_COLUMN}' not found -- output will have no label column.")

    return features, label


def _apply_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot Encode all categorical feature columns.
    Uses pandas get_dummies (drop_first=False) -- matches paper methodology.
    """
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    _info(f"Applying OHE to {len(cat_cols)} categorical column(s): {cat_cols}")

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=np.float32)
    n_new = df_encoded.shape[1] - df.shape[1] + len(cat_cols)
    _ok(f"OHE complete. Features expanded: {df.shape[1]} -> {df_encoded.shape[1]}  (+{n_new} dummy cols)")
    return df_encoded


def _apply_minmax(df: pd.DataFrame, scaler=None):
    """
    MinMax scale numeric feature columns to [0, 1].
    If scaler is None, a new scaler is fit on df (used for the full dataset).
    """
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    _info(f"Applying MinMax scaling to {len(num_cols)} numeric column(s): {num_cols}")

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[num_cols] = scaler.fit_transform(df[num_cols])
        _ok("MinMax scaler fit+transform on full dataset.")
    else:
        df[num_cols] = scaler.transform(df[num_cols])
        _ok("MinMax transform applied using pre-fit scaler.")

    return df, scaler


def _save_chunked(df: pd.DataFrame, out_path: str) -> None:
    """
    Write df to CSV in WRITE_CHUNK_SIZE-row chunks.
    Avoids holding the entire string-converted frame in RAM at once,
    which prevents ArrayMemoryError on 1M+ row datasets.
    """
    total = len(df)
    for i, start in enumerate(range(0, total, WRITE_CHUNK_SIZE)):
        chunk = df.iloc[start : start + WRITE_CHUNK_SIZE]
        chunk.to_csv(
            out_path,
            mode="a" if i > 0 else "w",
            index=False,
            header=(i == 0),
        )
        pct = min((start + WRITE_CHUNK_SIZE) / total * 100, 100)
        print(f"    writing... {pct:5.1f}%", end="\r")
    print()  # newline after \r


def _save_slice(df: pd.DataFrame, name: str) -> str:
    """Save a dataframe slice to outputs/ using chunked writing."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"preprocessed_data_{name}.csv")

    _info(f"Writing [{name}] slice ({len(df):,} rows, {df.shape[1]} cols)...")
    _save_chunked(df, out_path)

    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    _ok(f"Saved [{name:>5s}] -> {out_path}  ({size_mb:.1f} MB,  {len(df):,} rows)")
    return out_path


# ----------------------------------------------------------------
#  Public API
# ----------------------------------------------------------------

def run_pipeline(df: pd.DataFrame) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as returned by data_loader.load_and_verify().

    Returns
    -------
    dict[str, str]
        Mapping of scale name -> saved CSV path.
    """
    print(Fore.CYAN + "\n" + "="*60)
    print("  STEP 2 -- Preprocessing Pipeline")
    print("="*60 + Style.RESET_ALL)

    # -- 1. Feature selection ---------------------------------
    print(f"\n  [1/4] Feature Selection")
    features_df, label_series = _select_features(df)

    # Free the raw df to recover RAM before OHE expansion
    del df
    gc.collect()

    # -- 2. One-Hot Encoding ----------------------------------
    print(f"\n  [2/4] One-Hot Encoding")
    features_encoded = _apply_ohe(features_df)
    del features_df
    gc.collect()

    # -- 3. MinMax Scaling (fit on full data) -----------------
    print(f"\n  [3/4] MinMax Scaling")
    features_scaled, fitted_scaler = _apply_minmax(features_encoded)
    del features_encoded
    gc.collect()

    # -- 4. Re-attach label -----------------------------------
    if label_series is not None:
        features_scaled[LABEL_COLUMN] = label_series.values
        _ok(f"Label column '{LABEL_COLUMN}' re-attached (not scaled).")

    total_features = features_scaled.shape[1] - (1 if label_series is not None else 0)
    _ok(f"Final feature count after OHE: {total_features}")

    # -- 5. Slice & Save (chunked writes) ---------------------
    print(f"\n  [4/4] Saving Multi-Scale Slices")
    saved_paths = {}

    for name, n in SAMPLE_SIZES.items():
        if n is None:
            slice_df = features_scaled  # no copy -- full dataset
        else:
            # Stratified random sample to preserve class ratio
            if label_series is not None and features_scaled[LABEL_COLUMN].nunique() > 1:
                slice_df = (
                    features_scaled
                    .groupby(LABEL_COLUMN, group_keys=False)
                    .apply(lambda g: g.sample(
                        min(n, len(g)),
                        random_state=RANDOM_STATE
                    ))
                    .sample(frac=1, random_state=RANDOM_STATE)
                    .reset_index(drop=True)
                    .iloc[:n]
                )
            else:
                slice_df = features_scaled.sample(
                    n=min(n, len(features_scaled)),
                    random_state=RANDOM_STATE
                ).reset_index(drop=True)

        path = _save_slice(slice_df, name)
        saved_paths[name] = path

        # Free small slices immediately; keep full for last
        if n is not None:
            del slice_df
            gc.collect()

    print(Fore.GREEN + "\n  Preprocessing pipeline complete.\n" + Style.RESET_ALL)
    return saved_paths
