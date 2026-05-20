"""
isolation_forest.py — The Machine Learning Detection Layer (v2 — Behavioral Features)
======================================================================================

What is Isolation Forest?
--------------------------
Isolation Forest is an anomaly detection algorithm.  Instead of learning
what attacks look like (which requires labels), it learns what NORMAL
behaviour looks like — then flags anything that is unusually different.

Version 2 changes vs v1:
--------------------------
  [REMOVED] EventRecordID        — sequential log counter, zero anomaly signal
  [REMOVED] raw ProcessId        — random int PID, only adds noise
  [REMOVED] raw Execution_ProcessID — same; signal captured by same_process_ids
  [REPLACED] DestinationPortName LabelEncoder → port_freq_log
             LabelEncoder assigns arbitrary ordinal numbers: rare attack ports and
             rare benign ports get similar values even though they behave completely
             differently.  Frequency encoding (log of occurrence count) preserves
             the signal: common benign ports score high (normal); rare attack-only
             ports score low (anomalous).
  [REPLACED] Computer LabelEncoder → computer_freq_log  (same rationale)
  [NEW] src_ip_count_log       — log event volume per SourceIp
                                  Attack-only IPs: very low count (first-time IP).
                                  Scanning IPs: very high count (outlier volume).
                                  Normal clients: mid-range stable baseline.
  [NEW] src_ip_unique_ports    — distinct destination ports per SourceIp
                                  Normal hosts connect to 1–5 ports.
                                  Reconnaissance / lateral movement: 20–50+ ports.
  [NEW] src_ip_lm_count_log    — log LM-protocol connections per SourceIp
                                  High LM concentration from one IP = lateral
                                  movement signal beyond what any single rule covers.
  [NEW] eventid_port_freq_log  — log frequency of (EventID, Port) pair
                                  Attack-specific combinations are extremely rare
                                  in benign traffic → low score → easier to isolate.
  [IMPROVED] n_estimators 200 → 300   (more stable, lower variance)
  [IMPROVED] max_samples 512 → 1024   (covers more of the 1.6M benign distribution)
  [IMPROVED] threshold search 0.005 → 0.001 steps, range extended to [−0.30, +0.10]

Feature version:
  FEATURE_VERSION = 2
  Saved models carry this version number.  If encode_features() changes,
  increment FEATURE_VERSION → model_exists() returns False → Mission3 retrains
  automatically.  No manual deletion of pkl files required.

Reference:
  Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). Isolation Forest. ICDM.
  Smiliotopoulos et al. (2025), Computers & Security, Vol. 149.
"""

import gc
import os
import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, fbeta_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from colorama import Fore, Style, init as colorama_init

from src.config import OUTPUT_DIR, LABEL_COLUMN

colorama_init(autoreset=True)

MODEL_SAVE_PATH     = os.path.join(OUTPUT_DIR, "isolation_forest_model.pkl")
THRESHOLD_SAVE_PATH = os.path.join(OUTPUT_DIR, "isolation_forest_threshold.pkl")

# Increment when encode_features() changes to automatically invalidate stale pickles.
FEATURE_VERSION = 4

IF_CONTAMINATION = float(os.environ.get("IF_CONTAMINATION", "0.08"))

CATEGORICAL_OHE_FEATURES = [
    "Computer",
    "DestinationPortName",
    "EventID",
    "Initiated",
    "SourceIsIpv6",
]

NUMERIC_MINMAX_FEATURES = [
    "EventRecordID",
    "Execution_ProcessID",
    "ProcessId",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _ok(msg):   print(Fore.GREEN  + "  [OK]    " + Style.RESET_ALL + msg)
def _info(msg): print(Fore.CYAN   + "  [-->]   " + Style.RESET_ALL + msg)
def _warn(msg): print(Fore.YELLOW + "  [WARN]  " + Style.RESET_ALL + msg)


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# How many BENIGN records carry each Sysmon EventID in LMD-2023.
# Used to compute event_rarity_score: rarer in benign → higher anomaly signal.
BENIGN_COUNT_PER_EVENTID = {
    1:  19530,    2:    234,   3: 1380345,  4:    439,
    5: 172473,    6:     24,   7:       2,  8:    330,
    10:     1,   11:   9212,  12:     661, 13:  17751,
    15:   337,   16:    339,  17:       1, 18:      1,
    22:  9929,   23:       1,
}
TOTAL_BENIGN = 1_611_619

# EventIDs with ≤ 330 benign records (< 0.02% of benign traffic).
# Almost exclusive to attacks in LMD-2023.
HIGH_RISK_EVENT_IDS = frozenset({7, 8, 10, 17, 18, 23})

# Port names associated with lateral movement protocols.
# Numeric strings added to catch datasets where DestinationPortName holds the
# raw port number rather than the IANA service name.
LM_PORT_NAMES = frozenset({
    "ldap", "kerberos", "epmap", "ms-wbt-server",
    "microsoft-ds", "smb", "rdp", "netbios-ssn",
    "389", "445", "88", "135", "3389",
})


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENCODING — v2
# ─────────────────────────────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame):
    """
    Convert the paper-selected 8 Sysmon columns into a sparse matrix.

    v3 follows the paper-style representation:
      - One-hot encode: Computer, DestinationPortName, EventID, Initiated, SourceIp
      - MinMax scale: EventRecordID, Execution_ProcessID, ProcessId

    The output stays sparse CSR to fit 8 GB RAM machines comfortably.
    """
    _info(f"Building feature matrix (v{FEATURE_VERSION} — sparse OHE + MinMax) ...")

    work = df[CATEGORICAL_OHE_FEATURES + NUMERIC_MINMAX_FEATURES].copy()
    for col in CATEGORICAL_OHE_FEATURES:
        work[col] = work[col].astype(str).fillna("missing")
    for col in NUMERIC_MINMAX_FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
                CATEGORICAL_OHE_FEATURES,
            ),
            ("num", MinMaxScaler(), NUMERIC_MINMAX_FEATURES),
        ],
        sparse_threshold=1.0,
    )

    X = preprocessor.fit_transform(work)
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    X = X.astype(np.float32, copy=False).tocsr()

    # Keep the fitted transformer available for save_model().
    encode_features.last_preprocessor = preprocessor

    ohe_count = sum(len(cats) for cats in preprocessor.named_transformers_["cat"].categories_)
    ram_mb = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6
    _ok(f"Feature matrix complete: {X.shape[0]:,} rows × {X.shape[1]:,} columns")
    _info(f"OHE columns: {ohe_count:,}  | numeric columns: {len(NUMERIC_MINMAX_FEATURES)}")
    _info(f"Sparse nnz: {X.nnz:,}  | CSR RAM: {ram_mb:.1f} MB")
    return X


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_isolation_forest(X_all: pd.DataFrame,
                            labels: pd.Series) -> IsolationForest:
    """
    Train an Isolation Forest on the unlabeled/full feature matrix.

    v3 follows the paper-style IF setup more closely:
      n_estimators : 300
      contamination: 0.30
      max_samples  : 1024
      max_features : 1.0
    """
    n_records = X_all.shape[0]

    _info(f"Training set  : {n_records:,} unlabeled records")
    _info(f"Contamination : {IF_CONTAMINATION:.4f}")
    _info("Training 300 isolation trees  (n_estimators=300, max_samples=1024) ...")

    model = IsolationForest(
        n_estimators  = 300,
        contamination = IF_CONTAMINATION,
        max_samples   = 1024,
        max_features  = 1.0,
        warm_start    = True,
        random_state  = 42,
        n_jobs        = -1,           # all CPU cores
    )
    model.fit(X_all)

    _ok(f"Training complete.  300 trees built on {n_records:,} records.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(model: IsolationForest,
                            X_all: pd.DataFrame,
                            labels: pd.Series) -> float:
    """
    Find the anomaly-score cutoff that maximises F2-score.

    v2 improvements:
      - Search range extended to [−0.30, +0.10] (was [−0.25, +0.05])
      - Step size reduced to 0.001 (was 0.005) — 400 candidates vs 60
        Fine-grained search avoids the ±0.0025 rounding error of v1 and
        finds the true F2 optimum more accurately.

    F2 weights recall 2× over precision: we want the ML layer to recover as
    many rule-engine misses as possible.  The Mission 4 THRESHOLD_DELTA then
    applies a stricter cutoff for standalone (blind-spot) ML alerts to
    preserve the hybrid system's precision.
    """
    _info("Threshold search: F2-maximising, range [−0.30, +0.10], step 0.001 ...")
    _info(f"({int((0.10 - (-0.30)) / 0.001)} candidate thresholds evaluated)")

    scores = model.decision_function(X_all)
    y_true = (labels != 0).astype(int)
    n_pos  = int(y_true.sum())
    n_neg  = int((y_true == 0).sum())

    best_f2  = 0.0
    best_thr = 0.0
    best_met = {}

    for thr in np.arange(-0.30, 0.10, 0.001):
        y_pred = (scores < thr).astype(int)
        f2     = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2  = f2
            best_thr = thr
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            best_met = dict(
                TP=tp, FP=fp, TN=tn, FN=fn,
                DR   = tp / n_pos if n_pos > 0 else 0,
                FPR  = fp / n_neg if n_neg > 0 else 0,
                Prec = tp / (tp + fp) if (tp + fp) > 0 else 0,
                F1   = f1_score(y_true, y_pred, zero_division=0),
                F2   = best_f2,
            )

    _ok(f"Optimal threshold: {best_thr:.4f}  (F2={best_met['F2']:.4f})")
    _info(f"  Detection Rate : {best_met['DR']  * 100:.2f}%   "
          f"FPR: {best_met['FPR'] * 100:.2f}%")
    _info(f"  Precision      : {best_met['Prec'] * 100:.2f}%   "
          f"F1:  {best_met['F1']  * 100:.2f}%")
    _info(f"  TP={best_met['TP']:,}  FP={best_met['FP']:,}  "
          f"TN={best_met['TN']:,}  FN={best_met['FN']:,}")

    return float(best_thr)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def predict(model: IsolationForest,
            X_all: pd.DataFrame,
            threshold: float = 0.0) -> pd.DataFrame:
    """
    Score all records with decision_function() and apply the calibrated threshold.

    Returns
    -------
    pd.DataFrame with:
      'anomaly_score' : Raw IF score.  More negative → more anomalous.
      'ml_alert'      : True where score < threshold (record flagged as attack).
    """
    n_rows = X_all.shape[0]
    _info(f"Scoring {n_rows:,} records  (threshold: {threshold:.4f}) ...")
    scores = model.decision_function(X_all)
    alerts = scores < threshold
    n_alert = int(alerts.sum())
    _ok(f"Prediction complete.  {n_alert:,} alerts  "
        f"({n_alert / n_rows * 100:.2f}% of dataset).")
    return pd.DataFrame({"anomaly_score": scores, "ml_alert": alerts})


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD  (versioned to prevent stale-model bugs)
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: IsolationForest, threshold: float) -> str:
    """Save the trained model and calibrated threshold with feature version tag."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preprocessor = getattr(encode_features, "last_preprocessor", None)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "preprocessor": preprocessor,
                "version": FEATURE_VERSION,
                "feature_mode": "sparse_ohe_minmax",
                "contamination": IF_CONTAMINATION,
                "n_features": getattr(model, "n_features_in_", None),
            },
            f,
        )
    with open(THRESHOLD_SAVE_PATH, "wb") as f:
        pickle.dump({"threshold": threshold, "version": FEATURE_VERSION}, f)
    size_kb = os.path.getsize(MODEL_SAVE_PATH) / 1024
    _ok(f"Model saved     → {MODEL_SAVE_PATH}  ({size_kb:.0f} KB)")
    _ok(f"Threshold saved → {THRESHOLD_SAVE_PATH}  (value: {threshold:.4f})")
    return MODEL_SAVE_PATH


def load_model() -> tuple:
    """
    Load model and threshold from disk.
    Raises ValueError if the saved feature version does not match FEATURE_VERSION,
    preventing silent wrong-feature predictions.
    """
    with open(MODEL_SAVE_PATH, "rb") as f:
        pkg = pickle.load(f)
    with open(THRESHOLD_SAVE_PATH, "rb") as f:
        tpkg = pickle.load(f)

    # --- model ---
    if isinstance(pkg, dict):
        saved_ver = pkg.get("version")
        if saved_ver != FEATURE_VERSION:
            raise ValueError(
                f"Saved model is feature version {saved_ver!r}; "
                f"current version is {FEATURE_VERSION}.\n"
                "  Delete outputs/isolation_forest_model.pkl and "
                "outputs/isolation_forest_threshold.pkl\n"
                "  then run:  python Mission3_run.py"
            )
        model = pkg["model"]
        load_model.last_preprocessor = pkg.get("preprocessor")
    else:
        raise ValueError(
            "Saved model has no version tag (created before v2).\n"
            "  Delete outputs/isolation_forest_model.pkl and "
            "outputs/isolation_forest_threshold.pkl\n"
            "  then run:  python Mission3_run.py"
        )

    # --- threshold ---
    if isinstance(tpkg, dict):
        threshold = float(tpkg["threshold"])
    else:
        threshold = float(tpkg)   # old single-value format

    _ok(f"Model loaded  (feature v{FEATURE_VERSION}, threshold: {threshold:.4f})")
    return model, threshold


def model_exists() -> bool:
    """
    Return True only if both pkl files exist AND carry the current feature version.
    Returns False (triggering retraining) if files are absent OR stale.
    """
    if not (os.path.exists(MODEL_SAVE_PATH) and
            os.path.exists(THRESHOLD_SAVE_PATH)):
        return False
    try:
        with open(MODEL_SAVE_PATH, "rb") as f:
            pkg = pickle.load(f)
        if not isinstance(pkg, dict):
            _warn("Saved model has no version tag — will retrain with v2 features.")
            return False
        if pkg.get("version") != FEATURE_VERSION:
            _warn(
                f"Saved model is feature v{pkg.get('version')}; "
                f"current is v{FEATURE_VERSION}. Will retrain."
            )
            return False
        if float(pkg.get("contamination", IF_CONTAMINATION)) != IF_CONTAMINATION:
            _warn(
                f"Saved model contamination is {pkg.get('contamination')}; "
                f"current is {IF_CONTAMINATION}. Will retrain."
            )
            return False
        if pkg.get("n_features") is None:
            _warn("Saved model has no feature-count tag. Will retrain.")
            return False
        return True
    except Exception as exc:
        _warn(f"Could not read saved model ({exc}). Will retrain.")
        return False
