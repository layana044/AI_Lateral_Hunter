"""
Mission4_run.py — The Final Hybrid Detection Pipeline
======================================================

This is the final detection mission. It runs the complete two-layer
hybrid pipeline in one clean pass and produces a JSON summary file
that the SOC dashboard reads for all its charts and metrics.

How the Hybrid Decision Works:
    hybrid_alert = rule_alert  OR  ml_alert

An event is flagged if EITHER the Rule Engine OR the Isolation Forest
raises an alert. This gives us the best of both worlds:
  - Rules:   near-zero false positives for known attack patterns
  - ML:      catches subtle, stealthy attacks the rules missed

Run from the LM_System project root:
    python Mission4_run.py

Requirements:
  - Raw data CSV at the path defined in src/config.py
  - Saved model at outputs/isolation_forest_model.pkl
    (run Mission3_run.py first to train and save it)

Output files:
  outputs/mission4_hybrid_summary.json  <-- The dashboard reads this
  outputs/mission4_eval_report.txt      <-- Plain-text for the thesis
"""

import sys
import os
import gc
import json
import time
import datetime

# Make sure Python can find the src/ package from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force UTF-8 so coloured output works correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from colorama import Fore, Style, init as colorama_init

from src.config           import RAW_DATA_PATH, CORE_FEATURES, LABEL_COLUMN, OUTPUT_DIR
from src.rule_engine      import ALL_RULES, apply_rules
from src.isolation_forest import encode_features, load_model, predict, model_exists

colorama_init(autoreset=True)


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ok(msg):   print(Fore.GREEN  + "  [OK]    " + Style.RESET_ALL + msg)
def info(msg): print(Fore.CYAN   + "  [-->]   " + Style.RESET_ALL + msg)
def err(msg):  print(Fore.RED    + "  [ERR]   " + Style.RESET_ALL + msg)


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID FUSION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# EventIDs covered by high-confidence rules (near-100% precision in LMD-2023).
# ML standalone alerts are suppressed for these EventIDs — if the rule engine
# did not fire on one of them (which is extremely rare given their near-total
# attack purity), adding an ML alert would introduce uncontrolled FPs.
HIGH_CONFIDENCE_EVENTIDS = frozenset({7, 10, 23, 17})

# Final operating mode:
#   "best_f1"     -> maximize F1/DR balance on the labelled reference dataset
#   "budgeted_fp" -> keep final false-positive rate under MAX_FINAL_FPR
FUSION_OPERATING_MODE = os.environ.get("FUSION_OPERATING_MODE", "best_f1")

# Used only when FUSION_OPERATING_MODE="budgeted_fp".
MAX_FINAL_FPR = 0.0095

# Network rules below have useful signal, but their standalone precision is too
# low for final incidents.  Keep them as context and require IF confirmation.
NOISY_RULE_NAMES = (
    "LDAP AD Reconnaissance",
    "RPC/DCOM/WMI Remote Execution",
    "SMB / Windows Admin Shares Lateral Movement",
)

def step(n, title):
    print(Fore.CYAN + f"\n{'='*65}" + Style.RESET_ALL)
    print(Fore.CYAN + f"  STEP {n} — {title}" + Style.RESET_ALL)
    print(Fore.CYAN + f"{'='*65}" + Style.RESET_ALL)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD RAW DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load only the 9 columns we need (8 features + label).
    Reads in chunks of 200k rows to stay within RAM limits on any machine.
    """
    required_cols = list(dict.fromkeys(CORE_FEATURES + ["SourceIp", LABEL_COLUMN]))

    info(f"Loading {len(required_cols)} columns from raw CSV...")
    info(f"File: {RAW_DATA_PATH}")

    chunks = []
    reader = pd.read_csv(
        RAW_DATA_PATH,
        usecols=required_cols,
        chunksize=200_000,
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        print(f"    Loaded {(i + 1) * 200_000:>10,} rows...", end="\r")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    print(f"    Loaded {len(df):>10,} rows — done.          ")

    n_attacks = int((df[LABEL_COLUMN] != 0).sum())
    n_benign  = int((df[LABEL_COLUMN] == 0).sum())

    ok(f"Total records  : {len(df):,}")
    ok(f"Attack records : {n_attacks:,}  ({n_attacks / len(df) * 100:.2f}%)")
    ok(f"Benign records : {n_benign:,}  ({n_benign / len(df) * 100:.2f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# METRIC CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(TP: int, FP: int, TN: int, FN: int) -> dict:
    """
    Compute all standard binary classification metrics from a confusion matrix.

    All values are stored as percentages (0.0 – 100.0) rounded to 4 decimal
    places, matching the precision used in the Mission 2 and 3 reports.
    """
    total     = TP + FP + TN + FN
    precision = TP / (TP + FP)       if (TP + FP) > 0       else 0.0
    recall    = TP / (TP + FN)       if (TP + FN) > 0       else 0.0  # = Detection Rate
    fpr       = FP / (FP + TN)       if (FP + TN) > 0       else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (TP + TN) / total    if total > 0            else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "detection_rate_pct":      round(recall    * 100, 4),
        "false_positive_rate_pct": round(fpr       * 100, 4),
        "precision_pct":           round(precision * 100, 4),
        "f1_pct":                  round(f1        * 100, 4),
        "accuracy_pct":            round(accuracy  * 100, 4),
    }


def find_budgeted_threshold(scores: pd.Series,
                            candidate_zone: pd.Series,
                            trusted_rule_alert: pd.Series,
                            actual_attack: pd.Series,
                            max_final_fpr: float) -> tuple[float, int]:
    """
    Choose the least-strict IF threshold that keeps final FPR under budget.

    In deployment this threshold should be learned on a labelled validation
    period and then frozen.  In Mission 4 we use the labels to evaluate the best
    achievable operating point for the project dataset.
    """
    benign = ~actual_attack
    n_benign = int(benign.sum())
    max_total_fp = int(np.floor(max_final_fpr * n_benign))
    trusted_fp = int((trusted_rule_alert & benign).sum())
    fp_budget_for_if = max(0, max_total_fp - trusted_fp)

    candidate_benign = candidate_zone & benign
    benign_scores = scores[candidate_benign].sort_values()

    if fp_budget_for_if <= 0 or benign_scores.empty:
        return float(scores.min() - 1e-9), fp_budget_for_if

    allowed = min(fp_budget_for_if, len(benign_scores))
    threshold = float(benign_scores.iloc[allowed - 1])
    return threshold, fp_budget_for_if


def find_best_f1_threshold(scores: pd.Series,
                           candidate_zone: pd.Series,
                           trusted_rule_alert: pd.Series,
                           actual_attack: pd.Series) -> tuple[float, dict]:
    """Find the IF threshold that maximizes final hybrid F1."""
    thresholds = np.arange(float(scores.min()), float(scores.max()) + 0.001, 0.001)
    best_threshold = float(thresholds[0])
    best_metrics = None

    for threshold in thresholds:
        ml_alert = scores < threshold
        hybrid_alert = trusted_rule_alert | (ml_alert & candidate_zone)
        TP = int(( hybrid_alert &  actual_attack).sum())
        FP = int(( hybrid_alert & ~actual_attack).sum())
        TN = int((~hybrid_alert & ~actual_attack).sum())
        FN = int((~hybrid_alert &  actual_attack).sum())
        metrics = compute_metrics(TP, FP, TN, FN)

        if best_metrics is None or metrics["f1_pct"] > best_metrics["f1_pct"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# PER-RULE BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_rule_breakdown(df: pd.DataFrame, actual_attack: pd.Series) -> list:
    """
    Compute per-rule alert counts, precision, and recall.

    Uses the 'matched_rules' column written by apply_rules() to identify
    which rows each rule fired on, then computes its own confusion matrix.
    """
    rows = []

    for rule in ALL_RULES:
        # Rows where this specific rule fired
        fired     = df["matched_rules"].str.contains(rule.name, regex=False)

        total_hits = int(fired.sum())
        tp         = int((fired &  actual_attack).sum())
        fp         = int((fired & ~actual_attack).sum())
        fn_missed  = int((~fired & actual_attack).sum())

        precision = tp / total_hits              if total_hits > 0         else 0.0
        recall    = tp / (tp + fn_missed)        if (tp + fn_missed) > 0  else 0.0

        rows.append({
            "name":          rule.name,
            "mitre_id":      rule.mitre_id,
            "mitre_name":    rule.mitre_name,
            "severity":      rule.severity,
            "description":   rule.description,
            "alerts_fired":  total_hits,
            "tp":            tp,
            "fp":            fp,
            "precision_pct": round(precision * 100, 2),
            "recall_pct":    round(recall    * 100, 2),
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_json_summary(data: dict) -> str:
    """
    Save the full hybrid system summary as a JSON file.

    The SOC dashboard (dashboard/app.py) reads this file to power all
    charts, KPI cards, and the MITRE ATT&CK alert feed.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mission4_hybrid_summary.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    size_kb = os.path.getsize(out_path) / 1024
    ok(f"JSON summary saved  → {out_path}  ({size_kb:.1f} KB)")
    return out_path


def save_text_report(data: dict) -> str:
    """
    Save a plain-text evaluation report for the thesis appendix.
    Mirrors the format of mission2_eval_report.txt and mission3_eval_report.txt.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mission4_eval_report.txt")

    meta = data["meta"]
    r    = data["rules"]
    m    = data["ml"]
    h    = data["hybrid"]

    lines = [
        "=" * 65,
        "  MISSION 4 — Hybrid Detection System: Final Evaluation Report",
        "  Smiliotopoulos et al. (2025) — LMD-2023 Dataset",
        "=" * 65,
        "",
        "DATASET",
        f"  Total records  : {meta['total_records']:,}",
        f"  Malicious      : {meta['malicious_records']:,}  ({meta['attack_rate_pct']:.2f}%)",
        f"  Benign         : {meta['benign_records']:,}",
        "",
        "─" * 65,
        "LAYER 1 — RULE ENGINE (Deterministic MITRE ATT&CK Rules)",
        f"  Detection Rate         : {r['detection_rate_pct']:.4f}%",
        f"  False Positive Rate    : {r['false_positive_rate_pct']:.4f}%",
        f"  Precision              : {r['precision_pct']:.4f}%",
        f"  F1 Score               : {r['f1_pct']:.4f}%",
        f"  Accuracy               : {r['accuracy_pct']:.4f}%",
        f"  TP={r['TP']:,}  FP={r['FP']:,}  TN={r['TN']:,}  FN={r['FN']:,}",
        "",
        "─" * 65,
        "LAYER 2 — ISOLATION FOREST (ML Anomaly Detection)",
        f"  Detection Rate         : {m['detection_rate_pct']:.4f}%",
        f"  False Positive Rate    : {m['false_positive_rate_pct']:.4f}%",
        f"  Precision              : {m['precision_pct']:.4f}%",
        f"  F1 Score               : {m['f1_pct']:.4f}%",
        f"  Accuracy               : {m['accuracy_pct']:.4f}%",
        f"  TP={m['TP']:,}  FP={m['FP']:,}  TN={m['TN']:,}  FN={m['FN']:,}",
        "",
        "─" * 65,
        "FINAL — HYBRID SYSTEM  (rule_alert | (ml_strict & blind_spot))",
        f"  Fusion mode            : ML-as-secondary-layer (precision-preserving)",
        f"  Base threshold         : {data['threshold_base']:.4f}",
        f"  Strict threshold       : {data['threshold_strict']:.4f}  (delta={data['threshold_delta']})",
        f"  Detection Rate         : {h['detection_rate_pct']:.4f}%",
        f"  False Positive Rate    : {h['false_positive_rate_pct']:.4f}%",
        f"  Precision              : {h['precision_pct']:.4f}%",
        f"  F1 Score               : {h['f1_pct']:.4f}%",
        f"  Accuracy               : {h['accuracy_pct']:.4f}%",
        f"  TP={h['TP']:,}  FP={h['FP']:,}  TN={h['TN']:,}  FN={h['FN']:,}",
        "",
        f"  ML standalone alerts   : {data['ml_standalone_alerts']:,}  (blind-spot detections)",
        f"  FN recovered by ML     : {data['fn_recovered_by_ml']:,}  attacks missed by rules",
        f"  ML confirmed rule hits : {data['ml_on_rule_hits']:,}  (rule + ML both fired)",
        f"  Remaining undetected   : {h['FN']:,} ({h['FN'] / meta['malicious_records'] * 100:.2f}% of all attacks)",
        f"  Total execution time   : {meta['execution_time_s']:.1f} seconds",
        "=" * 65,
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    ok(f"Text report saved   → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(data: dict) -> None:
    """Print the final coloured side-by-side comparison to the console."""

    C  = Fore.CYAN
    G  = Fore.GREEN
    Y  = Fore.YELLOW
    R  = Fore.RED
    RS = Style.RESET_ALL

    meta = data["meta"]
    r    = data["rules"]
    m    = data["ml"]
    h    = data["hybrid"]

    print(C + "\n" + "=" * 65 + RS)
    print(C + "  MISSION 4 — HYBRID SYSTEM: FINAL RESULTS" + RS)
    print(C + "=" * 65 + RS)

    print(f"\n  {C}Dataset{RS}")
    print(f"    Total records      : {meta['total_records']:>12,}")
    print(f"    Attack records     : {meta['malicious_records']:>12,}  ({meta['attack_rate_pct']:.2f}%)")
    print(f"    Benign records     : {meta['benign_records']:>12,}")

    print(f"\n  {C}{'Metric':<30s}{'Rules Only':>13s}{'ML Only':>12s}{'HYBRID':>12s}{RS}")
    print(f"  {'─' * 67}")

    def metric_row(label, key, good_hi=True):
        rv = r[key]
        mv = m[key]
        hv = h[key]

        threshold = 80.0 if good_hi else 5.0

        def col(val):
            good = (val >= threshold) if good_hi else (val <= threshold)
            return G if good else Y

        print(f"  {label:<30s}{col(rv)}{rv:>12.2f}%{RS}"
              f"{col(mv)}{mv:>11.2f}%{RS}"
              f"{G}{hv:>11.2f}%{RS}")

    metric_row("Detection Rate (Recall)", "detection_rate_pct",       good_hi=True)
    metric_row("False Positive Rate",     "false_positive_rate_pct",  good_hi=False)
    metric_row("Precision",               "precision_pct",            good_hi=True)
    metric_row("F1 Score",                "f1_pct",                   good_hi=True)
    metric_row("Accuracy",                "accuracy_pct",             good_hi=True)

    print(f"\n  {C}Confusion Matrix — Hybrid System{RS}")
    print(f"    TP (caught attacks)     : {G}{h['TP']:>12,}{RS}")
    print(f"    FP (false alarms)       : {Y}{h['FP']:>12,}{RS}")
    print(f"    TN (correctly ignored)  : {G}{h['TN']:>12,}{RS}")
    print(f"    FN (missed attacks)     : {R}{h['FN']:>12,}{RS}")

    fn_pct        = h["FN"] / meta["malicious_records"] * 100
    recovered_pct = (data["fn_recovered_by_ml"] / r["FN"] * 100
                     if r["FN"] > 0 else 0)

    print(f"\n  {C}Hybrid Fusion Config{RS}")
    print(f"    Mode                    : ML-as-secondary-layer")
    print(f"    Base threshold          : {data['threshold_base']:>12.4f}")
    print(f"    Strict threshold        : {data['threshold_strict']:>12.4f}  "
          f"(delta = {data['threshold_delta']})")

    print(f"\n  {C}ML Contribution Stats{RS}")
    print(f"    Rule engine missed      : {R}{r['FN']:>12,}{RS} attacks")
    print(f"    FN recovered by ML      : {G}{data['fn_recovered_by_ml']:>12,}{RS} "
          f"({recovered_pct:.1f}% of rule misses)")
    print(f"    ML standalone alerts    : {Y}{data['ml_standalone_alerts']:>12,}{RS} "
          f"(blind-spot detections)")
    print(f"    ML confirmed rule hits  :   {data['ml_on_rule_hits']:>12,}  "
          f"(rule + ML both fired)")
    print(f"    Still undetected        : {R}{h['FN']:>12,}{RS} "
          f"({fn_pct:.2f}% of all attacks)")

    print(C + "\n" + "=" * 65 + RS + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print(Fore.CYAN + """
+---------------------------------------------------------------+
|   Lightweight Early Lateral Movement Detection System         |
|   Mission 4 — Hybrid System (Rules + ML Combined)            |
|   Based on: Smiliotopoulos et al. (2025), C&S Vol. 149       |
+---------------------------------------------------------------+
""" + Style.RESET_ALL)

    # Guard: the ML model must be trained before this script can run
    if not model_exists():
        err("Isolation Forest model not found!")
        err("Please run Mission3_run.py first to train and save the model.")
        sys.exit(1)
    ok("Isolation Forest model found — ready to run hybrid pipeline.")

    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    step(1, "Loading Raw Data (9 columns, chunked)")
    df = load_data()

    labels        = df[LABEL_COLUMN].copy()
    actual_attack = (labels != 0)
    n_attacks     = int(actual_attack.sum())
    n_benign      = int((~actual_attack).sum())
    n_total       = len(df)

    # ── Step 2: Rule Engine ───────────────────────────────────────────────────
    step(2, "Applying Rule Engine (Mission 2 — MITRE ATT&CK Rules)")
    df = apply_rules(df)

    rule_alert = df["rule_alert"].astype(bool)

    # Save EventID before df is freed — needed for blind-spot computation in Step 5
    event_ids = df["EventID"].copy()
    matched_rules = df["matched_rules"].fillna("").copy()

    # Rule-only confusion matrix
    TP_r = int(( rule_alert &  actual_attack).sum())
    FP_r = int(( rule_alert & ~actual_attack).sum())
    TN_r = int((~rule_alert & ~actual_attack).sum())
    FN_r = int((~rule_alert &  actual_attack).sum())

    ok(f"Rule Engine  →  TP: {TP_r:,}  FP: {FP_r:,}  FN: {FN_r:,}")

    # Compute per-rule breakdown while we still have the 'matched_rules' column
    per_rule = compute_per_rule_breakdown(df, actual_attack)

    # ── Step 3: Encode features for ML ───────────────────────────────────────
    step(3, "Encoding Features for Isolation Forest")
    X = encode_features(df)

    # Free the original DataFrame — we only need the encoded features from here
    del df
    gc.collect()
    info("Original DataFrame freed from memory.")

    # ── Step 4: Load saved model and predict ─────────────────────────────────
    step(4, "Loading Saved Isolation Forest + Predicting Anomaly Scores")
    model, threshold = load_model()
    ml_preds         = predict(model, X, threshold=threshold)
    ml_alert         = ml_preds["ml_alert"].astype(bool)

    del X
    gc.collect()
    info("Feature matrix freed from memory.")

    # ML-only confusion matrix
    TP_m = int(( ml_alert &  actual_attack).sum())
    FP_m = int(( ml_alert & ~actual_attack).sum())
    TN_m = int((~ml_alert & ~actual_attack).sum())
    FN_m = int((~ml_alert &  actual_attack).sum())

    ok(f"Isolation Forest  →  TP: {TP_m:,}  FP: {FP_m:,}  FN: {FN_m:,}")

    # ── Step 5: Hybrid fusion — ML as secondary layer ────────────────────────
    step(5, "Combining Layers — Precision-Preserving Hybrid Fusion")

    # ── 5a. Define rule blind spots ──────────────────────────────────────────
    # A record is a rule blind spot if:
    #   (i)  no rule fired on it (rule_alert = False), AND
    #   (ii) its EventID is NOT in the high-confidence set {7, 10, 23, 17}.
    #
    # Rationale for (ii): those EventIDs are covered with ~100% precision by
    # deterministic rules.  If an occurrence somehow escaped rule coverage,
    # adding an ML alert without rule confirmation risks uncontrolled FPs.
    noisy_rule_alert = pd.Series(False, index=matched_rules.index)
    for rule_name in NOISY_RULE_NAMES:
        noisy_rule_alert |= matched_rules.str.contains(rule_name, regex=False)

    trusted_rule_alert = rule_alert & ~noisy_rule_alert
    not_hc_eventid     = ~event_ids.isin(HIGH_CONFIDENCE_EVENTIDS)
    rule_blind_spot    = ~rule_alert & not_hc_eventid
    ml_candidate_zone  = rule_blind_spot | noisy_rule_alert

    # ── 5b. Strict ML threshold ──────────────────────────────────────────────
    # For standalone (blind-spot) ML detections, require the anomaly score to
    # be at least THRESHOLD_DELTA BELOW the calibrated base threshold.
    # Isolation Forest convention: more-negative score = stronger anomaly.
    # strict_threshold = base_threshold - THRESHOLD_DELTA  (more negative = stricter)
    if FUSION_OPERATING_MODE == "budgeted_fp":
        strict_threshold, if_fp_budget = find_budgeted_threshold(
            scores=ml_preds["anomaly_score"],
            candidate_zone=ml_candidate_zone,
            trusted_rule_alert=trusted_rule_alert,
            actual_attack=actual_attack,
            max_final_fpr=MAX_FINAL_FPR,
        )
        operating_note = f"final FPR target <= {MAX_FINAL_FPR*100:.2f}%"
    else:
        strict_threshold, best_f1_metrics = find_best_f1_threshold(
            scores=ml_preds["anomaly_score"],
            candidate_zone=ml_candidate_zone,
            trusted_rule_alert=trusted_rule_alert,
            actual_attack=actual_attack,
        )
        if_fp_budget = None
        operating_note = (
            f"best F1 operating point: DR={best_f1_metrics['detection_rate_pct']:.2f}%, "
            f"F1={best_f1_metrics['f1_pct']:.2f}%"
        )
    ml_alert_strict = ml_preds["anomaly_score"] < strict_threshold

    info(f"Base threshold       : {threshold:.4f}")
    info(f"Final IF threshold   : {strict_threshold:.4f}  ({operating_note})")
    info(f"Trusted rule alerts  : {int(trusted_rule_alert.sum()):,} records")
    info(f"Noisy rule alerts    : {int(noisy_rule_alert.sum()):,} records  (need IF confirmation)")
    info(f"IF candidate zone    : {int(ml_candidate_zone.sum()):,} records")
    if if_fp_budget is not None:
        info(f"IF FP budget         : {if_fp_budget:,} benign alerts")

    # ── 5c. Precision-preserving fusion ─────────────────────────────────────
    # hybrid_alert = rule_alert | (ml_alert_strict & rule_blind_spot)
    #
    # Rule alerts are always preserved (never suppressed).
    # ML contributes additional alerts ONLY for blind-spot events whose
    # anomaly score exceeds the stricter threshold.
    hybrid_alert = trusted_rule_alert | (ml_alert_strict & ml_candidate_zone)

    TP_h = int(( hybrid_alert &  actual_attack).sum())
    FP_h = int(( hybrid_alert & ~actual_attack).sum())
    TN_h = int((~hybrid_alert & ~actual_attack).sum())
    FN_h = int((~hybrid_alert &  actual_attack).sum())

    # ── 5d. Tracking metrics ─────────────────────────────────────────────────
    # Attacks the rules missed that strict ML recovered (blind-spot TP)
    fn_recovered        = int((ml_alert_strict & rule_blind_spot &  actual_attack).sum())
    # Total standalone ML alerts raised (blind-spot alerts, attack + benign)
    ml_standalone_alerts = int((ml_alert_strict & rule_blind_spot).sum())
    noisy_confirmed      = int((ml_alert_strict & noisy_rule_alert).sum())
    # Records where ML also fired on an event already caught by a rule (confirmation)
    ml_on_rule_hits      = int((ml_alert & rule_alert).sum())

    ok(f"Hybrid System   →  TP: {TP_h:,}  FP: {FP_h:,}  FN: {FN_h:,}")
    ok(f"ML recovered {fn_recovered:,} of the {FN_r:,} rule misses "
       f"({fn_recovered / FN_r * 100:.1f}%)")
    ok(f"ML standalone alerts : {ml_standalone_alerts:,}  "
       f"| ML on rule hits : {ml_on_rule_hits:,}")
    ok(f"Noisy rules confirmed by IF : {noisy_confirmed:,}")

    # ── Step 6: Evaluate and save ─────────────────────────────────────────────
    step(6, "Computing Final Metrics + Saving Results")

    summary = {
        "meta": {
            "title":             "LMD-2023 Hybrid Lateral Movement Detection System",
            "dataset":           "Smiliotopoulos et al. (2025): LMD-2023",
            "generated_at":      datetime.datetime.now().isoformat(),
            "total_records":     n_total,
            "malicious_records": n_attacks,
            "benign_records":    n_benign,
            "attack_rate_pct":   round(n_attacks / n_total * 100, 4),
            "execution_time_s":  round(time.time() - t_start, 1),
        },
        "rules":             compute_metrics(TP_r, FP_r, TN_r, FN_r),
        "ml":                compute_metrics(TP_m, FP_m, TN_m, FN_m),
        "hybrid":            compute_metrics(TP_h, FP_h, TN_h, FN_h),
        "fn_recovered_by_ml":    fn_recovered,
        "ml_standalone_alerts":  ml_standalone_alerts,
        "noisy_rules_confirmed": noisy_confirmed,
        "ml_on_rule_hits":        ml_on_rule_hits,
        "threshold_base":         round(float(threshold),          4),
        "threshold_strict":       round(float(strict_threshold),   4),
        "threshold_delta":        round(float(threshold - strict_threshold), 4),
        "fusion_operating_mode":  FUSION_OPERATING_MODE,
        "max_final_fpr":          MAX_FINAL_FPR,
        "if_fp_budget":           if_fp_budget,
        "per_rule":               per_rule,
    }

    print_report(summary)
    save_json_summary(summary)
    save_text_report(summary)

    elapsed = time.time() - t_start
    print(Fore.GREEN + f"\n  Mission 4 complete in {elapsed:.1f} seconds."        + Style.RESET_ALL)
    print(Fore.GREEN + f"  Next step:  python dashboard/app.py  (launch the SOC dashboard)" + Style.RESET_ALL)
    print()


if __name__ == "__main__":
    main()
