"""
rule_evaluator.py — Detection Performance Evaluator
=====================================================

Evaluates the performance of the rule-based engine by comparing its
predictions against the ground-truth labels in the LMD-2023 dataset.

Metrics computed (following Smiliotopoulos et al. (2025), Section 4):
  - True Positives  (TP) : Malicious records correctly flagged.
  - False Positives (FP) : Benign records incorrectly flagged.
  - True Negatives  (TN) : Benign records correctly ignored.
  - False Negatives (FN) : Malicious records missed — the "gap" that the
                           ML layer (Isolation Forest, Mission 3) must cover.

  - Detection Rate (DR) = Recall = TP / (TP + FN)
      "Of all real attacks, what fraction did we detect?"
      → Higher is better. Paper target: ~95%+.

  - False Positive Rate (FPR) = FP / (FP + TN)
      "Of all benign events, what fraction did we wrongly flag?"
      → Lower is better. Analysts want FPR < 5%.

  - Precision = TP / (TP + FP)
      "Of all our alerts, what fraction are genuine attacks?"

  - F1 Score = 2 × (Precision × DR) / (Precision + DR)
      The harmonic mean of Precision and Recall — the primary comparison
      metric against the paper's Table results.

  - Specificity = TN / (TN + FP)
      "Of all benign events, how many did we correctly ignore?"

Output:
  - Console report (coloured, human-readable)
  - outputs/mission2_predictions.csv   — per-record predictions + label
  - outputs/mission2_eval_report.txt   — plain-text evaluation summary
"""

import os
import json

import pandas as pd
from colorama import Fore, Style, init as colorama_init

from src.config import OUTPUT_DIR, LABEL_COLUMN
from src.rule_engine import ALL_RULES

colorama_init(autoreset=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LABEL NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _is_malicious(label_val) -> bool:
    """
    Convert any label encoding to a boolean: True = malicious, False = benign.

    Supports multiple encoding styles found in cybersecurity datasets:
      - Binary : 0 = benign,  1 or 2 = attack  (LMD-2023 uses 0/1/2)
      - String : "Benign", "Normal", "0" = benign; anything else = attack

    LMD-2023 specific: Label 0 = benign, Label 1 = LM attack type 1,
                                          Label 2 = LM attack type 2.
    """
    try:
        numeric = int(float(label_val))
        return numeric != 0           # 0 = benign, 1/2/... = attack
    except (ValueError, TypeError):
        val = str(label_val).strip().lower()
        return val not in ("0", "benign", "normal", "legitimate", "false")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_confusion_matrix(df: pd.DataFrame) -> dict:
    """
    Build the four-cell confusion matrix from the prediction results.

    Requires columns:
      - LABEL_COLUMN  : ground-truth label (any encoding)
      - 'rule_alert'  : bool — True if the rule engine raised an alert

    Returns
    -------
    dict with keys: TP, FP, TN, FN, total, n_malicious, n_benign
    """
    actual_malicious = df[LABEL_COLUMN].apply(_is_malicious)
    predicted_alert  = df["rule_alert"].astype(bool)

    TP = int(( predicted_alert &  actual_malicious).sum())
    FP = int(( predicted_alert & ~actual_malicious).sum())
    TN = int((~predicted_alert & ~actual_malicious).sum())
    FN = int((~predicted_alert &  actual_malicious).sum())

    return {
        "TP"          : TP,
        "FP"          : FP,
        "TN"          : TN,
        "FN"          : FN,
        "total"       : TP + FP + TN + FN,
        "n_malicious" : TP + FN,
        "n_benign"    : TN + FP,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(cm: dict) -> dict:
    """
    Compute standard binary classification metrics from a confusion matrix.

    All metrics are rounded to 6 decimal places for consistency with the
    paper's reported precision.
    """
    TP, FP, TN, FN = cm["TP"], cm["FP"], cm["TN"], cm["FN"]

    # Avoid division-by-zero for degenerate cases
    precision    = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall       = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # = Detection Rate
    fpr          = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    specificity  = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1           = (2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else 0.0)
    accuracy     = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0.0

    return {
        "Detection_Rate"    : round(recall,      6),   # = Recall
        "False_Positive_Rate": round(fpr,         6),
        "Precision"         : round(precision,    6),
        "F1_Score"          : round(f1,           6),
        "Specificity"       : round(specificity,  6),
        "Accuracy"          : round(accuracy,     6),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PER-RULE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def _per_rule_breakdown(df: pd.DataFrame) -> list:
    """
    Compute per-rule precision, recall, and alert counts.

    For each rule, we determine:
      - How many rows it flagged (alerts).
      - Of those, how many were genuinely malicious (TP for that rule alone).
      - Rule-level precision = TP / (TP + FP).

    This breakdown helps identify:
      - Rules with high recall  : good at catching attacks.
      - Rules with low precision: generating too many false alarms.
      - Gaps (no rule fires)    : the FN cases the ML layer must cover.
    """
    actual_malicious = df[LABEL_COLUMN].apply(_is_malicious)
    rows = []

    for rule in ALL_RULES:
        # Identify rows where this specific rule fired
        rule_fired = df["matched_rules"].str.contains(rule.name, regex=False)

        total_alerts = int(rule_fired.sum())
        tp           = int(( rule_fired & actual_malicious).sum())
        fp           = int(( rule_fired & ~actual_malicious).sum())
        fn_missed    = int((~rule_fired & actual_malicious).sum())

        precision = tp / total_alerts if total_alerts > 0 else 0.0
        recall    = tp / (tp + fn_missed) if (tp + fn_missed) > 0 else 0.0

        rows.append({
            "Rule"          : rule.name,
            "MITRE_ID"      : rule.mitre_id,
            "Severity"      : rule.severity,
            "Alerts_Fired"  : total_alerts,
            "TP"            : tp,
            "FP"            : fp,
            "Precision"     : round(precision, 4),
            "Recall"        : round(recall, 4),
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE REPORT PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_report(cm: dict, metrics: dict, per_rule: list) -> None:
    """Print a formatted evaluation report to the console."""

    C  = Fore.CYAN
    G  = Fore.GREEN
    Y  = Fore.YELLOW
    R  = Fore.RED
    RS = Style.RESET_ALL

    print(C + "\n" + "="*65)
    print("  MISSION 2 — Rule Engine Evaluation Report")
    print("="*65 + RS)

    # ── Dataset Summary ──────────────────────────────────────────────
    print(f"\n  {C}Dataset Summary{RS}")
    print(f"    Total records   : {cm['total']:>10,}")
    print(f"    Malicious (pos) : {cm['n_malicious']:>10,}  ({cm['n_malicious']/cm['total']*100:.2f}%)")
    print(f"    Benign (neg)    : {cm['n_benign']:>10,}  ({cm['n_benign']/cm['total']*100:.2f}%)")

    # ── Confusion Matrix ──────────────────────────────────────────────
    print(f"\n  {C}Confusion Matrix{RS}")
    print(f"                       Predicted: Alert   Predicted: Silent")
    print(f"    Actual: Attack               {G}{cm['TP']:>6,}{RS}  (TP)          {R}{cm['FN']:>6,}{RS}  (FN — missed!)")
    print(f"    Actual: Benign               {Y}{cm['FP']:>6,}{RS}  (FP — noise)  {G}{cm['TN']:>6,}{RS}  (TN)")

    # ── Performance Metrics ───────────────────────────────────────────
    print(f"\n  {C}Performance Metrics (compare against paper Table){RS}")

    def _metric_line(label, val, target_good, higher_is_better=True):
        colour = G if (val >= target_good) == higher_is_better else Y
        return f"    {label:<25s} {colour}{val*100:6.2f}%{RS}"

    print(_metric_line("Detection Rate (Recall)",  metrics["Detection_Rate"],     0.80))
    print(_metric_line("False Positive Rate",       metrics["False_Positive_Rate"],0.05, higher_is_better=False))
    print(_metric_line("Precision",                 metrics["Precision"],          0.80))
    print(_metric_line("F1 Score",                  metrics["F1_Score"],           0.80))
    print(_metric_line("Specificity",               metrics["Specificity"],        0.95))
    print(_metric_line("Accuracy",                  metrics["Accuracy"],           0.90))

    # ── Per-Rule Breakdown ────────────────────────────────────────────
    print(f"\n  {C}Per-Rule Breakdown{RS}")
    header = f"    {'Rule':<35s} {'MITRE':<12s} {'Sev':<7s} {'Alerts':>8s} {'TP':>7s} {'FP':>7s} {'Prec':>7s} {'Rec':>7s}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for r in per_rule:
        sev_colour = R if r["Severity"] == "HIGH" else Y
        print(f"    {r['Rule']:<35s} {r['MITRE_ID']:<12s} "
              f"{sev_colour}{r['Severity']:<7s}{RS} "
              f"{r['Alerts_Fired']:>8,} {r['TP']:>7,} {r['FP']:>7,} "
              f"{r['Precision']*100:>6.1f}% {r['Recall']*100:>6.1f}%")

    # ── False Negative Warning ────────────────────────────────────────
    fn_pct = cm["FN"] / cm["n_malicious"] * 100 if cm["n_malicious"] > 0 else 0
    print(f"\n  {R}⚠  False Negatives (attacks NOT caught by rules): "
          f"{cm['FN']:,}  ({fn_pct:.1f}% of all attacks){RS}")
    print(f"     → These {cm['FN']:,} records are the exact cases Mission 3")
    print(f"       (Isolation Forest) must detect using statistical anomaly detection.")

    print(C + "\n" + "="*65 + RS + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  FILE SAVE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def save_predictions(df: pd.DataFrame) -> str:
    """
    Save the per-record prediction results to a CSV for use in Mission 3/4.

    Saved columns:
      - All original 8 core features
      - rule_alert, matched_rules, severity    (from rule engine)
      - LABEL_COLUMN                           (ground truth)

    The Isolation Forest (Mission 3) will load this file and add its own
    'ml_anomaly' and 'anomaly_score' columns, producing the hybrid dataset.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mission2_predictions.csv")

    df.to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(Fore.GREEN + "  [OK]   " + Style.RESET_ALL
          + f"Predictions saved → {out_path}  ({size_mb:.1f} MB)")
    return out_path


def save_text_report(cm: dict, metrics: dict, per_rule: list) -> str:
    """Save a plain-text evaluation report for inclusion in thesis appendix."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mission2_eval_report.txt")

    lines = [
        "=" * 65,
        "  MISSION 2 — Rule-Based Engine Evaluation Report",
        "  Smiliotopoulos et al. (2025) — LMD-2023 Dataset",
        "=" * 65,
        "",
        "DATASET",
        f"  Total records : {cm['total']:,}",
        f"  Malicious     : {cm['n_malicious']:,}  ({cm['n_malicious']/cm['total']*100:.2f}%)",
        f"  Benign        : {cm['n_benign']:,}  ({cm['n_benign']/cm['total']*100:.2f}%)",
        "",
        "CONFUSION MATRIX",
        f"  TP (caught attacks)       : {cm['TP']:,}",
        f"  FP (false alarms)         : {cm['FP']:,}",
        f"  TN (correctly ignored)    : {cm['TN']:,}",
        f"  FN (missed attacks!!)     : {cm['FN']:,}",
        "",
        "PERFORMANCE METRICS",
    ]
    for k, v in metrics.items():
        lines.append(f"  {k:<25s}: {v*100:.4f}%")

    lines += ["", "PER-RULE BREAKDOWN",
              f"  {'Rule':<35s} {'MITRE':<12s} {'Sev':<7s} {'Alerts':>8s} {'TP':>7s} {'FP':>7s} {'Prec':>6s} {'Rec':>6s}"]
    for r in per_rule:
        lines.append(
            f"  {r['Rule']:<35s} {r['MITRE_ID']:<12s} {r['Severity']:<7s} "
            f"{r['Alerts_Fired']:>8,} {r['TP']:>7,} {r['FP']:>7,} "
            f"{r['Precision']*100:>5.1f}% {r['Recall']*100:>5.1f}%"
        )

    lines += [
        "",
        f"FALSE NEGATIVES: {cm['FN']:,} attacks missed by rules",
        "  → These are passed to Isolation Forest (Mission 3) for ML-based detection.",
        "=" * 65,
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(Fore.GREEN + "  [OK]   " + Style.RESET_ALL
          + f"Evaluation report saved → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — Single Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_and_report(df: pd.DataFrame) -> dict:
    """
    Full evaluation pipeline.

    1. Builds confusion matrix from predictions vs ground truth.
    2. Computes DR, FPR, Precision, F1, Specificity, Accuracy.
    3. Generates per-rule breakdown.
    4. Prints coloured console report.
    5. Saves CSV predictions + plain-text report for thesis use.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after run_rules() has been applied. Must contain columns:
        LABEL_COLUMN, 'rule_alert', 'matched_rules', 'severity'.

    Returns
    -------
    dict containing: 'confusion_matrix', 'metrics', 'per_rule_breakdown',
                     'predictions_path', 'report_path'
    """
    print(Fore.CYAN + "\n" + "="*60)
    print("  STEP 3 — Evaluation")
    print("="*60 + Style.RESET_ALL)

    cm       = _build_confusion_matrix(df)
    metrics  = _compute_metrics(cm)
    per_rule = _per_rule_breakdown(df)

    print_report(cm, metrics, per_rule)

    pred_path   = save_predictions(df)
    report_path = save_text_report(cm, metrics, per_rule)

    return {
        "confusion_matrix"   : cm,
        "metrics"            : metrics,
        "per_rule_breakdown" : per_rule,
        "predictions_path"   : pred_path,
        "report_path"        : report_path,
    }
