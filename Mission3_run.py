"""
Mission3_run.py — Entry Point for Mission 3: Isolation Forest ML Detection
===========================================================================

Run from the LM_System project root:
    python Mission3_run.py

What This Mission Does (in plain English):
-------------------------------------------
Mission 2 (the rule engine) was very precise — when it raised an alert,
it was almost always right. But it missed 37,284 attacks (26.4%) because
rules can only catch what they were designed to look for.

Mission 3 adds a second "pair of eyes" — a machine learning model called
Isolation Forest. Instead of using rules, it learns what NORMAL Sysmon
traffic looks like, then flags anything that looks statistically unusual.

The Isolation Forest is our answer to the question:
  "What about the attacks the rules missed?"

The 5 Steps This Script Runs:
-------------------------------
  Step 1  Load the raw data (8 features + label, all 1.75M records)
  Step 2  Encode features into numbers the ML model can work with
  Step 3  Train the Isolation Forest on BENIGN records only
  Step 4  Predict anomaly scores for ALL records
  Step 5  Evaluate: how many of the 37,284 missed attacks did ML catch?
          Preview the hybrid system (Rules + ML combined)

Output Files:
--------------
  outputs/isolation_forest_model.pkl    — Saved trained model (for Mission 4)
  outputs/mission3_predictions.csv      — Per-record ML predictions + scores
  outputs/mission3_eval_report.txt      — Plain-text evaluation report

RAM Budget (why this is safe on 8 GB):
----------------------------------------
  Raw data (9 cols × 1.75M rows)  : ~197 MB
  Encoded feature matrix (8 cols) : ~112 MB
  Isolation Forest model           : ~30  MB
  Predictions array                : ~14  MB
  ─────────────────────────────────────────
  Total peak estimate              : ~360 MB   ← well within 8 GB limit
"""

import sys
import os
import time
import gc

# Make sure we can import from the src folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force UTF-8 so coloured output works correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from colorama import Fore, Style, init as colorama_init

from src.config           import RAW_DATA_PATH, CORE_FEATURES, LABEL_COLUMN, OUTPUT_DIR
from src.isolation_forest import (
    encode_features, train_isolation_forest, find_optimal_threshold,
    predict, save_model, load_model, model_exists
)

colorama_init(autoreset=True)

BANNER = """
+--------------------------------------------------------------+
|   Lightweight Early Lateral Movement Detection System        |
|   Mission 3 — Isolation Forest Anomaly Detection            |
|   Based on: Smiliotopoulos et al. (2025), C&S Vol. 149      |
+--------------------------------------------------------------+
"""

# ─────────────────────────────────────────────────────────────────────────────
# HELPER PRINT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def ok(msg):    print(Fore.GREEN  + "  [OK]    " + Style.RESET_ALL + msg)
def info(msg):  print(Fore.CYAN   + "  [-->]   " + Style.RESET_ALL + msg)
def step(n, title):
    print(Fore.CYAN + f"\n{'='*60}" + Style.RESET_ALL)
    print(Fore.CYAN + f"  STEP {n} — {title}" + Style.RESET_ALL)
    print(Fore.CYAN + f"{'='*60}" + Style.RESET_ALL)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD RAW DATA
#
# We load the raw CSV using only 9 of its 93 columns.
# Loading just 9 columns = 197 MB in RAM instead of ~1.5 GB for all columns.
# Smart column selection is how we stay within the 8 GB RAM budget.
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """
    Load the 8 core Sysmon features + the Label column from the raw CSV.
    Uses pandas chunked reading to stay RAM-efficient during the load phase.

    We load the RAW data (not the preprocessed OHE version) because:
      - The Isolation Forest needs clean numeric encoding, not one-hot columns
      - OHE of hostnames + IPs creates 103 columns — wasteful for a tree model
      - 9 raw columns → 197 MB; 103 OHE columns → 700+ MB in RAM

    Returns
    -------
    pd.DataFrame with columns: CORE_FEATURES + [LABEL_COLUMN]
    """
    required_cols = CORE_FEATURES + [LABEL_COLUMN]

    print(f"  [-->]  Loading {len(required_cols)} columns from raw CSV...")
    print(f"  [-->]  File: {RAW_DATA_PATH}")
    print()

    chunks     = []
    chunk_size = 200_000   # Read 200k rows at a time — safe for any RAM size

    reader = pd.read_csv(
        RAW_DATA_PATH,
        usecols    = required_cols,   # Load ONLY the columns we need
        chunksize  = chunk_size,
        low_memory = False,
    )

    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        print(f"    Loaded {(i+1)*chunk_size:>10,} rows...", end="\r")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    print(f"    Loaded {len(df):>10,} rows — done.            ")

    n_mal = int((df[LABEL_COLUMN] != 0).sum())
    n_ben = int((df[LABEL_COLUMN] == 0).sum())

    ok(f"Total records  : {len(df):,}")
    ok(f"Benign records : {n_ben:,}  ({n_ben/len(df)*100:.2f}%)")
    ok(f"Attack records : {n_mal:,}  ({n_mal/len(df)*100:.2f}%)")
    ok(f"RAM in use     : {df.memory_usage(deep=True).sum()/1e6:.1f} MB")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — EVALUATE THE ML MODEL
#
# This is where we answer the key research questions:
#   1. How many of the 37,284 rule-engine misses did the ML catch?
#   2. What is the overall ML-only detection rate and false positive rate?
#   3. What does the hybrid (Rules + ML) look like as a preview?
#
# We compare ML predictions against the ground-truth Label column.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(df_with_predictions: pd.DataFrame) -> dict:
    """
    Evaluate the Isolation Forest's performance and print a full report.

    This function answers three questions:
      A) How did the ML model perform on its own?
      B) How many of the rule engine's False Negatives did ML recover?
      C) What does the combined (hybrid) system look like overall?

    Parameters
    ----------
    df_with_predictions : DataFrame containing LABEL_COLUMN, 'ml_alert',
                          'anomaly_score', and optionally 'rule_alert'
                          (loaded from Mission 2 predictions CSV).

    Returns
    -------
    dict with metrics, counts, and paths for use in Mission 4.
    """
    label  = df_with_predictions[LABEL_COLUMN]
    ml_alert = df_with_predictions["ml_alert"].astype(bool)

    # Ground truth: is this record actually an attack?
    actual_attack = (label != 0)

    # ── ML-only metrics ─────────────────────────────────────────────────────
    TP_ml = int(( ml_alert &  actual_attack).sum())
    FP_ml = int(( ml_alert & ~actual_attack).sum())
    TN_ml = int((~ml_alert & ~actual_attack).sum())
    FN_ml = int((~ml_alert &  actual_attack).sum())

    n_attacks = int(actual_attack.sum())
    n_benign  = int((~actual_attack).sum())
    n_total   = len(df_with_predictions)

    DR_ml   = TP_ml / n_attacks if n_attacks > 0 else 0
    FPR_ml  = FP_ml / n_benign  if n_benign > 0 else 0
    Prec_ml = TP_ml / (TP_ml + FP_ml) if (TP_ml + FP_ml) > 0 else 0
    F1_ml   = (2 * Prec_ml * DR_ml / (Prec_ml + DR_ml)
               if (Prec_ml + DR_ml) > 0 else 0)

    # ── False Negatives recovered from Mission 2 ─────────────────────────────
    # The rule engine had 37,284 False Negatives (attacks it missed).
    # Did the Isolation Forest catch any of them?
    m2_fn_recovered = 0
    has_rule_col    = "rule_alert" in df_with_predictions.columns
    if has_rule_col:
        rule_alert    = df_with_predictions["rule_alert"].astype(bool)
        # Attacks that rules MISSED but ML caught
        ml_only_catch = ml_alert & ~rule_alert & actual_attack
        m2_fn_recovered = int(ml_only_catch.sum())

        # ── Hybrid System preview (Rules OR ML) ──────────────────────────────
        hybrid_alert  = ml_alert | rule_alert
        TP_h  = int(( hybrid_alert &  actual_attack).sum())
        FP_h  = int(( hybrid_alert & ~actual_attack).sum())
        TN_h  = int((~hybrid_alert & ~actual_attack).sum())
        FN_h  = int((~hybrid_alert &  actual_attack).sum())
        DR_h  = TP_h / n_attacks if n_attacks > 0 else 0
        FPR_h = FP_h / n_benign  if n_benign > 0 else 0
        Prec_h= TP_h / (TP_h + FP_h) if (TP_h + FP_h) > 0 else 0
        F1_h  = (2 * Prec_h * DR_h / (Prec_h + DR_h)
                 if (Prec_h + DR_h) > 0 else 0)
    else:
        TP_h = FP_h = TN_h = FN_h = 0
        DR_h = FPR_h = Prec_h = F1_h = 0.0

    # ── Print the report ─────────────────────────────────────────────────────
    C    = Fore.CYAN
    G    = Fore.GREEN
    Y    = Fore.YELLOW
    R    = Fore.RED
    BOLD = Style.BRIGHT
    RS   = Style.RESET_ALL

    print(C + "\n" + "="*65 + RS)
    print(C + "  MISSION 3 — Isolation Forest Evaluation Report" + RS)
    print(C + "="*65 + RS)

    print(f"\n  {C}Dataset Summary{RS}")
    print(f"    Total records    : {n_total:>10,}")
    print(f"    Attack records   : {n_attacks:>10,}  ({n_attacks/n_total*100:.2f}%)")
    print(f"    Benign records   : {n_benign:>10,}  ({n_benign/n_total*100:.2f}%)")

    print(f"\n  {C}── A. Isolation Forest Alone ──{RS}")
    print(f"    Confusion Matrix:")
    print(f"      True  Positives (attacks caught)  : {G}{TP_ml:>8,}{RS}  ({TP_ml/n_attacks*100:.1f}% of attacks)")
    print(f"      False Positives (false alarms)     : {Y}{FP_ml:>8,}{RS}  ({FP_ml/n_benign*100:.2f}% of benign)")
    print(f"      True  Negatives (correctly ignored): {G}{TN_ml:>8,}{RS}")
    print(f"      False Negatives (still missed)     : {R}{FN_ml:>8,}{RS}  ({FN_ml/n_attacks*100:.1f}% of attacks)")
    print()

    def colour(val, good_hi=True, threshold=0.80):
        c = G if (val >= threshold) == good_hi else Y
        return f"{c}{val*100:.2f}%{RS}"

    print(f"    Detection Rate (Recall) : {colour(DR_ml)}")
    print(f"    False Positive Rate     : {colour(FPR_ml, good_hi=False, threshold=0.05)}")
    print(f"    Precision               : {colour(Prec_ml)}")
    print(f"    F1 Score                : {colour(F1_ml)}")

    if has_rule_col:
        print(f"\n  {C}── B. What Did ML Recover from Rule Engine Misses? ──{RS}")
        # Count the actual rule engine FNs from the data (attacks missed by rules)
        rule_fn_count = int((actual_attack & ~rule_alert).sum())
        print(f"    Rule engine False Negatives : {R}{rule_fn_count:,}{RS} attacks")
        print(f"    ML additionally caught      : {G}{m2_fn_recovered:,}{RS} of those missed attacks")
        pct_recovered = m2_fn_recovered / rule_fn_count * 100 if rule_fn_count > 0 else 0
        print(f"    Recovery rate               : {G}{pct_recovered:.1f}%{RS} of rule engine misses")

        print(f"\n  {C}── C. Hybrid System Preview (Rules OR ML) ──{RS}")
        print(f"    This is a preview of Mission 4 — the combined system.")
        print()
        print(f"                          Rule Engine    ML Only    Hybrid (both)")
        print(f"    ─────────────────────────────────────────────────────────────")
        print(f"    Detection Rate      : {G}73.60%{RS}        {colour(DR_ml):<10s}   {G}{DR_h*100:.2f}%{RS}")
        print(f"    False Positive Rate : {G}0.15%{RS}         {colour(FPR_ml, good_hi=False, threshold=0.05):<10s}   {colour(FPR_h, good_hi=False, threshold=0.05)}")
        print(f"    F1 Score            : {G}83.95%{RS}        {colour(F1_ml):<10s}   {G}{F1_h*100:.2f}%{RS}")
        print()
        if FN_h <= FN_ml:
            print(f"    Attacks still missed by HYBRID: {R}{FN_h:,}{RS}  ({FN_h/n_attacks*100:.1f}% of all attacks)")
            print(f"    → These remaining cases are very challenging edge cases")
            print(f"      that neither rules nor anomaly detection can easily catch.")

    print(C + "\n" + "="*65 + RS)

    return {
        "ml": dict(TP=TP_ml, FP=FP_ml, TN=TN_ml, FN=FN_ml,
                   DR=DR_ml, FPR=FPR_ml, Precision=Prec_ml, F1=F1_ml),
        "hybrid": dict(TP=TP_h, FP=FP_h, TN=TN_h, FN=FN_h,
                       DR=DR_h, FPR=FPR_h, Precision=Prec_h, F1=F1_h),
        "fn_recovered": m2_fn_recovered,
    }


def save_predictions(df: pd.DataFrame) -> str:
    """Save the per-record ML predictions to a CSV for use in Mission 4."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mission3_predictions.csv")
    df.to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    ok(f"Predictions saved → {out_path}  ({size_mb:.1f} MB)")
    return out_path


def save_text_report(results: dict, elapsed: float) -> str:
    """Save a plain-text evaluation report for the thesis appendix."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mission3_eval_report.txt")
    ml = results["ml"]
    hy = results["hybrid"]
    lines = [
        "=" * 65,
        "  MISSION 3 — Isolation Forest Evaluation Report",
        "  Smiliotopoulos et al. (2025) — LMD-2023 Dataset",
        "=" * 65,
        "",
        "ISOLATION FOREST ALONE",
        f"  True Positives  (attacks caught)  : {ml['TP']:,}",
        f"  False Positives (false alarms)     : {ml['FP']:,}",
        f"  True Negatives  (correct benign)   : {ml['TN']:,}",
        f"  False Negatives (attacks missed)   : {ml['FN']:,}",
        f"  Detection Rate                     : {ml['DR']*100:.4f}%",
        f"  False Positive Rate                : {ml['FPR']*100:.4f}%",
        f"  Precision                          : {ml['Precision']*100:.4f}%",
        f"  F1 Score                           : {ml['F1']*100:.4f}%",
        "",
        "HYBRID SYSTEM (Rules + ML combined)",
        f"  True Positives                     : {hy['TP']:,}",
        f"  False Positives                    : {hy['FP']:,}",
        f"  Detection Rate                     : {hy['DR']*100:.4f}%",
        f"  False Positive Rate                : {hy['FPR']*100:.4f}%",
        f"  F1 Score                           : {hy['F1']*100:.4f}%",
        "",
        f"  FN recovered from rule engine misses: {results['fn_recovered']:,}",
        f"  Execution time                     : {elapsed:.1f} seconds",
        "=" * 65,
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    ok(f"Report saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Tie everything together
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(Fore.CYAN + BANNER + Style.RESET_ALL)
    t_start = time.time()

    # ── Step 1: Load the raw data ─────────────────────────────────────────────
    step(1, "Loading Raw Data (9 columns, RAM-efficient)")
    df_raw = load_raw_data()
    labels = df_raw[LABEL_COLUMN].copy()

    # ── Step 2: Encode features ───────────────────────────────────────────────
    step(2, "Encoding Features into Numbers for the ML Model")
    X = encode_features(df_raw)

    # Free the raw DataFrame — we no longer need the text columns
    del df_raw
    gc.collect()
    info("Raw DataFrame freed from memory after encoding.")

    # ── Step 3: Train (or load saved model) ───────────────────────────────────
    step(3, "Training the Isolation Forest on Normal (Benign) Records")

    if model_exists():
        info("A saved model was found. Loading it to skip retraining...")
        info("(Delete outputs/isolation_forest_model.pkl to force retraining)")
        model, threshold = load_model()
    else:
        info("No saved model found. Training from scratch...")
        model     = train_isolation_forest(X, labels)
        threshold = find_optimal_threshold(model, X, labels)
        save_model(model, threshold)

    # ── Step 4: Predict on all records ───────────────────────────────────────
    step(4, "Predicting Anomaly Scores for All 1.75M Records")
    ml_preds = predict(model, X, threshold=threshold)

    # Free the feature matrix — predictions are all we need going forward
    del X
    gc.collect()
    info("Feature matrix freed from memory after prediction.")

    # ── Step 5: Evaluate ─────────────────────────────────────────────────────
    step(5, "Evaluating Results")

    # Build the full result DataFrame
    # We also try to load Mission 2 rule_alert column for hybrid preview
    result_df = pd.DataFrame({
        LABEL_COLUMN    : labels.values,
        "anomaly_score" : ml_preds["anomaly_score"].values,
        "ml_alert"      : ml_preds["ml_alert"].values,
    })

    m2_path = os.path.join(OUTPUT_DIR, "mission2_predictions.csv")
    if os.path.exists(m2_path):
        info("Loading Mission 2 rule_alert column for hybrid preview...")
        m2 = pd.read_csv(m2_path, usecols=["rule_alert"])
        result_df["rule_alert"] = m2["rule_alert"].values
        ok("Rule alert column loaded — hybrid preview will be shown.")
    else:
        info("Mission 2 predictions not found — skipping hybrid preview.")

    results  = evaluate(result_df)
    pred_path = save_predictions(result_df)
    save_text_report(results, time.time() - t_start)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    m  = results["ml"]
    hy = results["hybrid"]

    print(Fore.CYAN + "="*60 + Style.RESET_ALL)
    print(Fore.CYAN + "  MISSION 3 COMPLETE" + Style.RESET_ALL)
    print(Fore.CYAN + "="*60 + Style.RESET_ALL)
    print(f"\n  Time elapsed           : {elapsed:.1f} seconds")
    print(f"\n  Isolation Forest alone :")
    print(f"    Detection Rate       : {m['DR']*100:.2f}%")
    print(f"    False Positive Rate  : {m['FPR']*100:.2f}%")
    print(f"    F1 Score             : {m['F1']*100:.2f}%")
    if hy["TP"] > 0:
        print(f"\n  Hybrid Preview (Rules + ML) :")
        print(f"    Detection Rate       : {hy['DR']*100:.2f}%")
        print(f"    False Positive Rate  : {hy['FPR']*100:.2f}%")
        print(f"    F1 Score             : {hy['F1']*100:.2f}%")
    print(f"\n  Outputs:")
    print(f"     {pred_path}")
    print(f"     {os.path.join(OUTPUT_DIR, 'mission3_eval_report.txt')}")
    print(f"\n  Next: Mission 4 — Hybrid System (Rules + ML Combined)\n")


if __name__ == "__main__":
    main()
