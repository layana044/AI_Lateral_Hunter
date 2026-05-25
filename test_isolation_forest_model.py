from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the trained LMD-2023 Isolation Forest model.")
    parser.add_argument("--csv", default="LMD-2023 [1.75M Elements][Labelled]checked.csv")
    parser.add_argument("--model", default="models/isolation_forest_lmd2023.joblib")
    parser.add_argument("--threshold", choices=["validated", "paper"], default="validated")
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--max-rows", type=int, default=200000, help="Set 0 to test the full CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model)
    threshold_key = "validation_tuned_threshold" if args.threshold == "validated" else "paper_native_threshold"
    threshold = artifact[threshold_key]
    features = artifact["features"]

    y_true_parts = []
    score_parts = []
    total_rows = 0

    usecols = ["Label"] + features
    for chunk in pd.read_csv(args.csv, usecols=usecols, chunksize=args.chunksize, low_memory=False):
        if args.max_rows and total_rows + len(chunk) > args.max_rows:
            chunk = chunk.iloc[: args.max_rows - total_rows]
        if chunk.empty:
            break

        labels = (pd.to_numeric(chunk["Label"], errors="coerce").fillna(0) > 0).astype("int8").to_numpy()
        for col in features:
            if col not in chunk:
                chunk[col] = ""

        x = artifact["preprocessor"].transform(chunk[features])
        scores = -artifact["model"].decision_function(x)

        y_true_parts.append(labels)
        score_parts.append(scores)
        total_rows += len(chunk)

        if args.max_rows and total_rows >= args.max_rows:
            break

    y_true = np.concatenate(y_true_parts)
    scores = np.concatenate(score_parts)
    y_pred = (scores >= threshold).astype("int8")

    print(f"Rows tested: {len(y_true)}")
    print(f"Threshold: {threshold:.12f} ({args.threshold})")
    if len(np.unique(y_true)) > 1:
        print(f"AUC: {roc_auc_score(y_true, scores):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"F1: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    else:
        print("AUC/Precision/Recall/F1: not available because this slice has only one label class.")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Alerts: {int(y_pred.sum())}")


if __name__ == "__main__":
    main()
