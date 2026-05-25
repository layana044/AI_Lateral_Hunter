from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


LABEL = "Label"
CATEGORICAL_FEATURES = ["Computer", "DestinationPortName", "EventID", "Initiated", "SourceIsIpv6"]
NUMERIC_FEATURES = ["EventRecordID", "Execution_ProcessID", "ProcessId"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Isolation Forest using the paper's future-work direction.")
    parser.add_argument("--csv", default="LMD-2023 [1.75M Elements][Labelled]checked.csv")
    parser.add_argument("--model-out", default="models/isolation_forest_lmd2023_tuned.joblib")
    parser.add_argument("--report-out", default="reports/isolation_forest_lmd2023_tuned_metrics.json")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.10,
        help="Stratified fraction used only to choose the best lightweight config.",
    )
    return parser.parse_args()


def read_selected_columns(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=[LABEL] + FEATURES, low_memory=False)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").fillna("missing")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df[LABEL] = (pd.to_numeric(df[LABEL], errors="coerce").fillna(0) > 0).astype("int8")
    return df


def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
        ]
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("minmax", MinMaxScaler())])
    return ColumnTransformer(
        [("categorical", categorical, CATEGORICAL_FEATURES), ("numeric", numeric, NUMERIC_FEATURES)],
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def scores_for(model: IsolationForest, matrix) -> np.ndarray:
    return -model.decision_function(matrix)


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 500)))
    best = float(candidates[0])
    best_f1 = -1.0
    for candidate in candidates:
        pred = (scores >= candidate).astype("int8")
        value = f1_score(y_true, pred, zero_division=0)
        if value > best_f1:
            best = float(candidate)
            best_f1 = value
    return best


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype("int8")
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "threshold": float(threshold),
    }


def split_xy(df: pd.DataFrame, random_state: int):
    y = df[LABEL].to_numpy(dtype="int8")
    x = df[FEATURES]
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x, y, test_size=0.40, stratify=y, random_state=random_state
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.50, stratify=y_holdout, random_state=random_state
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def main() -> None:
    args = parse_args()
    df = read_selected_columns(args.csv)

    if args.sample_frac < 1.0:
        _, tuning_df = train_test_split(
            df,
            test_size=args.sample_frac,
            stratify=df[LABEL],
            random_state=args.random_state,
        )
    else:
        tuning_df = df

    tune_x_train, tune_x_val, _, _, tune_y_val, _ = split_xy(tuning_df, args.random_state)
    tune_preprocessor = build_preprocessor()
    tune_x_train_prep = tune_preprocessor.fit_transform(tune_x_train)
    tune_x_val_prep = tune_preprocessor.transform(tune_x_val)

    # Tiny search space for an 8 GB RAM machine. Contamination mainly changes
    # IsolationForest's native offset, so this fast run varies the tree shape
    # knobs that can actually change anomaly ranking.
    search_space = [
        (300, 0.30, 1, "auto"),
        (300, 0.30, 0.5, "auto"),
        (300, 0.30, 1.0, "auto"),
        (300, 0.30, 1, 4096),
    ]

    results = []
    best_result: dict[str, Any] | None = None
    best_model: IsolationForest | None = None
    start_all = perf_counter()

    for index, (n_estimators, contamination, max_features, max_samples) in enumerate(search_space, start=1):
        start = perf_counter()
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            max_samples=max_samples,
            warm_start=True,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(tune_x_train_prep)
        val_scores = scores_for(model, tune_x_val_prep)
        threshold = best_threshold(tune_y_val, val_scores)
        val_metrics = evaluate(tune_y_val, val_scores, threshold)
        row = {
            "rank_input_order": index,
            "params": {
                "n_estimators": n_estimators,
                "contamination": contamination,
                "max_features": max_features,
                "max_samples": max_samples,
                "warm_start": True,
            },
            "validation": val_metrics,
            "seconds": round(perf_counter() - start, 3),
        }
        results.append(row)
        if best_result is None or val_metrics["f1"] > best_result["validation"]["f1"]:
            best_result = row
        print(
            f"{index}/{len(search_space)} f1={val_metrics['f1']:.4f} "
            f"auc={val_metrics['auc']:.4f} params={row['params']}",
            flush=True,
        )

    assert best_result is not None

    del tune_x_train, tune_x_val, tune_x_train_prep, tune_x_val_prep, tuning_df

    full_x_train, full_x_val, full_x_test, _, full_y_val, full_y_test = split_xy(df, args.random_state)
    full_preprocessor = build_preprocessor()
    full_x_train_prep = full_preprocessor.fit_transform(full_x_train)
    full_x_val_prep = full_preprocessor.transform(full_x_val)
    full_x_test_prep = full_preprocessor.transform(full_x_test)

    final_model = IsolationForest(
        **best_result["params"],
        random_state=args.random_state,
        n_jobs=-1,
    )
    final_model.fit(full_x_train_prep)
    full_val_scores = scores_for(final_model, full_x_val_prep)
    final_threshold = best_threshold(full_y_val, full_val_scores)
    full_val_metrics = evaluate(full_y_val, full_val_scores, final_threshold)
    test_scores = scores_for(final_model, full_x_test_prep)
    test_metrics = evaluate(full_y_test, test_scores, final_threshold)

    report = {
        "paper_if_reported_percent": {
            "auc": 94.71,
            "precision": 88.92,
            "recall": 98.50,
            "f1": 93.02,
            "accuracy": 97.33,
        },
        "split": {
            "train_percent": 60,
            "validation_percent": 20,
            "test_percent": 20,
            "train_rows": int(len(full_x_train)),
            "validation_rows": int(len(full_x_val)),
            "test_rows": int(len(full_x_test)),
        },
        "best": {
            **best_result,
            "full_validation": full_val_metrics,
            "test": test_metrics,
        },
        "all_results": sorted(results, key=lambda item: item["validation"]["f1"], reverse=True),
        "total_seconds": round(perf_counter() - start_all, 3),
    }

    artifact = {
        "preprocessor": full_preprocessor,
        "model": final_model,
        "features": FEATURES,
        "threshold": final_threshold,
        "params": best_result["params"],
        "score_meaning": "Use -decision_function output; higher values are more anomalous.",
    }

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_out, compress=3)
    Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["best"], indent=2))


if __name__ == "__main__":
    main()
