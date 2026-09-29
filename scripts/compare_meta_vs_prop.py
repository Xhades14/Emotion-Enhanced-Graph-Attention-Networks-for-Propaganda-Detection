"""Compare saved meta-classifier against propaganda baseline using existing scores."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


FEATURE_CSV = Path("artifacts") / "prop_datasets" / "tree_width" / "meta_features.csv"
META_MODEL_PATH = Path("artifacts") / "models" / "meta_classifier.joblib"

EMOTION_FEATURES: List[str] = [
    "mean_anger",
    "mean_disgust",
    "mean_fear",
    "mean_joy",
    "mean_sadness",
    "mean_surprise",
    "mean_neutral",
]
ADDITIONAL_FEATURES = ["entropy", "variance"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved meta-classifier versus propaganda baseline")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=FEATURE_CSV,
        help="Path to meta feature CSV (from build_meta_dataset.py)",
    )
    parser.add_argument(
        "--meta-model",
        type=Path,
        default=META_MODEL_PATH,
        help="Serialized meta-classifier pipeline (.joblib)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for propaganda probability baseline",
    )
    return parser.parse_args()


def load_meta_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV not found at {path}; run build_meta_dataset.py")
    table = pd.read_csv(path)
    required = {"true_label", "text_pred"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"meta_features.csv missing columns: {sorted(missing)}")
    return table


def evaluate_baseline(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> None:
    predictions = (probabilities >= threshold).astype(int)
    print("Propaganda baseline (full dataset)")
    print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
    print(f"F1-score: {f1_score(labels, predictions, zero_division=0):.4f}")
    print(classification_report(labels, predictions, zero_division=0))


def evaluate_meta_pipeline(table: pd.DataFrame, model, feature_columns: List[str]) -> None:
    missing = [col for col in feature_columns if col not in table.columns]
    if missing:
        raise ValueError(f"Meta feature table missing columns: {missing}")

    X = table[feature_columns]
    y = table["true_label"].astype(int).to_numpy()
    start = time.perf_counter()
    preds = model.predict(X)
    print(f"Meta-classifier prediction time {time.perf_counter() - start:.2f}s")
    print("Meta-classifier (full dataset)")
    print(f"Accuracy: {accuracy_score(y, preds):.4f}")
    print(f"F1-score: {f1_score(y, preds, zero_division=0):.4f}")
    print(classification_report(y, preds, zero_division=0))

    coef = model.named_steps["clf"].coef_[0]
    print("Meta-classifier coefficients:")
    for name, weight in zip(feature_columns, coef):
        print(f"  {name:>12}: {weight:+.4f}")


def main() -> None:
    args = parse_args()
    overall_start = time.perf_counter()
    print(f"Loading meta features from {args.features_csv}", flush=True)
    table = load_meta_table(args.features_csv)
    print(f"Loaded meta feature table with shape {table.shape}")

    print("Evaluating propaganda baseline from stored probabilities", flush=True)
    evaluate_baseline(table["text_pred"].to_numpy(), table["true_label"].astype(int).to_numpy(), args.threshold)

    print(f"Loading meta-classifier from {args.meta_model}", flush=True)
    if not args.meta_model.exists():
        raise FileNotFoundError(f"Meta-classifier file not found at {args.meta_model}")
    meta_model = joblib.load(args.meta_model)

    feature_columns = ["text_pred", *EMOTION_FEATURES, *ADDITIONAL_FEATURES]
    evaluate_meta_pipeline(table, meta_model, feature_columns)

    print(f"Total comparison runtime {time.perf_counter() - overall_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
