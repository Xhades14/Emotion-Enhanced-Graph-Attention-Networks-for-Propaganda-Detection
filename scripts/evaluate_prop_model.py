"""Evaluate the propaganda detection model on a labeled CSV dataset."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_DATASET_PATH = Path("artifacts") / "prop_datasets" / "covid_3k_labeled.csv"
DEFAULT_TEXT_COLUMN = "tweet"
DEFAULT_LABEL_COLUMN = "labels"
DEFAULT_BATCH_SIZE = 32
DEFAULT_POSITIVE_INDEX = 1
DEFAULT_MODEL_DIR = Path("artifacts") / "models" / "eng_prop_model" / "SemEval_Trained_Intermediate(final)"


def batch_iterable(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model


def predict(texts: List[str], tokenizer, model, device: torch.device):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate propaganda model on labeled CSV")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to CSV file containing columns for tweet text and labels.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Name of the column with tweet text.",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help="Name of the column with labels (-1 for non-propaganda, otherwise propaganda).",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory with the fine-tuned propaganda model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of examples to process per forward pass.",
    )
    parser.add_argument(
        "--positive-index",
        type=int,
        default=DEFAULT_POSITIVE_INDEX,
        help="Model class index that corresponds to propaganda.",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path, text_column: str, label_column: str) -> tuple[List[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise KeyError(f"Missing text column '{text_column}' in {csv_path}")
    if label_column not in df.columns:
        raise KeyError(f"Missing label column '{label_column}' in {csv_path}")

    texts = df[text_column].astype(str).tolist()
    labels = (
        df[label_column]
        .astype(str)
        .str.strip()
        .apply(lambda value: 0 if value == "-1" else 1)
        .to_numpy(dtype=np.int64)
    )
    return texts, labels


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    texts, labels = load_dataset(csv_path, args.text_column, args.label_column)

    tokenizer, model = load_model(Path(args.model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_probs = []
    for chunk in batch_iterable(texts, args.batch_size):
        probs_chunk = predict(chunk, tokenizer, model, device)
        all_probs.append(probs_chunk)
    probs_tensor = torch.cat(all_probs, dim=0)

    predicted_indices = probs_tensor.argmax(dim=1).cpu().numpy()
    y_pred = (predicted_indices == args.positive_index).astype(np.int64)

    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred, zero_division=0)
    recall = recall_score(labels, y_pred, zero_division=0)
    f1 = f1_score(labels, y_pred, zero_division=0)
    cm = confusion_matrix(labels, y_pred)

    print("Evaluation on", csv_path)
    print("Examples:", len(labels))
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm)

    id2label = model.config.id2label or {index: f"LABEL_{index}" for index in range(probs_tensor.shape[1])}
    target_names = [
        id2label.get(0, "NON_PROPAGANDA"),
        id2label.get(args.positive_index, "PROPAGANDA"),
    ]
    print("\nClassification report:")
    print(classification_report(labels, y_pred, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    main()
