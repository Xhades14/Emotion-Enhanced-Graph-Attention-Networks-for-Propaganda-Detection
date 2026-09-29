"""Utility script to run the English propaganda detection model on sample texts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_DIR = Path("artifacts") / "models" / "eng_prop_model" / "SemEval_Trained_Intermediate(final)"
DEFAULT_TEXTS = [
    "BREAKING: Opposition leaders are secretly plotting to sell India's natural resources to foreign powers—share before they silence us!",
    "Forward this now: the central government's vaccine drive is actually a foreign plan to make Indians infertile!",
    "Proof leaked: the ruling party pays influencers to brainwash citizens and erase real news across India!",
]


def load_model(model_dir: Path):
    """Load tokenizer and sequence classification model from disk."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model


def predict(texts: List[str], tokenizer, model, device: torch.device):
    """Run model predictions and return probabilities for each label."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run propaganda detector on input texts")
    parser.add_argument(
        "texts",
        nargs="*",
        help="Texts to classify. If omitted, default examples are used.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to the propaganda model directory.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    texts = args.texts if args.texts else DEFAULT_TEXTS

    tokenizer, model = load_model(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    probs = predict(texts, tokenizer, model, device)

    id2label = model.config.id2label or {index: f"LABEL_{index}" for index in range(probs.shape[1])}

    for text, scores in zip(texts, probs):
        predicted_index = int(torch.argmax(scores).item())
        predicted_label = id2label.get(predicted_index, f"LABEL_{predicted_index}")
        score_pairs = " | ".join(
            f"{id2label.get(i, f'LABEL_{i}')}: {score:.2%}" for i, score in enumerate(scores)
        )
        print("=" * 80)
        print(f"Text: {text}")
        print(f"Prediction: {predicted_label}")
        print(f"Scores: {score_pairs}")
    print("=" * 80)


if __name__ == "__main__":
    main()
