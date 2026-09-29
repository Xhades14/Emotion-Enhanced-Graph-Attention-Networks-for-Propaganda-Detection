"""Quick smoke test for the English emotion detection model."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_DIR = Path("artifacts") / "models" / "eng_emo_model" / "models"
DEFAULT_TEXTS = [
    "I can't believe they betrayed us—I'm furious and shaking.",
    "This is the best news I've heard all week, I'm so happy!",
    "Honestly I'm just tired and a little worried about what's coming next.",
]

GOEMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

EKMAN_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
EKMAN_TO_FINE = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": [
        "admiration",
        "amusement",
        "approval",
        "caring",
        "desire",
        "excitement",
        "gratitude",
        "joy",
        "love",
        "optimism",
        "pride",
        "relief",
    ],
    "sadness": ["disappointment", "embarrassment", "grief", "remorse", "sadness"],
    "surprise": ["confusion", "curiosity", "realization", "surprise"],
    "neutral": ["neutral"],
}

LABEL_OVERRIDES = {str(idx): name for idx, name in enumerate(GOEMOTIONS_LABELS)}
EKMAN_TO_INDICES = {
    group: [GOEMOTIONS_LABELS.index(name) for name in names]
    for group, names in EKMAN_TO_FINE.items()
}


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
        probs = torch.sigmoid(outputs.logits)
    return probs.cpu()


def collapse_to_ekman(prob_vector: torch.Tensor) -> torch.Tensor:
    collapsed = []
    for label in EKMAN_LABELS:
        indices = EKMAN_TO_INDICES[label]
        if indices:
            values = prob_vector[indices]
            collapsed.append(values.max())
        else:
            collapsed.append(prob_vector.new_tensor(0.0))
    stacked = torch.stack(collapsed)
    total = stacked.sum()
    if total.item() > 0:
        stacked = stacked / total
    return stacked


def main() -> None:
    parser = argparse.ArgumentParser(description="Run emotion detector on sample sentences")
    parser.add_argument(
        "texts",
        nargs="*",
        help="Texts to analyse. If omitted, a small default list is used.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to the emotion model directory.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of highest-probability emotions to display per sentence.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Emotion model directory not found: {model_dir}")

    texts = args.texts if args.texts else DEFAULT_TEXTS

    tokenizer, model = load_model(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    probs = predict(texts, tokenizer, model, device)

    id2label = model.config.id2label or {index: f"LABEL_{index}" for index in range(probs.shape[1])}

    for text, scores in zip(texts, probs):
        print("=" * 80)
        print(f"Text: {text}")
        topk = min(args.topk, scores.numel())
        sorted_indices = torch.topk(scores, k=topk).indices.tolist()
        print("- Top fine-grained emotions:")
        for idx in sorted_indices:
            base_label = id2label.get(str(idx), id2label.get(idx, f"LABEL_{idx}"))
            label = LABEL_OVERRIDES.get(str(idx), base_label)
            print(f"  {label:>12}: {scores[idx]:.2%}")
        ekman_scores = collapse_to_ekman(scores)
        print("- Ekman clusters:")
        for label, value in zip(EKMAN_LABELS, ekman_scores):
            print(f"  {label:>12}: {value.item():.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
