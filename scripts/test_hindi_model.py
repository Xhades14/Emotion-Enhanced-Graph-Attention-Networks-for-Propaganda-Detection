"""Run inference with the Hindi LoRA adapter on the propaganda classifier."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

BASE_MODEL_DIR = Path("artifacts") / "models" / "eng_prop_model" / "SemEval_Trained_Intermediate(final)"
ADAPTER_DIR = Path("artifacts") / "models" / "hprop-lora-adapter"

DEFAULT_TEXTS = [
    "प्रधानमंत्री ने कहा कि देश का मीडिया झूठ फैला रहा है।",
    "यह वैज्ञानिक रिपोर्ट बताती है कि वैक्सीन सुरक्षित है।",
    "खबर आई है कि विपक्ष देश तोड़ने की साजिश कर रहा है, इसे तुरंत शेयर करें!",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hindi LoRA-adapted propaganda detector")
    parser.add_argument(
        "--base-model",
        type=Path,
        default=BASE_MODEL_DIR,
        help="Path to the base SemEval propaganda model",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ADAPTER_DIR,
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs="*",
        default=None,
        help="Input texts to score (space separated). Defaults to built-in Hindi samples.",
    )
    parser.add_argument(
        "--json-input",
        type=Path,
        default=None,
        help="Optional JSON file containing a list of texts.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    return parser.parse_args()


def load_texts(text_args: List[str] | None, json_path: Path | None) -> List[str]:
    if json_path:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of strings")
        return [str(item) for item in data]
    if text_args:
        return text_args
    return DEFAULT_TEXTS


def load_model(base_model: Path, adapter_dir: Path, device: str):
    if not base_model.exists():
        raise FileNotFoundError(f"Base model path {base_model} does not exist")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter path {adapter_dir} does not exist")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict(texts: List[str], tokenizer, model, device: str, batch_size: int) -> torch.Tensor:
    probabilities: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
        probabilities.append(probs.cpu())
    return torch.cat(probabilities, dim=0)


def main() -> None:
    args = parse_args()
    texts = load_texts(args.texts, args.json_input)
    if not texts:
        raise ValueError("No texts provided for inference")

    tokenizer, model = load_model(args.base_model, args.adapter, args.device)
    probabilities = predict(texts, tokenizer, model, args.device, args.batch_size)
    id2label = model.config.id2label or {index: f"LABEL_{index}" for index in range(probabilities.shape[1])}

    print("Predictions (higher = more propaganda-like):")
    for text, scores in zip(texts, probabilities.tolist()):
        predicted_index = int(torch.argmax(torch.tensor(scores)).item())
        label = id2label.get(predicted_index, f"LABEL_{predicted_index}")
        prop_score = scores[1] if len(scores) > 1 else scores[0]
        print("=" * 80)
        print(f"Text: {text}")
        score_pairs = " | ".join(
            f"{id2label.get(i, f'LABEL_{i}')}: {score:.2%}"
            for i, score in enumerate(scores)
        )
        print(f"Predicted label: {label}")
        print(f"Score (propaganda class): {prop_score:.4f}")
        print(f"Scores: {score_pairs}")
    print("=" * 80)


if __name__ == "__main__":
    main()
