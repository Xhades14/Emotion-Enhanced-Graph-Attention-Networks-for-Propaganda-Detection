"""Generate meta-classifier features from propaganda and emotion models."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TREE_WIDTH_DIR = Path("artifacts") / "prop_datasets" / "tree_width"
MERGED_JSONL = TREE_WIDTH_DIR / "merged_conversations.jsonl"
FEATURE_CSV = TREE_WIDTH_DIR / "meta_features.csv"
PROP_MODEL_DIR = Path("artifacts") / "models" / "eng_prop_model" / "SemEval_Trained_Intermediate(final)"
EMO_MODEL_DIR = Path("artifacts") / "models" / "eng_emo_model" / "models"

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
EKMAN_TO_INDICES = {
	cluster: [GOEMOTIONS_LABELS.index(label) for label in fine]
	for cluster, fine in EKMAN_TO_FINE.items()
}


@dataclass
class Conversation:
	post_id: str
	root_text: str
	replies: List[str]
	label: int


class ProgressLogger:
	def __init__(self, total: int, label: str):
		self.total = max(total, 1)
		self.label = label
		self.count = 0
		self.start = time.perf_counter()

	def update(self, increment: int) -> None:
		self.count += increment
		elapsed = time.perf_counter() - self.start
		rate = self.count / elapsed if elapsed > 0 else 0.0
		remaining = (self.total - self.count) / rate if rate > 0 else float("inf")
		percent = min(self.count / self.total, 1.0)
		eta_str = f"{remaining:.1f}s" if remaining != float("inf") else "?"
		print(
			f"[{self.label}] {self.count}/{self.total} ({percent:.0%}) elapsed={elapsed:.1f}s eta={eta_str}",
			flush=True,
		)

	def finish(self) -> None:
		elapsed = time.perf_counter() - self.start
		print(f"[{self.label}] done in {elapsed:.1f}s", flush=True)


def read_jsonl(path: Path) -> List[Conversation]:
	conversations: List[Conversation] = []
	with path.open("r", encoding="utf-8") as fh:
		for line in fh:
			data = json.loads(line)
			conversations.append(
				Conversation(
					post_id=data["post_id"],
					root_text=data["root_text"],
					replies=data.get("replies", []),
					label=int(data["label"]),
				)
			)
	conversations.sort(key=lambda item: item.post_id)
	return conversations


def batch_texts(texts: List[str], batch_size: int) -> Iterable[List[str]]:
	for start in range(0, len(texts), batch_size):
		yield texts[start : start + batch_size]


def load_prop_model(model_dir: Path):
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	model = AutoModelForSequenceClassification.from_pretrained(model_dir)
	return tokenizer, model


def load_emo_model(model_dir: Path):
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	model = AutoModelForSequenceClassification.from_pretrained(model_dir)
	return tokenizer, model


def get_prop_probabilities(
	texts: List[str],
	tokenizer,
	model,
	device: torch.device,
	batch_size: int = 32,
	progress_label: str | None = None,
) -> np.ndarray:
	outputs: List[torch.Tensor] = []
	model.eval()
	logger = ProgressLogger(len(texts), progress_label) if progress_label else None
	for chunk in batch_texts(texts, batch_size):
		encoded = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
		encoded = {key: value.to(device) for key, value in encoded.items()}
		with torch.no_grad():
			logits = model(**encoded).logits
			probs = torch.softmax(logits, dim=-1)[:, 1]
		outputs.append(probs.cpu())
		if logger:
			logger.update(len(chunk))
	if logger:
		logger.finish()
	return torch.cat(outputs, dim=0).numpy()


def get_emotion_probabilities(
	texts: List[str],
	tokenizer,
	model,
	device: torch.device,
	batch_size: int = 32,
	progress_label: str | None = None,
) -> np.ndarray:
	if not texts:
		return np.zeros((0, len(GOEMOTIONS_LABELS)), dtype=np.float32)

	outputs: List[torch.Tensor] = []
	model.eval()
	logger = ProgressLogger(len(texts), progress_label) if progress_label else None
	for chunk in batch_texts(texts, batch_size):
		encoded = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
		encoded = {key: value.to(device) for key, value in encoded.items()}
		with torch.no_grad():
			logits = model(**encoded).logits
			probs = torch.sigmoid(logits)
		outputs.append(probs.cpu())
		if logger:
			logger.update(len(chunk))
	if logger:
		logger.finish()
	return torch.cat(outputs, dim=0).numpy()


def collapse_to_ekman(vector: np.ndarray) -> np.ndarray:
	collapsed = np.zeros(len(EKMAN_LABELS), dtype=np.float32)
	for idx, label in enumerate(EKMAN_LABELS):
		indices = EKMAN_TO_INDICES[label]
		if indices:
			collapsed[idx] = float(np.max(vector[indices]))
	total = collapsed.sum()
	if total > 0:
		collapsed /= total
	return collapsed


def aggregate_emotions(conversations: List[Conversation], emotion_probs: np.ndarray, reply_index: List[int]) -> pd.DataFrame:
	grouped: defaultdict[int, List[np.ndarray]] = defaultdict(list)
	for row_idx, conv_idx in enumerate(reply_index):
		grouped[conv_idx].append(collapse_to_ekman(emotion_probs[row_idx]))

	rows = []
	for conv_idx, conversation in enumerate(conversations):
		vectors = grouped.get(conv_idx, [])
		if vectors:
			stack = np.vstack(vectors)
			mean_vec = stack.mean(axis=0)
			variance = float(stack.var(axis=0).mean())
		else:
			mean_vec = np.zeros(len(EKMAN_LABELS), dtype=np.float32)
			variance = 0.0

		total = mean_vec.sum()
		if total > 0:
			normalized = mean_vec / total
			entropy = float(-np.sum(normalized * np.log(normalized + 1e-12)))
		else:
			entropy = 0.0

		row = {
			"post_id": conversation.post_id,
			"entropy": entropy,
			"variance": variance,
		}
		for idx, label in enumerate(EKMAN_LABELS):
			row[f"mean_{label}"] = float(mean_vec[idx])
		rows.append(row)

	return pd.DataFrame(rows)


def build_feature_table(
	conversations: List[Conversation],
	prop_model_dir: Path,
	emo_model_dir: Path,
	batch_size: int,
) -> pd.DataFrame:
	tokenizer_prop, model_prop = load_prop_model(prop_model_dir)
	tokenizer_emo, model_emo = load_emo_model(emo_model_dir)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_prop = model_prop.to(device)
	model_emo = model_emo.to(device)

	root_texts = [conv.root_text for conv in conversations]
	print(f"Running propaganda model on {len(root_texts)} root posts", flush=True)
	prop_probs = get_prop_probabilities(
		root_texts,
		tokenizer_prop,
		model_prop,
		device,
		batch_size=batch_size,
		progress_label="Propaganda",
	)

	reply_texts: List[str] = []
	reply_owner: List[int] = []
	for idx, conv in enumerate(conversations):
		for reply in conv.replies:
			reply_texts.append(reply)
			reply_owner.append(idx)

	print(f"Running emotion model on {len(reply_texts)} replies", flush=True)
	emotion_probs = get_emotion_probabilities(
		reply_texts,
		tokenizer_emo,
		model_emo,
		device,
		batch_size=batch_size,
		progress_label="Emotion",
	)
	emotion_features = aggregate_emotions(conversations, emotion_probs, reply_owner)

	feature_table = pd.DataFrame(
		{
			"post_id": [conv.post_id for conv in conversations],
			"text_pred": prop_probs,
			"true_label": [conv.label for conv in conversations],
		}
	)

	feature_table = feature_table.merge(emotion_features, on="post_id", how="left")
	feature_table.fillna(0.0, inplace=True)
	return feature_table


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Build meta-classifier dataset from merged conversations")
	parser.add_argument(
		"--merged-jsonl",
		type=Path,
		default=MERGED_JSONL,
		help="Path to the merged conversations JSONL file.",
	)
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=FEATURE_CSV,
		help="Destination CSV file for the meta features.",
	)
	parser.add_argument(
		"--prop-model-dir",
		type=Path,
		default=PROP_MODEL_DIR,
		help="Path to the propaganda model directory.",
	)
	parser.add_argument(
		"--emo-model-dir",
		type=Path,
		default=EMO_MODEL_DIR,
		help="Path to the emotion model directory.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=32,
		help="Batch size to use for model inference.",
	)
	return parser.parse_args()


def main() -> None:
	overall_start = time.perf_counter()
	args = parse_args()

	conversations = read_jsonl(args.merged_jsonl)
	if not conversations:
		raise RuntimeError("Merged JSONL file is empty; run merge_tree_width.py first.")
	print(f"Loaded {len(conversations)} conversations", flush=True)

	feature_table = build_feature_table(
		conversations,
		prop_model_dir=args.prop_model_dir,
		emo_model_dir=args.emo_model_dir,
		batch_size=args.batch_size,
	)

	args.output_csv.parent.mkdir(parents=True, exist_ok=True)
	feature_table.to_csv(args.output_csv, index=False)
	print(f"Saved meta-feature table with {len(feature_table)} rows to {args.output_csv}")
	elapsed = time.perf_counter() - overall_start
	print(f"Total runtime {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
	main()

