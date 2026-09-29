"""Merge tree_width datasets into a single JSONL with text and labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


TREE_WIDTH_DIR = Path("artifacts") / "prop_datasets" / "tree_width"
DEFAULT_POSITIVE_JSONL = TREE_WIDTH_DIR / "train.jsonl"
DEFAULT_NEGATIVE_JSONL = TREE_WIDTH_DIR / "train_false.jsonl"
DEFAULT_OUTPUT_JSONL = TREE_WIDTH_DIR / "merged_conversations.jsonl"


def read_jsonl(path: Path) -> List[list]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def split_conversation(raw_text: str) -> List[str]:
    normalized = raw_text.replace("<s>", "</s>")
    segments: List[str] = []
    for chunk in normalized.split("</s>"):
        cleaned = chunk.strip()
        if cleaned:
            segments.append(cleaned)
    return segments


def merge_records(positive_path: Path, negative_path: Path) -> List[dict]:
    merged: List[dict] = []
    for source_path, prefix in ((positive_path, "tw_pos"), (negative_path, "tw_neg")):
        records = read_jsonl(source_path)
        for idx, record in enumerate(records):
            if len(record) < 2:
                continue
            raw_text, label_str = record[0], record[1]
            pieces = split_conversation(raw_text)
            if not pieces:
                continue
            merged.append(
                {
                    "post_id": f"{prefix}_{idx}",
                    "root_text": pieces[0],
                    "replies": pieces[1:],
                    "label": int(label_str),
                }
            )
    merged.sort(key=lambda item: item["post_id"])
    return merged


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge tree_width datasets into a single JSONL file")
    parser.add_argument(
        "--positive-jsonl",
        type=Path,
        default=DEFAULT_POSITIVE_JSONL,
        help="Path to the propaganda/rumour JSONL file.",
    )
    parser.add_argument(
        "--negative-jsonl",
        type=Path,
        default=DEFAULT_NEGATIVE_JSONL,
        help="Path to the non-propaganda/non-rumour JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Destination path for the merged JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    merged = merge_records(args.positive_jsonl, args.negative_jsonl)
    if not merged:
        raise RuntimeError("No conversations were merged; check the source files.")

    write_jsonl(merged, args.output)
    print(f"Merged {len(merged)} conversations into {args.output}")


if __name__ == "__main__":
    main()
