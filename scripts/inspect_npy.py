"""Inspect and optionally convert a pickled NumPy .npy file to a readable format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_NPY_PATH = Path("artifacts") / "prop_datasets" / "time_order" / "train_false.npy"


def to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types into JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return [to_serializable(item) for item in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): to_serializable(value) for key, value in obj.items()}
    return obj


def summarize_array(array: np.ndarray, limit: int) -> None:
    print(f"type: {type(array)}")
    if hasattr(array, "shape"):
        print(f"shape: {array.shape}")
    else:
        print("shape: n/a")
    print(f"dtype: {getattr(array, 'dtype', 'n/a')}")
    length = len(array) if hasattr(array, "__len__") else "n/a"
    print(f"length: {length}")

    if isinstance(length, int):
        sample_count = min(limit, length)
        print(f"\nShowing first {sample_count} entries:")
        for idx in range(sample_count):
            print(f"--- entry {idx} ---")
            print(to_serializable(array[idx]))
    else:
        print("Array does not expose a length; nothing to preview.")


def write_jsonl(array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for item in array:
            json.dump(to_serializable(item), fh, ensure_ascii=False)
            fh.write("\n")
    print(f"Wrote {len(array)} entries to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or export data from a NumPy .npy file")
    parser.add_argument(
        "--npy-path",
        type=Path,
        default=DEFAULT_NPY_PATH,
        help="Path to the .npy file (expects allow_pickle=True data).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of entries to display in the console preview.",
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        help="Optional path to write the converted data as JSON Lines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npy_path: Path = args.npy_path
    if not npy_path.exists():
        raise FileNotFoundError(f"File not found: {npy_path}")

    array = np.load(npy_path, allow_pickle=True)
    summarize_array(array, limit=args.preview)

    if args.jsonl_output:
        write_jsonl(array, args.jsonl_output)


if __name__ == "__main__":
    main()
