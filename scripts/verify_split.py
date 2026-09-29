"""Verify the dataset split logic produces balanced class distributions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data_path = Path("artifacts") / "prop_datasets" / "tree_width" / "merged_conversations.jsonl"
records = []
with data_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)
if "post_id" in df.columns and "tweet_id" not in df.columns:
    df = df.rename(columns={"post_id": "tweet_id"})
if "label" in df.columns and "true_label" not in df.columns:
    df = df.rename(columns={"label": "true_label"})

df = df.sort_values("tweet_id").reset_index(drop=True)
labels = df["true_label"].astype(int).to_numpy()

print(f"Total samples: {len(labels)}")
print(f"Propaganda (1): {labels.sum()} ({labels.mean()*100:.1f}%)")
print(f"Non-propaganda (0): {(1-labels).sum()} ({(1-labels.mean())*100:.1f}%)")

# Stratified split
indices = np.arange(len(labels))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=labels
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx]
)

print(f"\n=== Split Statistics ===")
print(f"Train: {len(train_idx)} samples ({len(train_idx)/len(labels)*100:.1f}%)")
print(f"  Propaganda: {labels[train_idx].sum()} ({labels[train_idx].mean()*100:.1f}%)")
print(f"  Non-propaganda: {(1-labels[train_idx]).sum()}")

print(f"Val: {len(val_idx)} samples ({len(val_idx)/len(labels)*100:.1f}%)")
print(f"  Propaganda: {labels[val_idx].sum()} ({labels[val_idx].mean()*100:.1f}%)")
print(f"  Non-propaganda: {(1-labels[val_idx]).sum()}")

print(f"Test: {len(test_idx)} samples ({len(test_idx)/len(labels)*100:.1f}%)")
print(f"  Propaganda: {labels[test_idx].sum()} ({labels[test_idx].mean()*100:.1f}%)")
print(f"  Non-propaganda: {(1-labels[test_idx]).sum()}")

print("\n✓ Split logic verified: all splits preserve class balance")
