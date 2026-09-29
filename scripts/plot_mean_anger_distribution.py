"""Describe emotion distributions by label and compute correlations."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FEATURE_CSV = Path("artifacts") / "prop_datasets" / "tree_width" / "meta_features.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize emotion distributions and correlations")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=FEATURE_CSV,
        help="Path to meta feature CSV (from build_meta_dataset.py)",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV not found at {path}; run build_meta_dataset.py")
    table = pd.read_csv(path)
    missing = {"mean_anger", "mean_surprise", "true_label"} - set(table.columns)
    if missing:
        raise ValueError(f"meta_features.csv missing columns: {sorted(missing)}")
    return table


def describe_distribution(table: pd.DataFrame, column: str) -> None:
    group_stats = (
        table.groupby("true_label")[column]
        .agg(["count", "mean", "std", "min", "max"])
        .rename(columns={"count": "n"})
    )
    print(f"Group-wise summary ({column} by true_label):")
    print(group_stats)

    overall_stats = table[column].describe()
    print(f"\nOverall distribution of {column}:")
    print(overall_stats)

    corr = table[[column, "true_label"]].corr(method="pearson")
    print(f"\nPearson correlation between {column} and true_label:")
    print(corr.loc[column, "true_label"])

    cov = table[[column, "true_label"]].cov()
    print("\nCovariance:")
    print(cov.loc[column, "true_label"])

    try:
        from scipy.stats import pointbiserialr

        r, p = pointbiserialr(table["true_label"], table[column])
        print(f"\nPoint-biserial correlation (true_label vs {column}):")
        print(f"r={r:.4f}, p={p:.4e}")
    except ImportError:
        print("\nscipy not installed; skipping point-biserial correlation")


def main() -> None:
    args = parse_args()
    table = load_table(args.features_csv)
    for column in ("mean_anger", "mean_surprise"):
        describe_distribution(table, column)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
