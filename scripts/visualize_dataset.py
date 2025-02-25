"""
Visualization script for human segmentation dataset.
Displays random samples of image-mask pairs from the dataset.

Usage:
    python visualize_dataset.py --csv_path path/to/data.csv --num_samples 5
"""

import pandas as pd
import numpy as np
import argparse

from src.utils.visualization import show_original_and_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the dataset")
    parser.add_argument(
        "--csv_path",
        default="data/train.csv",
        help="Path to CSV file containing image and mask paths",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of random samples to visualize",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    print(f"\nDisplaying {args.num_samples} random samples from the dataset...")
    for _ in range(args.num_samples):
        idx = np.random.randint(0, len(df))
        print(f"\nShowing sample {idx}")
        show_original_and_mask(df, idx)


if __name__ == "__main__":
    main()
