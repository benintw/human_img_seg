#!/bin/bash

# path to csv file
CSV_PATH="data/train.csv"

# number of samples to visualize
NUM_SAMPLES=5

python scripts/visualize_dataset.py --csv_path "$CSV_PATH" --num_samples "$NUM_SAMPLES"
