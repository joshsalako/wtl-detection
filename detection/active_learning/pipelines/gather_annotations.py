#!/usr/bin/env python3
import os
import sys
import csv
import shutil
import argparse
import pandas as pd
from pathlib import Path

# Add active_learning to path
PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEARNING_DIR = os.path.dirname(PIPELINES_DIR)
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Gather sampled oracle images for manual annotation."
    )
    parser.add_argument(
        "--candidates_csv",
        type=str,
        required=True,
        help="Path to the AL query candidates CSV.",
    )
    parser.add_argument(
        "--cycle", type=int, required=True, help="Active learning cycle number."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type (yolo, rtdetr, faster_rcnn)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.candidates_csv):
        print(f"Error: candidates CSV not found at {args.candidates_csv}")
        sys.exit(1)

    df = pd.read_csv(args.candidates_csv)
    if df.empty:
        print("No candidates to gather.")
        return

    # Define output directory
    output_dir = os.path.join(
        ACTIVE_LEARNING_DIR, "to_annotate", f"{args.model_type}_cycle_{args.cycle}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Tracker for already sampled images
    sampled_tracker_csv = os.path.join(ACTIVE_LEARNING_DIR, "already_sampled.csv")

    gathered_count = 0
    new_sampled_records = []

    for idx, row in df.iterrows():
        img_path_str = row["image_path"]
        img_path = Path(img_path_str)

        if not img_path.exists():
            print(f"Warning: Source image not found: {img_path_str}")
            continue

        base_name = img_path.name
        dest_path = os.path.join(output_dir, base_name)

        # Handle duplicate file names
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(base_name)
            dest_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
            counter += 1

        shutil.copy2(img_path, dest_path)
        gathered_count += 1
        new_sampled_records.append(
            {
                "image_path": img_path_str,
                "cycle": args.cycle,
                "model_type": args.model_type,
            }
        )

    print(f"\nSuccessfully gathered {gathered_count} images to: {output_dir}")

    # Append to already_sampled.csv
    tracker_df = pd.DataFrame(new_sampled_records)
    if os.path.exists(sampled_tracker_csv):
        tracker_df.to_csv(sampled_tracker_csv, mode="a", header=False, index=False)
    else:
        tracker_df.to_csv(sampled_tracker_csv, index=False)

    print(f"Updated tracking list at: {sampled_tracker_csv}")


if __name__ == "__main__":
    main()
