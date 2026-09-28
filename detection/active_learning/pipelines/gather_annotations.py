#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd

# Add active_learning to path
PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEARNING_DIR = os.path.dirname(PIPELINES_DIR)
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)

try:
    from ..central_config import CLASSES
except ImportError:  # Script execution from the pipelines directory.
    from central_config import CLASSES
try:
    from .annotation_handoff import AnnotationBatchError, export_annotation_batch
except ImportError:  # Script execution from the pipelines directory.
    from annotation_handoff import AnnotationBatchError, export_annotation_batch


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
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional experiment name used to isolate annotation batches.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.candidates_csv):
        print(f"Error: candidates CSV not found at {args.candidates_csv}")
        sys.exit(1)

    df = pd.read_csv(args.candidates_csv)
    if df.empty:
        print("No candidates to gather.")
        return

    # Define output directory. The default path remains unchanged.
    output_dir = os.path.join(ACTIVE_LEARNING_DIR, "to_annotate")
    if args.experiment_name:
        output_dir = os.path.join(output_dir, args.experiment_name)
    output_dir = os.path.join(output_dir, f"{args.model_type}_cycle_{args.cycle}")

    candidates = df.to_dict("records")
    try:
        manifest = export_annotation_batch(
            candidates,
            output_dir,
            model_type=args.model_type,
            cycle=args.cycle,
            classes=CLASSES,
            candidates_csv=args.candidates_csv,
        )
    except AnnotationBatchError as exc:
        print("\nAnnotation batch export failed:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    # Tracker for already sampled images. Preserve the existing three columns.
    sampled_tracker_csv = os.path.join(ACTIVE_LEARNING_DIR, "already_sampled.csv")
    new_sampled_records = [
        {
            "image_path": image.source_path,
            "cycle": args.cycle,
            "model_type": args.model_type,
        }
        for image in manifest.images
    ]

    print(
        f"\nSuccessfully gathered {len(manifest.images)} images to: {output_dir}"
    )
    print(f"Wrote annotation manifest to: {os.path.join(output_dir, 'manifest.json')}")

    # Append to already_sampled.csv
    tracker_df = pd.DataFrame(
        new_sampled_records, columns=["image_path", "cycle", "model_type"]
    )
    if os.path.exists(sampled_tracker_csv) and not tracker_df.empty:
        tracker_df.to_csv(sampled_tracker_csv, mode="a", header=False, index=False)
    elif not os.path.exists(sampled_tracker_csv):
        tracker_df.to_csv(sampled_tracker_csv, index=False)

    print(f"Updated tracking list at: {sampled_tracker_csv}")


if __name__ == "__main__":
    main()
