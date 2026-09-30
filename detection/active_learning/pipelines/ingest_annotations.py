#!/usr/bin/env python3
"""Validate and ingest one completed active-learning annotation batch."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

try:
    import cv2

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except ImportError:  # pragma: no cover - existing runtime dependency
    cv2 = None

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEARNING_DIR = os.path.dirname(PIPELINES_DIR)
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)

try:
    from .annotation_handoff import (
        AnnotationBatchError,
        ValidationIssue,
        validate_annotation_batch,
    )
except ImportError:  # Script execution from the pipelines directory.
    from annotation_handoff import (
        AnnotationBatchError,
        ValidationIssue,
        validate_annotation_batch,
    )
try:
    from .dataset_utils import load_original_classes
except ImportError:  # Script execution from the pipelines directory.
    from dataset_utils import load_original_classes


def load_state(state_file: str | Path) -> dict:
    with Path(state_file).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_batch_name(annotated_dir: Path) -> tuple[str, int]:
    parts = annotated_dir.name.rsplit("_cycle_", 1)
    if len(parts) != 2:
        raise ValueError(
            "Annotated directory name must follow the pattern "
            f"'model_type_cycle_N', got '{annotated_dir.name}'"
        )
    try:
        return parts[0], int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Could not parse cycle number from '{parts[1]}'") from exc


def _dataset_dir(
    active_learning_dir: Path,
    model_type: str,
    cycle: int,
    experiment_name: str | None,
) -> Path:
    path = active_learning_dir / "data" / f"{model_type}_clahe" / "pretrained"
    if experiment_name:
        path /= experiment_name
    return path / f"cycle_{cycle}"


def ingest_annotation_batch(
    annotated_dir: str | Path,
    *,
    experiment_name: str | None = None,
    active_learning_dir: str | Path | None = None,
) -> Path:
    """Validate a batch completely, then merge it into the next dataset cycle."""

    annotated_path = Path(annotated_dir).expanduser().resolve()
    if not annotated_path.is_dir():
        raise FileNotFoundError(f"Annotated directory not found at {annotated_path}")
    model_type, current_cycle = _parse_batch_name(annotated_path)
    next_cycle = current_cycle + 1
    root = Path(active_learning_dir or ACTIVE_LEARNING_DIR).expanduser().resolve()

    state_file = root / "pipelines" / f"al_state_{model_type}_clahe_pretrained.json"
    if experiment_name:
        state_file = (
            root
            / "pipelines"
            / (f"al_state_{model_type}_clahe_pretrained_{experiment_name}.json")
        )
    state = load_state(state_file) if state_file.is_file() else None
    if state is None:
        print(f"Warning: State file {state_file} not found.")

    src_dataset_dir = _dataset_dir(root, model_type, current_cycle, experiment_name)
    dst_dataset_dir = _dataset_dir(root, model_type, next_cycle, experiment_name)
    if not src_dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Source cycle {current_cycle} dataset not found at {src_dataset_dir}"
        )

    expected_classes = load_original_classes(str(src_dataset_dir))
    validated = validate_annotation_batch(
        annotated_path,
        expected_model_type=model_type,
        expected_cycle=current_cycle,
        expected_classes=expected_classes,
        expected_state_cycle=(state.get("cycle") if state is not None else None),
    )

    print("\n=======================================================")
    print(f"INGESTING ANNOTATIONS FOR {model_type.upper()}")
    print(f"  Detected Model Type: {model_type}")
    print(f"  Detected Current Cycle: {current_cycle}")
    print(f"  Target Next Cycle: {next_cycle}")
    print(f"  Validated Images: {len(validated.images)}")
    print("=======================================================\n")

    if state is not None and state.get("cycle", next_cycle) != next_cycle:
        # Defensive guard; validate_annotation_batch normally catches this first.
        raise AnnotationBatchError(
            [
                ValidationIssue(
                    "cycle_mismatch",
                    state_file,
                    "State file cycle does not match target next cycle",
                )
            ]
        )

    if dst_dataset_dir.exists():
        print(
            f"Warning: Target cycle {next_cycle} dataset already exists at "
            f"{dst_dataset_dir}. Merging into it."
        )
    else:
        print(f"Copying dataset from cycle {current_cycle} to cycle {next_cycle}...")
        shutil.copytree(src_dataset_dir, dst_dataset_dir, dirs_exist_ok=True)

    train_dir = dst_dataset_dir / "train"
    if train_dir.is_dir():
        img_dest_dir = train_dir / "images"
        lbl_dest_dir = train_dir / "labels"
    else:
        img_dest_dir = dst_dataset_dir / "images" / "train"
        lbl_dest_dir = dst_dataset_dir / "labels" / "train"
    img_dest_dir.mkdir(parents=True, exist_ok=True)
    lbl_dest_dir.mkdir(parents=True, exist_ok=True)

    src_images_dir = annotated_path / "images"
    src_labels_dir = annotated_path / "labels"
    print(
        f"\nMerging {len(validated.images)} annotated images into cycle {next_cycle}..."
    )
    for image in tqdm(validated.images, desc="Ingesting Annotations"):
        shutil.copy2(src_images_dir / image.file_name, img_dest_dir / image.file_name)
        label_name = f"{Path(image.file_name).stem}.txt"
        shutil.copy2(src_labels_dir / label_name, lbl_dest_dir / label_name)

    cache_path = lbl_dest_dir.parent / "labels.cache"
    if cache_path.exists():
        cache_path.unlink()
    cache_path2 = train_dir / "labels.cache" if train_dir.is_dir() else None
    if cache_path2 is not None and cache_path2.exists():
        cache_path2.unlink()

    print("\n=======================================================")
    print(f"INGESTION COMPLETE: {model_type.upper()}")
    print(f"  Merged Data into: cycle_{next_cycle}")
    print("=======================================================\n")
    return dst_dataset_dir


def main():
    parser = argparse.ArgumentParser(
        description="Ingest validated annotations for the next AL cycle."
    )
    parser.add_argument(
        "--annotated_dir",
        type=str,
        required=True,
        help="Path to the annotated folder (e.g., annotated/yolo_cycle_0)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional experiment name used to select an isolated dataset cycle.",
    )
    args = parser.parse_args()

    try:
        ingest_annotation_batch(
            args.annotated_dir,
            experiment_name=args.experiment_name,
        )
    except AnnotationBatchError as exc:
        print("\nAnnotation batch validation failed:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
