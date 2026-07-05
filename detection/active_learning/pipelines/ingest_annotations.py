#!/usr/bin/env python3
import os
import sys
import cv2
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEARNING_DIR = os.path.dirname(PIPELINES_DIR)
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)


def load_state(state_file):
    with open(state_file, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest annotations for the next AL cycle."
    )
    parser.add_argument(
        "--annotated_dir",
        type=str,
        required=True,
        help="Path to the annotated folder (e.g., annotated/yolo_cycle_0)",
    )
    args = parser.parse_args()

    annotated_dir = os.path.abspath(args.annotated_dir)
    if not os.path.exists(annotated_dir):
        print(f"Error: Annotated directory not found at {annotated_dir}")
        sys.exit(1)

    dir_name = os.path.basename(annotated_dir)
    # Expected format: {model_type}_cycle_{cycle}
    # e.g. yolo_cycle_0, faster_rcnn_cycle_0, rtdetr_cycle_0

    parts = dir_name.rsplit("_cycle_", 1)
    if len(parts) != 2:
        print(
            f"Error: Annotated directory name must follow the pattern 'model_type_cycle_N', got '{dir_name}'"
        )
        sys.exit(1)

    m_type = parts[0]
    try:
        current_cycle = int(parts[1])
    except ValueError:
        print(f"Error: Could not parse cycle number from '{parts[1]}'")
        sys.exit(1)

    next_cycle = current_cycle + 1
    print("\n=======================================================")
    print(f"INGESTING ANNOTATIONS FOR {m_type.upper()}")
    print(f"  Detected Model Type: {m_type}")
    print(f"  Detected Current Cycle: {current_cycle}")
    print(f"  Target Next Cycle: {next_cycle}")
    print("=======================================================\n")

    state_file = os.path.join(PIPELINES_DIR, f"al_state_{m_type}_clahe_pretrained.json")
    if os.path.exists(state_file):
        state = load_state(state_file)
        if state["cycle"] < next_cycle:
            print(
                f"Warning: State file indicates the loop is at cycle {state['cycle']}, but we are ingesting data to create cycle {next_cycle}."
            )
    else:
        print(f"Warning: State file {state_file} not found.")

    src_dataset_dir = os.path.join(
        ACTIVE_LEARNING_DIR,
        "data",
        f"{m_type}_clahe",
        "pretrained",
        f"cycle_{current_cycle}",
    )
    dst_dataset_dir = os.path.join(
        ACTIVE_LEARNING_DIR,
        "data",
        f"{m_type}_clahe",
        "pretrained",
        f"cycle_{next_cycle}",
    )

    if not os.path.exists(src_dataset_dir):
        print(
            f"Error: Source cycle {current_cycle} dataset not found at {src_dataset_dir}"
        )
        sys.exit(1)

    if os.path.exists(dst_dataset_dir):
        print(
            f"Warning: Target cycle {next_cycle} dataset already exists at {dst_dataset_dir}. Merging into it."
        )
    else:
        print(f"Copying dataset from cycle {current_cycle} to cycle {next_cycle}...")
        shutil.copytree(src_dataset_dir, dst_dataset_dir, dirs_exist_ok=True)

    # Images and Labels directories inside train
    # Handling both standard YOLO format and custom structure
    train_dir = os.path.join(dst_dataset_dir, "train")
    if not os.path.exists(train_dir):
        # Fallback to YOLO generic structure if 'train' folder doesn't exist
        img_dest_dir = os.path.join(dst_dataset_dir, "images", "train")
        lbl_dest_dir = os.path.join(dst_dataset_dir, "labels", "train")
    else:
        img_dest_dir = os.path.join(train_dir, "images")
        lbl_dest_dir = os.path.join(train_dir, "labels")

    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(lbl_dest_dir, exist_ok=True)

    src_images_dir = os.path.join(annotated_dir, "images")
    src_labels_dir = os.path.join(annotated_dir, "labels")

    if not os.path.exists(src_images_dir) or not os.path.exists(src_labels_dir):
        print(
            f"Error: The annotated directory must contain 'images' and 'labels' subfolders."
        )
        sys.exit(1)

    images = [
        f
        for f in os.listdir(src_images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(
        f"\nMerging {len(images)} annotated images into cycle {next_cycle} dataset..."
    )

    for img_name in tqdm(images, desc="Ingesting Annotations"):
        img_src_path = os.path.join(src_images_dir, img_name)
        img_dst_path = os.path.join(img_dest_dir, img_name)

        shutil.copy2(img_src_path, img_dst_path)

        # Copy label
        base_name = os.path.splitext(img_name)[0]
        lbl_src_path = os.path.join(src_labels_dir, base_name + ".txt")
        lbl_dst_path = os.path.join(lbl_dest_dir, base_name + ".txt")

        if os.path.exists(lbl_src_path):
            shutil.copy2(lbl_src_path, lbl_dst_path)

    # Clear labels cache to force rebuild during next training
    cache_path = os.path.join(lbl_dest_dir, "..", "labels.cache")
    if os.path.exists(cache_path):
        os.remove(cache_path)
    cache_path2 = (
        os.path.join(train_dir, "labels.cache") if os.path.exists(train_dir) else None
    )
    if cache_path2 and os.path.exists(cache_path2):
        os.remove(cache_path2)

    print("\n=======================================================")
    print(f"INGESTION COMPLETE: {m_type.upper()}")
    print(f"  Merged Data into: cycle_{next_cycle}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
