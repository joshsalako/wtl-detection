import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm


def main():
    CONSENSUS_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(CONSENSUS_DIR, "outputs")
    YOLO_LABELS_DIR = os.path.join(OUTPUT_DIR, "yolo_labels")
    DATASET_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "dataset_consensus")
    RAW_BASE_DIR = "/srv/shared_leopard_toad"

    if not os.path.exists(YOLO_LABELS_DIR):
        print(f"Error: {YOLO_LABELS_DIR} does not exist.")
        return

    # Find all generated label files
    label_files = glob.glob(os.path.join(YOLO_LABELS_DIR, "**/*.txt"), recursive=True)

    if not label_files:
        print("No label files found!")
        return

    print(f"Found {len(label_files)} label files. Constructing dataset structure...")

    # Create directories
    for split in ["test", "val"]:
        os.makedirs(os.path.join(DATASET_OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_OUTPUT_DIR, split, "labels"), exist_ok=True)

    copied_count = 0
    missing_images = 0

    for label_path in tqdm(label_files):
        # Infer the relative path from the label path
        rel_path = os.path.relpath(label_path, YOLO_LABELS_DIR)

        # Determine split based on camera
        if "5Z" in rel_path:
            split = "test"
        elif "4R" in rel_path:
            split = "val"
        else:
            # Fallback if neither is in the path
            continue

        # Reconstruct the expected raw image path without extension
        raw_image_base = os.path.join(RAW_BASE_DIR, os.path.splitext(rel_path)[0])

        # Find the actual image file (could be .JPG, .jpg, .jpeg, etc.)
        image_path = None
        for ext in [".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"]:
            if os.path.exists(raw_image_base + ext):
                image_path = raw_image_base + ext
                break

        if image_path is None:
            missing_images += 1
            continue

        img_filename = os.path.basename(image_path)
        lbl_filename = os.path.basename(label_path)

        # Make the path traceable by encoding the entire relative directory structure into the filename
        # e.g., 2025/Documents/4R/100MEDIA/img.JPG -> 2025_Documents_4R_100MEDIA_img.JPG
        rel_dir = os.path.dirname(rel_path)
        if rel_dir and rel_dir != ".":
            path_prefix = rel_dir.replace(os.sep, "_") + "_"
        else:
            path_prefix = ""

        safe_img_name = f"{path_prefix}{img_filename}"
        safe_lbl_name = f"{path_prefix}{lbl_filename}"

        target_img_path = os.path.join(
            DATASET_OUTPUT_DIR, split, "images", safe_img_name
        )
        target_lbl_path = os.path.join(
            DATASET_OUTPUT_DIR, split, "labels", safe_lbl_name
        )

        # Copy label
        shutil.copy2(label_path, target_lbl_path)

        # Symlink image to save disk space and time
        if not os.path.exists(target_img_path):
            os.symlink(image_path, target_img_path)

        copied_count += 1

    print(f"\nDataset creation complete!")
    print(f"Successfully processed {copied_count} files.")
    if missing_images > 0:
        print(f"Warning: Could not find raw images for {missing_images} labels.")
    print(f"Dataset location: {DATASET_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
