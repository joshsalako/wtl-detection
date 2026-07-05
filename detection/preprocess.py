"""Offline CLAHE Preprocessing Pipeline for target object detection datasets.

Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to training,
validation, and test splits to amplify low-contrast details, saving CPU overhead
during training.
"""

import os
import shutil
import glob
import cv2
from tqdm import tqdm
import concurrent.futures

# Import central configurations
from config import SRC_DATASET_DIR, CLAHE_DATASET_DIR


def apply_clahe_to_image(im):
    """Apply CLAHE preprocessing on the L-channel of a BGR image."""
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    im_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return im_clahe


def process_single_image(args):
    """Worker function to process a single image using CLAHE.

    args: (src_path, dst_path, overwrite)
    """
    src_path, dst_path, overwrite = args
    try:
        if not overwrite and os.path.exists(dst_path):
            return True, "skipped"

        im = cv2.imread(src_path)
        if im is None:
            return False, f"Failed to read image: {src_path}"

        im_clahe = apply_clahe_to_image(im)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, im_clahe)
        return True, "success"
    except Exception as e:
        return False, str(e)


def preprocess_dataset(overwrite=False, num_workers=None):
    """Main pipeline function to discover, preprocess and copy dataset splits."""
    print("=================================================================")
    print("      OFFLINE CLAHE PREPROCESSING PIPELINE FOR DATASET          ")
    print("=================================================================")
    print(f"Source Dataset: {SRC_DATASET_DIR}")
    print(f"CLAHE Dataset:  {CLAHE_DATASET_DIR}")
    print("=================================================================\n")

    # Discover splits dynamically as folders containing an 'images' subfolder
    splits = []
    if os.path.exists(SRC_DATASET_DIR):
        for entry in sorted(os.listdir(SRC_DATASET_DIR)):
            entry_path = os.path.join(SRC_DATASET_DIR, entry)
            if os.path.isdir(entry_path) and os.path.exists(
                os.path.join(entry_path, "images")
            ):
                splits.append(entry)
    if not splits:
        splits = ["train", "val", "test"]  # fallback

    jobs = []

    # 1. Discover all images to process
    for split in splits:
        split_src_img_dir = os.path.join(SRC_DATASET_DIR, split, "images")
        split_dst_img_dir = os.path.join(CLAHE_DATASET_DIR, split, "images")
        split_src_lbl_dir = os.path.join(SRC_DATASET_DIR, split, "labels")
        split_dst_lbl_dir = os.path.join(CLAHE_DATASET_DIR, split, "labels")

        # Discover source images
        img_files = sorted(glob.glob(os.path.join(split_src_img_dir, "*.*")))
        img_files = [
            f
            for f in img_files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]

        # Check if this split is already complete in destination
        if not overwrite and len(img_files) > 0:
            is_complete = True
            for src_path in img_files:
                base_name = os.path.basename(src_path)
                dst_path = os.path.join(split_dst_img_dir, base_name)
                if not os.path.exists(dst_path):
                    is_complete = False
                    break

            # Also check if label directory exists and contains files if source has labels
            if is_complete and os.path.exists(split_src_lbl_dir):
                src_lbls = glob.glob(os.path.join(split_src_lbl_dir, "*.*"))
                if len(src_lbls) > 0:
                    if not os.path.exists(split_dst_lbl_dir) or len(
                        glob.glob(os.path.join(split_dst_lbl_dir, "*.*"))
                    ) < len(src_lbls):
                        is_complete = False

            if is_complete:
                print(
                    f"Split '{split}': Found {len(img_files)} images. Already fully processed. Skipping split."
                )
                continue

        print(f"Split '{split}': Found {len(img_files)} images. Processing split...")

        for src_path in img_files:
            base_name = os.path.basename(src_path)
            dst_path = os.path.join(split_dst_img_dir, base_name)
            jobs.append((src_path, dst_path, overwrite))

        # Synchronize label files
        if os.path.exists(split_src_lbl_dir):
            os.makedirs(split_dst_lbl_dir, exist_ok=True)
            lbl_files = glob.glob(os.path.join(split_src_lbl_dir, "*.*"))
            for lbl_file in lbl_files:
                dst_lbl_file = os.path.join(
                    split_dst_lbl_dir, os.path.basename(lbl_file)
                )
                if overwrite or not os.path.exists(dst_lbl_file):
                    shutil.copy2(lbl_file, dst_lbl_file)
            print(f"Split '{split}': Copied labels.")

        # Copy split metadata files if present
        for meta_file in ["classes.txt", "notes.json"]:
            src_meta = os.path.join(SRC_DATASET_DIR, split, meta_file)
            if os.path.exists(src_meta):
                dst_meta = os.path.join(CLAHE_DATASET_DIR, split, meta_file)
                if overwrite or not os.path.exists(dst_meta):
                    shutil.copy2(src_meta, dst_meta)

    # 2. Process images in parallel
    if not jobs:
        print("No images found to preprocess.")
        return

    workers = num_workers or os.cpu_count() or 1
    print(f"\nProcessing {len(jobs)} images with {workers} parallel workers...")

    success_count = 0
    skipped_count = 0
    failed_count = 0
    failures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(
                executor.map(process_single_image, jobs),
                total=len(jobs),
                desc="Preprocessing CLAHE",
            )
        )

        for (success, msg), (src_path, _, _) in zip(results, jobs):
            if success:
                if msg == "skipped":
                    skipped_count += 1
                else:
                    success_count += 1
            else:
                failed_count += 1
                failures.append((src_path, msg))

    print("\n=================================================================")
    print("                      PREPROCESSING COMPLETE                     ")
    print("=================================================================")
    print(f"Successfully Preprocessed: {success_count}")
    print(f"Skipped (Already Exists):   {skipped_count}")
    print(f"Failed to Process:         {failed_count}")
    print("=================================================================")

    if failures:
        print("\nFailures encountered:")
        for path, err in failures[:10]:
            print(f" - {path}: {err}")


if __name__ == "__main__":
    preprocess_dataset()
