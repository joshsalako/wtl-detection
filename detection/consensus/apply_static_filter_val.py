import os
import sys
import glob

# Add the evaluation directory to sys.path so we can import spatial_filter
EVAL_DIR = "/home/Joshua/Downloads/active_learning_research/detection/evaluation"
if EVAL_DIR not in sys.path:
    sys.path.append(EVAL_DIR)

from eval_utils.spatial_filter import apply_spatial_filter


def main():
    VAL_LABELS_DIR = "/home/Joshua/Downloads/active_learning_research/detection/consensus/outputs/dataset_consensus/val/labels"
    VAL_IMAGES_DIR = "/home/Joshua/Downloads/active_learning_research/detection/consensus/outputs/dataset_consensus/val/images"

    CLEAN_VAL_DIR = "/home/Joshua/Downloads/active_learning_research/detection/consensus/outputs/dataset_consensus/val_filtered"
    CLEAN_LABELS_DIR = os.path.join(CLEAN_VAL_DIR, "labels")
    CLEAN_IMAGES_DIR = os.path.join(CLEAN_VAL_DIR, "images")

    os.makedirs(CLEAN_LABELS_DIR, exist_ok=True)
    os.makedirs(CLEAN_IMAGES_DIR, exist_ok=True)

    label_files = glob.glob(os.path.join(VAL_LABELS_DIR, "*.txt"))

    if not label_files:
        print("No label files found in val set.")
        return

    print(f"Loaded {len(label_files)} label files from {VAL_LABELS_DIR}.")

    # 1. Parse labels into the `results` format
    results = []

    for fpath in label_files:
        filename = os.path.basename(fpath)
        # We need to simulate the original path so the spatial filter can extract the camera and year
        simulated_path = filename.replace("_", "/")

        preds = []
        with open(fpath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    preds.append(
                        {
                            "cls": cls_id,
                            "conf": 1.0,  # Ground truth, assume max confidence
                            "bbox": [cx, cy, w, h],
                        }
                    )

        results.append(
            {
                "real_path": fpath,
                "path": simulated_path,
                "filename": filename,
                "predictions": preds,
            }
        )

    # 2. Run spatial filter
    print("Applying spatial filter to remove static background false positives...")
    filtered_results = apply_spatial_filter(results, min_conf_threshold=0.0)

    # 3. Write back the filtered results into the clean directory
    print("Writing clean dataset...")
    kept_images = 0
    discarded_images = 0

    for res in filtered_results:
        filename = res["filename"]
        preds = res.get("predictions", [])

        if len(preds) == 0:
            discarded_images += 1
            continue

        # Write label
        out_lbl_path = os.path.join(CLEAN_LABELS_DIR, filename)
        with open(out_lbl_path, "w") as f:
            for pred in preds:
                cls_id = pred["cls"]
                cx, cy, w, h = pred["bbox"]
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # Link corresponding image
        img_filename = os.path.splitext(filename)[0] + ".JPG"
        src_img_path = os.path.join(VAL_IMAGES_DIR, img_filename)
        out_img_path = os.path.join(CLEAN_IMAGES_DIR, img_filename)

        if os.path.exists(src_img_path) and not os.path.exists(out_img_path):
            os.symlink(os.path.realpath(src_img_path), out_img_path)

        kept_images += 1

    print("\nPost-processing complete!")
    print(f"Kept {kept_images} images with valid boxes.")
    print(
        f"Completely discarded {discarded_images} images (all boxes were static false positives)."
    )
    print(f"Clean dataset is available at: {CLEAN_VAL_DIR}")


if __name__ == "__main__":
    main()
