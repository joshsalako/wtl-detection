import os
import glob
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from consensus_config import (
    CSV_OUTPUT_DIR,
    YOLO_LABELS_OUTPUT_DIR,
    IOU_THRESHOLD,
    MIN_CONSENSUS_MODELS,
    RESEARCH_DIR,
    RAW_DATASET_BASE,
)


def compute_iou(boxA, boxB):
    # box format: [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if float(boxAArea + boxBArea - interArea) == 0:
        return 0.0

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def apply_clustering_consensus(df, iou_thresh, min_models):
    """
    Groups bounding boxes by IoU. A cluster is valid if it contains
    predictions from at least `min_models` DISTINCT models.
    """
    final_boxes = []

    # Sort by confidence
    df = df.sort_values(by="confidence", ascending=False).reset_index(drop=True)

    used = np.zeros(len(df), dtype=bool)

    boxes = df[["xmin", "ymin", "xmax", "ymax"]].values
    models = df["model_name"].values
    confidences = df["confidence"].values

    for i in range(len(df)):
        if used[i]:
            continue

        seed_box = boxes[i]
        cluster_indices = [i]
        used[i] = True

        # Find matches
        for j in range(i + 1, len(df)):
            if not used[j]:
                iou = compute_iou(seed_box, boxes[j])
                if iou >= iou_thresh:
                    cluster_indices.append(j)
                    used[j] = True

        # Check model consensus
        cluster_models = set(models[cluster_indices])
        if len(cluster_models) >= min_models:
            # Average the boxes
            cluster_boxes = boxes[cluster_indices]
            avg_box = np.mean(cluster_boxes, axis=0)
            avg_conf = np.mean(confidences[cluster_indices])

            final_boxes.append(
                {
                    "xmin": avg_box[0],
                    "ymin": avg_box[1],
                    "xmax": avg_box[2],
                    "ymax": avg_box[3],
                    "confidence": avg_conf,
                    "num_models_agreed": len(cluster_models),
                }
            )

    return final_boxes


def get_yolo_format(box, img_w, img_h):
    xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

    # Clip coordinates to image boundaries
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)

    w = xmax - xmin
    h = ymax - ymin
    cx = xmin + (w / 2.0)
    cy = ymin + (h / 2.0)

    return cx / img_w, cy / img_h, w / img_w, h / img_h


def main():
    csv_files = glob.glob(os.path.join(CSV_OUTPUT_DIR, "*_predictions.csv"))
    if not csv_files:
        print(f"No CSVs found in {CSV_OUTPUT_DIR}. Run inference first.")
        return

    print("Loading predictions...")
    all_dfs = []
    for f in csv_files:
        model_name = os.path.basename(f).replace("_predictions.csv", "")
        df = pd.read_csv(f)
        if not df.empty:
            df["model_name"] = model_name
            all_dfs.append(df)

    if not all_dfs:
        print("All CSVs are empty.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total raw predictions across all models: {len(master_df)}")

    grouped = master_df.groupby(["image_path", "class_id"])

    print("Applying consensus filtering...")
    final_predictions = {}  # image_path -> list of (class_id, yolo_box)

    total_preserved = 0
    for (img_path, cls_id), group_df in tqdm(grouped):
        valid_boxes = apply_clustering_consensus(
            group_df, IOU_THRESHOLD, MIN_CONSENSUS_MODELS
        )

        if valid_boxes:
            if img_path not in final_predictions:
                final_predictions[img_path] = []
            for b in valid_boxes:
                final_predictions[img_path].append((cls_id, b))
                total_preserved += 1

    print(f"\nConsensus resulted in {total_preserved} preserved bounding boxes.")

    print("Writing YOLO format labels...")
    os.makedirs(YOLO_LABELS_OUTPUT_DIR, exist_ok=True)

    for img_path, preds in tqdm(final_predictions.items()):
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        if img_path.startswith(RAW_DATASET_BASE):
            rel_path = os.path.relpath(img_path, RAW_DATASET_BASE)
            out_dir = os.path.join(YOLO_LABELS_OUTPUT_DIR, os.path.dirname(rel_path))
            os.makedirs(out_dir, exist_ok=True)

            img_name = os.path.basename(img_path)
            out_txt_path = os.path.join(out_dir, os.path.splitext(img_name)[0] + ".txt")

            with open(out_txt_path, "w") as f_out:
                for cls_id, box in preds:
                    cx, cy, w, h = get_yolo_format(box, img_w, img_h)
                    f_out.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        else:
            out_dir = os.path.join(YOLO_LABELS_OUTPUT_DIR, "fallback")
            os.makedirs(out_dir, exist_ok=True)
            img_name = os.path.basename(img_path)
            out_txt_path = os.path.join(out_dir, os.path.splitext(img_name)[0] + ".txt")
            with open(out_txt_path, "w") as f_out:
                for cls_id, box in preds:
                    cx, cy, w, h = get_yolo_format(box, img_w, img_h)
                    f_out.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"\nDone! Labels saved to {YOLO_LABELS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
