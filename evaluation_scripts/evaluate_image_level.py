import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

sys.path.append("/home/Joshua/Downloads/active_learning_research/detection")
from config import *


def get_ground_truth_for_image(txt_path):
    gt_boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    gt_boxes.append(
                        {"cls": cls_id, "bbox": [float(x) for x in parts[1:5]]}
                    )
    return gt_boxes


def plot_roc(y_true, y_score, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()
    return roc_auc


def evaluate_model_from_json(json_path, name):
    print(f"Evaluating from JSON for {name}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    gts = []
    max_scores = {0: [], 1: [], 2: [], "combined": []}

    # Load the mapping CSV
    mapping_df = pd.read_csv(IMAGE_MAPPING_CSV)
    path_to_unique_name = {}
    for _, row in mapping_df.iterrows():
        orig_path = row["original_path"]
        new_path = orig_path.replace(
            "/srv/shared_leopard_toad/",
            "/media/Project-drive/shared_leopard_toad_2/shared_leopard_toad_clahe/",
        )
        path_to_unique_name[new_path] = row["unique_name"]

    test_labels_dir = os.path.join(TEST_FULL_DIR, "labels")

    for item in tqdm(data, desc=f"Parsing JSON for {name}"):
        img_path = item["path"]

        # We ignore duplicate inferences on the test_full directory to avoid double counting
        if "test_full" in img_path:
            continue

        # Ground truth mapping
        gt_boxes = []
        if img_path in path_to_unique_name:
            unique_name = path_to_unique_name[img_path]
            name_no_ext, _ = os.path.splitext(unique_name)
            txt_path = os.path.join(test_labels_dir, name_no_ext + ".txt")
            gt_boxes = get_ground_truth_for_image(txt_path)

        has_cls = {0: False, 1: False, 2: False, "combined": False}
        for gt in gt_boxes:
            cls_id = gt["cls"]
            has_cls[cls_id] = True
            has_cls["combined"] = True
        gts.append(has_cls)

        # Predictions
        cls_probs = {0: 0.0, 1: 0.0, 2: 0.0, "combined": 0.0}
        for pred in item.get("predictions", []):
            c = pred["cls"]
            conf = pred["conf"]
            if conf > cls_probs.get(c, 0.0):
                cls_probs[c] = conf
            if conf > cls_probs["combined"]:
                cls_probs["combined"] = conf

        for k in max_scores:
            max_scores[k].append(cls_probs[k])

    # Calculate ROC AUC
    auc_results = {}
    for k in [0, 1, 2, "combined"]:
        y_true = [1 if gt[k] else 0 for gt in gts]
        y_score = max_scores[k]

        # Check if we have both positive and negative samples
        if sum(y_true) > 0 and sum(y_true) < len(y_true):
            auc_val = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_results[k] = (auc_val, fpr, tpr)
        else:
            auc_results[k] = (0.5, None, None)

    # Threshold sweeping for F1 Score
    f1_results = {}
    thresholds = np.arange(0.05, 1.0, 0.05)
    for k in [0, 1, 2, "combined"]:
        y_true = np.array([1 if gt[k] else 0 for gt in gts])
        y_score = np.array(max_scores[k])

        best_f1 = -1
        best_thresh = 0.05
        best_prec = 0
        best_rec = 0

        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            # Avoid division by zero warnings by using zero_division=0
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                best_prec = prec
                best_rec = rec

        f1_results[k] = {
            "best_thresh": best_thresh,
            "precision": best_prec,
            "recall": best_rec,
            "f1": best_f1,
        }

    return auc_results, f1_results


def main():
    results = []
    f1_results_list = []

    model_rocs = {0: [], 1: [], 2: [], "combined": []}

    models = [
        ("yolo", "YOLO"),
        ("rtdetr", "RT-DETR"),
        ("faster_rcnn", "Faster R-CNN"),
    ]

    for dir_name, m_name in tqdm(models, desc="Overall Image-Level Evaluation"):
        json_path = os.path.join(RESULTS_DIR, dir_name, "predictions.json")
        if not os.path.exists(json_path):
            print(
                f"JSON not found for {m_name} at {json_path}. Please run generate_predictions.py first."
            )
            continue

        try:
            auc_results, f1_results = evaluate_model_from_json(json_path, m_name)

            # Save results
            for k in [0, 1, 2, "combined"]:
                cls_name = CLASS_NAMES.get(k, "Combined_3_Classes")
                auc_val, fpr, tpr = auc_results[k]

                if fpr is not None:
                    model_rocs[k].append((m_name, auc_val, fpr, tpr))

                results.append({"Model": m_name, "Class": cls_name, "ROC_AUC": auc_val})

                f1_results_list.append(
                    {
                        "Model": m_name,
                        "Class": cls_name,
                        "Optimal_Threshold": f1_results[k]["best_thresh"],
                        "Precision": f1_results[k]["precision"],
                        "Recall": f1_results[k]["recall"],
                        "F1_Score": f1_results[k]["f1"],
                    }
                )

        except Exception as e:
            print(f"Error evaluating {m_name}: {e}")

    if results:
        for k in [0, 1, 2, "combined"]:
            cls_name = CLASS_NAMES.get(k, "Combined_3_Classes")
            if model_rocs[k]:
                plt.figure(figsize=(10, 8))
                for m_name, auc_val, fpr, tpr in model_rocs[k]:
                    plt.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.3f})")

                plt.plot([0, 1], [0, 1], "k--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"Image-Level ROC Curve ({cls_name})")
                plt.legend(loc="lower right")
                plot_path = os.path.join(
                    OUTPUT_DIR, f"image_level_roc_curve_{cls_name.lower()}.png"
                )
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved ROC curve for {cls_name} to {plot_path}")

        # Output AUC table
        df = pd.DataFrame(results)
        out_csv = os.path.join(OUTPUT_DIR, "image_level_evaluation_results.csv")
        df.to_csv(out_csv, index=False)
        out_md = os.path.join(OUTPUT_DIR, "image_level_evaluation_results.md")
        df.to_markdown(out_md, index=False)
        print(f"Saved image level results to {out_csv} and {out_md}")

        # Output F1 table
        df_f1 = pd.DataFrame(f1_results_list)
        out_f1_csv = os.path.join(OUTPUT_DIR, "image_level_f1_results.csv")
        df_f1.to_csv(out_f1_csv, index=False)
        out_f1_md = os.path.join(OUTPUT_DIR, "image_level_f1_results.md")
        df_f1.to_markdown(out_f1_md, index=False)
        print(f"Saved image level F1 results to {out_f1_csv} and {out_f1_md}")


if __name__ == "__main__":
    main()
