import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

sys.path.append("/home/Joshua/Downloads/active_learning_research/detection")
from config import *


# Helper to compute IoU between two boxes in [xmin, ymin, xmax, ymax]
def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


# Convert [x_center, y_center, w, h] normalized to [x_min, y_min, x_max, y_max] normalized
def yolo_to_xyxy(bbox):
    x_c, y_c, w, h = bbox
    return [x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2]


def get_ground_truth_for_image(txt_path):
    gt_boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    # Original is YOLO format: cls, x_c, y_c, w, h
                    yolo_bbox = [float(x) for x in parts[1:5]]
                    xyxy = yolo_to_xyxy(yolo_bbox)
                    gt_boxes.append({"cls": cls_id, "bbox": xyxy})
    return gt_boxes


def evaluate_model_detection(json_path, m_name):
    print(f"Evaluating Detection from JSON for {m_name}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Load mapping
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

    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True
    )

    # For manual sweeping
    all_gts = []
    all_preds = []

    for item in tqdm(data, desc=f"Parsing JSON for {m_name}"):
        img_path = item["path"]

        # Ignore double counting from test_full
        if "test_full" in img_path:
            continue

        # We ONLY evaluate images in the image_mapping.csv (ground truth test set)
        if img_path not in path_to_unique_name:
            continue

        unique_name = path_to_unique_name[img_path]
        name_no_ext, _ = os.path.splitext(unique_name)
        txt_path = os.path.join(test_labels_dir, name_no_ext + ".txt")

        gt_boxes = get_ground_truth_for_image(txt_path)
        pred_boxes = item.get("predictions", [])

        # Prepare for MeanAveragePrecision
        target = {
            "boxes": torch.tensor([gt["bbox"] for gt in gt_boxes], dtype=torch.float32)
            if gt_boxes
            else torch.empty((0, 4)),
            "labels": torch.tensor([gt["cls"] for gt in gt_boxes], dtype=torch.int64)
            if gt_boxes
            else torch.empty((0,), dtype=torch.int64),
        }

        pred = {
            "boxes": torch.tensor([p["bbox"] for p in pred_boxes], dtype=torch.float32)
            if pred_boxes
            else torch.empty((0, 4)),
            "scores": torch.tensor([p["conf"] for p in pred_boxes], dtype=torch.float32)
            if pred_boxes
            else torch.empty((0,), dtype=torch.float32),
            "labels": torch.tensor([p["cls"] for p in pred_boxes], dtype=torch.int64)
            if pred_boxes
            else torch.empty((0,), dtype=torch.int64),
        }

        metric.update([pred], [target])

        # Store for manual sweep
        all_gts.append(gt_boxes)
        all_preds.append(pred_boxes)

    # Compute mAP
    print(f"Computing mAP for {m_name}...")
    res_metrics = metric.compute()
    map_res = {
        "Model": m_name,
        "mAP50": res_metrics["map_50"].item(),
        "mAP75": res_metrics["map_75"].item(),
        "mAP50-95": res_metrics["map"].item(),
    }

    if "map_per_class" in res_metrics:
        classes = res_metrics["classes"].cpu().numpy()
        map_per_cls = res_metrics["map_per_class"].cpu().numpy()
        for cls_idx, c_ap in zip(classes, map_per_cls):
            cls_name = CLASS_NAMES.get(int(cls_idx), f"Class_{int(cls_idx)}")
            map_res[f"{cls_name}_mAP50-95"] = c_ap

    # Manual Sweep for F1 and PR curve
    print(f"Sweeping thresholds for {m_name} F1 Score and PR Curve...")
    thresholds = np.linspace(0.0, 1.0, 100)
    f1_results = {}
    pr_curve_data = {}

    for k in [0, 1, 2, "combined"]:
        best_f1 = -1
        best_thresh = 0.05
        best_prec = 0
        best_rec = 0

        precisions_for_curve = []
        recalls_for_curve = []

        for t in thresholds:
            total_tp = 0
            total_fp = 0
            total_fn = 0

            for gt_boxes, pred_boxes in zip(all_gts, all_preds):
                if k == "combined":
                    img_gt = [gt for gt in gt_boxes]
                    img_pred = [p for p in pred_boxes if p["conf"] >= t]
                else:
                    img_gt = [gt for gt in gt_boxes if gt["cls"] == k]
                    img_pred = [
                        p for p in pred_boxes if p["cls"] == k and p["conf"] >= t
                    ]

                matched_gts = set()
                img_pred.sort(key=lambda x: x["conf"], reverse=True)

                for p in img_pred:
                    best_iou = 0
                    best_gt_idx = -1
                    for idx, gt in enumerate(img_gt):
                        if idx in matched_gts:
                            continue
                        iou = compute_iou(p["bbox"], gt["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                    if best_iou >= 0.5:
                        total_tp += 1
                        matched_gts.add(best_gt_idx)
                    else:
                        total_fp += 1

                total_fn += len(img_gt) - len(matched_gts)

            prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

            precisions_for_curve.append(prec)
            recalls_for_curve.append(rec)

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

        pr_curve_data[k] = {
            "precisions": precisions_for_curve,
            "recalls": recalls_for_curve,
        }

    return map_res, f1_results, pr_curve_data


import matplotlib.pyplot as plt


def plot_pr_curve(pr_curve_data, m_name, dir_name):
    plt.figure(figsize=(8, 6))
    for k in [0, 1, 2, "combined"]:
        if k not in pr_curve_data:
            continue

        rec = pr_curve_data[k]["recalls"]
        prec = pr_curve_data[k]["precisions"]

        sort_idx = np.argsort(rec)
        rec = np.array(rec)[sort_idx]
        prec = np.array(prec)[sort_idx]

        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])

        if k == "combined":
            cls_name = "all classes"
        else:
            cls_name = CLASS_NAMES.get(k, f"Class_{k}")

        linewidth = 3 if k == "combined" else 1.5
        plt.plot(rec, prec, label=f"{cls_name}", linewidth=linewidth)

    plt.title(f"Precision-Recall Curve ({m_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower left")
    plt.tight_layout()

    out_dir = os.path.join(RESULTS_DIR, dir_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "BoxPR_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved PR curve to {out_path}")


def main():
    map_results = []
    f1_results_list = []

    models = [
        ("yolo", "YOLO"),
        ("rtdetr", "RT-DETR"),
        ("faster_rcnn", "Faster R-CNN"),
    ]

    for dir_name, m_name in tqdm(models, desc="Overall Detection Evaluation"):
        json_path = os.path.join(RESULTS_DIR, dir_name, "predictions.json")
        if not os.path.exists(json_path):
            print(
                f"JSON not found for {m_name}. Please run generate_predictions.py first."
            )
            continue

        try:
            map_res, f1_results, pr_curve_data = evaluate_model_detection(
                json_path, m_name
            )
            map_results.append(map_res)

            # Plot PR curve
            plot_pr_curve(pr_curve_data, m_name, dir_name)

            for k in [0, 1, 2, "combined"]:
                cls_name = CLASS_NAMES.get(k, "Combined_3_Classes")
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

    if map_results:
        df_map = pd.DataFrame(map_results)
        out_map_csv = os.path.join(OUTPUT_DIR, "detection_evaluation_results.csv")
        df_map.to_csv(out_map_csv, index=False)
        out_map_md = os.path.join(OUTPUT_DIR, "detection_evaluation_results.md")
        df_map.to_markdown(out_map_md, index=False)
        print(f"Saved detection mAP results to {out_map_md}")

        df_f1 = pd.DataFrame(f1_results_list)
        out_f1_csv = os.path.join(OUTPUT_DIR, "detection_f1_results.csv")
        df_f1.to_csv(out_f1_csv, index=False)
        out_f1_md = os.path.join(OUTPUT_DIR, "detection_f1_results.md")
        df_f1.to_markdown(out_f1_md, index=False)
        print(f"Saved detection F1 results to {out_f1_md}")


if __name__ == "__main__":
    main()
