import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import (
    RESULTS_DIR,
    TEST_DIR,
    VAL_DIR,
    FILES_DIR,
    CLASSES,
    WLT_CLASS_ID,
    SMALL_MAMMAL_CLASS_ID,
    OTHER_AMPHIBIAN_CLASS_ID,
    DETECTION_METRICS_CSV,
    IMAGE_LEVEL_WLT_CSV,
    IMAGE_LEVEL_AGNOSTIC_CSV,
    IMAGE_LEVEL_WLT_JSON,
    IMAGE_LEVEL_AGNOSTIC_JSON,
)


def format_cvat_filename(abs_path):
    if "/srv/shared_leopard_toad/" in abs_path:
        rel = abs_path.split("/srv/shared_leopard_toad/")[-1]
        cvat_name = rel.replace("/", "_").replace("\\", "_")
        return cvat_name
    return os.path.basename(abs_path)


def load_ground_truth(dataset_dir):
    """Loads GT labels into a dictionary: {path: [{"bbox": [x1, y1, x2, y2], "cls": cls_id}]}
    Also returns binary image-level labels for WLT (class 2) and Class Agnostic (any class)."""
    gt_dict = {}
    img_dir = os.path.join(dataset_dir, "images")
    lbl_dir = os.path.join(dataset_dir, "labels")

    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        return {}

    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        # YOLO format is cx, cy, w, h normalized
                        cx, cy, w, h = map(float, parts[1:5])
                        # Convert to x1, y1, x2, y2 format for AP calculation
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        boxes.append({"bbox": [x1, y1, x2, y2], "cls": cls_id})
        gt_dict[img_name] = boxes
    return gt_dict


def calculate_detection_metrics(gt_dict, preds_list):
    """Calculates AP metrics using torchmetrics."""
    metric_50_95 = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True
    )
    metric_50 = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True, iou_thresholds=[0.5]
    )

    # Map predictions to dictionary by CVAT filename format for easy access
    pred_dict = {}
    for p in preds_list:
        cvat_name = format_cvat_filename(p["path"])
        pred_dict[cvat_name] = p.get("predictions", [])

        import urllib.parse

        encoded_name = urllib.parse.quote(cvat_name)
        if encoded_name != cvat_name:
            pred_dict[encoded_name] = p.get("predictions", [])

    preds_fmt = []
    targets_fmt = []

    for img_name, gt_boxes in gt_dict.items():
        # GT format
        gt_b = []
        gt_l = []
        for box in gt_boxes:
            gt_b.append(box["bbox"])
            gt_l.append(box["cls"])

        if len(gt_b) > 0:
            targets_fmt.append(
                {
                    "boxes": torch.tensor(gt_b, dtype=torch.float32),
                    "labels": torch.tensor(gt_l, dtype=torch.int64),
                }
            )
        else:
            targets_fmt.append(
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }
            )

        # Pred format
        p_boxes = pred_dict.get(img_name, [])
        pr_b = []
        pr_l = []
        pr_s = []
        for box in p_boxes:
            cx, cy, w, h = box["bbox"]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            pr_b.append([x1, y1, x2, y2])
            pr_l.append(box["cls"])
            pr_s.append(box["conf"])

        if len(pr_b) > 0:
            preds_fmt.append(
                {
                    "boxes": torch.tensor(pr_b, dtype=torch.float32),
                    "scores": torch.tensor(pr_s, dtype=torch.float32),
                    "labels": torch.tensor(pr_l, dtype=torch.int64),
                }
            )
        else:
            preds_fmt.append(
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "scores": torch.empty((0,), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }
            )

    metric_50_95.update(preds_fmt, targets_fmt)
    metric_50.update(preds_fmt, targets_fmt)
    return metric_50_95.compute(), metric_50.compute()


def get_image_level_probs(gt_dict, preds_list, target_class=None):
    """
    Returns (y_true, y_score) arrays for image level binary classification.
    If target_class is None, it is class-agnostic (any object).
    Otherwise it looks for the specific class.
    """
    pred_dict = {}
    for p in preds_list:
        cvat_name = format_cvat_filename(p["path"])
        pred_dict[cvat_name] = p.get("predictions", [])

        import urllib.parse

        encoded_name = urllib.parse.quote(cvat_name)
        if encoded_name != cvat_name:
            pred_dict[encoded_name] = p.get("predictions", [])
    all_names = set(gt_dict.keys()).union(set(pred_dict.keys()))

    y_true = []
    y_score = []

    for img_name in all_names:
        if img_name in gt_dict:
            gt_boxes = gt_dict[img_name]
            if target_class is None:
                is_pos = int(len(gt_boxes) > 0)
            else:
                is_pos = int(any(b["cls"] == target_class for b in gt_boxes))
        else:
            is_pos = 0

        y_true.append(is_pos)

        p_boxes = pred_dict.get(img_name, [])
        if target_class is None:
            max_conf = max([b["conf"] for b in p_boxes] + [0.0])
        else:
            max_conf = max(
                [b["conf"] for b in p_boxes if b["cls"] == target_class] + [0.0]
            )

        y_score.append(max_conf)

    return np.array(y_true), np.array(y_score)


def calculate_threshold_sweep(y_true, y_score):
    """Finds best F1 threshold and returns metrics."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    p_valid = precisions[:-1]
    r_valid = recalls[:-1]

    f1_scores = np.divide(
        2 * (p_valid * r_valid),
        (p_valid + r_valid),
        out=np.zeros_like(p_valid),
        where=(p_valid + r_valid) != 0,
    )

    if len(f1_scores) > 0:
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_p = p_valid[best_idx]
        best_r = r_valid[best_idx]
    else:
        best_thresh, best_f1, best_p, best_r = 0, 0, 0, 0

    # Calculate confusion matrix for best threshold.
    y_pred = (y_score >= best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "best_threshold": best_thresh,
        "best_f1": best_f1,
        "precision": best_p,
        "recall": best_r,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "y_true": y_true,
        "y_score": y_score,
    }


def main():
    print("Loading Ground Truth...")
    test_gt = load_ground_truth(TEST_DIR)
    val_gt = load_ground_truth(VAL_DIR)

    detection_results = []
    image_level_wlt_results = []
    image_level_agnostic_results = []

    filtered_files = glob.glob(os.path.join(RESULTS_DIR, "*", "*_filtered.json"))
    # Exclude megadetector from filtered files (handled separately below)
    filtered_files = [f for f in filtered_files if "megadetector" not in f]

    md_test_file = os.path.join(
        RESULTS_DIR, "megadetector", "cycle_0_pretrained_test_full_seq_filtered.json"
    )
    md_val_file = os.path.join(RESULTS_DIR, "megadetector", "val_full_filtered.json")
    md_files = [f for f in [md_test_file, md_val_file] if os.path.exists(f)]

    all_files_to_eval = filtered_files + md_files

    for f_path in tqdm(all_files_to_eval, desc="Evaluating models"):
        parts = f_path.split(os.sep)
        model_name = parts[-2]
        file_name = parts[-1]

        if "megadetector" in model_name:
            if "test_full" in file_name:
                dataset_name = "test_full"
            elif "val_full" in file_name:
                dataset_name = "val_full"
            else:
                dataset_name = "test" if "test" in file_name else "val"
            cycle = "N/A"
        else:
            name_parts = file_name.replace("_filtered.json", "").split("_")
            dataset_name = f"{name_parts[0]}_{name_parts[1]}"
            cycle = f"{name_parts[2]}_{name_parts[3]}"  # e.g. cycle_0

        with open(f_path, "r") as f:
            preds = json.load(f)

        gt_dict = test_gt if "test" in dataset_name else val_gt

        # 1. Detection Metrics (Automatically evaluated only on images that exist in GT)
        det_metrics_50_95, det_metrics_50 = calculate_detection_metrics(gt_dict, preds)

        mAP50 = det_metrics_50["map"].item() if "map" in det_metrics_50 else 0.0
        mAP50_95 = (
            det_metrics_50_95["map"].item() if "map" in det_metrics_50_95 else 0.0
        )

        # Class APs — use the 'classes' tensor returned by torchmetrics to build a
        # label-id → AP lookup, avoiding index-shift errors when a class is absent.
        map_per_class_50 = det_metrics_50.get("map_per_class", torch.tensor([]))
        map_per_class_50_95 = det_metrics_50_95.get("map_per_class", torch.tensor([]))
        class_ids_50 = det_metrics_50.get("classes", torch.tensor([]))
        class_ids_50_95 = det_metrics_50_95.get("classes", torch.tensor([]))

        ap50_by_cls = {
            cls_id.item(): ap.item()
            for cls_id, ap in zip(class_ids_50, map_per_class_50)
        }
        ap50_95_by_cls = {
            cls_id.item(): ap.item()
            for cls_id, ap in zip(class_ids_50_95, map_per_class_50_95)
        }

        row_det = {
            "Model": model_name,
            "Cycle": cycle,
            "Dataset": dataset_name,
            "mAP50": mAP50,
            "mAP50-95": mAP50_95,
            "Other_Amphibian_AP50": ap50_by_cls.get(OTHER_AMPHIBIAN_CLASS_ID, -1),
            "Other_Amphibian_AP95": ap50_95_by_cls.get(OTHER_AMPHIBIAN_CLASS_ID, -1),
            "Small_Mammal_AP50": ap50_by_cls.get(SMALL_MAMMAL_CLASS_ID, -1),
            "Small_Mammal_AP95": ap50_95_by_cls.get(SMALL_MAMMAL_CLASS_ID, -1),
            "WLT_AP50": ap50_by_cls.get(WLT_CLASS_ID, -1),
            "WLT_AP95": ap50_95_by_cls.get(WLT_CLASS_ID, -1),
        }

        detection_results.append(row_det)

        # 2. Image Level Metrics (Evaluated across the full sequence)
        y_true_wlt, _ = get_image_level_probs(gt_dict, preds, target_class=WLT_CLASS_ID)

        if "megadetector" in model_name:
            _, y_score_wlt = get_image_level_probs(gt_dict, preds, target_class=None)
        else:
            _, y_score_wlt = get_image_level_probs(
                gt_dict, preds, target_class=WLT_CLASS_ID
            )

        if len(np.unique(y_true_wlt)) > 1:
            auc_wlt = roc_auc_score(y_true_wlt, y_score_wlt)
            sweep_wlt = calculate_threshold_sweep(y_true_wlt, y_score_wlt)
            image_level_wlt_results.append(
                {
                    "Model": model_name,
                    "Cycle": cycle,
                    "Dataset": dataset_name,
                    "AUC": auc_wlt,
                    "Best_F1": sweep_wlt["best_f1"],
                    "Best_Threshold": sweep_wlt["best_threshold"],
                    "Precision": sweep_wlt["precision"],
                    "Recall": sweep_wlt["recall"],
                    "TP": sweep_wlt["tp"],
                    "FP": sweep_wlt["fp"],
                    "FN": sweep_wlt["fn"],
                    "TN": sweep_wlt["tn"],
                    "y_true": sweep_wlt["y_true"].tolist(),
                    "y_score": sweep_wlt["y_score"].tolist(),
                }
            )

        # Class Agnostic (All models)
        y_true_agn, y_score_agn = get_image_level_probs(
            gt_dict, preds, target_class=None
        )
        if len(np.unique(y_true_agn)) > 1:
            auc_agn = roc_auc_score(y_true_agn, y_score_agn)
            sweep_agn = calculate_threshold_sweep(y_true_agn, y_score_agn)
            image_level_agnostic_results.append(
                {
                    "Model": model_name,
                    "Cycle": cycle,
                    "Dataset": dataset_name,
                    "AUC": auc_agn,
                    "Best_F1": sweep_agn["best_f1"],
                    "Best_Threshold": sweep_agn["best_threshold"],
                    "Precision": sweep_agn["precision"],
                    "Recall": sweep_agn["recall"],
                    "TP": sweep_agn["tp"],
                    "FP": sweep_agn["fp"],
                    "FN": sweep_agn["fn"],
                    "TN": sweep_agn["tn"],
                    "y_true": sweep_agn["y_true"].tolist(),
                    "y_score": sweep_agn["y_score"].tolist(),
                }
            )

    # Save CSVs
    df_det = pd.DataFrame(detection_results)
    df_det.to_csv(os.path.join(FILES_DIR, DETECTION_METRICS_CSV), index=False)

    # Exclude y_true and y_score arrays for the CSV tables
    if len(image_level_wlt_results) > 0:
        df_wlt = pd.DataFrame(image_level_wlt_results)
        df_wlt_csv = df_wlt.drop(columns=["y_true", "y_score"])
        df_wlt_csv.to_csv(os.path.join(FILES_DIR, IMAGE_LEVEL_WLT_CSV), index=False)
        # Save full data for plotting
        df_wlt.to_json(os.path.join(FILES_DIR, IMAGE_LEVEL_WLT_JSON), orient="records")

    if len(image_level_agnostic_results) > 0:
        df_ag = pd.DataFrame(image_level_agnostic_results)
        df_ag_csv = df_ag.drop(columns=["y_true", "y_score"])
        df_ag_csv.to_csv(os.path.join(FILES_DIR, IMAGE_LEVEL_AGNOSTIC_CSV), index=False)
        df_ag.to_json(
            os.path.join(FILES_DIR, IMAGE_LEVEL_AGNOSTIC_JSON), orient="records"
        )


if __name__ == "__main__":
    main()
