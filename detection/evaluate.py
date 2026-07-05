"""Evaluation and Comparison Script for Finetuned Models.

Loads the best checkpoints of YOLO, RT-DETR, and Faster R-CNN from training,
evaluates them on the test split of the CLAHE dataset, computes standard object
detection metrics (Precision, Recall, mAP50), and generates comparative plots.
"""

import os
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from ultralytics import YOLO, RTDETR
import matplotlib.pyplot as plt
import seaborn as sns

# Import config settings
from config import (
    DEVICE,
    CLAHE_DATASET_DIR,
    RESULTS_DIR,
    CLASSES,
    NUM_CLASSES,
    IMG_SIZE,
)

# Import Faster R-CNN utilities
from faster_rcnn_utils import (
    ActiveLearningFasterRCNNDataset,
    collate_fn,
    get_faster_rcnn_model,
    load_compatible_weights,
)


def evaluate_ultralytics(model_type, dataset_yaml, split="test"):
    """Evaluates YOLO or RT-DETR on the specified set using Ultralytics val mode."""
    print(f"\nEvaluating {model_type.upper()} on the {split} set...")
    best_weights = os.path.join(RESULTS_DIR, model_type, "phase2", "weights", "best.pt")

    if not os.path.exists(best_weights):
        print(f"Warning: Best weight file not found at {best_weights}. Skipping.")
        return None

    # Ensure the requested split is defined in the dataset yaml for Ultralytics
    if os.path.exists(dataset_yaml):
        try:
            with open(dataset_yaml, "r") as f:
                yaml_data = yaml.safe_load(f) or {}
            if split not in yaml_data:
                yaml_data[split] = f"{split}/images"
                with open(dataset_yaml, "w") as f:
                    yaml.dump(yaml_data, f, default_flow_style=False)
                print(f"[Config] Dynamically added split '{split}' to dataset YAML.")
        except Exception as e:
            print(f"Warning: Failed to update dataset YAML for split '{split}': {e}")

    model_class = YOLO if model_type == "yolo" else RTDETR
    model = model_class(best_weights)

    # Run validation on the specified split
    results = model.val(
        data=dataset_yaml,
        split=split,
        imgsz=IMG_SIZE,
        device="0" if torch.cuda.is_available() else "cpu",
        plots=True,
        verbose=False,
        project=os.path.join(RESULTS_DIR, model_type, "phase2"),
        name=split,
        exist_ok=True,
    )

    # Extract metrics
    metrics_dict = results.results_dict
    mAP50_all = metrics_dict.get("metrics/mAP50(B)", 0.0)
    precision_all = metrics_dict.get("metrics/precision(B)", 0.0)
    recall_all = metrics_dict.get("metrics/recall(B)", 0.0)

    # Class-specific AP50
    class_aps = {}
    for i, cls_idx in enumerate(results.box.ap_class_index):
        cls_name = CLASSES[cls_idx]
        cls_ap50 = results.box.ap50[i]
        class_aps[cls_name] = cls_ap50

    model_metrics = {
        "model": model_type.upper(),
        "precision": precision_all,
        "recall": recall_all,
        "mAP50": mAP50_all,
        **class_aps,
    }

    return model_metrics


def compute_faster_rcnn_metrics_and_curves(
    all_predictions, all_targets, iou_threshold=0.5
):
    """Computes standard AP50 metrics, and gathers data for curves."""
    pr_data = {
        c: {"scores": [], "matches": [], "num_gt": 0} for c in range(1, NUM_CLASSES + 1)
    }

    total_gt_boxes = 0

    for pred, tgt in zip(all_predictions, all_targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]

        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]

        total_gt_boxes += len(gt_boxes)

        # Count GT boxes per class
        for c in range(1, NUM_CLASSES + 1):
            pr_data[c]["num_gt"] += (gt_labels == c).sum().item()

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes)).numpy()
            matched_gt = set()
            sorted_idx = np.argsort(pred_scores)[::-1]

            for i in sorted_idx:
                p_label = pred_labels[i]
                p_score = pred_scores[i]
                best_iou = 0.0
                best_gt_idx = -1

                for j in range(len(gt_boxes)):
                    if j not in matched_gt and gt_labels[j] == p_label:
                        iou = ious[i, j]
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j

                is_match = 0
                if best_iou >= iou_threshold:
                    matched_gt.add(best_gt_idx)
                    is_match = 1

                pr_data[p_label]["scores"].append(p_score)
                pr_data[p_label]["matches"].append(is_match)

        elif len(pred_boxes) > 0:
            for lbl, score in zip(pred_labels, pred_scores):
                pr_data[lbl]["scores"].append(score)
                pr_data[lbl]["matches"].append(0)

    class_aps = {}
    ap_list = []

    conf_grid = np.linspace(0.0, 1.0, 100)

    curves_data = {
        c: {"precision_conf": [], "recall_conf": [], "f1_conf": []}
        for c in range(1, NUM_CLASSES + 1)
    }

    pr_curves = {}

    for c in range(1, NUM_CLASSES + 1):
        cls_name = CLASSES[c - 1]
        data = pr_data[c]
        scores = np.array(data["scores"])
        matches = np.array(data["matches"])
        num_gt = data["num_gt"]

        if num_gt == 0:
            class_aps[cls_name] = 0.0
            pr_curves[c] = {
                "recall": np.array([0.0, 1.0]),
                "precision": np.array([0.0, 0.0]),
                "ap": 0.0,
            }
            curves_data[c]["precision_conf"] = np.ones_like(conf_grid)
            curves_data[c]["recall_conf"] = np.zeros_like(conf_grid)
            curves_data[c]["f1_conf"] = np.zeros_like(conf_grid)
            continue

        if len(scores) == 0:
            class_aps[cls_name] = 0.0
            ap_list.append(0.0)
            pr_curves[c] = {
                "recall": np.array([0.0, 1.0]),
                "precision": np.array([0.0, 0.0]),
                "ap": 0.0,
            }
            curves_data[c]["precision_conf"] = np.ones_like(conf_grid)
            curves_data[c]["recall_conf"] = np.zeros_like(conf_grid)
            curves_data[c]["f1_conf"] = np.zeros_like(conf_grid)
            continue

        indices = np.argsort(scores)[::-1]
        matches_sorted = matches[indices]
        scores_sorted = scores[indices]

        tp_cumsum = np.cumsum(matches_sorted)
        fp_cumsum = np.cumsum(1 - matches_sorted)

        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        rec = np.concatenate(([0.0], recalls, [1.0]))
        prec = np.concatenate(([1.0], precisions, [0.0]))
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        ap = np.sum((rec[1:] - rec[:-1]) * prec[1:])
        class_aps[cls_name] = ap
        ap_list.append(ap)

        pr_curves[c] = {"recall": rec, "precision": prec, "ap": ap}

        p_c_list = []
        r_c_list = []
        f1_c_list = []

        for t in conf_grid:
            valid_mask = scores_sorted >= t
            tp_t = np.sum(matches_sorted[valid_mask])
            fp_t = np.sum(1 - matches_sorted[valid_mask])

            p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 1.0
            r_t = tp_t / num_gt
            f1_t = 2 * p_t * r_t / (p_t + r_t + 1e-16)

            p_c_list.append(p_t)
            r_c_list.append(r_t)
            f1_c_list.append(f1_t)

        curves_data[c]["precision_conf"] = np.array(p_c_list)
        curves_data[c]["recall_conf"] = np.array(r_c_list)
        curves_data[c]["f1_conf"] = np.array(f1_c_list)

    mAP50 = np.mean(ap_list) if ap_list else 0.0

    overall_precision = 0.0
    overall_recall = 0.0

    all_scores = []
    all_matches = []
    for c in range(1, NUM_CLASSES + 1):
        all_scores.extend(pr_data[c]["scores"])
        all_matches.extend(pr_data[c]["matches"])

    if len(all_scores) > 0:
        all_scores = np.array(all_scores)
        all_matches = np.array(all_matches)
        valid_idx = all_scores >= 0.25
        tp = np.sum(all_matches[valid_idx])
        fp = np.sum(1 - all_matches[valid_idx])

        overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        overall_recall = tp / total_gt_boxes if total_gt_boxes > 0 else 0.0

    metrics = {
        "precision": overall_precision,
        "recall": overall_recall,
        "mAP50": mAP50,
        **class_aps,
    }

    return metrics, pr_curves, curves_data, conf_grid


def compute_confusion_matrix(
    all_predictions, all_targets, num_classes=3, iou_threshold=0.5
):
    """Computes a (num_classes + 1) x (num_classes + 1) confusion matrix."""
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]

        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        indices = np.argsort(pred_scores)[::-1]
        matched_gts = set()

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes)).numpy()
        else:
            ious = None

        for p_idx in indices:
            p_label = pred_labels[p_idx]
            p_class_idx = p_label - 1

            best_iou = 0.0
            best_gt_idx = -1

            if ious is not None:
                for g_idx in range(len(gt_boxes)):
                    if g_idx in matched_gts:
                        continue
                    iou = ious[p_idx, g_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            if best_iou >= iou_threshold:
                matched_gts.add(best_gt_idx)
                gt_label = gt_labels[best_gt_idx]
                gt_class_idx = gt_label - 1
                cm[gt_class_idx, p_class_idx] += 1
            else:
                cm[num_classes, p_class_idx] += 1

        for g_idx in range(len(gt_boxes)):
            if g_idx not in matched_gts:
                gt_label = gt_labels[g_idx]
                gt_class_idx = gt_label - 1
                cm[gt_class_idx, num_classes] += 1

    return cm


def plot_evaluation_curves(pr_curves, curves_data, conf_grid, mAP50, output_dir):
    """Generates and saves PR, F1, P, and R curves matching Ultralytics format."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. P-R Curve
    plt.figure(figsize=(8, 6))
    recall_grid = np.linspace(0.0, 1.0, 100)
    all_class_precisions = []

    for c in range(1, NUM_CLASSES + 1):
        cls_name = CLASSES[c - 1]
        rec = pr_curves[c]["recall"]
        prec = pr_curves[c]["precision"]
        ap = pr_curves[c]["ap"]

        plt.plot(rec, prec, label=f"{cls_name} {ap:.3f}", linewidth=1.5)

        sort_idx = np.argsort(rec)
        prec_interp = np.interp(recall_grid, rec[sort_idx], prec[sort_idx])
        all_class_precisions.append(prec_interp)

    mean_precision = np.mean(all_class_precisions, axis=0)
    plt.plot(
        recall_grid,
        mean_precision,
        label=f"all classes {mAP50:.3f} mAP@0.5",
        linewidth=3,
        color="blue",
    )

    plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "BoxPR_curve.png"), dpi=300)
    plt.close()

    # 2. F1-Confidence Curve
    plt.figure(figsize=(8, 6))
    all_f1s = []
    for c in range(1, NUM_CLASSES + 1):
        cls_name = CLASSES[c - 1]
        f1 = curves_data[c]["f1_conf"]
        plt.plot(conf_grid, f1, label=f"{cls_name}", linewidth=1.5)
        all_f1s.append(f1)

    mean_f1 = np.mean(all_f1s, axis=0)
    peak_idx = np.argmax(mean_f1)
    peak_f1 = mean_f1[peak_idx]
    peak_conf = conf_grid[peak_idx]

    plt.plot(
        conf_grid,
        mean_f1,
        label=f"all classes {peak_f1:.2f} at {peak_conf:.2f}",
        linewidth=3,
        color="blue",
    )

    plt.title("F1-Confidence Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("F1", fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "BoxF1_curve.png"), dpi=300)
    plt.close()

    # 3. Precision-Confidence Curve
    plt.figure(figsize=(8, 6))
    all_ps = []
    for c in range(1, NUM_CLASSES + 1):
        cls_name = CLASSES[c - 1]
        p = curves_data[c]["precision_conf"]
        plt.plot(conf_grid, p, label=f"{cls_name}", linewidth=1.5)
        all_ps.append(p)

    mean_p = np.mean(all_ps, axis=0)
    plt.plot(conf_grid, mean_p, label="all classes", linewidth=3, color="blue")

    plt.title("Precision-Confidence Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "BoxP_curve.png"), dpi=300)
    plt.close()

    # 4. Recall-Confidence Curve
    plt.figure(figsize=(8, 6))
    all_rs = []
    for c in range(1, NUM_CLASSES + 1):
        cls_name = CLASSES[c - 1]
        r = curves_data[c]["recall_conf"]
        plt.plot(conf_grid, r, label=f"{cls_name}", linewidth=1.5)
        all_rs.append(r)

    mean_r = np.mean(all_rs, axis=0)
    plt.plot(conf_grid, mean_r, label="all classes", linewidth=3, color="blue")

    plt.title("Recall-Confidence Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Recall", fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "BoxR_curve.png"), dpi=300)
    plt.close()


def plot_confusion_matrix(cm, output_dir):
    """Plots and saves confusion matrix and normalized confusion matrix."""
    labels = CLASSES + ["background"]

    # Plot raw confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Plot normalized confusion matrix
    plt.figure(figsize=(8, 6))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = cm.astype(np.float64) / row_sums

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_normalized.png"), dpi=300)
    plt.close()


def plot_batch_visualizations(dataloader, model, output_dir):
    """Generates val_batch0_labels.jpg (ground truth) and val_batch0_pred.jpg (predictions) grids."""
    model.eval()
    images_to_plot = []
    gts_to_plot = []
    preds_to_plot = []

    max_images = 8

    with torch.no_grad():
        for images, targets in dataloader:
            device_images = [img.to(DEVICE) for img in images]
            outputs = model(device_images)

            for img, tgt, out in zip(images, targets, outputs):
                images_to_plot.append(img.cpu())
                gts_to_plot.append(tgt)
                preds_to_plot.append(
                    {
                        "boxes": out["boxes"].cpu().numpy(),
                        "labels": out["labels"].cpu().numpy(),
                        "scores": out["scores"].cpu().numpy(),
                    }
                )
                if len(images_to_plot) >= max_images:
                    break
            if len(images_to_plot) >= max_images:
                break

    if not images_to_plot:
        return

    num_imgs = len(images_to_plot)
    cols = min(4, num_imgs)
    rows = int(np.ceil(num_imgs / cols))

    # 1. Ground truth labels plot
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_imgs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(num_imgs):
        ax = axes[idx]
        img = images_to_plot[idx].permute(1, 2, 0).numpy()
        img = np.clip(img, 0.0, 1.0)
        ax.imshow(img)
        ax.axis("off")

        gt = gts_to_plot[idx]
        for box, label in zip(gt["boxes"], gt["labels"]):
            x1, y1, x2, y2 = box.numpy()
            w, h = x2 - x1, y2 - y1
            rect = plt.Rectangle(
                (x1, y1), w, h, fill=False, edgecolor="green", linewidth=2
            )
            ax.add_patch(rect)
            cls_name = CLASSES[label.item() - 1]
            ax.text(
                x1,
                max(0, y1 - 5),
                f"{cls_name}",
                color="green",
                fontsize=8,
                weight="bold",
                bbox=dict(facecolor="white", alpha=0.5, pad=0.1, edgecolor="none"),
            )

    for idx in range(num_imgs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_batch0_labels.jpg"), dpi=200)
    plt.close()

    # 2. Predictions plot
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_imgs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(num_imgs):
        ax = axes[idx]
        img = images_to_plot[idx].permute(1, 2, 0).numpy()
        img = np.clip(img, 0.0, 1.0)
        ax.imshow(img)
        ax.axis("off")

        pred = preds_to_plot[idx]
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            if score < 0.25:
                continue
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = plt.Rectangle(
                (x1, y1), w, h, fill=False, edgecolor="red", linewidth=2
            )
            ax.add_patch(rect)
            cls_name = CLASSES[label - 1]
            ax.text(
                x1,
                max(0, y1 - 5),
                f"{cls_name} {score:.2f}",
                color="red",
                fontsize=8,
                weight="bold",
                bbox=dict(facecolor="white", alpha=0.5, pad=0.1, edgecolor="none"),
            )

    for idx in range(num_imgs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_batch0_pred.jpg"), dpi=200)
    plt.close()


def evaluate_faster_rcnn(split="test"):
    """Evaluates Faster R-CNN best checkpoint on the specified split and saves plots in the run directory."""
    print(f"\nEvaluating FASTER R-CNN on the {split} set...")
    best_weights = os.path.join(RESULTS_DIR, "faster_rcnn", "phase2", "best.pt")

    if not os.path.exists(best_weights):
        print(f"Warning: Best weight file not found at {best_weights}. Skipping.")
        return None

    model = get_faster_rcnn_model(num_classes=NUM_CLASSES, freeze_backbone=False)
    load_compatible_weights(model, best_weights)
    model.to(DEVICE)
    model.eval()

    test_dataset = ActiveLearningFasterRCNNDataset(
        CLAHE_DATASET_DIR, split=split, img_size=IMG_SIZE, augment=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            device_images = [img.to(DEVICE) for img in images]
            outputs = model(device_images)

            for out, tgt in zip(outputs, targets):
                all_predictions.append(
                    {
                        "boxes": out["boxes"].cpu().numpy(),
                        "labels": out["labels"].cpu().numpy(),
                        "scores": out["scores"].cpu().numpy(),
                    }
                )
                all_targets.append(
                    {
                        "boxes": tgt["boxes"].cpu().numpy(),
                        "labels": tgt["labels"].cpu().numpy(),
                    }
                )

    metrics, pr_curves, curves_data, conf_grid = compute_faster_rcnn_metrics_and_curves(
        all_predictions, all_targets, iou_threshold=0.5
    )

    run_dir = os.path.join(RESULTS_DIR, "faster_rcnn", "phase2", split)
    os.makedirs(run_dir, exist_ok=True)
    plot_evaluation_curves(pr_curves, curves_data, conf_grid, metrics["mAP50"], run_dir)

    cm = compute_confusion_matrix(
        all_predictions, all_targets, num_classes=NUM_CLASSES, iou_threshold=0.5
    )
    plot_confusion_matrix(cm, run_dir)

    plot_batch_visualizations(test_loader, model, run_dir)

    metrics["model"] = "FASTER_RCNN"
    return metrics


def generate_comparison_plots(df, output_dir):
    """Generates comparison bar charts and saving comparison results csv."""
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(output_dir, "comparison_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Evaluator] Saved comparison metrics table to: {csv_path}")

    # Set styling
    sns.set_theme(style="whitegrid")
    plt.style.use("default")

    # Melt DataFrame for plotting class-specific APs
    ap_cols = [c for c in CLASSES if c in df.columns]
    df_melted = df.melt(
        id_vars=["model", "mAP50"],
        value_vars=ap_cols,
        var_name="Class",
        value_name="AP50",
    )

    # 1. Plot overall mAP50 comparison
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        x="model", y="mAP50", data=df, palette="viridis", hue="model", legend=False
    )
    plt.title(
        "Model Comparison - Mean Average Precision (mAP@0.5)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylabel("mAP@0.5", fontsize=12)
    plt.ylim(0, 1.05)

    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 8),
            textcoords="offset points",
            fontweight="bold",
        )

    plt.tight_layout()
    mAP_plot_path = os.path.join(output_dir, "map_comparison.png")
    plt.savefig(mAP_plot_path, dpi=300)
    plt.close()
    print(f"[Evaluator] Saved mAP comparison plot to: {mAP_plot_path}")

    # 2. Plot class-specific AP comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Class", y="AP50", hue="model", data=df_melted, palette="muted")
    plt.title(
        "Model Comparison - Class-Specific Average Precision (AP@0.5)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Target Classes", fontsize=12)
    plt.ylabel("AP@0.5", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title="Model", loc="upper right")
    plt.tight_layout()

    class_plot_path = os.path.join(output_dir, "class_ap_comparison.png")
    plt.savefig(class_plot_path, dpi=300)
    plt.close()
    print(f"[Evaluator] Saved class AP comparison plot to: {class_plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Finetuned Models on Dataset Splits"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Specify which split to evaluate on. Default: 'test'.",
    )
    parser.parse_args()
    args = parser.parse_args()

    dataset_yaml = os.path.join(RESULTS_DIR, "dataset_clahe.yaml")
    results_list = []

    # Evaluate YOLO
    yolo_metrics = evaluate_ultralytics("yolo", dataset_yaml, split=args.split)
    if yolo_metrics:
        results_list.append(yolo_metrics)

    # Evaluate RT-DETR
    rtdetr_metrics = evaluate_ultralytics("rtdetr", dataset_yaml, split=args.split)
    if rtdetr_metrics:
        results_list.append(rtdetr_metrics)

    # Evaluate Faster R-CNN
    faster_rcnn_metrics = evaluate_faster_rcnn(split=args.split)
    if faster_rcnn_metrics:
        results_list.append(faster_rcnn_metrics)

    if not results_list:
        print("\nNo trained models found to evaluate. Make sure to run train.py first.")
        return

    # Build and print comparison table
    df = pd.DataFrame(results_list)

    print("\n=======================================================")
    print(f"             {args.split.upper()} SET EVALUATION SUMMARY               ")
    print("=======================================================")
    print(df.to_markdown(index=False))
    print("=======================================================\n")

    # Generate comparison plots and save outputs
    comparison_dir = os.path.join(RESULTS_DIR, "comparison")
    generate_comparison_plots(df, comparison_dir)


if __name__ == "__main__":
    main()
