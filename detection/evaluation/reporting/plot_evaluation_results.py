import os
import sys
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.parse
from sklearn.metrics import roc_curve


plt.rcParams.update(
    {
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 20,
        "lines.linewidth": 3,
    }
)

MODEL_NAME_MAP = {
    "yolo_clahe": "YOLO",
    "faster_rcnn_clahe": "Faster R-CNN",
    "rtdetr_clahe": "RT-DETR",
    "megadetector": "MegaDetector",
}

MODEL_COLORS = {
    "yolo_clahe": "tab:blue",
    "faster_rcnn_clahe": "tab:orange",
    "rtdetr_clahe": "tab:green",
    "megadetector": "tab:red",
}


def format_cycle(cycle_str):
    if cycle_str == "N/A" or not cycle_str.startswith("cycle_"):
        return ""
    try:
        c_num = int(cycle_str.split("_")[1])
        return f"Cycle {c_num + 1}"
    except:
        return cycle_str


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_utils.config import (
    FILES_DIR,
    PLOTS_DIR,
    EVAL_DIR,
    RESULTS_DIR,
    TEST_DIR,
    CLASSES,
    WLT_PLOT_TITLE,
    AGNOSTIC_PLOT_TITLE,
    WLT_PREFIX,
    AGNOSTIC_PREFIX,
    WLT_CLASS_ID,
)


def plot_roc(df_json_path, output_prefix, title_suffix):
    """Plots ROC curves for image-level binary classification evaluation.
    PR curves are NOT plotted here; this evaluation uses ROC + confusion matrix.
    """
    if not os.path.exists(df_json_path):
        return

    with open(df_json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if df.empty:
        return

    # We only plot for test_full dataset
    df_test = df[df["Dataset"] == "test_full"]

    cycles_to_plot = ["cycle_0", "cycle_1", "cycle_2", "cycle_3", "cycle_4"]

    for current_cycle in cycles_to_plot:
        # Filter to only the current cycle and megadetector
        df_cycle = df_test[
            df_test["Cycle"].isin([current_cycle, "N/A"])
            | (df_test["Model"] == "megadetector")
        ]

        if df_cycle.empty:
            continue

        plt.figure(figsize=(12, 10))
        for _, row in df_cycle.iterrows():
            model_clean = MODEL_NAME_MAP.get(row["Model"], row["Model"])
            cycle_clean = format_cycle(row["Cycle"])

            # Format label cleanly
            if cycle_clean:
                label = f"{model_clean} {cycle_clean} (AUC: {row['AUC']:.3f})"
            else:
                label = f"{model_clean} (AUC: {row['AUC']:.3f})"

            color = MODEL_COLORS.get(row["Model"], "black")

            y_true = row["y_true"]
            y_score = row["y_score"]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            plt.plot(fpr, tpr, label=label, color=color)

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"{output_prefix}_{current_cycle}_roc_curve.pdf"),
            format="pdf",
        )
        plt.close()


def plot_confusion_matrices(df_json_path, output_prefix, title_suffix):
    if not os.path.exists(df_json_path):
        return

    with open(df_json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df_test = df[df["Dataset"] == "test_full"]

    # Determine positive label based on title_suffix
    pos_label = "WLT" if "WLT" in title_suffix else "Object"

    for _, row in df_test.iterrows():
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])

        plt.figure(figsize=(8, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 36},
            xticklabels=["Background", pos_label],
            yticklabels=["Background", pos_label],
        )
        plt.xlabel("Predicted", fontsize=22)
        plt.ylabel("Ground truth", fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()

        # Clean names for filename
        model_clean = row["Model"].replace("_clahe", "")
        cycle_val = str(row["Cycle"])
        cycle_clean = "baseline" if cycle_val in ["nan", "N/A"] else cycle_val

        fname = f"{output_prefix}_{model_clean}_{cycle_clean}_cm.pdf"
        plt.savefig(os.path.join(PLOTS_DIR, fname), format="pdf")
        plt.close()


def _load_gt_wlt(gt_dir):
    """Returns {img_name: 1_if_contains_wlt} for images in gt_dir."""
    gt = {}
    lbl_dir = os.path.join(gt_dir, "labels")
    img_dir = os.path.join(gt_dir, "images")
    if not os.path.exists(img_dir):
        return gt
    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        base = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(lbl_dir, f"{base}.txt")
        has_wlt = False
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1 and int(parts[0]) == WLT_CLASS_ID:
                        has_wlt = True
                        break
        gt[img_name] = int(has_wlt)
    return gt


def _format_cvat(path):
    if "/srv/shared_leopard_toad/" in path:
        rel = path.split("/srv/shared_leopard_toad/")[-1]
        return rel.replace("/", "_").replace("\\", "_")
    return os.path.basename(path)


def plot_detection_wlt_pr_curve():
    """Plots detection-level WLT PR curves: 5 figures (one per cycle),
    each containing 3 curves (YOLO, Faster R-CNN, RT-DETR).

    At each confidence threshold a box is a TP if it matches a WLT GT box at
    IoU >= 0.5; otherwise FP.  Unmatched WLT GT boxes are FN.
    MegaDetector is excluded (class-agnostic, no WLT class score).
    """
    import glob

    gt = _load_gt_wlt(TEST_DIR)
    if not gt:
        print("  [skip] GT directory not found or empty")
        return

    # Pre-load GT WLT boxes once (shared across models)
    gt_wlt_boxes_by_img = {}
    for img_name in gt:
        base = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(TEST_DIR, "labels", f"{base}.txt")
        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5 and int(parts[0]) == WLT_CLASS_ID:
                        cx, cy, w, h = map(float, parts[1:])
                        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        gt_wlt_boxes_by_img[img_name] = boxes

    total_gt_global = sum(len(b) for b in gt_wlt_boxes_by_img.values())

    cycles_order = ["cycle_0", "cycle_1", "cycle_2", "cycle_3", "cycle_4"]
    models = ["yolo_clahe", "faster_rcnn_clahe", "rtdetr_clahe"]

    # --- Step 1: compute PR curve data for every (model, cycle) pair ---
    pr_data = {}  # (model, cycle) -> {"precisions": arr, "recalls": arr, "ap": float}

    for model in models:
        for cycle in cycles_order:
            pattern = os.path.join(
                RESULTS_DIR, model, f"test_full_{cycle}_filtered.json"
            )
            matches = glob.glob(pattern)
            if not matches:
                continue

            with open(matches[0]) as f:
                preds_list = json.load(f)

            pred_by_name = {}
            for item in preds_list:
                key = _format_cvat(item["path"])
                pred_by_name[key] = item.get("predictions", [])
                encoded_key = urllib.parse.quote(key)
                if encoded_key != key:
                    pred_by_name[encoded_key] = item.get("predictions", [])

            scores = []
            total_gt = 0

            for img_name, gt_wlt_boxes in gt_wlt_boxes_by_img.items():
                total_gt += len(gt_wlt_boxes)
                matched_gt = set()

                pred_boxes = sorted(
                    [
                        b
                        for b in pred_by_name.get(img_name, [])
                        if b["cls"] == WLT_CLASS_ID
                    ],
                    key=lambda b: b["conf"],
                    reverse=True,
                )

                for pb in pred_boxes:
                    cx, cy, w, h = pb["bbox"]
                    px1, py1, px2, py2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

                    best_iou, best_j = 0.0, -1
                    for j, (gx1, gy1, gx2, gy2) in enumerate(gt_wlt_boxes):
                        if j in matched_gt:
                            continue
                        ix1, iy1 = max(px1, gx1), max(py1, gy1)
                        ix2, iy2 = min(px2, gx2), min(py2, gy2)
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        union = (
                            (px2 - px1) * (py2 - py1)
                            + (gx2 - gx1) * (gy2 - gy1)
                            - inter
                        )
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou, best_j = iou, j

                    is_tp = int(best_iou >= 0.5)
                    if is_tp:
                        matched_gt.add(best_j)
                    scores.append((pb["conf"], is_tp))

            if total_gt == 0 or not scores:
                continue

            scores.sort(key=lambda x: x[0], reverse=True)
            is_tps = np.array([s[1] for s in scores])
            tp_cum = np.cumsum(is_tps)
            fp_cum = np.cumsum(1 - is_tps)

            prec = tp_cum / (tp_cum + fp_cum)
            rec = tp_cum / total_gt

            # Sentinel for clean curve origin
            prec = np.concatenate([[1.0], prec])
            rec = np.concatenate([[0.0], rec])

            ap = float(np.trapezoid(prec, rec))

            # Drop the curve vertically at the last recall point
            if rec[-1] < 1.0:
                prec = np.concatenate([prec, [0.0]])
                rec = np.concatenate([rec, [rec[-1]]])

            pr_data[(model, cycle)] = {"precisions": prec, "recalls": rec, "ap": ap}

    # --- Step 2: one plot per cycle, 3 model curves each ---
    for cycle in cycles_order:
        plt.figure(figsize=(12, 10))
        plotted = False

        for model in models:
            if (model, cycle) not in pr_data:
                continue
            d = pr_data[(model, cycle)]
            model_clean = MODEL_NAME_MAP.get(model, model)
            color = MODEL_COLORS.get(model, "black")
            plt.plot(
                d["recalls"],
                d["precisions"],
                label=f"{model_clean} (AP@0.5: {d['ap']:.3f})",
                color=color,
            )
            plotted = True

        if not plotted:
            plt.close()
            continue

        cycle_clean = format_cycle(cycle)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = f"detection_wlt_pr_{cycle}.pdf"
        plt.savefig(os.path.join(PLOTS_DIR, fname), format="pdf")
        plt.close()
        print(f"  Saved {fname}")


def generate_readme():
    det_csv = os.path.join(FILES_DIR, "detection_metrics.csv")
    wlt_csv = os.path.join(FILES_DIR, "image_level_wlt.csv")
    ag_csv = os.path.join(FILES_DIR, "image_level_agnostic.csv")

    content = "# Active Learning Models Evaluation\n\n"
    content += "This repository contains the evaluation results of YOLO, Faster R-CNN, and RT-DETR models across 5 active learning cycles (0-4), as well as a MegaDetector baseline.\n\n"
    content += "## 1. Detection Level Metrics (mAP)\n\n"

    if os.path.exists(det_csv):
        df_det = pd.read_csv(det_csv).sort_values(by=["Dataset", "Model", "Cycle"])
        content += df_det.to_markdown(index=False) + "\n\n"

    content += "## 2. Image Level Metrics (WLT vs Background)\n\n"
    if os.path.exists(wlt_csv):
        df_wlt = pd.read_csv(wlt_csv).sort_values(by=["Dataset", "Model", "Cycle"])
        content += df_wlt.to_markdown(index=False) + "\n\n"

    content += "## 3. Image Level Metrics (Class Agnostic vs Background)\n\n"
    if os.path.exists(ag_csv):
        df_ag = pd.read_csv(ag_csv).sort_values(by=["Dataset", "Model", "Cycle"])
        content += df_ag.to_markdown(index=False) + "\n\n"

    content += "## 4. Plots\n\n"
    content += "- ROC and PR curves are saved as PDFs in `results/plots/`.\n"
    content += "- Confusion matrices are also saved as PDFs in the same directory.\n"

    readme_path = os.path.join(FILES_DIR, "final_evaluation_results.md")
    with open(readme_path, "w") as f:
        f.write(content)


def main():
    print(f"Generating plots for {WLT_PLOT_TITLE} (Image-Level)...")
    plot_roc(
        os.path.join(FILES_DIR, "image_level_wlt_full.json"), WLT_PREFIX, WLT_PLOT_TITLE
    )
    plot_confusion_matrices(
        os.path.join(FILES_DIR, "image_level_wlt_full.json"), WLT_PREFIX, WLT_PLOT_TITLE
    )

    print(f"Generating plots for {AGNOSTIC_PLOT_TITLE} (Image-Level)...")
    plot_roc(
        os.path.join(FILES_DIR, "image_level_agnostic_full.json"),
        AGNOSTIC_PREFIX,
        AGNOSTIC_PLOT_TITLE,
    )
    plot_confusion_matrices(
        os.path.join(FILES_DIR, "image_level_agnostic_full.json"),
        AGNOSTIC_PREFIX,
        AGNOSTIC_PLOT_TITLE,
    )

    print("Generating Detection-Level WLT PR curves...")
    plot_detection_wlt_pr_curve()

    print("Generating README.md...")
    generate_readme()
    print("All tasks complete.")


if __name__ == "__main__":
    main()
