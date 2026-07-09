import os
import sys
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
        "lines.linewidth": 3,
    }
)

MODEL_NAME_MAP = {
    "yolo_clahe": "YOLO",
    "faster_rcnn_clahe": "Faster R-CNN",
    "rtdetr_clahe": "RT-DETR",
    "megadetector": "MegaDetector",
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
    CLASSES,
    WLT_PLOT_TITLE,
    AGNOSTIC_PLOT_TITLE,
    WLT_PREFIX,
    AGNOSTIC_PREFIX,
)


def plot_roc_pr(df_json_path, output_prefix, title_suffix):
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

            y_true = row["y_true"]
            y_score = row["y_score"]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            plt.plot(fpr, tpr, label=label)

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"{output_prefix}_{current_cycle}_roc_curve.pdf"),
            format="pdf",
        )
        plt.close()

        # PR Curve
        plt.figure(figsize=(12, 10))
        for _, row in df_cycle.iterrows():
            model_clean = MODEL_NAME_MAP.get(row["Model"], row["Model"])
            cycle_clean = format_cycle(row["Cycle"])

            if cycle_clean:
                label = f"{model_clean} {cycle_clean} (Best F1: {row['Best_F1']:.3f})"
            else:
                label = f"{model_clean} (Best F1: {row['Best_F1']:.3f})"

            y_true = row["y_true"]
            y_score = row["y_score"]
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            plt.plot(recall, precision, label=label)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"{output_prefix}_{current_cycle}_pr_curve.pdf"),
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
    plot_roc_pr(
        os.path.join(FILES_DIR, "image_level_wlt_full.json"), WLT_PREFIX, WLT_PLOT_TITLE
    )
    plot_confusion_matrices(
        os.path.join(FILES_DIR, "image_level_wlt_full.json"), WLT_PREFIX, WLT_PLOT_TITLE
    )

    print(f"Generating plots for {AGNOSTIC_PLOT_TITLE} (Image-Level)...")
    plot_roc_pr(
        os.path.join(FILES_DIR, "image_level_agnostic_full.json"),
        AGNOSTIC_PREFIX,
        AGNOSTIC_PLOT_TITLE,
    )
    plot_confusion_matrices(
        os.path.join(FILES_DIR, "image_level_agnostic_full.json"),
        AGNOSTIC_PREFIX,
        AGNOSTIC_PLOT_TITLE,
    )

    print("Generating README.md...")
    generate_readme()
    print("All tasks complete.")


if __name__ == "__main__":
    main()
