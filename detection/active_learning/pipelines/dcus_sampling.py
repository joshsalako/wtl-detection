#!/usr/bin/env python3
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
import os

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEARNING_DIR = os.path.dirname(PIPELINES_DIR)
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)

from central_config import (
    CLASSES,
    DCUS_CONF_THRESHOLD,
    CURATION_TARGET_CLASS,
    DCUS_POOL_RATIO_UNCERTAIN,
    DCUS_POOL_RATIO_CERTAIN,
    DCUS_POOL_RATIO_RANDOM,
    DCUS_BUDGET_MULTIPLIER,
)


def load_detector_model(model_path, device, num_classes=3):
    """Loads YOLO, RT-DETR, or Faster R-CNN model dynamically."""
    model_name = os.path.basename(model_path).lower()
    if "rtdetr" in model_name or "rtdetr" in model_path.lower():
        from ultralytics import RTDETR

        print(f"DCUS: Loading RT-DETR model from {model_path}")
        return RTDETR(model_path)
    elif "faster_rcnn" in model_name or "faster_rcnn" in model_path.lower():
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        print(f"DCUS: Loading Faster R-CNN model from {model_path}")
        model = fasterrcnn_resnet50_fpn_v2()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    else:
        from ultralytics import YOLO

        print(f"DCUS: Loading YOLO model from {model_path}")
        return YOLO(model_path)


def get_ap_from_json(model_path):
    """Retrieves class-wise AP50 scores from results_dict.json if available."""
    # Sibling eval folders search
    run_dir = os.path.dirname(os.path.dirname(model_path))
    for split in ["val_eval", "test_eval"]:
        json_path = os.path.join(run_dir, split, "results_dict.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                ap_dict = {}
                for k, v in data.items():
                    if k.startswith("AP50_"):
                        class_name = k.replace("AP50_", "")
                        ap_dict[class_name] = float(v)
                if ap_dict:
                    print(f"DCUS: Loaded class APs from {json_path}")
                    return ap_dict
            except Exception as e:
                print(f"DCUS Warning: Error reading {json_path}: {e}")
    return None


def compute_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0


def load_yolo_labels(label_path, img_width, img_height):
    """Loads and converts YOLO format labels into absolute coordinates."""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    x1 = (xc - w / 2) * img_width
                    y1 = (yc - h / 2) * img_height
                    x2 = (xc + w / 2) * img_width
                    y2 = (yc + h / 2) * img_height
                    boxes.append({"class_id": cls_id, "bbox": [x1, y1, x2, y2]})
    return boxes


def run_test_inference(model, test_dir, device, xi=0.5, max_images=200):
    """Runs inference on a test subset to compute empirical class difficulties using PPAL formula."""
    # Find test images and labels
    img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    test_images_dir = None
    test_labels_dir = None

    # Try different structures
    for img_sub, lbl_sub in [
        ("test/images", "test/labels"),
        ("images/test", "labels/test"),
    ]:
        img_p = os.path.join(test_dir, img_sub)
        lbl_p = os.path.join(test_dir, lbl_sub)
        if os.path.exists(img_p):
            test_images_dir = img_p
            test_labels_dir = lbl_p
            break

    if not test_images_dir or not os.path.exists(test_images_dir):
        print("DCUS: Test images directory not found.")
        return None

    img_files = sorted(
        [
            os.path.join(test_images_dir, f)
            for f in os.listdir(test_images_dir)
            if os.path.splitext(f)[1].lower() in img_extensions
        ]
    )[:max_images]

    if not img_files:
        print("DCUS: No test images found.")
        return None

    print(f"DCUS: Running test inference on {len(img_files)} images...")

    class_difficulties = {cid: [] for cid in range(len(CLASSES))}

    is_ultralytics = hasattr(model, "predict")

    for img_path in img_files:
        # Get dimensions
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            continue

        # Load ground truths
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = (
            os.path.join(test_labels_dir, base_name + ".txt") if test_labels_dir else ""
        )
        gt_boxes = load_yolo_labels(label_path, w, h)

        # Predict
        preds = []
        if is_ultralytics:
            res = model.predict(img_path, imgsz=640, verbose=False, device=device)[0]
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                preds.append({"class_id": cls_id, "conf": conf, "bbox": coords})
        else:  # Faster R-CNN PyTorch model
            with torch.no_grad():
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    continue
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                outputs = model(img_tensor)
                out = outputs[0]
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                for j in range(len(scores)):
                    preds.append(
                        {
                            "class_id": int(labels[j])
                            - 1,  # Convert 1-indexed back to 0-indexed
                            "conf": float(scores[j]),
                            "bbox": boxes[j].tolist(),
                        }
                    )

        # Match predicted boxes to ground truth
        for pred in preds:
            pred_cls = pred["class_id"]
            if pred_cls not in class_difficulties:
                continue
            # Find best match in ground truth of the same class
            best_iou = 0.0
            for gt in gt_boxes:
                if gt["class_id"] == pred_cls:
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou

            # Match condition
            if best_iou >= 0.5:
                classification_prob = pred["conf"]
                localization_acc = best_iou
                # PPAL Instance-level difficulty calculation: q = 1 - P^xi * IoU^(1-xi)
                difficulty = 1.0 - (classification_prob**xi) * (
                    localization_acc ** (1.0 - xi)
                )
                class_difficulties[pred_cls].append(difficulty)

    # Calculate average difficulty
    avg_difficulties = {}
    for cid, diffs in class_difficulties.items():
        if diffs:
            avg_difficulties[CLASSES[cid]] = float(np.mean(diffs))
        else:
            avg_difficulties[CLASSES[cid]] = 1.0  # Max PPAL difficulty is 1.0

    return avg_difficulties


def compute_shannon_entropy(conf, num_classes):
    """Computes Shannon entropy for top-1 class confidence score."""
    if num_classes <= 1:
        return 0.0
    p = np.clip(conf, 1e-6, 1.0 - 1e-6)
    p_other = np.clip((1.0 - p) / (num_classes - 1), 1e-6, 1.0 - 1e-6)
    return -p * np.log(p) - (num_classes - 1) * p_other * np.log(p_other)


def main():
    parser = argparse.ArgumentParser(
        description="DCUS Query Uncertainty Sampling Script."
    )
    parser.add_argument(
        "--predictions_csv", type=str, required=True, help="Path topredictions CSV."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save output uncertainty CSV.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Path to test dataset directory.",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=0.5,
        help="Balance between classification and localization error.",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Logarithmic smoothing scale factor."
    )
    parser.add_argument(
        "--beta", type=float, default=2.0, help="Difficulty weight multiplier/cap."
    )
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device.")
    parser.add_argument(
        "--budget", type=int, default=100, help="Annotation budget for the cycle."
    )

    args = parser.parse_args()

    # Load predictions
    print(f"DCUS: Loading predictions from {args.predictions_csv}")
    df = pd.read_csv(args.predictions_csv)

    # Filter out sub-threshold false positives to prevent them from inflating entropy
    initial_len = len(df)
    df = df[df["confidence"] >= DCUS_CONF_THRESHOLD].copy()
    print(
        f"DCUS: Filtered out {initial_len - len(df)} predictions with confidence < {DCUS_CONF_THRESHOLD}"
    )

    if df.empty:
        print("DCUS: Empty predictions CSV. Copying to output.")
        df["entropy"] = []
        df["difficulty_coeff"] = []
        df["box_uncertainty"] = []
        df["uncertainty"] = []
        df.to_csv(args.output_csv, index=False)
        return

    num_classes = len(CLASSES)

    # 1. Resolve Class Difficulty weights
    difficulty_weights = {}

    # Attempt empirical test matching
    empirical_difficulties = None
    if args.test_dir and os.path.exists(args.test_dir):
        try:
            model = load_detector_model(args.model_path, args.device, num_classes)
            empirical_difficulties = run_test_inference(
                model, args.test_dir, args.device, xi=args.xi
            )
            # Free model weights
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"DCUS Warning: Test inference failed: {e}")

    # PPAL Logarithmic Smoothing constant
    gamma = np.exp(1.0 / args.alpha) - 1.0

    if empirical_difficulties:
        print(
            "DCUS: Computing difficulty coefficients from test matching statistics (PPAL):"
        )
        for name, diff in empirical_difficulties.items():
            w = 1.0 + (args.alpha / args.beta) * np.log(1.0 + gamma * diff)
            difficulty_weights[name] = w
            print(
                f"  Class '{name}': difficulty (di) = {diff:.4f}, weight (w_c) = {w:.4f}"
            )
    else:
        # Fall back to results_dict.json AP50 values
        ap_dict = get_ap_from_json(args.model_path)
        if ap_dict:
            print(
                "DCUS: Computing difficulty coefficients from class AP50 scores (PPAL fallback):"
            )
            for idx, name in enumerate(CLASSES):
                ap = ap_dict.get(name, 0.0)
                diff = 1.0 - ap
                w = 1.0 + (args.alpha / args.beta) * np.log(1.0 + gamma * diff)
                difficulty_weights[name] = w
                print(
                    f"  Class '{name}': AP50 = {ap:.4f}, diff = {diff:.4f}, weight (w_c) = {w:.4f}"
                )
        else:
            # Fall back to hardcoded default values
            print("DCUS: Falling back to default class difficulty weights:")
            for idx, name in enumerate(CLASSES):
                if name == CURATION_TARGET_CLASS:
                    # Give target class max difficulty
                    diff = 1.0
                else:
                    diff = 0.5
                w = 1.0 + (args.alpha / args.beta) * np.log(1.0 + gamma * diff)
                difficulty_weights[name] = w
                print(
                    f"  Class '{name}': fallback diff = {diff:.4f}, weight (w_c) = {w:.4f}"
                )

    # 2. Compute entropy for each prediction box
    print("DCUS: Computing object-level Shannon entropy...")
    df["entropy"] = df["confidence"].apply(
        lambda p: compute_shannon_entropy(p, num_classes)
    )

    # 3. Apply difficulty coefficients
    df["difficulty_coeff"] = df["class_name"].map(difficulty_weights).fillna(2.0)
    df["box_uncertainty"] = df["difficulty_coeff"] * df["entropy"]

    # 4. Aggregate image-level uncertainty (sum of box difficulties)
    print("DCUS: Aggregating image-level uncertainty scores...")
    image_uncertainties = df.groupby("image_path")["box_uncertainty"].sum().to_dict()
    df["uncertainty"] = df["image_path"].map(image_uncertainties)

    # 5. Pre-filter top K images using a distributed sampling strategy based on budget
    top_k = args.budget * DCUS_BUDGET_MULTIPLIER
    print(
        f"DCUS: Filtering down to {top_k} distributed images (Budget: {args.budget} x Multiplier: {DCUS_BUDGET_MULTIPLIER})"
    )
    print(
        f"DCUS: Distribution strategy: "
        f"{int(DCUS_POOL_RATIO_UNCERTAIN * 100)}% uncertain, "
        f"{int(DCUS_POOL_RATIO_CERTAIN * 100)}% certain, "
        f"{int(DCUS_POOL_RATIO_RANDOM * 100)}% random."
    )

    # Assign a primary class to each image based on its most confident prediction
    idx_max_conf = df.groupby("image_path")["confidence"].idxmax()
    primary_classes = df.loc[idx_max_conf, ["image_path", "class_name"]]

    unique_images = (
        df[["image_path", "uncertainty"]]
        .drop_duplicates()
        .merge(primary_classes, on="image_path", how="left")
        .sort_values(by="uncertainty", ascending=False)
    )

    total_available = len(unique_images)
    top_k = min(top_k, total_available)

    n_uncertain = int(top_k * DCUS_POOL_RATIO_UNCERTAIN)
    n_certain = int(top_k * DCUS_POOL_RATIO_CERTAIN)
    n_random = top_k - n_uncertain - n_certain

    # Get most uncertain
    uncertain_pool = unique_images.head(n_uncertain)
    remaining_images = unique_images.iloc[n_uncertain:]

    # Get most certain from remaining, equally distributed across classes
    if n_certain > 0 and len(remaining_images) > 0:
        classes_present = remaining_images["class_name"].dropna().unique()
        n_per_class = (
            max(1, n_certain // len(classes_present)) if len(classes_present) > 0 else 0
        )

        certain_pool = remaining_images.groupby("class_name", group_keys=False).apply(
            lambda x: x.tail(n_per_class)
        )

        # If short due to some classes lacking samples, backfill from absolute lowest uncertainty overall
        shortfall = n_certain - len(certain_pool)
        if shortfall > 0:
            leftovers = remaining_images.loc[
                ~remaining_images.index.isin(certain_pool.index)
            ]
            certain_pool = pd.concat([certain_pool, leftovers.tail(shortfall)])

        # Ensure we don't exceed n_certain
        if len(certain_pool) > n_certain:
            certain_pool = certain_pool.tail(n_certain)

        # Remove certain from remaining
        remaining_images = remaining_images.loc[
            ~remaining_images.index.isin(certain_pool.index)
        ]
    else:
        certain_pool = pd.DataFrame()

    # Get random from what's left using stratified quantile sampling to spread across variance
    if len(remaining_images) > 0 and n_random > 0:
        remaining_images = remaining_images.copy()

        # Determine number of bins (max 10)
        n_bins = min(10, len(remaining_images))

        try:
            # Try splitting into equal frequency quantiles
            remaining_images["bin"] = pd.qcut(
                remaining_images["uncertainty"], q=n_bins, duplicates="drop"
            )
        except Exception:
            # Fallback if quantiles fail (e.g., too many identical values)
            remaining_images["bin"] = pd.cut(
                remaining_images["uncertainty"], bins=n_bins
            )

        # Sample proportionally from each bin
        bin_counts = remaining_images["bin"].nunique()
        samples_per_bin = max(1, n_random // bin_counts)

        random_pool = remaining_images.groupby(
            "bin", observed=False, group_keys=False
        ).apply(lambda x: x.sample(n=min(len(x), samples_per_bin), random_state=42))

        # If we are short of n_random due to uneven bins or rounding, sample the rest uniformly
        shortfall = n_random - len(random_pool)
        if shortfall > 0:
            leftovers = remaining_images.loc[
                ~remaining_images.index.isin(random_pool.index)
            ]
            if len(leftovers) > 0:
                extra = leftovers.sample(
                    n=min(shortfall, len(leftovers)), random_state=42
                )
                random_pool = pd.concat([random_pool, extra])

        # If we exceeded n_random, trim it
        if len(random_pool) > n_random:
            random_pool = random_pool.sample(n=n_random, random_state=42)

        # Clean up
        random_pool = random_pool.drop(columns=["bin"], errors="ignore")
    else:
        random_pool = pd.DataFrame()

    # Combine pools
    selected_images_df = pd.concat([uncertain_pool, certain_pool, random_pool])
    top_image_paths = selected_images_df["image_path"]

    df = df[df["image_path"].isin(top_image_paths)]

    # Save output
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"DCUS: Saved uncertainty-scored predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
