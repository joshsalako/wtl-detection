#!/usr/bin/env python3
import os
import sys
import math
import cv2
import csv
import argparse
from pathlib import Path
import torch
import numpy as np
from ultralytics import RTDETR, YOLO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEARNING_DIR = os.path.dirname(PIPELINES_DIR)
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)

from central_config import (
    CLASSES,
    DETECTION_THRESHOLDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_IMG_SIZE,
    ULTRALYTICS_BATCH_SIZE,
    FASTER_RCNN_BATCH_SIZE,
    FASTER_RCNN_INFERENCE_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_DEVICE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_OCCURRENCE_THRESHOLD,
    UNLABELED_POOL_DIRS,
    EXCLUDED_CAMERAS,
    INFERENCE_CONF_THRESHOLD,
)

from filter_static_false_positives import filter_static_detections
from faster_rcnn_utils import get_faster_rcnn_model, load_compatible_weights


class ActiveLearningInferenceDataset(Dataset):
    """Custom Dataset for fast, asynchronous image loading and CLAHE preprocessing."""

    def __init__(
        self, image_paths, img_size=640, apply_clahe_flag=True, model_type="yolo"
    ):
        self.image_paths = image_paths
        self.img_size = img_size
        self.apply_clahe_flag = apply_clahe_flag
        self.model_type = model_type

    def __len__(self):
        return len(self.image_paths)

    def apply_clahe(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        return cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_bgr = cv2.imread(str(img_path))

        if img_bgr is None:
            # Provide default dimensions for corrupted images
            orig_h, orig_w = 3024, 5376
            is_corrupt = True
        else:
            orig_h, orig_w = img_bgr.shape[:2]
            is_corrupt = False

        if self.model_type in ["yolo", "faster_rcnn"]:
            if orig_h < orig_w:
                new_h = 640
                new_w = math.ceil((orig_w * (640 / orig_h)) / 32.0) * 32
            else:
                new_w = 640
                new_h = math.ceil((orig_h * (640 / orig_w)) / 32.0) * 32
        else:
            if orig_h > orig_w:
                new_h = 640
                new_w = math.ceil((orig_w * (640 / orig_h)) / 32.0) * 32
            else:
                new_w = 640
                new_h = math.ceil((orig_h * (640 / orig_w)) / 32.0) * 32

        if is_corrupt:
            if self.model_type in ["yolo", "rtdetr", "megadetector"]:
                return (
                    np.zeros((new_h, new_w, 3), dtype=np.uint8),
                    str(img_path),
                    orig_w,
                    orig_h,
                )
            else:
                return (
                    torch.zeros((3, new_h, new_w), dtype=torch.uint8),
                    str(img_path),
                    orig_w,
                    orig_h,
                )

        # Resize before CLAHE to drastically reduce CPU time
        img_resized = cv2.resize(
            img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        if self.model_type == "megadetector":
            pass  # MegaDetector expects native RGB, do no transformation
        elif self.apply_clahe_flag:
            img_resized = self.apply_clahe(img_resized)
        else:
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        if self.model_type in ["yolo", "rtdetr", "megadetector"]:
            return img_rgb, str(img_path), orig_w, orig_h

        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)

        return img_tensor, str(img_path), orig_w, orig_h


def inference_collate_fn(batch):
    imgs = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    orig_ws = [item[2] for item in batch]
    orig_hs = [item[3] for item in batch]

    if isinstance(imgs[0], torch.Tensor):
        try:
            imgs = torch.stack(imgs, dim=0)
        except Exception:
            pass
    return imgs, paths, orig_ws, orig_hs


def process_all_images(
    images,
    model,
    model_type,
    img_size,
    batch_size,
    device,
    all_writer,
    apply_clahe_flag=True,
):
    dataset = ActiveLearningInferenceDataset(
        images,
        img_size=img_size,
        apply_clahe_flag=apply_clahe_flag,
        model_type=model_type,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=inference_collate_fn,
    )

    grand_total_boxes = 0

    for batch_imgs, img_paths, orig_ws, orig_hs in tqdm(
        dataloader, desc="Processing Unlabeled Pool"
    ):
        if model_type in ["yolo", "rtdetr"]:
            img_size_arg = 1152 if model_type == "yolo" else 640
            
            preds = model.predict(
                source=batch_imgs,
                imgsz=img_size_arg,
                device=device,
                conf=INFERENCE_CONF_THRESHOLD,
                verbose=False,
                stream=False,
                half=(device != "cpu"),
            )

            for i, pred in enumerate(preds):
                img_path_str = img_paths[i]
                orig_w = orig_ws[i]
                orig_h = orig_hs[i]
                subfolder_name = Path(img_path_str).parent.name

                if pred.boxes is not None and len(pred.boxes) > 0:
                    boxes_norm = pred.boxes.xyxyn.cpu().numpy()
                    confs = pred.boxes.conf.cpu().numpy()
                    clss = pred.boxes.cls.cpu().numpy()

                    for b, conf, cls_id in zip(boxes_norm, confs, clss):
                        cls_id = int(cls_id)
                        if conf >= DETECTION_THRESHOLDS.get(cls_id, 0.25):
                            class_name = model.names.get(
                                cls_id,
                                CLASSES[cls_id]
                                if cls_id < len(CLASSES)
                                else f"class_{cls_id}",
                            )
                            x1, y1, x2, y2 = (
                                b[0] * orig_w,
                                b[1] * orig_h,
                                b[2] * orig_w,
                                b[3] * orig_h,
                            )

                            row_data = [
                                img_path_str,
                                Path(img_path_str).name,
                                subfolder_name,
                                cls_id,
                                class_name,
                                f"{conf:.4f}",
                                round(x1, 1),
                                round(y1, 1),
                                round(x2, 1),
                                round(y2, 1),
                            ]
                            all_writer.writerow(row_data)
                            grand_total_boxes += 1

        elif model_type == "faster_rcnn":
            if isinstance(batch_imgs, torch.Tensor):
                batch_imgs = batch_imgs.to(device, non_blocking=True).float() / 255.0
            else:
                batch_imgs = [img.to(device).float() / 255.0 for img in batch_imgs]

            with torch.no_grad():
                if device != "cpu":
                    with torch.amp.autocast("cuda"):
                        outputs = model(batch_imgs)
                else:
                    outputs = model(batch_imgs)

            if isinstance(batch_imgs, torch.Tensor):
                new_h, new_w = batch_imgs.shape[2], batch_imgs.shape[3]
            else:
                new_h, new_w = batch_imgs[0].shape[1], batch_imgs[0].shape[2]

            for out, img_path_str, ow, oh in zip(outputs, img_paths, orig_ws, orig_hs):
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy() - 1
                boxes = out["boxes"].cpu().numpy()
                orig_w = ow
                orig_h = oh
                subfolder_name = Path(img_path_str).parent.name

                for s, l, b in zip(scores, labels, boxes):
                    cls_id = int(l)
                    if s >= DETECTION_THRESHOLDS.get(cls_id, 0.25):
                        class_name = (
                            CLASSES[cls_id]
                            if cls_id < len(CLASSES)
                            else f"class_{cls_id}"
                        )
                        x1 = (b[0] / new_w) * orig_w
                        y1 = (b[1] / new_h) * orig_h
                        x2 = (b[2] / new_w) * orig_w
                        y2 = (b[3] / new_h) * orig_h

                        row_data = [
                            img_path_str,
                            Path(img_path_str).name,
                            subfolder_name,
                            cls_id,
                            class_name,
                            f"{s:.4f}",
                            round(x1, 1),
                            round(y1, 1),
                            round(x2, 1),
                            round(y2, 1),
                        ]
                        all_writer.writerow(row_data)
                        grand_total_boxes += 1

    return grand_total_boxes


def main():
    parser = argparse.ArgumentParser(
        description="Run batch object detection inference on camera trap directories."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained weights file (.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output folder for prediction CSVs.",
    )
    parser.add_argument(
        "--img_size", type=int, default=DEFAULT_IMG_SIZE, help="Inference image size."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config defaults).",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cpu, cuda, 0, 1)."
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU threshold for static filter.",
    )
    parser.add_argument(
        "--occurrence_threshold",
        type=int,
        default=DEFAULT_OCCURRENCE_THRESHOLD,
        help="Occurrence threshold.",
    )
    parser.add_argument(
        "--filter_static",
        action="store_true",
        help="Apply static false positive filter.",
    )
    parser.add_argument(
        "--apply_clahe",
        action="store_true",
        help="Apply CLAHE preprocessing on the fly.",
    )
    parser.add_argument(
        "--no_clahe",
        action="store_false",
        dest="apply_clahe",
        help="Disable CLAHE preprocessing.",
    )
    parser.set_defaults(apply_clahe=True)
    parser.add_argument("--force", action="store_true", help="Force re-run inference.")

    args = parser.parse_args()
    device = args.device if args.device else DEFAULT_DEVICE

    print("\n=========================================")
    print(f"LOADING INFERENCE PIPELINE")
    print(f"  Model:         {args.model_path}")
    print(f"  Output Dir:    {args.output_dir}")
    print(f"  CLAHE:         {args.apply_clahe}")
    print(f"  Auto-Filter:   {args.filter_static}")
    print("=========================================")

    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist.")
        return

    model_name = os.path.basename(args.model_path).lower()
    parent_dirs = args.model_path.lower()

    if "faster_rcnn" in model_name or "faster_rcnn" in parent_dirs:
        model_type = "faster_rcnn"
        print(f"Loading Faster R-CNN model from {args.model_path}")
        model = get_faster_rcnn_model(num_classes=len(CLASSES))
        load_compatible_weights(model, args.model_path)
        model.to(device)
        model.eval()
    elif "rtdetr" in model_name or "rtdetr" in parent_dirs:
        model_type = "rtdetr"
        print(f"Loading RT-DETR model from {args.model_path}")
        model = RTDETR(args.model_path)
    else:
        model_type = "yolo"
        print(f"Loading YOLO model from {args.model_path}")
        model = YOLO(args.model_path)

    active_batch_size = (
        args.batch_size
        if args.batch_size
        else (
            FASTER_RCNN_INFERENCE_BATCH_SIZE
            if model_type == "faster_rcnn"
            else ULTRALYTICS_BATCH_SIZE
        )
    )

    unified_csv_path = os.path.join(args.output_dir, "all_unlabeled_predictions.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.force and os.path.exists(unified_csv_path):
        print(
            f"Predictions already exist at {unified_csv_path}. Skipping batch inference. Use --force to override."
        )
    else:
        # Load sampled images
        sampled_csv_path = os.path.join(ACTIVE_LEARNING_DIR, "already_sampled.csv")
        sampled_images = set()
        if os.path.exists(sampled_csv_path):
            try:
                sampled_df = pd.read_csv(sampled_csv_path)
                if "image_path" in sampled_df.columns:
                    sampled_images = set(sampled_df["image_path"].tolist())
            except Exception as e:
                print(f"Warning: Could not read {sampled_csv_path}: {e}")

        # Gather all images across all years
        print("Gathering and filtering unlabeled images from 2023-2025...")
        all_images = []
        for year, base_input_dir in UNLABELED_POOL_DIRS.items():
            if not os.path.exists(base_input_dir):
                print(f"  Year directory {base_input_dir} not found. Skipping.")
                continue

            base_path = Path(base_input_dir)

            for folder_path in base_path.iterdir():
                if folder_path.is_dir():
                    images = [
                        f
                        for f in folder_path.rglob("*")
                        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                    ]

                    # Apply filters
                    filtered_images = [
                        f
                        for f in images
                        if str(f) not in sampled_images
                        and not any(cam in str(f) for cam in EXCLUDED_CAMERAS)
                    ]
                    all_images.extend(filtered_images)

        if not all_images:
            print("No valid unlabeled images found after filtering! Exiting.")
            return

        print(f"Successfully gathered {len(all_images)} total images for inference.")

        with open(unified_csv_path, mode="w", newline="") as f_all:
            all_writer = csv.writer(f_all)
            headers = [
                "image_path",
                "image_name",
                "subfolder",
                "class_id",
                "class_name",
                "confidence",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ]
            all_writer.writerow(headers)

            grand_total_boxes = process_all_images(
                images=all_images,
                model=model,
                model_type=model_type,
                img_size=args.img_size,
                batch_size=active_batch_size,
                device=device,
                all_writer=all_writer,
                apply_clahe_flag=args.apply_clahe,
            )

        print("\n=========================================")
        print("INFERENCE COMPLETE")
        print(f"  Unified CSV: {unified_csv_path}")
        print(f"  Total BBoxes: {grand_total_boxes}")
        print("=========================================")

    if args.filter_static:
        print("\n=========================================")
        print("RUNNING INTEGRATED POST-PROCESSING STATIC FILTER")
        print(f"  IoU Threshold: {args.iou_threshold}")
        print(f"  Occurrence Threshold: {args.occurrence_threshold}")
        print("=========================================")

        df = pd.read_csv(unified_csv_path)
        if df.empty:
            print("No detections found. Skipping static filter.")
            filtered_csv_path = unified_csv_path.replace(".csv", "_filtered.csv")
            df.to_csv(filtered_csv_path, index=False)
            return

        filtered_df, removed_count = filter_static_detections(
            df, args.iou_threshold, args.occurrence_threshold
        )
        filtered_csv_path = unified_csv_path.replace(".csv", "_filtered.csv")
        filtered_df.to_csv(filtered_csv_path, index=False)

        print(f"Static filter removed {removed_count} false positive bounding boxes.")
        print(f"Filtered predictions saved to: {filtered_csv_path}\n")


if __name__ == "__main__":
    main()
