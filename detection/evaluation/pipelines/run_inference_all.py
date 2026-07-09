import os
import sys
import json
import torch
import cv2
import math
import numpy as np
import argparse
import cv2
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_utils.config import (
    TEST_DIR,
    VAL_DIR,
    AL_DIR,
    MODELS,
    CYCLES,
    MIN_CONF_THRESHOLD,
    RESULTS_DIR,
    DEVICE,
    IMAGE_SIZE,
    YOLO_BATCH_SIZE,
    FASTER_RCNN_BATCH_SIZE,
    MD_BATCH_SIZE,
    MD_WEIGHTS_PATH,
    TEST_FULL_CAMERA,
    VAL_FULL_CAMERA,
    UNLABELED_POOL_DIRS,
)

sys.path.append(AL_DIR)
sys.path.append(os.path.join(AL_DIR, "pipelines"))
from pipelines.faster_rcnn_utils import get_faster_rcnn_model, load_compatible_weights
from central_config import CLASSES
from ultralytics import YOLO, RTDETR
from torch.utils.data import DataLoader, Dataset
from pipelines.run_inference_pipeline import (
    ActiveLearningInferenceDataset,
    DEFAULT_NUM_WORKERS,
)


def apply_clahe(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    return cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)


def resize_image(img_bgr, model_type="yolo"):
    orig_h, orig_w = img_bgr.shape[:2]
    if model_type in ["yolo", "faster_rcnn"]:
        if orig_h < orig_w:
            new_h = IMAGE_SIZE
            new_w = math.ceil((orig_w * (IMAGE_SIZE / orig_h)) / 32.0) * 32
        else:
            new_w = IMAGE_SIZE
            new_h = math.ceil((orig_h * (IMAGE_SIZE / orig_w)) / 32.0) * 32
    else:
        if orig_h > orig_w:
            new_h = IMAGE_SIZE
            new_w = math.ceil((orig_w * (IMAGE_SIZE / orig_h)) / 32.0) * 32
        else:
            new_w = IMAGE_SIZE
            new_h = math.ceil((orig_h * (IMAGE_SIZE / orig_w)) / 32.0) * 32

    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img_resized, orig_w, orig_h, new_w, new_h


def get_image_paths(dataset_dir):
    img_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(img_dir):
        return []
    paths = []
    for f in os.listdir(img_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(img_dir, f))
    return paths


def get_full_sequence_paths(camera_string):
    paths = []
    for year, base_input_dir in UNLABELED_POOL_DIRS.items():
        if not os.path.exists(base_input_dir):
            continue
        base_path = Path(base_input_dir)
        for folder_path in base_path.iterdir():
            if folder_path.is_dir():
                for f in folder_path.rglob("*"):
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        if camera_string in f.parts:
                            paths.append(str(f))
    return paths


def get_predictions_ultralytics(
    model,
    img_paths,
    batch_size=YOLO_BATCH_SIZE,
    apply_clahe_flag=True,
    model_type="yolo",
):
    results = []
    dataset = ActiveLearningInferenceDataset(
        img_paths,
        img_size=IMAGE_SIZE,
        apply_clahe_flag=apply_clahe_flag,
        model_type=model_type,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=True,
    )

    for batch_imgs, paths, orig_ws, orig_hs in tqdm(
        dataloader, desc=f"Inference {model_type}"
    ):
        batch_imgs = batch_imgs.to(DEVICE, non_blocking=True).float() / 255.0

        preds = model.predict(
            source=batch_imgs, device=DEVICE, conf=MIN_CONF_THRESHOLD, verbose=False
        )
        for j, pred in enumerate(preds):
            pred_list = []
            if pred.boxes is not None and len(pred.boxes) > 0:
                boxes_xywhn = pred.boxes.xywhn.cpu().numpy()
                confs = pred.boxes.conf.cpu().numpy()
                clss = pred.boxes.cls.cpu().numpy()
                for b, conf, cls_id in zip(boxes_xywhn, confs, clss):
                    pred_list.append(
                        {
                            "bbox": [
                                float(b[0]),
                                float(b[1]),
                                float(b[2]),
                                float(b[3]),
                            ],
                            "conf": float(conf),
                            "cls": int(cls_id),
                        }
                    )
            results.append({"path": paths[j], "predictions": pred_list})
    return results


def get_predictions_faster_rcnn(
    model, img_paths, batch_size=FASTER_RCNN_BATCH_SIZE, apply_clahe_flag=True
):
    results = []
    dataset = ActiveLearningInferenceDataset(
        img_paths,
        img_size=IMAGE_SIZE,
        apply_clahe_flag=apply_clahe_flag,
        model_type="faster_rcnn",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=True,
    )

    for batch_imgs, paths, orig_ws, orig_hs in tqdm(
        dataloader, desc="Inference faster_rcnn"
    ):
        batch_imgs = batch_imgs.to(DEVICE, non_blocking=True).float() / 255.0

        with torch.no_grad():
            outputs = model(batch_imgs)

        for j, out in enumerate(outputs):
            pred_list = []
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy() - 1
            boxes = out["boxes"].cpu().numpy()

            new_h, new_w = batch_imgs.shape[2], batch_imgs.shape[3]

            for s, l, b in zip(scores, labels, boxes):
                if s >= MIN_CONF_THRESHOLD:
                    x1, y1, x2, y2 = (
                        b[0] / new_w,
                        b[1] / new_h,
                        b[2] / new_w,
                        b[3] / new_h,
                    )
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = x2 - x1
                    h = y2 - y1
                    pred_list.append(
                        {
                            "bbox": [float(cx), float(cy), float(w), float(h)],
                            "conf": float(s),
                            "cls": int(l),
                        }
                    )
            results.append({"path": paths[j], "predictions": pred_list})
    return results


def get_predictions_megadetector(model, img_paths, batch_size=MD_BATCH_SIZE):
    results = []

    dataset = ActiveLearningInferenceDataset(
        img_paths,
        img_size=IMAGE_SIZE,
        apply_clahe_flag=False,
        model_type="megadetector",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        prefetch_factor=2,
        pin_memory=True,
    )

    for batch_imgs, batch_paths, orig_ws, orig_hs in tqdm(
        dataloader, desc="Megadetector Inference"
    ):
        # Convert PyTorch tensor (B, 3, H, W) back to list of numpy arrays (H, W, 3) for AutoShape
        batch_imgs_np = [img.permute(1, 2, 0).cpu().numpy() for img in batch_imgs]

        preds = model(batch_imgs_np)

        for j, pred_tensor in enumerate(preds.xywhn):
            pred_list = []
            if pred_tensor is not None and len(pred_tensor) > 0:
                boxes_xywhn = pred_tensor.cpu().numpy()
                for b in boxes_xywhn:
                    conf = b[4]
                    cls_id = b[5]
                    if conf >= MIN_CONF_THRESHOLD:
                        if int(cls_id) == 0:
                            pred_list.append(
                                {
                                    "bbox": [
                                        float(b[0]),
                                        float(b[1]),
                                        float(b[2]),
                                        float(b[3]),
                                    ],
                                    "conf": float(conf),
                                    "cls": -1,
                                }
                            )
            results.append({"path": batch_paths[j], "predictions": pred_list})
    return results


def main():
    datasets_to_run = [("test_full", TEST_FULL_CAMERA), ("val_full", VAL_FULL_CAMERA)]

    # 1. Run MegaDetector
    if os.path.exists(MD_WEIGHTS_PATH):
        print(f"Loading MegaDetector from {MD_WEIGHTS_PATH} via torch.hub")
        md_model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=MD_WEIGHTS_PATH, force_reload=False
        )
        md_model.to(DEVICE)
        md_dir = os.path.join(RESULTS_DIR, "megadetector")
        os.makedirs(md_dir, exist_ok=True)
        for ds_name, ds_path in datasets_to_run:
            out_path = os.path.join(md_dir, f"{ds_name}_raw.json")
            if not os.path.exists(out_path):
                img_paths = get_full_sequence_paths(ds_path)
                if img_paths:
                    print(f"Running MD on {ds_name} (Images: {len(img_paths)})...")
                    res = get_predictions_megadetector(md_model, img_paths)
                    with open(out_path, "w") as f:
                        json.dump(res, f, indent=4)
                    print(f"Saved {out_path}")

    # Cleanup MD model from memory
    del md_model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Run AL Models
    for model_name in MODELS:
        is_clahe = "clahe" in model_name
        is_faster = "faster_rcnn" in model_name
        is_rtdetr = "rtdetr" in model_name

        model_res_dir = os.path.join(RESULTS_DIR, model_name)
        os.makedirs(model_res_dir, exist_ok=True)

        for cycle in CYCLES:
            weights_path = os.path.join(
                AL_DIR,
                model_name,
                "runs",
                f"{cycle}_pretrained_phase2",
                "weights",
                "best.pt",
            )
            if not os.path.exists(weights_path):
                print(f"Missing {weights_path}, skipping...")
                continue

            print(f"\nLoading {model_name} {cycle}...")
            model = None
            if is_faster:
                model = get_faster_rcnn_model(num_classes=len(CLASSES))
                load_compatible_weights(model, weights_path)
                model.to(DEVICE)
                model.eval()
            elif is_rtdetr:
                model = RTDETR(weights_path)
            else:
                model = YOLO(weights_path)

            for ds_name, ds_path in datasets_to_run:
                out_path = os.path.join(model_res_dir, f"{ds_name}_{cycle}_raw.json")
                if os.path.exists(out_path):
                    print(f"{out_path} already exists, skipping.")
                    continue

                img_paths = get_full_sequence_paths(ds_path)

                if not img_paths:
                    continue

                print(
                    f"Running {model_name} {cycle} on {ds_name} ({len(img_paths)} images)..."
                )
                if is_faster:
                    res = get_predictions_faster_rcnn(
                        model, img_paths, apply_clahe_flag=is_clahe
                    )
                else:
                    m_type = "rtdetr" if is_rtdetr else "yolo"
                    res = get_predictions_ultralytics(
                        model, img_paths, apply_clahe_flag=is_clahe, model_type=m_type
                    )

                with open(out_path, "w") as f:
                    json.dump(res, f, indent=4)
                print(f"Saved {out_path}")

            # Cleanup AL model from memory
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
