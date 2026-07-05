import os
import sys
import glob
import json
import torch
from ultralytics import YOLO, RTDETR
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

sys.path.append("/home/Joshua/Downloads/active_learning_research/detection")
from config import *
from faster_rcnn_utils import get_faster_rcnn_model, load_compatible_weights


import cv2
import numpy as np


class ImagePoolDataset(Dataset):
    def __init__(self, image_paths, img_size=IMG_SIZE):
        self.image_paths = image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Robust loading with OpenCV
        img = cv2.imread(img_path)
        if img is None:
            # Fallback to black image if completely corrupted
            img_tensor = torch.zeros(
                (3, self.img_size, self.img_size), dtype=torch.float32
            )
            return img_tensor, img_path, self.img_size, self.img_size

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(
            img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        img_np = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        return img_tensor, img_path, orig_w, orig_h


def get_ground_truth_for_image(img_path, labels_dir):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_file = os.path.join(labels_dir, base_name + ".txt")
    gt_boxes = []
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    # We store just the cls, bounding boxes not strictly needed for image level
                    # But storing just in case. They are YOLO format (cls_id, x_center, y_center, w, h)
                    gt_boxes.append(
                        {"cls": cls_id, "bbox": [float(x) for x in parts[1:5]]}
                    )
    return gt_boxes


def generate_ultralytics_predictions(
    model_path, model_class, name, dir_name, all_paths, test_labels_dir
):
    model_dir = os.path.join(RESULTS_DIR, dir_name)
    os.makedirs(model_dir, exist_ok=True)
    out_file = os.path.join(model_dir, "predictions.json")

    if os.path.exists(out_file):
        print(f"\n{name} predictions already exist at {out_file}, skipping inference.")
        return

    print(f"\nRunning {name} Inference on {len(all_paths)} images...")
    model = model_class(model_path)

    dataset = ImagePoolDataset(all_paths, img_size=IMG_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=ULTRALYTICS_BATCH_SIZE,
        shuffle=False,
        num_workers=EVAL_NUM_WORKERS,
    )

    results_list = []

    for images, img_paths, orig_ws, orig_hs in tqdm(
        dataloader, desc=f"{name} Inference"
    ):
        images = images.to(DEVICE)

        preds = model.predict(
            source=images,
            imgsz=IMG_SIZE,
            device=DEVICE,
            conf=CONF_THRESHOLD,
            verbose=False,
            stream=False,
        )

        for i, pred in enumerate(preds):
            pred_data = {"path": img_paths[i], "predictions": [], "gt_boxes": []}

            # Ground truth insertion
            if img_paths[i].startswith(TEST_FULL_DIR):
                base_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
                lbl_path = os.path.join(test_labels_dir, base_name + ".txt")
                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                pred_data["gt_boxes"].append(
                                    {
                                        "cls": int(parts[0]),
                                        "bbox": [float(x) for x in parts[1:5]],
                                    }
                                )

            # Boxes
            if pred.boxes is not None and len(pred.boxes) > 0:
                boxes = pred.boxes.xyxyn.cpu().numpy().tolist()
                confs = pred.boxes.conf.cpu().numpy().tolist()
                clss = pred.boxes.cls.cpu().numpy().tolist()

                for b, c, cls_id in zip(boxes, confs, clss):
                    pred_data["predictions"].append(
                        {"cls": int(cls_id), "conf": float(c), "bbox": b}
                    )

            results_list.append(pred_data)

    with open(out_file, "w") as f:
        json.dump(results_list, f, indent=4)
    print(f"Saved {name} predictions to {out_file}")


def generate_faster_rcnn_predictions(
    all_paths, test_labels_dir, dir_name="faster_rcnn"
):
    model_dir = os.path.join(RESULTS_DIR, dir_name)
    os.makedirs(model_dir, exist_ok=True)
    out_file = os.path.join(model_dir, "predictions.json")

    if os.path.exists(out_file):
        print(
            f"\nFaster R-CNN predictions already exist at {out_file}, skipping inference."
        )
        return

    print(f"\nGenerating predictions for Faster R-CNN...")
    model = get_faster_rcnn_model(num_classes=NUM_CLASSES)
    load_compatible_weights(model, FASTER_RCNN_WEIGHTS)
    model.to(DEVICE)
    model.eval()

    dataset = ImagePoolDataset(all_paths, img_size=IMG_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=FASTER_RCNN_BATCH_SIZE,
        shuffle=False,
        num_workers=FASTER_RCNN_NUM_WORKERS,
    )

    results_list = []

    with torch.no_grad():
        for images, img_paths, orig_ws, orig_hs in tqdm(
            dataloader, desc="Faster R-CNN Inference"
        ):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for out, img_path, ow, oh in zip(outputs, img_paths, orig_ws, orig_hs):
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy() - 1  # Shift back background
                boxes = out["boxes"].cpu().numpy()

                img_preds = []
                for s, l, b in zip(scores, labels, boxes):
                    if s >= CONF_THRESHOLD:
                        # Normalize box [xmin, ymin, xmax, ymax]
                        img_preds.append(
                            {
                                "cls": int(l),
                                "conf": float(s),
                                "bbox": [
                                    float(
                                        b[0] / float(IMG_SIZE)
                                    ),  # Assuming model outputs IMG_SIZE scale
                                    float(b[1] / float(IMG_SIZE)),
                                    float(b[2] / float(IMG_SIZE)),
                                    float(b[3] / float(IMG_SIZE)),
                                ],
                            }
                        )

                if img_path.startswith(TEST_FULL_DIR):
                    gt_boxes = get_ground_truth_for_image(img_path, test_labels_dir)
                else:
                    gt_boxes = []

                results_list.append(
                    {"path": img_path, "predictions": img_preds, "gt_boxes": gt_boxes}
                )

    with open(out_file, "w") as f:
        json.dump(results_list, f)
    print(f"Saved Faster R-CNN predictions to {out_file}")


def main():
    print("Collecting images...")
    test_images_dir = os.path.join(TEST_FULL_DIR, "images")
    test_labels_dir = os.path.join(TEST_FULL_DIR, "labels")

    test_paths = sorted(glob.glob(os.path.join(test_images_dir, "*.*")))
    test_paths = [
        p for p in test_paths if p.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    unlabelled_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        unlabelled_paths.extend(
            glob.glob(os.path.join(UNLABELLED_POOL_DIR, "**", ext), recursive=True)
        )
    unlabelled_paths = sorted(list(set(unlabelled_paths)))

    all_paths = test_paths + unlabelled_paths
    print(f"Total images to infer: {len(all_paths)}")

    generate_ultralytics_predictions(
        YOLO_WEIGHTS, YOLO, "YOLO", "yolo", all_paths, test_labels_dir
    )
    generate_ultralytics_predictions(
        RTDETR_WEIGHTS, RTDETR, "RT-DETR", "rtdetr", all_paths, test_labels_dir
    )
    generate_faster_rcnn_predictions(all_paths, test_labels_dir, "faster_rcnn")


if __name__ == "__main__":
    main()
