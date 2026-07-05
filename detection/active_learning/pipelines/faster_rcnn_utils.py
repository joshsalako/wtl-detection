"""Utilities for Faster R-CNN model training and evaluation.

Contains dataset classes, model loading utilities with shape-checking, early
stopping, and training/validation functions.
"""

import os
import glob
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import PIL.ImageEnhance as ImageEnhance
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import sys
import os
import cv2
import albumentations as A
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchvision.ops import box_iou

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEARNING_DIR = os.path.dirname(PIPELINES_DIR)
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)

from central_config import CLASSES

from central_config import DEVICE


class ActiveLearningFasterRCNNDataset(Dataset):
    """Custom PyTorch dataset for loading images and YOLO format labels."""

    def __init__(self, dataset_dir, split="train", img_size=640, augment=False):
        """Initialize dataset paths and files."""
        self.dataset_dir = dataset_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.img_dir = os.path.join(dataset_dir, split, "images")
        self.lbl_dir = os.path.join(dataset_dir, split, "labels")

        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        self.img_files = [
            f
            for f in self.img_files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]

        if self.augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0, p=0.8
                    ),
                    A.Affine(
                        scale=(0.5, 1.5),
                        translate_percent=(-0.1, 0.1),
                        rotate=(-10, 10),
                        cval=(114, 114, 114),
                        p=0.5,
                    ),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    min_visibility=0.2,
                    label_fields=["class_labels"],
                ),
            )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]

        try:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            if orig_h < orig_w:
                new_h = 640
                new_w = math.ceil((orig_w * (640 / orig_h)) / 32.0) * 32
            else:
                new_w = 640
                new_h = math.ceil((orig_h * (640 / orig_w)) / 32.0) * 32

            # Apply CLAHE preprocessing
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(gray)
            img_clahe = cv2.cvtColor(cl, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(img_clahe)
        except Exception as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            new_w, new_h = 1152, 640
            img = Image.new("RGB", (new_w, new_h), (0, 0, 0))

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, base_name + ".txt")

        boxes = []
        labels = []

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                        xmin = (xc - w / 2) * new_w
                        ymin = (yc - h / 2) * new_h
                        xmax = (xc + w / 2) * new_w
                        ymax = (yc + h / 2) * new_h

                        xmin = max(0.0, xmin)
                        ymin = max(0.0, ymin)
                        xmax = min(float(new_w), xmax)
                        ymax = min(float(new_h), ymax)

                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            # PyTorch Faster R-CNN uses label 0 for background
                            labels.append(cls_id + 1)

        # Resize image FIRST so bounding box scales match
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img_np_cv = np.array(img)

        # Apply robust online augmentations during training via Albumentations
        if self.augment:
            if len(boxes) > 0:
                transformed = self.transform(
                    image=img_np_cv, bboxes=boxes, class_labels=labels
                )
                img_np_cv = transformed["image"]
                boxes = transformed["bboxes"]
                labels = transformed["class_labels"]
            else:
                transformed = self.transform(
                    image=img_np_cv, bboxes=[], class_labels=[]
                )
                img_np_cv = transformed["image"]

        img_np = img_np_cv.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            final_boxes = []
            final_labels = []
            for i, b in enumerate(boxes):
                x1 = max(0.0, min(b[0], float(new_w)))
                y1 = max(0.0, min(b[1], float(new_h)))
                x2 = max(0.0, min(b[2], float(new_w)))
                y2 = max(0.0, min(b[3], float(new_h)))
                if (x2 > x1 + 1) and (y2 > y1 + 1):
                    final_boxes.append([x1, y1, x2, y2])
                    final_labels.append(labels[i])

            if not final_boxes:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.as_tensor(final_boxes, dtype=torch.float32)
                labels = torch.as_tensor(final_labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return img_tensor, target


def collate_fn(batch):
    """Collate function to batch variable-length targets."""
    return tuple(zip(*batch))


class EarlyStopping:
    """Early stopping utility to monitor validation loss."""

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_faster_rcnn_model(num_classes=3, freeze_backbone=False):
    """Constructs Faster R-CNN model with torchvision ResNet50 backbone."""
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def load_compatible_weights(model, weights_path):
    """Loads weights from weights_path, filtering out shape-mismatched classification/regression heads."""
    if not os.path.exists(weights_path):
        print(f"Weights path {weights_path} not found. Starting from default weights.")
        return

    print(f"Loading pretrained weights state dict from {weights_path}")
    pretrained_dict = torch.load(weights_path, map_location=DEVICE)
    model_dict = model.state_dict()

    # Filter out layers that do not match in shape
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                print(
                    f"Skipping key '{k}' due to shape mismatch: "
                    f"pretrained {list(v.shape)} vs model {list(model_dict[k].shape)}"
                )
        else:
            print(f"Skipping key '{k}' as it is not in the model state dict.")

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print(f"Successfully loaded {len(filtered_dict)} layers from {weights_path}.")


def train_one_epoch(model, dataloader, optimizer):
    """Runs one training epoch on Faster R-CNN."""
    model.train()
    total_loss = 0
    loss_comps = {
        "loss_classifier": 0,
        "loss_box_reg": 0,
        "loss_objectness": 0,
        "loss_rpn_box_reg": 0,
    }

    for images, targets in dataloader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        for k in loss_comps.keys():
            if k in loss_dict:
                loss_comps[k] += loss_dict[k].item()

    avg_loss = total_loss / max(1, len(dataloader))
    avg_comps = {k: v / max(1, len(dataloader)) for k, v in loss_comps.items()}
    return avg_loss, avg_comps


def validate_one_epoch(model, dataloader):
    """Computes losses on the validation set using training mode forward pass."""
    model.train()  # Model must be in training mode to calculate losses
    total_loss = 0
    loss_comps = {
        "loss_classifier": 0,
        "loss_box_reg": 0,
        "loss_objectness": 0,
        "loss_rpn_box_reg": 0,
    }

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            total_loss += losses.item()
            for k in loss_comps.keys():
                if k in loss_dict:
                    loss_comps[k] += loss_dict[k].item()

    avg_loss = total_loss / max(1, len(dataloader))
    avg_comps = {k: v / max(1, len(dataloader)) for k, v in loss_comps.items()}
    return avg_loss, avg_comps


def plot_batch_grid(images, targets, save_path, class_names, max_images=16):
    num_images = min(len(images), max_images)
    grid_size = math.ceil(math.sqrt(num_images))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    if grid_size == 1:
        axes = np.array([[axes]])
    elif len(axes.shape) == 1:
        axes = axes.reshape(grid_size, grid_size)

    for i in range(grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        ax.axis("off")
        if i < num_images:
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)

            boxes = targets[i]["boxes"].cpu().numpy()
            labels = targets[i]["labels"].cpu().numpy()
            for b, l in zip(boxes, labels):
                xmin, ymin, xmax, ymax = b
                rect = patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)
                cls_name = class_names[l - 1] if (l - 1) < len(class_names) else str(l)
                ax.text(
                    xmin,
                    ymin,
                    cls_name,
                    color="white",
                    fontsize=8,
                    bbox=dict(facecolor="red", alpha=0.5, pad=1),
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_and_plot_metrics(model, dataloader, run_dir, class_names):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(DEVICE) for img in images]
            preds = model(images)
            for p in preds:
                all_preds.append({k: v.cpu() for k, v in p.items()})
            for t in targets:
                all_targets.append({k: v.cpu() for k, v in t.items()})

    num_classes = len(class_names)
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    for pred, target in zip(all_preds, all_targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"] - 1

        target_boxes = target["boxes"]
        target_labels = target["labels"] - 1

        keep = pred_scores > 0.25
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        if len(target_boxes) == 0:
            for l in pred_labels:
                conf_matrix[num_classes, int(l)] += 1
            continue

        if len(pred_boxes) == 0:
            for l in target_labels:
                conf_matrix[int(l), num_classes] += 1
            continue

        ious = box_iou(target_boxes, pred_boxes)
        matched_preds = set()
        for i, t_box in enumerate(target_boxes):
            t_label = int(target_labels[i])
            best_iou = 0
            best_j = -1
            for j, p_box in enumerate(pred_boxes):
                if j in matched_preds:
                    continue
                if pred_labels[j] == t_label and ious[i, j] > best_iou:
                    best_iou = ious[i, j]
                    best_j = j

            if best_iou > 0.5:
                matched_preds.add(best_j)
                conf_matrix[t_label, t_label] += 1
            else:
                conf_matrix[t_label, num_classes] += 1

        for j, p_box in enumerate(pred_boxes):
            if j not in matched_preds:
                p_label = int(pred_labels[j])
                conf_matrix[num_classes, p_label] += 1

    plt.figure(figsize=(10, 8))
    labels_ext = class_names + ["Background"]
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        xticklabels=labels_ext,
        yticklabels=labels_ext,
        cmap="Blues",
    )
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.title("Confusion Matrix (IoU > 0.5, Conf > 0.25)")
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    for c in range(num_classes):
        scores = []
        true_labels = []
        for pred, target in zip(all_preds, all_targets):
            p_boxes = pred["boxes"][pred["labels"] == c + 1]
            p_scores = pred["scores"][pred["labels"] == c + 1]
            t_boxes = target["boxes"][target["labels"] == c + 1]

            if len(t_boxes) == 0 and len(p_boxes) == 0:
                continue

            if len(p_boxes) == 0:
                for _ in range(len(t_boxes)):
                    scores.append(0.0)
                    true_labels.append(1)
                continue

            if len(t_boxes) == 0:
                for s in p_scores:
                    scores.append(s.item())
                    true_labels.append(0)
                continue

            ious = box_iou(t_boxes, p_boxes)
            matched_preds = set()
            for i in range(len(t_boxes)):
                best_iou = 0
                best_j = -1
                for j in range(len(p_boxes)):
                    if j in matched_preds:
                        continue
                    if ious[i, j] > best_iou:
                        best_iou = ious[i, j]
                        best_j = j
                if best_iou > 0.5:
                    matched_preds.add(best_j)
                    scores.append(p_scores[best_j].item())
                    true_labels.append(1)
                else:
                    scores.append(0.0)
                    true_labels.append(1)

            for j in range(len(p_boxes)):
                if j not in matched_preds:
                    scores.append(p_scores[j].item())
                    true_labels.append(0)

        if len(true_labels) > 0:
            p, r, _ = precision_recall_curve(true_labels, scores)
            ap = average_precision_score(true_labels, scores)
            plt.plot(r, p, label=f"{class_names[c]} (AP: {ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (IoU > 0.5)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "BoxPR_curve.png"), dpi=150)
    plt.close()


def train_faster_rcnn(
    weights,
    run_name,
    project_dir,
    dataset_dir,
    freeze_backbone,
    epochs,
    patience,
    batch_size,
    apply_clahe,
    num_classes,
):
    """Finetunes Faster R-CNN using torch native classes."""
    from torch.utils.data import DataLoader

    print(
        f"\n--- Training Faster R-CNN: {run_name} (Freeze Backbone: {freeze_backbone}) ---"
    )

    train_dataset = ActiveLearningFasterRCNNDataset(
        dataset_dir, split="train", img_size=640, augment=True
    )
    val_dataset = ActiveLearningFasterRCNNDataset(
        dataset_dir, split="val", img_size=640, augment=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    run_dir = os.path.join(project_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)

    results_csv = os.path.join(run_dir, "results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train/loss",
                "train/loss_classifier",
                "train/loss_box_reg",
                "train/loss_objectness",
                "train/loss_rpn_box_reg",
                "val/loss",
                "val/loss_classifier",
                "val/loss_box_reg",
                "val/loss_objectness",
                "val/loss_rpn_box_reg",
            ]
        )

    # Plot batch 0
    try:
        t_images, t_targets = next(iter(train_dataloader))
        plot_batch_grid(
            t_images, t_targets, os.path.join(run_dir, "train_batch0.jpg"), CLASSES
        )
        v_images, v_targets = next(iter(val_dataloader))
        plot_batch_grid(
            v_images, v_targets, os.path.join(run_dir, "val_batch0_labels.jpg"), CLASSES
        )
    except Exception as e:
        print(f"Warning: Failed to plot batch grids: {e}")

    model = get_faster_rcnn_model(
        num_classes=num_classes, freeze_backbone=freeze_backbone
    )
    load_compatible_weights(model, weights)
    model.to(DEVICE)

    lr = 0.0001 if freeze_backbone else 0.00005
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )

    run_dir = os.path.join(project_dir, run_name)
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    best_model_path = os.path.join(run_dir, "weights", "best.pt")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_comps = train_one_epoch(model, train_dataloader, optimizer)
        val_loss, val_comps = validate_one_epoch(model, val_dataloader)

        print(
            f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    train_comps["loss_classifier"],
                    train_comps["loss_box_reg"],
                    train_comps["loss_objectness"],
                    train_comps["loss_rpn_box_reg"],
                    val_loss,
                    val_comps["loss_classifier"],
                    val_comps["loss_box_reg"],
                    val_comps["loss_objectness"],
                    val_comps["loss_rpn_box_reg"],
                ]
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print("Evaluating best model to generate plots...")
    model.load_state_dict(torch.load(best_model_path))
    evaluate_and_plot_metrics(model, val_dataloader, run_dir, CLASSES)

    return best_model_path
