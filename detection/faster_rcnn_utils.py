"""Utilities for Faster R-CNN model training and evaluation.

Contains dataset classes, model loading utilities with shape-checking, early
stopping, and training/validation functions.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import PIL.ImageEnhance as ImageEnhance
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import DEVICE


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

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

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
                        xmin = (xc - w / 2) * self.img_size
                        ymin = (yc - h / 2) * self.img_size
                        xmax = (xc + w / 2) * self.img_size
                        ymax = (yc + h / 2) * self.img_size

                        xmin = max(0.0, xmin)
                        ymin = max(0.0, ymin)
                        xmax = min(float(self.img_size), xmax)
                        ymax = min(float(self.img_size), ymax)

                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            # PyTorch Faster R-CNN uses label 0 for background
                            labels.append(cls_id + 1)

        # Apply basic online augmentations during training
        if self.augment:
            # 1. CutMix Augmentation (mix two images)
            if np.random.random() > 0.6:  # 40% probability
                other_idx = np.random.randint(0, len(self.img_files))
                other_img_path = self.img_files[other_idx]
                try:
                    other_img = Image.open(other_img_path).convert("RGB")
                    other_img = other_img.resize(
                        (self.img_size, self.img_size), Image.BILINEAR
                    )

                    other_base_name = os.path.splitext(
                        os.path.basename(other_img_path)
                    )[0]
                    other_lbl_path = os.path.join(
                        self.lbl_dir, other_base_name + ".txt"
                    )
                    other_boxes = []
                    other_labels = []
                    if os.path.exists(other_lbl_path):
                        with open(other_lbl_path, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    cls_id = int(parts[0])
                                    xc, yc, w_box, h_box = map(float, parts[1:])
                                    xmin = (xc - w_box / 2) * self.img_size
                                    ymin = (yc - h_box / 2) * self.img_size
                                    xmax = (xc + w_box / 2) * self.img_size
                                    ymax = (yc + h_box / 2) * self.img_size
                                    xmin = max(0.0, xmin)
                                    ymin = max(0.0, ymin)
                                    xmax = min(float(self.img_size), xmax)
                                    ymax = min(float(self.img_size), ymax)
                                    if xmax > xmin and ymax > ymin:
                                        other_boxes.append([xmin, ymin, xmax, ymax])
                                        other_labels.append(cls_id + 1)

                    # Cut out a random patch (size between 30% and 60% of image size)
                    cut_w = np.random.randint(
                        int(self.img_size * 0.3), int(self.img_size * 0.6)
                    )
                    cut_h = np.random.randint(
                        int(self.img_size * 0.3), int(self.img_size * 0.6)
                    )
                    cut_x = np.random.randint(0, self.img_size - cut_w)
                    cut_y = np.random.randint(0, self.img_size - cut_h)

                    # Cut and paste patch
                    patch = other_img.crop((cut_x, cut_y, cut_x + cut_w, cut_y + cut_h))
                    img.paste(patch, (cut_x, cut_y))

                    merged_boxes = []
                    merged_labels = []

                    # Original image boxes: keep if they are not heavily covered by the cut region
                    for b, lbl in zip(boxes, labels):
                        # Calculate overlap area
                        x1_overlap = max(b[0], float(cut_x))
                        y1_overlap = max(b[1], float(cut_y))
                        x2_overlap = min(b[2], float(cut_x + cut_w))
                        y2_overlap = min(b[3], float(cut_y + cut_h))

                        overlap_w = max(0.0, x2_overlap - x1_overlap)
                        overlap_h = max(0.0, y2_overlap - y1_overlap)
                        overlap_area = overlap_w * overlap_h
                        box_area = (b[2] - b[0]) * (b[3] - b[1])

                        # Keep original boxes that are less than 60% covered
                        if box_area > 0 and (overlap_area / box_area) < 0.6:
                            merged_boxes.append(b)
                            merged_labels.append(lbl)

                    # Other image boxes: keep if their center is inside the cut region
                    for ob, ol in zip(other_boxes, other_labels):
                        center_x = (ob[0] + ob[2]) / 2.0
                        center_y = (ob[1] + ob[3]) / 2.0
                        if (
                            cut_x <= center_x <= cut_x + cut_w
                            and cut_y <= center_y <= cut_y + cut_h
                        ):
                            x1 = max(ob[0], float(cut_x))
                            y1 = max(ob[1], float(cut_y))
                            x2 = min(ob[2], float(cut_x + cut_w))
                            y2 = min(ob[3], float(cut_y + cut_h))
                            if x2 > x1 + 1 and y2 > y1 + 1:
                                merged_boxes.append([x1, y1, x2, y2])
                                merged_labels.append(ol)

                    boxes = merged_boxes
                    labels = merged_labels
                except Exception as e:
                    print(f"Warning: CutMix failed: {e}")

            # 2. Random Horizontal Flip
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if len(boxes) > 0:
                    new_boxes = []
                    for b in boxes:
                        new_boxes.append(
                            [self.img_size - b[2], b[1], self.img_size - b[0], b[3]]
                        )
                    boxes = new_boxes

            # 3. Random Vertical Flip
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                if len(boxes) > 0:
                    new_boxes = []
                    for b in boxes:
                        new_boxes.append(
                            [b[0], self.img_size - b[3], b[2], self.img_size - b[1]]
                        )
                    boxes = new_boxes

            # 4. Color Jitter (Brightness, Contrast, Saturation)
            if np.random.random() > 0.5:
                img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.6, 1.4))
            if np.random.random() > 0.5:
                img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.6, 1.4))
            if np.random.random() > 0.5:
                img = ImageEnhance.Color(img).enhance(np.random.uniform(0.6, 1.4))

            # 5. Random Zoom / Scale & Crop
            if np.random.random() > 0.5:
                scale = np.random.uniform(1.0, 1.3)
                w, h = img.size
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = img.resize((new_w, new_h), Image.BILINEAR)

                left = np.random.randint(0, new_w - w) if new_w > w else 0
                top = np.random.randint(0, new_h - h) if new_h > h else 0
                img = img_resized.crop((left, top, left + w, top + h))

                if len(boxes) > 0:
                    new_boxes = []
                    for b in boxes:
                        x1 = b[0] * scale - left
                        y1 = b[1] * scale - top
                        x2 = b[2] * scale - left
                        y2 = b[3] * scale - top
                        x1 = max(0.0, min(x1, float(w)))
                        y1 = max(0.0, min(y1, float(h)))
                        x2 = max(0.0, min(x2, float(w)))
                        y2 = max(0.0, min(y2, float(h)))
                        if x2 > x1 + 1 and y2 > y1 + 1:
                            new_boxes.append([x1, y1, x2, y2])
                    if new_boxes:
                        boxes = new_boxes

            # 6. Random Rotation (+/- 10 degrees)
            if np.random.random() > 0.5:
                angle = np.random.uniform(-10, 10)
                img = img.rotate(angle, resample=Image.BILINEAR)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            final_boxes = []
            final_labels = []
            for i, b in enumerate(boxes):
                x1 = max(0.0, min(b[0], float(self.img_size)))
                y1 = max(0.0, min(b[1], float(self.img_size)))
                x2 = max(0.0, min(b[2], float(self.img_size)))
                y2 = max(0.0, min(b[3], float(self.img_size)))
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
