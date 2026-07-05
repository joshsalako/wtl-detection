"""Phased Freezing Training Script for YOLO, RT-DETR, and Faster R-CNN.

Executes a two-phase training loop:
1. Phase 1: Adapt the detection head only (backbone frozen).
2. Phase 2: Optimize the entire network globally (backbone unfrozen).
Uses the val split for early stopping and saves checkpoints in the results directory.
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO, RTDETR

# Import config settings
from config import (
    DEVICE,
    CLAHE_DATASET_DIR,
    RESULTS_DIR,
    PRETRAINED_YOLO,
    PRETRAINED_RTDETR,
    PRETRAINED_FASTER_RCNN,
    CLASSES,
    NUM_CLASSES,
    IMG_SIZE,
    YOLO_TRAIN_CONFIG,
    RTDETR_TRAIN_CONFIG,
    FASTER_RCNN_TRAIN_CONFIG,
    AUGMENTATION_CONFIG,
)

# Import Faster R-CNN utilities
from faster_rcnn_utils import (
    ActiveLearningFasterRCNNDataset,
    collate_fn,
    get_faster_rcnn_model,
    load_compatible_weights,
    train_one_epoch,
    validate_one_epoch,
    EarlyStopping,
)


def create_dataset_yaml():
    """Generates the dataset.yaml file required for Ultralytics training."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    yaml_path = os.path.join(RESULTS_DIR, "dataset_clahe.yaml")

    yaml_data = {
        "path": os.path.abspath(CLAHE_DATASET_DIR),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(CLASSES)},
    }

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"[Config] Created dataset YAML at: {yaml_path}")
    return yaml_path


def train_yolo_rtdetr(model_type, dataset_yaml):
    """Finetunes YOLO or RT-DETR using phased freezing with Ultralytics APIs."""
    print("\n=======================================================")
    print(f"STARTING {model_type.upper()} FINETUNING")
    print("=======================================================")

    model_class = YOLO if model_type == "yolo" else RTDETR
    pretrained_weights = PRETRAINED_YOLO if model_type == "yolo" else PRETRAINED_RTDETR
    train_cfg = YOLO_TRAIN_CONFIG if model_type == "yolo" else RTDETR_TRAIN_CONFIG

    # --- Phase 1: Train Head Only (Backbone Frozen) ---
    p1_cfg = train_cfg["phase1"]
    print(
        f"\n--- Phase 1: Training Head Only (Freeze Layer count: {p1_cfg['freeze']}) ---"
    )

    # Instantiate model with domain pretrained weights
    model = model_class(pretrained_weights)
    project_path = os.path.join(RESULTS_DIR, model_type)

    model.train(
        data=dataset_yaml,
        epochs=p1_cfg["epochs"],
        patience=p1_cfg["patience"],
        imgsz=IMG_SIZE,
        batch=p1_cfg["batch_size"],
        project=project_path,
        name="phase1",
        freeze=p1_cfg["freeze"],
        device="0" if torch.cuda.is_available() else "cpu",
        verbose=False,
        **AUGMENTATION_CONFIG,
    )

    p1_best_weights = os.path.join(project_path, "phase1", "weights", "best.pt")
    print(f"Phase 1 training finished. Best weights saved to: {p1_best_weights}")

    # --- Phase 2: Global Finetuning (Backbone Unfrozen) ---
    p2_cfg = train_cfg["phase2"]
    print("\n--- Phase 2: Global Finetuning (Unfreeze Backbone) ---")

    # Load the best weights from Phase 1
    model_p2 = model_class(p1_best_weights)

    model_p2.train(
        data=dataset_yaml,
        epochs=p2_cfg["epochs"],
        patience=p2_cfg["patience"],
        imgsz=IMG_SIZE,
        batch=p2_cfg["batch_size"],
        project=project_path,
        name="phase2",
        freeze=p2_cfg["freeze"],  # freeze=0 (train all layers)
        device="0" if torch.cuda.is_available() else "cpu",
        verbose=False,
        **AUGMENTATION_CONFIG,
    )

    p2_best_weights = os.path.join(project_path, "phase2", "weights", "best.pt")
    print(
        f"Phase 2 global finetuning finished. Final best weights saved to: {p2_best_weights}"
    )
    print("=======================================================\n")


def train_faster_rcnn_model():
    """Finetunes Faster R-CNN using torch native classes and phased freezing."""
    print("\n=======================================================")
    print("STARTING FASTER R-CNN FINETUNING")
    print("=======================================================")

    project_path = os.path.join(RESULTS_DIR, "faster_rcnn")
    os.makedirs(project_path, exist_ok=True)

    # Prepare datasets & loaders
    train_dataset = ActiveLearningFasterRCNNDataset(
        CLAHE_DATASET_DIR, split="train", img_size=IMG_SIZE, augment=True
    )
    val_dataset = ActiveLearningFasterRCNNDataset(
        CLAHE_DATASET_DIR, split="val", img_size=IMG_SIZE, augment=False
    )

    p1_cfg = FASTER_RCNN_TRAIN_CONFIG["phase1"]
    p2_cfg = FASTER_RCNN_TRAIN_CONFIG["phase2"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=p1_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=p1_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # --- Phase 1: Train Head Only (Backbone Frozen) ---
    print("\n--- Phase 1: Training Head Only (Backbone Frozen) ---")
    model = get_faster_rcnn_model(num_classes=NUM_CLASSES, freeze_backbone=True)
    load_compatible_weights(model, PRETRAINED_FASTER_RCNN)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=p1_cfg["lr"],
        weight_decay=1e-4,
    )
    early_stopping = EarlyStopping(patience=p1_cfg["patience"])

    p1_dir = os.path.join(project_path, "phase1")
    os.makedirs(p1_dir, exist_ok=True)
    p1_weights_path = os.path.join(p1_dir, "best.pt")

    best_val_loss = float("inf")
    history = []

    for epoch in range(p1_cfg["epochs"]):
        train_loss, train_comps = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_comps = validate_one_epoch(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{p1_cfg['epochs']}: "
            f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), p1_weights_path)
            print(f"  --> Saved new best Phase 1 model (Val Loss: {val_loss:.4f})")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    print(f"Phase 1 complete. Best weights saved to: {p1_weights_path}")

    # --- Phase 2: Global Finetuning (Backbone Unfrozen) ---
    print("\n--- Phase 2: Global Finetuning (Unfreeze Backbone) ---")
    model_p2 = get_faster_rcnn_model(num_classes=NUM_CLASSES, freeze_backbone=False)
    load_compatible_weights(model_p2, p1_weights_path)
    model_p2.to(DEVICE)

    optimizer_p2 = torch.optim.Adam(
        [p for p in model_p2.parameters() if p.requires_grad],
        lr=p2_cfg["lr"],
        weight_decay=1e-4,
    )
    early_stopping_p2 = EarlyStopping(patience=p2_cfg["patience"])

    p2_dir = os.path.join(project_path, "phase2")
    os.makedirs(p2_dir, exist_ok=True)
    p2_weights_path = os.path.join(p2_dir, "best.pt")

    best_val_loss_p2 = float("inf")
    history_p2 = []

    for epoch in range(p2_cfg["epochs"]):
        train_loss, train_comps = train_one_epoch(model_p2, train_loader, optimizer_p2)
        val_loss, val_comps = validate_one_epoch(model_p2, val_loader)

        print(
            f"Epoch {epoch + 1}/{p2_cfg['epochs']}: "
            f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        history_p2.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val_loss_p2:
            best_val_loss_p2 = val_loss
            torch.save(model_p2.state_dict(), p2_weights_path)
            print(f"  --> Saved new best Phase 2 model (Val Loss: {val_loss:.4f})")

        early_stopping_p2(val_loss)
        if early_stopping_p2.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    print(f"Phase 2 complete. Final best weights saved to: {p2_weights_path}")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Finetuning Pipeline for Object Detectors"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["yolo", "rtdetr", "faster_rcnn", "all"],
        help="Specify which model to finetune. 'all' trains all models sequentially.",
    )
    args = parser.parse_args()

    # Pre-check CLAHE dataset exists
    if not os.path.exists(CLAHE_DATASET_DIR):
        raise FileNotFoundError(
            f"Preprocessed CLAHE dataset not found at {CLAHE_DATASET_DIR}. "
            f"Please run preprocess.py first."
        )

    # Create dataset yaml file for YOLO & RT-DETR
    dataset_yaml = create_dataset_yaml()

    # Launch training tasks
    if args.model in ["yolo", "all"]:
        train_yolo_rtdetr("yolo", dataset_yaml)

    if args.model in ["rtdetr", "all"]:
        train_yolo_rtdetr("rtdetr", dataset_yaml)

    if args.model in ["faster_rcnn", "all"]:
        train_faster_rcnn_model()


if __name__ == "__main__":
    main()
