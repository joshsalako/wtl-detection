"""Centralized configuration file for the object detection finetuning pipeline.

Contains paths, training parameters, model configurations, and device settings.
"""

import os
import torch

# --- Device Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Directory Pathing Resolution ---
DETECTION_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.dirname(DETECTION_DIR)

# Dataset Paths
SRC_DATASET_DIR = os.path.join(RESEARCH_DIR, "dataset")
CLAHE_DATASET_DIR = os.path.join(RESEARCH_DIR, "dataset_clahe")
RESULTS_DIR = os.path.join(DETECTION_DIR, "results")

# Pretrained Weights Paths
PRETRAINED_YOLO = "/home/Joshua/Downloads/leopard_toad_identification/detection/pretraining/runs/detect/yolo_model/weights/best.pt"
PRETRAINED_RTDETR = "/home/Joshua/Downloads/leopard_toad_identification/detection/pretraining/runs/detect/rtdetr_finetuning/weights/best.pt"
PRETRAINED_FASTER_RCNN = "/home/Joshua/Downloads/leopard_toad_identification/detection/pretraining/runs/faster_rcnn/train_resnet50_1/weights/best.pt"

# --- Class Definitions ---
CLASSES = ["Other_Amphibian", "Small_Mammal", "Western_Leopard_Toad"]
NUM_CLASSES = len(CLASSES)

# --- Hyperparameters & Training Configurations ---
IMG_SIZE = 640

# Phased Freezing Configuration
YOLO_TRAIN_CONFIG = {
    "phase1": {"epochs": 100, "patience": 25, "batch_size": 16, "freeze": 15},
    "phase2": {"epochs": 100, "patience": 25, "batch_size": 16, "freeze": 0},
}

RTDETR_TRAIN_CONFIG = {
    "phase1": {"epochs": 100, "patience": 25, "batch_size": 16, "freeze": 15},
    "phase2": {"epochs": 100, "patience": 25, "batch_size": 16, "freeze": 0},
}

FASTER_RCNN_TRAIN_CONFIG = {
    "phase1": {
        "epochs": 100,
        "patience": 25,
        "batch_size": 16,
        "freeze_backbone": True,
        "lr": 0.0001,
    },
    "phase2": {
        "epochs": 100,
        "patience": 25,
        "batch_size": 16,
        "freeze_backbone": False,
        "lr": 0.00005,
    },
}

# --- Advanced Data Augmentation Configurations (for YOLO & RT-DETR) ---
AUGMENTATION_CONFIG = {
    "degrees": 15.0,  # random rotation (+/- degrees)
    "translate": 0.1,  # random translation (+/- fraction)
    "scale": 0.5,  # random scale (+/- gain)
    "shear": 2.0,  # random shear (+/- degrees)
    "perspective": 0.0001,  # random perspective (+/- fraction)
    "flipud": 0.5,  # random vertical flip probability
    "fliplr": 0.5,  # random horizontal flip probability
    "mosaic": 1.0,  # mosaic probability
    "mixup": 0.15,  # mixup probability
    "copy_paste": 0.2,  # copy-paste probability (excellent for small/big objects)
}
