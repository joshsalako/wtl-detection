import os
import sys

# --- Directory Pathing Resolution ---
ACTIVE_LEARNING_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.dirname(ACTIVE_LEARNING_DIR)
RESEARCH_DIR = os.path.dirname(DETECTION_DIR)

# Insert the parent directory into sys.path to access the global config
if DETECTION_DIR not in sys.path:
    sys.path.append(DETECTION_DIR)

from config import (
    CLASSES,
    NUM_CLASSES,
    PRETRAINED_YOLO,
    PRETRAINED_RTDETR,
    PRETRAINED_FASTER_RCNN,
    YOLO_TRAIN_CONFIG,
    RTDETR_TRAIN_CONFIG,
    FASTER_RCNN_TRAIN_CONFIG,
    DEVICE,
    CLAHE_DATASET_DIR,
    AUGMENTATION_CONFIG,
)

# Convert CLASSES list to dictionary format needed for AL pipeline
CLASSES_DICT = {i: name for i, name in enumerate(CLASSES)}

# --- Default Pipeline File Paths ---
DEFAULT_OUTPUT_DIR = os.path.join(
    DETECTION_DIR, "results", "active_learning_predictions"
)

# --- Default Inference Hyperparameters ---
DEFAULT_IMG_SIZE = 640
DEFAULT_DEVICE = DEVICE

# Batch Sizes and Workers
ULTRALYTICS_BATCH_SIZE = 256
FASTER_RCNN_BATCH_SIZE = 64
FASTER_RCNN_INFERENCE_BATCH_SIZE = 112
FEATURE_EXTRACTION_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 16

# --- Bounding Box Spatial Filtering Settings ---
DEFAULT_IOU_THRESHOLD = 0.8
DEFAULT_OCCURRENCE_THRESHOLD = 30
INFERENCE_CONF_THRESHOLD = 0.05
DCUS_CONF_THRESHOLD = 0.25
EXCLUDED_CAMERAS = {"4R", "5Z"}

# Optimal validation analytical thresholds based on F1-Score maximization
ORIGINAL_DETECTION_THRESHOLDS = {
    "Other_Amphibian": 0.1,
    "Small_Mammal": 0.1,
    "Western_Leopard_Toad": 0.1,
}

# Resolve DETECTION_THRESHOLDS dynamically for target classes
DETECTION_THRESHOLDS = {}
for i, name in CLASSES_DICT.items():
    if name in ORIGINAL_DETECTION_THRESHOLDS:
        DETECTION_THRESHOLDS[i] = ORIGINAL_DETECTION_THRESHOLDS[name]
    else:
        DETECTION_THRESHOLDS[i] = 0.25  # Generic default threshold

# --- Active Learning Curation Settings ---
CURATION_TARGET_CLASS = "Western_Leopard_Toad"

DCUS_BUDGET_MULTIPLIER = (
    200  # Multiplier to determine how many top uncertain images DCUS hands to CCMS
)

DCUS_POOL_RATIO_UNCERTAIN = 0.4
DCUS_POOL_RATIO_CERTAIN = 0.5
DCUS_POOL_RATIO_RANDOM = 0.1

# Default total human annotation budget (n_clusters)
DEFAULT_CURATION_BUDGET = 100

# Input year directories to run inference on for unlabeled pool
UNLABELED_POOL_DIRS = {
    "2023": "/srv/shared_leopard_toad/2023",
    "2024": "/srv/shared_leopard_toad/2024",
    "2025": "/srv/shared_leopard_toad/2025/Documents",
}
