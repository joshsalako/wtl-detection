import os
import sys

# Add root paths
CONSENSUS_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.dirname(CONSENSUS_DIR)
RESEARCH_DIR = os.path.dirname(DETECTION_DIR)

if DETECTION_DIR not in sys.path:
    sys.path.append(DETECTION_DIR)

from central_config import UNLABELED_POOL_DIRS

# The val and test images are drawn exclusively from these withheld cameras.
TARGET_CAMERAS = {"4R", "5Z"}
RAW_DATASET_BASE = "/srv/shared_leopard_toad"

# Output Directories
OUTPUT_DIR = os.path.join(CONSENSUS_DIR, "outputs")
CSV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "raw_csvs")
YOLO_LABELS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "yolo_labels")

# Consensus Parameters
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5
MIN_CONSENSUS_MODELS = 3

# Model Paths
# Note: As per user instructions, all cycle 4 models are assumed to have phase 2 completed soon.
MODEL_ZOO = {
    "yolo_cycle3": os.path.join(
        DETECTION_DIR,
        "active_learning/yolo_clahe/runs/cycle_3_pretrained_phase2/weights/best.pt",
    ),
    "yolo_cycle4": os.path.join(
        DETECTION_DIR,
        "active_learning/yolo_clahe/runs/cycle_4_pretrained_phase2/weights/best.pt",
    ),
    "rtdetr_cycle3": os.path.join(
        DETECTION_DIR,
        "active_learning/rtdetr_clahe/runs/cycle_3_pretrained_phase2/weights/best.pt",
    ),
    "rtdetr_cycle4": os.path.join(
        DETECTION_DIR,
        "active_learning/rtdetr_clahe/runs/cycle_4_pretrained_phase2/weights/best.pt",
    ),
    "faster_rcnn_cycle3": os.path.join(
        DETECTION_DIR,
        "active_learning/faster_rcnn_clahe/runs/cycle_3_pretrained_phase2/weights/best.pt",
    ),
    "faster_rcnn_cycle4": os.path.join(
        DETECTION_DIR,
        "active_learning/faster_rcnn_clahe/runs/cycle_4_pretrained_phase2/weights/best.pt",
    ),
}
