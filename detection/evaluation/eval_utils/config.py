import os
import torch
import sys

CLASSES = ["Other_Amphibian", "Small_Mammal", "Western_Leopard_Toad"]
CLASSES_DICT = {i: name for i, name in enumerate(CLASSES)}

TEST_DIR = "/home/Joshua/Downloads/active_learning_research/dataset/test_final"
VAL_DIR = "/home/Joshua/Downloads/active_learning_research/dataset/val_final"

TEST_FULL_CAMERA = "5Z"
VAL_FULL_CAMERA = "4R"

EVAL_DIR = "/home/Joshua/Downloads/active_learning_research/detection/evaluation"
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
FILES_DIR = os.path.join(RESULTS_DIR, "files")

AL_DIR = "/home/Joshua/Downloads/active_learning_research/detection/active_learning"
if AL_DIR not in sys.path:
    sys.path.append(AL_DIR)
from central_config import UNLABELED_POOL_DIRS

MODELS = ["yolo_clahe", "faster_rcnn_clahe", "rtdetr_clahe"]
CYCLES = ["cycle_0", "cycle_1", "cycle_2", "cycle_3", "cycle_4"]

POST_PROCESS_IOU_THRESHOLD = 0.8
POST_PROCESS_OCCURRENCE_THRESHOLD = 35
MIN_CONF_THRESHOLD = 0.05

# --- Hardware & Inference Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 640
YOLO_BATCH_SIZE = 128
FASTER_RCNN_BATCH_SIZE = 112
MD_BATCH_SIZE = 128
MD_WEIGHTS_PATH = os.path.join(EVAL_DIR, "weights", "md_v5a.0.0.pt")

# --- Class Definitions ---
WLT_CLASS_ID = 2
SMALL_MAMMAL_CLASS_ID = 1
OTHER_AMPHIBIAN_CLASS_ID = 0

# --- Plotting & Reporting Settings ---
WLT_PLOT_TITLE = "WLT vs Background"
AGNOSTIC_PLOT_TITLE = "Class Agnostic vs Background"
WLT_PREFIX = "wlt"
AGNOSTIC_PREFIX = "agnostic"

# --- Output Files ---
DETECTION_METRICS_CSV = "detection_metrics.csv"
IMAGE_LEVEL_WLT_CSV = "image_level_wlt.csv"
IMAGE_LEVEL_AGNOSTIC_CSV = "image_level_agnostic.csv"
IMAGE_LEVEL_WLT_JSON = "image_level_wlt_full.json"
IMAGE_LEVEL_AGNOSTIC_JSON = "image_level_agnostic_full.json"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)
