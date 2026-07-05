import os

# Root directories
BASE_DIR = "/home/Joshua/Downloads/active_learning_research"
RESULTS_DIR = os.path.join(BASE_DIR, "detection/results")

# Dataset paths
TEST_FULL_DIR = os.path.join(BASE_DIR, "dataset_clahe/test_full")
UNLABELLED_POOL_DIR = (
    "/media/Project-drive/shared_leopard_toad_2/shared_leopard_toad_clahe/"
)
IMAGE_MAPPING_CSV = os.path.join(BASE_DIR, "dataset_clahe/image_mapping.csv")

# Model paths
YOLO_WEIGHTS = os.path.join(RESULTS_DIR, "yolo/phase2/weights/best.pt")
RTDETR_WEIGHTS = os.path.join(RESULTS_DIR, "rtdetr/phase2/weights/best.pt")
FASTER_RCNN_WEIGHTS = os.path.join(RESULTS_DIR, "faster_rcnn/phase2/best.pt")
DATASET_YAML = os.path.join(RESULTS_DIR, "dataset_clahe.yaml")

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class mappings
CLASS_NAMES = {0: "Other_Amphibian", 1: "Small_Mammal", 2: "Western_Leopard_Toad"}
NUM_CLASSES = 3

DEVICE = "cuda"

# Inference Parameters
IMG_SIZE = 640
CONF_THRESHOLD = 0.05

# Batch Sizes & Workers
ULTRALYTICS_BATCH_SIZE = 256
FASTER_RCNN_BATCH_SIZE = 64
FASTER_RCNN_NUM_WORKERS = 16
EVAL_BATCH_SIZE = 128
EVAL_NUM_WORKERS = 16
