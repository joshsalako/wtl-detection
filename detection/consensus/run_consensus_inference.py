import os
import sys
import csv
from pathlib import Path

CONSENSUS_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.dirname(CONSENSUS_DIR)
ACTIVE_LEARNING_DIR = os.path.join(DETECTION_DIR, "active_learning")

if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)
if DETECTION_DIR not in sys.path:
    sys.path.append(DETECTION_DIR)

PIPELINES_DIR = os.path.join(ACTIVE_LEARNING_DIR, "pipelines")
if PIPELINES_DIR not in sys.path:
    sys.path.append(PIPELINES_DIR)

from consensus_config import (
    UNLABELED_POOL_DIRS,
    TARGET_CAMERAS,
    MODEL_ZOO,
    CSV_OUTPUT_DIR,
    CONFIDENCE_THRESHOLD,
)

from pipelines.run_inference_pipeline import process_all_images
from pipelines.faster_rcnn_utils import get_faster_rcnn_model, load_compatible_weights
import central_config
from central_config import (
    CLASSES,
    DEFAULT_DEVICE,
    ULTRALYTICS_BATCH_SIZE,
    FASTER_RCNN_INFERENCE_BATCH_SIZE,
    DEFAULT_IMG_SIZE,
)
from ultralytics import RTDETR, YOLO

# Override inference threshold in central_config
central_config.INFERENCE_CONF_THRESHOLD = CONFIDENCE_THRESHOLD
# Override DETECTION_THRESHOLDS to ensure it doesn't filter things out higher than our threshold
for i in range(len(CLASSES)):
    central_config.DETECTION_THRESHOLDS[i] = CONFIDENCE_THRESHOLD


def main():
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    device = DEFAULT_DEVICE

    # Gather all images
    print("Gathering test and val images from target cameras...")
    all_images = []
    for year, dir_path in UNLABELED_POOL_DIRS.items():
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
        base_path = Path(dir_path)
        for f in base_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                if any(cam in str(f) for cam in TARGET_CAMERAS):
                    all_images.append(f)

    if not all_images:
        print("No images found to process. Exiting.")
        return

    print(f"Found {len(all_images)} images to process.")

    # Run inference for each model
    for model_key, model_path in MODEL_ZOO.items():
        print(f"\n=========================================")
        print(f"Processing model: {model_key}")
        print(f"Model path: {model_path}")
        print(f"=========================================")

        output_csv_path = os.path.join(CSV_OUTPUT_DIR, f"{model_key}_predictions.csv")
        if os.path.exists(output_csv_path):
            print(f"Predictions already exist at {output_csv_path}. Skipping.")
            continue

        if not os.path.exists(model_path):
            print(f"WARNING: Model file {model_path} does not exist. Skipping.")
            continue

        if "faster_rcnn" in model_key:
            model_type = "faster_rcnn"
            model = get_faster_rcnn_model(num_classes=len(CLASSES))
            load_compatible_weights(model, model_path)
            model.to(device)
            model.eval()
            batch_size = FASTER_RCNN_INFERENCE_BATCH_SIZE
        elif "rtdetr" in model_key:
            model_type = "rtdetr"
            model = RTDETR(model_path)
            batch_size = ULTRALYTICS_BATCH_SIZE
        else:
            model_type = "yolo"
            model = YOLO(model_path)
            batch_size = ULTRALYTICS_BATCH_SIZE

        with open(output_csv_path, mode="w", newline="") as f_all:
            all_writer = csv.writer(f_all)
            headers = [
                "image_path",
                "image_name",
                "subfolder",
                "class_id",
                "class_name",
                "confidence",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ]
            all_writer.writerow(headers)

            grand_total_boxes = process_all_images(
                images=all_images,
                model=model,
                model_type=model_type,
                img_size=DEFAULT_IMG_SIZE,
                batch_size=batch_size,
                device=device,
                all_writer=all_writer,
                apply_clahe_flag=True,  # We apply CLAHE as these models were trained on it
            )
            print(
                f"Model {model_key} generated {grand_total_boxes} total bounding boxes."
            )


if __name__ == "__main__":
    main()
