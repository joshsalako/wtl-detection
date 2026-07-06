# Consensus Inference 

This module runs batch inference across all our model architectures (YOLO, RT-DETR, Faster R-CNN) and evaluates the results on the target holdout test/val cameras (`4R` and `5Z`). It uses a consensus clustering algorithm to identify ground-truth bounding boxes by discarding false positives and background artifacts that do not have agreement across the models.

## Usage

### 1. Configuration
Check `consensus_config.py` to adjust hyper-parameters:
- `TARGET_CAMERAS`: Cameras dedicated as your test/val set (defaults to `{"4R", "5Z"}`).
- `CONFIDENCE_THRESHOLD`: Prediction confidence threshold across all models (defaults to `0.1`).
- `IOU_THRESHOLD`: IoU threshold for overlapping prediction boxes to be grouped (defaults to `0.5`).
- `MIN_CONSENSUS_MODELS`: The minimum number of *unique* models that must predict a box in the cluster for it to be accepted as a ground truth (defaults to `3`).
- `MODEL_ZOO`: Dictionary defining paths to cycle 3 and cycle 4 pretrained weights.

### 2. Run Ensemble Inference
The inference script scours the raw `/srv/shared_leopard_toad` data lake for all imagery belonging to the target cameras. It will load each of the 6 models sequentially, run inference with CLAHE preprocessing applied, and output a detailed CSV of predictions.
```bash
python3 run_consensus_inference.py
```
> Note: If cycle 4 models are still training, the script will gracefully skip them and continue with the available ones.

### 3. Apply Consensus Filtering
Once inference finishes, trigger the clustering logic.
```bash
python3 apply_consensus.py
```
This script aggregates all raw predictions from the `outputs/raw_csvs/` folder. It uses a custom NMS/clustering implementation to intersect the predictions, keeping only those that pass the minimum model consensus count.

The accepted ground-truth predictions will be averaged together into YOLO-normalized `.txt` format and written to `outputs/yolo_labels/`. The output structure mirrors the original `/srv/shared_leopard_toad` folder structure so that they can be easily moved or visually evaluated.
