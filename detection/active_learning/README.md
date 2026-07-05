# Active Learning Pipeline

This directory contains the orchestrator and pipeline scripts to run Active Learning iterations for object detection models (YOLO, RT-DETR, Faster R-CNN).

## Directory Structure
- `central_config.py`: Global configuration containing class definitions, hyperparameter presets for training, and data augmentation settings.
- `pipelines/`: Contains the individual python scripts executed by the main orchestrator loop.
  - `run_active_learning_loop.py`: The main orchestrator script.
  - `train_model.py`: Script to train/fine-tune models using phased freezing.
  - `run_inference_pipeline.py`: Runs batch inference over the `/srv/` unlabeled pool and applies on-the-fly CLAHE.
  - `dcus_sampling.py`: Difficulty Calibrated Uncertainty Sampling logic.
  - `ccms_sampling.py`: Category Conditioned Matching Similarity logic for final diverse sample selection.
  - `gather_annotations.py`: Collects the queried images, copies them to the `to_annotate` folder, and logs them in `already_sampled.csv`.
  - `ingest_annotations.py`: Merges newly annotated images and labels into the dataset to advance to the next cycle, automatically applying CLAHE to the new images.

## Pre-requisites
1. **Pretrained Base Weights**: The pipeline expects base pretraining weights to exist for the models you want to use. These paths are configured in `central_config.py`.
2. **Cycle Data**: For each cycle (e.g. `cycle_0`), the pipeline expects the corresponding training dataset to be available at:
   `data/{model_type}/pretrained/cycle_{cycle}/`
   This folder should contain a standard YOLO format dataset structure (`train/`, `val/`, `test/`).
3. **Unlabeled Pool**: The inference script automatically scans the `/srv/shared_leopard_toad` directory for unlabeled data. 

## Continuing from Pre-trained Cycle 0 Models
We have automatically copied your trained Cycle 0 models from `detection/results/` into the `runs/` folder of this active learning directory. 

Because the trained weights already exist in the target paths, `run_active_learning_loop.py` will **automatically skip Phase 1 (Model Training)** for Cycle 0 and proceed directly to Phase 2 (Batch Inference).

**Important**: Ensure that you copy or symlink your Cycle 0 validation dataset into `data/{model_type}/pretrained/cycle_0/val` (and similarly for other models). The DCUS sampling algorithm requires the validation set to calibrate uncertainty thresholds!

## How to Run

To execute the full active learning loop for a cycle (defaults to 100 annotation budget):
```bash
python pipelines/run_active_learning_loop.py --model_type yolo rtdetr faster_rcnn --budget 100
```

By default, the script enforces the `clahe` and `pretrained` parameters internally, meaning you do not need to specify them.

## Ingesting New Annotations

When a cycle completes, the images selected for manual labeling are placed in the `to_annotate/` directory. Once you have annotated these images, you must merge them into the dataset to create the next cycle before running the loop again.

Place your annotated images and YOLO `.txt` labels in an `images` and `labels` subfolder within the `annotated/` directory (e.g., `annotated/yolo_cycle_0/images` and `annotated/yolo_cycle_0/labels`). Then, run the ingestion script:

```bash
python pipelines/ingest_annotations.py --annotated_dir annotated/yolo_cycle_0
```

This script will automatically:
1. Detect the current cycle and create the dataset for the next cycle (e.g., `cycle_1`).
2. Copy all data from the previous cycle.
3. Apply CLAHE preprocessing to your new images and merge them.
4. Merge the new labels and clear dataset caches.

*Note: You can pass `--no_clahe` if you wish to skip applying CLAHE to the newly ingested images.*

## Outputs
- **Models**: Saved in `{model_type}_clahe/runs/cycle_{X}_pretrained_phase2/`
- **Predictions**: Saved in `../results/detect_{model_type}_cycle{X}_clahe_pretrained/`
- **Annotations**: Gathered candidates are copied into `to_annotate/{model_type}_cycle_{X}/`
