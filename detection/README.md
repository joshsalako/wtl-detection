# Object Detection Finetuning Pipeline

This directory contains the central components for preprocessing datasets, training, and evaluating various object detection models (YOLO, RT-DETR, and Faster R-CNN). The pipeline focuses on finetuning domain-pretrained weights using a phased freezing strategy.

## Directory Structure & Files

- **`config.py`**: The central configuration hub. Contains dataset paths, hyperparameters, pretrained weight references, class definitions, and advanced augmentation configurations used across the models.
- **`preprocess.py`**: An offline preprocessing script that applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the dataset images to amplify low-contrast details. It automatically discovers and processes the `dataset` into a new `dataset_clahe` directory.
- **`faster_rcnn_utils.py`**: Custom PyTorch utilities specifically for handling Faster R-CNN. This includes the dataset loader (`ActiveLearningFasterRCNNDataset`) which implements **online letterboxing** (to ensure aspect-ratio preservation), CutMix, and other augmentations, as well as the training/validation epoch loops.
- **`train.py`**: The unified training script. It runs a **two-phase freezing strategy**:
  - **Phase 1**: Adapts the detection head only while keeping the backbone frozen.
  - **Phase 2**: Unfreezes the backbone for global network optimization.
  - Usage: `python train.py --model [yolo|rtdetr|faster_rcnn|all]`
- **`evaluate.py`**: The evaluation script. Evaluates the best checkpoints on the test split, computes standard metrics (Precision, Recall, mAP@0.5), generates Precision-Recall curves, confusion matrices, and outputs bounding box visualization grids (`val_batch...labels.jpg`) for comparison.
  - Usage: `python evaluate.py --split test`
- **`results/`**: (Auto-generated) Contains all saved artifacts during training and evaluation, organized by model and phase, including best weights and validation plots.

## Key Features

1. **Consistent Aspect-Ratio Preservation**: All models (YOLO, RT-DETR, Faster R-CNN) evaluate and train on identically letterboxed images (gray padding to perfectly square 640x640 dimensions) to prevent object distortion. This is natively managed in Ultralytics by turning off rectangular batches (`rect=False`), and managed dynamically for PyTorch Faster R-CNN via the custom dataloader.
2. **Phased Freezing**: Fine-tuning is stabilized by first locking the backbone weights to train the head, and subsequently unlocking the whole network.
3. **CLAHE Preprocessing**: The dataset contrast is enhanced offline prior to training to help distinguish objects (like frogs) in difficult lighting or camouflage environments.

## Getting Started

1. **Preprocess Data**: Run `python preprocess.py` to create the CLAHE-enhanced dataset.
2. **Train Models**: Run `python train.py --model all` to sequentially finetune YOLO, RT-DETR, and Faster R-CNN.
3. **Evaluate Models**: Run `python evaluate.py --split test` to generate comparison metrics and plots in the `results/comparison/` directory.
