# Western Leopard Toad Detection & Active Learning

This repository contains the codebase for research into the automated detection of the endangered **Western Leopard Toad (WLT)** using camera trap imagery. It implements an end-to-end computer vision pipeline featuring multi-model evaluation and an Active Learning framework to drastically reduce human annotation effort.

## Overview
Camera traps generate massive volumes of imagery, making manual curation unscalable. This research aims to identify the optimal object detection architecture for detecting WLTs (alongside `Small_Mammal` and `Other_Amphibian` classes) and implements an Active Learning pipeline to autonomously surface the most informative, highly-uncertain images from an unlabeled pool of nearly 1 million images for human review.

## Key Implementations

* **Multi-Model Object Detection:** End-to-end training, inference, and evaluation pipelines for three state-of-the-art architectures:
  * **YOLOv8**
  * **RT-DETR**
  * **Faster R-CNN**
* **Dynamic Preprocessing:** On-the-fly image preprocessing featuring Contrast Limited Adaptive Histogram Equalization (CLAHE) to handle extreme variations in nighttime illumination and camera flash washouts.
* **Robust Augmentations:** A customized `Albumentations` pipeline that applies scale-aware geometric transforms, bounding box tracking, and safe dropping for heavily truncated boxes.
* **Active Learning Framework:** 
  * **DCUS (Dynamic Curation using Uncertainty Sampling):** Analyzes model confidence distributions to surface boundary-case images.
  * **CCMS (Cross-Camera Marginalization Sampling):** Ensures curated images are spatially diverse and not biased toward a single high-activity camera.
* **Comprehensive Evaluation:** Dedicated evaluation scripts mapping bounding-box metrics (mAP, PR curves) to actionable Image-Level metrics (ROC, F1-Scores) to determine true biological presence/absence.

## Structure
* `detection/`: Core training, inference, and architectural scripts.
* `detection/active_learning/`: The active learning sampler (DCUS/CCMS) and central configuration schemas.
* `evaluation_scripts/`: Scripts for generating threshold-optimized evaluation reports and plots.
* `dataset/`: Ground-truth bounding box datasets (YOLO format).
