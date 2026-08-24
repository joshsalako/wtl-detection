# Annotation-Efficient Western Leopard Toad Detection

Code and evaluation outputs for:

> Annotation-Efficient Object Detection of Endangered Western Leopard Toads in Camera Trap Imagery for Assessing Wildlife Tunnel Use

Joshua Salako, Kim Gordon, and Lorène Jeantet.

## Overview

This repository implements the computer-vision pipeline developed for the Western Leopard Toad (WLT) Underpass Project in South Africa. The project uses camera-trap imagery from wildlife tunnels to monitor amphibian movement while reducing the manual annotation required for large image collections.

The study begins with 20 annotated images and combines CLAHE contrast enhancement, domain-specific transfer learning, object detection, active learning, and spatial post-processing. YOLO, RT-DETR, Faster R-CNN, and MegaDetector are evaluated on three classes:

- `Other_Amphibian`
- `Small_Mammal`
- `Western_Leopard_Toad`

Faster R-CNN produced the strongest WLT detection results in the study, with an F1 score of 0.86 and AP50 of 0.94. Applied to more than one million images, the final pipeline identified 730 WLT occurrences across 387 non-redundant detection events and mapped 15 re-encounters across opposite sides of the tunnels.

## Repository contents

```text
detection/             Training, inference, active-learning, consensus, and evaluation pipelines
evaluation_scripts/    Standalone prediction and metric-generation scripts
evaluation_outputs/    Summary tables and image-level ROC plots
```

The datasets, camera-trap image pool, and pretrained model weights are not included. Their locations are configured locally in `detection/config.py` and `detection/active_learning/central_config.py`.

## Reproduction workflow

From the repository root:

```bash
python detection/preprocess.py
python detection/train.py --model all
python detection/evaluate.py --split test
```

To run active learning, prepare the cycle dataset and pretrained weights, then run from `detection/active_learning`:

```bash
python pipelines/run_active_learning_loop.py \
  --model_type yolo rtdetr faster_rcnn \
  --budget 100
```

The active-learning loop uses Difficulty Calibrated Uncertainty Sampling (DCUS) followed by Category Conditioned Matching Similarity (CCMS) to select informative and visually diverse images for annotation. Newly annotated images are ingested with `pipelines/ingest_annotations.py` to create the next cycle.

## Evaluation outputs

The repository includes summary results in [`evaluation_outputs/`](evaluation_outputs/) and the complete active-learning evaluation report in [`detection/evaluation/results/files/final_evaluation_results.md`](detection/evaluation/results/files/final_evaluation_results.md). The evaluation framework reports detection-level mAP and class AP, as well as image-level ROC-AUC, F1, precision, recall, and confusion-matrix results.

## Citation

Salako, J., Gordon, K., and Jeantet, L. *Annotation-Efficient Object Detection of Endangered Western Leopard Toads in Camera Trap Imagery for Assessing Wildlife Tunnel Use.*
