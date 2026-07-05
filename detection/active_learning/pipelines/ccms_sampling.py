#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import sys
import os
import cv2

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEARNING_DIR = os.path.dirname(PIPELINES_DIR)
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)

# Import central configurations
from central_config import (
    CLASSES,
    CURATION_TARGET_CLASS,
    DEFAULT_CURATION_BUDGET,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_OCCURRENCE_THRESHOLD,
    DETECTION_THRESHOLDS,
    PRETRAINED_FASTER_RCNN,
    DEFAULT_NUM_WORKERS,
    FEATURE_EXTRACTION_BATCH_SIZE,
)


class ImageCropDataset(Dataset):
    """Dataset class that loads an image once, applies CLAHE, and returns all cropped patches for that image."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        # Group by image path to load each image exactly once
        self.image_paths = self.df["image_path"].unique()
        self.df_grouped = self.df.groupby("image_path")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        group = self.df_grouped.get_group(image_path)

        crops = []
        indices = []
        original_indices = group.index.values

        try:
            img = Image.open(image_path).convert("RGB")

            # Apply Grayscale and CLAHE preprocessing exactly ONCE
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(gray)
            img_gray_3c = cv2.cvtColor(cl, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(img_gray_3c)

            # Extract all crops
            for index, row in group.iterrows():
                xmin, ymin = max(0, int(row["xmin"])), max(0, int(row["ymin"]))
                xmax, ymax = (
                    min(img.width, int(row["xmax"])),
                    min(img.height, int(row["ymax"])),
                )

                if xmax <= xmin or ymax <= ymin:
                    crop = img.resize((224, 224))
                else:
                    crop = img.crop((xmin, ymin, xmax, ymax))

                if self.transform:
                    crop = self.transform(crop)

                crops.append(crop)
                indices.append(index)

            if crops:
                return torch.stack(crops), torch.tensor(indices, dtype=torch.long)
            else:
                return torch.empty((0, 3, 224, 224)), torch.empty(
                    (0,), dtype=torch.long
                )

        except Exception:
            # Fallback for failed image load
            if self.transform:
                return torch.zeros((len(original_indices), 3, 224, 224)), torch.tensor(
                    original_indices, dtype=torch.long
                )
            return None, original_indices


def collate_crops(batch):
    """Collates a list of (stacked_crops, indices) into a single batch."""
    crops_list, indices_list = zip(*batch)

    # Filter out empty tensors
    valid_crops = [c for c in crops_list if c is not None and c.size(0) > 0]
    valid_indices = [i for i in indices_list if i is not None and i.size(0) > 0]

    if not valid_crops:
        return torch.empty((0, 3, 224, 224)), torch.empty((0,), dtype=torch.long)

    return torch.cat(valid_crops, dim=0), torch.cat(valid_indices, dim=0)


class DomainFeatureExtractor(torch.nn.Module):
    """Wraps a trained Faster R-CNN ResNet50 FPN backbone and pools final feature maps to 2048-dim embeddings."""

    def __init__(self, resnet_body):
        super().__init__()
        self.resnet_body = resnet_body
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features_dict = self.resnet_body(x)
        features = features_dict["3"]
        pooled = self.pool(features)
        return pooled.view(pooled.size(0), -1)


class ImageNetFeatureExtractor(torch.nn.Module):
    """Wraps standard ImageNet ResNet50 model and pools to 2048-dim embeddings."""

    def __init__(self, resnet):
        super().__init__()
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        out = self.features(x)
        return out.view(out.size(0), -1)


def extract_features(model, dataloader, device):
    """Extracts deep embeddings from cropped patches in the loader."""
    model.eval()
    features = []
    indices = []

    with torch.no_grad():
        for batch_imgs, batch_idx in tqdm(
            dataloader, desc="Extracting visual features"
        ):
            batch_imgs = batch_imgs.to(device)
            out = model(batch_imgs)
            features.append(out.cpu().numpy())
            indices.append(batch_idx.numpy())

    return np.concatenate(features, axis=0), np.concatenate(indices, axis=0)


def extract_camera_id(subfolder):
    """Extracts camera ID from the subfolder string."""
    if pd.isna(subfolder):
        return "unknown"
    parts = str(subfolder).split("/")
    return parts[0]


def flag_static_triggers(df, iou_threshold=0.7, occurrence_threshold=15):
    """Clusters bounding boxes spatially across fixed cameras to flag stationary repeat triggers."""
    if len(df) == 0:
        return df

    df = df.copy()
    df["camera_id"] = df["subfolder"].apply(extract_camera_id)
    df["is_static_trigger"] = False

    for cam_id, group in df.groupby("camera_id"):
        if cam_id in ["unknown", "Observed", "Seen"]:
            continue

        boxes = group[["xmin", "ymin", "xmax", "ymax"]].values
        indices = group.index.values
        n = len(boxes)

        clusters = []
        for i in range(n):
            box = boxes[i]
            idx = indices[i]

            matched = False
            for cluster in clusters:
                rep = cluster["rep"]

                # Spatial IoU
                inter_x1 = max(box[0], rep[0])
                inter_y1 = max(box[1], rep[1])
                inter_x2 = min(box[2], rep[2])
                inter_y2 = min(box[3], rep[3])

                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area1 = (box[2] - box[0]) * (box[3] - box[1])
                area2 = (rep[2] - rep[0]) * (rep[3] - rep[1])
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                if iou >= iou_threshold:
                    cluster["indices"].append(idx)
                    m = len(cluster["indices"])
                    cluster["rep"] = [
                        (rep[0] * (m - 1) + box[0]) / m,
                        (rep[1] * (m - 1) + box[1]) / m,
                        (rep[2] * (m - 1) + box[2]) / m,
                        (rep[3] * (m - 1) + box[3]) / m,
                    ]
                    matched = True
                    break
            if not matched:
                clusters.append({"rep": list(box), "indices": [idx]})

        for cluster in clusters:
            count = len(cluster["indices"])
            if count > occurrence_threshold:
                df.loc[cluster["indices"], "is_static_trigger"] = True

    df = df.drop(columns=["camera_id"])
    return df


def compute_ccms_matrix(unique_images, df_boxes, embeddings, device):
    """Computes Category Conditioned Matching Similarity (CCMS) matrix between images using PyTorch."""
    df_boxes_reset = df_boxes.reset_index(drop=True)
    M = len(df_boxes_reset)
    N = len(unique_images)

    # Move embeddings to GPU and normalize
    emb_t = torch.tensor(embeddings, dtype=torch.float32, device=device)
    norms = torch.norm(emb_t, dim=1, keepdim=True) + 1e-8
    norm_emb = emb_t / norms

    # Map image paths and classes to integer indices
    img_to_idx = {img: i for i, img in enumerate(unique_images)}
    unique_classes = df_boxes_reset["class_name"].unique()
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}

    # Construct tensors
    img_indices = torch.tensor(
        [img_to_idx[row["image_path"]] for _, row in df_boxes_reset.iterrows()],
        dtype=torch.long,
        device=device,
    )
    class_indices = torch.tensor(
        [class_to_idx[row["class_name"]] for _, row in df_boxes_reset.iterrows()],
        dtype=torch.long,
        device=device,
    )
    confidences = torch.tensor(
        df_boxes_reset["confidence"].values, dtype=torch.float32, device=device
    )

    max_sim = torch.zeros((M, N), dtype=torch.float32, device=device)

    for j in range(N):
        objs_j_mask = img_indices == j
        if not objs_j_mask.any():
            continue

        objs_j_idx = torch.where(objs_j_mask)[0]
        emb_j = norm_emb[objs_j_idx]  # [K, D] where K is objects in image j
        class_j = class_indices[objs_j_idx]  # [K]

        # Calculate similarity of all M objects to the K objects in image j
        sim_to_j = torch.mm(norm_emb, emb_j.t())  # [M, K]

        # Zero out similarities between different classes
        match_mask = class_indices.unsqueeze(1) == class_j.unsqueeze(0)
        sim_to_j.masked_fill_(~match_mask, 0.0)

        # Get the max similarity for each of the M objects to any object in image j
        max_sim[:, j] = sim_to_j.max(dim=1).values

    # 4. Weight max similarities by confidence
    weighted_max_sim = max_sim * confidences.unsqueeze(1)

    # 5. Sum weighted max similarities over all objects inside image i
    img_indicator = torch.zeros((N, M), dtype=torch.float32, device=device)
    img_indicator[img_indices, torch.arange(M, device=device)] = 1.0

    S_prime_unnorm = torch.mm(img_indicator, weighted_max_sim)
    sum_conf = torch.mv(img_indicator, confidences)
    sum_conf_safe = torch.where(sum_conf == 0, torch.ones_like(sum_conf), sum_conf)

    S_prime = S_prime_unnorm / sum_conf_safe.unsqueeze(1)

    # Make symmetric
    S = 0.5 * (S_prime + S_prime.t())
    return S


def run_ccms_clustering(unique_images, S, k, max_iter=15):
    """Performs k-Center Greedy initialization and modified k-Means++ refinement based on CCMS using PyTorch."""
    N = len(unique_images)
    if N <= k:
        return unique_images

    # Distance matrix D = 1.0 - S
    D = torch.clamp(1.0 - S, min=0.0, max=1.0)

    # 1. k-Center Greedy Initialization
    torch.manual_seed(42)
    centers = [torch.randint(0, N, (1,)).item()]
    min_dist = D[:, centers[0]].clone()

    for _ in range(1, k):
        next_center = torch.argmax(min_dist).item()
        centers.append(next_center)
        min_dist = torch.minimum(min_dist, D[:, next_center])

    # 2. Modified k-Means++ Refinement
    centers_tensor = torch.tensor(centers, dtype=torch.long, device=S.device)
    for iteration in range(max_iter):
        assignments = torch.argmin(D[:, centers_tensor], dim=1)

        new_centers = []
        changed = False

        for cluster_idx in range(k):
            members = torch.where(assignments == cluster_idx)[0]
            if len(members) == 0:
                new_centers.append(centers_tensor[cluster_idx].item())
                continue

            sub_S = S[members][:, members]
            summed_sims = torch.sum(sub_S, dim=1)
            best_idx = members[torch.argmax(summed_sims)]
            new_centers.append(best_idx.item())

            if best_idx.item() != centers_tensor[cluster_idx].item():
                changed = True

        centers_tensor = torch.tensor(new_centers, dtype=torch.long, device=S.device)
        if not changed:
            print(f"CCMS Clustering: Converged after {iteration + 1} iterations.")
            break

    return [unique_images[idx] for idx in centers_tensor.tolist()]


def perform_diversity_sampling(
    curation_df, budget, feature_extractor, transform, batch_size, device, category_name
):
    """Runs crop feature extraction, CCMS scoring, and clustering to select diverse priority queries."""
    if len(curation_df) == 0:
        return pd.DataFrame()

    curation_df = curation_df.copy()
    curation_df["is_representative"] = False

    unique_images = curation_df["image_path"].unique().tolist()
    n_samples = min(budget, len(unique_images))
    if n_samples <= 0:
        return curation_df

    # Extract features for all boxes in curation pool
    dataset = ImageCropDataset(curation_df, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=FEATURE_EXTRACTION_BATCH_SIZE,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        collate_fn=collate_crops,
    )
    embeddings, idxs = extract_features(feature_extractor, dataloader, device)

    # Reorder embeddings to align with curation_df rows
    order = np.argsort(idxs)
    embeddings = embeddings[order]

    # Compute CCMS Matrix between unique images
    print(
        f"CCMS: Computing similarity matrix for '{category_name}' with {len(unique_images)} images..."
    )
    S = compute_ccms_matrix(unique_images, curation_df, embeddings, device)

    # Cluster using k-Center Greedy and Refinement
    print(
        f"CCMS: Running two-stage clustering for '{category_name}' ({n_samples} clusters)..."
    )
    representative_images = run_ccms_clustering(unique_images, S, n_samples)

    # Select the highest uncertainty box per representative image
    rep_mask = curation_df["image_path"].isin(representative_images)
    curation_df.loc[rep_mask, "is_representative"] = True

    # We want each representative image to be marked, but only one box per image is kept
    # So we sort by uncertainty descending and drop duplicate image paths for representatives
    curation_df = curation_df.sort_values(
        by=["is_representative", "uncertainty"], ascending=[False, False]
    )

    # Mark duplicates as not representative
    representatives_only = curation_df[curation_df["is_representative"] == True].copy()
    representatives_only = representatives_only.drop_duplicates(
        subset=["image_path"], keep="first"
    )

    curation_df["is_representative"] = False
    curation_df.loc[representatives_only.index, "is_representative"] = True

    return curation_df


def main():
    parser = argparse.ArgumentParser(
        description="CCMS & Diversity Curation Sampling Script."
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        required=True,
        help="Path to predictions with uncertainty.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output curation_priority.csv.",
    )
    parser.add_argument(
        "--resnet_weights",
        type=str,
        default=PRETRAINED_FASTER_RCNN,
        help="Path to ResNet50 weights.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=DEFAULT_CURATION_BUDGET,
        help="Curation budget.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="Filter IoU threshold.",
    )
    parser.add_argument(
        "--occurrence_threshold",
        type=int,
        default=DEFAULT_OCCURRENCE_THRESHOLD,
        help="Static trigger threshold.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Feature extraction batch size."
    )

    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)
    if df.empty:
        print("CCMS: Empty predictions CSV. Graceful exit.")
        df["is_representative"] = []
        df["curation_reason"] = []
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        return

    # Ensure uncertainty is in columns
    if "uncertainty" not in df.columns:
        df["uncertainty"] = 1.0 - df["confidence"]

    # 1. Flag Static Triggers
    df = flag_static_triggers(df, args.iou_threshold, args.occurrence_threshold)

    # 2. Assign Curation Reasons (for Label Studio)
    df["curation_reason"] = "Other active support class"
    target_mask = (df["class_name"] == CURATION_TARGET_CLASS) & (
        df["is_static_trigger"] == False
    )
    hard_neg_mask = df["is_static_trigger"] == True
    df.loc[target_mask, "curation_reason"] = (
        f"Target positive ({CURATION_TARGET_CLASS})"
    )
    df.loc[hard_neg_mask, "curation_reason"] = "Hard Negative (Static Background)"

    print(
        f"CCMS: Operating on global pool of {len(df['image_path'].unique())} unique images with budget {args.n_clusters}."
    )

    # 3. Load Domain-Pretrained Feature Extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = args.resnet_weights

    if not os.path.exists(weights_path):
        print(
            f"CCMS: Pretrained weights not found. Falling back to ImageNet ResNet50..."
        )
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        feature_extractor = ImageNetFeatureExtractor(resnet).to(device)
    else:
        print(f"CCMS: Loading domain-pretrained weights from {weights_path}")
        frcnn_model = fasterrcnn_resnet50_fpn_v2()
        in_features = frcnn_model.roi_heads.box_predictor.cls_score.in_features
        frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
        frcnn_model.load_state_dict(torch.load(weights_path, map_location=device))
        feature_extractor = DomainFeatureExtractor(frcnn_model.backbone.body).to(device)

    transform = v2.Compose(
        [
            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 4. Run global diversity clustering
    all_curated = perform_diversity_sampling(
        df,
        args.n_clusters,
        feature_extractor,
        transform,
        args.batch_size,
        device,
        "Global Curation Pool",
    )
    if all_curated.empty:
        print("CCMS: No predictions require curation.")
        return

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    all_curated.to_csv(args.output_csv, index=False)

    rep_count = all_curated["is_representative"].sum()
    print(
        f"CCMS: Saved {rep_count} representative priority candidates to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
