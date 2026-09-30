"""Lossless class-score export and explicit entropy semantics for AL CSVs."""

import csv
import json
import math

PROBABILITY_COLUMNS = ("class_probabilities", "probability_kind")


def compute_class_entropy(values, kind, num_classes):
    """Use actual outputs: categorical softmax or summed Bernoulli entropy.

    Faster R-CNN vectors include background at index zero. Ultralytics
    sigmoid scores are independent probabilities, not a categorical simplex.
    """
    if isinstance(values, str):
        values = json.loads(values)
    if kind not in ("sigmoid", "softmax_background"):
        raise ValueError(f"Unsupported probability_kind: {kind!r}")
    expected = num_classes + (kind == "softmax_background")
    if not isinstance(values, (list, tuple)) or len(values) != expected:
        raise ValueError(f"Expected {expected} class probabilities, got {values!r}")
    probabilities = [float(p) for p in values]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Class probabilities must be finite and in [0, 1]")
    if kind == "softmax_background" and not math.isclose(
        sum(probabilities), 1.0, abs_tol=1e-3
    ):
        raise ValueError("Softmax class probabilities must sum to one")

    def xlogx(p):
        return p * math.log(p) if p > 0 else 0.0

    entropy = -sum(xlogx(p) for p in probabilities)
    if kind == "sigmoid":
        entropy -= sum(xlogx(1 - p) for p in probabilities)
    return entropy


def require_probability_csv(path):
    """Reject legacy caches rather than reconstruct missing class scores."""
    with open(path, newline="") as stream:
        fields = csv.DictReader(stream).fieldnames or []
    if not set(PROBABILITY_COLUMNS).issubset(fields):
        raise ValueError(
            f"{path} lacks full class probabilities. Re-run inference with --force; "
            "top-confidence-only CSVs cannot provide actual class entropy."
        )


def ultralytics_probability_predictor(model_type):
    """Predictors for full-score YOLO and RT-DETR heads (Ultralytics 8.3.200).

    Unsupported/top-k-only head layouts fail explicitly instead of guessing
    the missing probabilities. The retained class scores are never rounded.
    """
    import torch
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.models.rtdetr.predict import RTDETRPredictor
    from ultralytics.utils import ops

    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    class ProbabilityYOLOPredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            raw = preds[0] if isinstance(preds, (tuple, list)) else preds
            nc = len(self.model.names)
            if (
                getattr(self.model, "end2end", False)
                or raw.ndim != 3
                or raw.shape[1] != 4 + nc
            ):
                raise ValueError("Full-score YOLO head required for entropy export")
            # NMS preserves extra channels as per-candidate metadata. Carry the
            # complete score vector through the very same keep/reorder indices.
            enriched = torch.cat((raw, raw[:, 4 : 4 + nc].clone()), dim=1)
            kept = non_max_suppression(
                enriched,
                self.args.conf,
                self.args.iou,
                self.args.classes,
                self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=nc,
            )
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            results = []
            for pred, original, path in zip(kept, orig_imgs, self.batch[0]):
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:], pred[:, :4], original.shape
                )
                result = Results(
                    original, path=path, names=self.model.names, boxes=pred[:, :6]
                )
                result.class_probabilities = pred[:, 6 : 6 + nc]
                results.append(result)
            return results

    class ProbabilityRTDETRPredictor(RTDETRPredictor):
        def postprocess(self, preds, img, orig_imgs):
            raw = preds[0] if isinstance(preds, (tuple, list)) else preds
            nc = len(self.model.names)
            if raw.ndim != 3 or raw.shape[-1] != 4 + nc:
                raise ValueError("Full-score RT-DETR head required for entropy export")
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            results = []
            for pred, original, path in zip(raw, orig_imgs, self.batch[0]):
                probabilities = pred[:, 4:]
                confidence, labels = probabilities.max(-1)
                mask = confidence > self.args.conf
                if self.args.classes is not None:
                    allowed = torch.tensor(self.args.classes, device=labels.device)
                    mask &= (labels[:, None] == allowed).any(1)
                # Match Ultralytics' confidence sort and max_det truncation.
                selected = torch.where(mask)[0]
                selected = selected[confidence[selected].argsort(descending=True)][
                    : self.args.max_det
                ]
                boxes = ops.xywh2xyxy(pred[:, :4])[selected]
                boxes[:, [0, 2]] *= original.shape[1]
                boxes[:, [1, 3]] *= original.shape[0]
                output = torch.cat(
                    (boxes, confidence[selected, None], labels[selected, None]), dim=1
                )
                result = Results(
                    original, path=path, names=self.model.names, boxes=output
                )
                result.class_probabilities = probabilities[selected]
                results.append(result)
            return results

    return {"yolo": ProbabilityYOLOPredictor, "rtdetr": ProbabilityRTDETRPredictor}[
        model_type
    ]


def enable_faster_rcnn_probabilities(model):
    """Carry proposal softmax vectors through torchvision's detection filtering.

    Preserve foreground box decoding, score thresholds, small-box removal,
    classwise NMS and detections_per_img; background stays in the score vector.
    """
    import types
    import torch
    from torchvision.ops import boxes as box_ops

    if getattr(model.roi_heads, "_exports_class_probabilities", False):
        return

    def postprocess(self, class_logits, box_regression, proposals, image_shapes):
        nc = class_logits.shape[-1]
        counts = [len(p) for p in proposals]
        decoded = self.box_coder.decode(box_regression, proposals).split(counts, 0)
        probabilities = class_logits.softmax(-1).split(counts, 0)
        all_boxes, all_scores, all_labels, all_probabilities = [], [], [], []
        for boxes, probs, shape in zip(decoded, probabilities, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, shape)[:, 1:].reshape(-1, 4)
            scores = probs[:, 1:].reshape(-1)
            labels = (
                torch.arange(1, nc, device=probs.device)
                .expand(len(probs), -1)
                .reshape(-1)
            )
            proposal_ids = torch.arange(
                len(probs), device=probs.device
            ).repeat_interleave(nc - 1)
            keep = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels, proposal_ids = (
                boxes[keep],
                scores[keep],
                labels[keep],
                proposal_ids[keep],
            )
            keep = box_ops.remove_small_boxes(boxes, 1e-2)
            boxes, scores, labels, proposal_ids = (
                boxes[keep],
                scores[keep],
                labels[keep],
                proposal_ids[keep],
            )
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)[
                : self.detections_per_img
            ]
            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_labels.append(labels[keep])
            all_probabilities.append(probs[proposal_ids[keep]])
        self._retained_probabilities = all_probabilities
        return all_boxes, all_scores, all_labels

    def attach(module, inputs, output):
        detections, losses = output
        if not module.training:
            for detection, probabilities in zip(
                detections, module._retained_probabilities
            ):
                detection["class_probabilities"] = probabilities
            del module._retained_probabilities
        return detections, losses

    model.roi_heads.postprocess_detections = types.MethodType(
        postprocess, model.roi_heads
    )
    model.roi_heads.register_forward_hook(attach)
    model.roi_heads._exports_class_probabilities = True
