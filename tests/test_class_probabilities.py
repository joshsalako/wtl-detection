"""Check entropy semantics and score/box alignment using real detector libraries."""

import csv
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

PIPELINES = Path(__file__).resolve().parents[1] / "detection/active_learning/pipelines"
sys.path.insert(0, str(PIPELINES))
from class_probabilities import (
    compute_class_entropy,
    require_probability_csv,
    ultralytics_probability_predictor,
    enable_faster_rcnn_probabilities,
)


class EntropyTests(unittest.TestCase):
    def test_full_vector_changes_entropy_at_same_top_confidence(self):
        first = compute_class_entropy([0.8, 0.1, 0.1], "softmax_background", 2)
        second = compute_class_entropy([0.8, 0.19, 0.01], "softmax_background", 2)
        self.assertGreater(first, second)

    def test_sigmoid_is_bernoulli_not_synthetic_categorical(self):
        self.assertAlmostEqual(
            compute_class_entropy("[0.5,0.5,0.5]", "sigmoid", 3), 3 * math.log(2)
        )
        self.assertAlmostEqual(compute_class_entropy([0.5], "sigmoid", 1), math.log(2))
        self.assertEqual(compute_class_entropy([0, 1, 0], "sigmoid", 3), 0)

    def test_background_is_retained(self):
        self.assertAlmostEqual(
            compute_class_entropy([0.5, 0.5], "softmax_background", 1), math.log(2)
        )

    def test_bad_vectors_fail(self):
        for values, kind, nc in [
            ([0.8], "sigmoid", 3),
            ([float("nan")], "sigmoid", 1),
            ([1.2], "sigmoid", 1),
            ([0.5, 0.2], "softmax_background", 1),
            ([0.5], "unknown", 1),
        ]:
            with self.subTest(values=values, kind=kind), self.assertRaises(ValueError):
                compute_class_entropy(values, kind, nc)

    def test_legacy_cache_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerow(["confidence"])
            with self.assertRaisesRegex(ValueError, "--force"):
                require_probability_csv(path)
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerow(["class_probabilities", "probability_kind"])
            require_probability_csv(path)


class DetectorAlignmentTests(unittest.TestCase):
    def setup_predictor(self, cls):
        predictor = object.__new__(cls)
        predictor.args = SimpleNamespace(
            conf=0.25,
            iou=0.5,
            classes=None,
            agnostic_nms=False,
            max_det=10,
            task="detect",
        )
        predictor.model = SimpleNamespace(names={0: "a", 1: "b", 2: "c"}, end2end=False)
        predictor.batch = (["first.jpg", "empty.jpg"], None, None)
        return predictor

    def test_yolo_vectors_follow_nms_and_batch(self):
        import numpy as np
        import torch
        from ultralytics.models.yolo.detect.predict import DetectionPredictor

        custom = self.setup_predictor(ultralytics_probability_predictor("yolo"))
        baseline = self.setup_predictor(DetectionPredictor)
        # Candidates 0/1 overlap; keep 1. Candidate 2 survives for another class.
        first = torch.tensor(
            [
                [16, 16, 10, 10, 0.6, 0.1, 0.2],
                [16, 16, 10, 10, 0.9, 0.05, 0.1],
                [45, 45, 8, 8, 0.1, 0.8, 0.2],
            ]
        ).T
        raw = torch.stack((first, first.clone()))
        raw[1, 4:] = 0
        originals = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        img = torch.zeros(2, 3, 64, 64)
        expected = baseline.postprocess(raw.clone(), img, originals)
        actual = custom.postprocess(raw.clone(), img, originals)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a.boxes.data, e.boxes.data)
        torch.testing.assert_close(actual[0].class_probabilities, first[4:, [1, 2]].T)
        self.assertEqual(tuple(actual[1].class_probabilities.shape), (0, 3))

    def test_rtdetr_vectors_follow_filter(self):
        import numpy as np
        import torch
        from ultralytics.models.rtdetr.predict import RTDETRPredictor

        custom = self.setup_predictor(ultralytics_probability_predictor("rtdetr"))
        baseline = self.setup_predictor(RTDETRPredictor)
        custom.args.classes = baseline.args.classes = [1]
        first = torch.tensor(
            [[0.2, 0.2, 0.1, 0.1, 0.1, 0.7, 0.2], [0.6, 0.6, 0.2, 0.2, 0.1, 0.8, 0.3]]
        )
        raw = torch.stack((first, first.clone()))
        raw[1, :, 4:] = 0
        originals = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        img = torch.zeros(2, 3, 64, 64)
        expected = baseline.postprocess(raw.clone(), img, originals)
        actual = custom.postprocess(raw.clone(), img, originals)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a.boxes.data, e.boxes.data)
        torch.testing.assert_close(actual[0].class_probabilities, first[[1, 0], 4:])
        self.assertEqual(tuple(actual[1].class_probabilities.shape), (0, 3))

    def test_unsupported_yolo_head_fails(self):
        import torch

        custom = self.setup_predictor(ultralytics_probability_predictor("yolo"))
        custom.model.end2end = True
        with self.assertRaisesRegex(ValueError, "Full-score"):
            custom.postprocess(torch.zeros(1, 10, 6), None, None)

    def test_faster_rcnn_exact_postprocess_and_forward(self):
        import torch
        from torch import nn
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator

        torch.manual_seed(12)
        backbone = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        backbone.out_channels = 8
        model = FasterRCNN(
            backbone,
            num_classes=4,
            min_size=32,
            max_size=32,
            rpn_anchor_generator=AnchorGenerator(((8, 16),), ((1.0,),)),
            rpn_pre_nms_top_n_test=20,
            rpn_post_nms_top_n_test=10,
            box_detections_per_img=10,
        ).eval()
        proposals = [
            torch.tensor(
                [
                    [2.0, 2.0, 10.0, 10.0],
                    [2.0, 2.0, 10.0, 10.0],
                    [15.0, 15.0, 25.0, 25.0],
                ]
            ),
            torch.empty(0, 4),
        ]
        logits = torch.tensor(
            [[0.0, 4.0, 1.0, 2.0], [0.0, 3.0, 1.0, 2.0], [0.0, 1.0, 4.0, 2.0]]
        )
        regression = torch.zeros(3, 16)
        shapes = [(32, 32), (32, 32)]
        expected = model.roi_heads.postprocess_detections(
            logits, regression, proposals, shapes
        )
        enable_faster_rcnn_probabilities(model)
        actual = model.roi_heads.postprocess_detections(
            logits, regression, proposals, shapes
        )
        for group_a, group_e in zip(actual, expected):
            for a, e in zip(group_a, group_e):
                torch.testing.assert_close(a, e)
        for scores, labels, probs in zip(
            actual[1], actual[2], model.roi_heads._retained_probabilities
        ):
            torch.testing.assert_close(probs.sum(1), torch.ones(len(probs)))
            torch.testing.assert_close(probs[torch.arange(len(probs)), labels], scores)
            for row in probs:
                self.assertTrue(any(torch.allclose(row, p) for p in logits.softmax(-1)))
        with torch.no_grad():
            results = model([torch.rand(3, 32, 32)])
        self.assertEqual(
            len(results[0]["class_probabilities"]), len(results[0]["boxes"])
        )
        self.assertEqual(results[0]["class_probabilities"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
