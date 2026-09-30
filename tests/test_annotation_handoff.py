import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from detection.active_learning.pipelines.annotation_handoff import (
    AnnotationBatchError,
    export_annotation_batch,
    validate_annotation_batch,
)
from detection.active_learning.pipelines import gather_annotations
from detection.active_learning.pipelines.ingest_annotations import (
    ingest_annotation_batch,
)


CLASSES = ["Other_Amphibian", "Small_Mammal", "Western_Leopard_Toad"]


class AnnotationHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _image(self, name: str, value: int = 80) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((8, 10, 3), value, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def _export(self, sources):
        output = self.root / "to_annotate" / "yolo_cycle_0"
        manifest = export_annotation_batch(
            [{"image_path": str(path), "image_name": path.name} for path in sources],
            output,
            model_type="yolo",
            cycle=0,
            classes=CLASSES,
            candidates_csv=self.root / "candidates.csv",
        )
        return output, manifest

    def _annotated_batch(self, output: Path, manifest, labels=None) -> Path:
        batch = self.root / "annotated" / "yolo_cycle_0"
        images = batch / "images"
        label_dir = batch / "labels"
        images.mkdir(parents=True)
        label_dir.mkdir()
        shutil.copy2(output / "manifest.json", batch / "manifest.json")
        shutil.copy2(output / "classes.txt", batch / "classes.txt")
        labels = labels or {}
        for record in manifest.images:
            shutil.copy2(output / record.file_name, images / record.file_name)
            (label_dir / f"{Path(record.file_name).stem}.txt").write_text(
                labels.get(record.file_name, "0 0.5 0.5 0.25 0.25\n"),
                encoding="utf-8",
            )
        return batch

    def _dataset_root(self) -> Path:
        active = self.root / "active_learning"
        dataset = active / "data" / "yolo_clahe" / "pretrained" / "cycle_0" / "train"
        (dataset / "images").mkdir(parents=True)
        (dataset / "labels").mkdir()
        (dataset / "classes.txt").write_text(
            "\n".join(CLASSES) + "\n", encoding="utf-8"
        )
        (dataset / "images" / "base.jpg").write_bytes(b"original")
        (dataset / "labels" / "base.txt").write_text("", encoding="utf-8")
        state_dir = active / "pipelines"
        state_dir.mkdir()
        (state_dir / "al_state_yolo_clahe_pretrained.json").write_text(
            json.dumps({"cycle": 1}), encoding="utf-8"
        )
        return active

    def test_valid_export_and_validation_preserves_sources(self):
        source = self._image("source.jpg")
        original = source.read_bytes()

        output, manifest = self._export([source])
        batch = self._annotated_batch(output, manifest)

        validated = validate_annotation_batch(
            batch,
            expected_model_type="yolo",
            expected_cycle=0,
            expected_classes=CLASSES,
            expected_state_cycle=1,
        )

        self.assertEqual(validated.batch_id, "yolo_cycle_0")
        self.assertEqual(manifest.images[0].width, 10)
        self.assertEqual(source.read_bytes(), original)
        self.assertTrue((output / "manifest.json").is_file())
        self.assertEqual(
            json.loads((output / "manifest.json").read_text())["format"],
            "amphilens-active-learning-yolo-v1",
        )

    def test_repeated_export_reuses_an_identical_batch(self):
        source = self._image("repeat.jpg")
        output, first = self._export([source])

        second = export_annotation_batch(
            [{"image_path": str(source), "image_name": source.name}],
            output,
            model_type="yolo",
            cycle=0,
            classes=CLASSES,
            candidates_csv=self.root / "candidates.csv",
        )

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_gather_cli_writes_manifest_and_preserves_tracker_columns(self):
        source = self._image("cli.jpg")
        candidates = self.root / "candidates.csv"
        candidates.write_text(
            f"image_path,image_name\n{source},{source.name}\n", encoding="utf-8"
        )
        active = self.root / "active_learning"
        original_root = gather_annotations.ACTIVE_LEARNING_DIR
        original_argv = sys.argv
        gather_annotations.ACTIVE_LEARNING_DIR = str(active)
        sys.argv = [
            "gather_annotations.py",
            "--candidates_csv",
            str(candidates),
            "--cycle",
            "0",
            "--model_type",
            "yolo",
        ]
        try:
            gather_annotations.main()
        finally:
            gather_annotations.ACTIVE_LEARNING_DIR = original_root
            sys.argv = original_argv

        output = active / "to_annotate" / "yolo_cycle_0"
        self.assertTrue((output / "manifest.json").is_file())
        self.assertTrue((output / "classes.txt").is_file())
        self.assertEqual(
            (active / "already_sampled.csv")
            .read_text(encoding="utf-8")
            .splitlines()[0],
            "image_path,cycle,model_type",
        )

    def test_collision_names_use_stable_hash_suffix(self):
        first = self._image("one" + "".join(["/", "frame.jpg"]))
        second_dir = self.root / "two"
        second_dir.mkdir()
        second = self._image("frame.jpg", value=90)
        moved_second = second_dir / second.name
        second.rename(moved_second)

        output, manifest = self._export([first, moved_second])

        names = [record.file_name for record in manifest.images]
        self.assertEqual(len(names), 2)
        self.assertNotEqual(names[0], names[1])
        self.assertTrue(any("__" in name for name in names))
        self.assertTrue(all((output / name).is_file() for name in names))

    def test_missing_image_is_reported(self):
        source = self._image("missing.jpg")
        output, manifest = self._export([source])
        batch = self._annotated_batch(output, manifest)
        (batch / "images" / manifest.images[0].file_name).unlink()

        with self.assertRaises(AnnotationBatchError) as raised:
            validate_annotation_batch(
                batch,
                expected_model_type="yolo",
                expected_cycle=0,
                expected_classes=CLASSES,
                expected_state_cycle=1,
            )

        self.assertIn(
            "missing_image", {issue.code for issue in raised.exception.issues}
        )

    def test_unknown_class_and_malformed_yolo_are_reported(self):
        source = self._image("bad.jpg")
        output, manifest = self._export([source])
        batch = self._annotated_batch(
            output,
            manifest,
            labels={manifest.images[0].file_name: "9 0.5 0.5\n"},
        )

        with self.assertRaises(AnnotationBatchError) as raised:
            validate_annotation_batch(
                batch,
                expected_model_type="yolo",
                expected_cycle=0,
                expected_classes=CLASSES,
                expected_state_cycle=1,
            )

        codes = {issue.code for issue in raised.exception.issues}
        self.assertIn("malformed_label", codes)

        (
            batch / "labels" / f"{Path(manifest.images[0].file_name).stem}.txt"
        ).write_text("9 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        with self.assertRaises(AnnotationBatchError) as raised:
            validate_annotation_batch(
                batch,
                expected_model_type="yolo",
                expected_cycle=0,
                expected_classes=CLASSES,
                expected_state_cycle=1,
            )
        self.assertIn(
            "unknown_class", {issue.code for issue in raised.exception.issues}
        )

    def test_duplicate_content_is_rejected(self):
        first = self._image("first.jpg", value=100)
        second = self.root / "second.jpg"
        shutil.copy2(first, second)

        with self.assertRaises(AnnotationBatchError) as raised:
            self._export([first, second])

        self.assertIn(
            "duplicate_content", {issue.code for issue in raised.exception.issues}
        )

    def test_validator_rejects_duplicate_content_in_manifest(self):
        first = self._image("first-valid.jpg", value=100)
        second = self._image("second-valid.jpg", value=120)
        output, manifest = self._export([first, second])
        batch = self._annotated_batch(output, manifest)
        manifest_path = batch / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["images"][1]["sha256"] = payload["images"][0]["sha256"]
        shutil.copy2(
            batch / "images" / manifest.images[0].file_name,
            batch / "images" / manifest.images[1].file_name,
        )
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaises(AnnotationBatchError) as raised:
            validate_annotation_batch(
                batch,
                expected_model_type="yolo",
                expected_cycle=0,
                expected_classes=CLASSES,
                expected_state_cycle=1,
            )

        self.assertIn(
            "duplicate_content", {issue.code for issue in raised.exception.issues}
        )

    def test_cycle_mismatch_is_rejected(self):
        source = self._image("cycle.jpg")
        output, manifest = self._export([source])
        batch = self._annotated_batch(output, manifest)
        manifest_path = batch / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["cycle"] = 1
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaises(AnnotationBatchError) as raised:
            validate_annotation_batch(
                batch,
                expected_model_type="yolo",
                expected_cycle=0,
                expected_classes=CLASSES,
                expected_state_cycle=1,
            )

        self.assertIn(
            "cycle_mismatch", {issue.code for issue in raised.exception.issues}
        )

    def test_missing_and_extra_labels_are_rejected(self):
        first = self._image("first-label.jpg")
        second = self._image("second-label.jpg", value=120)
        output, manifest = self._export([first, second])
        batch = self._annotated_batch(output, manifest)
        first_label = (
            batch / "labels" / f"{Path(manifest.images[0].file_name).stem}.txt"
        )
        first_label.unlink()
        (batch / "labels" / "untracked.txt").write_text("", encoding="utf-8")

        with self.assertRaises(AnnotationBatchError) as raised:
            validate_annotation_batch(
                batch,
                expected_model_type="yolo",
                expected_cycle=0,
                expected_classes=CLASSES,
                expected_state_cycle=1,
            )

        codes = {issue.code for issue in raised.exception.issues}
        self.assertIn("missing_label", codes)
        self.assertIn("unexpected_label", codes)

    def test_valid_batch_is_ingested_after_validation(self):
        active = self._dataset_root()
        source = self._image("ingest.jpg")
        output, manifest = self._export([source])
        batch = self._annotated_batch(output, manifest)

        destination = ingest_annotation_batch(batch, active_learning_dir=active)

        self.assertTrue(
            (destination / "train" / "images" / manifest.images[0].file_name).is_file()
        )
        self.assertTrue((destination / "train" / "labels" / "ingest.txt").is_file())
        self.assertTrue(
            (active / "data" / "yolo_clahe" / "pretrained" / "cycle_0").is_dir()
        )

    def test_invalid_batch_does_not_create_next_dataset(self):
        active = self._dataset_root()
        source = self._image("invalid-ingest.jpg")
        output, manifest = self._export([source])
        batch = self._annotated_batch(output, manifest)
        (batch / "images" / manifest.images[0].file_name).unlink()
        destination = active / "data" / "yolo_clahe" / "pretrained" / "cycle_1"

        with self.assertRaises(AnnotationBatchError):
            ingest_annotation_batch(batch, active_learning_dir=active)

        self.assertFalse(destination.exists())

    def test_missing_source_does_not_publish_partial_export(self):
        existing = self._image("existing.jpg")
        missing = self.root / "missing.jpg"
        output = self.root / "to_annotate" / "yolo_cycle_0"

        with self.assertRaises(AnnotationBatchError) as raised:
            export_annotation_batch(
                [
                    {"image_path": str(existing), "image_name": existing.name},
                    {"image_path": str(missing), "image_name": missing.name},
                ],
                output,
                model_type="yolo",
                cycle=0,
                classes=CLASSES,
            )

        self.assertIn(
            "missing_source", {issue.code for issue in raised.exception.issues}
        )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
