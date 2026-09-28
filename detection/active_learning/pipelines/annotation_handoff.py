"""Deterministic, manifest-backed exchange for active-learning annotations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_FORMAT = "amphilens-active-learning-yolo-v1"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
CLASSES_FILENAME = "classes.txt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A stable, machine-readable validation failure."""

    code: str
    path: str
    message: str

    def __str__(self) -> str:
        location = f" ({self.path})" if self.path else ""
        return f"[{self.code}] {self.message}{location}"


class AnnotationBatchError(ValueError):
    """Raised when an export or annotation batch violates the handoff contract."""

    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("AnnotationBatchError requires at least one issue")
        super().__init__("\n".join(str(issue) for issue in self.issues))


@dataclass(frozen=True, slots=True)
class AnnotationImage:
    """Manifest metadata for one exported image."""

    file_name: str
    source_path: str
    sha256: str
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str) -> AnnotationImage:
        try:
            return cls(
                file_name=str(value["file_name"]),
                source_path=str(value["source_path"]),
                sha256=str(value["sha256"]),
                width=int(value["width"]),
                height=int(value["height"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AnnotationBatchError(
                [ValidationIssue("manifest_invalid", path, "Image record is malformed")]
            ) from exc


@dataclass(frozen=True, slots=True)
class AnnotationBatchManifest:
    """Portable metadata describing one active-learning annotation batch."""

    batch_id: str
    model_type: str
    cycle: int
    next_cycle: int
    classes: tuple[str, ...]
    source_provenance: dict[str, Any]
    images: tuple[AnnotationImage, ...]
    format: str = MANIFEST_FORMAT
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "model_type": self.model_type,
            "cycle": self.cycle,
            "next_cycle": self.next_cycle,
            "classes": list(self.classes),
            "source_provenance": dict(self.source_provenance),
            "images": [image.to_dict() for image in self.images],
        }


def _issue(code: str, path: str | Path, message: str) -> ValidationIssue:
    return ValidationIssue(code, str(path), message)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without changing it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - existing pipeline dependency
        raise AnnotationBatchError(
            [_issue("image_unreadable", path, "OpenCV is required to inspect images")]
        ) from exc
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise AnnotationBatchError(
            [_issue("image_unreadable", path, "Image cannot be decoded")]
        )
    return int(image.shape[1]), int(image.shape[0])


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _safe_manifest(value: Mapping[str, Any], manifest_path: Path) -> AnnotationBatchManifest:
    issues: list[ValidationIssue] = []
    if value.get("format") != MANIFEST_FORMAT:
        issues.append(_issue("manifest_invalid", manifest_path, "Unsupported manifest format"))
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        issues.append(
            _issue(
                "manifest_invalid",
                manifest_path,
                f"Unsupported manifest schema version: {value.get('schema_version')!r}",
            )
        )
    classes_value = value.get("classes")
    if not isinstance(classes_value, list) or any(
        not isinstance(item, str) or not item.strip() for item in classes_value
    ):
        issues.append(_issue("manifest_invalid", manifest_path, "Manifest classes must be non-empty strings"))
        classes: tuple[str, ...] = ()
    else:
        classes = tuple(item.strip() for item in classes_value)
        if len(classes) != len(set(classes)):
            issues.append(_issue("manifest_invalid", manifest_path, "Manifest classes must be unique"))

    images_value = value.get("images")
    if not isinstance(images_value, list):
        issues.append(_issue("manifest_invalid", manifest_path, "Manifest images must be a list"))
        images: tuple[AnnotationImage, ...] = ()
    else:
        parsed_images: list[AnnotationImage] = []
        for index, item in enumerate(images_value):
            if not isinstance(item, Mapping):
                issues.append(
                    _issue("manifest_invalid", f"{manifest_path}:images[{index}]", "Image record must be an object")
                )
                continue
            try:
                parsed_images.append(AnnotationImage.from_dict(item, f"{manifest_path}:images[{index}]"))
            except AnnotationBatchError as exc:
                issues.extend(exc.issues)
        images = tuple(parsed_images)

    source_provenance = value.get("source_provenance", {})
    if not isinstance(source_provenance, dict):
        issues.append(_issue("manifest_invalid", manifest_path, "source_provenance must be an object"))
        source_provenance = {}

    try:
        batch_id = str(value["batch_id"])
        model_type = str(value["model_type"])
        cycle = int(value["cycle"])
        next_cycle = int(value["next_cycle"])
    except (KeyError, TypeError, ValueError):
        issues.append(_issue("manifest_invalid", manifest_path, "Manifest identity fields are malformed"))
        batch_id = ""
        model_type = ""
        cycle = -1
        next_cycle = -1

    if issues:
        raise AnnotationBatchError(issues)
    return AnnotationBatchManifest(
        batch_id=batch_id,
        model_type=model_type,
        cycle=cycle,
        next_cycle=next_cycle,
        classes=classes,
        source_provenance=source_provenance,
        images=images,
        format=str(value["format"]),
        schema_version=int(value["schema_version"]),
    )


def load_manifest(batch_dir_or_manifest: str | Path) -> AnnotationBatchManifest:
    path = Path(batch_dir_or_manifest).expanduser().resolve()
    manifest_path = path / MANIFEST_FILENAME if path.is_dir() else path
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnnotationBatchError(
            [_issue("manifest_missing", manifest_path, "manifest.json is required")]
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationBatchError(
            [_issue("manifest_invalid", manifest_path, "manifest.json is not valid JSON")]
        ) from exc
    if not isinstance(value, dict):
        raise AnnotationBatchError(
            [_issue("manifest_invalid", manifest_path, "manifest.json must contain an object")]
        )
    return _safe_manifest(value, manifest_path)


def _stable_path_suffix(source: Path) -> str:
    return hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]


def _deterministic_names(sources: Sequence[Path]) -> dict[Path, str]:
    groups: dict[str, list[Path]] = {}
    for source in sources:
        groups.setdefault(source.name, []).append(source)
    names: dict[Path, str] = {}
    for basename, grouped in groups.items():
        if len(grouped) == 1:
            names[grouped[0]] = basename
            continue
        path = Path(basename)
        for source in sorted(grouped, key=str):
            names[source] = f"{path.stem}__{_stable_path_suffix(source)}{path.suffix}"
    return names


def export_annotation_batch(
    candidates: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    model_type: str,
    cycle: int,
    classes: Iterable[str],
    candidates_csv: str | Path | None = None,
) -> AnnotationBatchManifest:
    """Stage and atomically publish a deterministic annotation batch."""

    destination = Path(output_dir).expanduser().resolve()
    clean_classes = tuple(str(value).strip() for value in classes)
    issues: list[ValidationIssue] = []
    if not clean_classes or any(not value for value in clean_classes):
        issues.append(_issue("class_schema_mismatch", destination, "Classes must be non-empty"))
    if len(clean_classes) != len(set(clean_classes)):
        issues.append(_issue("class_schema_mismatch", destination, "Classes must be unique"))
    if cycle < 0:
        issues.append(_issue("cycle_mismatch", destination, "Cycle must be non-negative"))

    source_paths: list[Path] = []
    seen_sources: set[Path] = set()
    for index, candidate in enumerate(candidates):
        value = candidate.get("image_path")
        if not isinstance(value, str) or not value.strip():
            issues.append(_issue("invalid_candidate", f"candidate[{index}]", "image_path is required"))
            continue
        source = Path(value).expanduser().resolve()
        if source in seen_sources:
            issues.append(_issue("duplicate_source", source, "Candidate source is repeated"))
        seen_sources.add(source)
        source_paths.append(source)

    missing = [path for path in source_paths if not path.is_file()]
    issues.extend(_issue("missing_source", path, "Source image does not exist") for path in missing)
    source_paths = [path for path in source_paths if path.is_file()]
    if not source_paths:
        issues.append(_issue("empty_batch", destination, "No source images were exported"))

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        issues.append(_issue("output_error", destination, f"Cannot create output parent: {exc}"))

    for source in source_paths:
        try:
            if destination == source or source.is_relative_to(destination):
                issues.append(_issue("source_collision", source, "Output directory overlaps a source image"))
        except ValueError:
            pass

    if issues:
        raise AnnotationBatchError(issues)

    names = _deterministic_names(source_paths)
    images: list[AnnotationImage] = []
    source_hashes: dict[str, Path] = {}
    try:
        for source in sorted(source_paths, key=str):
            digest = sha256_file(source)
            if digest in source_hashes:
                issues.append(
                    _issue(
                        "duplicate_content",
                        source,
                        f"Image content duplicates {source_hashes[digest]}",
                    )
                )
            source_hashes[digest] = source
            width, height = _image_dimensions(source)
            images.append(
                AnnotationImage(
                    file_name=names[source],
                    source_path=str(source),
                    sha256=digest,
                    width=width,
                    height=height,
                )
            )
    except AnnotationBatchError:
        raise
    except (OSError, ValueError) as exc:
        raise AnnotationBatchError([_issue("source_unreadable", source_paths[0], str(exc))]) from exc
    if issues:
        raise AnnotationBatchError(issues)

    manifest = AnnotationBatchManifest(
        batch_id=f"{model_type}_cycle_{cycle}",
        model_type=str(model_type),
        cycle=int(cycle),
        next_cycle=int(cycle) + 1,
        classes=clean_classes,
        source_provenance={
            "candidates_csv": str(Path(candidates_csv).expanduser().resolve())
            if candidates_csv is not None
            else None,
            "requested_count": len(source_paths),
            "exported_count": len(images),
        },
        images=tuple(images),
    )

    if destination.exists():
        if not destination.is_dir() or not (destination / MANIFEST_FILENAME).is_file():
            raise AnnotationBatchError(
                [
                    _issue(
                        "existing_output",
                        destination,
                        "Batch output already exists without a reusable manifest",
                    )
                ]
            )
        try:
            existing = load_manifest(destination)
        except AnnotationBatchError as exc:
            raise AnnotationBatchError(
                [
                    _issue(
                        "existing_output",
                        destination,
                        "Existing batch manifest is invalid and cannot be overwritten",
                    ),
                    *exc.issues,
                ]
            ) from exc
        if existing.to_dict() != manifest.to_dict():
            raise AnnotationBatchError(
                [
                    _issue(
                        "existing_output",
                        destination,
                        "Existing batch does not match the requested export",
                    )
                ]
            )
        existing_issues: list[ValidationIssue] = []
        for image in existing.images:
            exported = destination / image.file_name
            if not exported.is_file():
                existing_issues.append(_issue("existing_output", exported, "Existing export image is missing"))
                continue
            if sha256_file(exported) != image.sha256:
                existing_issues.append(_issue("existing_output", exported, "Existing export image has changed"))
        if existing_issues:
            raise AnnotationBatchError(existing_issues)
        return existing

    temporary: Path | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.", dir=destination.parent
        ) as staging:
            temporary = Path(staging)
            for image in manifest.images:
                shutil.copy2(image.source_path, temporary / image.file_name)
            (temporary / CLASSES_FILENAME).write_text(
                "\n".join(manifest.classes) + "\n", encoding="utf-8"
            )
            _atomic_write_json(temporary / MANIFEST_FILENAME, manifest.to_dict())
            os.replace(temporary, destination)
            temporary = None
    except FileExistsError as exc:
        raise AnnotationBatchError(
            [_issue("existing_output", destination, "Batch output was created concurrently")]
        ) from exc
    except OSError as exc:
        raise AnnotationBatchError([_issue("output_error", destination, str(exc))]) from exc
    return manifest


def _label_values(line: str, label_path: Path, line_number: int) -> tuple[int, list[float]]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Expected five fields at line {line_number}")
    try:
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"Non-numeric YOLO value at line {line_number}") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite YOLO value at line {line_number}")
    return class_id, values


def validate_annotation_batch(
    batch_dir: str | Path,
    *,
    expected_model_type: str,
    expected_cycle: int,
    expected_classes: Iterable[str],
    expected_state_cycle: int | None = None,
) -> AnnotationBatchManifest:
    """Validate a completed ``images``/``labels`` annotation batch."""

    batch = Path(batch_dir).expanduser().resolve()
    manifest = load_manifest(batch)
    issues: list[ValidationIssue] = []
    classes = tuple(str(value).strip() for value in expected_classes)
    manifest_path = batch / MANIFEST_FILENAME

    if manifest.model_type != expected_model_type:
        issues.append(_issue("model_mismatch", manifest_path, "Manifest model_type does not match the batch directory"))
    if manifest.batch_id != f"{expected_model_type}_cycle_{expected_cycle}":
        issues.append(_issue("cycle_mismatch", manifest_path, "Manifest batch_id does not match the batch directory"))
    if manifest.cycle != expected_cycle or manifest.next_cycle != expected_cycle + 1:
        issues.append(_issue("cycle_mismatch", manifest_path, "Manifest cycle and next_cycle do not match ingestion"))
    if expected_state_cycle is not None and expected_state_cycle != manifest.next_cycle:
        issues.append(_issue("cycle_mismatch", manifest_path, "State file cycle does not match manifest next_cycle"))
    if manifest.classes != classes:
        issues.append(_issue("class_schema_mismatch", manifest_path, "Manifest classes do not match the current dataset"))

    provenance = manifest.source_provenance
    if provenance.get("requested_count") != provenance.get("exported_count"):
        issues.append(_issue("missing_source", manifest_path, "Manifest records an incomplete export"))
    missing_sources = provenance.get("missing_sources", [])
    if isinstance(missing_sources, list) and missing_sources:
        issues.append(_issue("missing_source", manifest_path, "Manifest records missing source images"))

    manifest_names: list[str] = []
    manifest_hashes: set[str] = set()
    manifest_sources: set[str] = set()
    label_names: set[str] = set()
    for image in manifest.images:
        image_name = Path(image.file_name)
        if image_name.name != image.file_name or image_name.is_absolute() or ".." in image_name.parts:
            issues.append(_issue("manifest_invalid", manifest_path, f"Unsafe image filename: {image.file_name!r}"))
            continue
        if image.file_name in manifest_names:
            issues.append(_issue("duplicate_name", manifest_path, f"Duplicate image filename: {image.file_name}"))
        manifest_names.append(image.file_name)
        if image.sha256 in manifest_hashes:
            issues.append(_issue("duplicate_content", manifest_path, f"Duplicate image hash: {image.file_name}"))
        manifest_hashes.add(image.sha256)
        if image.source_path in manifest_sources:
            issues.append(_issue("duplicate_source", manifest_path, f"Duplicate source path: {image.source_path}"))
        manifest_sources.add(image.source_path)
        if not _SHA256_RE.fullmatch(image.sha256):
            issues.append(_issue("manifest_invalid", manifest_path, f"Invalid image hash: {image.file_name}"))
        if image.width <= 0 or image.height <= 0:
            issues.append(_issue("manifest_invalid", manifest_path, f"Invalid image dimensions: {image.file_name}"))
        label_name = f"{Path(image.file_name).stem}.txt"
        if label_name in label_names:
            issues.append(_issue("duplicate_name", manifest_path, f"Duplicate derived label filename: {label_name}"))
        label_names.add(label_name)

    images_dir = batch / "images"
    labels_dir = batch / "labels"
    actual_images = {
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    } if images_dir.is_dir() else set()
    actual_labels = {
        path.name for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
    } if labels_dir.is_dir() else set()

    for file_name in sorted(set(manifest_names) - actual_images):
        issues.append(_issue("missing_image", images_dir / file_name, "Manifest image is missing"))
    for file_name in sorted(actual_images - set(manifest_names)):
        issues.append(_issue("unexpected_image", images_dir / file_name, "Image is not listed in manifest"))
    for file_name in sorted(label_names - actual_labels):
        issues.append(_issue("missing_label", labels_dir / file_name, "Label file is missing"))
    for file_name in sorted(actual_labels - label_names):
        issues.append(_issue("unexpected_label", labels_dir / file_name, "Label file is not associated with a manifest image"))

    actual_hashes: dict[str, str] = {}
    for image in manifest.images:
        image_path = images_dir / image.file_name
        if not image_path.is_file():
            continue
        try:
            actual_hash = sha256_file(image_path)
            actual_hashes[image.file_name] = actual_hash
            if actual_hash != image.sha256:
                issues.append(_issue("image_hash_mismatch", image_path, "Image bytes differ from export manifest"))
            if list(actual_hashes.values()).count(actual_hash) > 1:
                issues.append(_issue("duplicate_content", image_path, "Image content is duplicated in the batch"))
            dimensions = _image_dimensions(image_path)
            if dimensions != (image.width, image.height):
                issues.append(_issue("image_dimensions_mismatch", image_path, "Image dimensions differ from manifest"))
        except AnnotationBatchError as exc:
            issues.extend(exc.issues)

    for image in manifest.images:
        label_path = labels_dir / f"{Path(image.file_name).stem}.txt"
        if not label_path.is_file():
            continue
        try:
            label_lines = label_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            issues.append(_issue("malformed_label", label_path, f"Label file cannot be read: {exc}"))
            continue
        for line_number, raw_line in enumerate(label_lines, start=1):
            if not raw_line.strip():
                continue
            try:
                class_id, values = _label_values(raw_line, label_path, line_number)
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                issues.append(_issue("malformed_label", f"{label_path}:{line_number}", str(exc)))
                continue
            if class_id < 0 or class_id >= len(classes):
                issues.append(_issue("unknown_class", f"{label_path}:{line_number}", f"Unknown YOLO class id: {class_id}"))
            center_x, center_y, width, height = values
            if width <= 0 or height <= 0 or any(value < 0 or value > 1 for value in values):
                issues.append(_issue("malformed_label", f"{label_path}:{line_number}", "YOLO coordinates must be normalized and positive"))
            if (
                center_x - width / 2 < 0
                or center_y - height / 2 < 0
                or center_x + width / 2 > 1
                or center_y + height / 2 > 1
            ):
                issues.append(_issue("malformed_label", f"{label_path}:{line_number}", "YOLO bounding box exceeds image bounds"))

    if issues:
        raise AnnotationBatchError(issues)
    return manifest
