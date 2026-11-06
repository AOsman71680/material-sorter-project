"""Validate the raw image dataset (one sub-folder per material class).

Validation is read-only: images are never modified, moved, or deleted.
Run ``python -m src.dataset`` to print a report for the configured dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from src.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

# Reporting heuristics (warnings only, never errors).
DEFAULT_MAX_IMBALANCE_RATIO = 2.0
DEFAULT_MIN_RECOMMENDED_IMAGES = 50


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_image_files(directory: Path) -> list[Path]:
    """Image files directly inside ``directory`` (not recursive), sorted by name."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if is_image_file(p))


def count_images(directory: Path) -> int:
    """Cheap image count by extension only (no decoding)."""
    return len(list_image_files(directory))


@dataclass(frozen=True)
class ImageIssue:
    path: Path
    reason: str


@dataclass
class ClassReport:
    name: str
    directory: Path
    exists: bool
    valid_count: int = 0
    valid_paths: list[Path] = field(default_factory=list)
    invalid: list[ImageIssue] = field(default_factory=list)
    ignored_files: list[Path] = field(default_factory=list)
    nested_dirs: list[Path] = field(default_factory=list)

    @property
    def image_file_count(self) -> int:
        return self.valid_count + len(self.invalid)


@dataclass
class DatasetReport:
    dataset_dir: Path
    dataset_dir_exists: bool
    classes: dict[str, ClassReport]
    unexpected_entries: list[Path]
    duplicate_groups: list[list[Path]]
    imbalance_ratio: float | None
    errors: list[str]
    warnings: list[str]
    max_imbalance_ratio: float
    min_recommended_images: int

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def total_valid_images(self) -> int:
        return sum(c.valid_count for c in self.classes.values())


@dataclass(frozen=True)
class _Decoded:
    digest: str | None  # sha256 of file bytes when decodable
    problem: str | None  # reason the image is unusable


def _check_image(path: Path) -> _Decoded:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return _Decoded(None, f"cannot read file ({exc.strerror or exc})")
    if not data:
        return _Decoded(None, "file is empty (0 bytes)")
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    if image is None or image.size == 0:
        return _Decoded(None, "cannot be decoded as an image (corrupted or not an image)")
    return _Decoded(hashlib.sha256(data).hexdigest(), None)


def validate_dataset(
    dataset_dir: Path,
    classes: tuple[str, ...],
    *,
    max_imbalance_ratio: float = DEFAULT_MAX_IMBALANCE_RATIO,
    min_recommended_images: int = DEFAULT_MIN_RECOMMENDED_IMAGES,
) -> DatasetReport:
    """Inspect ``dataset_dir`` and return a report. Never changes any file."""
    if max_imbalance_ratio < 1.0:
        raise ValueError(f"max_imbalance_ratio must be >= 1, got {max_imbalance_ratio}")
    if min_recommended_images < 0:
        raise ValueError(f"min_recommended_images must be >= 0, got {min_recommended_images}")

    errors: list[str] = []
    warnings: list[str] = []
    class_reports: dict[str, ClassReport] = {}
    unexpected: list[Path] = []
    by_digest: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    dataset_exists = dataset_dir.is_dir()
    if not dataset_exists:
        errors.append(f"dataset directory does not exist: {dataset_dir}")

    for name in classes:
        class_dir = dataset_dir / name
        report = ClassReport(name=name, directory=class_dir, exists=class_dir.is_dir())
        class_reports[name] = report
        if not report.exists:
            if dataset_exists:
                errors.append(f"class directory missing: {name}/")
            continue

        for entry in sorted(class_dir.iterdir()):
            if entry.is_dir():
                report.nested_dirs.append(entry)
            elif entry.suffix.lower() in IMAGE_EXTENSIONS:
                result = _check_image(entry)
                if result.problem is not None:
                    report.invalid.append(ImageIssue(entry, result.problem))
                else:
                    report.valid_count += 1
                    report.valid_paths.append(entry)
                    assert result.digest is not None
                    by_digest[result.digest].append((name, entry))
            else:
                report.ignored_files.append(entry)

        if report.valid_count == 0:
            errors.append(f"class '{name}' has no valid images")
        for issue in report.invalid:
            errors.append(f"invalid image {name}/{issue.path.name}: {issue.reason}")
        if report.nested_dirs:
            warnings.append(
                f"class '{name}' contains sub-folder(s) that are not scanned: "
                + _name_list(report.nested_dirs)
            )
        if report.ignored_files:
            warnings.append(
                f"class '{name}' contains non-image file(s) that are ignored: "
                + _name_list(report.ignored_files)
            )

    if dataset_exists:
        for entry in sorted(dataset_dir.iterdir()):
            if entry.name not in classes:
                unexpected.append(entry)
        if unexpected:
            warnings.append(
                "entries in dataset directory that are not configured classes (ignored): "
                + _name_list(unexpected)
            )

    duplicate_groups: list[list[Path]] = []
    for entries in by_digest.values():
        if len(entries) < 2:
            continue
        paths = sorted(p for _, p in entries)
        duplicate_groups.append(paths)
        names = ", ".join(f"{c}/{p.name}" for c, p in sorted(entries, key=lambda e: e[1]))
        if len({c for c, _ in entries}) > 1:
            errors.append(f"identical image content in different classes (label conflict): {names}")
        else:
            warnings.append(f"duplicate images within one class: {names}")
    duplicate_groups.sort()

    counts = {n: r.valid_count for n, r in class_reports.items() if r.valid_count > 0}
    imbalance: float | None = None
    if len(counts) >= 2:
        largest = max(counts, key=lambda n: counts[n])
        smallest = min(counts, key=lambda n: counts[n])
        imbalance = counts[largest] / counts[smallest]
        if imbalance > max_imbalance_ratio:
            warnings.append(
                f"class imbalance: '{largest}' has {counts[largest]} images vs "
                f"'{smallest}' with {counts[smallest]} "
                f"({imbalance:.1f}x, limit {max_imbalance_ratio:g}x)"
            )
    for name, count in counts.items():
        if count < min_recommended_images:
            warnings.append(
                f"class '{name}' has only {count} valid image(s); "
                f"at least {min_recommended_images} are recommended"
            )

    return DatasetReport(
        dataset_dir=dataset_dir,
        dataset_dir_exists=dataset_exists,
        classes=class_reports,
        unexpected_entries=unexpected,
        duplicate_groups=duplicate_groups,
        imbalance_ratio=imbalance,
        errors=errors,
        warnings=warnings,
        max_imbalance_ratio=max_imbalance_ratio,
        min_recommended_images=min_recommended_images,
    )


def validate_dataset_from_config(
    config: Config,
    *,
    max_imbalance_ratio: float = DEFAULT_MAX_IMBALANCE_RATIO,
    min_recommended_images: int = DEFAULT_MIN_RECOMMENDED_IMAGES,
) -> DatasetReport:
    """Validate the dataset described by a loaded ``Config``."""
    return validate_dataset(
        config.paths.dataset_dir,
        config.classes,
        max_imbalance_ratio=max_imbalance_ratio,
        min_recommended_images=min_recommended_images,
    )


MANIFEST_COLUMNS: tuple[str, ...] = (
    "relative_path",
    "material_class",
    "split",
    "group_id",
    "object_id",
    "session_id",
)
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class ManifestRow:
    """One image's split assignment, as stored in the manifest CSV.

    ``relative_path`` is POSIX-style (forward slashes) and relative to the
    dataset directory, e.g. ``metal/metal_20261024_103501_123456.jpg``.
    ``object_id`` and ``session_id`` are empty strings when not recorded.
    """

    relative_path: str
    material_class: str
    split: str
    group_id: str
    object_id: str
    session_id: str


def normalize_relative_path(value: str) -> str:
    """Return the POSIX form of a relative image path, accepting Windows separators.

    Raises ``ValueError`` for absolute paths or paths that climb out of the
    dataset directory (``..``), so a manifest cannot point outside it.
    """
    posix = value.strip().replace("\\", "/")
    parts = PurePosixPath(posix).parts
    if not parts or posix.startswith("/") or PurePosixPath(posix).is_absolute() or ".." in parts:
        raise ValueError(f"manifest path must be relative to the dataset folder: {value!r}")
    if len(posix) > 1 and posix[1] == ":":  # Windows drive letter such as C:...
        raise ValueError(f"manifest path must be relative to the dataset folder: {value!r}")
    return "/".join(parts)


def image_path_for(dataset_dir: Path, relative_path: str) -> Path:
    """Turn a manifest path into a local path, valid on Windows and POSIX."""
    return dataset_dir.joinpath(*normalize_relative_path(relative_path).split("/"))


def read_manifest(path: Path, classes: tuple[str, ...]) -> list[ManifestRow]:
    """Read and check a manifest CSV written by ``src.prepare_dataset``."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != MANIFEST_COLUMNS:
                raise ValueError(
                    f"manifest {path} has columns {reader.fieldnames}; expected {list(MANIFEST_COLUMNS)}"
                )
            rows: list[ManifestRow] = []
            seen: set[str] = set()
            for line_no, raw in enumerate(reader, start=2):
                rel = normalize_relative_path(raw["relative_path"] or "")
                if rel in seen:
                    raise ValueError(f"manifest {path} line {line_no}: duplicate image {rel!r}")
                seen.add(rel)
                if raw["material_class"] not in classes:
                    raise ValueError(
                        f"manifest {path} line {line_no}: unknown class {raw['material_class']!r}"
                    )
                if raw["split"] not in SPLIT_NAMES:
                    raise ValueError(f"manifest {path} line {line_no}: unknown split {raw['split']!r}")
                rows.append(
                    ManifestRow(
                        relative_path=rel,
                        material_class=raw["material_class"],
                        split=raw["split"],
                        group_id=raw["group_id"],
                        object_id=raw["object_id"],
                        session_id=raw["session_id"],
                    )
                )
    except FileNotFoundError:
        raise ValueError(f"manifest not found: {path}") from None
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"cannot read manifest {path}: {exc}") from exc
    return rows


def _name_list(paths: list[Path], limit: int = 5) -> str:
    names = [p.name + ("/" if p.is_dir() else "") for p in paths[:limit]]
    extra = len(paths) - limit
    return ", ".join(names) + (f" (+{extra} more)" if extra > 0 else "")


def format_report(report: DatasetReport) -> str:
    """Render a human-readable report."""
    lines = ["Dataset report", f"Directory: {report.dataset_dir}", ""]

    total = report.total_valid_images
    width = max([len(n) for n in report.classes] + [5])
    lines.append(f"{'Class'.ljust(width)}  {'Valid':>6}  {'Invalid':>7}  {'Share':>6}  Status")
    for name, c in report.classes.items():
        if not c.exists:
            status, share = "MISSING", "-"
        elif c.valid_count == 0:
            status, share = "EMPTY", "-"
        else:
            status = "ok" if not c.invalid else "has invalid images"
            share = f"{100 * c.valid_count / total:.1f}%"
        lines.append(
            f"{name.ljust(width)}  {c.valid_count:>6}  {len(c.invalid):>7}  {share:>6}  {status}"
        )
    lines.append(f"{'Total'.ljust(width)}  {total:>6}  {sum(len(c.invalid) for c in report.classes.values()):>7}")
    lines.append("")

    if report.imbalance_ratio is None:
        lines.append("Imbalance: n/a (needs at least two classes with valid images)")
    else:
        verdict = "above limit" if report.imbalance_ratio > report.max_imbalance_ratio else "within limit"
        lines.append(
            f"Imbalance: largest/smallest = {report.imbalance_ratio:.2f}x "
            f"({verdict}, limit {report.max_imbalance_ratio:g}x)"
        )
    lines.append(f"Duplicate content groups: {len(report.duplicate_groups)}")
    lines.append("")

    if report.errors:
        lines.append(f"ERRORS ({len(report.errors)}):")
        lines.extend(f"  - {e}" for e in report.errors)
        lines.append("")
    if report.warnings:
        lines.append(f"WARNINGS ({len(report.warnings)}):")
        lines.extend(f"  - {w}" for w in report.warnings)
        lines.append("")

    if report.errors:
        lines.append("Result: INVALID (fix the errors above; no files were changed)")
    elif report.warnings:
        lines.append("Result: VALID with warnings")
    else:
        lines.append("Result: VALID")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the raw image dataset (read-only).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument(
        "--max-imbalance",
        type=float,
        default=DEFAULT_MAX_IMBALANCE_RATIO,
        help="warn when largest/smallest class ratio exceeds this (default: %(default)s)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=DEFAULT_MIN_RECOMMENDED_IMAGES,
        help="warn when a class has fewer valid images than this (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        report = validate_dataset_from_config(
            config,
            max_imbalance_ratio=args.max_imbalance,
            min_recommended_images=args.min_images,
        )
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(format_report(report))
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
