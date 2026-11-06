"""Split the validated raw dataset into train / validation / test sets.

What it does:

1. Validates the raw dataset with ``src.dataset`` (reused, not duplicated).
   Any error (missing or empty class, corrupt image, identical image in two
   classes) stops the run before anything is written.
2. Reads the optional grouping metadata CSV (``dataset.group_metadata``).
   Images that share an ``object_id`` form one group, and exact duplicate
   files are always kept in one group.
3. Splits each class separately (stratified) so that every group lands
   entirely in one split. Class proportions are kept as close to the
   configured ratios as the group sizes allow. A class with fewer than three
   separate objects (for example one object photographed many times) cannot be
   split by object, so that class is split image by image instead (identical
   files still stay together). This leaks near-copies of the object across the
   splits, so it is reported as a warning and recorded in the manifest.
4. Writes ``manifest.csv`` (one row per image), ``manifest.json`` (seed, ratios
   and counts, with no absolute paths) and a text report.

The raw images are only read, never moved, renamed or changed. Existing
outputs are never overwritten unless ``--overwrite`` is given; in that case the
old files are first renamed to ``<name>.<timestamp>.bak``.

Run from the project root:

    python -m src.prepare_dataset
    python -m src.prepare_dataset --dry-run      # report only, write nothing
    python -m src.prepare_dataset --overwrite    # replace old outputs (keeps .bak copies)

Exit codes: 0 success (warnings allowed), 1 invalid dataset / impossible split /
outputs exist, 2 bad config or arguments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from src.dataset import (
    SPLIT_NAMES,
    MANIFEST_COLUMNS,
    DatasetReport,
    ManifestRow,
    format_report,
    normalize_relative_path,
    validate_dataset_from_config,
)

MANIFEST_SCHEMA_VERSION = 1
GROUP_METADATA_REQUIRED_COLUMNS: tuple[str, ...] = ("path", "object_id")
GROUP_METADATA_OPTIONAL_COLUMNS: tuple[str, ...] = ("session_id",)

# A train/val/test split needs at least one group per split in every class.
MIN_GROUPS_PER_CLASS: int = len(SPLIT_NAMES)
# Warn when a class has fewer images than this in a split (reporting only).
RECOMMENDED_MIN_IMAGES_PER_SPLIT: int = 10
# Warn when a split's share of a class differs from its target by more than this.
MAX_SHARE_DEVIATION: float = 0.10


class SplitError(Exception):
    """The dataset cannot be split validly, or the outputs cannot be written."""

    def __init__(self, messages: list[str] | str) -> None:
        self.messages = [messages] if isinstance(messages, str) else list(messages)
        super().__init__("; ".join(self.messages))


@dataclass(frozen=True)
class SplitPlan:
    """Everything decided by a split, before anything is written to disk."""

    rows: list[ManifestRow]
    counts: dict[str, dict[str, int]]  # split -> class -> number of images
    groups: dict[str, int]  # class -> number of groups
    grouping: str  # human-readable description of how images were grouped
    metadata_sha256: str | None  # hash of the grouping CSV, if it was used
    images_without_metadata: int
    warnings: list[str]
    image_level_classes: tuple[str, ...] = ()  # classes split image by image (too few objects)

    @property
    def leakage_protection(self) -> bool:
        return self.metadata_sha256 is not None and not self.image_level_classes


# --------------------------------------------------------------------------
# Grouping metadata
# --------------------------------------------------------------------------


def read_group_metadata(path: Path, known_images: set[str]) -> tuple[dict[str, tuple[str, str]], str]:
    """Read the grouping CSV. Returns ``{relative_path: (object_id, session_id)}``.

    Every problem is collected and reported together in a ``SplitError``.
    """
    errors: list[str] = []
    result: dict[str, tuple[str, str]] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            allowed = GROUP_METADATA_REQUIRED_COLUMNS + GROUP_METADATA_OPTIONAL_COLUMNS
            missing = [c for c in GROUP_METADATA_REQUIRED_COLUMNS if c not in columns]
            unknown = [c for c in columns if c not in allowed]
            if missing:
                errors.append(f"{path.name}: missing column(s) {missing}")
            if unknown:
                errors.append(f"{path.name}: unknown column(s) {unknown}; expected {list(allowed)}")
            if missing or unknown:
                raise SplitError(errors)
            for line_no, row in enumerate(reader, start=2):
                try:
                    rel = normalize_relative_path(row["path"] or "")
                except ValueError as exc:
                    errors.append(f"{path.name} line {line_no}: {exc}")
                    continue
                if rel not in known_images:
                    errors.append(
                        f"{path.name} line {line_no}: {rel!r} is not a valid image in the dataset"
                    )
                    continue
                if rel in result:
                    errors.append(f"{path.name} line {line_no}: {rel!r} listed more than once")
                    continue
                object_id = (row["object_id"] or "").strip()
                if not object_id:
                    errors.append(f"{path.name} line {line_no}: empty object_id for {rel!r}")
                    continue
                if "|" in object_id:
                    errors.append(f"{path.name} line {line_no}: object_id may not contain '|'")
                    continue
                session_id = (row.get("session_id") or "").strip()
                result[rel] = (object_id, session_id)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise SplitError(f"cannot read grouping metadata {path}: {exc}") from exc
    if errors:
        raise SplitError(errors)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return result, digest


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[max(root_a, root_b)] = min(root_a, root_b)


@dataclass(frozen=True)
class _Group:
    group_id: str
    members: tuple[str, ...]  # relative paths, sorted


def _build_groups(
    images: dict[str, str],
    metadata: dict[str, tuple[str, str]],
    duplicate_groups: list[list[str]],
) -> tuple[list[_Group], list[str]]:
    """Merge images into groups: same object_id, or identical file content.

    Returns the groups and a list of errors (groups whose images have
    different material classes, i.e. conflicting labels).
    """
    names = sorted(images)
    index = {name: i for i, name in enumerate(names)}
    uf = _UnionFind(len(names))

    first_with_object: dict[str, int] = {}
    for name in names:
        object_id = metadata.get(name, ("", ""))[0]
        if not object_id:
            continue
        if object_id in first_with_object:
            uf.union(first_with_object[object_id], index[name])
        else:
            first_with_object[object_id] = index[name]

    for members in duplicate_groups:
        ids = [index[m] for m in members if m in index]
        for other in ids[1:]:
            uf.union(ids[0], other)

    components: dict[int, list[str]] = defaultdict(list)
    for name in names:
        components[uf.find(index[name])].append(name)

    errors: list[str] = []
    groups: list[_Group] = []
    for members in components.values():
        members = sorted(members)
        classes = {images[m] for m in members}
        object_ids = sorted({metadata[m][0] for m in members if m in metadata})
        if len(classes) > 1:
            shown = ", ".join(members[:5]) + (f" (+{len(members) - 5} more)" if len(members) > 5 else "")
            ids = f" (object_id {', '.join(repr(o) for o in object_ids)})" if object_ids else ""
            errors.append(
                f"conflicting labels in one group{ids}: the same object or identical images "
                f"appear in classes {sorted(classes)}: {shown}"
            )
            continue
        group_id = "object:" + "|".join(object_ids) if object_ids else "image:" + members[0]
        groups.append(_Group(group_id=group_id, members=tuple(members)))
    groups.sort(key=lambda g: g.group_id)
    return groups, errors


# --------------------------------------------------------------------------
# Stratified assignment
# --------------------------------------------------------------------------


def assign_groups(
    groups: list[_Group],
    ratios: dict[str, float],
    rng: random.Random,
) -> dict[str, str]:
    """Assign each group to one split. Returns ``{group_id: split}``.

    Groups are shuffled with the seeded RNG, then the largest first. The first
    group goes to each split, so no split can be empty. Each remaining group
    goes to the split that is furthest below its target image count, which
    keeps the class close to the configured ratios.
    """
    order = sorted(groups, key=lambda g: g.group_id)
    rng.shuffle(order)
    order.sort(key=lambda g: len(g.members), reverse=True)  # stable: keeps the random order within a size

    names = list(ratios)
    total = sum(len(g.members) for g in groups)
    targets = {name: ratios[name] * total for name in names}
    filled = {name: 0 for name in names}
    assignment: dict[str, str] = {}

    for name, group in zip(names, order):  # one group per split first
        assignment[group.group_id] = name
        filled[name] += len(group.members)
    for group in order[len(names):]:
        best = max(names, key=lambda n: (targets[n] - filled[n], -names.index(n)))
        assignment[group.group_id] = best
        filled[best] += len(group.members)
    return assignment


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def plan_split(
    config: Config,
    report: DatasetReport,
    metadata_path: Path | None = None,
) -> SplitPlan:
    """Decide the split for a validated dataset. Writes nothing.

    Raises ``SplitError`` with every blocking problem found.
    """
    if not report.is_valid:
        raise SplitError("the dataset is invalid; fix the errors in the validation report first")

    dataset_dir = config.paths.dataset_dir
    metadata_path = metadata_path or config.dataset.group_metadata

    images: dict[str, str] = {}
    for name in config.classes:
        class_report = report.classes[name]
        for path in class_report.valid_paths:
            images[path.relative_to(dataset_dir).as_posix()] = name

    metadata: dict[str, tuple[str, str]] = {}
    digest: str | None = None
    if metadata_path.exists():
        if not metadata_path.is_file():
            raise SplitError(f"grouping metadata path is not a file: {metadata_path}")
        metadata, digest = read_group_metadata(metadata_path, set(images))

    duplicate_groups = [
        [p.relative_to(dataset_dir).as_posix() for p in group] for group in report.duplicate_groups
    ]
    groups, errors = _build_groups(images, metadata, duplicate_groups)
    if errors:
        raise SplitError(errors)

    by_class: dict[str, list[_Group]] = {name: [] for name in config.classes}
    for group in groups:
        by_class[images[group.members[0]]].append(group)

    # A class with too few objects is split image by image instead (identical files still stay together).
    image_level: list[str] = []
    too_few: list[str] = []
    object_counts = {name: len(gs) for name, gs in by_class.items()}
    for name in config.classes:
        if len(by_class[name]) >= MIN_GROUPS_PER_CLASS:
            continue
        class_images = {path: c for path, c in images.items() if c == name}
        class_duplicates = [g for g in duplicate_groups if g and g[0] in class_images]
        per_image, _ = _build_groups(class_images, {}, class_duplicates)
        if len(per_image) < MIN_GROUPS_PER_CLASS:
            too_few.append(
                f"class '{name}' has only {len(per_image)} separate group(s); a train/val/test split needs "
                f"at least {MIN_GROUPS_PER_CLASS}. Add more images of distinct objects."
            )
            continue
        by_class[name] = per_image
        image_level.append(name)
    if too_few:
        raise SplitError(too_few)

    ratios = {"train": config.split.train, "val": config.split.val, "test": config.split.test}
    rng = random.Random(config.random_seed)
    rows: list[ManifestRow] = []
    counts: dict[str, dict[str, int]] = {s: {c: 0 for c in config.classes} for s in SPLIT_NAMES}
    group_counts: dict[str, int] = {}
    for name in config.classes:  # stratify: each class is split on its own
        assignment = assign_groups(by_class[name], ratios, rng)
        group_counts[name] = len(by_class[name])
        for group in by_class[name]:
            split = assignment[group.group_id]
            for member in group.members:
                object_id, session_id = metadata.get(member, ("", ""))
                rows.append(
                    ManifestRow(
                        relative_path=member,
                        material_class=name,
                        split=split,
                        group_id=group.group_id,
                        object_id=object_id,
                        session_id=session_id,
                    )
                )
                counts[split][name] += 1

    without_metadata = sum(1 for name in images if name not in metadata)
    warnings = _split_warnings(config, counts, ratios, metadata_path, digest, without_metadata)
    for name in image_level:
        n = object_counts[name]
        found = "one object" if n == 1 else f"only {n} objects"
        warnings.insert(
            0,
            f"class '{name}' has {found} (a split by object needs at least {MIN_GROUPS_PER_CLASS}), so it was split "
            "by image. Near-copies of the same object appear in train, val and test, so validation and test "
            "accuracy for this class is optimistic and does not show how the model handles a new object. "
            "Add photos of more objects to get a real test.",
        )
    grouping = (
        f"object_id groups from {metadata_path.name} plus identical-file groups"
        if digest is not None
        else "identical-file groups only (no grouping metadata)"
    )
    rows.sort(key=lambda r: (config.classes.index(r.material_class), r.relative_path))
    return SplitPlan(
        rows=rows,
        counts=counts,
        groups=group_counts,
        grouping=grouping,
        metadata_sha256=digest,
        images_without_metadata=without_metadata if digest is not None else len(images),
        warnings=warnings,
        image_level_classes=tuple(image_level),
    )


def _split_warnings(
    config: Config,
    counts: dict[str, dict[str, int]],
    ratios: dict[str, float],
    metadata_path: Path,
    digest: str | None,
    without_metadata: int,
) -> list[str]:
    warnings: list[str] = []
    if digest is None:
        warnings.append(
            f"no grouping metadata file found ({metadata_path.name}, set by dataset.group_metadata "
            "in config.yaml). Repeated photos of the same object "
            "may be placed in different splits (data leakage, inflated test accuracy). Only "
            "identical files are kept together. Add the CSV described in the README to prevent this."
        )
    elif without_metadata:
        warnings.append(
            f"{without_metadata} image(s) have no row in the grouping metadata; each is treated as "
            "its own object, so repeated photos of those objects may leak across splits."
        )

    for name in config.classes:
        total = sum(counts[s][name] for s in SPLIT_NAMES)
        for split in SPLIT_NAMES:
            n = counts[split][name]
            if n < RECOMMENDED_MIN_IMAGES_PER_SPLIT:
                warnings.append(
                    f"class '{name}' has only {n} image(s) in the '{split}' split "
                    f"(at least {RECOMMENDED_MIN_IMAGES_PER_SPLIT} recommended)"
                )
            if total and abs(n / total - ratios[split]) > MAX_SHARE_DEVIATION:
                warnings.append(
                    f"class '{name}': '{split}' holds {100 * n / total:.1f}% of its images "
                    f"(target {100 * ratios[split]:.1f}%), because whole groups must stay together"
                )
    return warnings


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def manifest_json_dict(plan: SplitPlan, config: Config) -> dict[str, object]:
    """Metadata written next to the CSV. No timestamps or absolute paths, so it is reproducible."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "random_seed": config.random_seed,
        "split_ratios": {"train": config.split.train, "val": config.split.val, "test": config.split.test},
        "classes": list(config.classes),
        "grouping": plan.grouping,
        "leakage_protection": plan.leakage_protection,
        "group_metadata_sha256": plan.metadata_sha256,
        "image_path_base": "dataset_dir (paths are POSIX-style and relative to it)",
        "total_images": len(plan.rows),
        "image_level_classes": list(plan.image_level_classes),
        "groups_per_class": plan.groups,
        "image_counts": {s: plan.counts[s] for s in SPLIT_NAMES},
        "manifest_columns": list(MANIFEST_COLUMNS),
        "warnings": plan.warnings,
    }


def format_split_report(report: DatasetReport, plan: SplitPlan, config: Config) -> str:
    """Validation report followed by the split summary and grouping status."""
    lines = [format_report(report), "", "Split summary", "-------------"]
    lines.append(f"Random seed: {config.random_seed}")
    lines.append(
        f"Ratios: train {config.split.train:g}, val {config.split.val:g}, test {config.split.test:g}"
    )
    if plan.image_level_classes:
        lines.append(
            "Grouping: classes split image by image because they have too few objects: "
            + ", ".join(plan.image_level_classes)
            + " (leakage prevention OFF for them)"
        )
    elif plan.leakage_protection:
        lines.append("Grouping: object groups from metadata + identical-file groups (leakage prevention ON)")
    else:
        lines.append("Grouping: identical-file groups only (leakage prevention OFF: no grouping metadata)")
    lines.append("")

    width = max([len(c) for c in config.classes] + [7])
    header = "Split".ljust(6) + "  " + "  ".join(c.rjust(width) for c in config.classes) + "  " + "Total".rjust(width)
    lines.append(header)
    for split in SPLIT_NAMES:
        row_counts = [plan.counts[split][c] for c in config.classes]
        lines.append(
            split.ljust(6) + "  " + "  ".join(str(n).rjust(width) for n in row_counts)
            + "  " + str(sum(row_counts)).rjust(width)
        )
    totals = [sum(plan.counts[s][c] for s in SPLIT_NAMES) for c in config.classes]
    lines.append(
        "Total".ljust(6) + "  " + "  ".join(str(n).rjust(width) for n in totals)
        + "  " + str(sum(totals)).rjust(width)
    )
    lines.append("")
    lines.append("Groups per class: " + ", ".join(f"{c}={plan.groups[c]}" for c in config.classes))
    lines.append(f"Images written to manifest: {len(plan.rows)} (no images were dropped)")
    lines.append("")
    if plan.warnings:
        lines.append(f"SPLIT WARNINGS ({len(plan.warnings)}):")
        lines.extend(f"  - {w}" for w in plan.warnings)
    else:
        lines.append("Split warnings: none")
    return "\n".join(lines)


def _backup_path(path: Path, stamp: str) -> Path:
    return path.with_name(f"{path.name}.{stamp}.bak")


def output_paths(config: Config) -> list[Path]:
    """Manifest CSV, manifest JSON and report files, in that order."""
    manifest_csv = config.dataset.manifest
    return [manifest_csv, manifest_csv.with_name(manifest_csv.stem + ".json"), config.dataset.report]


def check_outputs_free(config: Config, overwrite: bool) -> list[Path]:
    """Return existing output files that a write would replace. Raise if not allowed."""
    existing = [p for p in output_paths(config) if p.exists()]
    if existing and not overwrite:
        raise SplitError(
            "refusing to overwrite existing file(s): "
            + ", ".join(str(p) for p in existing)
            + ". Re-run with --overwrite to replace them (the old files are kept as .bak copies)."
        )
    return existing


def write_outputs(
    plan: SplitPlan,
    config: Config,
    report_text: str,
    *,
    overwrite: bool = False,
    stamp: str | None = None,
) -> list[Path]:
    """Write manifest CSV, manifest JSON and report. Returns the files written.

    Existing files are never replaced unless ``overwrite`` is True, and then
    they are first renamed to ``<name>.<timestamp>.bak``.
    """
    manifest_csv, manifest_json, report_file = output_paths(config)
    targets = [manifest_csv, manifest_json, report_file]

    existing = check_outputs_free(config, overwrite)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in existing:
        path.replace(_backup_path(path, stamp))

    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = manifest_csv.with_name(manifest_csv.name + ".tmp")
    with temp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        for row in plan.rows:
            writer.writerow(
                [row.relative_path, row.material_class, row.split, row.group_id, row.object_id, row.session_id]
            )
    _write_text_atomic(manifest_json, json.dumps(manifest_json_dict(plan, config), indent=2, sort_keys=True) + "\n")
    _write_text_atomic(report_file, report_text + "\n")
    os.replace(temp_csv, manifest_csv)
    return targets


def _write_text_atomic(path: Path, text: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split the raw dataset into train/val/test with group-aware stratification."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing manifest/report; old files are renamed to *.<timestamp>.bak",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the split report, write nothing")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2

    report = validate_dataset_from_config(config)
    if not report.is_valid:
        print(format_report(report))
        print("\nerror: dataset is invalid, no split was created.")
        return 1

    try:
        if not args.dry_run:
            check_outputs_free(config, args.overwrite)  # fail before printing a "success" report
        plan = plan_split(config, report)
        text = format_split_report(report, plan, config)
        print(text)
        if args.dry_run:
            print("\nDry run: no files were written.")
            return 0
        written = write_outputs(plan, config, text, overwrite=args.overwrite)
    except SplitError as exc:
        print("\nerror: no split was written:")
        for message in exc.messages:
            print(f"  - {message}")
        return 1

    print("\nWritten:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
