"""Tests for src.prepare_dataset (splitting, grouping, manifests, CLI).

Uses small synthetic images in temporary folders. No camera or GPU is needed.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from helpers import make_dataset, snapshot, write_image
from src.config import DEFAULT_CONFIG_PATH, MATERIAL_CLASSES, Config, load_config
from src.dataset import (
    image_path_for,
    normalize_relative_path,
    read_manifest,
    validate_dataset_from_config,
)
from src.prepare_dataset import (
    SplitError,
    main,
    plan_split,
    manifest_json_dict,
    write_outputs,
)

CLASSES = MATERIAL_CLASSES
FOUR_EACH = {c: 4 for c in CLASSES}


class ProjectTestCase(unittest.TestCase):
    """A temporary project with its own config/config.yaml and data/ folder."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "config").mkdir()
        shutil.copy(DEFAULT_CONFIG_PATH, self.root / "config" / "config.yaml")
        self.config_path = self.root / "config" / "config.yaml"
        self.config: Config = load_config(self.config_path)
        self.dataset = self.config.paths.dataset_dir
        self.metadata_path = self.config.dataset.group_metadata
        self.manifest_path = self.config.dataset.manifest
        self.report_path = self.config.dataset.report

    def write_metadata(self, rows: list[tuple[str, ...]], header: tuple[str, ...] = ("path", "object_id", "session_id")) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def plan(self):  # type: ignore[no-untyped-def]
        report = validate_dataset_from_config(self.config, min_recommended_images=0)
        return plan_split(self.config, report)

    def split_of(self, plan) -> dict[str, str]:  # type: ignore[no-untyped-def]
        return {row.relative_path: row.split for row in plan.rows}

    def make_objects(self, objects_per_class: int, photos: int) -> list[tuple[str, ...]]:
        """Write objects_per_class objects per class, each photographed `photos` times.

        Returns grouping-metadata rows (path, object_id, session_id).
        """
        rows: list[tuple[str, ...]] = []
        for name in CLASSES:
            for i in range(objects_per_class * photos):
                write_image(self.dataset / name / f"{name}_{i:03d}.jpg")
                obj = i // photos
                rows.append((f"{name}/{name}_{i:03d}.jpg", f"{name}-object-{obj:02d}", f"session-{obj % 3}"))
        return rows


class ValidSplitTests(ProjectTestCase):
    def test_every_image_is_assigned_exactly_once(self) -> None:
        make_dataset(self.dataset, {c: 40 for c in CLASSES})
        plan = self.plan()
        self.assertEqual(len(plan.rows), 160)
        paths = [row.relative_path for row in plan.rows]
        self.assertEqual(len(paths), len(set(paths)))
        for name in CLASSES:
            on_disk = {f"{name}/{p.name}" for p in (self.dataset / name).iterdir()}
            self.assertEqual({r.relative_path for r in plan.rows if r.material_class == name}, on_disk)

    def test_class_counts_follow_configured_ratios(self) -> None:
        make_dataset(self.dataset, {c: 100 for c in CLASSES})
        plan = self.plan()
        for name in CLASSES:
            self.assertEqual(plan.counts["train"][name], 70)
            self.assertEqual(plan.counts["val"][name], 15)
            self.assertEqual(plan.counts["test"][name], 15)

    def test_splits_do_not_overlap(self) -> None:
        make_dataset(self.dataset, {c: 30 for c in CLASSES})
        plan = self.plan()
        by_split = {s: {r.relative_path for r in plan.rows if r.split == s} for s in ("train", "val", "test")}
        self.assertFalse(by_split["train"] & by_split["val"])
        self.assertFalse(by_split["train"] & by_split["test"])
        self.assertFalse(by_split["val"] & by_split["test"])
        self.assertEqual(sum(len(v) for v in by_split.values()), 120)

    def test_class_names_and_counts_match_config(self) -> None:
        make_dataset(self.dataset, {"metal": 10, "plastic": 20, "paper": 30, "other": 40})
        plan = self.plan()
        self.assertEqual({r.material_class for r in plan.rows}, set(CLASSES))
        totals = {c: sum(plan.counts[s][c] for s in ("train", "val", "test")) for c in CLASSES}
        self.assertEqual(totals, {"metal": 10, "plastic": 20, "paper": 30, "other": 40})

    def test_planning_writes_nothing(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        before = snapshot(self.root)
        self.plan()
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(self.manifest_path.exists())

    def test_raw_images_are_not_changed(self) -> None:
        make_dataset(self.dataset, {c: 40 for c in CLASSES})
        before = {p: p.read_bytes() for p in self.dataset.rglob("*.jpg")}
        write_outputs(self.plan(), self.config, "report", overwrite=False)
        after = {p: p.read_bytes() for p in self.dataset.rglob("*.jpg")}
        self.assertEqual(before, after)


class ReproducibilityTests(ProjectTestCase):
    def test_same_seed_gives_identical_assignments(self) -> None:
        make_dataset(self.dataset, {c: 50 for c in CLASSES})
        self.assertEqual(self.split_of(self.plan()), self.split_of(self.plan()))

    def test_different_seed_gives_different_assignments(self) -> None:
        make_dataset(self.dataset, {c: 50 for c in CLASSES})
        first = self.split_of(self.plan())
        other = replace(self.config, random_seed=self.config.random_seed + 1)
        report = validate_dataset_from_config(other, min_recommended_images=0)
        second = self.split_of(plan_split(other, report))
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first, second)

    def test_written_outputs_are_byte_identical_across_runs(self) -> None:
        make_dataset(self.dataset, {c: 40 for c in CLASSES})
        self.write_metadata([(f"{c}/img_000.jpg", f"obj_{c}", "s1") for c in CLASSES])
        write_outputs(self.plan(), self.config, "report text", overwrite=False)
        first = {p: p.read_bytes() for p in (self.manifest_path, self.manifest_path.with_suffix(".json"), self.report_path)}
        write_outputs(self.plan(), self.config, "report text", overwrite=True, stamp="fixed")
        second = {p: p.read_bytes() for p in first}
        self.assertEqual(first, second)


class GroupAwareSplitTests(ProjectTestCase):
    def test_photos_of_one_object_stay_in_one_split(self) -> None:
        rows = self.make_objects(objects_per_class=8, photos=5)
        self.write_metadata(rows)
        plan = self.plan()
        self.assertTrue(plan.leakage_protection)
        splits_by_object: dict[str, set[str]] = {}
        for row in plan.rows:
            splits_by_object.setdefault(row.object_id, set()).add(row.split)
        self.assertEqual(len(splits_by_object), 32)
        for obj, splits in splits_by_object.items():
            with self.subTest(object=obj):
                self.assertEqual(len(splits), 1)

    def test_group_split_keeps_class_representation(self) -> None:
        rows = self.make_objects(objects_per_class=20, photos=3)  # 60 images per class
        self.write_metadata(rows)
        plan = self.plan()
        for name in CLASSES:
            total = sum(plan.counts[s][name] for s in ("train", "val", "test"))
            self.assertEqual(total, 60)
            self.assertGreater(plan.counts["train"][name], 0)
            self.assertGreater(plan.counts["val"][name], 0)
            self.assertGreater(plan.counts["test"][name], 0)
            self.assertLess(abs(plan.counts["train"][name] / total - 0.70), 0.10)

    def test_session_id_is_recorded_in_manifest_rows(self) -> None:
        rows = self.make_objects(objects_per_class=4, photos=2)
        self.write_metadata(rows)
        plan = self.plan()
        sample = next(r for r in plan.rows if r.object_id == "metal-object-00")
        self.assertEqual(sample.session_id, "session-0")
        self.assertEqual(sample.group_id, "object:metal-object-00")

    def test_identical_files_are_kept_together_without_metadata(self) -> None:
        make_dataset(self.dataset, {c: 10 for c in CLASSES})
        shutil.copy(self.dataset / "metal" / "img_000.jpg", self.dataset / "metal" / "copy.jpg")
        plan = self.plan()
        self.assertFalse(plan.leakage_protection)
        splits = self.split_of(plan)
        self.assertEqual(splits["metal/img_000.jpg"], splits["metal/copy.jpg"])

    def test_one_object_per_class_splits_by_image_with_a_warning(self) -> None:
        rows = self.make_objects(objects_per_class=1, photos=30)  # 30 photos of ONE object per class
        self.write_metadata(rows)
        plan = self.plan()
        self.assertEqual(len(plan.rows), 120)
        for name in CLASSES:
            self.assertGreater(plan.counts["train"][name], 0, name)
            self.assertGreater(plan.counts["val"][name], 0, name)
            self.assertGreater(plan.counts["test"][name], 0, name)
            self.assertEqual(sum(plan.counts[s][name] for s in ("train", "val", "test")), 30)
            self.assertTrue(any(f"class '{name}'" in w and "one object" in w and "by image" in w for w in plan.warnings), name)
        self.assertFalse(plan.leakage_protection)
        self.assertEqual(plan.image_level_classes, CLASSES)

    def test_two_objects_in_a_class_also_falls_back_to_images(self) -> None:
        rows = self.make_objects(objects_per_class=2, photos=10)
        self.write_metadata(rows)
        plan = self.plan()
        for name in CLASSES:
            self.assertGreater(plan.counts["test"][name], 0, name)
        self.assertEqual(plan.image_level_classes, CLASSES)

    def test_classes_with_enough_objects_stay_group_aware_while_others_fall_back(self) -> None:
        rows = []
        for i in range(24):  # metal: 6 objects x 4 photos
            write_image(self.dataset / "metal" / f"metal_{i:03d}.jpg")
            rows.append((f"metal/metal_{i:03d}.jpg", f"metal-object-{i // 4:02d}", "s"))
        for name in ("plastic", "paper", "other"):  # these three have a single object
            for i in range(20):
                write_image(self.dataset / name / f"{name}_{i:03d}.jpg")
                rows.append((f"{name}/{name}_{i:03d}.jpg", f"{name}-only", "s"))
        self.write_metadata(rows)
        plan = self.plan()
        self.assertEqual(plan.image_level_classes, ("plastic", "paper", "other"))
        splits_by_object: dict[str, set[str]] = {}
        for row in plan.rows:
            if row.material_class == "metal":
                splits_by_object.setdefault(row.object_id, set()).add(row.split)
        self.assertEqual(len(splits_by_object), 6)
        self.assertTrue(all(len(v) == 1 for v in splits_by_object.values()))
        self.assertFalse(any("class 'metal'" in w and "by image" in w for w in plan.warnings))

    def test_identical_files_stay_together_even_in_the_image_level_fallback(self) -> None:
        rows = self.make_objects(objects_per_class=1, photos=12)
        shutil.copy(self.dataset / "metal" / "metal_000.jpg", self.dataset / "metal" / "copy.jpg")
        self.write_metadata(rows)
        splits = self.split_of(self.plan())
        self.assertEqual(splits["metal/metal_000.jpg"], splits["metal/copy.jpg"])

    def test_image_level_fallback_is_recorded_in_the_manifest_json(self) -> None:
        self.write_metadata(self.make_objects(objects_per_class=1, photos=12))
        data = manifest_json_dict(self.plan(), self.config)
        self.assertFalse(data["leakage_protection"])
        self.assertEqual(data["image_level_classes"], list(CLASSES))

    def test_image_level_fallback_still_fails_when_a_class_has_too_few_images(self) -> None:
        make_dataset(self.dataset, {"metal": 2, "plastic": 10, "paper": 10, "other": 10})
        with self.assertRaises(SplitError) as ctx:
            self.plan()
        self.assertIn("class 'metal' has only 2 separate group(s)", "\n".join(ctx.exception.messages))

    def test_conflicting_labels_within_an_object_are_reported(self) -> None:
        make_dataset(self.dataset, {c: 6 for c in CLASSES})
        self.write_metadata([("metal/img_000.jpg", "bottle-1", ""), ("plastic/img_000.jpg", "bottle-1", "")])
        with self.assertRaises(SplitError) as ctx:
            self.plan()
        message = "\n".join(ctx.exception.messages)
        self.assertIn("conflicting labels", message)
        self.assertIn("bottle-1", message)

    def test_metadata_rows_must_match_valid_images(self) -> None:
        make_dataset(self.dataset, {c: 6 for c in CLASSES})
        self.write_metadata([("metal/does_not_exist.jpg", "x", "")])
        with self.assertRaises(SplitError) as ctx:
            self.plan()
        self.assertIn("is not a valid image in the dataset", "\n".join(ctx.exception.messages))

    def test_metadata_structure_errors_are_all_reported(self) -> None:
        make_dataset(self.dataset, {c: 6 for c in CLASSES})
        cases = {
            "missing object column": ([("metal/img_000.jpg", "")], ("path", "objectid")),
            "empty object_id": ([("metal/img_000.jpg", "  ", "")], ("path", "object_id", "session_id")),
            "duplicate row": ([("metal/img_000.jpg", "a", ""), ("metal/img_000.jpg", "b", "")], ("path", "object_id", "session_id")),
            "pipe in object_id": ([("metal/img_000.jpg", "a|b", "")], ("path", "object_id", "session_id")),
        }
        for label, (rows, header) in cases.items():
            with self.subTest(case=label):
                self.write_metadata(rows, header=header)
                with self.assertRaises(SplitError):
                    self.plan()

    def test_missing_metadata_warns_and_disables_leakage_protection(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        plan = self.plan()
        self.assertFalse(plan.leakage_protection)
        self.assertTrue(any("no grouping metadata" in w for w in plan.warnings))
        self.assertIsNone(plan.metadata_sha256)

    def test_images_missing_from_metadata_are_warned_about(self) -> None:
        rows = self.make_objects(objects_per_class=6, photos=2)
        self.write_metadata(rows[:-3])  # the last three images (in "other") have no row
        plan = self.plan()
        self.assertTrue(plan.leakage_protection)
        self.assertTrue(any("3 image(s) have no row in the grouping metadata" in w for w in plan.warnings))

    def test_too_few_groups_fails_clearly(self) -> None:
        make_dataset(self.dataset, {"metal": 2, "plastic": 4, "paper": 4, "other": 4})
        with self.assertRaises(SplitError) as ctx:
            self.plan()
        self.assertIn("class 'metal' has only 2 separate group(s)", "\n".join(ctx.exception.messages))

    def test_large_group_triggers_deviation_warning(self) -> None:
        make_dataset(self.dataset, {c: 3 for c in CLASSES})
        extra = [self.dataset / "metal" / f"big_{i}.jpg" for i in range(40)]
        for i, path in enumerate(extra):
            write_image(path, seed=10_000 + i)
        rows = [(f"metal/big_{i}.jpg", "giant", "") for i in range(40)]
        rows += [(f"metal/img_{i:03d}.jpg", f"small-{i}", "") for i in range(3)]
        self.write_metadata(rows)
        plan = self.plan()
        self.assertTrue(any("class 'metal'" in w and "target" in w for w in plan.warnings))


class InvalidDatasetTests(ProjectTestCase):
    def test_missing_class_blocks_split(self) -> None:
        make_dataset(self.dataset, {"metal": 10, "paper": 10, "other": 10})
        with self.assertRaises(SplitError) as ctx:
            self.plan()
        self.assertIn("dataset is invalid", str(ctx.exception))

    def test_empty_class_blocks_split(self) -> None:
        make_dataset(self.dataset, {"metal": 10, "plastic": 0, "paper": 10, "other": 10})
        (self.dataset / "plastic").mkdir(exist_ok=True)
        with self.assertRaises(SplitError):
            self.plan()

    def test_corrupt_image_blocks_split_and_is_not_silently_dropped(self) -> None:
        make_dataset(self.dataset, {c: 20 for c in CLASSES})
        (self.dataset / "metal" / "broken.jpg").write_bytes(b"not an image")
        with self.assertRaises(SplitError):
            self.plan()
        self.assertFalse(self.manifest_path.exists())

    def test_identical_images_in_two_classes_block_split(self) -> None:
        make_dataset(self.dataset, {c: 20 for c in CLASSES})
        shutil.copy(self.dataset / "metal" / "img_000.jpg", self.dataset / "paper" / "same.jpg")
        with self.assertRaises(SplitError):
            self.plan()


class ManifestTests(ProjectTestCase):
    def test_manifest_round_trips_through_reader(self) -> None:
        rows = self.make_objects(objects_per_class=5, photos=2)
        self.write_metadata(rows)
        plan = self.plan()
        write_outputs(plan, self.config, "report", overwrite=False)
        loaded = read_manifest(self.manifest_path, CLASSES)
        self.assertEqual(loaded, plan.rows)

    def test_manifest_paths_are_relative_and_posix(self) -> None:
        make_dataset(self.dataset, {c: 10 for c in CLASSES})
        plan = self.plan()
        write_outputs(plan, self.config, "report", overwrite=False)
        text = self.manifest_path.read_text(encoding="utf-8")
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("\\", text)
        for row in plan.rows:
            self.assertFalse(Path(row.relative_path).is_absolute())
            self.assertEqual(image_path_for(self.dataset, row.relative_path).exists(), True)

    def test_json_metadata_records_seed_and_ratios_without_absolute_paths(self) -> None:
        make_dataset(self.dataset, {c: 10 for c in CLASSES})
        plan = self.plan()
        write_outputs(plan, self.config, "report", overwrite=False)
        text = self.manifest_path.with_suffix(".json").read_text(encoding="utf-8")
        data = json.loads(text)
        self.assertEqual(data["random_seed"], 42)
        self.assertEqual(data["split_ratios"], {"train": 0.7, "val": 0.15, "test": 0.15})
        self.assertEqual(data["classes"], list(CLASSES))
        self.assertFalse(data["leakage_protection"])
        self.assertNotIn(str(self.root), text)

    def test_existing_manifest_is_not_overwritten_without_flag(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        self.manifest_path.parent.mkdir(parents=True)
        self.manifest_path.write_text("keep me\n", encoding="utf-8")
        with self.assertRaises(SplitError) as ctx:
            write_outputs(self.plan(), self.config, "report", overwrite=False)
        self.assertIn("--overwrite", str(ctx.exception))
        self.assertEqual(self.manifest_path.read_text(encoding="utf-8"), "keep me\n")
        self.assertFalse(self.report_path.exists())

    def test_overwrite_keeps_a_backup_of_the_old_manifest(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        self.manifest_path.parent.mkdir(parents=True)
        self.manifest_path.write_text("old\n", encoding="utf-8")
        write_outputs(self.plan(), self.config, "report", overwrite=True, stamp="20260101_000000")
        backup = self.manifest_path.with_name(self.manifest_path.name + ".20260101_000000.bak")
        self.assertEqual(backup.read_text(encoding="utf-8"), "old\n")
        self.assertNotEqual(self.manifest_path.read_text(encoding="utf-8"), "old\n")

    def test_reader_rejects_bad_manifests(self) -> None:
        self.manifest_path.parent.mkdir(parents=True)
        cases = {
            "wrong columns": "a,b\n1,2\n",
            "unknown class": "relative_path,material_class,split,group_id,object_id,session_id\nmetal/a.jpg,glass,train,g,,\n",
            "unknown split": "relative_path,material_class,split,group_id,object_id,session_id\nmetal/a.jpg,metal,dev,g,,\n",
            "escaping path": "relative_path,material_class,split,group_id,object_id,session_id\n../x.jpg,metal,train,g,,\n",
            "absolute path": "relative_path,material_class,split,group_id,object_id,session_id\nC:\\x.jpg,metal,train,g,,\n",
        }
        for label, text in cases.items():
            with self.subTest(case=label):
                self.manifest_path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_manifest(self.manifest_path, CLASSES)

    def test_windows_style_paths_are_accepted(self) -> None:
        self.assertEqual(normalize_relative_path("metal\\img.jpg"), "metal/img.jpg")
        self.assertEqual(normalize_relative_path("paper/sub/x.png"), "paper/sub/x.png")
        for bad in ("../secret.jpg", "C:\\data\\x.jpg", "/etc/x.jpg", ""):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                normalize_relative_path(bad)
        path = image_path_for(self.dataset, "metal\\a.jpg")
        self.assertEqual(path, self.dataset / "metal" / "a.jpg")


class CommandLineTests(ProjectTestCase):
    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--config", str(self.config_path), *args])
        return code, out.getvalue()

    def test_valid_run_writes_outputs_and_returns_zero(self) -> None:
        make_dataset(self.dataset, {c: 30 for c in CLASSES})
        code, out = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertIn("Split summary", out)
        self.assertTrue(self.manifest_path.exists())
        self.assertTrue(self.manifest_path.with_suffix(".json").exists())
        self.assertTrue(self.report_path.exists())

    def test_dry_run_writes_nothing(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        before = snapshot(self.root)
        code, out = self.run_main("--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("Dry run", out)
        self.assertEqual(snapshot(self.root), before)

    def test_invalid_dataset_returns_one_and_writes_nothing(self) -> None:
        make_dataset(self.dataset, {"metal": 10})
        before = snapshot(self.root)
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("no split was created", out)
        self.assertEqual(snapshot(self.root), before)

    def test_existing_outputs_return_one_without_overwrite(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        self.manifest_path.parent.mkdir(parents=True)
        self.manifest_path.write_text("x", encoding="utf-8")
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("--overwrite", out)
        self.assertEqual(self.manifest_path.read_text(encoding="utf-8"), "x")

    def test_bad_config_returns_two(self) -> None:
        code, out = self.run_main_with_config(self.root / "missing.yaml")
        self.assertEqual(code, 2)
        self.assertIn("config file not found", out)

    def run_main_with_config(self, path: Path) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--config", str(path)])
        return code, out.getvalue()


if __name__ == "__main__":
    unittest.main()
