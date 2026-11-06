"""Tests for src.dataset."""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from helpers import encode_image, make_dataset, snapshot, write_image
from src.config import DEFAULT_CONFIG_PATH, MATERIAL_CLASSES, parse_config
from src.dataset import (
    count_images,
    format_report,
    list_image_files,
    main,
    validate_dataset,
    validate_dataset_from_config,
)

CLASSES = MATERIAL_CLASSES
FOUR_EACH = {c: 4 for c in CLASSES}


class DatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.dataset = self.root / "dataset"

    def validate(self, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("min_recommended_images", 0)
        return validate_dataset(self.dataset, CLASSES, **kwargs)


class ValidDatasetTests(DatasetTestCase):
    def test_clean_dataset_is_valid(self) -> None:
        make_dataset(self.dataset, {"metal": 4, "plastic": 5, "paper": 4, "other": 5})
        report = self.validate()
        self.assertTrue(report.is_valid)
        self.assertEqual(report.errors, [])
        self.assertEqual(report.warnings, [])
        self.assertEqual({n: c.valid_count for n, c in report.classes.items()},
                         {"metal": 4, "plastic": 5, "paper": 4, "other": 5})
        self.assertEqual(report.total_valid_images, 18)
        self.assertAlmostEqual(report.imbalance_ratio or 0, 1.25)
        self.assertIn("Result: VALID", format_report(report))

    def test_extensions_are_case_insensitive_and_png_bmp_supported(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        write_image(self.dataset / "metal" / "upper.JPG")
        write_image(self.dataset / "metal" / "a.png")
        write_image(self.dataset / "metal" / "b.bmp")
        self.assertEqual(self.validate().classes["metal"].valid_count, 7)

    def test_count_and_list_helpers(self) -> None:
        make_dataset(self.dataset, {"metal": 3})
        (self.dataset / "metal" / "notes.txt").write_text("x")
        self.assertEqual(count_images(self.dataset / "metal"), 3)
        self.assertEqual(len(list_image_files(self.dataset / "metal")), 3)
        self.assertEqual(count_images(self.dataset / "does_not_exist"), 0)

    def test_validate_from_config(self) -> None:
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config = parse_config(raw, self.root)  # dataset_dir -> root/data/dataset
        make_dataset(config.paths.dataset_dir, FOUR_EACH)
        report = validate_dataset_from_config(config, min_recommended_images=0)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_valid_images, 16)


class MissingAndEmptyTests(DatasetTestCase):
    def test_missing_dataset_directory(self) -> None:
        report = self.validate()
        self.assertFalse(report.is_valid)
        self.assertFalse(report.dataset_dir_exists)
        self.assertTrue(any("dataset directory does not exist" in e for e in report.errors))

    def test_missing_class_directory(self) -> None:
        make_dataset(self.dataset, {"metal": 3, "paper": 3, "other": 3})
        report = self.validate()
        self.assertFalse(report.is_valid)
        self.assertFalse(report.classes["plastic"].exists)
        self.assertIn("class directory missing: plastic/", report.errors)
        self.assertIn("MISSING", format_report(report))

    def test_empty_class_directory(self) -> None:
        make_dataset(self.dataset, {"metal": 3, "plastic": 0, "paper": 3, "other": 3})
        (self.dataset / "plastic").mkdir(exist_ok=True)
        report = self.validate()
        self.assertFalse(report.is_valid)
        self.assertIn("class 'plastic' has no valid images", report.errors)
        self.assertIn("EMPTY", format_report(report))

    def test_class_with_only_corrupt_images_counts_as_empty(self) -> None:
        make_dataset(self.dataset, {"metal": 3, "paper": 3, "other": 3})
        (self.dataset / "plastic").mkdir()
        (self.dataset / "plastic" / "bad.jpg").write_bytes(b"garbage")
        report = self.validate()
        self.assertIn("class 'plastic' has no valid images", report.errors)
        self.assertEqual(report.classes["plastic"].valid_count, 0)


class CorruptImageTests(DatasetTestCase):
    def test_corrupt_zero_byte_and_truncated_files_are_reported_not_deleted(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        metal = self.dataset / "metal"
        (metal / "garbage.jpg").write_bytes(b"this is not an image")
        (metal / "zero.png").write_bytes(b"")
        (metal / "cut.png").write_bytes(encode_image(ext=".png")[:200])
        before = snapshot(self.root)

        report = self.validate()

        self.assertFalse(report.is_valid)
        reasons = {i.path.name: i.reason for i in report.classes["metal"].invalid}
        self.assertEqual(set(reasons), {"garbage.jpg", "zero.png", "cut.png"})
        self.assertIn("empty", reasons["zero.png"])
        self.assertIn("cannot be decoded", reasons["garbage.jpg"])
        self.assertEqual(report.classes["metal"].valid_count, 4)
        self.assertEqual(sum("invalid image metal/" in e for e in report.errors), 3)
        self.assertEqual(snapshot(self.root), before)  # nothing deleted, moved or changed

    def test_folder_named_like_an_image_is_not_scanned_as_one(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        (self.dataset / "metal" / "weird.jpg").mkdir()  # a folder named like an image
        report = self.validate()
        self.assertEqual(report.classes["metal"].valid_count, 4)
        self.assertEqual(len(report.classes["metal"].nested_dirs), 1)


class ExtraContentTests(DatasetTestCase):
    def test_non_image_files_nested_dirs_and_unexpected_entries_warn(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        (self.dataset / "metal" / "notes.txt").write_text("hi")
        (self.dataset / "paper" / "old").mkdir()
        (self.dataset / "glass").mkdir()
        (self.dataset / "readme.md").write_text("x")
        before = snapshot(self.root)

        report = self.validate()

        self.assertTrue(report.is_valid)
        text = "\n".join(report.warnings)
        self.assertIn("non-image file(s) that are ignored: notes.txt", text)
        self.assertIn("sub-folder(s) that are not scanned: old/", text)
        self.assertIn("not configured classes", text)
        self.assertIn("glass/", text)
        self.assertEqual(snapshot(self.root), before)


class DuplicateTests(DatasetTestCase):
    def test_duplicate_within_one_class_is_warning(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        shutil.copy(self.dataset / "metal" / "img_000.jpg", self.dataset / "metal" / "copy.jpg")
        report = self.validate()
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.duplicate_groups), 1)
        self.assertTrue(any("duplicate images within one class" in w for w in report.warnings))

    def test_duplicate_across_classes_is_error(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        shutil.copy(self.dataset / "metal" / "img_000.jpg", self.dataset / "paper" / "oops.jpg")
        report = self.validate()
        self.assertFalse(report.is_valid)
        self.assertTrue(any("label conflict" in e for e in report.errors))
        self.assertEqual(len(report.duplicate_groups), 1)

    def test_same_pixels_but_different_files_are_not_flagged(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        data = encode_image(seed=999)
        (self.dataset / "metal" / "x.jpg").write_bytes(data)
        (self.dataset / "paper" / "y.png").write_bytes(encode_image(seed=999, ext=".png"))
        self.assertEqual(self.validate().duplicate_groups, [])


class ImbalanceTests(DatasetTestCase):
    def test_imbalance_above_limit_warns(self) -> None:
        make_dataset(self.dataset, {"metal": 30, "plastic": 10, "paper": 10, "other": 10})
        report = self.validate()
        self.assertTrue(report.is_valid)
        self.assertAlmostEqual(report.imbalance_ratio or 0, 3.0)
        self.assertTrue(any("class imbalance: 'metal' has 30" in w for w in report.warnings))
        self.assertIn("above limit", format_report(report))

    def test_imbalance_within_limit_does_not_warn(self) -> None:
        make_dataset(self.dataset, {"metal": 15, "plastic": 10, "paper": 10, "other": 10})
        report = self.validate()
        self.assertFalse(any("imbalance" in w for w in report.warnings))

    def test_custom_limit(self) -> None:
        make_dataset(self.dataset, {"metal": 15, "plastic": 10, "paper": 10, "other": 10})
        report = self.validate(max_imbalance_ratio=1.2)
        self.assertTrue(any("class imbalance" in w for w in report.warnings))

    def test_imbalance_not_computed_with_fewer_than_two_populated_classes(self) -> None:
        make_dataset(self.dataset, {"metal": 5})
        report = self.validate()
        self.assertIsNone(report.imbalance_ratio)
        self.assertIn("n/a", format_report(report))

    def test_min_recommended_images_warns(self) -> None:
        make_dataset(self.dataset, FOUR_EACH)
        report = self.validate(min_recommended_images=10)
        self.assertEqual(sum("only 4 valid image(s)" in w for w in report.warnings), 4)

    def test_invalid_parameters_raise(self) -> None:
        with self.assertRaises(ValueError):
            self.validate(max_imbalance_ratio=0.5)
        with self.assertRaises(ValueError):
            self.validate(min_recommended_images=-1)


class CommandLineTests(DatasetTestCase):
    def make_project(self) -> Path:
        (self.root / "config").mkdir()
        config_path = self.root / "config" / "config.yaml"
        shutil.copy(DEFAULT_CONFIG_PATH, config_path)
        self.dataset = self.root / "data" / "dataset"
        return config_path

    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(list(args))
        return code, out.getvalue()

    def test_exit_code_0_for_valid_dataset(self) -> None:
        config_path = self.make_project()
        make_dataset(self.dataset, FOUR_EACH)
        code, out = self.run_main("--config", str(config_path), "--min-images", "1")
        self.assertEqual(code, 0, out)
        self.assertIn("Result: VALID", out)

    def test_exit_code_1_for_invalid_dataset(self) -> None:
        config_path = self.make_project()
        make_dataset(self.dataset, {"metal": 4})
        code, out = self.run_main("--config", str(config_path))
        self.assertEqual(code, 1)
        self.assertIn("Result: INVALID", out)

    def test_exit_code_2_for_bad_config_or_arguments(self) -> None:
        code, out = self.run_main("--config", str(self.root / "missing.yaml"))
        self.assertEqual(code, 2)
        self.assertIn("config file not found", out)
        config_path = self.make_project()
        code, out = self.run_main("--config", str(config_path), "--max-imbalance", "0.5")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
