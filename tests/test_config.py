"""Tests for src.config.

Written with unittest so they run with either:
    python -m unittest discover -s tests -v
    python -m pytest
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from src.config import (
    DEFAULT_CONFIG_PATH,
    MATERIAL_CLASSES,
    Config,
    ConfigError,
    load_config,
    parse_config,
)

BASE_DIR = Path(tempfile.gettempdir()) / "sorter_test_root"

VALID: dict[str, Any] = {
    "classes": ["metal", "plastic", "paper", "other"],
    "cups": {"metal": 1, "plastic": 2, "paper": 3, "other": 4},
    "paths": {"dataset_dir": "data/dataset", "model_dir": "models"},
    "dataset": {
        "group_metadata": "data/group_metadata.csv",
        "manifest": "data/splits/manifest.csv",
        "report": "data/splits/split_report.txt",
    },
    "image": {"width": 224, "height": 224},
    "augmentation": {
        "max_rotation_degrees": 10.0,
        "max_translation": 0.05,
        "min_scale": 0.9,
        "max_scale": 1.1,
        "brightness": 0.15,
        "contrast": 0.15,
    },
    "split": {"train": 0.7, "val": 0.15, "test": 0.15},
    "random_seed": 42,
    "training": {
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "early_stopping_patience": 5,
        "num_workers": 0,
        "device": "auto",
        "pretrained": True,
    },
    "model": {"architecture": "mobilenet_v2"},
    "inference": {"confidence_threshold": 0.7},
    "camera": {"index": 0},
}


def make_raw() -> dict[str, Any]:
    return copy.deepcopy(VALID)


def set_in(raw: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    *parents, leaf = dotted.split(".")
    target = raw
    for part in parents:
        target = target[part]
    target[leaf] = value
    return raw


def delete_in(raw: dict[str, Any], dotted: str) -> dict[str, Any]:
    *parents, leaf = dotted.split(".")
    target = raw
    for part in parents:
        target = target[part]
    del target[leaf]
    return raw


class ValidConfigTests(unittest.TestCase):
    def test_valid_dict_parses(self) -> None:
        config = parse_config(make_raw(), BASE_DIR)
        self.assertIsInstance(config, Config)
        self.assertEqual(config.classes, MATERIAL_CLASSES)
        self.assertEqual(config.cups, {"metal": 1, "plastic": 2, "paper": 3, "other": 4})
        self.assertEqual((config.image.width, config.image.height), (224, 224))
        self.assertEqual(config.training.batch_size, 32)
        self.assertEqual(config.training.epochs, 30)
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.camera_index, 0)
        self.assertEqual(config.model.architecture, "mobilenet_v2")

    def test_project_config_file_loads_with_default_mapping(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(config.cup_for("metal"), 1)
        self.assertEqual(config.cup_for("plastic"), 2)
        self.assertEqual(config.cup_for("paper"), 3)
        self.assertEqual(config.cup_for("other"), 4)
        self.assertNotIn("uncertain", config.classes)

    def test_load_config_without_arguments_uses_project_config(self) -> None:
        self.assertEqual(load_config().classes, MATERIAL_CLASSES)

    def test_custom_cup_mapping_is_allowed(self) -> None:
        raw = make_raw()
        raw["cups"] = {"metal": 4, "plastic": 3, "paper": 2, "other": 1}
        config = parse_config(raw, BASE_DIR)
        self.assertEqual(config.cup_for("metal"), 4)
        self.assertEqual(config.cup_for("other"), 1)

    def test_cup_ids_do_not_have_to_be_1_to_4(self) -> None:
        raw = make_raw()
        raw["cups"] = {"metal": 10, "plastic": 20, "paper": 30, "other": 40}
        self.assertEqual(parse_config(raw, BASE_DIR).cup_for("paper"), 30)

    def test_relative_paths_resolve_against_base_dir(self) -> None:
        config = parse_config(make_raw(), BASE_DIR)
        self.assertEqual(config.paths.dataset_dir, BASE_DIR / "data" / "dataset")
        self.assertEqual(config.paths.model_dir, BASE_DIR / "models")

    def test_absolute_paths_are_kept(self) -> None:
        absolute = (BASE_DIR / "elsewhere").resolve()
        raw = set_in(make_raw(), "paths.dataset_dir", str(absolute))
        self.assertEqual(parse_config(raw, BASE_DIR).paths.dataset_dir, absolute)

    def test_paths_do_not_need_to_exist(self) -> None:
        config = parse_config(make_raw(), Path(tempfile.gettempdir()) / "definitely_missing_xyz")
        self.assertFalse(config.paths.dataset_dir.exists())

    def test_alternate_valid_split_accepted(self) -> None:
        raw = set_in(make_raw(), "split", {"train": 0.6, "val": 0.2, "test": 0.2})
        self.assertAlmostEqual(parse_config(raw, BASE_DIR).split.train, 0.6)

    def test_every_supported_architecture_accepted(self) -> None:
        for arch in ("mobilenet_v2", "efficientnet_b0", "resnet18"):
            with self.subTest(arch=arch):
                raw = set_in(make_raw(), "model.architecture", arch)
                self.assertEqual(parse_config(raw, BASE_DIR).model.architecture, arch)

    def test_cup_for_unknown_class_raises(self) -> None:
        config = parse_config(make_raw(), BASE_DIR)
        with self.assertRaisesRegex(ConfigError, "uncertain"):
            config.cup_for("uncertain")


class InvalidConfigTests(unittest.TestCase):
    def assert_invalid(self, raw: Any, fragment: str) -> None:
        with self.assertRaises(ConfigError) as ctx:
            parse_config(raw, BASE_DIR)
        self.assertIn(fragment, str(ctx.exception))

    def test_top_level_must_be_mapping(self) -> None:
        for raw in (None, [], "text", 5):
            with self.subTest(raw=raw):
                self.assert_invalid(raw, "must be a mapping")

    def test_missing_top_level_keys(self) -> None:
        for key in VALID:
            with self.subTest(key=key):
                raw = delete_in(make_raw(), key)
                self.assert_invalid(raw, f"missing required key(s): ['{key}']")

    def test_unknown_top_level_key(self) -> None:
        raw = make_raw()
        raw["confidence_treshold"] = 0.5  # typo must not be ignored
        self.assert_invalid(raw, "unknown key(s): ['confidence_treshold']")

    def test_missing_nested_keys(self) -> None:
        nested = [
            "cups.metal",
            "paths.dataset_dir",
            "paths.model_dir",
            "image.width",
            "image.height",
            "split.train",
            "split.val",
            "split.test",
            "training.batch_size",
            "training.epochs",
            "training.learning_rate",
            "model.architecture",
            "inference.confidence_threshold",
            "camera.index",
        ]
        for dotted in nested:
            with self.subTest(key=dotted):
                with self.assertRaises(ConfigError) as ctx:
                    parse_config(delete_in(make_raw(), dotted), BASE_DIR)
                self.assertIn("missing required key(s)", str(ctx.exception))
                self.assertIn(dotted.split(".")[-1], str(ctx.exception))

    def test_unknown_nested_key(self) -> None:
        raw = set_in(make_raw(), "training.momentum", 0.9)
        self.assert_invalid(raw, "'training' has unknown key(s): ['momentum']")

    # --- classes -----------------------------------------------------------

    def test_missing_class(self) -> None:
        raw = set_in(make_raw(), "classes", ["metal", "plastic", "paper"])
        self.assert_invalid(raw, "missing: ['other']")

    def test_extra_class_including_uncertain(self) -> None:
        for extra in ("uncertain", "glass"):
            with self.subTest(extra=extra):
                raw = set_in(make_raw(), "classes", [*MATERIAL_CLASSES, extra])
                self.assert_invalid(raw, f"unexpected: ['{extra}']")

    def test_renamed_or_wrong_case_class(self) -> None:
        raw = set_in(make_raw(), "classes", ["Metal", "plastic", "paper", "other"])
        self.assert_invalid(raw, "unexpected: ['Metal']")

    def test_duplicate_class(self) -> None:
        raw = set_in(make_raw(), "classes", ["metal", "metal", "paper", "other"])
        self.assert_invalid(raw, "duplicate entries: ['metal']")

    def test_classes_wrong_type(self) -> None:
        for value in ("metal", ["metal", 1, "paper", "other"], None):
            with self.subTest(value=value):
                self.assert_invalid(set_in(make_raw(), "classes", value), "list of strings")

    # --- cups --------------------------------------------------------------

    def test_duplicate_cup_ids(self) -> None:
        raw = set_in(make_raw(), "cups.plastic", 1)
        self.assert_invalid(raw, "both map to cup 1")

    def test_cup_for_extra_class(self) -> None:
        raw = set_in(make_raw(), "cups.uncertain", 5)
        self.assert_invalid(raw, "unknown key(s): ['uncertain']")

    def test_cup_ids_must_be_positive_integers(self) -> None:
        for value in (0, -1, 1.5, "2", True, None):
            with self.subTest(value=value):
                raw = set_in(make_raw(), "cups.plastic", value)
                self.assert_invalid(raw, "cups.plastic")

    def test_cups_must_be_mapping(self) -> None:
        self.assert_invalid(set_in(make_raw(), "cups", [1, 2, 3, 4]), "'cups' must be a mapping")

    # --- paths -------------------------------------------------------------

    def test_empty_or_non_string_paths(self) -> None:
        for value in ("", "   ", None, 5, ["a"]):
            with self.subTest(value=value):
                raw = set_in(make_raw(), "paths.dataset_dir", value)
                self.assert_invalid(raw, "paths.dataset_dir")

    def test_dataset_and_model_dirs_must_be_separate(self) -> None:
        cases = [
            ("data", "data"),
            ("data", "data/models"),
            ("data/dataset", "data"),
        ]
        for dataset_dir, model_dir in cases:
            with self.subTest(dataset_dir=dataset_dir, model_dir=model_dir):
                raw = set_in(make_raw(), "paths", {"dataset_dir": dataset_dir, "model_dir": model_dir})
                self.assert_invalid(raw, "separate")

    # --- image / training / seed / camera -----------------------------------

    def test_invalid_image_dimensions(self) -> None:
        for key in ("image.width", "image.height"):
            for value in (0, -224, 16, 100000, 224.0, "224", True, None):
                with self.subTest(key=key, value=value):
                    self.assert_invalid(set_in(make_raw(), key, value), key)

    def test_invalid_batch_size_and_epochs(self) -> None:
        for key in ("training.batch_size", "training.epochs"):
            for value in (0, -1, 1.5, "10", True, None):
                with self.subTest(key=key, value=value):
                    self.assert_invalid(set_in(make_raw(), key, value), key)

    def test_invalid_learning_rate(self) -> None:
        for value in (0, 0.0, -0.1, 1.5, "0.01", True, None, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assert_invalid(
                    set_in(make_raw(), "training.learning_rate", value), "training.learning_rate"
                )

    def test_invalid_random_seed(self) -> None:
        for value in (-1, 1.5, "42", True, None):
            with self.subTest(value=value):
                self.assert_invalid(set_in(make_raw(), "random_seed", value), "random_seed")

    def test_invalid_camera_index(self) -> None:
        for value in (-1, 0.5, "0", True, None):
            with self.subTest(value=value):
                self.assert_invalid(set_in(make_raw(), "camera.index", value), "camera.index")

    # --- split -------------------------------------------------------------

    def test_split_must_sum_to_one(self) -> None:
        cases = [
            {"train": 0.7, "val": 0.2, "test": 0.2},
            {"train": 0.5, "val": 0.2, "test": 0.2},
            {"train": 0.8, "val": 0.1, "test": 0.0999},
        ]
        for split in cases:
            with self.subTest(split=split):
                self.assert_invalid(set_in(make_raw(), "split", split), "must sum to 1")

    def test_split_ratios_must_be_between_zero_and_one(self) -> None:
        cases = [
            {"train": 1.0, "val": 0.0, "test": 0.0},
            {"train": 1.2, "val": -0.1, "test": -0.1},
            {"train": 0.9, "val": 0.1, "test": 0.0},
            {"train": "0.7", "val": 0.15, "test": 0.15},
        ]
        for split in cases:
            with self.subTest(split=split):
                with self.assertRaises(ConfigError):
                    parse_config(set_in(make_raw(), "split", split), BASE_DIR)

    # --- confidence threshold ----------------------------------------------

    def test_confidence_threshold_must_be_strictly_between_zero_and_one(self) -> None:
        for value in (0, 0.0, 1, 1.0, -0.1, 1.5, "0.7", True, None, float("nan")):
            with self.subTest(value=value):
                raw = set_in(make_raw(), "inference.confidence_threshold", value)
                self.assert_invalid(raw, "inference.confidence_threshold")

    # --- model -------------------------------------------------------------

    def test_unsupported_architecture(self) -> None:
        for value in ("vgg16", "", None, 3):
            with self.subTest(value=value):
                raw = set_in(make_raw(), "model.architecture", value)
                self.assert_invalid(raw, "model.architecture")


class TrainingConfigTests(unittest.TestCase):
    def test_training_values_are_parsed(self) -> None:
        training = parse_config(make_raw(), BASE_DIR).training
        self.assertEqual(training.weight_decay, 0.0001)
        self.assertEqual(training.early_stopping_patience, 5)
        self.assertEqual(training.num_workers, 0)
        self.assertEqual(training.device, "auto")
        self.assertIs(training.pretrained, True)

    def test_new_training_keys_are_required(self) -> None:
        for key in ("weight_decay", "early_stopping_patience", "num_workers", "device", "pretrained"):
            with self.subTest(key=key):
                raw = delete_in(make_raw(), f"training.{key}")
                with self.assertRaisesRegex(ConfigError, "training"):
                    parse_config(raw, BASE_DIR)

    def test_invalid_training_values_are_rejected(self) -> None:
        cases = [
            ("training.pretrained", "yes", "must be true or false"),
            ("training.pretrained", 1, "must be true or false"),
            ("training.device", "tpu", "'training.device' must be one of"),
            ("training.weight_decay", 1.0, "[0, 1)"),
            ("training.weight_decay", -0.1, "[0, 1)"),
            ("training.early_stopping_patience", 0, "must be >= 1"),
            ("training.num_workers", 17, "must be <= 16"),
            ("training.num_workers", -1, "must be >= 0"),
        ]
        for dotted, value, fragment in cases:
            with self.subTest(key=dotted, value=value):
                with self.assertRaises(ConfigError) as ctx:
                    parse_config(set_in(make_raw(), dotted, value), BASE_DIR)
                self.assertIn(fragment, str(ctx.exception))

    def test_training_unknown_key_still_rejected(self) -> None:
        raw = set_in(make_raw(), "training.momentum", 0.9)
        with self.assertRaises(ConfigError) as ctx:
            parse_config(raw, BASE_DIR)
        self.assertIn("unknown key(s): ['momentum']", str(ctx.exception))

    def test_project_config_selects_mobilenet_v3_small(self) -> None:
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config = parse_config(raw, Path(tempfile.gettempdir()) / "sorter_root")
        self.assertEqual(config.model.architecture, "mobilenet_v3_small")
        self.assertIs(config.training.pretrained, True)


class DatasetAndAugmentationConfigTests(unittest.TestCase):
    def test_dataset_and_augmentation_values_are_parsed(self) -> None:
        config = parse_config(make_raw(), BASE_DIR)
        self.assertEqual(config.dataset.group_metadata, BASE_DIR / "data" / "group_metadata.csv")
        self.assertEqual(config.dataset.manifest, BASE_DIR / "data" / "splits" / "manifest.csv")
        self.assertEqual(config.dataset.report, BASE_DIR / "data" / "splits" / "split_report.txt")
        self.assertEqual(config.augmentation.max_rotation_degrees, 10.0)
        self.assertEqual(config.augmentation.max_scale, 1.1)

    def test_dataset_files_must_not_be_inside_dataset_dir(self) -> None:
        for key, value in (
            ("manifest", "data/dataset/manifest.csv"),
            ("report", "data/dataset/report.txt"),
            ("group_metadata", "data/dataset/groups.csv"),
            ("manifest", "data/dataset"),
        ):
            with self.subTest(key=key, value=value):
                raw = set_in(make_raw(), f"dataset.{key}", value)
                self.assert_invalid(raw, "must not be inside 'paths.dataset_dir'")

    def test_manifest_and_report_must_differ(self) -> None:
        raw = set_in(make_raw(), "dataset.report", "data/splits/manifest.csv")
        self.assert_invalid(raw, "must be different files")

    def test_dataset_section_keys_are_strict(self) -> None:
        raw = set_in(make_raw(), "dataset.grup", "x")  # typo of group_metadata
        self.assert_invalid(raw, "unknown key(s): ['grup']")

    def test_augmentation_missing_and_unknown_keys(self) -> None:
        raw = delete_in(make_raw(), "augmentation.contrast")
        self.assert_invalid(raw, "missing required key(s): ['contrast']")
        raw = make_raw()
        raw["augmentation"]["brightnes"] = 0.1  # typo; all real keys still present
        self.assert_invalid(raw, "unknown key(s): ['brightnes']")

    def test_augmentation_ranges_are_enforced(self) -> None:
        cases = [
            ("augmentation.max_rotation_degrees", 50.0, "between 0 and 45"),
            ("augmentation.max_rotation_degrees", -1.0, "between 0 and 45"),
            ("augmentation.max_translation", 0.5, "[0, 0.5)"),
            ("augmentation.min_scale", 0.0, "0 < min_scale <= max_scale"),
            ("augmentation.min_scale", 1.2, "0 < min_scale <= max_scale"),
            ("augmentation.brightness", 1.0, "[0, 1)"),
            ("augmentation.contrast", -0.1, "[0, 1)"),
            ("augmentation.contrast", "0.1", "must be a number"),
        ]
        for dotted, value, fragment in cases:
            with self.subTest(key=dotted, value=value):
                self.assert_invalid(set_in(make_raw(), dotted, value), fragment)

    def test_project_config_has_dataset_and_augmentation_sections(self) -> None:
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config = parse_config(raw, Path(tempfile.gettempdir()) / "sorter_root")
        self.assertEqual(config.dataset.manifest.name, "manifest.csv")
        self.assertEqual(config.augmentation.max_translation, 0.05)

    def assert_invalid(self, raw: Any, fragment: str) -> None:
        with self.assertRaises(ConfigError) as ctx:
            parse_config(raw, BASE_DIR)
        self.assertIn(fragment, str(ctx.exception))


class LoadFromFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "config").mkdir()
        self.config_path = self.root / "config" / "config.yaml"

    def write(self, text: str) -> None:
        self.config_path.write_text(text, encoding="utf-8")

    def test_load_valid_file_resolves_paths_against_project_root(self) -> None:
        self.write(yaml.safe_dump(VALID))
        config = load_config(self.config_path)
        self.assertEqual(config.paths.dataset_dir, self.root.resolve() / "data" / "dataset")

    def test_explicit_base_dir_overrides_default(self) -> None:
        self.write(yaml.safe_dump(VALID))
        other = self.root / "other_base"
        self.assertEqual(
            load_config(self.config_path, base_dir=other).paths.model_dir, other / "models"
        )

    def test_missing_file(self) -> None:
        with self.assertRaisesRegex(ConfigError, "not found"):
            load_config(self.root / "config" / "nope.yaml")

    def test_empty_file(self) -> None:
        self.write("")
        with self.assertRaisesRegex(ConfigError, "must be a mapping"):
            load_config(self.config_path)

    def test_malformed_yaml(self) -> None:
        self.write("classes: [metal, plastic\ncups: {")
        with self.assertRaisesRegex(ConfigError, "invalid YAML"):
            load_config(self.config_path)

    def test_duplicate_yaml_keys_are_rejected(self) -> None:
        text = yaml.safe_dump(VALID) + "random_seed: 7\n"
        self.write(text)
        with self.assertRaisesRegex(ConfigError, "duplicate key 'random_seed'"):
            load_config(self.config_path)

    def test_utf8_bom_is_tolerated(self) -> None:
        self.config_path.write_bytes(b"\xef\xbb\xbf" + yaml.safe_dump(VALID).encode("utf-8"))
        self.assertEqual(load_config(self.config_path).classes, MATERIAL_CLASSES)

    def test_yaml_boolean_for_number_is_rejected(self) -> None:
        text = yaml.safe_dump(VALID).replace("epochs: 30", "epochs: yes")
        self.write(text)
        with self.assertRaisesRegex(ConfigError, "training.epochs"):
            load_config(self.config_path)


if __name__ == "__main__":
    unittest.main()
