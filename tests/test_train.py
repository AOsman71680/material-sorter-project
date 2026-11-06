"""Tests for src.model and src.train.

Everything runs on the CPU with small synthetic images (32x32, 2 epochs), and
no test downloads pretrained weights. Run with:
    python -m unittest discover -s tests -v
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
from unittest import mock

import numpy as np
import torch
import yaml

from helpers import make_dataset
from src.config import DEFAULT_CONFIG_PATH, MATERIAL_CLASSES, Config, ImageConfig, load_config
from src.dataset import read_manifest, validate_dataset_from_config
from src.model import ModelError, build_model, load_checkpoint, save_checkpoint
from src.prepare_dataset import plan_split, write_outputs
from src.train import (
    BEST_MODEL_NAME,
    HISTORY_NAME,
    LAST_MODEL_NAME,
    SUMMARY_NAME,
    EarlyStopper,
    ManifestImageDataset,
    TrainingError,
    load_training_rows,
    main,
    resolve_device,
    run_training,
)

CLASSES = MATERIAL_CLASSES
SMALL = ImageConfig(width=32, height=32)


class ProjectTestCase(unittest.TestCase):
    """A temporary project with a config, a dataset, and a Task 3 manifest."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "config").mkdir()
        self.config_path = self.root / "config" / "config.yaml"
        shutil.copy(DEFAULT_CONFIG_PATH, self.config_path)
        self.base: Config = load_config(self.config_path)
        make_dataset(self.base.paths.dataset_dir, {c: 10 for c in CLASSES})
        report = validate_dataset_from_config(self.base, min_recommended_images=0)
        write_outputs(plan_split(self.base, report), self.base, "report", overwrite=False)
        self.config = replace(
            self.base,
            image=SMALL,
            training=replace(
                self.base.training,
                epochs=2,
                batch_size=8,
                pretrained=False,
                device="cpu",
                num_workers=0,
                early_stopping_patience=5,
            ),
        )
        self.output = self.root / "models"

    def run_quiet(self, config: Config | None = None, output: Path | None = None, **kwargs) -> object:  # type: ignore[no-untyped-def]
        return run_training(config or self.config, output or self.output, log=lambda _m: None, **kwargs)

    def rewrite_manifest(self, mutate) -> None:  # type: ignore[no-untyped-def]
        path = self.config.dataset.manifest
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        mutate(rows)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


class ModelBuildTests(unittest.TestCase):
    def test_every_architecture_outputs_one_logit_per_class(self) -> None:
        for arch in ("mobilenet_v3_small", "mobilenet_v2", "efficientnet_b0", "resnet18"):
            with self.subTest(architecture=arch):
                model = build_model(arch, 4, pretrained=False).eval()
                with torch.no_grad():
                    out = model(torch.zeros(2, 3, 32, 32))
                self.assertEqual(tuple(out.shape), (2, 4))

    def test_default_architecture_is_mobilenet_v3_small_with_four_outputs(self) -> None:
        model = build_model("mobilenet_v3_small", 4, pretrained=False).eval()
        self.assertEqual(model.classifier[-1].out_features, 4)  # type: ignore[attr-defined]

    def test_unsupported_architecture_and_too_few_classes(self) -> None:
        with self.assertRaisesRegex(ModelError, "unsupported architecture"):
            build_model("vgg16", 4, pretrained=False)
        with self.assertRaisesRegex(ModelError, "at least 2 classes"):
            build_model("mobilenet_v3_small", 1, pretrained=False)

    def test_pretrained_false_never_downloads_weights(self) -> None:
        with mock.patch("torchvision.models._api.WeightsEnum.get_state_dict") as fetch:
            build_model("mobilenet_v3_small", 4, pretrained=False)
        fetch.assert_not_called()

    def test_pretrained_true_requests_imagenet_weights(self) -> None:
        # The download is stubbed to fail, so the test shows the request was made without using the network.
        with mock.patch(
            "torchvision.models._api.WeightsEnum.get_state_dict",
            side_effect=RuntimeError("network blocked in tests"),
        ) as fetch:
            with self.assertRaisesRegex(RuntimeError, "network blocked in tests"):
                build_model("mobilenet_v3_small", 4, pretrained=True)
        fetch.assert_called_once()


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.metadata = {
            "architecture": "mobilenet_v3_small",
            "classes": list(CLASSES),
            "image_width": 32,
            "image_height": 32,
            "normalization": "imagenet",
            "pretrained": False,
            "seed": 42,
            "epoch": 3,
            "val_loss": 0.5,
            "val_accuracy": 0.75,
        }

    def test_round_trip_restores_weights_and_metadata(self) -> None:
        torch.manual_seed(0)
        model = build_model("mobilenet_v3_small", 4, pretrained=False)
        path = self.dir / BEST_MODEL_NAME
        save_checkpoint(path, model, self.metadata)
        loaded, meta = load_checkpoint(path, classes=CLASSES)
        for (name, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
            self.assertTrue(torch.equal(a, b), name)
        self.assertFalse(loaded.training)  # loaded in eval mode
        self.assertEqual(meta["epoch"], 3)
        self.assertEqual(meta["classes"], list(CLASSES))
        self.assertNotIn("state_dict", meta)
        self.assertFalse((self.dir / (BEST_MODEL_NAME + ".tmp")).exists())

    def test_class_order_mismatch_is_rejected(self) -> None:
        model = build_model("mobilenet_v3_small", 4, pretrained=False)
        path = self.dir / BEST_MODEL_NAME
        save_checkpoint(path, model, self.metadata)
        with self.assertRaisesRegex(ModelError, "do not match"):
            load_checkpoint(path, classes=("paper", "metal", "plastic", "other"))

    def test_missing_corrupt_and_foreign_files_are_rejected(self) -> None:
        with self.assertRaisesRegex(ModelError, "not found"):
            load_checkpoint(self.dir / "absent.pth", classes=CLASSES)
        corrupt = self.dir / "corrupt.pth"
        corrupt.write_bytes(b"garbage")
        with self.assertRaisesRegex(ModelError, "cannot read checkpoint"):
            load_checkpoint(corrupt, classes=CLASSES)
        foreign = self.dir / "foreign.pth"
        torch.save({"just": "a dict"}, foreign)
        with self.assertRaisesRegex(ModelError, "missing key"):
            load_checkpoint(foreign, classes=CLASSES)


class EarlyStoppingTests(unittest.TestCase):
    def test_stops_after_patience_epochs_without_improvement(self) -> None:
        stopper = EarlyStopper(patience=2)
        losses = [1.0, 0.9, 0.95, 0.96, 0.97]
        improved = []
        stop_at = None
        for epoch, value in enumerate(losses, start=1):
            improved.append(stopper.update(value, epoch))
            if stopper.should_stop and stop_at is None:
                stop_at = epoch
        self.assertEqual(improved, [True, True, False, False, False])
        self.assertEqual(stop_at, 4)
        self.assertEqual(stopper.best, 0.9)
        self.assertEqual(stopper.best_epoch, 2)

    def test_nan_never_counts_as_improvement(self) -> None:
        stopper = EarlyStopper(patience=1)
        self.assertTrue(stopper.update(1.0, 1))
        self.assertFalse(stopper.update(float("nan"), 2))
        self.assertTrue(stopper.should_stop)


class ManifestSplitHandlingTests(ProjectTestCase):
    def test_only_train_and_val_rows_are_loaded(self) -> None:
        rows = load_training_rows(self.config)
        self.assertEqual(set(rows), {"train", "val"})
        train_paths = {r.relative_path for r in rows["train"]}
        val_paths = {r.relative_path for r in rows["val"]}
        all_rows = read_manifest(self.config.dataset.manifest, CLASSES)
        test_paths = {r.relative_path for r in all_rows if r.split == "test"}
        self.assertTrue(test_paths)  # the test split exists in the manifest...
        self.assertFalse(test_paths & (train_paths | val_paths))  # ...and is never loaded
        self.assertFalse(train_paths & val_paths)

    def test_training_dataset_contains_only_train_images(self) -> None:
        rows = load_training_rows(self.config)
        dataset = ManifestImageDataset(
            rows["train"],
            dataset_dir=self.config.paths.dataset_dir,
            classes=CLASSES,
            image=SMALL,
            augmentation=self.config.augmentation,
            training=True,
            seed=42,
        )
        self.assertEqual(len(dataset), len(rows["train"]))

    def test_missing_manifest_is_reported(self) -> None:
        self.config.dataset.manifest.unlink()
        with self.assertRaisesRegex(TrainingError, "prepare_dataset"):
            load_training_rows(self.config)

    def test_empty_val_split_is_reported(self) -> None:
        self.rewrite_manifest(lambda rows: [r.update(split="train") for r in rows if r["split"] == "val"])
        with self.assertRaisesRegex(TrainingError, "no 'val' images"):
            load_training_rows(self.config)

    def test_class_missing_from_a_split_is_reported(self) -> None:
        self.rewrite_manifest(
            lambda rows: [r.update(split="test") for r in rows if r["split"] == "val" and r["material_class"] == "paper"]
        )
        with self.assertRaisesRegex(TrainingError, "no images of class"):
            load_training_rows(self.config)

    def test_group_shared_between_train_and_val_is_rejected(self) -> None:
        def leak(rows):  # type: ignore[no-untyped-def]
            val_row = next(r for r in rows if r["split"] == "val")
            train_row = next(r for r in rows if r["split"] == "train")
            train_row["group_id"] = val_row["group_id"]

        self.rewrite_manifest(leak)
        with self.assertRaisesRegex(TrainingError, "appear in both train and val"):
            load_training_rows(self.config)

    def test_missing_image_file_is_reported(self) -> None:
        rows = read_manifest(self.config.dataset.manifest, CLASSES)
        victim = next(r for r in rows if r.split == "train")
        (self.config.paths.dataset_dir / victim.relative_path).unlink()
        with self.assertRaisesRegex(TrainingError, "missing"):
            load_training_rows(self.config)

    def test_manifest_written_for_other_classes_is_rejected(self) -> None:
        json_path = self.config.dataset.manifest.with_name("manifest.json")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        data["classes"] = ["metal", "plastic", "paper", "glass"]
        json_path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(TrainingError, "was created for classes"):
            load_training_rows(self.config)


class AugmentationRoutingTests(ProjectTestCase):
    def test_training_items_are_augmented_and_validation_items_are_not(self) -> None:
        rows = load_training_rows(self.config)
        common = dict(
            dataset_dir=self.config.paths.dataset_dir,
            classes=CLASSES,
            image=SMALL,
            augmentation=self.config.augmentation,
            seed=42,
        )
        train_ds = ManifestImageDataset(rows["train"], training=True, **common)
        val_ds = ManifestImageDataset(rows["val"], training=False, **common)

        train_ds.epoch = 1
        first, _ = train_ds[0]
        again, _ = train_ds[0]
        train_ds.epoch = 2
        next_epoch, _ = train_ds[0]
        self.assertTrue(torch.equal(first, again))  # same epoch and index: same augmentation
        self.assertFalse(torch.equal(first, next_epoch))  # a new epoch draws new augmentation

        v1, _ = val_ds[0]
        v2, _ = val_ds[0]
        self.assertTrue(torch.equal(v1, v2))  # evaluation is deterministic

    def test_validation_images_match_eval_preprocessing_exactly(self) -> None:
        from src.preprocessing import load_rgb, preprocess_eval
        from src.dataset import image_path_for

        rows = load_training_rows(self.config)
        dataset = ManifestImageDataset(
            rows["val"],
            dataset_dir=self.config.paths.dataset_dir,
            classes=CLASSES,
            image=SMALL,
            augmentation=self.config.augmentation,
            training=False,
            seed=42,
        )
        row = rows["val"][0]
        expected = preprocess_eval(load_rgb(image_path_for(self.config.paths.dataset_dir, row.relative_path)), SMALL)
        tensor, label = dataset[0]
        np.testing.assert_allclose(tensor.numpy(), expected, rtol=0, atol=0)
        self.assertEqual(label, CLASSES.index(row.material_class))


class DeviceTests(unittest.TestCase):
    def test_cuda_request_without_gpu_is_a_clear_error(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(TrainingError, "CUDA was requested but is not available"):
                resolve_device("cuda")
            self.assertEqual(resolve_device("auto").type, "cpu")
        self.assertEqual(resolve_device("cpu").type, "cpu")


class ShortCpuTrainingRunTests(ProjectTestCase):
    def test_run_writes_all_four_outputs_and_a_loadable_best_model(self) -> None:
        result = self.run_quiet()
        for name in (BEST_MODEL_NAME, LAST_MODEL_NAME, HISTORY_NAME, SUMMARY_NAME):
            self.assertTrue((self.output / name).is_file(), name)
        self.assertEqual(result.epochs_run, 2)  # type: ignore[attr-defined]
        with (self.output / HISTORY_NAME).open(encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        self.assertEqual([row["epoch"] for row in history], ["1", "2"])
        model, meta = load_checkpoint(self.output / BEST_MODEL_NAME, classes=CLASSES)
        self.assertEqual(meta["classes"], list(CLASSES))
        self.assertEqual(tuple(model(torch.zeros(1, 3, 32, 32)).shape), (1, 4))

    def test_best_checkpoint_is_the_lowest_validation_loss(self) -> None:
        self.run_quiet()
        with (self.output / HISTORY_NAME).open(encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        best_row = min(history, key=lambda r: float(r["val_loss"]))
        flagged = [r for r in history if r["is_best"] == "True"]
        # is_best marks every epoch that set a new best, so the last flag must be the overall minimum.
        self.assertTrue(flagged)
        self.assertEqual(flagged[-1]["epoch"], best_row["epoch"])
        _, meta = load_checkpoint(self.output / BEST_MODEL_NAME, classes=CLASSES)
        self.assertEqual(meta["epoch"], int(best_row["epoch"]))

    def test_summary_records_settings_and_that_test_split_was_not_used(self) -> None:
        self.run_quiet()
        summary = json.loads((self.output / SUMMARY_NAME).read_text(encoding="utf-8"))
        self.assertEqual(summary["architecture"], "mobilenet_v3_small")
        self.assertEqual(summary["classes"], list(CLASSES))
        self.assertFalse(summary["test_split_used"])
        self.assertEqual(summary["device"], "cpu")
        self.assertEqual(summary["seed"], 42)
        self.assertEqual(summary["epochs_run"], 2)
        self.assertEqual(set(summary["manifest_counts"]), {"train", "val"})
        self.assertNotIn(str(self.root), (self.output / SUMMARY_NAME).read_text(encoding="utf-8"))

    def test_same_seed_gives_identical_history(self) -> None:
        first = self.root / "run_a"
        second = self.root / "run_b"
        self.run_quiet(output=first)
        self.run_quiet(output=second)
        self.assertEqual(self._losses(first), self._losses(second))

    @staticmethod
    def _losses(directory: Path) -> list[tuple[str, str]]:
        with (directory / HISTORY_NAME).open(encoding="utf-8") as handle:
            return [(r["train_loss"], r["val_loss"]) for r in csv.DictReader(handle)]

    def test_early_stopping_ends_the_run(self) -> None:
        config = replace(self.config, training=replace(self.config.training, epochs=6, early_stopping_patience=1))
        with mock.patch("src.train.evaluate", return_value=(1.0, 0.5)):
            result = self.run_quiet(config=config)
        self.assertEqual(result.epochs_run, 2)  # type: ignore[attr-defined]
        self.assertTrue(result.stopped_early)  # type: ignore[attr-defined]
        summary = json.loads((self.output / SUMMARY_NAME).read_text(encoding="utf-8"))
        self.assertTrue(summary["stopped_early"])
        self.assertTrue((self.output / LAST_MODEL_NAME).is_file())

    def test_non_finite_loss_stops_with_a_clear_error(self) -> None:
        with mock.patch("src.train.evaluate", return_value=(float("nan"), 0.0)):
            with self.assertRaisesRegex(TrainingError, "NaN or infinite"):
                self.run_quiet()

    def test_existing_outputs_are_not_overwritten_silently(self) -> None:
        self.run_quiet()
        before = (self.output / BEST_MODEL_NAME).read_bytes()
        with self.assertRaisesRegex(TrainingError, "refusing to overwrite"):
            self.run_quiet()
        self.assertEqual((self.output / BEST_MODEL_NAME).read_bytes(), before)

    def test_overwrite_keeps_backups_of_old_outputs(self) -> None:
        self.run_quiet()
        old_summary = (self.output / SUMMARY_NAME).read_bytes()
        self.run_quiet(overwrite=True, stamp="20260101_000000")
        backup = self.output / f"{SUMMARY_NAME}.20260101_000000.bak"
        self.assertEqual(backup.read_bytes(), old_summary)
        self.assertTrue((self.output / SUMMARY_NAME).is_file())

    def test_pretrained_true_is_not_attempted_in_tests(self) -> None:
        config = replace(self.config, training=replace(self.config.training, pretrained=True, epochs=1))
        with mock.patch(
            "torchvision.models._api.WeightsEnum.get_state_dict",
            side_effect=RuntimeError("network blocked in tests"),
        ):
            with self.assertRaisesRegex(TrainingError, "could not download the ImageNet weights.*--no-pretrained"):
                self.run_quiet(config=config)


class CommandLineTests(ProjectTestCase):
    def write_config(self, **training_changes) -> Path:  # type: ignore[no-untyped-def]
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["image"] = {"width": 32, "height": 32}
        raw["training"].update(epochs=2, batch_size=8, pretrained=False, device="cpu", num_workers=0)
        raw["training"].update(training_changes)
        raw["paths"] = {"dataset_dir": "data/dataset", "model_dir": "models"}
        self.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return self.config_path

    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(list(args))
        return code, out.getvalue()

    def test_quick_run_from_command_line_succeeds(self) -> None:
        path = self.write_config()
        code, out = self.run_main("--config", str(path), "--output-dir", str(self.output))
        self.assertEqual(code, 0, out)
        self.assertIn("Outputs written", out)
        self.assertTrue((self.output / BEST_MODEL_NAME).is_file())

    def test_bad_config_and_bad_epochs_return_two(self) -> None:
        code, _ = self.run_main("--config", str(self.root / "missing.yaml"))
        self.assertEqual(code, 2)
        path = self.write_config()
        code, out = self.run_main("--config", str(path), "--epochs", "0")
        self.assertEqual(code, 2)
        self.assertIn("--epochs", out)

    def test_existing_outputs_return_one(self) -> None:
        path = self.write_config()
        self.assertEqual(self.run_main("--config", str(path), "--output-dir", str(self.output))[0], 0)
        code, out = self.run_main("--config", str(path), "--output-dir", str(self.output))
        self.assertEqual(code, 1)
        self.assertIn("--overwrite", out)

    def test_cuda_without_gpu_returns_one(self) -> None:
        path = self.write_config()
        with mock.patch("torch.cuda.is_available", return_value=False):
            code, out = self.run_main("--config", str(path), "--output-dir", str(self.output), "--device", "cuda")
        self.assertEqual(code, 1)
        self.assertIn("CUDA", out)


if __name__ == "__main__":
    unittest.main()
