"""Tests for training on all images with no validation split. Tiny images, CPU, random weights."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from src import train_all as ta
from src.config import DEFAULT_CONFIG_PATH, Config, load_config
from src.model import load_checkpoint
from src.predict import load_model_for_inference
from src.train import TrainingError

CLASSES = ("metal", "plastic", "paper", "other")
PER_CLASS = 6


class TrainAllTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["image"] = {"width": 32, "height": 32}
        raw["training"].update(batch_size=4, epochs=2, num_workers=0, device="cpu", pretrained=False)
        raw["paths"] = {"dataset_dir": str(self.root / "dataset"), "model_dir": str(self.root / "models")}
        raw["dataset"] = {
            "group_metadata": str(self.root / "group_metadata.csv"),
            "manifest": str(self.root / "splits" / "manifest.csv"),
            "report": str(self.root / "splits" / "split_report.txt"),
        }
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        self.config: Config = load_config(self.config_path)
        self.make_dataset()

    def make_dataset(self, per_class: int = PER_CLASS) -> None:
        rng = np.random.default_rng(0)
        for name in CLASSES:
            folder = self.config.paths.dataset_dir / name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                img = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(folder / f"synth_{i:04d}.jpg"), img)


class CollectRowsTests(TrainAllTestCase):
    def test_every_image_becomes_one_training_row(self) -> None:
        rows = ta.collect_rows(self.config)
        self.assertEqual(len(rows), PER_CLASS * 4)
        self.assertEqual({r.split for r in rows}, {"train"})
        self.assertEqual(sum(1 for r in rows if r.material_class == "other"), PER_CLASS)

    def test_paths_are_relative_to_the_dataset_folder(self) -> None:
        rows = ta.collect_rows(self.config)
        self.assertIn("metal/synth_0000.jpg", {r.relative_path for r in rows})

    def test_non_image_files_are_ignored(self) -> None:
        (self.config.paths.dataset_dir / "metal" / "notes.txt").write_text("x", encoding="utf-8")
        self.assertEqual(len(ta.collect_rows(self.config)), PER_CLASS * 4)

    def test_a_class_without_images_is_refused(self) -> None:
        for path in (self.config.paths.dataset_dir / "paper").glob("*.jpg"):
            path.unlink()
        with self.assertRaisesRegex(TrainingError, "no images in .*paper"):
            ta.collect_rows(self.config)

    def test_missing_dataset_folder_is_refused(self) -> None:
        for path in self.config.paths.dataset_dir.rglob("*.jpg"):
            path.unlink()
        with self.assertRaisesRegex(TrainingError, "dataset folder not found|no images"):
            ta.collect_rows(self.config)

    def test_no_manifest_or_split_file_is_needed(self) -> None:
        self.assertFalse(self.config.dataset.manifest.exists())
        rows = ta.collect_rows(self.config)
        self.assertTrue(rows)


class TrainAllRunTests(TrainAllTestCase):
    def run_train(self, **kwargs) -> ta.TrainAllResult:
        log: list[str] = []
        kwargs.setdefault("epochs", 2)
        kwargs.setdefault("device_setting", "cpu")
        kwargs.setdefault("pretrained", False)
        return ta.train_all(self.config, self.config.paths.model_dir, log=log.append, **kwargs)

    def test_saves_a_checkpoint_that_predict_and_the_sorter_can_load(self) -> None:
        self.run_train()
        checkpoint = self.config.paths.model_dir / "best_model.pth"
        self.assertTrue(checkpoint.is_file())
        model, metadata = load_checkpoint(checkpoint, classes=CLASSES, device="cpu")
        self.assertEqual(metadata["classes"], list(CLASSES))
        loaded = load_model_for_inference(self.config, checkpoint, device_setting="cpu")
        self.assertEqual(loaded.metadata["architecture"], "mobilenet_v3_small")

    def test_saved_model_is_the_last_epoch(self) -> None:
        self.run_train(epochs=3)
        _, metadata = load_checkpoint(self.config.paths.model_dir / "best_model.pth", classes=CLASSES, device="cpu")
        self.assertEqual(metadata["epoch"], 3)
        self.assertEqual(metadata["trained_on"], "all images, no validation split")

    def test_history_has_one_row_per_epoch_without_validation_values(self) -> None:
        self.run_train(epochs=2)
        lines = (self.config.paths.model_dir / "training_history.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1 + 2)
        header = lines[0].split(",")
        self.assertIn("train_loss", header)
        self.assertEqual(lines[1].split(",")[header.index("val_loss")], "")

    def test_summary_states_that_nothing_was_held_out(self) -> None:
        self.run_train(epochs=1)
        summary = json.loads((self.config.paths.model_dir / "training_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["mode"], "train_all")
        self.assertEqual(summary["total_images"], PER_CLASS * 4)
        self.assertEqual(summary["images_per_class"], {"metal": PER_CLASS, "plastic": PER_CLASS, "paper": PER_CLASS, "other": PER_CLASS})
        self.assertTrue(any("no validation or test split" in note for note in summary["notes"]))

    def test_result_reports_images_and_training_numbers(self) -> None:
        result = self.run_train(epochs=1)
        self.assertEqual(result.images, PER_CLASS * 4)
        self.assertTrue(0.0 <= result.final_train_accuracy <= 1.0)
        self.assertTrue(np.isfinite(result.final_train_loss))

    def test_zero_epochs_is_refused(self) -> None:
        with self.assertRaisesRegex(TrainingError, "epochs must be at least 1"):
            self.run_train(epochs=0)

    def test_existing_outputs_are_not_replaced_without_overwrite(self) -> None:
        self.run_train(epochs=1, stamp="first")
        with self.assertRaisesRegex(TrainingError, "refusing to overwrite"):
            self.run_train(epochs=1, stamp="second")

    def test_overwrite_keeps_the_old_files_as_backups(self) -> None:
        self.run_train(epochs=1, stamp="first")
        self.run_train(epochs=1, overwrite=True, stamp="second")
        models = self.config.paths.model_dir
        self.assertTrue((models / "best_model.pth.second.bak").is_file())
        self.assertTrue((models / "best_model.pth").is_file())

    def test_training_is_reproducible_with_the_same_seed(self) -> None:
        self.run_train(epochs=1, stamp="a")
        first = load_checkpoint(self.config.paths.model_dir / "best_model.pth", classes=CLASSES, device="cpu")[0].state_dict()
        first_values = {k: v.clone() for k, v in first.items()}
        self.run_train(epochs=1, overwrite=True, stamp="b")
        second = load_checkpoint(self.config.paths.model_dir / "best_model.pth", classes=CLASSES, device="cpu")[0].state_dict()
        for key in first_values:
            self.assertTrue(np.allclose(first_values[key].numpy(), second[key].numpy(), atol=1e-6), key)


class CommandLineTests(TrainAllTestCase):
    def run_cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ta.main(["--config", str(self.config_path), "--device", "cpu", "--no-pretrained", *argv])
        return code, out.getvalue()

    def test_success_exits_zero_and_says_no_split_was_used(self) -> None:
        code, out = self.run_cli("--epochs", "1")
        self.assertEqual(code, 0)
        self.assertIn("No validation or test split was used", out)
        self.assertIn("not a test score", out)

    def test_bad_config_exits_2(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ta.main(["--config", str(self.root / "missing.yaml")])
        self.assertEqual(code, 2)

    def test_zero_epochs_exits_2(self) -> None:
        code, out = self.run_cli("--epochs", "0")
        self.assertEqual(code, 2)
        self.assertIn("--epochs must be at least 1", out)

    def test_missing_class_images_exit_1(self) -> None:
        for path in (self.config.paths.dataset_dir / "plastic").glob("*.jpg"):
            path.unlink()
        code, out = self.run_cli("--epochs", "1")
        self.assertEqual(code, 1)
        self.assertIn("no images in", out)


class SyntheticEndToEndTests(TrainAllTestCase):
    def test_images_from_the_synthesis_step_train_without_a_split(self) -> None:
        """The synthesis writes a group file and class folders; training must need neither a split nor the group file."""
        import shutil

        from src.synthesize_dataset import SynthesisSettings, find_templates, generate_images

        shutil.rmtree(self.config.paths.dataset_dir)  # start from an empty dataset, as the synthesis step would

        template_dir = self.root / "template"
        template_dir.mkdir()
        background = np.full((120, 160, 3), (120, 135, 150), np.uint8)
        cv2.imwrite(str(template_dir / "empty.jpg"), background)
        for name in ("Metal", "Plastic", "Paper", "Organic"):
            photo = background.copy()
            photo[40:70, 50:90] = (40, 60, 220)
            cv2.imwrite(str(template_dir / f"{name}.jpg"), photo)
        templates = find_templates(template_dir, CLASSES)
        generate_images(templates, SynthesisSettings(per_template=3, seed=5), self.config.paths.dataset_dir)
        result = ta.train_all(self.config, self.config.paths.model_dir, epochs=1, device_setting="cpu", pretrained=False, log=lambda _: None)
        self.assertEqual(result.images, 12)


if __name__ == "__main__":
    unittest.main()
