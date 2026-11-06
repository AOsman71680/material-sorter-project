"""Tests for src.evaluate (test-split evaluation, metrics, outputs, errors).

Fast and deterministic: 32x32 synthetic images, random untrained models built
without pretrained weights, and no GPU. Run with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from helpers import make_dataset
from src.config import DEFAULT_CONFIG_PATH, MATERIAL_CLASSES, Config, ImageConfig, load_config
from src.dataset import ManifestRow, read_manifest, validate_dataset_from_config
from src.evaluate import (
    CONFUSION_NAME,
    OUTPUT_NAMES,
    PLOT_NAME,
    PREDICTIONS_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
    EvaluationError,
    Inference,
    build_notes,
    check_checkpoint_matches_config,
    compute_metrics,
    load_test_rows,
    main,
    portable_path,
    run_evaluation,
    run_inference,
    write_predictions,
)
from src.model import build_model, load_checkpoint, save_checkpoint
from src.prepare_dataset import plan_split, write_outputs

CLASSES = MATERIAL_CLASSES
SMALL = ImageConfig(width=32, height=32)


# --------------------------------------------------------------------------
# Metrics on small known examples (no files, no model)
# --------------------------------------------------------------------------


class MetricTests(unittest.TestCase):
    def test_confusion_matrix_rows_are_actual_and_columns_predicted(self) -> None:
        labels = np.array([0, 0, 1, 2])  # metal, metal, plastic, paper
        predictions = np.array([0, 1, 1, 2])  # one metal predicted as plastic
        result = compute_metrics(labels, predictions, CLASSES)
        matrix = result["confusion_matrix"]
        self.assertEqual(matrix[0, 1], 1)  # actual metal, predicted plastic
        self.assertEqual(matrix[1, 0], 0)  # actual plastic, predicted metal: none
        self.assertEqual(matrix.sum(), 4)
        self.assertEqual(matrix.shape, (4, 4))

    def test_accuracy_and_per_class_metrics_on_known_example(self) -> None:
        # Correct: 6 of 8. Hand-computed values below.
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        predictions = np.array([0, 1, 1, 1, 2, 0, 3, 3])
        result = compute_metrics(labels, predictions, CLASSES)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertEqual((result["correct"], result["incorrect"], result["test_samples"]), (6, 2, 8))
        metal = result["per_class"]["metal"]
        self.assertAlmostEqual(metal["precision"], 0.5)
        self.assertAlmostEqual(metal["recall"], 0.5)
        self.assertAlmostEqual(metal["f1"], 0.5)
        plastic = result["per_class"]["plastic"]
        self.assertAlmostEqual(plastic["precision"], 2 / 3)
        self.assertAlmostEqual(plastic["recall"], 1.0)
        self.assertAlmostEqual(plastic["f1"], 0.8)
        paper = result["per_class"]["paper"]
        self.assertAlmostEqual(paper["precision"], 1.0)
        self.assertAlmostEqual(paper["recall"], 0.5)
        self.assertAlmostEqual(paper["f1"], 2 / 3)
        self.assertEqual(result["per_class"]["other"]["support"], 2)
        self.assertAlmostEqual(result["macro"]["precision"], (0.5 + 2 / 3 + 1.0 + 1.0) / 4)
        self.assertAlmostEqual(result["macro"]["f1"], (0.5 + 0.8 + 2 / 3 + 1.0) / 4)
        # All classes have support 2, so weighted equals macro here.
        self.assertAlmostEqual(result["weighted"]["recall"], result["macro"]["recall"])

    def test_zero_denominators_give_zero_not_nan(self) -> None:
        # plastic and other never appear in the labels, and plastic is never predicted.
        labels = np.array([0, 0, 2, 2])
        predictions = np.array([0, 0, 0, 2])
        result = compute_metrics(labels, predictions, CLASSES)
        for name in ("plastic", "other"):
            for key in ("precision", "recall", "f1"):
                value = result["per_class"][name][key]
                self.assertEqual(value, 0.0, f"{name} {key}")
                self.assertFalse(math.isnan(value))
        self.assertFalse(any(math.isnan(v) for v in result["macro"].values()))
        self.assertFalse(any(math.isnan(v) for v in result["weighted"].values()))
        notes = " ".join(build_notes(result, CLASSES))
        self.assertIn("No test images of class 'plastic'", notes)
        self.assertIn("No test images of class 'other'", notes)

    def test_class_never_predicted_is_noted(self) -> None:
        labels = np.array([0, 1, 2, 3])
        predictions = np.array([0, 0, 0, 0])
        notes = " ".join(build_notes(compute_metrics(labels, predictions, CLASSES), CLASSES))
        self.assertIn("never predicted class 'plastic'", notes)

    def test_small_support_is_flagged_as_unstable(self) -> None:
        labels = np.array([0, 1, 2, 3])
        result = compute_metrics(labels, labels, CLASSES)
        notes = " ".join(build_notes(result, CLASSES))
        self.assertIn("Only 1 test image(s) of class 'metal' (fewer than 30)", notes)
        self.assertEqual(result["accuracy"], 1.0)

    def test_predictions_csv_maps_indices_to_class_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            rows = [
                ManifestRow("paper/a.jpg", "paper", "test", "g1", "", ""),
                ManifestRow("other/b.jpg", "other", "test", "g2", "", ""),
            ]
            inference = Inference(
                labels=np.array([2, 3]),
                predictions=np.array([2, 0]),
                probabilities=np.array([[0.1, 0.2, 0.6, 0.1], [0.5, 0.2, 0.2, 0.1]]),
            )
            write_predictions(path, rows, inference, CLASSES)
            with path.open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
        self.assertEqual(records[0]["actual_class"], "paper")
        self.assertEqual(records[0]["predicted_class"], "paper")
        self.assertEqual(records[0]["correct"], "true")
        self.assertEqual(records[1]["actual_class"], "other")
        self.assertEqual(records[1]["predicted_class"], "metal")
        self.assertEqual(records[1]["correct"], "false")

    def test_probability_columns_are_named_by_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            rows = [ManifestRow("metal/a.jpg", "metal", "test", "g", "", "")]
            inference = Inference(
                labels=np.array([0]),
                predictions=np.array([1]),
                probabilities=np.array([[0.1, 0.7, 0.1, 0.1]]),
            )
            write_predictions(path, rows, inference, CLASSES)
            header = path.read_text(encoding="utf-8").splitlines()[0]
            values = path.read_text(encoding="utf-8").splitlines()[1].split(",")
        self.assertEqual(header, "relative_path,actual_class,predicted_class,correct,prob_metal,prob_plastic,prob_paper,prob_other")
        self.assertEqual(values[4:], ["0.100000", "0.700000", "0.100000", "0.100000"])


# --------------------------------------------------------------------------
# Project-based tests (synthetic dataset, manifest, checkpoint)
# --------------------------------------------------------------------------


class EvalProjectTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "config").mkdir()
        shutil.copy(DEFAULT_CONFIG_PATH, self.root / "config" / "config.yaml")
        base = load_config(self.root / "config" / "config.yaml")
        make_dataset(base.paths.dataset_dir, {c: 20 for c in CLASSES})
        report = validate_dataset_from_config(base, min_recommended_images=0)
        write_outputs(plan_split(base, report), base, "report", overwrite=False)
        self.config: Config = replace(
            base,
            image=SMALL,
            training=replace(base.training, batch_size=8, num_workers=0, device="cpu", pretrained=False),
        )
        self.manifest = self.config.dataset.manifest
        self.checkpoint = self.root / "models" / "best_model.pth"
        self.output = self.root / "evaluation"

    def make_checkpoint(
        self,
        *,
        favour: str | None = None,
        architecture: str = "mobilenet_v3_small",
        image: ImageConfig = SMALL,
        normalization: str = "imagenet",
        path: Path | None = None,
    ) -> Path:
        """Save a random untrained checkpoint. With ``favour``, its output always prefers that class."""
        torch.manual_seed(0)
        model = build_model(architecture, len(CLASSES), pretrained=False)
        if favour is not None:
            head = model.classifier[-1]  # type: ignore[index]
            with torch.no_grad():
                head.weight.zero_()  # type: ignore[attr-defined]
                head.bias.copy_(torch.tensor([10.0 if c == favour else 0.0 for c in CLASSES]))  # type: ignore[attr-defined]
        metadata = {
            "architecture": architecture,
            "classes": list(CLASSES),
            "image_width": image.width,
            "image_height": image.height,
            "normalization": normalization,
            "pretrained": False,
            "seed": 42,
            "epoch": 1,
            "val_loss": 1.0,
            "val_accuracy": 0.5,
        }
        target = path or self.checkpoint
        save_checkpoint(target, model, metadata)
        return target

    def run_quiet(self, checkpoint: Path | None = None, **kwargs) -> object:  # type: ignore[no-untyped-def]
        return run_evaluation(
            self.config,
            checkpoint or self.checkpoint,
            self.manifest,
            self.output,
            log=lambda _m: None,
            **kwargs,
        )

    def manifest_rows(self, split: str) -> list[ManifestRow]:
        return [r for r in read_manifest(self.manifest, CLASSES) if r.split == split]

    def rewrite_manifest(self, mutate) -> None:  # type: ignore[no-untyped-def]
        with self.manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        mutate(rows)
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


class TestRecordSelectionTests(EvalProjectTestCase):
    def test_only_test_rows_are_selected(self) -> None:
        rows, counts = load_test_rows(self.config, self.manifest)
        self.assertTrue(rows)
        self.assertEqual({r.split for r in rows}, {"test"})
        self.assertEqual(counts["test_rows"], len(rows))
        self.assertEqual(len(rows), len(self.manifest_rows("test")))

    def test_training_and_validation_records_are_excluded(self) -> None:
        rows, _ = load_test_rows(self.config, self.manifest)
        selected = {r.relative_path for r in rows}
        for other in ("train", "val"):
            self.assertFalse(selected & {r.relative_path for r in self.manifest_rows(other)}, other)

    def test_every_test_record_is_evaluated_exactly_once(self) -> None:
        result = self.run_quiet(
            checkpoint=self.make_checkpoint(favour="metal"),
        )
        with (self.output / PREDICTIONS_NAME).open(newline="", encoding="utf-8") as handle:
            predicted_paths = [r["relative_path"] for r in csv.DictReader(handle)]
        expected = [r.relative_path for r in self.manifest_rows("test")]
        self.assertEqual(predicted_paths, expected)  # same records, same order, no duplicates
        self.assertEqual(len(set(predicted_paths)), len(predicted_paths))
        self.assertEqual(result.summary["test_samples"], len(expected))  # type: ignore[attr-defined]

    def test_test_group_shared_with_training_is_rejected(self) -> None:
        def leak(rows):  # type: ignore[no-untyped-def]
            test_row = next(r for r in rows if r["split"] == "test")
            train_row = next(r for r in rows if r["split"] == "train")
            test_row["group_id"] = train_row["group_id"]

        self.rewrite_manifest(leak)
        with self.assertRaisesRegex(EvaluationError, "also appear in train or val"):
            load_test_rows(self.config, self.manifest)

    def test_manifest_without_test_rows_is_rejected(self) -> None:
        self.rewrite_manifest(lambda rows: [r.update(split="train") for r in rows if r["split"] == "test"])
        with self.assertRaisesRegex(EvaluationError, "no 'test' images"):
            load_test_rows(self.config, self.manifest)


class MissingAndInvalidInputTests(EvalProjectTestCase):
    def test_missing_checkpoint_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(EvaluationError, "checkpoint not found"):
            self.run_quiet(checkpoint=self.root / "absent.pth")

    def test_missing_manifest_is_a_clear_error(self) -> None:
        self.manifest.unlink()
        with self.assertRaisesRegex(EvaluationError, "prepare_dataset"):
            self.run_quiet(checkpoint=self.make_checkpoint())

    def test_missing_test_image_stops_the_run(self) -> None:
        victim = self.manifest_rows("test")[0]
        (self.config.paths.dataset_dir / victim.relative_path).unlink()
        with self.assertRaisesRegex(EvaluationError, "missing or unreadable"):
            self.run_quiet(checkpoint=self.make_checkpoint())
        self.assertFalse(self.output.exists())  # no partial results written

    def test_corrupt_checkpoint_is_a_clear_error(self) -> None:
        bad = self.root / "bad.pth"
        bad.write_bytes(b"not a checkpoint")
        with self.assertRaisesRegex(EvaluationError, "cannot read checkpoint"):
            self.run_quiet(checkpoint=bad)

    def test_architecture_image_size_and_normalization_must_match_config(self) -> None:
        cases = [
            ({"architecture": "resnet18"}, "architecture"),
            ({"image": ImageConfig(width=64, height=64)}, "trained at 64x64"),
            ({"normalization": "none"}, "normalization"),
        ]
        for kwargs, fragment in cases:
            with self.subTest(kwargs=kwargs):
                path = self.make_checkpoint(path=self.root / f"mismatch_{fragment.split()[0]}.pth", **kwargs)
                with self.assertRaisesRegex(EvaluationError, fragment):
                    self.run_quiet(checkpoint=path)

    def test_class_order_mismatch_is_rejected(self) -> None:
        model = build_model("mobilenet_v3_small", 4, pretrained=False)
        path = self.root / "reordered.pth"
        save_checkpoint(
            path,
            model,
            {
                "architecture": "mobilenet_v3_small",
                "classes": ["paper", "metal", "plastic", "other"],
                "image_width": 32,
                "image_height": 32,
                "normalization": "imagenet",
            },
        )
        with self.assertRaisesRegex(EvaluationError, "do not match"):
            self.run_quiet(checkpoint=path)

    def test_check_checkpoint_matches_config_accepts_matching_metadata(self) -> None:
        check_checkpoint_matches_config(
            {"architecture": "mobilenet_v3_small", "image_width": 32, "image_height": 32, "normalization": "imagenet"},
            self.config,
        )


class OutputProtectionTests(EvalProjectTestCase):
    def test_existing_results_are_not_overwritten_without_permission(self) -> None:
        checkpoint = self.make_checkpoint()
        self.run_quiet(checkpoint=checkpoint)
        before = (self.output / SUMMARY_NAME).read_bytes()
        with self.assertRaisesRegex(EvaluationError, "refusing to overwrite"):
            self.run_quiet(checkpoint=checkpoint)
        self.assertEqual((self.output / SUMMARY_NAME).read_bytes(), before)

    def test_overwrite_keeps_backups(self) -> None:
        checkpoint = self.make_checkpoint()
        self.run_quiet(checkpoint=checkpoint)
        old = (self.output / REPORT_NAME).read_bytes()
        self.run_quiet(checkpoint=checkpoint, overwrite=True, stamp="20260101_000000")
        self.assertEqual((self.output / f"{REPORT_NAME}.20260101_000000.bak").read_bytes(), old)
        self.assertTrue((self.output / SUMMARY_NAME).is_file())


class InferenceTests(EvalProjectTestCase):
    def test_probabilities_use_softmax_over_classes(self) -> None:
        class FixedLogits(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                row = torch.tensor([2.0, 0.0, 0.0, 0.0])
                return row.expand(x.shape[0], 4)

        rows = self.manifest_rows("test")[:3]
        inference = run_inference(FixedLogits(), rows, self.config, SMALL, torch.device("cpu"), batch_size=2)
        expected_metal = math.exp(2.0) / (math.exp(2.0) + 3)
        np.testing.assert_allclose(inference.probabilities.sum(axis=1), 1.0, rtol=1e-12)
        np.testing.assert_allclose(inference.probabilities[:, 0], expected_metal, rtol=1e-9)
        self.assertTrue((inference.predictions == 0).all())  # metal is index 0

    def test_inference_does_not_change_weights_and_is_repeatable(self) -> None:
        checkpoint = self.make_checkpoint()
        model, _ = load_checkpoint(checkpoint, classes=CLASSES)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rows = self.manifest_rows("test")
        first = run_inference(model, rows, self.config, SMALL, torch.device("cpu"), batch_size=5)
        second = run_inference(model, rows, self.config, SMALL, torch.device("cpu"), batch_size=3)
        np.testing.assert_array_equal(first.probabilities, second.probabilities)  # no randomness
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_portable_path_hides_machine_specific_directories(self) -> None:
        self.assertEqual(portable_path(self.root / "models" / "best_model.pth"), "best_model.pth")


class EndToEndEvaluationTests(EvalProjectTestCase):
    def test_known_expected_results_when_model_always_predicts_metal(self) -> None:
        checkpoint = self.make_checkpoint(favour="metal")
        result = self.run_quiet(checkpoint=checkpoint)
        s = result.summary  # type: ignore[attr-defined]
        # Twenty images per class: 14 train, 3 val, 3 test. Test set = 12 images, 3 per class.
        self.assertEqual(s["test_samples"], 12)
        self.assertEqual(s["correct"], 3)
        self.assertEqual(s["incorrect"], 9)
        self.assertAlmostEqual(s["accuracy"], 0.25)
        self.assertEqual(s["per_class"]["metal"]["recall"], 1.0)
        self.assertAlmostEqual(s["per_class"]["metal"]["precision"], 0.25)
        for name in ("plastic", "paper", "other"):
            self.assertEqual(s["per_class"][name]["recall"], 0.0)
            self.assertEqual(s["per_class"][name]["precision"], 0.0)
        self.assertAlmostEqual(s["macro"]["recall"], 0.25)
        self.assertTrue(s["test_split_only"])
        self.assertFalse(s["model_modified"])

    def test_confusion_csv_matches_the_known_matrix(self) -> None:
        self.run_quiet(checkpoint=self.make_checkpoint(favour="metal"))
        with (self.output / CONFUSION_NAME).open(newline="", encoding="utf-8") as handle:
            table = list(csv.reader(handle))
        self.assertEqual(table[0], ["actual \\ predicted", *CLASSES])
        for i, name in enumerate(CLASSES, start=1):
            self.assertEqual(table[i][0], name)
            # Every image is predicted as metal, so only the metal column is non-zero (3 per actual class).
            self.assertEqual(table[i][1:], ["3" if j == 0 else "0" for j in range(4)])

    def test_artifacts_are_written_and_consistent_with_predictions(self) -> None:
        self.run_quiet(checkpoint=self.make_checkpoint(favour="metal"))
        for name in OUTPUT_NAMES:
            self.assertTrue((self.output / name).is_file(), name)
        plot = (self.output / PLOT_NAME).read_bytes()
        self.assertEqual(plot[:8], b"\x89PNG\r\n\x1a\n")

        with (self.output / PREDICTIONS_NAME).open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        correct = sum(r["correct"] == "true" for r in records)
        summary = json.loads((self.output / SUMMARY_NAME).read_text(encoding="utf-8"))
        self.assertEqual(correct, summary["correct"])
        self.assertAlmostEqual(summary["accuracy"], correct / len(records))
        self.assertEqual(summary["checkpoint"]["file"], "best_model.pth")
        self.assertEqual(summary["device"], "cpu")
        self.assertEqual(summary["architecture"], "mobilenet_v3_small")
        self.assertEqual(summary["classes"], list(CLASSES))
        self.assertNotIn(str(self.root), (self.output / SUMMARY_NAME).read_text(encoding="utf-8"))
        self.assertIn("evaluation_timestamp", summary)

    def test_report_csv_lists_every_class_and_averages(self) -> None:
        self.run_quiet(checkpoint=self.make_checkpoint(favour="metal"))
        with (self.output / REPORT_NAME).open(newline="", encoding="utf-8") as handle:
            rows = {r["class"]: r for r in csv.DictReader(handle)}
        for name in (*CLASSES, "macro avg", "weighted avg", "accuracy"):
            self.assertIn(name, rows)
        self.assertEqual(rows["metal"]["support"], "3")
        self.assertEqual(rows["accuracy"]["f1"], "0.250000")
        self.assertFalse(any("nan" in v.lower() for r in rows.values() for v in r.values()))

    def test_missing_test_class_is_reported_as_a_limitation(self) -> None:
        def drop_other(rows):  # type: ignore[no-untyped-def]
            for r in rows:
                if r["split"] == "test" and r["material_class"] == "other":
                    r["split"] = "train"

        self.rewrite_manifest(drop_other)
        result = self.run_quiet(checkpoint=self.make_checkpoint(favour="metal"))
        s = result.summary  # type: ignore[attr-defined]
        self.assertEqual(s["missing_classes_in_test"], ["other"])
        self.assertTrue(any("No test images of class 'other'" in n for n in s["notes"]))


class CommandLineTests(EvalProjectTestCase):
    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--config", str(self.config_path), *args])
        return code, out.getvalue()

    def setUp(self) -> None:
        super().setUp()
        self.config_path = self.root / "config" / "config.yaml"
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["image"] = {"width": 32, "height": 32}
        raw["training"].update(batch_size=8, num_workers=0, device="cpu", pretrained=False)
        self.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    def test_successful_run_returns_zero(self) -> None:
        self.make_checkpoint(favour="metal")
        code, out = self.run_main("--checkpoint", str(self.checkpoint), "--output-dir", str(self.output))
        self.assertEqual(code, 0, out)
        self.assertIn("accuracy", out)
        self.assertTrue((self.output / SUMMARY_NAME).is_file())

    def test_missing_checkpoint_returns_one_with_message(self) -> None:
        code, out = self.run_main("--checkpoint", str(self.root / "nope.pth"), "--output-dir", str(self.output))
        self.assertEqual(code, 1)
        self.assertIn("checkpoint not found", out)

    def test_existing_results_return_one(self) -> None:
        self.make_checkpoint(favour="metal")
        args = ("--checkpoint", str(self.checkpoint), "--output-dir", str(self.output))
        self.assertEqual(self.run_main(*args)[0], 0)
        code, out = self.run_main(*args)
        self.assertEqual(code, 1)
        self.assertIn("--overwrite", out)

    def test_bad_config_returns_two(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--config", str(self.root / "missing.yaml")])
        self.assertEqual(code, 2)

    def test_cuda_without_gpu_returns_one(self) -> None:
        from unittest import mock

        self.make_checkpoint(favour="metal")
        with mock.patch("torch.cuda.is_available", return_value=False):
            code, out = self.run_main(
                "--checkpoint", str(self.checkpoint), "--output-dir", str(self.output), "--device", "cuda"
            )
        self.assertEqual(code, 1)
        self.assertIn("CUDA", out)


if __name__ == "__main__":
    unittest.main()
