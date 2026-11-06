"""Tests for src.predict (single-image inference).

Fast and deterministic: 32x32 inference with random untrained models whose output
is fixed in advance, no pretrained weights, and no GPU. Run with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch
import yaml
from torch import nn

from helpers import encode_image, write_image
from src.config import DEFAULT_CONFIG_PATH, MATERIAL_CLASSES, Config, ImageConfig, load_config
from src.model import build_model, save_checkpoint
from src.predict import (
    PREDICTION_FORMAT_VERSION,
    PredictionError,
    format_text,
    main,
    predict_image,
    predict_probabilities,
    softmax,
)
from src.preprocessing import load_rgb, preprocess_eval

CLASSES = MATERIAL_CLASSES
SMALL = ImageConfig(width=32, height=32)


def expected_probabilities(favoured: str | None) -> dict[str, float]:
    """Softmax of logits [10 for the favoured class, 0 otherwise], computed by hand."""
    logits = np.array([10.0 if c == favoured else 0.0 for c in CLASSES])
    exp = np.exp(logits)
    return dict(zip(CLASSES, (exp / exp.sum()).tolist()))


class SoftmaxTests(unittest.TestCase):
    def test_equal_logits_give_equal_probabilities(self) -> None:
        np.testing.assert_allclose(softmax(np.zeros(4)), [0.25] * 4)

    def test_known_logits_match_the_formula(self) -> None:
        probs = softmax(np.array([2.0, 0.0, 0.0, 0.0]))
        e2 = math.exp(2.0)
        np.testing.assert_allclose(probs, [e2 / (e2 + 3), 1 / (e2 + 3), 1 / (e2 + 3), 1 / (e2 + 3)], rtol=1e-12)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=12)

    def test_large_logits_do_not_overflow(self) -> None:
        probs = softmax(np.array([1000.0, 0.0, 0.0, 0.0]))
        self.assertFalse(np.isnan(probs).any())
        np.testing.assert_allclose(probs, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_softmax_runs_over_the_class_axis(self) -> None:
        batch = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
        probs = softmax(batch)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])
        self.assertGreater(probs[1, 3], probs[1, 0])


class PredictProjectTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "config").mkdir()
        self.config_path = self.root / "config" / "config.yaml"
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["image"] = {"width": 32, "height": 32}
        raw["training"].update(batch_size=8, num_workers=0, device="cpu", pretrained=False)
        self.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        self.config: Config = load_config(self.config_path)
        self.model_dir = self.config.paths.model_dir
        self.checkpoint = self.model_dir / "best_model.pth"
        self.image = self.root / "object.jpg"
        write_image(self.image, seed=11)  # 48 x 64 synthetic JPEG

    def make_checkpoint(
        self,
        *,
        favour: str | None = None,
        architecture: str = "mobilenet_v3_small",
        image: ImageConfig = SMALL,
        classes: tuple[str, ...] = CLASSES,
        num_outputs: int = 4,
        path: Path | None = None,
    ) -> Path:
        torch.manual_seed(0)
        model = build_model(architecture, num_outputs, pretrained=False)
        if favour is not None:
            head = model.classifier[-1] if architecture != "resnet18" else model.fc  # type: ignore[index, attr-defined]
            with torch.no_grad():
                head.weight.zero_()
                head.bias.copy_(torch.tensor([10.0 if c == favour else 0.0 for c in CLASSES]))
        target = path or self.checkpoint
        save_checkpoint(
            target,
            model,
            {
                "architecture": architecture,
                "classes": list(classes),
                "image_width": image.width,
                "image_height": image.height,
                "normalization": "imagenet",
                "pretrained": False,
                "seed": 42,
                "epoch": 1,
                "val_loss": 1.0,
            },
        )
        return target

    def predict(self, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("device_setting", "cpu")
        return predict_image(self.config, kwargs.pop("checkpoint", self.checkpoint), kwargs.pop("image", self.image), **kwargs)

    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--config", str(self.config_path), *args])
        return code, out.getvalue()


class KnownOutputPredictionTests(PredictProjectTestCase):
    def test_known_output_model_predicts_the_expected_class_and_probabilities(self) -> None:
        self.make_checkpoint(favour="metal")
        result = self.predict()
        self.assertEqual(result.predicted_class, "metal")
        expected = expected_probabilities("metal")
        for name in CLASSES:
            self.assertAlmostEqual(result.probabilities[name], expected[name], places=9, msg=name)
        self.assertAlmostEqual(result.max_probability, expected["metal"], places=9)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=9)

    def test_probability_is_mapped_to_the_class_it_belongs_to(self) -> None:
        for favoured in CLASSES:
            with self.subTest(favoured=favoured):
                self.make_checkpoint(favour=favoured)
                result = self.predict()
                self.assertEqual(result.predicted_class, favoured)
                self.assertEqual(max(result.probabilities, key=result.probabilities.get), favoured)
                self.assertAlmostEqual(result.probabilities[favoured], expected_probabilities(favoured)[favoured], places=9)

    def test_exactly_four_classes_are_reported_in_checkpoint_order(self) -> None:
        self.make_checkpoint(favour="paper")
        result = self.predict()
        self.assertEqual(list(result.probabilities), list(CLASSES))
        self.assertEqual(len(result.probabilities), 4)
        self.assertNotIn("unknown", result.probabilities)
        self.assertNotIn("uncertain", result.probabilities)

    def test_model_output_has_four_values(self) -> None:
        self.make_checkpoint(favour="other")
        from src.model import load_checkpoint

        model, _ = load_checkpoint(self.checkpoint, classes=CLASSES)
        probs = predict_probabilities(model, load_rgb(self.image), SMALL, torch.device("cpu"))
        self.assertEqual(probs.shape, (4,))

    def test_prediction_is_deterministic_without_augmentation(self) -> None:
        self.make_checkpoint()
        first = self.predict()
        second = self.predict()
        self.assertEqual(first.probabilities, second.probabilities)
        self.assertEqual(first.predicted_class, second.predicted_class)

    def test_checkpoint_metadata_is_reported(self) -> None:
        self.make_checkpoint(favour="plastic")
        result = self.predict()
        self.assertEqual(result.architecture, "mobilenet_v3_small")
        self.assertEqual(result.checkpoint, "best_model.pth")
        self.assertEqual(len(result.checkpoint_sha256), 64)
        self.assertEqual(result.device, "cpu")


class PreprocessingUseTests(PredictProjectTestCase):
    def test_model_receives_exactly_the_evaluation_preprocessing(self) -> None:
        class Capture(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seen: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.seen.append(x.detach().clone())
                return torch.zeros(x.shape[0], 4)

        capture = Capture()
        rgb = load_rgb(self.image)
        predict_probabilities(capture, rgb, SMALL, torch.device("cpu"))
        expected = preprocess_eval(rgb, SMALL)
        np.testing.assert_array_equal(capture.seen[0].squeeze(0).numpy(), expected)

    def test_non_ascii_and_spaced_file_names_work(self) -> None:
        name = "Bild mit Ünïcode & Leerzeichen 1.jpg"
        path = self.root / name
        self.assertTrue(cv2.imwrite(str(path), cv2.imdecode(np.frombuffer(encode_image(seed=3), np.uint8), cv2.IMREAD_COLOR)))
        self.make_checkpoint(favour="other")
        result = self.predict(image=path)
        self.assertEqual(result.predicted_class, "other")
        self.assertEqual(Path(result.image_path).name, name)

    def test_grey_and_png_images_are_accepted(self) -> None:
        self.make_checkpoint(favour="metal")
        grey = self.root / "grey.png"
        cv2.imwrite(str(grey), np.full((40, 50), 90, dtype=np.uint8))
        self.assertEqual(self.predict(image=grey).predicted_class, "metal")


class InputErrorTests(PredictProjectTestCase):
    def test_missing_image_is_a_clear_error(self) -> None:
        self.make_checkpoint()
        with self.assertRaisesRegex(PredictionError, "image not found"):
            self.predict(image=self.root / "absent.jpg")

    def test_directory_given_as_image_is_an_error(self) -> None:
        self.make_checkpoint()
        with self.assertRaisesRegex(PredictionError, "not a file"):
            self.predict(image=self.root)

    def test_invalid_image_file_is_a_clear_error(self) -> None:
        self.make_checkpoint()
        bad = self.root / "broken.jpg"
        bad.write_bytes(b"this is not an image")
        with self.assertRaisesRegex(PredictionError, "not a readable image file"):
            self.predict(image=bad)
        empty = self.root / "empty.png"
        empty.write_bytes(b"")
        with self.assertRaisesRegex(PredictionError, "not a readable image file"):
            self.predict(image=empty)

    def test_invalid_threshold_is_rejected(self) -> None:
        self.make_checkpoint()
        for value in (0.0, 1.0, 1.5, -0.2):
            with self.subTest(value=value), self.assertRaisesRegex(PredictionError, "threshold"):
                self.predict(threshold=value)


class CheckpointErrorTests(PredictProjectTestCase):
    def test_missing_checkpoint_is_a_clear_error_with_no_fallback(self) -> None:
        other = self.model_dir / "other_model.pth"
        self.make_checkpoint(path=other, favour="metal")  # a different checkpoint exists...
        with self.assertRaisesRegex(PredictionError, "checkpoint not found.*No other checkpoint"):
            self.predict(checkpoint=self.model_dir / "missing.pth")  # ...but must not be used

    def test_wrong_architecture_is_refused(self) -> None:
        self.make_checkpoint(architecture="resnet18")
        with self.assertRaisesRegex(PredictionError, "architecture"):
            self.predict()

    def test_wrong_image_size_is_refused(self) -> None:
        self.make_checkpoint(image=ImageConfig(width=64, height=64))
        with self.assertRaisesRegex(PredictionError, "trained at 64x64"):
            self.predict()

    def test_wrong_class_order_is_refused(self) -> None:
        self.make_checkpoint(classes=("paper", "metal", "plastic", "other"))
        with self.assertRaisesRegex(PredictionError, "do not match"):
            self.predict()

    def test_checkpoint_with_three_outputs_is_refused(self) -> None:
        self.make_checkpoint(num_outputs=3, classes=CLASSES)
        with self.assertRaisesRegex(PredictionError, "does not fit architecture"):
            self.predict()

    def test_corrupt_checkpoint_is_refused(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint.write_bytes(b"garbage")
        with self.assertRaisesRegex(PredictionError, "cannot read checkpoint"):
            self.predict()

    def test_checkpoint_is_never_modified(self) -> None:
        self.make_checkpoint(favour="metal")
        before = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.predict()
        self.predict(threshold=0.5)
        self.assertEqual(hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(), before)

    def test_cuda_request_without_gpu_is_an_error(self) -> None:
        self.make_checkpoint()
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(PredictionError, "CUDA was requested"):
                self.predict(device_setting="cuda")


class ThresholdTests(PredictProjectTestCase):
    def test_threshold_flags_low_confidence_without_changing_the_class(self) -> None:
        self.make_checkpoint(favour="metal")  # max probability is e^10/(e^10+3) = 0.9998638...
        above = self.predict(threshold=0.9998)  # max is above the threshold: not low confidence
        self.assertFalse(above.low_confidence)
        below = self.predict(threshold=0.99999)  # max is below the threshold: low confidence
        self.assertTrue(below.low_confidence)
        high = below
        self.assertEqual(high.predicted_class, "metal")  # never a fifth class
        self.assertIn(high.predicted_class, CLASSES)

    def test_no_threshold_means_no_low_confidence_field(self) -> None:
        self.make_checkpoint()
        result = self.predict()
        self.assertIsNone(result.low_confidence)
        self.assertNotIn("low_confidence", result.to_dict())


class OutputTests(PredictProjectTestCase):
    def test_json_output_is_parseable_and_complete(self) -> None:
        self.make_checkpoint(favour="paper")
        code, out = self.run_main("--image", str(self.image), "--json")
        self.assertEqual(code, 0, out)
        data = json.loads(out)
        self.assertEqual(data["format_version"], PREDICTION_FORMAT_VERSION)
        self.assertEqual(data["predicted_class"], "paper")
        self.assertEqual(list(data["probabilities"]), list(CLASSES))
        self.assertAlmostEqual(sum(data["probabilities"].values()), 1.0, places=9)
        self.assertAlmostEqual(data["max_probability"], data["probabilities"]["paper"], places=12)
        self.assertEqual(data["device"], "cpu")
        self.assertIn("probability_note", data)
        self.assertNotIn(str(self.root), json.dumps({k: v for k, v in data.items() if k != "image"}))

    def test_json_threshold_fields(self) -> None:
        self.make_checkpoint(favour="metal")
        _, out = self.run_main("--image", str(self.image), "--json", "--threshold", "0.5")
        data = json.loads(out)
        self.assertEqual(data["threshold"], 0.5)
        self.assertFalse(data["low_confidence"])

    def test_json_error_for_missing_image_and_exit_code(self) -> None:
        self.make_checkpoint()
        code, out = self.run_main("--image", str(self.root / "nope.jpg"), "--json")
        self.assertEqual(code, 1)
        self.assertIn("image not found", json.loads(out)["error"])

    def test_text_output_lists_four_probabilities_and_disclaimer(self) -> None:
        self.make_checkpoint(favour="other")
        code, out = self.run_main("--image", str(self.image))
        self.assertEqual(code, 0, out)
        self.assertIn("Predicted:        other", out)
        for name in CLASSES:
            self.assertIn(name, out)
        self.assertIn("not calibrated", out)
        result = self.predict()
        self.assertIn(f"{result.max_probability:.6f}", format_text(result))

    def test_missing_checkpoint_on_command_line_returns_one(self) -> None:
        code, out = self.run_main("--image", str(self.image), "--checkpoint", str(self.root / "absent.pth"))
        self.assertEqual(code, 1)
        self.assertIn("checkpoint not found", out)

    def test_bad_threshold_on_command_line_returns_two(self) -> None:
        self.make_checkpoint()
        code, out = self.run_main("--image", str(self.image), "--threshold", "1.2")
        self.assertEqual(code, 2)
        self.assertIn("threshold", out)

    def test_bad_config_returns_two(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--image", str(self.image), "--config", str(self.root / "missing.yaml")])
        self.assertEqual(code, 2)

    def test_cpu_device_option_is_used(self) -> None:
        self.make_checkpoint(favour="metal")
        code, out = self.run_main("--image", str(self.image), "--device", "cpu", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
