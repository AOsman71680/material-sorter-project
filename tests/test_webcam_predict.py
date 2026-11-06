"""Tests for the live webcam preview. No physical camera, window, or GPU is used.

The camera and window are replaced by small fakes, and cv2.VideoCapture is
mocked wherever the real camera-open path is exercised.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import itertools
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch
import yaml

from src.config import DEFAULT_CONFIG_PATH, MATERIAL_CLASSES, Config, ImageConfig, load_config
from src.data_collection import CameraError, InvalidFrameError, open_camera
from src.model import build_model, save_checkpoint
from src.predict import PredictionError, load_model_for_inference
from src.webcam_predict import (
    WAITING_TEXT,
    LivePrediction,
    draw_prediction,
    frame_to_rgb,
    main,
    make_predictor,
    overlay_text,
    run_prediction_loop,
    should_run_inference,
    summarize_probabilities,
)

CLASSES = MATERIAL_CLASSES
ESC = 27
Q = ord("q")
SMALL = ImageConfig(width=32, height=32)
FRAME_SHAPE = (48, 64, 3)


def blank_frame(value: int = 0) -> np.ndarray:
    return np.full(FRAME_SHAPE, value, dtype=np.uint8)


def expected_probabilities(favoured: str) -> dict[str, float]:
    """Softmax of logits [10 for the favoured class, 0 otherwise], computed by hand."""
    logits = np.array([10.0 if c == favoured else 0.0 for c in CLASSES])
    exp = np.exp(logits)
    return dict(zip(CLASSES, (exp / exp.sum()).tolist()))


def prediction_from(probs: list[float], threshold: float | None = None) -> LivePrediction:
    return summarize_probabilities(np.array(probs, dtype=np.float64), CLASSES, threshold)


class FakeSource:
    """Stands in for cv2.VideoCapture. ``None`` items are failed reads."""

    def __init__(self, items: list[np.ndarray | None]) -> None:
        self.items = list(items)
        self.released = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.items:
            return False, None
        item = self.items.pop(0)
        return item is not None, item

    def release(self) -> None:
        self.released += 1


class FakeDisplay:
    """Stands in for the OpenCV window. Keys are returned in order; an empty list raises."""

    def __init__(self, keys: list[int] | None = None, *, open_window: bool = True) -> None:
        self.keys = list(keys or [])
        self.shown: list[np.ndarray] = []
        self.closed = 0
        self.open_window = open_window

    def show(self, image: np.ndarray) -> None:
        self.shown.append(image.copy())

    def wait_key(self, delay_ms: int) -> int:
        if not self.keys:
            raise AssertionError("the loop did not quit when it should have")
        return self.keys.pop(0)

    def is_open(self) -> bool:
        return self.open_window

    def close(self) -> None:
        self.closed += 1


class InterruptingDisplay(FakeDisplay):
    def show(self, image: np.ndarray) -> None:
        raise KeyboardInterrupt


def ticking_clock(step: float = 1.0):
    counter = itertools.count(0.0, step)
    return lambda: next(counter)


# --------------------------------------------------------------------------
# Predictions, classes, and cadence
# --------------------------------------------------------------------------


class PredictionMappingTests(unittest.TestCase):
    def test_top_probability_maps_to_its_class_name(self) -> None:
        prediction = prediction_from([0.05, 0.10, 0.80, 0.05])
        self.assertEqual(prediction.predicted_class, "paper")
        self.assertAlmostEqual(prediction.max_probability, 0.80)

    def test_probabilities_keep_the_configured_class_order(self) -> None:
        prediction = prediction_from([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(list(prediction.probabilities), ["metal", "plastic", "paper", "other"])
        self.assertEqual(prediction.predicted_class, "other")

    def test_exactly_four_classes_are_reported(self) -> None:
        prediction = prediction_from([0.25, 0.25, 0.25, 0.25])
        self.assertEqual(set(prediction.probabilities), {"metal", "plastic", "paper", "other"})
        self.assertEqual(len(prediction.probabilities), 4)

    def test_a_fifth_output_is_rejected_not_turned_into_a_class(self) -> None:
        with self.assertRaises(PredictionError):
            summarize_probabilities(np.array([0.2, 0.2, 0.2, 0.2, 0.2]), CLASSES, None)

    def test_too_few_outputs_are_rejected(self) -> None:
        with self.assertRaises(PredictionError):
            summarize_probabilities(np.array([0.5, 0.5]), CLASSES, None)

    def test_low_confidence_is_absent_without_a_threshold(self) -> None:
        prediction = prediction_from([0.4, 0.3, 0.2, 0.1])
        self.assertIsNone(prediction.threshold)
        self.assertIsNone(prediction.low_confidence)

    def test_probability_below_threshold_is_flagged_low(self) -> None:
        prediction = prediction_from([0.4, 0.3, 0.2, 0.1], threshold=0.5)
        self.assertTrue(prediction.low_confidence)
        self.assertEqual(prediction.predicted_class, "metal")  # still a real class

    def test_probability_at_threshold_is_not_low(self) -> None:
        prediction = prediction_from([0.5, 0.2, 0.2, 0.1], threshold=0.5)
        self.assertFalse(prediction.low_confidence)


class CadenceTests(unittest.TestCase):
    def test_inference_runs_on_every_nth_valid_frame(self) -> None:
        runs = [i for i in range(7) if should_run_inference(i, 3)]
        self.assertEqual(runs, [0, 3, 6])

    def test_every_frame_when_n_is_one(self) -> None:
        self.assertTrue(all(should_run_inference(i, 1) for i in range(5)))

    def test_first_frame_is_always_predicted(self) -> None:
        for n in (1, 2, 5, 30):
            self.assertTrue(should_run_inference(0, n))


# --------------------------------------------------------------------------
# Frames and overlay
# --------------------------------------------------------------------------


class FrameConversionTests(unittest.TestCase):
    def test_bgr_frame_is_converted_to_rgb(self) -> None:
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        frame[..., 0] = 255  # blue channel in BGR order
        rgb = frame_to_rgb(frame)
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue(np.all(rgb[..., 2] == 255))  # blue is the third channel in RGB
        self.assertTrue(np.all(rgb[..., 0] == 0))

    def test_conversion_does_not_modify_the_camera_frame(self) -> None:
        frame = blank_frame(7)
        frame_to_rgb(frame)
        self.assertTrue(np.all(frame == 7))

    def test_invalid_frames_are_rejected(self) -> None:
        bad_frames = [
            None,
            np.zeros(FRAME_SHAPE, dtype=np.float32),
            np.zeros((48, 64), dtype=np.uint8),  # grey, not the 3-channel camera format
            np.zeros((0, 64, 3), dtype=np.uint8),
        ]
        for frame in bad_frames:
            with self.subTest(frame=type(frame).__name__):
                with self.assertRaises(InvalidFrameError):
                    frame_to_rgb(frame)


class OverlayTests(unittest.TestCase):
    def test_before_the_first_prediction_the_overlay_says_so(self) -> None:
        self.assertEqual(overlay_text(None), [WAITING_TEXT])

    def test_overlay_shows_class_confidence_and_all_four_probabilities(self) -> None:
        prediction = prediction_from([0.1, 0.2, 0.62, 0.08])
        lines = overlay_text(prediction)
        self.assertEqual(lines[0], "Predicted: paper")
        self.assertEqual(lines[1], "Confidence: 62.0%")
        self.assertEqual(len(lines), 2 + 4)
        self.assertIn("> paper: 62.0%", lines)
        self.assertIn("  metal: 10.0%", lines)
        self.assertIn("  other: 8.0%", lines)

    def test_low_confidence_line_appears_only_when_flagged(self) -> None:
        low = overlay_text(prediction_from([0.4, 0.3, 0.2, 0.1], threshold=0.5))
        self.assertIn("LOW CONFIDENCE (below 0.5)", low)
        ok = overlay_text(prediction_from([0.9, 0.05, 0.03, 0.02], threshold=0.5))
        self.assertFalse(any("LOW CONFIDENCE" in line for line in ok))
        none = overlay_text(prediction_from([0.4, 0.3, 0.2, 0.1]))
        self.assertFalse(any("LOW CONFIDENCE" in line for line in none))

    def test_draw_prediction_returns_an_annotated_copy(self) -> None:
        frame = blank_frame(0)
        drawn = draw_prediction(frame, prediction_from([0.9, 0.05, 0.03, 0.02]))
        self.assertEqual(drawn.shape, frame.shape)
        self.assertEqual(drawn.dtype, np.uint8)
        self.assertTrue(np.all(frame == 0), "the camera frame must not be changed")
        self.assertGreater(int(np.count_nonzero(drawn)), 0)

    def test_draw_prediction_without_a_prediction_still_draws_the_waiting_text(self) -> None:
        drawn = draw_prediction(blank_frame(0), None)
        self.assertGreater(int(np.count_nonzero(drawn)), 0)


# --------------------------------------------------------------------------
# Live loop with fake camera and window
# --------------------------------------------------------------------------


class LoopTests(unittest.TestCase):
    def predictor(self, probs: list[float] | None = None):
        calls: list[np.ndarray] = []
        probs = probs or [0.9, 0.05, 0.03, 0.02]

        def predict(rgb: np.ndarray) -> LivePrediction:
            calls.append(rgb)
            return prediction_from(probs)

        return predict, calls

    def test_q_quits_and_releases_camera_and_window(self) -> None:
        source = FakeSource([blank_frame()])
        display = FakeDisplay(keys=[Q])
        predict, _ = self.predictor()
        stats = run_prediction_loop(predict, source, display)
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)
        self.assertEqual(stats.frames_shown, 1)

    def test_escape_quits(self) -> None:
        source = FakeSource([blank_frame(), blank_frame()])
        display = FakeDisplay(keys=[-1, ESC])
        predict, _ = self.predictor()
        stats = run_prediction_loop(predict, source, display)
        self.assertEqual(stats.frames_shown, 2)
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)

    def test_closing_the_window_stops_the_loop(self) -> None:
        source = FakeSource([blank_frame()])
        display = FakeDisplay(keys=[-1], open_window=False)
        predict, _ = self.predictor()
        run_prediction_loop(predict, source, display)
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)

    def test_other_keys_do_not_quit(self) -> None:
        source = FakeSource([blank_frame(), blank_frame(), blank_frame()])
        display = FakeDisplay(keys=[ord("x"), ord("1"), Q])
        predict, _ = self.predictor()
        stats = run_prediction_loop(predict, source, display)
        self.assertEqual(stats.frames_shown, 3)

    def test_failed_reads_raise_a_clear_error_and_still_release(self) -> None:
        source = FakeSource([blank_frame(), None] + [None] * 20)
        display = FakeDisplay(keys=[-1] * 20)
        predict, _ = self.predictor()
        with self.assertRaisesRegex(CameraError, "stopped delivering valid frames"):
            run_prediction_loop(predict, source, display, clock=ticking_clock(1.0))
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)

    def test_opencv_read_exception_counts_as_a_failed_read(self) -> None:
        class BrokenSource(FakeSource):
            def read(self):
                raise cv2.error("simulated read failure")

        source = BrokenSource([])
        display = FakeDisplay(keys=[-1] * 20)
        predict, _ = self.predictor()
        with self.assertRaises(CameraError):
            run_prediction_loop(predict, source, display, clock=ticking_clock(1.0))
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)

    def test_predictor_error_still_releases_camera_and_window(self) -> None:
        def broken(rgb: np.ndarray) -> LivePrediction:
            raise RuntimeError("simulated inference failure")

        source = FakeSource([blank_frame()])
        display = FakeDisplay(keys=[Q])
        with self.assertRaises(RuntimeError):
            run_prediction_loop(broken, source, display)
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)

    def test_keyboard_interrupt_still_releases_camera_and_window(self) -> None:
        source = FakeSource([blank_frame()])
        display = InterruptingDisplay()
        predict, _ = self.predictor()
        with self.assertRaises(KeyboardInterrupt):
            run_prediction_loop(predict, source, display)
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)

    def test_inference_runs_every_n_frames_and_the_latest_prediction_stays_on_screen(self) -> None:
        frames = [blank_frame(10), blank_frame(20), blank_frame(30), blank_frame(40), blank_frame(50)]
        source = FakeSource(frames)
        display = FakeDisplay(keys=[-1, -1, -1, -1, Q])
        first = [0.9, 0.05, 0.03, 0.02]
        predict, calls = self.predictor(first)
        stats = run_prediction_loop(predict, source, display, every_n=2)

        self.assertEqual(stats.frames_shown, 5)
        self.assertEqual(stats.inferences, 3)  # valid frames 0, 2 and 4
        self.assertEqual(len(calls), 3)
        # Frame 1 was not predicted, so it must show the prediction made on frame 0.
        expected = draw_prediction(frames[1], prediction_from(first))
        np.testing.assert_array_equal(display.shown[1], expected)

    def test_every_frame_is_predicted_when_n_is_one(self) -> None:
        source = FakeSource([blank_frame(), blank_frame(), blank_frame()])
        display = FakeDisplay(keys=[-1, -1, Q])
        predict, calls = self.predictor()
        stats = run_prediction_loop(predict, source, display, every_n=1)
        self.assertEqual(stats.inferences, 3)
        self.assertEqual(len(calls), 3)

    def test_failed_reads_do_not_count_toward_the_inference_cadence(self) -> None:
        source = FakeSource([None, blank_frame(), None, blank_frame()])
        display = FakeDisplay(keys=[-1, -1, -1, Q])
        predict, _ = self.predictor()
        stats = run_prediction_loop(predict, source, display, every_n=2, clock=ticking_clock(0.01))
        self.assertEqual(stats.frames_shown, 2)
        self.assertEqual(stats.inferences, 1)  # only the first valid frame

    def test_predicted_frames_are_converted_from_bgr(self) -> None:
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        frame[..., 0] = 255  # blue in BGR
        source = FakeSource([frame])
        display = FakeDisplay(keys=[Q])
        predict, calls = self.predictor()
        run_prediction_loop(predict, source, display)
        self.assertTrue(np.all(calls[0][..., 2] == 255))

    def test_zero_every_n_is_rejected(self) -> None:
        source = FakeSource([])
        display = FakeDisplay()
        predict, _ = self.predictor()
        with self.assertRaises(ValueError):
            run_prediction_loop(predict, source, display, every_n=0)
        self.assertEqual(source.released, 0)  # rejected before the loop, nothing to release


# --------------------------------------------------------------------------
# Camera opening (cv2.VideoCapture is mocked)
# --------------------------------------------------------------------------


class CameraOpenTests(unittest.TestCase):
    def test_camera_that_does_not_open_raises_a_clear_error(self) -> None:
        capture = mock.Mock()
        capture.isOpened.return_value = False
        with mock.patch("src.data_collection.cv2.VideoCapture", return_value=capture) as factory:
            with self.assertRaisesRegex(CameraError, "camera index 7"):
                open_camera(7)
        factory.assert_called()
        capture.release.assert_called()

    def test_camera_that_opens_but_never_gives_frames_raises(self) -> None:
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with mock.patch("src.data_collection.cv2.VideoCapture", return_value=capture):
            with mock.patch("src.data_collection.time.sleep"):
                with self.assertRaises(CameraError):
                    open_camera(0)
        capture.release.assert_called()

    def test_error_message_names_the_flag_of_the_tool_that_failed(self) -> None:
        capture = mock.Mock()
        capture.isOpened.return_value = False
        with mock.patch("src.data_collection.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(CameraError, r"try another index with --camera\.$"):
                open_camera(0, index_flag="--camera")
        with mock.patch("src.data_collection.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(CameraError, r"try another index with --camera-index\.$"):
                open_camera(0)

    def test_camera_with_a_valid_frame_is_returned(self) -> None:
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, blank_frame(5))
        with mock.patch("src.data_collection.cv2.VideoCapture", return_value=capture):
            self.assertIs(open_camera(0), capture)

    def test_importing_the_module_does_not_open_a_camera(self) -> None:
        import src.webcam_predict as module

        with mock.patch("cv2.VideoCapture") as factory:
            importlib.reload(module)
        factory.assert_not_called()


# --------------------------------------------------------------------------
# Command line and whole-session behaviour (fake camera, fake window)
# --------------------------------------------------------------------------


class SessionTestCase(unittest.TestCase):
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
        self.checkpoint = self.config.paths.model_dir / "best_model.pth"

    def make_checkpoint(self, *, favour: str | None = "metal", classes: tuple[str, ...] = CLASSES) -> Path:
        torch.manual_seed(0)
        model = build_model("mobilenet_v3_small", len(classes), pretrained=False)
        if favour is not None:
            with torch.no_grad():
                model.classifier[-1].weight.zero_()  # type: ignore[index, union-attr]
                model.classifier[-1].bias.copy_(  # type: ignore[index, union-attr]
                    torch.tensor([10.0 if c == favour else 0.0 for c in classes])
                )
        metadata = {
            "architecture": "mobilenet_v3_small",
            "classes": list(classes),
            "image_width": SMALL.width,
            "image_height": SMALL.height,
            "normalization": "imagenet",
            "pretrained": False,
            "seed": 0,
            "epoch": 0,
            "val_loss": 0.0,
            "val_accuracy": 0.0,
        }
        save_checkpoint(self.checkpoint, model, metadata)
        return self.checkpoint

    def run_main(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--config", str(self.config_path), *argv])
        return code, out.getvalue()


class ArgumentTests(SessionTestCase):
    def test_help_lists_every_option_and_exits_zero(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as caught:
            main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        for flag in ("--checkpoint", "--camera", "--device", "--threshold", "--inference-every-n-frames"):
            self.assertIn(flag, out.getvalue())

    def test_threshold_outside_zero_to_one_exits_2(self) -> None:
        for value in ("0", "1", "1.5", "-0.2"):
            with self.subTest(threshold=value):
                code, out = self.run_main("--threshold", value)
                self.assertEqual(code, 2)
                self.assertIn("--threshold", out)

    def test_inference_interval_must_be_at_least_one(self) -> None:
        for value in ("0", "-3"):
            with self.subTest(n=value):
                code, out = self.run_main("--inference-every-n-frames", value)
                self.assertEqual(code, 2)
                self.assertIn("--inference-every-n-frames", out)

    def test_negative_camera_index_exits_2(self) -> None:
        code, out = self.run_main("--camera", "-1")
        self.assertEqual(code, 2)
        self.assertIn("--camera", out)

    def test_unknown_device_is_rejected_by_argparse(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["--device", "gpu"])
        self.assertEqual(caught.exception.code, 2)

    def test_missing_config_exits_2(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--config", str(self.root / "missing.yaml")])
        self.assertEqual(code, 2)

    def test_default_camera_comes_from_the_config(self) -> None:
        self.make_checkpoint()
        with mock.patch("src.webcam_predict.open_camera", side_effect=CameraError("no camera")) as opener:
            code, _ = self.run_main("--checkpoint", str(self.checkpoint))
        self.assertEqual(code, 1)
        opener.assert_called_once_with(self.config.camera_index, index_flag="--camera")

    def test_explicit_camera_overrides_the_config(self) -> None:
        self.make_checkpoint()
        with mock.patch("src.webcam_predict.open_camera", side_effect=CameraError("no camera")) as opener:
            self.run_main("--checkpoint", str(self.checkpoint), "--camera", "3")
        opener.assert_called_once_with(3, index_flag="--camera")


class FailurePathTests(SessionTestCase):
    def test_missing_checkpoint_exits_1_and_never_opens_the_camera(self) -> None:
        with mock.patch("src.webcam_predict.open_camera") as opener:
            code, out = self.run_main("--checkpoint", str(self.root / "nope.pth"))
        self.assertEqual(code, 1)
        self.assertIn("checkpoint not found", out)
        opener.assert_not_called()

    def test_incompatible_checkpoint_exits_1_and_never_opens_the_camera(self) -> None:
        self.make_checkpoint(classes=("a", "b", "c", "d"))
        with mock.patch("src.webcam_predict.open_camera") as opener:
            code, _ = self.run_main("--checkpoint", str(self.checkpoint))
        self.assertEqual(code, 1)
        opener.assert_not_called()

    def test_cuda_unavailable_exits_1_with_a_clear_message(self) -> None:
        self.make_checkpoint()
        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("src.webcam_predict.open_camera") as opener:
                code, out = self.run_main("--checkpoint", str(self.checkpoint), "--device", "cuda")
        self.assertEqual(code, 1)
        self.assertIn("CUDA was requested but is not available", out)
        opener.assert_not_called()

    def test_camera_that_cannot_open_exits_1_and_creates_no_window(self) -> None:
        self.make_checkpoint()
        with mock.patch("src.webcam_predict.open_camera", side_effect=CameraError("camera index 0 failed")):
            with mock.patch("src.webcam_predict.OpenCvWindow") as window:
                code, out = self.run_main("--checkpoint", str(self.checkpoint))
        self.assertEqual(code, 1)
        self.assertIn("camera index 0 failed", out)
        window.assert_not_called()

    def test_window_creation_failure_releases_the_camera(self) -> None:
        self.make_checkpoint()
        source = FakeSource([blank_frame()])
        with mock.patch("src.webcam_predict.open_camera", return_value=source):
            with mock.patch("src.webcam_predict.OpenCvWindow", side_effect=cv2.error("no display")):
                code, out = self.run_main("--checkpoint", str(self.checkpoint))
        self.assertEqual(code, 1)
        self.assertIn("OpenCV error", out)
        self.assertEqual(source.released, 1)

    def test_camera_failure_during_the_session_exits_1_and_releases(self) -> None:
        self.make_checkpoint()
        source = FakeSource([None] * 50)
        display = FakeDisplay(keys=[-1] * 50)
        with mock.patch("src.webcam_predict.open_camera", return_value=source):
            with mock.patch("src.webcam_predict.OpenCvWindow", return_value=display):
                with mock.patch("src.webcam_predict.time.monotonic", ticking_clock(1.0)):
                    code, out = self.run_main("--checkpoint", str(self.checkpoint))
        self.assertEqual(code, 1)
        self.assertIn("stopped delivering valid frames", out)
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)


class SessionTests(SessionTestCase):
    def test_mocked_session_predicts_then_quits_with_q(self) -> None:
        self.make_checkpoint(favour="metal")
        source = FakeSource([blank_frame(), blank_frame()])
        display = FakeDisplay(keys=[-1, Q])
        with mock.patch("src.webcam_predict.open_camera", return_value=source):
            with mock.patch("src.webcam_predict.OpenCvWindow", return_value=display):
                code, out = self.run_main("--checkpoint", str(self.checkpoint), "--device", "cpu")
        self.assertEqual(code, 0)
        self.assertIn("Stopped after 2 frames and 2 predictions.", out)
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)
        # The frames were drawn with the metal prediction from the real (zeroed) checkpoint.
        self.assertGreater(int(np.count_nonzero(display.shown[0] != blank_frame())), 0)

    def test_mocked_session_with_escape_and_every_n_frames(self) -> None:
        self.make_checkpoint()
        frames = [blank_frame(v) for v in range(4)]
        source = FakeSource(frames)
        display = FakeDisplay(keys=[-1, -1, -1, ESC])
        with mock.patch("src.webcam_predict.open_camera", return_value=source):
            with mock.patch("src.webcam_predict.OpenCvWindow", return_value=display):
                code, out = self.run_main(
                    "--checkpoint", str(self.checkpoint), "--device", "cpu",
                    "--inference-every-n-frames", "2", "--threshold", "0.5",
                )
        self.assertEqual(code, 0)
        self.assertIn("Stopped after 4 frames and 2 predictions.", out)

    def test_keyboard_interrupt_exits_0_and_releases(self) -> None:
        self.make_checkpoint()
        source = FakeSource([blank_frame()])
        display = InterruptingDisplay()
        with mock.patch("src.webcam_predict.open_camera", return_value=source):
            with mock.patch("src.webcam_predict.OpenCvWindow", return_value=display):
                code, out = self.run_main("--checkpoint", str(self.checkpoint), "--device", "cpu")
        self.assertEqual(code, 0)
        self.assertIn("Interrupted.", out)
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)


class RealCheckpointPredictionTests(SessionTestCase):
    """The prediction path uses the real checkpoint loader, so these exercise true class mapping."""

    def test_real_checkpoint_maps_outputs_to_the_four_classes(self) -> None:
        self.make_checkpoint(favour="plastic")
        loaded = load_model_for_inference(self.config, self.checkpoint, device_setting="cpu")
        predict = make_predictor(loaded, self.config.classes, threshold=0.5)
        prediction = predict(np.zeros(FRAME_SHAPE, dtype=np.uint8))
        self.assertEqual(prediction.predicted_class, "plastic")
        self.assertEqual(list(prediction.probabilities), list(CLASSES))
        expected = expected_probabilities("plastic")
        for name in CLASSES:
            self.assertTrue(math.isclose(prediction.probabilities[name], expected[name], rel_tol=1e-6))
        self.assertAlmostEqual(sum(prediction.probabilities.values()), 1.0, places=6)
        self.assertFalse(prediction.low_confidence)

    def test_prediction_does_not_depend_on_the_frame_for_a_fixed_checkpoint(self) -> None:
        self.make_checkpoint(favour="other")
        loaded = load_model_for_inference(self.config, self.checkpoint, device_setting="cpu")
        predict = make_predictor(loaded, self.config.classes, threshold=None)
        dark = predict(np.zeros(FRAME_SHAPE, dtype=np.uint8))
        bright = predict(np.full(FRAME_SHAPE, 255, dtype=np.uint8))
        self.assertEqual(dark.predicted_class, bright.predicted_class)
        self.assertAlmostEqual(dark.max_probability, bright.max_probability, places=9)


if __name__ == "__main__":
    unittest.main()
