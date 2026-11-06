"""Tests for the live sorter. No camera, window, serial port, or GPU is used.

The camera, window, and serial port are replaced by fakes. The model is a real
tiny checkpoint with a fixed output, so the prediction path is exercised.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch
import yaml

from src import live_sorter as ls
from src.config import DEFAULT_CONFIG_PATH, MATERIAL_CLASSES, PROJECT_ROOT, Config, load_config
from src.data_collection import CameraError
from src.model import build_model, save_checkpoint
from src.webcam_predict import LivePrediction, summarize_probabilities

H, W = 240, 320
CLASSES = MATERIAL_CLASSES
Q = ord("q")
ESC = 27


# --------------------------------------------------------------------------
# Synthetic frames
# --------------------------------------------------------------------------


def platform() -> np.ndarray:
    rng = np.random.default_rng(0)
    base = np.full((H, W, 3), (120, 135, 150), np.float32)
    base += cv2.GaussianBlur(rng.normal(0, 4, (H, W)).astype(np.float32), (0, 0), 2)[..., None]
    return np.clip(base, 0, 255).astype(np.uint8)


def frame_with_object(colour=(40, 60, 220)) -> np.ndarray:
    img = platform().copy()
    img[90:150, 100:170] = colour
    return img


def brighter(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def prediction(probs: list[float], threshold: float | None = None) -> LivePrediction:
    return summarize_probabilities(np.array(probs, dtype=np.float64), CLASSES, threshold)


METAL = prediction([0.9, 0.05, 0.03, 0.02])
PLASTIC = prediction([0.05, 0.9, 0.03, 0.02])
UNSURE = prediction([0.4, 0.3, 0.2, 0.1])


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeSource:
    def __init__(self, items: list[np.ndarray | None]) -> None:
        self.items = list(items)
        self.released = 0

    def read(self):
        if not self.items:
            return False, None
        item = self.items.pop(0)
        return item is not None, item

    def release(self) -> None:
        self.released += 1


class FakeDisplay:
    def __init__(self, keys: list[int] | None = None) -> None:
        self.keys = list(keys or [])
        self.shown: list[np.ndarray] = []
        self.closed = 0

    def show(self, image: np.ndarray) -> None:
        self.shown.append(image.copy())

    def wait_key(self, delay_ms: int) -> int:
        if not self.keys:
            raise AssertionError("the loop did not stop when it should have")
        return self.keys.pop(0)

    def is_open(self) -> bool:
        return True

    def close(self) -> None:
        self.closed += 1


class FakeSerial:
    """Stands in for pyserial. ``replies`` are returned by readline() in order."""

    timeout = 0.01  # short poll, so silent-device tests finish quickly

    def __init__(self, replies: list[bytes] | None = None) -> None:
        self.replies = list(replies or [])
        self.written: list[bytes] = []
        self.flushes = 0
        self.resets = 0
        self.closed = False

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def flush(self) -> None:
        self.flushes += 1

    def reset_input_buffer(self) -> None:
        self.resets += 1

    def readline(self) -> bytes:
        return self.replies.pop(0) if self.replies else b""

    def close(self) -> None:
        self.closed = True


# The firmware's messages for one complete sort of metal (see the STM32 code).
COMPLETE_SORT = [
    b"REQUESTED:METAL\n",
    b"METAL\n",
    b"POSITION CONFIRMED\n",
    b"GATE OPEN\n",
    b"GATE CLOSED\n",
    b"SORTING COMPLETE\n",
]


def ticking_clock(step: float = 1.0):
    counter = iter(range(10_000))
    return lambda: float(next(counter)) * step


# --------------------------------------------------------------------------
# Serial settings file
# --------------------------------------------------------------------------


class SerialSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "serial.txt"

    def write(self, text: str) -> None:
        self.path.write_text(text, encoding="utf-8")

    def test_reads_port_baud_and_timeout(self) -> None:
        self.write("port = COM3\nbaud = 115200\nack_timeout = 3.5\n")
        settings = ls.load_serial_settings(self.path)
        self.assertEqual((settings.port, settings.baud, settings.ack_timeout), ("COM3", 115200, 3.5))

    def test_timeout_defaults_to_two_seconds(self) -> None:
        self.write("port = COM3\nbaud = 9600\n")
        self.assertEqual(ls.load_serial_settings(self.path).ack_timeout, 2.0)

    def test_comments_and_blank_lines_are_ignored(self) -> None:
        self.write("# header\n\nport = /dev/ttyACM0\n  # indented comment\nbaud = 115200\n")
        self.assertEqual(ls.load_serial_settings(self.path).port, "/dev/ttyACM0")

    def test_blank_port_asks_for_it(self) -> None:
        self.write("port = \nbaud = 115200\n")
        with self.assertRaisesRegex(ls.SerialConfigError, "set the serial port"):
            ls.load_serial_settings(self.path)

    def test_blank_baud_asks_for_it(self) -> None:
        self.write("port = COM3\nbaud = \n")
        with self.assertRaisesRegex(ls.SerialConfigError, "set the baud rate"):
            ls.load_serial_settings(self.path)

    def test_bad_baud_is_rejected(self) -> None:
        for value in ("fast", "0", "-9600", "115200.5"):
            with self.subTest(baud=value):
                self.write(f"port = COM3\nbaud = {value}\n")
                with self.assertRaises(ls.SerialConfigError):
                    ls.load_serial_settings(self.path)

    def test_unknown_key_is_rejected(self) -> None:
        self.write("port = COM3\nbaud = 115200\nparity = none\n")
        with self.assertRaisesRegex(ls.SerialConfigError, "unknown key 'parity'"):
            ls.load_serial_settings(self.path)

    def test_duplicate_key_is_rejected(self) -> None:
        self.write("port = COM3\nport = COM4\nbaud = 115200\n")
        with self.assertRaisesRegex(ls.SerialConfigError, "more than once"):
            ls.load_serial_settings(self.path)

    def test_line_without_equals_is_rejected(self) -> None:
        self.write("port COM3\nbaud = 115200\n")
        with self.assertRaisesRegex(ls.SerialConfigError, "expected 'key = value'"):
            ls.load_serial_settings(self.path)

    def test_timeout_out_of_range_is_rejected(self) -> None:
        self.write("port = COM3\nbaud = 115200\nack_timeout = 0\n")
        with self.assertRaisesRegex(ls.SerialConfigError, "between 0.1 and 30"):
            ls.load_serial_settings(self.path)

    def test_missing_file_is_reported(self) -> None:
        with self.assertRaisesRegex(ls.SerialConfigError, "not found"):
            ls.load_serial_settings(self.path.with_name("nope.txt"))


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------


class ProtocolTests(unittest.TestCase):
    def test_default_cups_give_the_four_codes(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        codes = ls.class_codes(config.cups, config.classes)
        self.assertEqual(codes, {"metal": b"1", "plastic": b"2", "paper": b"3", "other": b"4"})

    def test_firmware_names_match_the_firmware(self) -> None:
        self.assertEqual(ls.FIRMWARE_NAMES, {"metal": "METAL", "plastic": "PLASTIC", "paper": "PAPER", "other": "ORGANIC"})

    def test_cup_outside_one_to_four_is_rejected(self) -> None:
        with self.assertRaisesRegex(ls.SorterError, "1 to 4 only"):
            ls.class_codes({"metal": 5, "plastic": 2, "paper": 3, "other": 4}, CLASSES)

    def test_command_is_the_digit_followed_by_a_newline(self) -> None:
        port = FakeSerial(list(COMPLETE_SORT))
        ls.run_sort_command(port, "1", "METAL")
        self.assertEqual(port.written, [b"1\n"])
        self.assertEqual(port.resets, 1)

    def test_full_sort_completes(self) -> None:
        outcome = ls.run_sort_command(FakeSerial(list(COMPLETE_SORT)), "1", "METAL")
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.final_line, "SORTING COMPLETE")
        self.assertIn("GATE OPEN", outcome.lines)

    def test_organic_sort_uses_the_organic_name_and_reports_a_timeout(self) -> None:
        port = FakeSerial([b"REQUESTED:ORGANIC\n", b"POSITION TIMEOUT\n"])
        outcome = ls.run_sort_command(port, "4", "ORGANIC")
        self.assertFalse(outcome.completed)
        self.assertEqual(outcome.final_line, "POSITION TIMEOUT")

    def test_stopped_is_reported_not_raised(self) -> None:
        outcome = ls.run_sort_command(FakeSerial([b"REQUESTED:PAPER\n", b"STOPPED\n"]), "3", "PAPER")
        self.assertFalse(outcome.completed)

    def test_wrong_request_echo_is_a_protocol_error(self) -> None:
        with self.assertRaisesRegex(ls.SerialProtocolError, "'REQUESTED:PAPER' but 'REQUESTED:METAL' was expected"):
            ls.run_sort_command(FakeSerial([b"REQUESTED:PAPER\n"]), "1", "METAL")

    def test_busy_and_invalid_are_protocol_errors(self) -> None:
        with self.assertRaisesRegex(ls.SerialProtocolError, "BUSY"):
            ls.run_sort_command(FakeSerial([b"BUSY\n"]), "1", "METAL")
        with self.assertRaisesRegex(ls.SerialProtocolError, "invalid"):
            ls.run_sort_command(FakeSerial([b"INVALID COMMAND\n"]), "1", "METAL")

    def test_silent_stm32_is_a_protocol_error(self) -> None:
        with self.assertRaisesRegex(ls.SerialProtocolError, "no reply"):
            ls.run_sort_command(FakeSerial([]), "1", "METAL")

    def test_sort_that_never_finishes_is_a_protocol_error(self) -> None:
        port = FakeSerial([b"REQUESTED:METAL\n", b"GATE OPEN\n"])
        with self.assertRaisesRegex(ls.SerialProtocolError, "did not finish"):
            ls.run_sort_command(port, "1", "METAL")

    def test_message_split_across_reads_is_joined(self) -> None:
        port = FakeSerial([b"REQUESTED:", b"METAL\n", b"SORTING COMPLETE\n"])
        self.assertTrue(ls.run_sort_command(port, "1", "METAL").completed)

    def test_message_cut_off_by_silence_is_not_accepted(self) -> None:
        port = FakeSerial([b"REQUESTED:METAL"])  # no newline, then silence
        with self.assertRaisesRegex(ls.SerialProtocolError, "no reply"):
            ls.run_sort_command(port, "1", "METAL")

    def test_bin_lines_are_not_mistaken_for_replies(self) -> None:
        # The firmware reports the colour-sensor bin at any time; they must not break a sort.
        port = FakeSerial([b"BIN:NONE\n", b"REQUESTED:METAL\n", b"BIN:METAL\n", b"METAL\n", b"SORTING COMPLETE\n"])
        outcome = ls.run_sort_command(port, "1", "METAL")
        self.assertTrue(outcome.completed)
        self.assertNotIn("BIN:METAL", outcome.lines)

    def test_check_stm32_skips_boot_lines_and_accepts_status(self) -> None:
        port = FakeSerial([b"SYSTEM READY\n", b"LAST_SAVED:METAL\n"])
        self.assertEqual(ls.check_stm32(port), "LAST_SAVED:METAL")
        self.assertEqual(port.written, [b"STATUS\n"])

    def test_check_stm32_without_reply_is_an_error(self) -> None:
        with self.assertRaisesRegex(ls.SerialProtocolError, "did not answer STATUS"):
            ls.check_stm32(FakeSerial([]))

    def test_stop_sends_stop_and_reports_stopped(self) -> None:
        port = FakeSerial([b"STOPPED\n"])
        self.assertTrue(ls.stop_stm32(port))
        self.assertEqual(port.written, [b"STOP\n"])

    def test_stop_never_raises_when_the_port_is_gone(self) -> None:
        port = FakeSerial([])
        port.write = mock.Mock(side_effect=OSError("port gone"))
        self.assertFalse(ls.stop_stm32(port))

    def test_open_serial_reports_missing_pyserial(self) -> None:
        settings = ls.SerialSettings(port="COM3", baud=115200, ack_timeout=2.0)
        with mock.patch.dict("sys.modules", {"serial": None}):
            with self.assertRaisesRegex(ls.SorterError, "pip install pyserial"):
                ls.open_serial(settings)


# --------------------------------------------------------------------------
# Presence and decisions
# --------------------------------------------------------------------------


class PresenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = ls.make_reference(platform())

    def test_empty_platform_has_no_object(self) -> None:
        self.assertFalse(ls.object_present(platform(), self.reference))

    def test_object_on_platform_is_detected(self) -> None:
        self.assertTrue(ls.object_present(frame_with_object(), self.reference))

    def test_lighting_change_alone_is_not_an_object(self) -> None:
        for factor in (0.8, 1.15):
            with self.subTest(factor=factor):
                self.assertFalse(ls.object_present(brighter(platform(), factor), self.reference))

    def test_frame_size_different_from_the_photo_still_works(self) -> None:
        big = cv2.resize(frame_with_object(), (640, 480))
        self.assertTrue(ls.object_present(big, self.reference))
        self.assertFalse(ls.object_present(cv2.resize(platform(), (640, 480)), self.reference))


class SortStateTests(unittest.TestCase):
    def test_sends_once_after_stable_frames(self) -> None:
        state = ls.SortState(threshold=0.7, stable_frames=3)
        decisions = [state.update(True, METAL) for _ in range(5)]
        self.assertEqual([d is not None for d in decisions], [False, False, True, False, False])
        self.assertEqual(decisions[2].class_name, "metal")

    def test_no_second_send_while_the_object_stays(self) -> None:
        state = ls.SortState(threshold=0.7, stable_frames=1)
        self.assertIsNotNone(state.update(True, METAL))
        for _ in range(10):
            self.assertIsNone(state.update(True, METAL))

    def test_empty_platform_rearms_for_the_next_object(self) -> None:
        state = ls.SortState(threshold=0.7, stable_frames=1)
        self.assertIsNotNone(state.update(True, METAL))
        state.update(False, None)
        self.assertEqual(state.update(True, PLASTIC).class_name, "plastic")

    def test_low_confidence_is_never_sent(self) -> None:
        state = ls.SortState(threshold=0.7, stable_frames=1)
        for _ in range(5):
            self.assertIsNone(state.update(True, UNSURE))

    def test_prediction_changing_class_restarts_the_count(self) -> None:
        state = ls.SortState(threshold=0.7, stable_frames=3)
        self.assertIsNone(state.update(True, METAL))
        self.assertIsNone(state.update(True, METAL))
        self.assertIsNone(state.update(True, PLASTIC))  # restarts at 1
        self.assertIsNone(state.update(True, PLASTIC))
        self.assertEqual(state.update(True, PLASTIC).class_name, "plastic")

    def test_confidence_exactly_at_threshold_counts(self) -> None:
        exact = prediction([0.7, 0.1, 0.1, 0.1])
        state = ls.SortState(threshold=0.7, stable_frames=1)
        self.assertIsNotNone(state.update(True, exact))

    def test_no_prediction_while_present_does_not_send(self) -> None:
        state = ls.SortState(threshold=0.7, stable_frames=1)
        self.assertIsNone(state.update(True, None))


# --------------------------------------------------------------------------
# Live loop
# --------------------------------------------------------------------------


class SortingLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = ls.make_reference(platform())
        self.object_frame = frame_with_object()
        self.empty = platform()

    def run_loop(self, frames, keys, *, stable=2, send=None, clock=None):
        source = FakeSource(frames)
        display = FakeDisplay(keys)
        state = ls.SortState(threshold=0.7, stable_frames=stable)
        sent: list[ls.Decision] = []

        def default_send(decision: ls.Decision) -> str:
            sent.append(decision)
            return ls.FIRMWARE_NAMES[decision.class_name]

        stats = ls.run_sorting_loop(
            lambda rgb: METAL,
            source,
            display,
            self.reference,
            state,
            send or default_send,
            clock=clock,
        )
        return stats, sent, source, display

    def test_one_object_gives_one_send(self) -> None:
        frames = [self.object_frame] * 6
        stats, sent, _, _ = self.run_loop(frames, [-1] * 5 + [Q], stable=2)
        self.assertEqual([d.class_name for d in sent], ["metal"])
        self.assertEqual(stats.sent, ["metal"])

    def test_removing_and_placing_an_object_sends_again(self) -> None:
        frames = [self.object_frame] * 3 + [self.empty] * 2 + [self.object_frame] * 3
        stats, sent, _, _ = self.run_loop(frames, [-1] * 8 + [Q], stable=2)
        self.assertEqual(len(sent), 2)

    def test_prediction_only_runs_while_an_object_is_present(self) -> None:
        calls: list[int] = []

        def predict(rgb):
            calls.append(1)
            return METAL

        source = FakeSource([self.empty, self.empty, self.object_frame])
        display = FakeDisplay([-1, -1, Q])
        ls.run_sorting_loop(predict, source, display, self.reference, ls.SortState(0.7, 5), lambda d: "metal")
        self.assertEqual(len(calls), 1)

    def test_a_failed_send_stops_the_loop_and_releases_everything(self) -> None:
        def bad_send(decision):
            raise ls.SerialProtocolError("the STM32 replied 'paper' but 'metal' was expected")

        source = FakeSource([self.object_frame] * 5)
        display = FakeDisplay([-1] * 10)
        state = ls.SortState(threshold=0.7, stable_frames=1)
        with self.assertRaises(ls.SerialProtocolError):
            ls.run_sorting_loop(lambda rgb: METAL, source, display, self.reference, state, bad_send)
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)
        self.assertEqual(len(source.items), 4)  # stopped at once, not still sending

    def test_q_and_escape_stop_the_loop(self) -> None:
        for key in (Q, ESC):
            with self.subTest(key=key):
                stats, _, source, display = self.run_loop([self.empty, self.empty], [-1, key])
                self.assertEqual(stats.frames, 2)
                self.assertEqual(source.released, 1)
                self.assertEqual(display.closed, 1)

    def test_camera_read_failures_raise_and_release(self) -> None:
        source = FakeSource([None] * 40)
        display = FakeDisplay([-1] * 40)
        state = ls.SortState(threshold=0.7, stable_frames=1)
        with self.assertRaises(CameraError):
            ls.run_sorting_loop(lambda rgb: METAL, source, display, self.reference, state, lambda d: "x", clock=ticking_clock(1.0))
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)

    def test_each_frame_is_drawn(self) -> None:
        _, _, _, display = self.run_loop([self.empty, self.object_frame], [-1, Q])
        self.assertEqual(len(display.shown), 2)
        self.assertFalse(np.array_equal(display.shown[0], display.shown[1]))


# --------------------------------------------------------------------------
# Command line (fake camera, window and serial port)
# --------------------------------------------------------------------------


class CommandLineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))  # image size stays 224, matching the checkpoint
        raw["training"].update(batch_size=8, num_workers=0, device="cpu", pretrained=False)
        raw["paths"] = {"dataset_dir": str(self.root / "dataset"), "model_dir": str(self.root / "models")}
        raw["dataset"] = {
            "group_metadata": str(self.root / "group_metadata.csv"),
            "manifest": str(self.root / "splits" / "manifest.csv"),
            "report": str(self.root / "splits" / "split_report.txt"),
        }
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        self.config: Config = load_config(self.config_path)

        template = self.root / "template"
        template.mkdir()
        cv2.imwrite(str(template / "empty.jpg"), platform(), [cv2.IMWRITE_JPEG_QUALITY, 95])
        for name in ("Metal", "Plastic", "Paper", "Organic"):
            cv2.imwrite(str(template / f"{name}.jpg"), frame_with_object(), [cv2.IMWRITE_JPEG_QUALITY, 95])

        self.serial_path = self.root / "serial.txt"
        self.serial_path.write_text("port = COM3\nbaud = 115200\nack_timeout = 2\n", encoding="utf-8")
        self.checkpoint = self.make_checkpoint()

    def make_checkpoint(self) -> Path:
        torch.manual_seed(0)
        model = build_model("mobilenet_v3_small", 4, pretrained=False)
        with torch.no_grad():
            model.classifier[-1].weight.zero_()  # type: ignore[index, union-attr]
            model.classifier[-1].bias.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))  # type: ignore[index, union-attr]
        path = self.config.paths.model_dir / "best_model.pth"
        save_checkpoint(
            path,
            model,
            {
                "architecture": "mobilenet_v3_small",
                "classes": list(CLASSES),
                "image_width": 224,
                "image_height": 224,
                "normalization": "imagenet",
                "pretrained": False,
                "seed": 0,
                "epoch": 0,
                "val_loss": 0.0,
                "val_accuracy": 0.0,
            },
        )
        return path

    def run_main(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ls.main(["--config", str(self.config_path), "--serial-config", str(self.serial_path), *argv])
        return code, out.getvalue()

    def camera_frames(self, frames: list[np.ndarray], keys: list[int]):
        source = FakeSource(frames)
        display = FakeDisplay(keys)
        return mock.patch("src.live_sorter.open_camera", return_value=source), mock.patch(
            "src.live_sorter.OpenCvWindow", return_value=display
        ), source, display


class DryRunTests(CommandLineTestCase):
    def test_dry_run_prints_the_code_and_never_opens_serial(self) -> None:
        open_cam, window, source, display = self.camera_frames([frame_with_object()] * 4, [-1, -1, -1, Q])
        with open_cam, window, mock.patch("src.live_sorter.open_serial") as opener:
            code, out = self.run_main("--checkpoint", str(self.checkpoint), "--dry-run", "--device", "cpu", "--stable-frames", "2")
        self.assertEqual(code, 0)
        opener.assert_not_called()
        self.assertIn("[dry run] would send 1 (metal", out)
        self.assertEqual(out.count("would send"), 1)
        self.assertEqual(source.released, 1)
        self.assertEqual(display.closed, 1)

    def test_dry_run_does_not_need_the_serial_file(self) -> None:
        self.serial_path.unlink()
        open_cam, window, _, _ = self.camera_frames([frame_with_object()] * 3, [-1, -1, Q])
        with open_cam, window:
            code, _ = self.run_main("--checkpoint", str(self.checkpoint), "--dry-run", "--device", "cpu", "--stable-frames", "2")
        self.assertEqual(code, 0)


class RealSerialTests(CommandLineTestCase):
    def test_sends_the_digit_and_newline_and_closes_the_port(self) -> None:
        port = FakeSerial([b"LAST_SAVED:NONE\n"] + list(COMPLETE_SORT) + [b"STOPPED\n"])
        open_cam, window, _, _ = self.camera_frames([frame_with_object()] * 4, [-1, -1, -1, Q])
        with open_cam, window, mock.patch("src.live_sorter.open_serial", return_value=port) as opener:
            code, out = self.run_main("--checkpoint", str(self.checkpoint), "--device", "cpu", "--stable-frames", "2")
        self.assertEqual(code, 0)
        opener.assert_called_once()
        self.assertEqual(port.written, [b"STATUS\n", b"1\n", b"STOP\n"])
        self.assertTrue(port.closed)
        self.assertIn("SORTING COMPLETE", out)

    def test_wrong_reply_exits_1_closes_the_port_and_sends_stop(self) -> None:
        port = FakeSerial([b"LAST_SAVED:NONE\n", b"REQUESTED:PLASTIC\n", b"STOPPED\n"])
        open_cam, window, source, _ = self.camera_frames([frame_with_object()] * 4, [-1] * 4 + [Q])
        with open_cam, window, mock.patch("src.live_sorter.open_serial", return_value=port):
            code, out = self.run_main("--checkpoint", str(self.checkpoint), "--device", "cpu", "--stable-frames", "2")
        self.assertEqual(code, 1)
        self.assertIn("'REQUESTED:PLASTIC' but 'REQUESTED:METAL' was expected", out)
        self.assertEqual(port.written, [b"STATUS\n", b"1\n", b"STOP\n"])  # no second command after the bad reply
        self.assertTrue(port.closed)
        self.assertEqual(source.released, 1)

    def test_silent_board_exits_1_before_the_camera_opens(self) -> None:
        port = FakeSerial([])
        with mock.patch("src.live_sorter.open_camera") as camera, mock.patch("src.live_sorter.open_serial", return_value=port):
            code, out = self.run_main("--checkpoint", str(self.checkpoint), "--device", "cpu")
        self.assertEqual(code, 1)
        self.assertIn("did not answer STATUS", out)
        camera.assert_not_called()
        self.assertTrue(port.closed)

    def test_missing_serial_settings_exit_2_before_anything_opens(self) -> None:
        self.serial_path.write_text("port = \nbaud = \n", encoding="utf-8")
        with mock.patch("src.live_sorter.open_camera") as camera:
            code, out = self.run_main("--checkpoint", str(self.checkpoint), "--device", "cpu")
        self.assertEqual(code, 2)
        self.assertIn("set the serial port", out)
        camera.assert_not_called()

    def test_serial_port_that_cannot_open_exits_1(self) -> None:
        with mock.patch("src.live_sorter.open_camera") as camera, mock.patch(
            "src.live_sorter.open_serial", side_effect=ls.SorterError("could not open serial port COM3: busy")
        ):
            code, out = self.run_main("--checkpoint", str(self.checkpoint), "--device", "cpu")
        self.assertEqual(code, 1)
        self.assertIn("could not open serial port COM3", out)
        camera.assert_not_called()


class StartUpFailureTests(CommandLineTestCase):
    def test_missing_template_folder_exits_1(self) -> None:
        for item in (self.root / "template").iterdir():
            item.unlink()
        (self.root / "template").rmdir()
        with mock.patch("src.live_sorter.open_camera") as camera:
            code, out = self.run_main("--checkpoint", str(self.checkpoint), "--dry-run", "--device", "cpu")
        self.assertEqual(code, 1)
        self.assertIn("template folder not found", out)
        camera.assert_not_called()

    def test_missing_checkpoint_exits_1(self) -> None:
        with mock.patch("src.live_sorter.open_camera") as camera:
            code, out = self.run_main("--checkpoint", str(self.root / "nope.pth"), "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("checkpoint not found", out)
        camera.assert_not_called()

    def test_camera_that_cannot_open_exits_1_and_closes_serial(self) -> None:
        port = FakeSerial([b"LAST_SAVED:NONE\n", b"STOPPED\n"])
        with mock.patch("src.live_sorter.open_serial", return_value=port), mock.patch(
            "src.live_sorter.open_camera", side_effect=CameraError("camera index 0 failed")
        ):
            code, out = self.run_main("--checkpoint", str(self.checkpoint), "--device", "cpu")
        self.assertEqual(code, 1)
        self.assertIn("camera index 0 failed", out)
        self.assertTrue(port.closed)

    def test_bad_arguments_exit_2(self) -> None:
        for argv in (["--stable-frames", "0"], ["--camera", "-1"], ["--threshold", "1.5"]):
            with self.subTest(argv=argv):
                code, _ = self.run_main("--checkpoint", str(self.checkpoint), "--dry-run", *argv)
                self.assertEqual(code, 2)

    def test_cups_outside_one_to_four_exit_2(self) -> None:
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        raw["cups"]["other"] = 7
        self.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        code, out = self.run_main("--checkpoint", str(self.checkpoint), "--dry-run")
        self.assertEqual(code, 2)
        self.assertIn("1 to 4 only", out)

    def test_threshold_defaults_to_the_config_value(self) -> None:
        open_cam, window, _, _ = self.camera_frames([frame_with_object()] * 2, [-1, Q])
        with open_cam, window:
            _, out = self.run_main("--checkpoint", str(self.checkpoint), "--dry-run", "--device", "cpu")
        self.assertIn(f"Threshold: {self.config.confidence_threshold:g}", out)


class ImportTests(unittest.TestCase):
    def test_importing_the_module_opens_nothing(self) -> None:
        # Checked in a fresh interpreter: reloading the module here would replace its
        # exception classes, which other test modules still refer to.
        script = (
            "import cv2\n"
            "calls = []\n"
            "cv2.VideoCapture = lambda *args, **kwargs: calls.append(args)\n"
            "import src.live_sorter\n"
            "print(len(calls))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=180
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0")


if __name__ == "__main__":
    unittest.main()
