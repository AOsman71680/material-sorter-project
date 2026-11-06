"""Tests for src.data_collection.

No camera or GUI is used: the capture loop is driven by fake frame sources and
fake displays, and camera opening is tested with a mocked cv2.VideoCapture.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from helpers import snapshot
from src import data_collection as dc
from src.config import DEFAULT_CONFIG_PATH, MATERIAL_CLASSES

CLASSES = MATERIAL_CLASSES  # metal, plastic, paper, other
SPACE, ESC = 32, 27


def solid_frame(color: tuple[int, int, int] = (10, 200, 30), size: tuple[int, int] = (60, 80)) -> np.ndarray:
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:] = color
    return frame


class FakeSource:
    """Yields scripted (ok, frame) results, repeating the last one forever."""

    def __init__(self, results: list[tuple[bool, np.ndarray | None] | Exception]) -> None:
        self.results = results
        self.reads = 0
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        item = self.results[min(self.reads, len(self.results) - 1)]
        self.reads += 1
        if isinstance(item, Exception):
            raise item
        return item

    def release(self) -> None:
        self.released = True


class FakeDisplay:
    """Returns scripted keys, then 'q' so a loop can never run forever."""

    def __init__(self, keys: list[int], open_after: int | None = None) -> None:
        self.keys = list(keys)
        self.open_after = open_after  # window "closed" after this many shows
        self.shown: list[np.ndarray] = []
        self.closed = False

    def show(self, image: np.ndarray) -> None:
        self.shown.append(image)

    def wait_key(self, delay_ms: int) -> int:
        return self.keys.pop(0) if self.keys else ord("q")

    def is_open(self) -> bool:
        return self.open_after is None or len(self.shown) < self.open_after

    def close(self) -> None:
        self.closed = True


class TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.dataset = self.root / "dataset"


class FrameValidationTests(unittest.TestCase):
    def test_valid_frame(self) -> None:
        self.assertTrue(dc.is_valid_frame(solid_frame()))

    def test_invalid_frames(self) -> None:
        bad: list[object] = [
            None,
            "frame",
            np.zeros((0, 10, 3), dtype=np.uint8),
            np.zeros((10, 0, 3), dtype=np.uint8),
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10, 4), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.float32),
        ]
        for frame in bad:
            with self.subTest(frame=type(frame).__name__):
                self.assertFalse(dc.is_valid_frame(frame))


class SaveFrameTests(TempDirTestCase):
    def test_saves_decodable_image_with_class_prefix(self) -> None:
        path = dc.save_frame(solid_frame(), self.dataset / "metal", "metal")
        self.assertEqual(path.parent, self.dataset / "metal")
        self.assertTrue(path.name.startswith("metal_") and path.suffix == ".jpg")
        decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (60, 80, 3))

    def test_invalid_frames_raise_and_save_nothing(self) -> None:
        for frame in (None, np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((5, 5), dtype=np.uint8)):
            with self.subTest(frame=type(frame).__name__):
                with self.assertRaises(dc.InvalidFrameError):
                    dc.save_frame(frame, self.dataset / "metal", "metal")
        self.assertEqual(list((self.dataset / "metal").glob("*")) if (self.dataset / "metal").exists() else [], [])

    def test_same_timestamp_never_overwrites(self) -> None:
        when = datetime(2026, 1, 2, 3, 4, 5, 678)
        first = dc.save_frame(solid_frame((1, 1, 1)), self.dataset / "paper", "paper", now=when)
        original = first.read_bytes()
        second = dc.save_frame(solid_frame((250, 250, 250)), self.dataset / "paper", "paper", now=when)
        third = dc.save_frame(solid_frame((120, 0, 0)), self.dataset / "paper", "paper", now=when)
        self.assertEqual(len({first, second, third}), 3)
        self.assertEqual(first.read_bytes(), original)

    def test_existing_unrelated_file_with_same_name_is_preserved(self) -> None:
        when = datetime(2026, 1, 2, 3, 4, 5, 678)
        target = self.dataset / "other" / f"other_{when.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"KEEP ME")
        saved = dc.save_frame(solid_frame(), target.parent, "other", now=when)
        self.assertNotEqual(saved, target)
        self.assertEqual(target.read_bytes(), b"KEEP ME")

    def test_gives_up_instead_of_overwriting_when_names_exhausted(self) -> None:
        when = datetime(2026, 1, 2, 3, 4, 5, 678)
        first = dc.save_frame(solid_frame(), self.dataset / "metal", "metal", now=when)
        original = first.read_bytes()
        with mock.patch.object(dc, "MAX_NAME_ATTEMPTS", 1):
            with self.assertRaises(OSError):
                dc.save_frame(solid_frame((9, 9, 9)), self.dataset / "metal", "metal", now=when)
        self.assertEqual(first.read_bytes(), original)

    def test_partial_file_is_removed_when_write_fails(self) -> None:
        real_open = Path.open

        class FailingHandle:
            def __init__(self, handle) -> None:  # type: ignore[no-untyped-def]
                self.handle = handle

            def __enter__(self) -> "FailingHandle":
                return self

            def __exit__(self, *exc: object) -> None:
                self.handle.close()

            def write(self, data: bytes) -> int:
                self.handle.write(b"partial")
                raise OSError("disk full")

        with mock.patch.object(Path, "open", lambda self, mode="r", *a, **k: FailingHandle(real_open(self, mode))):
            with self.assertRaises(OSError):
                dc.save_frame(solid_frame(), self.dataset / "metal", "metal")
        self.assertEqual(list((self.dataset / "metal").iterdir()), [])


class SessionTests(TempDirTestCase):
    def test_creates_class_dirs_and_counts_existing_images(self) -> None:
        (self.dataset / "metal").mkdir(parents=True)
        (self.dataset / "metal" / "a.jpg").write_bytes(b"x")
        (self.dataset / "metal" / "notes.txt").write_text("x")
        session = dc.CollectionSession(CLASSES, self.dataset)
        for name in CLASSES:
            self.assertTrue((self.dataset / name).is_dir())
        self.assertEqual(session.counts, {"metal": 1, "plastic": 0, "paper": 0, "other": 0})
        self.assertIsNone(session.selected)

    def test_select_and_capture(self) -> None:
        session = dc.CollectionSession(CLASSES, self.dataset)
        self.assertTrue(session.select(2))
        path = session.capture(solid_frame())
        self.assertEqual(session.selected, "paper")
        self.assertEqual(path.parent, self.dataset / "paper")
        self.assertEqual(session.counts["paper"], 1)

    def test_capture_requires_selected_class(self) -> None:
        session = dc.CollectionSession(CLASSES, self.dataset)
        with self.assertRaises(dc.NoClassSelectedError):
            session.capture(solid_frame())
        self.assertEqual(sum(session.counts.values()), 0)

    def test_invalid_frame_does_not_change_count(self) -> None:
        session = dc.CollectionSession(CLASSES, self.dataset)
        session.select(0)
        with self.assertRaises(dc.InvalidFrameError):
            session.capture(None)
        self.assertEqual(session.counts["metal"], 0)

    def test_select_out_of_range(self) -> None:
        session = dc.CollectionSession(CLASSES, self.dataset)
        self.assertFalse(session.select(4))
        self.assertFalse(session.select(-1))
        self.assertIsNone(session.selected)

    def test_class_count_limits(self) -> None:
        with self.assertRaises(ValueError):
            dc.CollectionSession((), self.dataset)
        with self.assertRaises(ValueError):
            dc.CollectionSession(tuple(f"c{i}" for i in range(10)), self.dataset)

    def test_key_to_class_index(self) -> None:
        self.assertEqual([dc.class_index_for_key(ord(k), 4) for k in "1234"], [0, 1, 2, 3])
        for key in (ord("5"), ord("0"), -1, ord("a")):
            self.assertIsNone(dc.class_index_for_key(key, 4))


class OverlayTests(TempDirTestCase):
    def test_overlay_does_not_modify_input_and_keeps_shape(self) -> None:
        session = dc.CollectionSession(CLASSES, self.dataset)
        frame = solid_frame(size=(240, 320))
        original = frame.copy()
        canvas = dc.draw_overlay(frame, session, "hello")
        self.assertEqual(canvas.shape, frame.shape)
        np.testing.assert_array_equal(frame, original)
        self.assertFalse(np.array_equal(canvas, frame))  # text was drawn

    def test_placeholder_when_no_frame(self) -> None:
        session = dc.CollectionSession(CLASSES, self.dataset)
        canvas = dc.draw_overlay(None, session, None)
        self.assertEqual(canvas.shape, (*dc.PLACEHOLDER_SIZE, 3))


class RunCollectionTests(TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = dc.CollectionSession(CLASSES, self.dataset)

    def files(self, name: str) -> list[Path]:
        return sorted((self.dataset / name).glob("*.jpg"))

    def test_select_capture_quit_saves_raw_frame_in_right_class(self) -> None:
        frame = solid_frame((10, 200, 30))
        source = FakeSource([(True, frame)])
        display = FakeDisplay([ord("2"), SPACE, SPACE, ord("q")])
        dc.run_collection(self.session, source, display)

        saved = self.files("plastic")
        self.assertEqual(len(saved), 2)
        self.assertEqual(self.session.counts["plastic"], 2)
        for name in ("metal", "paper", "other"):
            self.assertEqual(self.files(name), [])
        decoded = cv2.imdecode(np.fromfile(saved[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertLess(int(np.abs(decoded.astype(int) - frame.astype(int)).max()), 8)  # no overlay text
        self.assertTrue(source.released)
        self.assertTrue(display.closed)

    def test_capture_key_c_and_esc_quit(self) -> None:
        display = FakeDisplay([ord("1"), ord("c"), ESC, ord("c")])
        dc.run_collection(self.session, FakeSource([(True, solid_frame())]), display)
        self.assertEqual(len(self.files("metal")), 1)  # the capture after Esc never ran

    def test_capture_without_selected_class_saves_nothing(self) -> None:
        before = snapshot(self.root)
        display = FakeDisplay([SPACE, ord("q")])
        dc.run_collection(self.session, FakeSource([(True, solid_frame())]), display)
        self.assertEqual(snapshot(self.root), before)

    def test_switching_class_changes_destination(self) -> None:
        display = FakeDisplay([ord("1"), SPACE, ord("4"), SPACE, ord("q")])
        dc.run_collection(self.session, FakeSource([(True, solid_frame())]), display)
        self.assertEqual((len(self.files("metal")), len(self.files("other"))), (1, 1))

    def test_unknown_keys_are_ignored(self) -> None:
        display = FakeDisplay([ord("9"), ord("x"), -1, SPACE, ord("q")])
        dc.run_collection(self.session, FakeSource([(True, solid_frame())]), display)
        self.assertEqual(sum(self.session.counts.values()), 0)

    def test_transient_bad_frames_are_survived_and_not_saved(self) -> None:
        source = FakeSource(
            [
                (True, solid_frame()),
                (False, None),
                (True, None),
                (True, np.zeros((0, 0, 3), dtype=np.uint8)),
                cv2.error("simulated driver error"),
                (True, solid_frame()),
            ]
        )
        # select a class on frame 1, press capture on each bad frame (2-5), then quit on frame 6
        display = FakeDisplay([ord("1"), SPACE, SPACE, SPACE, SPACE, ord("q")])
        dc.run_collection(self.session, source, display)
        self.assertEqual(self.files("metal"), [])
        self.assertTrue(source.released and display.closed)

    def test_persistent_failure_raises_camera_error_and_cleans_up(self) -> None:
        source = FakeSource([(False, None)])
        display = FakeDisplay([-1] * 50)  # no key presses
        ticks = iter(range(100))
        with self.assertRaises(dc.CameraError):
            dc.run_collection(self.session, source, display, clock=lambda: float(next(ticks)))
        self.assertTrue(source.released)
        self.assertTrue(display.closed)

    def test_recovery_resets_failure_timer(self) -> None:
        results: list[tuple[bool, np.ndarray | None]] = [(False, None), (False, None), (True, solid_frame())] * 3
        source = FakeSource(results)
        display = FakeDisplay([-1] * 8 + [ord("q")])
        ticks = iter(float(i) for i in range(1000))  # 1 s per failed read; timeout is 3 s
        dc.run_collection(self.session, source, display, clock=lambda: next(ticks))  # must not raise

    def test_closing_window_exits_cleanly(self) -> None:
        source = FakeSource([(True, solid_frame())])
        display = FakeDisplay([-1] * 50, open_after=3)
        dc.run_collection(self.session, source, display)
        self.assertLessEqual(source.reads, 3)
        self.assertTrue(source.released and display.closed)

    def test_save_failure_does_not_crash_loop(self) -> None:
        display = FakeDisplay([ord("1"), SPACE, ord("q")])
        with mock.patch.object(dc, "save_frame", side_effect=PermissionError(13, "Permission denied")):
            dc.run_collection(self.session, FakeSource([(True, solid_frame())]), display)
        self.assertEqual(self.session.counts["metal"], 0)
        self.assertTrue(display.closed)


class OpenCameraTests(unittest.TestCase):
    """Mocked cv2.VideoCapture: checks error handling only, not real hardware."""

    def fake_capture(self, opened: bool, frame: np.ndarray | None) -> mock.Mock:
        cap = mock.Mock()
        cap.isOpened.return_value = opened
        cap.read.return_value = (frame is not None, frame)
        return cap

    def test_unopenable_camera_raises_helpful_error(self) -> None:
        cap = self.fake_capture(False, None)
        with mock.patch.object(dc.cv2, "VideoCapture", return_value=cap), mock.patch.object(dc.time, "sleep"):
            with self.assertRaisesRegex(dc.CameraError, "camera index 3"):
                dc.open_camera(3)
        cap.release.assert_called()

    def test_camera_that_opens_but_gives_no_frames_raises_and_releases(self) -> None:
        cap = self.fake_capture(True, None)
        with mock.patch.object(dc.cv2, "VideoCapture", return_value=cap), mock.patch.object(dc.time, "sleep"):
            with self.assertRaises(dc.CameraError):
                dc.open_camera(0)
        cap.release.assert_called()

    def test_working_camera_is_returned(self) -> None:
        cap = self.fake_capture(True, solid_frame())
        with mock.patch.object(dc.cv2, "VideoCapture", return_value=cap), mock.patch.object(dc.time, "sleep"):
            self.assertIs(dc.open_camera(0), cap)


class MainTests(TempDirTestCase):
    def make_project(self) -> Path:
        (self.root / "config").mkdir()
        path = self.root / "config" / "config.yaml"
        shutil.copy(DEFAULT_CONFIG_PATH, path)
        return path

    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = dc.main(list(args))
        return code, out.getvalue()

    def test_bad_config_returns_2(self) -> None:
        code, out = self.run_main("--config", str(self.root / "nope.yaml"))
        self.assertEqual(code, 2)
        self.assertIn("config file not found", out)

    def test_negative_camera_index_returns_2(self) -> None:
        code, out = self.run_main("--config", str(self.make_project()), "--camera-index", "-1")
        self.assertEqual(code, 2)

    def test_camera_failure_returns_1_with_message(self) -> None:
        config_path = self.make_project()
        with mock.patch.object(dc, "open_camera", side_effect=dc.CameraError("no camera")):
            code, out = self.run_main("--config", str(config_path))
        self.assertEqual(code, 1)
        self.assertIn("camera error: no camera", out)

    def test_camera_index_override_is_used(self) -> None:
        config_path = self.make_project()
        with mock.patch.object(dc, "open_camera", side_effect=dc.CameraError("x")) as opener:
            self.run_main("--config", str(config_path), "--camera-index", "2")
        opener.assert_called_once_with(2)

    def test_config_camera_index_is_default(self) -> None:
        config_path = self.make_project()
        with mock.patch.object(dc, "open_camera", side_effect=dc.CameraError("x")) as opener:
            self.run_main("--config", str(config_path))
        opener.assert_called_once_with(dc.load_config(config_path).camera_index)

    def test_full_run_with_fakes_cleans_up_and_reports_counts(self) -> None:
        config_path = self.make_project()
        source = FakeSource([(True, solid_frame())])
        display = FakeDisplay([ord("3"), SPACE, ord("q")])
        with mock.patch.object(dc, "open_camera", return_value=source), mock.patch.object(
            dc, "OpenCvWindow", return_value=display
        ):
            code, out = self.run_main("--config", str(config_path))
        self.assertEqual(code, 0)
        self.assertIn("paper=1", out)
        self.assertEqual(len(list((self.root / "data" / "dataset" / "paper").glob("*.jpg"))), 1)
        self.assertTrue(source.released and display.closed)

    def test_window_creation_failure_releases_camera(self) -> None:
        config_path = self.make_project()
        source = FakeSource([(True, solid_frame())])
        with mock.patch.object(dc, "open_camera", return_value=source), mock.patch.object(
            dc, "OpenCvWindow", side_effect=cv2.error("no GUI")
        ):
            code, out = self.run_main("--config", str(config_path))
        self.assertEqual(code, 1)
        self.assertTrue(source.released)


if __name__ == "__main__":
    unittest.main()
