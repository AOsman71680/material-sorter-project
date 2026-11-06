"""Tests for the empty-platform check. Uses fake cameras and generated photos only."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml

from src import check_empty as ce
from src.config import DEFAULT_CONFIG_PATH
from src.live_sorter import make_reference, object_present


def scene(brightness: float = 1.0, blob: bool = False) -> np.ndarray:
    rng = np.random.default_rng(3)
    img = np.full((240, 320, 3), (120, 135, 150), np.float32)
    img += cv2.GaussianBlur(rng.normal(0, 4, (240, 320)).astype(np.float32), (0, 0), 2)[..., None]
    if blob:
        img[60:160, 100:220] = (30, 40, 200)
    return np.clip(img * brightness, 0, 255).astype(np.uint8)


class FakeCamera:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.released = False

    def read(self):  # type: ignore[no-untyped-def]
        return True, self.frame.copy()

    def release(self) -> None:
        self.released = True


class MeasurementTests(unittest.TestCase):
    def test_fraction_matches_the_sorters_own_decision(self) -> None:
        reference = make_reference(scene())
        for frame in (scene(), scene(1.05), scene(blob=True), scene(1.5)):
            fraction = ce.changed_fraction(frame, reference)
            self.assertEqual(fraction > ce.PRESENCE_FRACTION, object_present(frame, reference))

    def test_verdict_ok_when_empty_is_recognised(self) -> None:
        text = "\n".join(ce.verdict([0.0, 0.001], 0.5, (320, 240), (320, 240)))
        self.assertIn("OK", text)

    def test_verdict_points_to_exposure_when_the_picture_is_much_brighter(self) -> None:
        text = "\n".join(ce.verdict([0.9, 0.95], 40.0, (320, 240), (320, 240)))
        self.assertIn("PROBLEM", text)
        self.assertIn("brighter", text)

    def test_verdict_points_to_a_different_camera_when_the_shape_differs(self) -> None:
        text = "\n".join(ce.verdict([0.5], 0.0, (320, 240), (640, 360)))
        self.assertIn("different shape", text)

    def test_verdict_reports_flicker_when_only_some_frames_fail(self) -> None:
        text = "\n".join(ce.verdict([0.0, 0.0, 0.0, 0.5], 0.0, (320, 240), (320, 240)))
        self.assertIn("flickering", text)


class MainTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["paths"] = {"dataset_dir": str(self.root / "dataset"), "model_dir": str(self.root / "models")}
        raw["dataset"] = {k: str(self.root / v) for k, v in {"group_metadata": "g.csv", "manifest": "m.csv", "report": "r.txt"}.items()}
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        template = self.root / "template"
        template.mkdir()
        cv2.imwrite(str(template / "empty.jpg"), scene())
        for name in ("Metal", "Plastic", "Paper", "Organic"):
            cv2.imwrite(str(template / f"{name}.jpg"), scene(blob=True))
        self.output = self.root / "check.png"

    def run_main(self, camera: FakeCamera) -> tuple[int, str]:
        out = io.StringIO()
        args = ["--config", str(self.config_path), "--seconds", "0.2", "--output", str(self.output)]
        with mock.patch.object(ce, "open_camera", return_value=camera), contextlib.redirect_stdout(out):
            code = ce.main(args)
        return code, out.getvalue()

    def test_matching_camera_is_ok_and_a_picture_is_saved(self) -> None:
        camera = FakeCamera(scene(1.02))
        code, out = self.run_main(camera)
        self.assertEqual(code, 0)
        self.assertIn("OK", out)
        self.assertTrue(self.output.is_file())
        self.assertIsNotNone(cv2.imread(str(self.output)))
        self.assertTrue(camera.released)

    def test_a_camera_that_sees_something_else_is_reported_as_a_problem(self) -> None:
        code, out = self.run_main(FakeCamera(scene(blob=True)))
        self.assertEqual(code, 0)
        self.assertIn("PROBLEM", out)

    def test_missing_templates_exit_1(self) -> None:
        (self.root / "template" / "empty.jpg").unlink()
        code, out = self.run_main(FakeCamera(scene()))
        self.assertEqual(code, 1)
        self.assertIn("error:", out)

    def test_camera_that_never_delivers_frames_gives_up(self) -> None:
        class Dead:
            released = False

            def read(self):  # type: ignore[no-untyped-def]
                return False, None

            def release(self) -> None:
                self.released = True

        dead = Dead()
        with mock.patch.object(ce, "CAMERA_FAILURE_TIMEOUT_S", 0.1):
            code, out = self.run_main(dead)  # type: ignore[arg-type]
        self.assertEqual(code, 1)
        self.assertIn("stopped delivering", out)
        self.assertTrue(dead.released)


if __name__ == "__main__":
    unittest.main()
