"""Tests for the sorter window's logic: bars, the serial link, sorting, the session, and the window itself.

The window test runs only where tkinter can open a display; elsewhere it is skipped.
"""

from __future__ import annotations

import contextlib
import io
import threading
import time
import unittest
from unittest import mock

import numpy as np

from src import sorter_gui as sg
from src.live_sorter import Decision, SerialProtocolError, SortOutcome, SortState
from src.webcam_predict import LivePrediction

try:
    import tkinter as tk

    TK_IMPORT_ERROR: str | None = None
except ImportError as exc:  # no tkinter in this Python
    tk = None  # type: ignore[assignment]
    TK_IMPORT_ERROR = str(exc)

CODES = {"metal": b"1", "plastic": b"2", "paper": b"3", "other": b"4"}


class FakeLinePort:
    """Stands in for pyserial. Lines can be queued, or sent in reply to a write."""

    def __init__(self, lines: list[bytes] | None = None, replies: dict[bytes, list[bytes]] | None = None) -> None:
        self.lines = list(lines or [])
        self.replies = replies or {}
        self.written: list[bytes] = []
        self.timeout = 0.01
        self.closed = False

    def readline(self) -> bytes:
        if self.lines:
            return self.lines.pop(0)
        time.sleep(0.003)
        return b""

    def write(self, data: bytes) -> None:
        self.written.append(data)
        self.lines.extend(self.replies.get(data, []))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class RaisingPort(FakeLinePort):
    def readline(self) -> bytes:
        raise OSError("port unplugged")


def wait_until(predicate, seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def prediction(name: str, probability: float) -> LivePrediction:
    return LivePrediction(
        predicted_class=name,
        max_probability=probability,
        probabilities={"metal": 0.5, "plastic": 0.2, "paper": 0.2, "other": 0.1},
        threshold=None,
        low_confidence=None,
    )


# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------


class BarTests(unittest.TestCase):
    def test_nothing_on_the_platform(self) -> None:
        view = sg.material_view(False, None, 0.7)
        self.assertEqual(view.text, "Nothing on the platform")
        self.assertEqual(view.fraction, 0.0)
        self.assertIsNone(view.material)

    def test_confident_material_shows_firmware_name_and_percentage(self) -> None:
        view = sg.material_view(True, prediction("plastic", 0.91), 0.7)
        self.assertEqual(view.text, "PLASTIC  91%")
        self.assertEqual(view.material, "PLASTIC")
        self.assertAlmostEqual(view.fraction, 0.91)

    def test_organic_class_is_shown_as_organic(self) -> None:
        self.assertEqual(sg.material_view(True, prediction("other", 0.8), 0.7).material, "ORGANIC")

    def test_unsure_prediction_has_no_material_colour(self) -> None:
        view = sg.material_view(True, prediction("paper", 0.5), 0.7)
        self.assertTrue(view.text.startswith("Unsure: PAPER 50%"))
        self.assertIsNone(view.material)

    def test_bin_not_connected(self) -> None:
        self.assertIsNone(sg.bin_view(False, "METAL").active)

    def test_bin_waiting_for_first_reading(self) -> None:
        self.assertEqual(sg.bin_view(True, None).text, "waiting for the colour sensor...")

    def test_bin_nothing_in_front(self) -> None:
        view = sg.bin_view(True, sg.NO_BIN)
        self.assertIsNone(view.active)
        self.assertEqual(view.text, "nothing in front of the sensor")

    def test_bin_over_each_known_bin_lights_its_segment(self) -> None:
        for index, name in enumerate(sg.BIN_ORDER):
            self.assertEqual(sg.bin_view(True, name).active, index)

    def test_unknown_bin_name_is_shown_but_not_highlighted(self) -> None:
        view = sg.bin_view(True, "GLASS")
        self.assertIsNone(view.active)
        self.assertEqual(view.text, "reads GLASS")


# --------------------------------------------------------------------------
# Serial link and sorting
# --------------------------------------------------------------------------


class SerialLinkTests(unittest.TestCase):
    def test_bin_lines_update_the_bin_and_other_lines_go_to_the_inbox(self) -> None:
        port = FakeLinePort([b"BIN:METAL\n", b"SYSTEM READY\n", b"BIN:NONE\n"])
        link = sg.SerialLink(port)
        link.start()
        self.addCleanup(link.stop)
        self.assertEqual(link.next_line(2.0), "SYSTEM READY")
        self.assertTrue(wait_until(lambda: link.current_bin() == "NONE"))

    def test_bin_is_unknown_until_the_first_report(self) -> None:
        link = sg.SerialLink(FakeLinePort())
        self.assertIsNone(link.current_bin())

    def test_port_failure_is_reported_as_lost(self) -> None:
        link = sg.SerialLink(RaisingPort())
        link.start()
        self.addCleanup(link.stop)
        self.assertTrue(wait_until(lambda: link.lost))

    def test_sort_ignores_bin_lines_that_arrive_during_it(self) -> None:
        port = FakeLinePort(
            replies={
                b"1\n": [b"BIN:PLASTIC\n", b"REQUESTED:METAL\n", b"BIN:METAL\n", b"METAL\n", b"POSITION CONFIRMED\n", b"SORTING COMPLETE\n"],
            }
        )
        link = sg.SerialLink(port)
        link.start()
        self.addCleanup(link.stop)
        outcome = sg.run_sort_command(sg.SortPort(link), "1", "METAL")
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.lines, ("REQUESTED:METAL", "METAL", "POSITION CONFIRMED", "SORTING COMPLETE"))
        self.assertTrue(wait_until(lambda: link.current_bin() == "METAL"))

    def test_shut_down_sends_stop_and_closes_the_port(self) -> None:
        port = FakeLinePort(replies={b"STOP\n": [b"STOPPED\n"]})
        link = sg.SerialLink(port)
        link.start()
        sg.shut_down(port, link, None)
        self.assertIn(b"STOP\n", port.written)
        self.assertTrue(port.closed)

    def test_shut_down_with_nothing_open_is_safe(self) -> None:
        sg.shut_down(None, None, None)  # must not raise


class SorterWorkerTests(unittest.TestCase):
    def test_second_start_while_busy_sends_nothing(self) -> None:
        release = threading.Event()

        def slow_sort(port, code, name):
            release.wait(2.0)
            return SortOutcome(True, "SORTING COMPLETE", ("SORTING COMPLETE",))

        worker = sg.SortWorker(sg.SortPort(sg.SerialLink(FakeLinePort())), sort_fn=slow_sort)
        self.assertTrue(worker.start("1", "METAL"))
        self.assertFalse(worker.start("2", "PLASTIC"))
        release.set()
        worker.wait(2.0)
        self.assertFalse(worker.busy)

    def test_completed_sort_is_logged(self) -> None:
        worker = sg.SortWorker(sg.SortPort(sg.SerialLink(FakeLinePort())), sort_fn=lambda p, c, n: SortOutcome(True, "SORTING COMPLETE", ()))
        sender = sg.StmSender(CODES, worker)
        self.assertEqual(sender(Decision("metal", 0.9)), "sent 1 (METAL), waiting for the STM32")
        self.assertTrue(wait_until(lambda: not worker.busy and not worker.results.empty()))
        sender.poll()
        self.assertEqual(list(sender.log), ["STM32: SORTING COMPLETE"])

    def test_timeout_is_logged_as_not_completed(self) -> None:
        worker = sg.SortWorker(
            sg.SortPort(sg.SerialLink(FakeLinePort())),
            sort_fn=lambda p, c, n: SortOutcome(False, "POSITION TIMEOUT", ("POSITION TIMEOUT",)),
        )
        sender = sg.StmSender(CODES, worker)
        sender(Decision("paper", 0.9))
        self.assertTrue(wait_until(lambda: not worker.busy and not worker.results.empty()))
        sender.poll()
        self.assertEqual(list(sender.log), ["STM32: POSITION TIMEOUT (not completed)"])
        self.assertIsNone(sender.halted_reason)

    def test_protocol_error_halts_all_further_sends(self) -> None:
        def broken(port, code, name):
            raise SerialProtocolError("the STM32 replied 'BUSY'. Sorting has stopped.")

        worker = sg.SortWorker(sg.SortPort(sg.SerialLink(FakeLinePort())), sort_fn=broken)
        sender = sg.StmSender(CODES, worker)
        sender(Decision("metal", 0.9))
        self.assertTrue(wait_until(lambda: not worker.busy and not worker.results.empty()))
        sender.poll()
        self.assertIn("BUSY", sender.halted_reason or "")
        self.assertTrue(sender(Decision("plastic", 0.9)).startswith("not sent:"))

    def test_unexpected_exception_is_reported_not_raised(self) -> None:
        def crashing(port, code, name):
            raise ValueError("bug")

        worker = sg.SortWorker(sg.SortPort(sg.SerialLink(FakeLinePort())), sort_fn=crashing)
        sender = sg.StmSender(CODES, worker)
        sender(Decision("metal", 0.9))
        self.assertTrue(wait_until(lambda: not worker.busy and not worker.results.empty()))
        sender.poll()
        self.assertIn("unexpected problem", sender.halted_reason or "")

    def test_dry_run_only_describes_the_command(self) -> None:
        sender = sg.StmSender(CODES, None)
        self.assertEqual(sender(Decision("other", 0.9)), "dry run: would send 4 (ORGANIC)")
        sender.poll()  # no worker: nothing to read
        self.assertEqual(list(sender.log), [])


# --------------------------------------------------------------------------
# Session: one frame at a time
# --------------------------------------------------------------------------


class FakeSource:
    def __init__(self, frames: list[np.ndarray | None]) -> None:
        self.frames = list(frames)
        self.released = 0

    def read(self):
        if not self.frames:
            return False, None
        frame = self.frames.pop(0)
        return (frame is not None), frame

    def release(self) -> None:
        self.released += 1


def empty_frame() -> np.ndarray:
    return np.full((240, 320, 3), 110, dtype=np.uint8)


def object_frame() -> np.ndarray:
    frame = empty_frame()
    frame[60:180, 80:240] = (30, 30, 30)  # a large dark object; its grey level must differ from the background
    return frame


class SessionTests(unittest.TestCase):
    def make_session(self, frames, bin_reading=lambda: (True, "PLASTIC"), clock=time.monotonic, send=None):
        from src.sorter_gui import SorterSession
        from src.live_sorter import make_reference

        sent: list[Decision] = []

        def default_send(decision: Decision) -> str:
            sent.append(decision)
            return "sent"

        predict_calls: list[int] = []

        def predict(rgb: np.ndarray) -> LivePrediction:
            predict_calls.append(1)
            return prediction("metal", 0.95)

        session = SorterSession(
            source=FakeSource(frames),
            predict=predict,
            reference=make_reference(empty_frame()),
            state=SortState(threshold=0.7, stable_frames=2),
            send=send or default_send,
            bin_reading=bin_reading,
            threshold=0.7,
            clock=clock,
        )
        return session, sent, predict_calls

    def test_one_send_after_a_stable_object_and_none_while_it_stays(self) -> None:
        session, sent, _ = self.make_session([empty_frame(), object_frame(), object_frame(), object_frame(), empty_frame()])
        views = [session.step() for _ in range(5)]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].class_name, "metal")
        self.assertEqual(views[0].material.text, "Nothing on the platform")
        self.assertEqual(views[2].material.text, "METAL  95%")
        self.assertIn("Ready for next object: no", views[3].status)

    def test_bin_bar_follows_the_sensor_reading(self) -> None:
        session, _, _ = self.make_session([empty_frame()], bin_reading=lambda: (True, "PAPER"))
        view = session.step()
        self.assertEqual(view.bin.active, 2)

    def test_picture_is_rgb_and_matches_the_frame_size(self) -> None:
        session, _, _ = self.make_session([object_frame()])
        view = session.step()
        self.assertEqual(view.rgb.shape, (240, 320, 3))

    def test_frame_failure_keeps_the_last_picture_until_the_timeout(self) -> None:
        clock_values = iter([0.0, 3.5])
        session, _, _ = self.make_session([empty_frame(), None, None], clock=lambda: next(clock_values))
        first = session.step()
        second = session.step()
        self.assertIsNotNone(second.rgb)
        self.assertIn("no valid frame", second.status[0])
        self.assertIsNotNone(first.rgb)
        from src.data_collection import CameraError

        with self.assertRaisesRegex(CameraError, "stopped delivering valid frames"):
            session.step()

    def test_dry_run_bin_bar_says_not_connected(self) -> None:
        session, _, _ = self.make_session([empty_frame()], bin_reading=lambda: (False, None))
        self.assertIsNone(session.step().bin.active)

    def test_session_close_releases_the_camera(self) -> None:
        session, _, _ = self.make_session([])
        session.close()
        self.assertEqual(session._source.released, 1)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


class CommandLineTests(unittest.TestCase):
    def run_main(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = sg.main(list(argv))
        return code, out.getvalue()

    def test_negative_camera_is_refused(self) -> None:
        self.assertEqual(self.run_main("--dry-run", "--camera", "-1")[0], 2)

    def test_zero_stable_frames_is_refused(self) -> None:
        self.assertEqual(self.run_main("--dry-run", "--stable-frames", "0")[0], 2)

    def test_threshold_outside_zero_to_one_is_refused(self) -> None:
        code, out = self.run_main("--dry-run", "--threshold", "1.5")
        self.assertEqual(code, 2)
        self.assertIn("strictly between 0 and 1", out)

    def test_missing_serial_settings_are_refused_before_the_camera_opens(self) -> None:
        with mock.patch("src.sorter_gui.open_camera") as camera:
            code, out = self.run_main("--serial-config", "does_not_exist.txt")
        self.assertEqual(code, 2)
        self.assertIn("serial settings file not found", out)
        camera.assert_not_called()


# --------------------------------------------------------------------------
# Window (needs a display; skipped where tkinter cannot open one)
# --------------------------------------------------------------------------


@unittest.skipIf(TK_IMPORT_ERROR is not None, f"tkinter is not available: {TK_IMPORT_ERROR}")
class WindowTests(unittest.TestCase):
    def make_root(self):
        try:
            return tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display for the window test: {exc}")

    def test_window_draws_both_bars_and_the_status(self) -> None:
        from src.sorter_window import SorterWindow

        root = self.make_root()
        self.addCleanup(lambda: root.winfo_exists() and root.destroy())

        class StubSession:
            def step(self):
                return sg.FrameView(
                    rgb=np.zeros((240, 320, 3), dtype=np.uint8),
                    material=sg.material_view(True, prediction("plastic", 0.9), 0.7),
                    bin=sg.bin_view(True, "PLASTIC"),
                    status=("camera: ok", "Object on platform: yes"),
                )

        sender = sg.StmSender(CODES, None)
        window = SorterWindow(root, StubSession(), sender)
        window.refresh()
        material_texts = [window.material_bar.itemcget(item, "text") for item in window.material_bar.find_withtag("all") if window.material_bar.type(item) == "text"]
        self.assertEqual(material_texts, ["PLASTIC  90%"])
        self.assertEqual(len(window.bin_bar.find_withtag("all")), 8)  # four segments, four labels
        self.assertIn("PLASTIC", window.bin_title.cget("text"))
        self.assertIn("Object on platform: yes", window.status.cget("text"))

    def test_camera_failure_closes_the_window_with_the_error(self) -> None:
        from src.data_collection import CameraError
        from src.sorter_window import SorterWindow

        root = self.make_root()

        class BrokenSession:
            def step(self):
                raise CameraError("camera stopped delivering valid frames for 3 s")

        window = SorterWindow(root, BrokenSession(), sg.StmSender(CODES, None))
        window.tick()
        self.assertEqual(window.error, "camera stopped delivering valid frames for 3 s")


if __name__ == "__main__":
    unittest.main()
