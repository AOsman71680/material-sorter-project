"""Live sorter with a window: camera picture, detected-material bar, and colour-sensor bin bar.

It uses the same settings, protocol and decision rules as ``python -m src.live_sorter``.
The window shows:

* the camera picture;
* a bar for the material the camera model detected;
* a bar for the bin the STM32 colour sensor is over right now.

The bin bar needs the firmware that prints ``BIN:<name>`` lines
(firmware/stm32_sorter_bluepill/stm32_sorter_bluepill.ino). Without it, the bar
says it is waiting for the colour sensor.

This module holds the logic and has no window code, so it can be tested without a
display. The window is in ``src/sorter_window.py``.

Run from the project root:

    python -m src.sorter_gui --dry-run
    python -m src.sorter_gui

Close the window, or press q or Esc, to stop. Exit codes: 0 normal stop,
1 runtime error, 2 bad config or arguments.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from src.data_collection import CAMERA_FAILURE_TIMEOUT_S, CameraError, FrameSource, is_valid_frame, open_camera
from src.live_sorter import (
    BIN_PREFIX,
    DEFAULT_STABLE_FRAMES,
    FIRMWARE_NAMES,
    SERIAL_CONFIG_PATH,
    STOP_DEADLINE_S,
    Decision,
    SerialConfigError,
    SerialProtocolError,
    SerialSettings,
    SortOutcome,
    SorterError,
    SortState,
    check_stm32,
    class_codes,
    load_serial_settings,
    make_reference,
    object_present,
    open_serial,
    run_sort_command,
    stop_stm32,
)
from src.predict import PredictionError, load_model_for_inference
from src.synthesize_dataset import SynthesisError, find_templates, read_bgr
from src.webcam_predict import LivePrediction, frame_to_rgb, make_predictor

WINDOW_TITLE = "Material sorter"
BIN_ORDER = ("METAL", "PLASTIC", "PAPER", "ORGANIC")  # the firmware's names, left to right on the bin bar
NO_BIN = "NONE"  # what the firmware prints when nothing is in front of the sensor
READ_POLL_S = 0.2  # how long each serial read waits; the protocol code counts these waits
FRAME_WIDTH = 640  # the camera picture is scaled to this width
TICK_MS = 30  # time between window updates
LOG_LINES = 3  # how many recent sort messages the window keeps

# Bar colours per firmware name. ORGANIC is yellow.
MATERIAL_COLOURS: dict[str, str] = {
    "METAL": "#90a4ae",
    "PLASTIC": "#1e88e5",
    "PAPER": "#8e24aa",
    "ORGANIC": "#fdd835",
}
NEUTRAL_COLOUR = "#9e9e9e"


# --------------------------------------------------------------------------
# What the bars show (pure functions, tested without a window)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialView:
    text: str
    fraction: float  # how full the bar is, 0 to 1
    material: str | None  # firmware name when the prediction is confident, else None


@dataclass(frozen=True)
class BinView:
    text: str
    active: int | None  # index into BIN_ORDER of the bin under the sensor, else None


def material_view(present: bool, prediction: LivePrediction | None, threshold: float) -> MaterialView:
    """What the material bar says: nothing, unsure, or the detected material with its probability."""
    if not present:
        return MaterialView("Nothing on the platform", 0.0, None)
    if prediction is None:
        return MaterialView("Looking at the object...", 0.0, None)
    name = FIRMWARE_NAMES[prediction.predicted_class]
    probability = prediction.max_probability
    if probability < threshold:
        return MaterialView(f"Unsure: {name} {probability:.0%} (below {threshold:.0%})", probability, None)
    return MaterialView(f"{name}  {probability:.0%}", probability, name)


def bin_view(connected: bool, bin_name: str | None) -> BinView:
    """What the bin bar says, from the last ``BIN:`` line the STM32 sent."""
    if not connected:
        return BinView("not connected (dry run or serial lost)", None)
    if bin_name is None:
        return BinView("waiting for the colour sensor...", None)
    if bin_name == NO_BIN:
        return BinView("nothing in front of the sensor", None)
    if bin_name in BIN_ORDER:
        return BinView(f"{bin_name}", BIN_ORDER.index(bin_name))
    return BinView(f"reads {bin_name}", None)


# --------------------------------------------------------------------------
# Serial: one reader thread splits bin reports from replies
# --------------------------------------------------------------------------


class SerialLink:
    """Owns the open serial port after start-up.

    A reader thread takes every line from the STM32. ``BIN:`` lines update the bin
    reading. Every other line goes to a queue, where the sort protocol reads it.
    """

    def __init__(self, port: Any) -> None:
        self._port = port
        self._inbox: queue.Queue[str] = queue.Queue()
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._bin: str | None = None
        self._lost = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, name="stm32-reader", daemon=True)
        self.timeout = READ_POLL_S  # read by the protocol code to count waits

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def lost(self) -> bool:
        """True after the reader thread stopped because the port failed."""
        with self._state_lock:
            return self._lost

    def current_bin(self) -> str | None:
        with self._state_lock:
            return self._bin

    def write(self, data: bytes) -> None:
        with self._write_lock:
            self._port.write(data)
            self._port.flush()

    def next_line(self, timeout: float) -> str | None:
        try:
            return self._inbox.get(timeout=timeout)
        except queue.Empty:
            return None

    def drop_pending(self) -> None:
        while True:
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                return

    def _read_loop(self) -> None:
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = self._port.readline()
            except Exception:  # the port went away; sorting will report the silence
                with self._state_lock:
                    self._lost = True
                return
            if not chunk:
                continue
            buffer += chunk
            if not buffer.endswith(b"\n"):
                continue
            text = buffer.decode("ascii", errors="replace").strip()
            buffer = b""
            if text.startswith(BIN_PREFIX):
                with self._state_lock:
                    self._bin = text[len(BIN_PREFIX) :].strip()
            else:
                self._inbox.put(text)


class SortPort:
    """The port interface the protocol functions in live_sorter use, fed from a ``SerialLink``."""

    def __init__(self, link: SerialLink) -> None:
        self._link = link
        self.timeout = link.timeout

    def write(self, data: bytes) -> None:
        self._link.write(data)

    def flush(self) -> None:
        pass  # SerialLink.write flushes already

    def reset_input_buffer(self) -> None:
        self._link.drop_pending()

    def readline(self) -> bytes:
        line = self._link.next_line(self.timeout)
        if line is None:
            return b""
        return (line + "\n").encode("ascii", errors="replace")


class SortWorker:
    """Runs one sort at a time on a background thread, so the camera keeps updating."""

    def __init__(self, port: SortPort, sort_fn: Callable[[Any, str, str], SortOutcome] = run_sort_command) -> None:
        self._port = port
        self._sort_fn = sort_fn
        self._busy = False
        self._thread: threading.Thread | None = None
        self.results: queue.Queue[tuple[str, object]] = queue.Queue()

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, code: str, firmware_name: str) -> bool:
        """Start a sort. Returns False, and sends nothing, when a sort is already running."""
        if self._busy:
            return False
        self._busy = True
        self._thread = threading.Thread(target=self._run, args=(code, firmware_name), name="stm32-sort", daemon=True)
        self._thread.start()
        return True

    def wait(self, timeout: float) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self, code: str, firmware_name: str) -> None:
        try:
            outcome = self._sort_fn(self._port, code, firmware_name)
            result: tuple[str, object] = ("done", outcome)
        except SerialProtocolError as exc:
            result = ("error", str(exc))
        except Exception as exc:  # keep the window alive; report it as a stop
            result = ("error", f"unexpected problem during sorting: {exc}")
        self._busy = False  # cleared before the result is posted, so the next start() succeeds
        self.results.put(result)


class StmSender:
    """Turns a decision into a command for the STM32, and reports the sort results.

    After a protocol error the sender halts: nothing more is sent until the program restarts.
    """

    def __init__(self, codes: dict[str, bytes], worker: SortWorker | None) -> None:
        self._codes = codes
        self._worker = worker
        self.halted_reason: str | None = None
        self.log: deque[str] = deque(maxlen=LOG_LINES)

    def __call__(self, decision: Decision) -> str:
        code = self._codes[decision.class_name].decode()
        name = FIRMWARE_NAMES[decision.class_name]
        if self._worker is None:
            return f"dry run: would send {code} ({name})"
        if self.halted_reason is not None:
            return f"not sent: {self.halted_reason}"
        if not self._worker.start(code, name):
            return "not sent: the previous sort is still finishing"
        return f"sent {code} ({name}), waiting for the STM32"

    def poll(self) -> None:
        """Move finished sort results into the log, and stop sending after an error."""
        if self._worker is None:
            return
        while True:
            try:
                kind, value = self._worker.results.get_nowait()
            except queue.Empty:
                return
            if kind == "done":
                outcome = value
                assert isinstance(outcome, SortOutcome)
                note = "" if outcome.completed else " (not completed)"
                self.log.append(f"STM32: {outcome.final_line}{note}")
            else:
                self.halted_reason = str(value)
                self.log.append(f"STOPPED: {value}")


# --------------------------------------------------------------------------
# One frame at a time, without a window
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameView:
    rgb: np.ndarray | None  # latest camera picture, RGB
    material: MaterialView
    bin: BinView
    status: tuple[str, ...]  # short status lines


class SorterSession:
    """Per-frame work: read the camera, predict, decide, and report the bin. No window code."""

    def __init__(
        self,
        *,
        source: FrameSource,
        predict: Callable[[np.ndarray], LivePrediction],
        reference: np.ndarray,
        state: SortState,
        send: Callable[[Decision], str],
        bin_reading: Callable[[], tuple[bool, str | None]],
        threshold: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._predict = predict
        self._reference = reference
        self._state = state
        self._send = send
        self._bin_reading = bin_reading
        self._threshold = threshold
        self._clock = clock
        self._failing_since: float | None = None
        self._rgb: np.ndarray | None = None
        self._last_sent: str | None = None

    def step(self) -> FrameView:
        """Read one camera frame and update everything. Raises CameraError if the camera stops."""
        try:
            ok, raw = self._source.read()
        except cv2.error:
            ok, raw = False, None

        present = False
        prediction: LivePrediction | None = None
        if ok and is_valid_frame(raw):
            self._failing_since = None
            rgb = frame_to_rgb(raw)
            self._rgb = rgb
            present = object_present(raw, self._reference)
            if present:
                prediction = self._predict(rgb)
            decision = self._state.update(present, prediction)
            if decision is not None:
                self._last_sent = self._send(decision)
            camera_note = "camera: ok"
        else:
            now = self._clock()
            if self._failing_since is None:
                self._failing_since = now
            elif now - self._failing_since >= CAMERA_FAILURE_TIMEOUT_S:
                raise CameraError(f"camera stopped delivering valid frames for {CAMERA_FAILURE_TIMEOUT_S:g} s")
            camera_note = "camera: no valid frame (showing the last one)"

        connected, bin_name = self._bin_reading()
        status = (
            camera_note,
            f"Object on platform: {'yes' if present else 'no'}",
            f"Ready for next object: {'yes' if self._state.armed else 'no'}",
            f"Last sent: {self._last_sent or 'nothing yet'}",
        )
        return FrameView(
            rgb=self._rgb,
            material=material_view(present, prediction, self._threshold),
            bin=bin_view(connected, bin_name),
            status=status,
        )

    def close(self) -> None:
        self._source.release()


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live sorter with a window: camera, detected material, and the colour-sensor bin.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint to use (default: best_model.pth in paths.model_dir)")
    parser.add_argument("--camera", type=int, default=None, help="camera index (default: camera.index from the config)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="override training.device")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="send only when the top probability is at least this value (default: inference.confidence_threshold from the config)",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=DEFAULT_STABLE_FRAMES,
        help="frames in a row the prediction must agree (default: %(default)s)",
    )
    parser.add_argument("--serial-config", type=Path, default=SERIAL_CONFIG_PATH, help="serial settings file (default: config/serial.txt)")
    parser.add_argument("--dry-run", action="store_true", help="show the window, but do not open the serial port or send anything")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    return parser


def shut_down(port: Any, link: SerialLink | None, worker: SortWorker | None) -> None:
    """Leave the STM32 idle and close the port. Safe to call when nothing is open."""
    if port is None:
        return
    if link is None:
        port.close()
        return
    try:
        if worker is not None and worker.busy:
            link.write(b"STOP\n")  # ends a running sort; the worker then reports it
            worker.wait(STOP_DEADLINE_S + 2.0)
        stop_stm32(SortPort(link))
    finally:
        link.stop()
        port.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.camera is not None and args.camera < 0:
        print(f"error: --camera must be 0 or more, got {args.camera}")
        return 2
    if args.stable_frames < 1:
        print(f"error: --stable-frames must be at least 1, got {args.stable_frames}")
        return 2
    try:
        config: Config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    threshold = config.confidence_threshold if args.threshold is None else args.threshold
    if not 0.0 < threshold < 1.0:
        print(f"error: --threshold must be strictly between 0 and 1, got {threshold}")
        return 2
    try:
        codes = class_codes(config.cups, config.classes)
    except SorterError as exc:
        print(f"error: {exc}")
        return 2

    serial_settings: SerialSettings | None = None
    if not args.dry_run:
        try:
            serial_settings = load_serial_settings(args.serial_config)
        except SerialConfigError as exc:
            print(f"error: {exc}")
            return 2

    camera_index = config.camera_index if args.camera is None else args.camera
    checkpoint = args.checkpoint or (config.paths.model_dir / "best_model.pth")
    template_dir = config.paths.dataset_dir.parent / "template"

    port: Any = None
    link: SerialLink | None = None
    worker: SortWorker | None = None
    source: FrameSource | None = None
    try:
        templates = find_templates(template_dir, config.classes)
        reference = make_reference(read_bgr(templates.background))
        loaded = load_model_for_inference(config, checkpoint, device_setting=args.device)
        predict = make_predictor(loaded, config.classes, threshold=None)
        state = SortState(threshold=threshold, stable_frames=args.stable_frames)

        if args.dry_run:
            sender = StmSender(codes, None)

            def read_bin() -> tuple[bool, str | None]:
                return False, None

        else:
            assert serial_settings is not None
            port = open_serial(serial_settings)
            print(f"STM32 connected: {check_stm32(port)}")
            link = SerialLink(port)
            link.start()
            worker = SortWorker(SortPort(link))
            sender = StmSender(codes, worker)
            current_link = link

            def read_bin() -> tuple[bool, str | None]:
                return (not current_link.lost), current_link.current_bin()

        print(f"Model: {loaded.metadata['architecture']}   Device: {loaded.device.type}   Threshold: {threshold:g}")
        print(f"Opening camera {camera_index} ...")
        source = open_camera(camera_index, index_flag="--camera")
        session = SorterSession(
            source=source,
            predict=predict,
            reference=reference,
            state=state,
            send=sender,
            bin_reading=read_bin,
            threshold=threshold,
        )

        from src.sorter_window import run_window  # imported here, so the other parts run without tkinter

        error = run_window(session, sender)
        if error is not None:
            print(f"error: {error}")
            return 1
    except (SorterError, SynthesisError, PredictionError, CameraError) as exc:
        print(f"error: {exc}")
        return 1
    except ImportError as exc:
        print(f"error: the window needs tkinter, which comes with the standard Python installer ({exc})")
        return 1
    except cv2.error as exc:
        print(f"error: OpenCV error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted.")
        return 0
    finally:
        if source is not None:
            source.release()
        shut_down(port, link, worker)  # leave the STM32 idle whenever this program exits

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
