"""Live sorting: read the USB camera, classify the object, and send one byte to the STM32.

The script watches the platform. When an object appears, it predicts the
material over several frames. Once the prediction is stable and confident, it
sends one ASCII character to the STM32 over serial and waits for the STM32 to
reply with the material name and a newline. The next message is sent only after
the platform is empty again.

Codes (from ``cups`` in config/config.yaml):

    1 = metal    (reply "metal")
    2 = plastic  (reply "plastic")
    3 = paper    (reply "paper")
    4 = organic  (reply "organic"; the project class is "other")

Settings: ``config/serial.txt`` (port and baud rate). Use ``--dry-run`` to see
the codes without opening the serial port.

Run from the project root:

    python -m src.live_sorter --dry-run
    python -m src.live_sorter

Press q or Esc in the window to stop. If the STM32 does not reply with the
expected name, the script stops at once and does not send again.

Exit codes: 0 normal stop, 1 runtime error (model, camera, serial, or template),
2 bad config or arguments.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, Config, ConfigError, load_config
from src.data_collection import (
    CAMERA_FAILURE_TIMEOUT_S,
    KEY_QUIT,
    CameraError,
    Display,
    FrameSource,
    InvalidFrameError,
    OpenCvWindow,
    _GREEN,
    _WHITE,
    _put_text,
    is_valid_frame,
    open_camera,
)
from src.predict import PredictionError, load_model_for_inference
from src.synthesize_dataset import SynthesisError, find_templates, read_bgr
from src.webcam_predict import WINDOW_TITLE, LivePrediction, draw_prediction, frame_to_rgb, make_predictor

SERIAL_CONFIG_PATH = PROJECT_ROOT / "config" / "serial.txt"
SERIAL_KEYS = ("port", "baud", "ack_timeout")
DEFAULT_ACK_TIMEOUT_S = 2.0
DEFAULT_STABLE_FRAMES = 5

# The name the STM32 firmware uses for each class. "other" is the Organic material.
FIRMWARE_NAMES: dict[str, str] = {"metal": "METAL", "plastic": "PLASTIC", "paper": "PAPER", "other": "ORGANIC"}

# Presence check: compare the frame with the empty platform photo at a small size.
COMPARE_SIZE = (160, 120)
PRESENCE_PIXEL_DIFF = 30  # grey-level change that counts as "something is different here"
PRESENCE_FRACTION = 0.02  # object present when at least this share of pixels changed

class SorterError(RuntimeError):
    """A runtime problem: the model, camera, serial port, or template folder."""


class SerialConfigError(ValueError):
    """config/serial.txt is missing or invalid."""


class SerialProtocolError(SorterError):
    """The STM32 did not behave as the firmware describes. Sorting must stop."""


@dataclass(frozen=True)
class SerialSettings:
    port: str
    baud: int
    ack_timeout: float


@dataclass(frozen=True)
class Decision:
    class_name: str
    confidence: float


@dataclass
class SorterStats:
    frames: int = 0
    predictions: int = 0
    sent: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Settings and protocol
# --------------------------------------------------------------------------


def load_serial_settings(path: Path = SERIAL_CONFIG_PATH) -> SerialSettings:
    """Read ``port = ...``, ``baud = ...`` and optional ``ack_timeout = ...`` lines.

    Blank lines and lines starting with ``#`` are ignored.
    """
    if not path.is_file():
        raise SerialConfigError(f"serial settings file not found: {path}")
    values: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SerialConfigError(f"{path.name} line {line_no}: expected 'key = value', got {line!r}")
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in SERIAL_KEYS:
            raise SerialConfigError(f"{path.name} line {line_no}: unknown key {key!r}; expected one of {SERIAL_KEYS}")
        if key in values:
            raise SerialConfigError(f"{path.name} line {line_no}: {key!r} is given more than once")
        values[key] = value

    port = values.get("port", "")
    if not port:
        raise SerialConfigError(f"set the serial port in {path.name} (for example: port = COM3)")
    baud_text = values.get("baud", "")
    if not baud_text:
        raise SerialConfigError(f"set the baud rate in {path.name} (for example: baud = 115200)")
    try:
        baud = int(baud_text)
    except ValueError as exc:
        raise SerialConfigError(f"{path.name}: baud must be a whole number, got {baud_text!r}") from exc
    if baud <= 0:
        raise SerialConfigError(f"{path.name}: baud must be positive, got {baud}")
    try:
        ack_timeout = float(values.get("ack_timeout", DEFAULT_ACK_TIMEOUT_S))
    except ValueError as exc:
        raise SerialConfigError(f"{path.name}: ack_timeout must be a number of seconds") from exc
    if not 0.1 <= ack_timeout <= 30.0:
        raise SerialConfigError(f"{path.name}: ack_timeout must be between 0.1 and 30 seconds, got {ack_timeout}")
    return SerialSettings(port=port, baud=baud, ack_timeout=ack_timeout)


def class_codes(cups: dict[str, int], classes: tuple[str, ...]) -> dict[str, bytes]:
    """The one-byte code for each class, taken from ``cups`` (must be 1 to 4)."""
    codes: dict[str, bytes] = {}
    for name in classes:
        value = cups[name]
        if value not in (1, 2, 3, 4):
            raise SorterError(
                f"cups.{name} is {value}, but the STM32 protocol uses 1 to 4 only. Fix cups in config/config.yaml."
            )
        if name not in FIRMWARE_NAMES:
            raise SorterError(f"class {name!r} has no STM32 name")
        codes[name] = str(value).encode("ascii")
    return codes


SORT_DEADLINE_S = 60.0  # firmware rotation timeout is 20 s and the gate cycle about 5 s; 60 s leaves margin
STATUS_DEADLINE_S = 5.0
STOP_DEADLINE_S = 3.0


@dataclass(frozen=True)
class SortOutcome:
    completed: bool  # True only after the firmware's SORTING COMPLETE
    final_line: str  # the last message from the STM32
    lines: tuple[str, ...]


def _poll_budget(port: object, seconds: float) -> int:
    """How many empty reads fit in ``seconds``, based on the port's read timeout."""
    timeout = getattr(port, "timeout", None) or 1.0
    return max(1, math.ceil(seconds / max(float(timeout), 0.01)))


BIN_PREFIX = "BIN:"  # the firmware reports the colour-sensor bin as "BIN:<name>"; it is not a reply


def _read_message(port: object, budget: int) -> str | None:
    """Read one newline-terminated reply. Returns None after ``budget`` empty reads.

    ``BIN:`` lines are skipped: they report the colour sensor and can arrive at any time.
    """
    buffer = b""
    empty = 0
    while empty < budget:
        chunk = port.readline()  # type: ignore[attr-defined]
        if not chunk:
            empty += 1
            continue
        buffer += chunk
        if buffer.endswith(b"\n"):
            text = buffer.decode("ascii", errors="replace").strip()
            buffer = b""
            if text.startswith(BIN_PREFIX):
                continue
            return text
    return None


def run_sort_command(port: object, code: str, firmware_name: str) -> SortOutcome:
    """Send one command and follow the firmware's messages until the sort ends.

    The command is the digit followed by a newline, as the firmware requires. The
    first reply must be ``REQUESTED:<NAME>``. ``POSITION TIMEOUT`` and ``STOPPED``
    end the sort without completing it; they are reported, not raised. Any other
    unexpected message raises ``SerialProtocolError``, and the caller must stop the STM32.
    """
    reset = getattr(port, "reset_input_buffer", None)
    if reset is not None:
        reset()
    port.write(f"{code}\n".encode("ascii"))  # type: ignore[attr-defined]
    port.flush()  # type: ignore[attr-defined]
    budget = _poll_budget(port, SORT_DEADLINE_S)

    first = _read_message(port, budget)
    if first is None:
        raise SerialProtocolError(f"no reply from the STM32 to command {code!r}. Sorting has stopped.")
    if first == "BUSY":
        raise SerialProtocolError("the STM32 reports BUSY, so it is still doing an earlier sort. Sorting has stopped.")
    if first == "INVALID COMMAND":
        raise SerialProtocolError(f"the STM32 rejected command {code!r} as invalid. Sorting has stopped.")
    expected = f"REQUESTED:{firmware_name}"
    if first != expected:
        raise SerialProtocolError(f"the STM32 replied {first!r} but {expected!r} was expected. Sorting has stopped.")

    lines = [first]
    while True:
        line = _read_message(port, budget)
        if line is None:
            raise SerialProtocolError(
                f"the STM32 did not finish the sort in time (last message: {lines[-1]!r}). Sorting has stopped."
            )
        lines.append(line)
        if line == "SORTING COMPLETE":
            return SortOutcome(True, line, tuple(lines))
        if line in ("POSITION TIMEOUT", "STOPPED"):
            return SortOutcome(False, line, tuple(lines))


def check_stm32(port: object) -> str:
    """Ask the STM32 for its status. Confirms the port is connected and the firmware answers."""
    port.write(b"STATUS\n")  # type: ignore[attr-defined]
    port.flush()  # type: ignore[attr-defined]
    budget = _poll_budget(port, STATUS_DEADLINE_S)
    while True:
        line = _read_message(port, budget)
        if line is None:
            raise SerialProtocolError(
                "the STM32 did not answer STATUS. Check the COM port and baud rate, that the board is "
                "powered, and that the sorting firmware is running."
            )
        if line.startswith("LAST_SAVED:"):
            return line


def stop_stm32(port: object) -> bool:
    """Send STOP and wait briefly for STOPPED. Never raises: it runs while the program is closing."""
    try:
        port.write(b"STOP\n")  # type: ignore[attr-defined]
        port.flush()  # type: ignore[attr-defined]
    except Exception:  # the port may already be gone; there is nothing more to do
        return False
    budget = _poll_budget(port, STOP_DEADLINE_S)
    while True:
        line = _read_message(port, budget)
        if line is None:
            return False
        if line == "STOPPED":
            return True


def open_serial(settings: SerialSettings) -> object:
    """Open the serial port. Imports pyserial only when the port is actually used."""
    try:
        import serial  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SorterError("pyserial is not installed. Install it with: pip install pyserial") from exc
    try:
        return serial.Serial(settings.port, baudrate=settings.baud, timeout=settings.ack_timeout)
    except (serial.SerialException, OSError) as exc:
        raise SorterError(f"could not open serial port {settings.port}: {exc}") from exc


# --------------------------------------------------------------------------
# Presence and decisions
# --------------------------------------------------------------------------


def _compare_image(bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(bgr, COMPARE_SIZE, interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(grey, (5, 5), 0)


def make_reference(empty_bgr: np.ndarray) -> np.ndarray:
    """The empty platform, prepared the same way as each camera frame."""
    return _compare_image(empty_bgr)


def object_present(frame_bgr: np.ndarray, reference: np.ndarray) -> bool:
    """True when enough of the frame differs from the empty platform."""
    diff = cv2.absdiff(_compare_image(frame_bgr), reference)
    return float((diff > PRESENCE_PIXEL_DIFF).mean()) > PRESENCE_FRACTION


class SortState:
    """Decides when to send. One send per object, and only after the object has left the platform.

    * No object: the state re-arms (the next object may be sent).
    * Object present and armed: the prediction must be confident and the same class
      for ``stable_frames`` frames in a row, then one ``Decision`` is returned.
    * After a decision, nothing else is sent until the platform is empty again.
    """

    def __init__(self, threshold: float, stable_frames: int) -> None:
        self.threshold = threshold
        self.stable_frames = stable_frames
        self.armed = True
        self._candidate: str | None = None
        self._streak = 0

    def update(self, present: bool, prediction: LivePrediction | None) -> Decision | None:
        if not present:
            self.armed = True
            self._candidate, self._streak = None, 0
            return None
        if not self.armed:
            return None
        if prediction is None or prediction.max_probability < self.threshold:
            self._candidate, self._streak = None, 0
            return None
        if prediction.predicted_class == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = prediction.predicted_class, 1
        if self._streak >= self.stable_frames:
            decision = Decision(prediction.predicted_class, prediction.max_probability)
            self.armed = False
            self._candidate, self._streak = None, 0
            return decision
        return None


# --------------------------------------------------------------------------
# Live loop
# --------------------------------------------------------------------------


def _status_lines(present: bool, armed: bool, last_sent: str | None) -> list[str]:
    return [
        f"Object on platform: {'yes' if present else 'no'}",
        f"Ready for next object: {'yes' if armed else 'no'}",
        f"Last sent: {last_sent or 'nothing yet'}",
    ]


def run_sorting_loop(
    predict: Callable[[np.ndarray], LivePrediction],
    source: FrameSource,
    display: Display,
    reference: np.ndarray,
    state: SortState,
    send: Callable[[Decision], str],
    *,
    clock: Callable[[], float] | None = None,
) -> SorterStats:
    """Show the live feed and send a code when ``state`` decides. Always releases the camera and window.

    ``send`` is called once per decision. If it raises, the loop stops at once.
    """
    now_seconds = clock or time.monotonic
    stats = SorterStats()
    latest: LivePrediction | None = None
    last_sent: str | None = None
    failing_since: float | None = None
    try:
        while True:
            try:
                ok, raw = source.read()
            except cv2.error:
                ok, raw = False, None

            if ok and is_valid_frame(raw):
                failing_since = None
                stats.frames += 1
                present = object_present(raw, reference)
                prediction: LivePrediction | None = None
                if present:
                    prediction = predict(frame_to_rgb(raw))
                    stats.predictions += 1
                latest = prediction if present else None
                decision = state.update(present, prediction)
                if decision is not None:
                    reply = send(decision)
                    last_sent = f"{decision.class_name}: {reply}"
                    stats.sent.append(decision.class_name)
                canvas = draw_prediction(raw, latest)
                for index, line in enumerate(_status_lines(present, state.armed, last_sent)):
                    _put_text(canvas, line, (10, canvas.shape[0] - 110 + 22 * index), _GREEN if index == 2 else _WHITE)
                display.show(canvas)
                key = display.wait_key(1)
            else:
                now = now_seconds()
                if failing_since is None:
                    failing_since = now
                elif now - failing_since >= CAMERA_FAILURE_TIMEOUT_S:
                    raise CameraError(f"camera stopped delivering valid frames for {CAMERA_FAILURE_TIMEOUT_S:g} s")
                key = display.wait_key(30)

            if not display.is_open() or key in KEY_QUIT:
                break
    finally:
        source.release()
        display.close()
    return stats


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify objects on the platform and send the material code to the STM32.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint to use (default: best_model.pth in paths.model_dir)")
    parser.add_argument("--camera", type=int, default=None, help="camera index (default: camera.index from the config)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="override training.device")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="send only when the top probability is at least this value (default: inference.confidence_threshold from the config)",
    )
    parser.add_argument("--stable-frames", type=int, default=DEFAULT_STABLE_FRAMES, help="frames in a row the prediction must agree (default: %(default)s)")
    parser.add_argument("--serial-config", type=Path, default=SERIAL_CONFIG_PATH, help="serial settings file (default: config/serial.txt)")
    parser.add_argument("--dry-run", action="store_true", help="print what would be sent, and do not open the serial port")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    args = parser.parse_args(argv)

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

    port = None
    stats: SorterStats | None = None
    try:
        templates = find_templates(template_dir, config.classes)
        reference = make_reference(read_bgr(templates.background))
        loaded = load_model_for_inference(config, checkpoint, device_setting=args.device)
        predict = make_predictor(loaded, config.classes, threshold=None)
        state = SortState(threshold=threshold, stable_frames=args.stable_frames)

        if args.dry_run:
            print("[dry run] serial port is not opened; codes are printed only.")

            def send(decision: Decision) -> str:
                code = codes[decision.class_name].decode()
                print(f"[dry run] would send {code} ({decision.class_name}, {decision.confidence:.1%})")
                return "dry run"

        else:
            assert serial_settings is not None
            port = open_serial(serial_settings)
            print(f"STM32 connected: {check_stm32(port)}")

            def send(decision: Decision) -> str:
                code = codes[decision.class_name].decode()
                # A protocol error propagates; the finally block below sends STOP and closes the port.
                outcome = run_sort_command(port, code, FIRMWARE_NAMES[decision.class_name])
                print(f"sent {code} for {decision.class_name} ({decision.confidence:.1%}); STM32: {outcome.final_line}")
                return outcome.final_line

        print(f"Model: {loaded.metadata['architecture']}   Device: {loaded.device.type}   Threshold: {threshold:g}")
        print(f"Opening camera {camera_index} ... press q or Esc in the window to stop.")
        source = open_camera(camera_index, index_flag="--camera")
        try:
            display = OpenCvWindow(WINDOW_TITLE)
        except BaseException:
            source.release()
            raise
        stats = run_sorting_loop(predict, source, display, reference, state, send)
    except (SorterError, SynthesisError, PredictionError, CameraError, InvalidFrameError) as exc:
        print(f"error: {exc}")
        return 1
    except cv2.error as exc:
        print(f"error: OpenCV error (if this mentions imshow/namedWindow, install the GUI build): {exc}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted.")
        return 0
    finally:
        if port is not None:
            stop_stm32(port)  # leave the STM32 idle whenever this program exits
            port.close()  # type: ignore[attr-defined]

    print(f"Stopped after {stats.frames} frames, {stats.predictions} predictions, {len(stats.sent)} sends.")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    raise SystemExit(main())
