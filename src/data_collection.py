"""Collect labeled webcam images into ``<dataset_dir>/<class>/``.

Run with ``python -m src.data_collection``. Nothing is saved until a class has
been selected, and saved images are the raw camera frame (no overlay text).
Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from src.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from src.dataset import count_images

SAVE_EXTENSION = ".jpg"
JPEG_QUALITY = 95
MAX_NAME_ATTEMPTS = 1000

CAMERA_FAILURE_TIMEOUT_S = 3.0  # continuous bad frames before giving up
CAMERA_WARMUP_READS = 30
MESSAGE_FRAMES = 45  # how many frames a status message stays on screen

KEY_QUIT = (ord("q"), 27)  # q or Esc
KEY_CAPTURE = (32, ord("c"))  # Space or c
FIRST_CLASS_KEY = ord("1")
MAX_SELECTABLE_CLASSES = 9

WINDOW_TITLE = "Material sorter - data collection"
PLACEHOLDER_SIZE = (480, 640)  # height, width

_GREEN = (0, 220, 0)
_WHITE = (255, 255, 255)
_YELLOW = (0, 220, 255)
_RED = (60, 60, 255)


class CameraError(RuntimeError):
    """The camera could not be opened or stopped delivering frames."""


class InvalidFrameError(ValueError):
    """A frame was missing, empty, or not a 3-channel 8-bit image."""


class NoClassSelectedError(RuntimeError):
    """Capture was requested before a class was selected."""


class FrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


class Display(Protocol):
    def show(self, image: np.ndarray) -> None: ...

    def wait_key(self, delay_ms: int) -> int:
        """Return the pressed key code (0-255) or -1 if none."""
        ...

    def is_open(self) -> bool: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------
# Frames and saving
# --------------------------------------------------------------------------


def is_valid_frame(frame: object) -> bool:
    """True for a non-empty 8-bit, 3-channel (BGR) image array."""
    return (
        isinstance(frame, np.ndarray)
        and frame.dtype == np.uint8
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.shape[0] > 0
        and frame.shape[1] > 0
    )


def save_frame(
    frame: object, class_dir: Path, class_name: str, *, now: datetime | None = None
) -> Path:
    """Save ``frame`` into ``class_dir`` under a unique name and return the path.

    The file is created exclusively, so an existing image can never be
    overwritten; on a name clash a numeric suffix is added.
    """
    if not is_valid_frame(frame):
        raise InvalidFrameError("frame is missing, empty, or not a 3-channel 8-bit image")
    assert isinstance(frame, np.ndarray)
    ok, encoded = cv2.imencode(SAVE_EXTENSION, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise InvalidFrameError("frame could not be encoded as an image")
    data = encoded.tobytes()

    class_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    for attempt in range(MAX_NAME_ATTEMPTS):
        suffix = "" if attempt == 0 else f"_{attempt}"
        path = class_dir / f"{class_name}_{stamp}{suffix}{SAVE_EXTENSION}"
        try:
            handle = path.open("xb")  # fails if the file already exists
        except FileExistsError:
            continue
        try:
            with handle:
                handle.write(data)
        except OSError:
            path.unlink(missing_ok=True)  # remove only the partial file created just now
            raise
        return path
    raise OSError(f"could not find an unused file name in {class_dir}")


class CollectionSession:
    """Tracks the selected class and per-class image counts."""

    def __init__(self, classes: tuple[str, ...], dataset_dir: Path) -> None:
        if not 0 < len(classes) <= MAX_SELECTABLE_CLASSES:
            raise ValueError(f"need between 1 and {MAX_SELECTABLE_CLASSES} classes")
        self.classes = classes
        self.dataset_dir = dataset_dir
        self.selected: str | None = None
        for name in classes:
            self.class_dir(name).mkdir(parents=True, exist_ok=True)
        self.counts: dict[str, int] = {n: count_images(self.class_dir(n)) for n in classes}

    def class_dir(self, name: str) -> Path:
        return self.dataset_dir / name

    def select(self, index: int) -> bool:
        """Select a class by 0-based index. Returns False if out of range."""
        if not 0 <= index < len(self.classes):
            return False
        self.selected = self.classes[index]
        return True

    def capture(self, frame: object) -> Path:
        if self.selected is None:
            raise NoClassSelectedError("select a class first")
        path = save_frame(frame, self.class_dir(self.selected), self.selected)
        self.counts[self.selected] += 1
        return path


# --------------------------------------------------------------------------
# Camera and window
# --------------------------------------------------------------------------


def open_camera(index: int, *, index_flag: str = "--camera-index") -> cv2.VideoCapture:
    """Open the webcam and confirm it delivers a valid frame.

    ``index_flag`` is only used in the error message, so each tool names its own option.
    """
    backends = [("DirectShow", cv2.CAP_DSHOW), ("default", cv2.CAP_ANY)]
    if sys.platform != "win32":
        backends = backends[1:]
    for _, backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            continue
        for _ in range(CAMERA_WARMUP_READS):
            ok, frame = capture.read()
            if ok and is_valid_frame(frame):
                return capture
            time.sleep(0.05)
        capture.release()
    raise CameraError(
        f"could not get frames from camera index {index}. Check that it is plugged in, "
        "not used by another app, allowed in Windows Settings > Privacy > Camera, "
        f"or try another index with {index_flag}."
    )


class OpenCvWindow:
    """``Display`` backed by an OpenCV GUI window."""

    def __init__(self, title: str = WINDOW_TITLE) -> None:
        self.title = title
        self._shown = False
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def show(self, image: np.ndarray) -> None:
        cv2.imshow(self.title, image)
        self._shown = True

    def wait_key(self, delay_ms: int) -> int:
        key = cv2.waitKey(delay_ms)
        return -1 if key == -1 else key & 0xFF

    def is_open(self) -> bool:
        if not self._shown:
            return True
        return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1

    def close(self) -> None:
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# Overlay and main loop
# --------------------------------------------------------------------------


def _put_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)  # outline
    cv2.putText(image, text, origin, font, 0.55, color, 1, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray | None, session: CollectionSession, message: str | None
) -> np.ndarray:
    """Return a copy of ``frame`` (or a placeholder) with status text drawn on it."""
    if frame is None:
        canvas = np.zeros((*PLACEHOLDER_SIZE, 3), dtype=np.uint8)
        _put_text(canvas, "No valid camera frame...", (20, PLACEHOLDER_SIZE[0] // 2), _RED)
    else:
        canvas = frame.copy()
    height = canvas.shape[0]

    selected = session.selected
    header = f"Class: {selected}" if selected else "Class: none - press 1-%d to select" % len(session.classes)
    _put_text(canvas, header, (10, 22), _GREEN if selected else _YELLOW)
    for i, name in enumerate(session.classes):
        mark = ">" if name == selected else " "
        color = _GREEN if name == selected else _WHITE
        _put_text(canvas, f"{mark}[{i + 1}] {name}: {session.counts[name]}", (10, 46 + 22 * i), color)

    if message:
        _put_text(canvas, message, (10, height - 40), _YELLOW)
    _put_text(canvas, "SPACE/C capture   1-%d class   Q/Esc quit" % len(session.classes), (10, height - 14), _WHITE)
    return canvas


def class_index_for_key(key: int, class_count: int) -> int | None:
    index = key - FIRST_CLASS_KEY
    return index if 0 <= index < class_count else None


def run_collection(
    session: CollectionSession,
    source: FrameSource,
    display: Display,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Preview loop. Always releases the camera and closes the window on exit.

    Raises ``CameraError`` if no valid frame arrives for ``CAMERA_FAILURE_TIMEOUT_S``.
    """
    frame: np.ndarray | None = None
    failing_since: float | None = None
    message: str | None = None
    message_frames_left = 0

    try:
        while True:
            try:
                ok, raw = source.read()
            except cv2.error:
                ok, raw = False, None

            if ok and is_valid_frame(raw):
                frame, failing_since = raw, None
            else:
                frame = None
                now = clock()
                if failing_since is None:
                    failing_since = now
                elif now - failing_since >= CAMERA_FAILURE_TIMEOUT_S:
                    raise CameraError(
                        f"camera stopped delivering valid frames for {CAMERA_FAILURE_TIMEOUT_S:g} s"
                    )

            shown_message = message if message_frames_left > 0 else None
            message_frames_left = max(0, message_frames_left - 1)
            display.show(draw_overlay(frame, session, shown_message))

            key = display.wait_key(1 if frame is not None else 30)
            if not display.is_open() or key in KEY_QUIT:
                break

            index = class_index_for_key(key, len(session.classes))
            if index is not None:
                session.select(index)
            elif key in KEY_CAPTURE:
                try:
                    message = f"Saved {session.capture(frame).name}"
                except NoClassSelectedError:
                    message = "Select a class first (press 1-%d)" % len(session.classes)
                except InvalidFrameError:
                    message = "No valid frame - nothing saved"
                except OSError as exc:
                    message = f"Save failed: {exc.strerror or exc}"
                message_frames_left = MESSAGE_FRAMES
    finally:
        source.release()
        display.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect labeled webcam images for the dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument(
        "--camera-index", type=int, default=None, help="override camera.index from the config"
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    camera_index = config.camera_index if args.camera_index is None else args.camera_index
    if camera_index < 0:
        print(f"error: --camera-index must be >= 0, got {camera_index}")
        return 2

    session: CollectionSession | None = None
    try:
        print(f"Opening camera {camera_index} ...")
        source = open_camera(camera_index)
        try:
            session = CollectionSession(config.classes, config.paths.dataset_dir)
            display = OpenCvWindow()
        except BaseException:
            source.release()
            raise
        print(f"Saving to: {config.paths.dataset_dir}")
        run_collection(session, source, display)
    except CameraError as exc:
        print(f"camera error: {exc}")
        return 1
    except OSError as exc:
        print(f"file system error: {exc}")
        return 1
    except cv2.error as exc:
        print(
            "OpenCV error (if this mentions imshow/namedWindow, install the GUI build: "
            f"pip install opencv-python and uninstall opencv-python-headless): {exc}"
        )
        return 1
    except KeyboardInterrupt:
        print("Interrupted.")

    if session is not None:
        print("Images per class: " + ", ".join(f"{n}={session.counts[n]}" for n in config.classes))
    print("Check the dataset with: python -m src.dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
