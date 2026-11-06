"""Live webcam material prediction preview (informational only).

Opens the webcam, classifies frames with a trained checkpoint and draws the
prediction on the live picture. It reuses the Task 6 inference path
(``src.predict``) and the Task 2 camera helpers (``src.data_collection``).

Run from the project root:

    python -m src.webcam_predict
    python -m src.webcam_predict --checkpoint models\\best_model.pth --camera 0
    python -m src.webcam_predict --device cpu --threshold 0.8 --inference-every-n-frames 5

Nothing is saved, nothing is sent to any hardware, and nothing is sorted. Press
q or Esc to quit. The model's probabilities are not calibrated and the model
has not been validated on real materials.

Exit codes: 0 normal exit (q, Esc, window closed, or Ctrl+C), 1 runtime error
(checkpoint, camera, frame read, CUDA), 2 bad config or arguments.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from src.data_collection import (
    CAMERA_FAILURE_TIMEOUT_S,
    KEY_QUIT,
    CameraError,
    Display,
    FrameSource,
    InvalidFrameError,
    OpenCvWindow,
    _GREEN,
    _RED,
    _WHITE,
    _YELLOW,
    _put_text,
    is_valid_frame,
    open_camera,
)
from src.predict import LoadedModel, PredictionError, load_model_for_inference, predict_probabilities
from src.preprocessing import ensure_rgb

WINDOW_TITLE = "Material sorter - live prediction"
PREVIEW_NOTE = "Model output only - not verified on real materials"
PREVIEW_NOTE_2 = "Informational only - no sorting action. Q or Esc quits."
WAITING_TEXT = "Predicted: waiting for first prediction..."


@dataclass(frozen=True)
class LivePrediction:
    """One prediction. ``predicted_class`` is always one of the configured classes."""

    predicted_class: str
    max_probability: float
    probabilities: dict[str, float]  # class name -> softmax probability, in class order
    threshold: float | None
    low_confidence: bool | None  # None when no threshold was given


@dataclass
class LoopStats:
    frames_shown: int = 0
    inferences: int = 0


# --------------------------------------------------------------------------
# Predictions and frames
# --------------------------------------------------------------------------


def summarize_probabilities(
    probs: np.ndarray, classes: tuple[str, ...], threshold: float | None
) -> LivePrediction:
    """Map the four softmax probabilities to the predicted class, as Task 6 does.

    ``low_confidence`` means the top probability is below the threshold. It is
    a display-only flag and never adds a class.
    """
    if probs.shape != (len(classes),):
        raise PredictionError(f"expected {len(classes)} probabilities, got shape {probs.shape}")
    best = int(np.argmax(probs))
    max_probability = float(probs[best])
    low = None if threshold is None else bool(max_probability < threshold)
    return LivePrediction(
        predicted_class=classes[best],
        max_probability=max_probability,
        probabilities={name: float(p) for name, p in zip(classes, probs)},
        threshold=threshold,
        low_confidence=low,
    )


def make_predictor(
    loaded: LoadedModel,
    classes: tuple[str, ...],
    threshold: float | None,
) -> Callable[[np.ndarray], LivePrediction]:
    """Build a function that turns one RGB frame into a ``LivePrediction``."""

    def predict(rgb: np.ndarray) -> LivePrediction:
        probs = predict_probabilities(loaded.model, rgb, loaded.image, loaded.device)
        return summarize_probabilities(probs, classes, threshold)

    return predict


def frame_to_rgb(frame: object) -> np.ndarray:
    """Convert an OpenCV BGR camera frame to the RGB array the preprocessing expects."""
    if not is_valid_frame(frame):
        raise InvalidFrameError("camera frame is missing or is not an 8-bit 3-channel image")
    return ensure_rgb(frame, is_bgr=True)  # type: ignore[arg-type]


def should_run_inference(valid_frame_index: int, every_n: int) -> bool:
    """Run the model on valid frames 0, N, 2N, ... (so the first frame is always predicted)."""
    return valid_frame_index % every_n == 0


# --------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------


def overlay_text(prediction: LivePrediction | None) -> list[str]:
    """The text lines drawn on the frame (kept separate so it can be tested directly)."""
    if prediction is None:
        return [WAITING_TEXT]
    lines = [
        f"Predicted: {prediction.predicted_class}",
        f"Confidence: {prediction.max_probability * 100:.1f}%",
    ]
    for name, value in prediction.probabilities.items():
        mark = ">" if name == prediction.predicted_class else " "
        lines.append(f"{mark} {name}: {value * 100:.1f}%")
    if prediction.low_confidence:
        lines.append(f"LOW CONFIDENCE (below {prediction.threshold:g})")
    return lines


def draw_prediction(frame: np.ndarray, prediction: LivePrediction | None) -> np.ndarray:
    """Return a copy of ``frame`` with the prediction drawn on it. The input is not changed."""
    canvas = frame.copy()
    height = canvas.shape[0]
    for index, line in enumerate(overlay_text(prediction)):
        if prediction is None:
            color = _YELLOW
        elif line.startswith("LOW CONFIDENCE"):
            color = _RED
        elif index == 0:
            color = _GREEN
        else:
            color = _WHITE
        _put_text(canvas, line, (10, 24 + 22 * index), color)
    _put_text(canvas, PREVIEW_NOTE, (10, height - 40), _YELLOW)
    _put_text(canvas, PREVIEW_NOTE_2, (10, height - 14), _WHITE)
    return canvas


# --------------------------------------------------------------------------
# Live loop
# --------------------------------------------------------------------------


def run_prediction_loop(
    predict: Callable[[np.ndarray], LivePrediction],
    source: FrameSource,
    display: Display,
    *,
    every_n: int = 1,
    clock: Callable[[], float] | None = None,
) -> LoopStats:
    """Show the live feed, predicting every ``every_n``-th valid frame.

    Between predictions the latest prediction stays on screen. Always releases
    the camera and closes the window on exit, including after errors.
    Raises ``CameraError`` if no valid frame arrives for ``CAMERA_FAILURE_TIMEOUT_S``.
    """
    if every_n < 1:
        raise ValueError(f"every_n must be at least 1, got {every_n}")
    now_seconds = clock or time.monotonic
    stats = LoopStats()
    latest: LivePrediction | None = None
    valid_index = 0
    failing_since: float | None = None
    try:
        while True:
            try:
                ok, raw = source.read()
            except cv2.error:
                ok, raw = False, None

            if ok and is_valid_frame(raw):
                failing_since = None
                if should_run_inference(valid_index, every_n):
                    latest = predict(frame_to_rgb(raw))
                    stats.inferences += 1
                valid_index += 1
                display.show(draw_prediction(raw, latest))  # type: ignore[arg-type]
                stats.frames_shown += 1
                key = display.wait_key(1)
            else:
                now = now_seconds()
                if failing_since is None:
                    failing_since = now
                elif now - failing_since >= CAMERA_FAILURE_TIMEOUT_S:
                    raise CameraError(
                        f"camera stopped delivering valid frames for {CAMERA_FAILURE_TIMEOUT_S:g} s"
                    )
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
    parser = argparse.ArgumentParser(description="Live webcam material prediction (informational only).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint to use (default: best_model.pth in paths.model_dir)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="camera index (default: camera.index from the config, normally 0)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="override training.device (cuda fails clearly if no GPU is available)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="optional: show LOW CONFIDENCE when the top probability is below this value (0 to 1)",
    )
    parser.add_argument(
        "--inference-every-n-frames",
        type=int,
        default=1,
        help="run the model on every Nth camera frame (default 1); the last prediction stays on screen",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    args = parser.parse_args(argv)

    if args.threshold is not None and not 0.0 < args.threshold < 1.0:
        print(f"error: --threshold must be strictly between 0 and 1, got {args.threshold}")
        return 2
    if args.inference_every_n_frames < 1:
        print(f"error: --inference-every-n-frames must be at least 1, got {args.inference_every_n_frames}")
        return 2
    if args.camera is not None and args.camera < 0:
        print(f"error: --camera must be 0 or more, got {args.camera}")
        return 2
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2

    camera_index = config.camera_index if args.camera is None else args.camera
    checkpoint = args.checkpoint or (config.paths.model_dir / "best_model.pth")
    if args.checkpoint is None:
        print(f"no --checkpoint given; using {checkpoint}")

    stats: LoopStats | None = None
    try:
        loaded = load_model_for_inference(config, checkpoint, device_setting=args.device)
        predict = make_predictor(loaded, config.classes, args.threshold)
        print(f"Model: {loaded.metadata['architecture']}   Device: {loaded.device.type}")
        print(f"Opening camera {camera_index} ... press q or Esc in the window to quit.")
        source = open_camera(camera_index, index_flag="--camera")
        try:
            display = OpenCvWindow(WINDOW_TITLE)
        except BaseException:
            source.release()
            raise
        stats = run_prediction_loop(predict, source, display, every_n=args.inference_every_n_frames)
    except (PredictionError, CameraError, InvalidFrameError) as exc:
        print(f"error: {exc}")
        return 1
    except cv2.error as exc:
        print(
            "error: OpenCV error (if this mentions imshow/namedWindow, install the GUI build: "
            f"pip install opencv-python and uninstall opencv-python-headless): {exc}"
        )
        return 1
    except KeyboardInterrupt:
        print("Interrupted.")
        return 0

    print(f"Stopped after {stats.frames_shown} frames and {stats.inferences} predictions.")
    return 0


if __name__ == "__main__":
    # Keep printing safe when a file name has non-ASCII characters and output is redirected.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    raise SystemExit(main())
