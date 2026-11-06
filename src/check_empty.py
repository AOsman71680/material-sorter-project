"""Check how the live camera compares with the empty-platform photo.

Use this when the sorter says "Object on platform: yes" with nothing on the platform.
It takes the same measurement the sorter uses (``object_present`` in
``src.live_sorter``) on your real camera, prints the numbers, and saves a picture:
the empty photo, the live frame and the pixels that differ, side by side.

Run from the project root with the platform EMPTY:

    python -m src.check_empty
    python -m src.check_empty --camera 0 --seconds 8

Exit codes: 0 finished (see the printed verdict), 1 runtime error, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, ConfigError, load_config
from src.data_collection import CAMERA_FAILURE_TIMEOUT_S, CameraError, is_valid_frame, open_camera
from src.live_sorter import (
    COMPARE_SIZE,
    PRESENCE_FRACTION,
    PRESENCE_PIXEL_DIFF,
    _compare_image,
    make_reference,
)
from src.synthesize_dataset import SynthesisError, find_templates, read_bgr

WARM_UP_FRAMES = 15  # cameras adjust exposure for the first moments; these frames are not measured
DEFAULT_SECONDS = 5.0
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "empty_check.png"
BRIGHTNESS_SHIFT_WARN = 12.0  # mean grey-level difference that points to an exposure or lighting change
PANEL_HEIGHT = 240


def changed_fraction(frame_bgr: np.ndarray, reference: np.ndarray) -> float:
    """Share of pixels the sorter counts as changed. Above ``PRESENCE_FRACTION`` means "object present"."""
    diff = cv2.absdiff(_compare_image(frame_bgr), reference)
    return float((diff > PRESENCE_PIXEL_DIFF).mean())


def _panel(image_bgr: np.ndarray, label: str) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scaled = cv2.resize(image_bgr, (max(1, round(width * PANEL_HEIGHT / height)), PANEL_HEIGHT))
    cv2.putText(scaled, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(scaled, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return scaled


def comparison_image(template_bgr: np.ndarray, frame_bgr: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Empty photo | live frame | changed pixels in white, scaled to the same height."""
    diff = cv2.absdiff(_compare_image(frame_bgr), reference)
    mask = ((diff > PRESENCE_PIXEL_DIFF) * 255).astype(np.uint8)
    mask_bgr = cv2.cvtColor(cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    return np.hstack(
        [_panel(template_bgr, "empty photo"), _panel(frame_bgr, "live camera"), _panel(mask_bgr, "pixels that differ")]
    )


def verdict(fractions: list[float], brightness_shift: float, template_size: tuple[int, int], frame_size: tuple[int, int]) -> list[str]:
    """Plain-language reading of the measurements."""
    median = statistics.median(fractions)
    lines = [
        f"changed pixels while empty: median {100 * median:.1f}%, worst {100 * max(fractions):.1f}% "
        f"(the sorter says 'object' above {100 * PRESENCE_FRACTION:.0f}%)"
    ]
    if median <= PRESENCE_FRACTION and max(fractions) <= PRESENCE_FRACTION:
        lines.append("OK: the empty platform is recognised as empty on this camera.")
        return lines
    if median <= PRESENCE_FRACTION:
        lines.append("Mostly empty, but some frames cross the limit: lighting or the camera is flickering. Keep the light steady.")
        return lines
    lines.append("PROBLEM: the empty platform reads as 'object present'. Likely causes, from the numbers:")
    if abs(brightness_shift) >= BRIGHTNESS_SHIFT_WARN:
        direction = "brighter" if brightness_shift > 0 else "darker"
        lines.append(f"  - The live picture is {abs(brightness_shift):.0f} grey levels {direction} than the empty photo (lighting or exposure changed).")
    if abs(template_size[0] / template_size[1] - frame_size[0] / frame_size[1]) > 0.05:
        lines.append(f"  - The photo is {template_size[0]}x{template_size[1]} but the camera gives {frame_size[0]}x{frame_size[1]}: different shape, so probably a different camera or setting.")
    lines.append("  - Open the saved picture. If the white areas show the whole scene, the camera or its position differs from the one used for the empty photo.")
    lines.append("    If they show a few spots, something else is in view (cables, a hand, a shadow).")
    lines.append("  Fix: retake the empty photo with THIS camera, in its final position and lighting, and replace data\\template\\<empty photo>.")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether the empty platform is recognised as empty.")
    parser.add_argument("--camera", type=int, default=None, help="camera index (default: camera.index from the config)")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="how long to measure (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="where to save the comparison picture (default: data/empty_check.png)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seconds <= 0 or (args.camera is not None and args.camera < 0):
        print("error: --seconds must be positive and --camera 0 or more")
        return 2
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    camera_index = config.camera_index if args.camera is None else args.camera

    source = None
    try:
        templates = find_templates(config.paths.dataset_dir.parent / "template", config.classes)
        template_bgr = read_bgr(templates.background)
        reference = make_reference(template_bgr)
        print(f"empty photo: {templates.background.name} ({template_bgr.shape[1]}x{template_bgr.shape[0]})")
        print(f"opening camera {camera_index} - keep the platform EMPTY ...")
        source = open_camera(camera_index, index_flag="--camera")

        fractions: list[float] = []
        shifts: list[float] = []
        last_frame: np.ndarray | None = None
        seen = 0
        deadline: float | None = None
        failing_since: float | None = None
        while deadline is None or time.monotonic() < deadline:
            ok, raw = source.read()
            if not ok or not is_valid_frame(raw):
                now = time.monotonic()
                failing_since = now if failing_since is None else failing_since
                if now - failing_since >= CAMERA_FAILURE_TIMEOUT_S:
                    raise CameraError(f"camera stopped delivering valid frames for {CAMERA_FAILURE_TIMEOUT_S:g} s")
                time.sleep(0.02)
                continue
            failing_since = None
            seen += 1
            if seen <= WARM_UP_FRAMES:
                continue
            if deadline is None:
                deadline = time.monotonic() + args.seconds
            last_frame = raw
            fractions.append(changed_fraction(raw, reference))
            shifts.append(float(_compare_image(raw).mean()) - float(reference.mean()))
    except (SynthesisError, CameraError) as exc:
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted.")
        return 0
    finally:
        if source is not None:
            source.release()

    if last_frame is None or not fractions:
        print("error: no usable frames from the camera")
        return 1
    frame_size = (last_frame.shape[1], last_frame.shape[0])
    print(f"camera frame: {frame_size[0]}x{frame_size[1]}   frames measured: {len(fractions)}   (compared at {COMPARE_SIZE[0]}x{COMPARE_SIZE[1]})")
    for line in verdict(fractions, statistics.median(shifts), (template_bgr.shape[1], template_bgr.shape[0]), frame_size):
        print(line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), comparison_image(template_bgr, last_frame, reference))
    print(f"picture saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
