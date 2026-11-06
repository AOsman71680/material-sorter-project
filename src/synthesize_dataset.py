"""Build a training dataset from four template photos and one empty platform photo.

Folder layout (``paths.template_dir``, default ``data/template``)::

    Metal.jpg      one object on the platform
    Metal_2.jpg    optional: another face of the same Metal object
    Plastic.jpg    one object on the platform
    Paper.jpg      one object on the platform
    Organic.jpg    one object on the platform
    empty.jpg      the same platform with nothing on it

Several photos of one object (different faces, sides or poses) go in the same
folder. A photo counts as another face when its name is the class name followed by
a separator or number: ``Metal_2.jpg``, ``Metal 3.png``, ``Metal-side.jpg``,
``Metal2.jpg``. ``--per-template`` is the total per class and is split evenly
across its faces, so every class keeps the same number of images.

Each object is isolated by comparing it with the empty platform photo, and the
mask is refined with GrabCut. The object is then pasted onto the empty platform
many times with random position, rotation, scale, hue, saturation, lighting,
shadow, blur and noise. The results go into ``paths.dataset_dir/<class>/`` in the
same layout the collection tool writes, plus a grouping CSV so that every variant
of one template object (all its faces included) stays in one train, validation or test split.

Run from the project root:

    python -m src.synthesize_dataset
    python -m src.synthesize_dataset --per-template 300 --seed 7 --overwrite

Limits you should know about:

- Synthetic images are a rough stand-in for real photos. Cut-out edges, shadows
  and lighting differ from real captures, so validation numbers on synthetic data
  overstate real accuracy.
- Each class is one object, however many faces you photograph. Every variant lands in one split, so
  the test split cannot show how the model handles a *new* object of a class.
- The isolation step needs the template and the empty photo taken with the same
  camera, framing and platform. An object that fills more than half the frame
  breaks the exposure matching.

Exit codes: 0 success, 1 template or generation error, 2 bad config or arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from src.dataset import is_image_file
from src.preprocessing import load_rgb

# Template file names (case-insensitive, without extension) -> project class.
# "organic" is the fourth material in the template folder; the project's fourth
# class is called "other" in config/config.yaml, so it maps to that class.
TEMPLATE_TO_CLASS: dict[str, str] = {
    "metal": "metal",
    "plastic": "plastic",
    "paper": "paper",
    "organic": "other",
}

MIN_DIFFERENCE = 18  # smallest colour distance (0-255) that counts as object, not platform
MIN_OBJECT_FRACTION = 0.002  # an object must cover at least 0.2% of the photo
MAX_OBJECT_FRACTION = 0.5  # more than half the frame breaks exposure matching
FIT_FRACTION = 0.9  # a pasted object may use at most 90% of the frame in each direction
GRABCUT_ITERATIONS = 4
JPEG_QUALITY_RANGE = (85, 95)
PARTIAL_SUFFIX = ".partial"
GENERATION_REPORT_NAME = "synthetic_report.json"


class SynthesisError(Exception):
    """Templates or settings cannot be used to build the dataset."""


@dataclass(frozen=True)
class SynthesisSettings:
    per_template: int = 200  # images generated from each template object
    seed: int = 42
    min_scale: float = 0.6  # object size relative to the template photo
    max_scale: float = 1.2
    max_hue_shift_degrees: float = 30.0  # hue moves by up to +/- this many degrees
    saturation_jitter: float = 0.2  # +/- relative change
    value_jitter: float = 0.2  # +/- relative change of object brightness
    brightness_log_range: float = 0.25  # whole-scene exposure factor exp(+/- this)
    contrast_range: float = 0.15  # +/- relative contrast change
    tint_range: float = 0.08  # per-channel colour temperature gains, +/- this
    gradient_strength: float = 0.15  # one-sided light falloff across the frame
    blur_probability: float = 0.5  # chance of a slight defocus blur
    max_noise_sigma: float = 5.0  # gaussian sensor noise, 0 to this (0-255 scale)

    def __post_init__(self) -> None:
        if not 1 <= self.per_template <= 5000:
            raise SynthesisError(f"per_template must be between 1 and 5000, got {self.per_template}")
        if self.seed < 0:
            raise SynthesisError(f"seed must be >= 0, got {self.seed}")
        if not 0.1 <= self.min_scale <= self.max_scale <= 2.0:
            raise SynthesisError(
                f"scale range must satisfy 0.1 <= min_scale <= max_scale <= 2.0, "
                f"got {self.min_scale}..{self.max_scale}"
            )
        for name, value, high in (
            ("max_hue_shift_degrees", self.max_hue_shift_degrees, 180.0),
            ("saturation_jitter", self.saturation_jitter, 0.9),
            ("value_jitter", self.value_jitter, 0.9),
            ("brightness_log_range", self.brightness_log_range, 1.5),
            ("contrast_range", self.contrast_range, 0.9),
            ("tint_range", self.tint_range, 0.5),
            ("gradient_strength", self.gradient_strength, 0.9),
            ("max_noise_sigma", self.max_noise_sigma, 50.0),
        ):
            if not 0.0 <= value <= high:
                raise SynthesisError(f"{name} must be between 0 and {high:g}, got {value}")
        if not 0.0 <= self.blur_probability <= 1.0:
            raise SynthesisError(f"blur_probability must be between 0 and 1, got {self.blur_probability}")


@dataclass(frozen=True)
class TemplateObject:
    class_name: str  # project class, e.g. "other"
    files: tuple[Path, ...]  # one photo per face of the object; at least one
    stem: str  # canonical template name, e.g. "Organic"

    @property
    def object_id(self) -> str:
        """Group identifier: all variants of this object, from every face, share it."""
        return f"synthetic-{self.class_name}-{safe_name(self.stem)}"


@dataclass(frozen=True)
class TemplateSet:
    objects: tuple[TemplateObject, ...]
    background: Path  # the empty platform photo


def safe_name(value: str) -> str:
    """ASCII-only name for file names and group IDs."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "item"


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------


def _template_key(stem: str) -> str | None:
    """Template name a file stem belongs to, or None. ``Metal``, ``metal_2`` and ``Metal-side`` all give ``metal``;
    ``metallic`` does not, because the class name must end at a separator, digit or the end of the name."""
    text = stem.strip().lower()
    for key in TEMPLATE_TO_CLASS:
        if text.startswith(key) and (len(text) == len(key) or not text[len(key)].isalpha()):
            return key
    return None


def find_templates(template_dir: Path, classes: tuple[str, ...]) -> TemplateSet:
    """Find the four object templates and the empty platform photo in ``template_dir``.

    Each template name (Metal, Plastic, Paper, Organic) must have at least one photo.
    Extra photos named like ``Metal_2`` or ``Metal-side`` are more faces of the same
    object. Exactly one image must be the empty platform (any other name).
    """
    if not template_dir.is_dir():
        raise SynthesisError(f"template folder not found: {template_dir}")
    images = sorted(p for p in template_dir.iterdir() if is_image_file(p))
    if not images:
        raise SynthesisError(f"no images in {template_dir}. Expected Metal, Plastic, Paper, Organic and one empty platform photo.")

    found: dict[str, list[Path]] = {}
    background: list[Path] = []
    for path in images:
        key = _template_key(path.stem)
        if key is not None:
            found.setdefault(key, []).append(path)
        else:
            background.append(path)

    problems: list[str] = []
    for key in TEMPLATE_TO_CLASS:
        matches = found.get(key, [])
        if not matches:
            problems.append(f"missing template image for {key.capitalize()}")
    if len(background) != 1:
        names = ", ".join(p.name for p in background) or "none"
        problems.append(
            f"expected exactly one empty platform photo (a file whose name does not start with Metal, Plastic, Paper or Organic); found: {names}"
        )
    if problems:
        raise SynthesisError("; ".join(problems))

    objects: list[TemplateObject] = []
    for key, class_name in TEMPLATE_TO_CLASS.items():
        if class_name not in classes:
            raise SynthesisError(f"template '{key}' maps to class '{class_name}', which is not in config classes {classes}")
        objects.append(TemplateObject(class_name=class_name, files=tuple(found[key]), stem=key.capitalize()))
    return TemplateSet(objects=tuple(objects), background=background[0])


def read_bgr(path: Path) -> np.ndarray:
    try:
        rgb = load_rgb(path)
    except ValueError as exc:
        raise SynthesisError(f"{path.name} is not a readable image ({exc})") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# Isolation
# --------------------------------------------------------------------------


def _border_colour(image: np.ndarray) -> np.ndarray:
    """Median colour of the outer strip of the photo, where the platform is visible."""
    height, width = image.shape[:2]
    top, left = max(1, height // 12), max(1, width // 12)
    strip = np.concatenate(
        [
            image[:top].reshape(-1, 3),
            image[-top:].reshape(-1, 3),
            image[:, :left].reshape(-1, 3),
            image[:, -left:].reshape(-1, 3),
        ]
    )
    return np.median(strip, axis=0)


def _match_exposure(obj: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Scale each colour channel of the object photo so the platform colour matches the empty photo.

    The platform colour is read from the outer border of both photos. An object
    that fills the frame would bias a whole-photo median; an object that does
    not reach the border leaves the border reading correct.
    """
    obj_border = _border_colour(obj)
    bg_border = _border_colour(background)
    scaled = obj.astype(np.float32)
    for channel in range(3):
        ratio = float(bg_border[channel]) / max(float(obj_border[channel]), 1.0)
        scaled[..., channel] *= float(np.clip(ratio, 0.5, 2.0))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == best, 255, 0).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes. Assumes the corner pixel is background (the platform)."""
    if mask[0, 0] != 0:
        return mask
    flood = mask.copy()
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, padded, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def _refine_with_grabcut(obj: np.ndarray, component: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inner = cv2.erode(component, kernel)
    outer = cv2.dilate(component, kernel)
    labels = np.full(component.shape, cv2.GC_BGD, np.uint8)
    labels[outer > 0] = cv2.GC_PR_BGD
    labels[component > 0] = cv2.GC_PR_FGD
    labels[inner > 0] = cv2.GC_FGD
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(obj, labels, None, background_model, foreground_model, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
    refined = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return refined if refined.any() else component


def isolate_object(object_bgr: np.ndarray, background_bgr: np.ndarray, name: str = "template") -> np.ndarray:
    """Return a uint8 mask (255 = object) for the object photo, using the empty platform photo."""
    if object_bgr.shape != background_bgr.shape:
        raise SynthesisError(
            f"{name} is {object_bgr.shape[1]}x{object_bgr.shape[0]} but the empty platform photo is "
            f"{background_bgr.shape[1]}x{background_bgr.shape[0]}. Take all five photos with the same camera and framing."
        )
    matched = _match_exposure(object_bgr, background_bgr)
    obj_blur = cv2.GaussianBlur(matched, (5, 5), 0).astype(np.float32)
    bg_blur = cv2.GaussianBlur(background_bgr, (5, 5), 0).astype(np.float32)
    distance = np.clip(np.sqrt(((obj_blur - bg_blur) ** 2).sum(axis=2)), 0, 255).astype(np.uint8)

    otsu, _ = cv2.threshold(distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = np.where(distance >= max(otsu, MIN_DIFFERENCE), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    component = _largest_component(binary)
    if component is None:
        raise SynthesisError(f"could not find an object in {name}. Check that it is on the platform and that the photo matches the empty one.")
    component = _fill_holes(component)
    refined = _largest_component(_refine_with_grabcut(matched, component))
    if refined is None:
        refined = component
    refined = _fill_holes(refined)

    fraction = float(refined.mean()) / 255.0
    if fraction < MIN_OBJECT_FRACTION:
        raise SynthesisError(f"the object in {name} covers only {fraction:.2%} of the photo; it is too small to use.")
    if fraction > MAX_OBJECT_FRACTION:
        raise SynthesisError(
            f"the object in {name} covers {fraction:.0%} of the photo; the limit is {MAX_OBJECT_FRACTION:.0%}. "
            "Use a larger camera distance."
        )
    return refined


def make_cutout(object_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop the object with its mask as alpha (BGRA, uint8). The alpha edge is softened."""
    x, y, w, h = cv2.boundingRect(mask)
    pad = 2
    height, width = mask.shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
    bgr = object_bgr[y0:y1, x0:x1].copy()
    alpha = cv2.GaussianBlur(mask[y0:y1, x0:x1], (3, 3), 0)
    return np.dstack([bgr, alpha])


# --------------------------------------------------------------------------
# Augmentation and compositing
# --------------------------------------------------------------------------


def _resize(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    size = (max(2, round(width * scale)), max(2, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def _rotate_expand(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate with an enlarged canvas so nothing is cropped. Works for BGRA images."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return cv2.warpAffine(image, matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def _jitter_colour(bgr: np.ndarray, rng: np.random.Generator, settings: SynthesisSettings) -> np.ndarray:
    """Change the object's hue, saturation and brightness. Only the object is changed."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    shift_degrees = rng.uniform(-settings.max_hue_shift_degrees, settings.max_hue_shift_degrees)
    hsv[..., 0] = (hsv[..., 0] + shift_degrees / 2.0) % 180.0  # OpenCV hue runs 0-179
    hsv[..., 1] *= rng.uniform(1 - settings.saturation_jitter, 1 + settings.saturation_jitter)
    hsv[..., 2] *= rng.uniform(1 - settings.value_jitter, 1 + settings.value_jitter)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def compose_scene(
    background_bgr: np.ndarray,
    cutout_bgra: np.ndarray,
    rng: np.random.Generator,
    settings: SynthesisSettings,
) -> np.ndarray:
    """Place the object at a random spot on the empty platform with random lighting. Returns uint8 BGR."""
    height, width = background_bgr.shape[:2]
    obj = _resize(cutout_bgra, rng.uniform(settings.min_scale, settings.max_scale))
    obj = _rotate_expand(obj, rng.uniform(0.0, 360.0))
    fit = min(1.0, FIT_FRACTION * width / obj.shape[1], FIT_FRACTION * height / obj.shape[0])
    if fit < 1.0:
        obj = _resize(obj, fit)
    obj[..., :3] = _jitter_colour(obj[..., :3], rng, settings)

    obj_h, obj_w = obj.shape[:2]
    x = int(rng.integers(0, width - obj_w + 1))  # anywhere on the platform, fully in frame
    y = int(rng.integers(0, height - obj_h + 1))

    scene = background_bgr.astype(np.float32)

    # Soft shadow, offset in a random direction and darkening the platform.
    alpha_full = np.zeros((height, width), np.float32)
    alpha_full[y : y + obj_h, x : x + obj_w] = obj[..., 3] / 255.0
    direction = rng.uniform(0.0, 2.0 * math.pi)
    distance = rng.uniform(0.01, 0.04) * min(height, width)
    shift = np.float32([[1, 0, distance * math.cos(direction)], [0, 1, distance * math.sin(direction)]])
    shadow = cv2.warpAffine(alpha_full, shift, (width, height))
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=max(2.0, 0.01 * min(height, width)))
    scene *= (1.0 - rng.uniform(0.2, 0.5) * shadow)[..., None]

    # Alpha-blend the object over its spot.
    alpha = obj[..., 3:4].astype(np.float32) / 255.0
    region = scene[y : y + obj_h, x : x + obj_w]
    scene[y : y + obj_h, x : x + obj_w] = region * (1.0 - alpha) + obj[..., :3].astype(np.float32) * alpha

    # Scene lighting: exposure, colour temperature, contrast, and a light falloff across the frame.
    scene *= math.exp(rng.uniform(-settings.brightness_log_range, settings.brightness_log_range))
    scene *= rng.uniform(1 - settings.tint_range, 1 + settings.tint_range, size=3).astype(np.float32)
    mean = float(scene.mean())
    scene = (scene - mean) * rng.uniform(1 - settings.contrast_range, 1 + settings.contrast_range) + mean
    angle = rng.uniform(0.0, 2.0 * math.pi)
    strength = rng.uniform(0.0, settings.gradient_strength)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    projection = (xx / width - 0.5) * math.cos(angle) + (yy / height - 0.5) * math.sin(angle)
    scene *= (1.0 + 2.0 * strength * projection)[..., None]

    # Camera effects: optional slight defocus and sensor noise.
    if rng.random() < settings.blur_probability:
        scene = cv2.GaussianBlur(scene, (0, 0), sigmaX=rng.uniform(0.3, 1.2))
    scene += rng.normal(0.0, rng.uniform(0.0, settings.max_noise_sigma), scene.shape).astype(np.float32)
    return np.clip(scene, 0, 255).astype(np.uint8)


def encode_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise SynthesisError("could not encode a generated image as JPEG")
    return buffer.tobytes()


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


@dataclass
class GeneratedItem:
    relative_path: str  # e.g. "other/synth_Organic_0007.jpg"
    object_id: str
    session_id: str


@dataclass
class SynthesisResult:
    items: list[GeneratedItem] = field(default_factory=list)
    per_object: dict[str, dict[str, Any]] = field(default_factory=dict)


def _variant_rng(settings: SynthesisSettings, object_index: int, face_index: int, variant: int) -> np.random.Generator:
    return np.random.default_rng([settings.seed, object_index, face_index, variant])


def _split_evenly(total: int, parts: int) -> list[int]:
    """``total`` split into ``parts`` counts that differ by at most one; the first parts get the remainder."""
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _face_labels(files: tuple[Path, ...]) -> list[str]:
    """File-name label per face: the safe file stem, or ``<stem>-<n>`` when two faces would collide
    (for example Metal.png and Metal.jpg, or Metal.jpg and metal.jpg on a case-insensitive disk)."""
    stems = [safe_name(f.stem) for f in files]
    lowered = [s.lower() for s in stems]
    return [s if lowered.count(s.lower()) == 1 else f"{s}-{i + 1}" for i, s in enumerate(stems)]


def generate_images(
    templates: TemplateSet,
    settings: SynthesisSettings,
    output_dir: Path,
) -> SynthesisResult:
    """Write all generated images under ``output_dir/<class>/``. Returns what was written."""
    background = read_bgr(templates.background)
    result = SynthesisResult()
    session_id = f"synthetic-seed-{settings.seed}"

    for template in templates.objects:
        faces = len(template.files)
        if settings.per_template < faces:
            raise SynthesisError(
                f"{template.stem} has {faces} faces but per_template is {settings.per_template}; "
                f"every face needs at least one image"
            )

    for object_index, template in enumerate(templates.objects):
        class_dir = output_dir / template.class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        counts = _split_evenly(settings.per_template, len(template.files))
        labels = _face_labels(template.files)
        areas: list[float] = []
        per_face: dict[str, int] = {}

        for face_index, face_file in enumerate(template.files):
            object_bgr = read_bgr(face_file)
            mask = isolate_object(object_bgr, background, name=face_file.name)
            cutout = make_cutout(object_bgr, mask)
            areas.append(float(mask.mean()) / 255.0)
            per_face[face_file.name] = counts[face_index]
            for variant in range(counts[face_index]):
                rng = _variant_rng(settings, object_index, face_index, variant)
                scene = compose_scene(background, cutout, rng, settings)
                quality = int(rng.integers(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1] + 1))
                name = f"synth_{labels[face_index]}_{variant:04d}.jpg"
                (class_dir / name).write_bytes(encode_jpeg(scene, quality))
                result.items.append(
                    GeneratedItem(
                        relative_path=f"{template.class_name}/{name}",
                        object_id=template.object_id,
                        session_id=session_id,
                    )
                )

        result.per_object[template.object_id] = {
            "template_files": [f.name for f in template.files],
            "class": template.class_name,
            "object_area_fraction": round(sum(areas) / len(areas), 6),
            "images": settings.per_template,
            "images_per_face": per_face,
        }
    return result


def _backup_name(path: Path, stamp: str) -> Path:
    return path.with_name(f"{path.name}.{stamp}.bak")


def check_outputs_free(dataset_dir: Path, group_path: Path, overwrite: bool) -> list[Path]:
    """Return existing outputs that would be backed up. Refuse unless ``overwrite`` is set."""
    existing: list[Path] = []
    if dataset_dir.is_dir() and any(is_image_file(p) for p in dataset_dir.rglob("*")):
        existing.append(dataset_dir)
    if group_path.exists():
        existing.append(group_path)
    if existing and not overwrite:
        names = ", ".join(str(p) for p in existing)
        raise SynthesisError(
            f"outputs already exist ({names}). Use --overwrite to move them to a .bak copy first."
        )
    return existing


def _write_csv_atomic(path: Path, rows: list[GeneratedItem]) -> None:
    lines = ["path,object_id,session_id"]
    lines += [f"{row.relative_path},{row.object_id},{row.session_id}" for row in rows]
    temp = path.with_name(path.name + ".tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(path)


def run_synthesis(
    config: Config,
    template_dir: Path,
    settings: SynthesisSettings,
    *,
    overwrite: bool = False,
    stamp: str | None = None,
) -> tuple[SynthesisResult, Path]:
    """Build the dataset. Returns the result and the report path.

    Images are written to a staging folder first. Existing outputs are moved to
    ``.<stamp>.bak`` only after generation succeeds, so a failure leaves the
    current dataset untouched.
    """
    dataset_dir = config.paths.dataset_dir
    group_path = config.dataset.group_metadata
    report_path = group_path.parent / GENERATION_REPORT_NAME
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    templates = find_templates(template_dir, config.classes)
    check_outputs_free(dataset_dir, group_path, overwrite)

    staging = dataset_dir.with_name(dataset_dir.name + PARTIAL_SUFFIX)
    if staging.exists():
        shutil.rmtree(staging)
    try:
        result = generate_images(templates, settings, staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Back up what is being replaced, then move the staged dataset into place.
    # Anything already at dataset_dir is moved aside, never deleted, in case it holds notes or real photos.
    if dataset_dir.exists():
        dataset_dir.rename(_backup_name(dataset_dir, stamp))
    staging.rename(dataset_dir)
    if group_path.exists():
        group_path.rename(_backup_name(group_path, stamp))
    group_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(group_path, result.items)

    report = {
        "format_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": settings.seed,
        "settings": asdict(settings),
        "background_file": templates.background.name,
        "total_images": len(result.items),
        "objects": result.per_object,
        "notes": [
            "Synthetic images are a stand-in for real photos. Validation numbers on them overstate real accuracy.",
            "Each class is one object (all its faces share a split), so the test split cannot show performance on a new object of a class.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return result, report_path


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a training dataset from template photos.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument(
        "--template-dir", type=Path, default=None, help="folder with Metal, Plastic, Paper, Organic and the empty platform photo (default: data/template)"
    )
    parser.add_argument("--per-template", type=int, default=SynthesisSettings.per_template, help="images per class, split evenly across its faces (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=SynthesisSettings.seed, help="random seed (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true", help="move an existing dataset and group file to .bak copies first")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    try:
        settings = SynthesisSettings(per_template=args.per_template, seed=args.seed)
    except SynthesisError as exc:
        print(f"error: {exc}")
        return 2
    template_dir = args.template_dir or (config.paths.dataset_dir.parent / "template")

    try:
        result, report_path = run_synthesis(config, template_dir, settings, overwrite=args.overwrite)
    except SynthesisError as exc:
        print(f"error: {exc}")
        return 1
    except OSError as exc:
        print(f"file system error: {exc}")
        return 1

    counts = {name: 0 for name in config.classes}
    for item in result.items:
        counts[item.relative_path.split("/", 1)[0]] += 1
    print(f"Wrote {len(result.items)} images to {config.paths.dataset_dir}")
    print("Per class: " + ", ".join(f"{name}={counts[name]}" for name in config.classes))
    print(f"Grouping file: {config.dataset.group_metadata}")
    print(f"Report: {report_path}")
    print("Check the dataset with: python -m src.dataset")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    raise SystemExit(main())
