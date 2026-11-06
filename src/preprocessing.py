"""Image preprocessing shared by training, evaluation and (later) inference.

Pipeline for every image:

1. Load as RGB uint8 (``load_rgb``; BGR camera frames use ``ensure_rgb``).
2. Resize the shorter side to ``image.width`` / ``image.height`` keeping the
   aspect ratio, then take the centre crop. This is the *base* image.
3. Training only: random affine jitter (rotation, shift, zoom) and brightness /
   contrast changes, all drawn from a seeded ``numpy`` Generator.
4. Scale to [0, 1], then apply the ImageNet mean/std normalisation that the
   pretrained backbones (mobilenet_v2, efficientnet_b0, resnet18) expect.
5. Reorder HWC -> CHW (float32), the layout PyTorch models use.

Validation and test images use steps 1, 2, 4 and 5 only, so they are
deterministic. Inference must call ``preprocess_eval`` too, so the model sees
the same input as during evaluation.

Only NumPy and OpenCV are used here; no deep-learning framework is needed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import AugmentationConfig, ImageConfig

# ImageNet statistics used by the torchvision pretrained backbones.
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(path: Path) -> np.ndarray:
    """Load an image file as an RGB uint8 array of shape (H, W, 3).

    Reads the bytes with NumPy first, so paths containing non-ASCII characters
    work on Windows (``cv2.imread`` often fails on them). Raises ``ValueError``
    if the file cannot be decoded.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"cannot read image {path}: {exc}") from exc
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if bgr is None:
        raise ValueError(f"cannot decode image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def ensure_rgb(image: np.ndarray, *, is_bgr: bool) -> np.ndarray:
    """Convert an in-memory image (grey, BGR/RGB, or with alpha) to RGB uint8."""
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 pixels, got {image.dtype}")
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected an image with 1, 3 or 4 channels, got shape {image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image.copy()


def resize_and_center_crop(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale so both sides cover the target, then crop the centre (no padding)."""
    h, w = rgb.shape[:2]
    scale = max(width / w, height / h)
    new_w = max(width, round(w * scale))
    new_h = max(height, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interpolation)
    x0 = (new_w - width) // 2
    y0 = (new_h - height) // 2
    return np.ascontiguousarray(resized[y0 : y0 + height, x0 : x0 + width])


def _to_model_input(unit_hwc: np.ndarray) -> np.ndarray:
    """[0, 1] HWC float image -> ImageNet-normalised CHW float32."""
    normalized = (unit_hwc - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)


def preprocess_eval(rgb: np.ndarray, image: ImageConfig) -> np.ndarray:
    """Deterministic preprocessing for validation, test and inference."""
    base = resize_and_center_crop(rgb, image.width, image.height)
    return _to_model_input(base.astype(np.float32) / 255.0)


def augment_training(
    base: np.ndarray,
    augmentation: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random rotation, shift, zoom and brightness/contrast change of a base crop.

    Returns a [0, 1] float32 HWC image. The same ``rng`` state always gives the
    same output. Reflection is used at the borders so no black corners appear.
    """
    h, w = base.shape[:2]
    angle = rng.uniform(-augmentation.max_rotation_degrees, augmentation.max_rotation_degrees)
    zoom = rng.uniform(augmentation.min_scale, augmentation.max_scale)
    shift_x = rng.uniform(-augmentation.max_translation, augmentation.max_translation) * w
    shift_y = rng.uniform(-augmentation.max_translation, augmentation.max_translation) * h

    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), float(zoom))
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    warped = cv2.warpAffine(
        base,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    image = warped.astype(np.float32) / 255.0
    contrast = rng.uniform(1.0 - augmentation.contrast, 1.0 + augmentation.contrast)
    brightness = rng.uniform(1.0 - augmentation.brightness, 1.0 + augmentation.brightness)
    mean = float(image.mean())
    image = (image - mean) * contrast + mean
    image = image * brightness
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def preprocess_training(
    rgb: np.ndarray,
    image: ImageConfig,
    augmentation: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random augmentation followed by the same normalisation as ``preprocess_eval``."""
    base = resize_and_center_crop(rgb, image.width, image.height)
    return _to_model_input(augment_training(base, augmentation, rng))
