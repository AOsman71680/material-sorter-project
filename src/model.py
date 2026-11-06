"""Model construction and checkpoint saving/loading for transfer learning.

``build_model`` returns a torchvision backbone whose final layer is replaced
with a new layer for the configured material classes. With ``pretrained=True``
the backbone starts from ImageNet weights. The first use downloads them to the
torch cache (about 10 MB for MobileNetV3 Small). Tests never download weights.

Checkpoints are dictionaries, not bare state dicts. Each one records the class
order, input size and normalisation, so inference can check that it matches
the configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

from src.config import SUPPORTED_ARCHITECTURES

CHECKPOINT_FORMAT_VERSION = 1
REQUIRED_CHECKPOINT_KEYS: tuple[str, ...] = (
    "format_version",
    "architecture",
    "classes",
    "image_width",
    "image_height",
    "state_dict",
)

# architecture -> (torchvision constructor, ImageNet weights enum)
_BACKBONES: dict[str, tuple[Any, Any]] = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}


class ModelError(Exception):
    """A model or checkpoint cannot be built, saved, or loaded."""


def build_model(architecture: str, num_classes: int, *, pretrained: bool) -> nn.Module:
    """Create the backbone with a new ``num_classes``-way final layer."""
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ModelError(
            f"unsupported architecture {architecture!r}; expected one of {list(SUPPORTED_ARCHITECTURES)}"
        )
    if num_classes < 2:
        raise ModelError(f"a classifier needs at least 2 classes, got {num_classes}")
    constructor, weights = _BACKBONES[architecture]
    model = constructor(weights=weights if pretrained else None)
    _replace_head(model, architecture, num_classes)
    return model


def _replace_head(model: nn.Module, architecture: str, num_classes: int) -> None:
    if architecture == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # type: ignore[attr-defined]
        return
    # MobileNet and EfficientNet heads are nn.Sequential; the last layer is the Linear.
    head: nn.Sequential = model.classifier  # type: ignore[assignment]
    head[-1] = nn.Linear(head[-1].in_features, num_classes)  # type: ignore[attr-defined]


def save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, Any]) -> None:
    """Write a checkpoint atomically (temp file, then rename)."""
    payload = {"format_version": CHECKPOINT_FORMAT_VERSION, **metadata, "state_dict": model.state_dict()}
    temp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temp)
    os.replace(temp, path)


def load_checkpoint(
    path: Path,
    *,
    classes: tuple[str, ...],
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a checkpoint for inference. Returns the model (in eval mode) and its metadata.

    Raises ``ModelError`` if the file is missing or malformed, or if its class
    order differs from ``classes``.
    """
    if not path.is_file():
        raise ModelError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # torch raises several exception types for corrupt files
        raise ModelError(f"cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelError(f"checkpoint {path} is not a training checkpoint")
    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in payload]
    if missing:
        raise ModelError(f"checkpoint {path} is missing key(s): {missing}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ModelError(
            f"checkpoint {path} has format version {payload['format_version']}, "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    if list(payload["classes"]) != list(classes):
        raise ModelError(
            f"checkpoint classes {payload['classes']} do not match the configured classes {list(classes)}"
        )
    model = build_model(payload["architecture"], len(classes), pretrained=False)
    try:
        model.load_state_dict(payload["state_dict"])
    except (RuntimeError, KeyError) as exc:
        raise ModelError(f"checkpoint {path} does not fit architecture {payload['architecture']}: {exc}") from exc
    model.to(device)
    model.eval()
    metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, metadata
