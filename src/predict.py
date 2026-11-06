"""Predict the material of one image with a trained checkpoint (inference only).

The image gets the same deterministic preprocessing as validation and
evaluation (resize, centre crop, ImageNet normalisation). The four output
logits are turned into probabilities with softmax, and the class with the
highest probability is reported. Nothing is trained, fine-tuned, or written
to the checkpoint.

Run from the project root:

    python -m src.predict --image path\\to\\object.jpg
    python -m src.predict --image path\\to\\object.jpg --checkpoint models\\best_model.pth
    python -m src.predict --image path\\to\\object.jpg --device cpu --json
    python -m src.predict --image path\\to\\object.jpg --threshold 0.8

Exit codes: 0 success, 1 prediction error (missing or unreadable image, missing
or incompatible checkpoint), 2 bad config or arguments.

The probabilities are model outputs (softmax of the logits). They are not
calibrated, so they are not guaranteed real-world probabilities. A prediction
is not proof that the material is correct.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.config import DEFAULT_CONFIG_PATH, Config, ConfigError, ImageConfig, load_config
from src.evaluate import EvaluationError, check_checkpoint_matches_config, portable_path, sha256_file
from src.model import ModelError, load_checkpoint
from src.preprocessing import load_rgb, preprocess_eval
from src.train import TrainingError, resolve_device

PREDICTION_FORMAT_VERSION = 1
PROBABILITY_NOTE = (
    "Probabilities are softmax outputs of the model. They are not calibrated and are not "
    "guaranteed real-world probabilities."
)


class PredictionError(Exception):
    """The image or checkpoint cannot be used for a prediction."""


@dataclass(frozen=True)
class Prediction:
    image_path: str  # the path as the user gave it
    predicted_class: str
    max_probability: float
    probabilities: dict[str, float]  # class name -> probability, in checkpoint class order
    architecture: str
    checkpoint: str  # portable identifier: file name or project-relative path
    checkpoint_sha256: str
    device: str
    threshold: float | None = None
    low_confidence: bool | None = None  # only set when a threshold was given

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "format_version": PREDICTION_FORMAT_VERSION,
            "image": self.image_path,
            "predicted_class": self.predicted_class,
            "max_probability": self.max_probability,
            "probabilities": self.probabilities,
            "architecture": self.architecture,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
            "probability_note": PROBABILITY_NOTE,
        }
        if self.threshold is not None:
            data["threshold"] = self.threshold
            data["low_confidence"] = self.low_confidence
        return data


# --------------------------------------------------------------------------
# Core inference
# --------------------------------------------------------------------------


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis, computed in float64 with the max subtracted for stability."""
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def predict_probabilities(model: nn.Module, rgb: np.ndarray, image: ImageConfig, device: torch.device) -> np.ndarray:
    """Return the softmax probabilities (one per class, in output order) for one RGB image."""
    tensor = torch.from_numpy(preprocess_eval(rgb, image)).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
    values = logits.detach().cpu().numpy()
    if values.ndim != 2 or values.shape[0] != 1:
        raise PredictionError(f"model returned an unexpected output shape {values.shape}")
    return softmax(values[0])


def check_image_file(path: Path) -> None:
    if not path.exists():
        raise PredictionError(f"image not found: {path}")
    if not path.is_file():
        raise PredictionError(f"image path is not a file: {path}")


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module  # in eval mode
    image: ImageConfig  # input size recorded in the checkpoint
    device: torch.device
    metadata: dict[str, Any]  # checkpoint metadata without the state dict


def load_model_for_inference(
    config: Config,
    checkpoint: Path,
    *,
    device_setting: str | None = None,
) -> LoadedModel:
    """Resolve the device and load a compatible checkpoint for inference.

    Shared by single-image prediction and the live webcam preview. Raises
    ``PredictionError`` for any problem and never substitutes another checkpoint.
    """
    try:
        device = resolve_device(device_setting or config.training.device)
    except TrainingError as exc:
        raise PredictionError(str(exc)) from exc

    if not checkpoint.is_file():
        raise PredictionError(f"checkpoint not found: {checkpoint}. No other checkpoint is used in its place.")
    try:
        model, metadata = load_checkpoint(checkpoint, classes=config.classes, device=device)
        check_checkpoint_matches_config(metadata, config)
    except (ModelError, EvaluationError) as exc:
        raise PredictionError(str(exc)) from exc
    image = ImageConfig(width=metadata["image_width"], height=metadata["image_height"])
    return LoadedModel(model=model, image=image, device=device, metadata=metadata)


def predict_image(
    config: Config,
    checkpoint: Path,
    image_path: Path,
    *,
    device_setting: str | None = None,
    threshold: float | None = None,
) -> Prediction:
    """Predict the material of one image. Raises ``PredictionError`` for any problem."""
    if threshold is not None and not 0.0 < threshold < 1.0:
        raise PredictionError(f"threshold must be strictly between 0 and 1, got {threshold}")
    check_image_file(image_path)
    loaded = load_model_for_inference(config, checkpoint, device_setting=device_setting)

    try:
        rgb = load_rgb(image_path)
    except ValueError as exc:
        raise PredictionError(
            f"{image_path.name} is not a readable image file ({exc}). "
            "Supported formats include JPG, PNG and BMP."
        ) from exc

    probs = predict_probabilities(loaded.model, rgb, loaded.image, loaded.device)
    if probs.shape != (len(config.classes),):
        raise PredictionError(f"expected {len(config.classes)} probabilities, got {probs.shape}")
    best = int(np.argmax(probs))
    max_probability = float(probs[best])
    low = None if threshold is None else bool(max_probability < threshold)
    return Prediction(
        image_path=str(image_path),
        predicted_class=config.classes[best],
        max_probability=max_probability,
        probabilities={name: float(p) for name, p in zip(config.classes, probs)},
        architecture=str(loaded.metadata["architecture"]),
        checkpoint=portable_path(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        device=loaded.device.type,
        threshold=threshold,
        low_confidence=low,
    )


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def format_text(prediction: Prediction) -> str:
    name_width = max(len(c) for c in prediction.probabilities)
    lines = [
        f"Image:            {Path(prediction.image_path).name}",
        f"Predicted:        {prediction.predicted_class}",
        f"Max probability:  {prediction.max_probability:.6f}",
        "Probabilities:",
    ]
    for name, value in prediction.probabilities.items():
        marker = "  <- predicted" if name == prediction.predicted_class else ""
        lines.append(f"  {name.ljust(name_width)}  {value:.6f}{marker}")
    if prediction.threshold is not None:
        status = "yes" if prediction.low_confidence else "no"
        lines.append(f"Low confidence:   {status} (threshold {prediction.threshold:g})")
    lines += [
        f"Model:            {prediction.architecture}",
        f"Checkpoint:       {prediction.checkpoint}",
        f"Device:           {prediction.device}",
        "",
        PROBABILITY_NOTE,
    ]
    return "\n".join(lines)


def error_json(message: str) -> str:
    return json.dumps({"format_version": PREDICTION_FORMAT_VERSION, "error": message}, indent=2)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict the material of one image (inference only).")
    parser.add_argument("--image", type=Path, required=True, help="image file to classify")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint to use (default: best_model.pth in paths.model_dir)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="override training.device")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="optional: flag low_confidence when max_probability is below this value (0 to 1)",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    def fail(message: str, code: int) -> int:
        print(error_json(message) if args.json else f"error: {message}")
        return code

    if args.threshold is not None and not 0.0 < args.threshold < 1.0:
        return fail(f"--threshold must be strictly between 0 and 1, got {args.threshold}", 2)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        return fail(str(exc), 2)

    checkpoint = args.checkpoint or (config.paths.model_dir / "best_model.pth")
    if args.checkpoint is None and not args.json:
        print(f"no --checkpoint given; using {checkpoint}")
    try:
        prediction = predict_image(
            config,
            checkpoint,
            args.image,
            device_setting=args.device,
            threshold=args.threshold,
        )
    except PredictionError as exc:
        return fail(str(exc), 1)
    print(json.dumps(prediction.to_dict(), indent=2) if args.json else format_text(prediction))
    return 0


if __name__ == "__main__":
    # Keep printing safe when a file name has non-ASCII characters and output is redirected.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    raise SystemExit(main())
