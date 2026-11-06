"""Train the four-class material classifier with transfer learning.

Data comes only from the manifest written by ``src.prepare_dataset``. This
script never splits the dataset itself. It reads the ``train`` and ``val``
rows. The ``test`` rows are not loaded, because final test evaluation is a
separate step.

Per epoch: train on the training split (with augmentation), evaluate on the
validation split (no augmentation), write a history row, and keep the
checkpoint with the lowest validation loss. Training stops early when the
validation loss has not improved for ``training.early_stopping_patience``
epochs.

Outputs in the output directory (default: ``paths.model_dir``):

    best_model.pth          checkpoint with the lowest validation loss
    last_model.pth          checkpoint after the last epoch that ran
    training_history.csv    one row per epoch
    training_summary.json   settings, counts, best epoch and metrics

Run from the project root:

    python -m src.train
    python -m src.train --epochs 5 --device cpu
    python -m src.train --no-pretrained          # random initial weights
    python -m src.train --output-dir models\\run2 --overwrite

Exit codes: 0 success, 1 data, model or training error (or existing outputs
without --overwrite), 2 bad config or arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import DEFAULT_CONFIG_PATH, AugmentationConfig, Config, ConfigError, ImageConfig, load_config
from src.dataset import ManifestRow, image_path_for, read_manifest
from src.model import ModelError, build_model, save_checkpoint
from src.preprocessing import load_rgb, preprocess_eval, preprocess_training

BEST_MODEL_NAME = "best_model.pth"
LAST_MODEL_NAME = "last_model.pth"
HISTORY_NAME = "training_history.csv"
SUMMARY_NAME = "training_summary.json"
HISTORY_COLUMNS: tuple[str, ...] = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "learning_rate",
    "seconds",
    "is_best",
)
SUMMARY_FORMAT_VERSION = 1


class TrainingError(Exception):
    """Training cannot start or cannot finish: bad data, missing files, or unusable settings."""


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_training_rows(config: Config) -> dict[str, list[ManifestRow]]:
    """Return the ``train`` and ``val`` manifest rows, after checking them.

    Checks: the manifest exists and was written for the configured classes;
    every class is present in both splits; no group (object or identical
    file) appears in both splits; every image file exists.
    """
    manifest = config.dataset.manifest
    if not manifest.is_file():
        raise TrainingError(f"manifest not found: {manifest}. Run `python -m src.prepare_dataset` first.")
    try:
        rows = read_manifest(manifest, config.classes)
    except ValueError as exc:
        raise TrainingError(str(exc)) from exc
    check_manifest_metadata(manifest.with_name(manifest.stem + ".json"), config)

    by_split = {split: [r for r in rows if r.split == split] for split in ("train", "val")}
    for split, split_rows in by_split.items():
        if not split_rows:
            raise TrainingError(f"the manifest has no '{split}' images")
        present = {r.material_class for r in split_rows}
        missing = [c for c in config.classes if c not in present]
        if missing:
            raise TrainingError(f"the '{split}' split has no images of class(es) {missing}")

    shared_groups = {r.group_id for r in by_split["train"]} & {r.group_id for r in by_split["val"]}
    if shared_groups:
        examples = sorted(shared_groups)[:3]
        raise TrainingError(
            f"{len(shared_groups)} group(s) appear in both train and val (e.g. {examples}); "
            "re-run src.prepare_dataset with --overwrite"
        )

    for split_rows in by_split.values():
        absent = [r.relative_path for r in split_rows if not image_path_for(config.paths.dataset_dir, r.relative_path).is_file()]
        if absent:
            raise TrainingError(f"{len(absent)} image(s) listed in the manifest are missing, e.g. {absent[:3]}")
    return by_split


def check_manifest_metadata(path: Path, config: Config) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"cannot read {path}: {exc}") from exc
    if data.get("classes") != list(config.classes):
        raise TrainingError(
            f"{path.name} was created for classes {data.get('classes')}, "
            f"but the configuration has {list(config.classes)}"
        )
    return data


class ManifestImageDataset(Dataset):
    """Images of one manifest split, preprocessed with the Task 3 pipeline.

    Training items get random augmentation seeded by (seed, epoch, index), so
    the result does not depend on worker order. Evaluation items are
    deterministic.
    """

    def __init__(
        self,
        rows: Sequence[ManifestRow],
        *,
        dataset_dir: Path,
        classes: tuple[str, ...],
        image: ImageConfig,
        augmentation: AugmentationConfig,
        training: bool,
        seed: int,
    ) -> None:
        self._items = [
            (image_path_for(dataset_dir, r.relative_path), classes.index(r.material_class)) for r in rows
        ]
        self._image = image
        self._augmentation = augmentation
        self._training = training
        self._seed = seed
        self.epoch = 0  # set by the training loop before each epoch

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self._items[index]
        try:
            rgb = load_rgb(path)
        except ValueError as exc:
            raise TrainingError(str(exc)) from exc
        if self._training:
            rng = np.random.default_rng([self._seed, self.epoch, index])
            array = preprocess_training(rgb, self._image, self._augmentation, rng)
        else:
            array = preprocess_eval(rgb, self._image)
        return torch.from_numpy(array), label


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------


@dataclass
class EarlyStopper:
    """Stop when the monitored loss has not improved for ``patience`` epochs."""

    patience: int
    best: float = math.inf
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, value: float, epoch: int) -> bool:
        """Record one epoch's value. Returns True when it is a new best."""
        if value < self.best:  # NaN never counts as an improvement
            self.best = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise TrainingError(
            "CUDA was requested but is not available on this machine. "
            "Use --device cpu, or set training.device to auto."
        )
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over ``loader``. Trains when ``optimizer`` is given. Returns (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0
    with torch.enable_grad() if training else torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            count = labels.size(0)
            total_loss += loss.item() * count
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += count
    if seen == 0:
        raise TrainingError("a data split produced no batches")
    return total_loss / seen, correct / seen


def train_one_epoch(model, loader, criterion, device, optimizer) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    return run_epoch(model, loader, criterion, device, optimizer)


def evaluate(model, loader, criterion, device) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    return run_epoch(model, loader, criterion, device, None)


def write_history(path: Path, history: Sequence[dict[str, Any]]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    os.replace(temp, path)


def _backup_path(path: Path, stamp: str) -> Path:
    return path.with_name(f"{path.name}.{stamp}.bak")


def check_output_dir(output_dir: Path, overwrite: bool, stamp: str | None = None) -> list[Path]:
    """Refuse to replace existing outputs unless ``overwrite``; back them up when replacing."""
    names = (BEST_MODEL_NAME, LAST_MODEL_NAME, HISTORY_NAME, SUMMARY_NAME)
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise TrainingError(
            "refusing to overwrite existing training outputs: "
            + ", ".join(p.name for p in existing)
            + f" in {output_dir}. Use another --output-dir, or add --overwrite "
            "(the old files are kept as .bak copies)."
        )
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in existing:
        path.replace(_backup_path(path, stamp))
    output_dir.mkdir(parents=True, exist_ok=True)
    return existing


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    device: str
    epochs_run: int
    stopped_early: bool
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    summary: dict[str, Any]


def run_training(
    config: Config,
    output_dir: Path,
    *,
    overwrite: bool = False,
    stamp: str | None = None,
    log: Callable[[str], None] = print,
) -> TrainingResult:
    """Train, save the outputs, and return a summary of the run."""
    hp = config.training
    device = resolve_device(hp.device)
    check_output_dir(output_dir, overwrite, stamp)  # fail before any slow work
    rows = load_training_rows(config)
    manifest_meta = check_manifest_metadata(config.dataset.manifest.with_name(config.dataset.manifest.stem + ".json"), config)

    set_seed(config.random_seed)
    datasets = {
        "train": ManifestImageDataset(
            rows["train"],
            dataset_dir=config.paths.dataset_dir,
            classes=config.classes,
            image=config.image,
            augmentation=config.augmentation,
            training=True,
            seed=config.random_seed,
        ),
        "val": ManifestImageDataset(
            rows["val"],
            dataset_dir=config.paths.dataset_dir,
            classes=config.classes,
            image=config.image,
            augmentation=config.augmentation,
            training=False,
            seed=config.random_seed,
        ),
    }
    pin = device.type == "cuda"
    # BatchNorm cannot train on a batch of one, so skip a final batch of size 1.
    train_size = len(datasets["train"])
    drop_last = train_size > hp.batch_size and train_size % hp.batch_size == 1
    shuffle_rng = torch.Generator().manual_seed(config.random_seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=hp.batch_size,
        shuffle=True,
        generator=shuffle_rng,
        num_workers=hp.num_workers,
        pin_memory=pin,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=hp.batch_size,
        shuffle=False,
        num_workers=hp.num_workers,
        pin_memory=pin,
    )

    try:
        model = build_model(config.model.architecture, len(config.classes), pretrained=hp.pretrained).to(device)
    except ModelError:
        raise
    except Exception as exc:  # download failures raise several exception types from torch/urllib
        if not hp.pretrained:
            raise
        raise TrainingError(
            f"could not download the ImageNet weights for {config.model.architecture} ({exc}). "
            "Check the internet connection and try again, or run with --no-pretrained."
        ) from exc
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopper(patience=hp.early_stopping_patience)

    best_path = output_dir / BEST_MODEL_NAME
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    stopped_early = False
    started = time.perf_counter()

    def checkpoint_metadata(epoch: int, val_loss: float, val_accuracy: float) -> dict[str, Any]:
        return {
            "architecture": config.model.architecture,
            "classes": list(config.classes),
            "image_width": config.image.width,
            "image_height": config.image.height,
            "normalization": "imagenet",
            "pretrained": hp.pretrained,
            "seed": config.random_seed,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

    log(
        f"training on {device} | train {len(datasets['train'])} images, "
        f"val {len(datasets['val'])} images | {config.model.architecture}, "
        f"pretrained={hp.pretrained}"
    )
    epoch = 0
    for epoch in range(1, hp.epochs + 1):
        datasets["train"].epoch = epoch
        epoch_started = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise TrainingError(
                f"loss became NaN or infinite at epoch {epoch}; try a smaller training.learning_rate"
            )
        improved = stopper.update(val_loss, epoch)
        seconds = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_acc, 6),
                "learning_rate": hp.learning_rate,
                "seconds": round(seconds, 2),
                "is_best": improved,
            }
        )
        write_history(output_dir / HISTORY_NAME, history)
        if improved:
            save_checkpoint(best_path, model, checkpoint_metadata(epoch, val_loss, val_acc))
            best = history[-1]
        log(
            f"epoch {epoch:>3}/{hp.epochs}  train loss {train_loss:.4f} acc {train_acc:.3f}  |  "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}  {'(best)' if improved else ''}  {seconds:.1f}s"
        )
        if stopper.should_stop:
            stopped_early = True
            log(f"early stopping: no validation-loss improvement for {hp.early_stopping_patience} epoch(s)")
            break

    if best is None:  # cannot happen with finite losses, but never write a missing best model silently
        raise TrainingError("no epoch produced a validation loss, so no best model was saved")

    last_row = history[-1]
    save_checkpoint(
        output_dir / LAST_MODEL_NAME,
        model,
        checkpoint_metadata(epoch, last_row["val_loss"], last_row["val_accuracy"]),
    )
    counts = {
        split: {c: sum(1 for r in rows[split] if r.material_class == c) for c in config.classes}
        for split in ("train", "val")
    }
    summary: dict[str, Any] = {
        "summary_format_version": SUMMARY_FORMAT_VERSION,
        "architecture": config.model.architecture,
        "classes": list(config.classes),
        "pretrained": hp.pretrained,
        "device": device.type,
        "torch_version": torch.__version__,
        "seed": config.random_seed,
        "image": {"width": config.image.width, "height": config.image.height},
        "augmentation": {
            "max_rotation_degrees": config.augmentation.max_rotation_degrees,
            "max_translation": config.augmentation.max_translation,
            "min_scale": config.augmentation.min_scale,
            "max_scale": config.augmentation.max_scale,
            "brightness": config.augmentation.brightness,
            "contrast": config.augmentation.contrast,
        },
        "hyperparameters": {
            "epochs_requested": hp.epochs,
            "batch_size": hp.batch_size,
            "learning_rate": hp.learning_rate,
            "weight_decay": hp.weight_decay,
            "early_stopping_patience": hp.early_stopping_patience,
            "num_workers": hp.num_workers,
        },
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_accuracy": best["val_accuracy"],
        "last_val_loss": last_row["val_loss"],
        "last_val_accuracy": last_row["val_accuracy"],
        "manifest_counts": counts,
        "manifest_leakage_protection": manifest_meta.get("leakage_protection"),
        "test_split_used": False,
        "files": [BEST_MODEL_NAME, LAST_MODEL_NAME, HISTORY_NAME, SUMMARY_NAME],
        "duration_seconds": round(time.perf_counter() - started, 2),
    }
    temp = output_dir / (SUMMARY_NAME + ".tmp")
    temp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, output_dir / SUMMARY_NAME)
    log(f"best epoch {best['epoch']}: val loss {best['val_loss']:.4f}, val accuracy {best['val_accuracy']:.3f}")
    return TrainingResult(
        output_dir=output_dir,
        device=device.type,
        epochs_run=len(history),
        stopped_early=stopped_early,
        best_epoch=best["epoch"],
        best_val_loss=best["val_loss"],
        best_val_accuracy=best["val_accuracy"],
        summary=summary,
    )


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the material classifier (transfer learning).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--output-dir", type=Path, default=None, help="where to save outputs (default: paths.model_dir)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs (old files kept as .bak)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="override training.device")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--no-pretrained", action="store_true", help="train from random weights")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.epochs is not None and args.epochs < 1:
            raise ConfigError("--epochs must be at least 1")
        training = config.training
        training = replace(
            training,
            epochs=args.epochs if args.epochs is not None else training.epochs,
            device=args.device or training.device,
            pretrained=False if args.no_pretrained else training.pretrained,
        )
        config = replace(config, training=training)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2

    output_dir = args.output_dir or config.paths.model_dir
    try:
        result = run_training(config, output_dir, overwrite=args.overwrite)
    except (TrainingError, ModelError) as exc:
        print(f"\nerror: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted: files written so far are kept in the output directory")
        return 1
    print(f"\nOutputs written to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
