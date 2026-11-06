"""Train on every image in the dataset, with no validation or test split.

Use this when the data cannot be split by object, for example when each
material is one object (see src.synthesize_dataset). It uses the same model,
preprocessing, augmentation and loss as src.train, but:

* every image in ``paths.dataset_dir/<class>/`` is used for training;
* there is no validation set, so no early stopping and no best-epoch choice:
  the model after the last epoch is saved;
* no accuracy is measured on held-out images. The numbers printed are on
  training images (with augmentation), so they do not show how the model
  handles new photos.

Run from the project root:

    python -m src.train_all
    python -m src.train_all --epochs 15 --overwrite
    python -m src.train_all --no-pretrained

Outputs in ``paths.model_dir`` (default ``models``):

    best_model.pth            the checkpoint predict, webcam and live sorter load
    training_history.csv      one row per epoch
    training_summary.json     settings and the notes above

Exit codes: 0 success, 1 training error, 2 bad config or arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from src.dataset import ManifestRow, list_image_files
from src.model import ModelError, build_model, save_checkpoint
from src.train import (
    BEST_MODEL_NAME,
    HISTORY_NAME,
    SUMMARY_NAME,
    ManifestImageDataset,
    TrainingError,
    check_output_dir,
    resolve_device,
    set_seed,
    train_one_epoch,
    write_history,
)

SUMMARY_FORMAT_VERSION = 1
NOTES = (
    "Trained on every image in the dataset. There was no validation or test split.",
    "The saved model is the one after the last epoch. There was no early stopping.",
    "The accuracy shown during training is on training images, so it does not show performance on new photos.",
)


@dataclass
class TrainAllResult:
    images: int
    epochs: int
    final_train_loss: float
    final_train_accuracy: float
    per_class: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def collect_rows(config: Config) -> list[ManifestRow]:
    """One row per image in ``dataset_dir/<class>/``. Every class must have at least one image."""
    dataset_dir = config.paths.dataset_dir
    if not dataset_dir.is_dir():
        raise TrainingError(f"dataset folder not found: {dataset_dir}. Run `python -m src.synthesize_dataset` first.")
    rows: list[ManifestRow] = []
    for name in config.classes:
        class_dir = dataset_dir / name
        files = list_image_files(class_dir)
        if not files:
            raise TrainingError(f"no images in {class_dir}. Each class folder needs images before training.")
        for path in files:
            rows.append(
                ManifestRow(
                    relative_path=f"{name}/{path.name}",
                    material_class=name,
                    split="train",
                    group_id="",
                    object_id="",
                    session_id="",
                )
            )
    return rows


def train_all(
    config: Config,
    output_dir: Path,
    *,
    epochs: int,
    overwrite: bool = False,
    stamp: str | None = None,
    device_setting: str | None = None,
    pretrained: bool | None = None,
    log=print,
) -> TrainAllResult:
    """Train on all images and save the final model. Raises ``TrainingError`` or ``ModelError``."""
    if epochs < 1:
        raise TrainingError(f"epochs must be at least 1, got {epochs}")
    hp = config.training
    use_pretrained = hp.pretrained if pretrained is None else pretrained
    device = resolve_device(device_setting or hp.device)

    rows = collect_rows(config)
    check_output_dir(output_dir, overwrite, stamp)  # before any slow work; backs up existing outputs

    set_seed(config.random_seed)
    dataset = ManifestImageDataset(
        rows,
        dataset_dir=config.paths.dataset_dir,
        classes=config.classes,
        image=config.image,
        augmentation=config.augmentation,
        training=True,
        seed=config.random_seed,
    )
    # BatchNorm cannot train on a batch of one, so skip a final batch of size 1.
    drop_last = len(dataset) > hp.batch_size and len(dataset) % hp.batch_size == 1
    loader = DataLoader(
        dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.random_seed),
        num_workers=hp.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
    )

    try:
        model = build_model(config.model.architecture, len(config.classes), pretrained=use_pretrained).to(device)
    except ModelError:
        raise
    except Exception as exc:  # download failures raise several exception types
        if not use_pretrained:
            raise
        raise TrainingError(
            f"could not download the ImageNet weights for {config.model.architecture} ({exc}). "
            "Check the internet connection, or run with --no-pretrained."
        ) from exc

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    criterion = nn.CrossEntropyLoss()
    per_class = {name: sum(1 for r in rows if r.material_class == name) for name in config.classes}
    log(
        f"training on {device} | {len(rows)} images ({', '.join(f'{k}={v}' for k, v in per_class.items())}) | "
        f"{config.model.architecture}, pretrained={use_pretrained}, {epochs} epochs, no validation"
    )

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    train_loss = train_acc = float("nan")
    for epoch in range(1, epochs + 1):
        dataset.epoch = epoch
        epoch_started = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, loader, criterion, device, optimizer)
        if not math.isfinite(train_loss):
            raise TrainingError(f"loss became NaN or infinite at epoch {epoch}; try a smaller training.learning_rate")
        seconds = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": None,
                "val_accuracy": None,
                "learning_rate": hp.learning_rate,
                "seconds": round(seconds, 2),
                "is_best": None,
            }
        )
        log(f"epoch {epoch}/{epochs}  train loss {train_loss:.4f}  train acc {train_acc:.3f}  {seconds:.1f}s")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_dir / BEST_MODEL_NAME,
        model,
        {
            "architecture": config.model.architecture,
            "classes": list(config.classes),
            "image_width": config.image.width,
            "image_height": config.image.height,
            "normalization": "imagenet",
            "pretrained": use_pretrained,
            "seed": config.random_seed,
            "epoch": epochs,
            "val_loss": None,
            "val_accuracy": None,
            "trained_on": "all images, no validation split",
            "final_train_loss": train_loss,
            "final_train_accuracy": train_acc,
        },
    )
    write_history(output_dir / HISTORY_NAME, history)
    result = TrainAllResult(
        images=len(rows),
        epochs=epochs,
        final_train_loss=train_loss,
        final_train_accuracy=train_acc,
        per_class=per_class,
        seconds=time.perf_counter() - started,
    )
    summary = {
        "format_version": SUMMARY_FORMAT_VERSION,
        "mode": "train_all",
        "architecture": config.model.architecture,
        "classes": list(config.classes),
        "images_per_class": per_class,
        "total_images": len(rows),
        "epochs": epochs,
        "batch_size": hp.batch_size,
        "learning_rate": hp.learning_rate,
        "pretrained": use_pretrained,
        "device": device.type,
        "random_seed": config.random_seed,
        "final_train_loss": train_loss,
        "final_train_accuracy": train_acc,
        "seconds": round(result.seconds, 1),
        "files": [BEST_MODEL_NAME, HISTORY_NAME, SUMMARY_NAME],
        "notes": list(NOTES),
    }
    temp = output_dir / (SUMMARY_NAME + ".tmp")
    temp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, output_dir / SUMMARY_NAME)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train on all dataset images (no validation split).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--output-dir", type=Path, default=None, help="where to save outputs (default: paths.model_dir)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs (old files kept as .bak)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="override training.device")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--no-pretrained", action="store_true", help="train from random weights")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    epochs = config.training.epochs if args.epochs is None else args.epochs
    if epochs < 1:
        print(f"error: --epochs must be at least 1, got {epochs}")
        return 2
    output_dir = args.output_dir or config.paths.model_dir
    try:
        result = train_all(
            config,
            output_dir,
            epochs=epochs,
            overwrite=args.overwrite,
            device_setting=args.device,
            pretrained=False if args.no_pretrained else None,
        )
    except (TrainingError, ModelError) as exc:
        print(f"error: {exc}")
        return 1
    except OSError as exc:
        print(f"file system error: {exc}")
        return 1

    print(f"Saved {output_dir / BEST_MODEL_NAME} after {result.epochs} epochs on {result.images} images.")
    print(f"Final training accuracy {result.final_train_accuracy:.3f} (on training images; not a test score).")
    print("No validation or test split was used. See training_summary.json.")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    raise SystemExit(main())
