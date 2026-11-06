"""Evaluate a trained checkpoint on the held-out test split only.

Reads the manifest from ``src.prepare_dataset`` and keeps only the rows whose
split is ``test``. Each of those images is evaluated exactly once, in manifest
order, with the deterministic preprocessing used for validation. The model is
never trained, fine-tuned, or changed, and no split is generated here.

Run from the project root:

    python -m src.evaluate                                  # models\\best_model.pth
    python -m src.evaluate --checkpoint models\\best_model.pth
    python -m src.evaluate --checkpoint models\\best_model.pth --output-dir models\\eval_run2
    python -m src.evaluate --checkpoint models\\best_model.pth --overwrite

Outputs in the output directory (default: ``<paths.model_dir>\\evaluation``):

    evaluation_summary.json   metrics, settings, checkpoint identity, notes
    classification_report.csv per-class precision, recall, F1, support; macro and weighted averages
    confusion_matrix.csv      rows = actual class, columns = predicted class
    predictions.csv           one row per test image with actual, predicted and probabilities
    confusion_matrix.png      the same confusion matrix as a labelled plot

Exit codes: 0 success, 1 evaluation error (missing or incompatible checkpoint,
invalid test data, outputs exist without --overwrite), 2 bad config or arguments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, Config, ConfigError, ImageConfig, load_config
from src.dataset import ManifestRow, read_manifest
from src.model import ModelError, load_checkpoint
from src.train import ManifestImageDataset, TrainingError, check_manifest_metadata, resolve_device

SUMMARY_NAME = "evaluation_summary.json"
REPORT_NAME = "classification_report.csv"
CONFUSION_NAME = "confusion_matrix.csv"
PREDICTIONS_NAME = "predictions.csv"
PLOT_NAME = "confusion_matrix.png"
OUTPUT_NAMES: tuple[str, ...] = (SUMMARY_NAME, REPORT_NAME, CONFUSION_NAME, PREDICTIONS_NAME, PLOT_NAME)
EVALUATION_FORMAT_VERSION = 1
# Per-class estimates from fewer test images than this are flagged as unstable.
SMALL_SUPPORT_WARNING: int = 30


class EvaluationError(Exception):
    """The checkpoint, manifest, test images, or outputs cannot be used for an evaluation."""


# --------------------------------------------------------------------------
# Test records
# --------------------------------------------------------------------------


def portable_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    """Path relative to the project root when possible, otherwise just the file name."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def load_test_rows(config: Config, manifest_path: Path) -> tuple[list[ManifestRow], dict[str, int]]:
    """Return the ``test`` rows of the manifest and simple counts for the summary.

    Checks that the manifest matches the configured classes, that every test
    image exists on disk, and that no test group also appears in train or val.
    Nothing is skipped silently: any problem raises ``EvaluationError``.
    """
    if not manifest_path.is_file():
        raise EvaluationError(f"manifest not found: {manifest_path}. Run `python -m src.prepare_dataset` first.")
    try:
        rows = read_manifest(manifest_path, config.classes)
        check_manifest_metadata(manifest_path.with_name(manifest_path.stem + ".json"), config)
    except (ValueError, TrainingError) as exc:
        raise EvaluationError(str(exc)) from exc

    test_rows = [r for r in rows if r.split == "test"]
    if not test_rows:
        raise EvaluationError("the manifest has no 'test' images; there is nothing to evaluate")

    other_groups = {r.group_id for r in rows if r.split in ("train", "val")}
    shared = sorted({r.group_id for r in test_rows} & other_groups)
    if shared:
        raise EvaluationError(
            f"{len(shared)} test group(s) also appear in train or val (e.g. {shared[:3]}). "
            "The split is invalid; re-run src.prepare_dataset with --overwrite."
        )

    missing = [r.relative_path for r in test_rows if not (config.paths.dataset_dir / r.relative_path).is_file()]
    if missing:
        raise EvaluationError(
            f"{len(missing)} test image(s) listed in the manifest are missing or unreadable, "
            f"e.g. {missing[:3]}. The evaluation was not run, so no subset is reported as the full test set."
        )
    counts = {
        "manifest_rows": len(rows),
        "test_rows": len(test_rows),
        "train_rows": sum(r.split == "train" for r in rows),
        "val_rows": sum(r.split == "val" for r in rows),
    }
    return test_rows, counts


def check_checkpoint_matches_config(metadata: dict[str, Any], config: Config) -> None:
    """Refuse checkpoints trained with a different architecture, input size, or normalisation."""
    if metadata.get("architecture") != config.model.architecture:
        raise EvaluationError(
            f"checkpoint architecture {metadata.get('architecture')!r} differs from "
            f"model.architecture {config.model.architecture!r} in the config"
        )
    expected = (config.image.width, config.image.height)
    actual = (metadata.get("image_width"), metadata.get("image_height"))
    if actual != expected:
        raise EvaluationError(
            f"checkpoint was trained at {actual[0]}x{actual[1]} but the config image size is "
            f"{expected[0]}x{expected[1]}. Restore the training config or evaluate a matching checkpoint."
        )
    if metadata.get("normalization") != "imagenet":
        raise EvaluationError(f"unsupported normalization in checkpoint: {metadata.get('normalization')!r}")


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Inference:
    labels: np.ndarray  # ground-truth class indices, in manifest order
    predictions: np.ndarray  # predicted class indices
    probabilities: np.ndarray  # (N, classes) softmax of the logits, float64


def run_inference(
    model: torch.nn.Module,
    rows: list[ManifestRow],
    config: Config,
    image: ImageConfig,
    device: torch.device,
    batch_size: int,
) -> Inference:
    """Evaluate every row exactly once, in order, without gradients or augmentation."""
    dataset = ManifestImageDataset(
        rows,
        dataset_dir=config.paths.dataset_dir,
        classes=config.classes,
        image=image,
        augmentation=config.augmentation,
        training=False,  # deterministic preprocessing only
        seed=config.random_seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=config.training.num_workers)
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            logits_parts.append(logits.float().cpu().numpy().astype(np.float64))
            label_parts.append(labels.numpy())
    logits = np.concatenate(logits_parts)
    labels = np.concatenate(label_parts).astype(np.int64)

    expected = np.array([config.classes.index(r.material_class) for r in rows], dtype=np.int64)
    if len(labels) != len(rows) or not np.array_equal(labels, expected):
        raise EvaluationError("internal check failed: evaluated items do not match the manifest order")
    shifted = logits - logits.max(axis=1, keepdims=True)  # numerically stable softmax over classes
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    return Inference(labels=labels, predictions=logits.argmax(axis=1).astype(np.int64), probabilities=probabilities)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
# Convention: a precision or recall with zero denominator is reported as 0.0
# (sklearn zero_division=0). Macro averages are the unweighted mean over all four
# classes, so a class with no test images or no predictions lowers them. Weighted
# averages weight each class by its support, so classes with no test images
# contribute nothing.


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, classes: tuple[str, ...]) -> dict[str, Any]:
    indices = list(range(len(classes)))
    matrix = confusion_matrix(labels, predictions, labels=indices)  # rows = actual, columns = predicted
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=indices, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(labels, predictions, labels=indices, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(labels, predictions, labels=indices, average="weighted", zero_division=0)
    total = int(len(labels))
    correct = int(np.trace(matrix))
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "predicted_count": int(matrix[:, i].sum()),
        }
        for i, name in enumerate(classes)
    }
    return {
        "confusion_matrix": matrix,
        "per_class": per_class,
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
        "accuracy": float(accuracy_score(labels, predictions)) if total else 0.0,
        "test_samples": total,
        "correct": correct,
        "incorrect": total - correct,
    }


def build_notes(metrics: dict[str, Any], classes: tuple[str, ...]) -> list[str]:
    notes = [
        "Metrics use zero_division=0: an undefined precision or recall is reported as 0.0.",
        "Macro averages are unweighted means over all four classes; weighted averages use class support.",
        f"Test sample count is {metrics['test_samples']}. Results describe this test set only.",
    ]
    for name in classes:
        stats = metrics["per_class"][name]
        if stats["support"] == 0:
            notes.append(
                f"No test images of class '{name}': its precision, recall and F1 are not measured, and "
                "it lowers the macro averages. The model has not been evaluated on this class."
            )
        elif stats["predicted_count"] == 0:
            notes.append(f"The model never predicted class '{name}' on the test set (precision and F1 are 0).")
        if 0 < stats["support"] < SMALL_SUPPORT_WARNING:
            notes.append(
                f"Only {stats['support']} test image(s) of class '{name}' (fewer than {SMALL_SUPPORT_WARNING}); "
                "its per-class estimates are unstable."
            )
    return notes


# --------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------


def _backup(path: Path, stamp: str) -> Path:
    return path.with_name(f"{path.name}.{stamp}.bak")


def check_output_dir(output_dir: Path, overwrite: bool, stamp: str | None = None) -> None:
    """Refuse to replace existing results unless ``overwrite``; back up the old files when replacing."""
    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise EvaluationError(
            "refusing to overwrite existing evaluation results: "
            + ", ".join(p.name for p in existing)
            + f" in {output_dir}. Use another --output-dir, or add --overwrite "
            "(the old files are kept as .bak copies)."
        )
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in existing:
        path.replace(_backup(path, stamp))  # the folder is created only when results are written


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temp, path)


def write_classification_report(path: Path, metrics: dict[str, Any], classes: tuple[str, ...]) -> None:
    rows: list[list[Any]] = []
    for name in classes:
        s = metrics["per_class"][name]
        rows.append([name, f"{s['precision']:.6f}", f"{s['recall']:.6f}", f"{s['f1']:.6f}", s["support"]])
    total = metrics["test_samples"]
    for label, key in (("macro avg", "macro"), ("weighted avg", "weighted")):
        m = metrics[key]
        rows.append([label, f"{m['precision']:.6f}", f"{m['recall']:.6f}", f"{m['f1']:.6f}", total])
    rows.append(["accuracy", "", "", f"{metrics['accuracy']:.6f}", total])
    _write_csv(path, ["class", "precision", "recall", "f1", "support"], rows)


def write_confusion_csv(path: Path, matrix: np.ndarray, classes: tuple[str, ...]) -> None:
    header = ["actual \\ predicted", *classes]
    rows = [[name, *[int(v) for v in matrix[i]]] for i, name in enumerate(classes)]
    _write_csv(path, header, rows)


def write_predictions(
    path: Path,
    rows: list[ManifestRow],
    inference: Inference,
    classes: tuple[str, ...],
) -> None:
    header = ["relative_path", "actual_class", "predicted_class", "correct", *[f"prob_{c}" for c in classes]]
    body: list[list[Any]] = []
    for i, row in enumerate(rows):
        actual = int(inference.labels[i])
        predicted = int(inference.predictions[i])
        body.append(
            [
                row.relative_path,
                classes[actual],
                classes[predicted],
                "true" if actual == predicted else "false",
                *[f"{p:.6f}" for p in inference.probabilities[i]],
            ]
        )
    _write_csv(path, header, body)


def write_confusion_plot(path: Path, matrix: np.ndarray, classes: tuple[str, ...], accuracy: float) -> None:
    """Heat map of the actual confusion matrix, with counts printed in each cell."""
    figure = Figure(figsize=(6.4, 5.4), dpi=120)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="count")
    axis.set_xticks(range(len(classes)), labels=classes)
    axis.set_yticks(range(len(classes)), labels=classes)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(f"Confusion matrix (test set, accuracy {accuracy:.3f})")
    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            axis.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    figure.tight_layout()
    temp = path.with_name(path.name + ".tmp.png")
    figure.savefig(temp, format="png")
    os.replace(temp, path)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    summary: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_evaluation(
    config: Config,
    checkpoint: Path,
    manifest: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    stamp: str | None = None,
    log: Any = print,
) -> EvaluationResult:
    """Evaluate ``checkpoint`` on the test split and write the result files."""
    if not checkpoint.is_file():
        raise EvaluationError(f"checkpoint not found: {checkpoint}")
    check_output_dir(output_dir, overwrite, stamp)  # refuse early, before any slow work

    test_rows, counts = load_test_rows(config, manifest)
    device = resolve_device(config.training.device)
    try:
        model, metadata = load_checkpoint(checkpoint, classes=config.classes, device=device)
    except ModelError as exc:
        raise EvaluationError(str(exc)) from exc
    check_checkpoint_matches_config(metadata, config)
    image = ImageConfig(width=metadata["image_width"], height=metadata["image_height"])

    started = time.perf_counter()
    inference = run_inference(model, test_rows, config, image, device, config.training.batch_size)
    metrics = compute_metrics(inference.labels, inference.predictions, config.classes)
    if metrics["test_samples"] != counts["test_rows"]:
        raise EvaluationError("internal check failed: not every test record was evaluated exactly once")
    notes = build_notes(metrics, config.classes)
    for note in notes:
        log(f"note: {note}")

    summary: dict[str, Any] = {
        "evaluation_format_version": EVALUATION_FORMAT_VERSION,
        "checkpoint": {
            "file": portable_path(checkpoint),
            "sha256": sha256_file(checkpoint),
            "epoch": metadata.get("epoch"),
            "trained_val_loss": metadata.get("val_loss"),
            "pretrained": metadata.get("pretrained"),
            "seed": metadata.get("seed"),
        },
        "architecture": metadata["architecture"],
        "classes": list(config.classes),
        "class_order_source": "checkpoint (verified against the configuration)",
        "image": {"width": image.width, "height": image.height},
        "normalization": metadata["normalization"],
        "manifest": {"file": portable_path(manifest), "split": "test", **counts},
        "test_samples": metrics["test_samples"],
        "correct": metrics["correct"],
        "incorrect": metrics["incorrect"],
        "accuracy": metrics["accuracy"],
        "macro": metrics["macro"],
        "weighted": metrics["weighted"],
        "per_class": metrics["per_class"],
        "missing_classes_in_test": [c for c in config.classes if metrics["per_class"][c]["support"] == 0],
        "device": device.type,
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_seconds": round(time.perf_counter() - started, 2),
        "test_split_only": True,
        "model_modified": False,
        "notes": notes,
        "files": list(OUTPUT_NAMES),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_classification_report(output_dir / REPORT_NAME, metrics, config.classes)
    write_confusion_csv(output_dir / CONFUSION_NAME, metrics["confusion_matrix"], config.classes)
    write_predictions(output_dir / PREDICTIONS_NAME, test_rows, inference, config.classes)
    write_confusion_plot(output_dir / PLOT_NAME, metrics["confusion_matrix"], config.classes, metrics["accuracy"])
    temp = output_dir / (SUMMARY_NAME + ".tmp")
    temp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, output_dir / SUMMARY_NAME)
    return EvaluationResult(output_dir=output_dir, summary=summary)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test split only.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint to evaluate (default: best_model.pth in paths.model_dir)",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="manifest CSV (default: dataset.manifest)")
    parser.add_argument("--output-dir", type=Path, default=None, help="results folder (default: <model_dir>/evaluation)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="override training.device")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing results (old files are kept as .bak)",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    if args.device:
        config = replace(config, training=replace(config.training, device=args.device))

    checkpoint = args.checkpoint or (config.paths.model_dir / "best_model.pth")
    manifest = args.manifest or config.dataset.manifest
    output_dir = args.output_dir or (config.paths.model_dir / "evaluation")
    if args.checkpoint is None:
        print(f"no --checkpoint given; using {checkpoint}")

    try:
        result = run_evaluation(config, checkpoint, manifest, output_dir, overwrite=args.overwrite)
    except (EvaluationError, TrainingError) as exc:
        print(f"\nerror: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted: no complete evaluation was written")
        return 1

    s = result.summary
    print("\nTest split evaluation")
    print(f"  checkpoint     {s['checkpoint']['file']}")
    print(f"  test images    {s['test_samples']} (correct {s['correct']}, incorrect {s['incorrect']})")
    print(f"  accuracy       {s['accuracy']:.4f}")
    print(f"  macro F1       {s['macro']['f1']:.4f}   weighted F1 {s['weighted']['f1']:.4f}")
    print(f"  device         {s['device']}")
    print(f"\nResults written to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
