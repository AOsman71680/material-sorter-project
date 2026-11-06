"""Load and validate the material sorter configuration.

Nothing here falls back to a default: a missing, unknown, or invalid setting
raises ``ConfigError`` with a message naming the offending key.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MATERIAL_CLASSES: tuple[str, ...] = ("metal", "plastic", "paper", "other")
SUPPORTED_ARCHITECTURES: tuple[str, ...] = (
    "mobilenet_v3_small",
    "mobilenet_v2",
    "efficientnet_b0",
    "resnet18",
)
SUPPORTED_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"

MIN_IMAGE_SIDE = 32
MAX_IMAGE_SIDE = 2048
SPLIT_SUM_TOLERANCE = 1e-6


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class PathsConfig:
    dataset_dir: Path
    model_dir: Path


@dataclass(frozen=True)
class ImageConfig:
    width: int
    height: int


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    num_workers: int
    device: str
    pretrained: bool


@dataclass(frozen=True)
class ModelConfig:
    architecture: str


@dataclass(frozen=True)
class DatasetConfig:
    group_metadata: Path
    manifest: Path
    report: Path


@dataclass(frozen=True)
class AugmentationConfig:
    max_rotation_degrees: float
    max_translation: float
    min_scale: float
    max_scale: float
    brightness: float
    contrast: float


@dataclass(frozen=True)
class Config:
    classes: tuple[str, ...]
    cups: dict[str, int]
    paths: PathsConfig
    dataset: DatasetConfig
    image: ImageConfig
    augmentation: AugmentationConfig
    split: SplitConfig
    random_seed: int
    training: TrainingConfig
    model: ModelConfig
    confidence_threshold: float
    camera_index: int

    def cup_for(self, material: str) -> int:
        """Return the cup ID for a material class."""
        try:
            return self.cups[material]
        except KeyError:
            raise ConfigError(
                f"unknown material class {material!r}; expected one of {list(self.classes)}"
            ) from None


# --------------------------------------------------------------------------
# YAML loading
# --------------------------------------------------------------------------


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if isinstance(key, Hashable):
                if key in seen:
                    line = key_node.start_mark.line + 1
                    raise ConfigError(f"duplicate key {key!r} in YAML (line {line})")
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _read_yaml(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8-sig")  # tolerate a Notepad BOM
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 (SafeLoader subclass)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc


# --------------------------------------------------------------------------
# Field validators
# --------------------------------------------------------------------------


def _describe(value: Any) -> str:
    return f"{value!r} ({type(value).__name__})"


def _mapping(value: Any, name: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a mapping, got {_describe(value)}")
    return value


def _check_keys(mapping: Mapping[Any, Any], expected: tuple[str, ...], name: str) -> None:
    present = {str(k) for k in mapping}
    missing = sorted(set(expected) - present)
    unknown = sorted(present - set(expected))
    if missing:
        raise ConfigError(f"'{name}' is missing required key(s): {missing}")
    if unknown:
        raise ConfigError(f"'{name}' has unknown key(s): {unknown}; expected {list(expected)}")


def _int(value: Any, name: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{name}' must be an integer, got {_describe(value)}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"'{name}' must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"'{name}' must be <= {maximum}, got {value}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be a number, got {_describe(value)}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"'{name}' must be a finite number, got {value}")
    return number


def _open_unit_interval(value: Any, name: str) -> float:
    number = _number(value, name)
    if not 0.0 < number < 1.0:
        raise ConfigError(f"'{name}' must be strictly between 0 and 1, got {number}")
    return number


def _classes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(c, str) for c in value):
        raise ConfigError(f"'classes' must be a list of strings, got {_describe(value)}")
    duplicates = sorted({c for c in value if value.count(c) > 1})
    if duplicates:
        raise ConfigError(f"'classes' contains duplicate entries: {duplicates}")
    missing = [c for c in MATERIAL_CLASSES if c not in value]
    unexpected = [c for c in value if c not in MATERIAL_CLASSES]
    if missing or unexpected:
        raise ConfigError(
            f"'classes' must be exactly {list(MATERIAL_CLASSES)}; "
            f"missing: {missing}, unexpected: {unexpected}"
        )
    return tuple(value)


def _cups(value: Any, classes: tuple[str, ...]) -> dict[str, int]:
    mapping = _mapping(value, "cups")
    _check_keys(mapping, classes, "cups")
    cups = {c: _int(mapping[c], f"cups.{c}", minimum=1) for c in classes}
    owner: dict[int, str] = {}
    for material, cup in cups.items():
        if cup in owner:
            raise ConfigError(
                f"'cups.{owner[cup]}' and 'cups.{material}' both map to cup {cup}; "
                "each class needs a unique cup ID"
            )
        owner[cup] = material
    return cups


def _path(value: Any, name: str, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{name}' must be a non-empty path string, got {_describe(value)}")
    path = Path(value.strip())
    return path if path.is_absolute() else base_dir / path


def _paths(value: Any, base_dir: Path) -> PathsConfig:
    mapping = _mapping(value, "paths")
    _check_keys(mapping, ("dataset_dir", "model_dir"), "paths")
    dataset_dir = _path(mapping["dataset_dir"], "paths.dataset_dir", base_dir)
    model_dir = _path(mapping["model_dir"], "paths.model_dir", base_dir)
    if dataset_dir == model_dir or dataset_dir in model_dir.parents or model_dir in dataset_dir.parents:
        raise ConfigError(
            "'paths.dataset_dir' and 'paths.model_dir' must be separate, "
            f"non-nested folders, got {dataset_dir} and {model_dir}"
        )
    return PathsConfig(dataset_dir=dataset_dir, model_dir=model_dir)


def _image(value: Any) -> ImageConfig:
    mapping = _mapping(value, "image")
    _check_keys(mapping, ("width", "height"), "image")
    return ImageConfig(
        width=_int(mapping["width"], "image.width", MIN_IMAGE_SIDE, MAX_IMAGE_SIDE),
        height=_int(mapping["height"], "image.height", MIN_IMAGE_SIDE, MAX_IMAGE_SIDE),
    )


def _split(value: Any) -> SplitConfig:
    mapping = _mapping(value, "split")
    _check_keys(mapping, ("train", "val", "test"), "split")
    train = _open_unit_interval(mapping["train"], "split.train")
    val = _open_unit_interval(mapping["val"], "split.val")
    test = _open_unit_interval(mapping["test"], "split.test")
    total = train + val + test
    if not math.isclose(total, 1.0, abs_tol=SPLIT_SUM_TOLERANCE):
        raise ConfigError(f"'split' ratios must sum to 1, got {train} + {val} + {test} = {total}")
    return SplitConfig(train=train, val=val, test=test)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{name}' must be true or false, got {_describe(value)}")
    return value


def _training(value: Any) -> TrainingConfig:
    mapping = _mapping(value, "training")
    _check_keys(
        mapping,
        (
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "early_stopping_patience",
            "num_workers",
            "device",
            "pretrained",
        ),
        "training",
    )
    learning_rate = _number(mapping["learning_rate"], "training.learning_rate")
    if not 0.0 < learning_rate <= 1.0:
        raise ConfigError(
            f"'training.learning_rate' must be in (0, 1], got {learning_rate}"
        )
    weight_decay = _number(mapping["weight_decay"], "training.weight_decay")
    if not 0.0 <= weight_decay < 1.0:
        raise ConfigError(f"'training.weight_decay' must be in [0, 1), got {weight_decay}")
    device = mapping["device"]
    if device not in SUPPORTED_DEVICES:
        raise ConfigError(
            f"'training.device' must be one of {list(SUPPORTED_DEVICES)}, got {_describe(device)}"
        )
    return TrainingConfig(
        batch_size=_int(mapping["batch_size"], "training.batch_size", minimum=1),
        epochs=_int(mapping["epochs"], "training.epochs", minimum=1),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=_int(
            mapping["early_stopping_patience"], "training.early_stopping_patience", minimum=1
        ),
        num_workers=_int(mapping["num_workers"], "training.num_workers", minimum=0, maximum=16),
        device=device,
        pretrained=_bool(mapping["pretrained"], "training.pretrained"),
    )


def _model(value: Any) -> ModelConfig:
    mapping = _mapping(value, "model")
    _check_keys(mapping, ("architecture",), "model")
    architecture = mapping["architecture"]
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ConfigError(
            f"'model.architecture' must be one of {list(SUPPORTED_ARCHITECTURES)}, "
            f"got {_describe(architecture)}"
        )
    return ModelConfig(architecture=architecture)


def _dataset(value: Any, dataset_dir: Path, base_dir: Path) -> DatasetConfig:
    mapping = _mapping(value, "dataset")
    _check_keys(mapping, ("group_metadata", "manifest", "report"), "dataset")
    files = {
        key: _path(mapping[key], f"dataset.{key}", base_dir)
        for key in ("group_metadata", "manifest", "report")
    }
    for key, path in files.items():
        if path == dataset_dir or dataset_dir in path.parents:
            raise ConfigError(
                f"'dataset.{key}' must not be inside 'paths.dataset_dir' "
                f"({dataset_dir}), got {path}"
            )
    if files["manifest"] == files["report"]:
        raise ConfigError("'dataset.manifest' and 'dataset.report' must be different files")
    return DatasetConfig(
        group_metadata=files["group_metadata"],
        manifest=files["manifest"],
        report=files["report"],
    )


def _augmentation(value: Any) -> AugmentationConfig:
    mapping = _mapping(value, "augmentation")
    _check_keys(
        mapping,
        ("max_rotation_degrees", "max_translation", "min_scale", "max_scale", "brightness", "contrast"),
        "augmentation",
    )
    max_rotation = _number(mapping["max_rotation_degrees"], "augmentation.max_rotation_degrees")
    if not 0.0 <= max_rotation <= 45.0:
        raise ConfigError(
            f"'augmentation.max_rotation_degrees' must be between 0 and 45, got {max_rotation}"
        )
    max_translation = _number(mapping["max_translation"], "augmentation.max_translation")
    if not 0.0 <= max_translation < 0.5:
        raise ConfigError(
            f"'augmentation.max_translation' must be in [0, 0.5), got {max_translation}"
        )
    min_scale = _number(mapping["min_scale"], "augmentation.min_scale")
    max_scale = _number(mapping["max_scale"], "augmentation.max_scale")
    if not 0.0 < min_scale <= max_scale:
        raise ConfigError(
            "'augmentation.min_scale' and 'augmentation.max_scale' must satisfy "
            f"0 < min_scale <= max_scale, got {min_scale} and {max_scale}"
        )
    brightness = _number(mapping["brightness"], "augmentation.brightness")
    contrast = _number(mapping["contrast"], "augmentation.contrast")
    for name, amount in (("brightness", brightness), ("contrast", contrast)):
        if not 0.0 <= amount < 1.0:
            raise ConfigError(f"'augmentation.{name}' must be in [0, 1), got {amount}")
    return AugmentationConfig(
        max_rotation_degrees=max_rotation,
        max_translation=max_translation,
        min_scale=min_scale,
        max_scale=max_scale,
        brightness=brightness,
        contrast=contrast,
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "classes",
    "cups",
    "paths",
    "dataset",
    "image",
    "augmentation",
    "split",
    "random_seed",
    "training",
    "model",
    "inference",
    "camera",
)


def parse_config(raw: Any, base_dir: Path) -> Config:
    """Validate an already-parsed config mapping and build a ``Config``.

    Relative paths in the config are resolved against ``base_dir``.
    """
    top = _mapping(raw, "config")
    _check_keys(top, _TOP_LEVEL_KEYS, "config")

    classes = _classes(top["classes"])

    inference = _mapping(top["inference"], "inference")
    _check_keys(inference, ("confidence_threshold",), "inference")
    camera = _mapping(top["camera"], "camera")
    _check_keys(camera, ("index",), "camera")

    paths = _paths(top["paths"], base_dir)
    return Config(
        classes=classes,
        cups=_cups(top["cups"], classes),
        paths=paths,
        dataset=_dataset(top["dataset"], paths.dataset_dir, base_dir),
        image=_image(top["image"]),
        augmentation=_augmentation(top["augmentation"]),
        split=_split(top["split"]),
        random_seed=_int(top["random_seed"], "random_seed", minimum=0),
        training=_training(top["training"]),
        model=_model(top["model"]),
        confidence_threshold=_open_unit_interval(
            inference["confidence_threshold"], "inference.confidence_threshold"
        ),
        camera_index=_int(camera["index"], "camera.index", minimum=0),
    )


def load_config(path: Path | str | None = None, base_dir: Path | None = None) -> Config:
    """Load and validate a YAML config file.

    ``path`` defaults to ``config/config.yaml`` in the project root. Relative
    paths inside the config are resolved against ``base_dir``, which defaults
    to the parent of the folder containing the config file (the project root
    for the standard layout).
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    root = base_dir if base_dir is not None else config_path.resolve().parent.parent
    return parse_config(_read_yaml(config_path), root)
