"""Shared test helpers: build small real images and datasets in temp folders."""

from __future__ import annotations

import itertools
from pathlib import Path

import cv2
import numpy as np

_seeds = itertools.count(1)


def encode_image(seed: int | None = None, ext: str = ".jpg", shape: tuple[int, int, int] = (48, 64, 3)) -> bytes:
    """Encode a random-noise image. Different seeds give different bytes."""
    rng = np.random.default_rng(next(_seeds) if seed is None else seed)
    image = rng.integers(0, 256, size=shape, dtype=np.uint8)
    ok, buf = cv2.imencode(ext, image)
    assert ok
    return buf.tobytes()


def write_image(path: Path, seed: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_image(seed, ext=path.suffix.lower() or ".jpg"))
    return path


def make_dataset(root: Path, counts: dict[str, int]) -> None:
    """Create ``root/<class>/img_N.jpg`` with ``counts[class]`` unique valid images."""
    for name, count in counts.items():
        (root / name).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            write_image(root / name / f"img_{i:03d}.jpg")


def snapshot(root: Path) -> dict[str, bytes | None]:
    """Relative path -> file bytes (None for directories), for read-only checks."""
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }
