"""Tests for src.preprocessing (RGB conversion, resizing, augmentation, normalisation)."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from src.config import AugmentationConfig, ImageConfig
from src.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    augment_training,
    ensure_rgb,
    load_rgb,
    preprocess_eval,
    preprocess_training,
    resize_and_center_crop,
)

IMAGE = ImageConfig(width=64, height=64)
AUGMENT = AugmentationConfig(
    max_rotation_degrees=10.0,
    max_translation=0.05,
    min_scale=0.9,
    max_scale=1.1,
    brightness=0.15,
    contrast=0.15,
)
NO_AUGMENT = AugmentationConfig(
    max_rotation_degrees=0.0,
    max_translation=0.0,
    min_scale=1.0,
    max_scale=1.0,
    brightness=0.0,
    contrast=0.0,
)


def textured_rgb(height: int = 90, width: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


class LoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_file_is_loaded_as_rgb(self) -> None:
        bgr_red = np.zeros((8, 8, 3), dtype=np.uint8)
        bgr_red[:, :, 2] = 255  # OpenCV stores this as BGR: pure red
        path = self.root / "red.png"
        self.assertTrue(cv2.imwrite(str(path), bgr_red))
        rgb = load_rgb(path)
        self.assertEqual(rgb.shape, (8, 8, 3))
        self.assertEqual(rgb[0, 0].tolist(), [255, 0, 0])

    def test_non_ascii_path_loads(self) -> None:
        path = self.root / "صورة_ö.jpg"
        self.assertTrue(cv2.imwrite(str(path), np.full((10, 10, 3), 128, dtype=np.uint8)))
        self.assertEqual(load_rgb(path).shape, (10, 10, 3))

    def test_undecodable_file_raises_value_error(self) -> None:
        path = self.root / "broken.jpg"
        path.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "cannot decode"):
            load_rgb(path)
        empty = self.root / "empty.png"
        empty.write_bytes(b"")
        with self.assertRaises(ValueError):
            load_rgb(empty)

    def test_grey_and_alpha_inputs_become_rgb(self) -> None:
        grey = np.full((4, 4), 100, dtype=np.uint8)
        out = ensure_rgb(grey, is_bgr=False)
        self.assertEqual(out.shape, (4, 4, 3))
        self.assertTrue((out == 100).all())
        bgra = np.zeros((4, 4, 4), dtype=np.uint8)
        bgra[:, :, 0] = 255  # blue in BGR
        bgra[:, :, 3] = 0
        out = ensure_rgb(bgra, is_bgr=True)
        self.assertEqual(out.shape, (4, 4, 3))
        self.assertEqual(out[0, 0].tolist(), [0, 0, 255])  # blue moves to the last RGB channel

    def test_bad_in_memory_arrays_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ensure_rgb(np.zeros((4, 4), dtype=np.float32), is_bgr=False)
        with self.assertRaises(ValueError):
            ensure_rgb(np.zeros((4, 4, 2), dtype=np.uint8), is_bgr=False)


class ResizeTests(unittest.TestCase):
    def test_output_has_exact_target_size(self) -> None:
        for shape in ((90, 120), (200, 50), (30, 30), (10, 300)):
            with self.subTest(shape=shape):
                out = resize_and_center_crop(textured_rgb(*shape), 64, 48)
                self.assertEqual(out.shape, (48, 64, 3))
                self.assertEqual(out.dtype, np.uint8)

    def test_centre_crop_keeps_the_middle(self) -> None:
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[40:60, 40:60] = 255  # bright square in the middle
        out = resize_and_center_crop(rgb, 50, 50)
        self.assertGreater(out[25, 25].mean(), 200)


class EvalPreprocessingTests(unittest.TestCase):
    def test_output_is_chw_float32_and_normalised(self) -> None:
        out = preprocess_eval(textured_rgb(), IMAGE)
        self.assertEqual(out.shape, (3, 64, 64))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(out.min(), -3.0)
        self.assertLess(out.max(), 3.0)

    def test_normalisation_uses_imagenet_statistics(self) -> None:
        white = np.full((40, 40, 3), 255, dtype=np.uint8)
        out = preprocess_eval(white, IMAGE)
        expected = ((1.0 - IMAGENET_MEAN) / IMAGENET_STD).reshape(3, 1, 1)
        np.testing.assert_allclose(out, np.broadcast_to(expected, out.shape), rtol=1e-5, atol=1e-5)

    def test_validation_and_test_preprocessing_is_deterministic(self) -> None:
        rgb = textured_rgb(seed=3)
        first = preprocess_eval(rgb, IMAGE)
        second = preprocess_eval(rgb.copy(), IMAGE)
        np.testing.assert_array_equal(first, second)

    def test_eval_does_not_depend_on_any_random_state(self) -> None:
        rgb = textured_rgb(seed=4)
        np.random.seed(1)
        first = preprocess_eval(rgb, IMAGE)
        np.random.seed(999)
        np.testing.assert_array_equal(first, preprocess_eval(rgb, IMAGE))


class TrainingAugmentationTests(unittest.TestCase):
    def test_training_augmentation_differs_from_eval(self) -> None:
        rgb = textured_rgb(seed=7)
        eval_out = preprocess_eval(rgb, IMAGE)
        train_out = preprocess_training(rgb, IMAGE, AUGMENT, np.random.default_rng(0))
        self.assertEqual(train_out.shape, eval_out.shape)
        self.assertFalse(np.allclose(train_out, eval_out))

    def test_augmentation_is_reproducible_with_the_same_seed(self) -> None:
        rgb = textured_rgb(seed=8)
        a = preprocess_training(rgb, IMAGE, AUGMENT, np.random.default_rng(5))
        b = preprocess_training(rgb, IMAGE, AUGMENT, np.random.default_rng(5))
        c = preprocess_training(rgb, IMAGE, AUGMENT, np.random.default_rng(6))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.allclose(a, c))

    def test_zero_augmentation_matches_eval_exactly(self) -> None:
        # Guards against the training and evaluation pipelines drifting apart.
        rgb = textured_rgb(seed=9)
        np.testing.assert_allclose(
            preprocess_training(rgb, IMAGE, NO_AUGMENT, np.random.default_rng(0)),
            preprocess_eval(rgb, IMAGE),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_augmented_base_stays_in_unit_range_and_keeps_channel_order(self) -> None:
        red = np.zeros((80, 80, 3), dtype=np.uint8)
        red[:, :, 0] = 255  # pure red in RGB
        base = resize_and_center_crop(red, 64, 64)
        for seed in range(5):
            with self.subTest(seed=seed):
                out = augment_training(base, AUGMENT, np.random.default_rng(seed))
                self.assertGreaterEqual(out.min(), 0.0)
                self.assertLessEqual(out.max(), 1.0)
                # Red stays the dominant channel: augmentation never swaps channels.
                self.assertGreater(out[..., 0].mean(), out[..., 1].mean() + 0.3)

    def test_augmentation_strength_is_taken_from_config(self) -> None:
        # With every amount set to zero, augmentation is a no-op, so the settings are what matter.
        rgb = textured_rgb(seed=10)
        base = resize_and_center_crop(rgb, 64, 64).astype(np.float32) / 255.0
        identity = augment_training(
            resize_and_center_crop(rgb, 64, 64), replace(NO_AUGMENT), np.random.default_rng(2)
        )
        np.testing.assert_allclose(identity, base, rtol=1e-5, atol=1e-5)
        jittered = augment_training(
            resize_and_center_crop(rgb, 64, 64), replace(AUGMENT, brightness=0.5), np.random.default_rng(2)
        )
        self.assertFalse(np.allclose(jittered, base))


if __name__ == "__main__":
    unittest.main()
