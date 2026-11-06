"""Tests for the template-based synthetic dataset builder. Uses small generated photos only."""

from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.config import DEFAULT_CONFIG_PATH, Config, load_config
from src.dataset import validate_dataset
from src.prepare_dataset import read_group_metadata
from src.synthesize_dataset import (
    SynthesisError,
    SynthesisSettings,
    compose_scene,
    find_templates,
    isolate_object,
    main,
    make_cutout,
    read_bgr,
    run_synthesis,
    _jitter_colour,
    _rotate_expand,
)

H, W = 240, 320
CLASSES = ("metal", "plastic", "paper", "other")
FAST = SynthesisSettings(per_template=3, seed=1)


def platform(seed: int = 0) -> np.ndarray:
    """A textured grey platform, BGR uint8."""
    rng = np.random.default_rng(seed)
    base = np.full((H, W, 3), (120, 135, 150), np.float32)
    base += cv2.GaussianBlur(rng.normal(0, 4, (H, W)).astype(np.float32), (0, 0), 2)[..., None]
    return np.clip(base, 0, 255).astype(np.uint8)


def object_photo(shape_mask: np.ndarray, colour=(40, 60, 220), brighten: float = 1.0) -> np.ndarray:
    """The platform with a coloured object drawn on it, optionally with different exposure."""
    img = platform().astype(np.float32)
    img[shape_mask > 0] = colour
    img *= brighten
    return np.clip(img, 0, 255).astype(np.uint8)


def rectangle(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def write_template_set(folder: Path, *, names=("Metal", "Plastic", "Paper", "Organic"), background="empty") -> dict[str, np.ndarray]:
    """Write the four template photos and the empty platform. Returns the true masks by name."""
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{background}.jpg"), platform(), [cv2.IMWRITE_JPEG_QUALITY, 95])
    masks = {}
    layouts = {
        "Metal": rectangle(40, 40, 110, 100),
        "Plastic": rectangle(180, 120, 260, 190),
        "Paper": rectangle(60, 150, 140, 210),
        "Organic": rectangle(220, 30, 290, 80),
    }
    for name in names:
        mask = layouts[name]
        masks[name] = mask
        cv2.imwrite(str(folder / f"{name}.jpg"), object_photo(mask, brighten=1.06), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return masks


def make_config(root: Path) -> Config:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["paths"] = {"dataset_dir": str(root / "dataset"), "model_dir": str(root / "models")}
    raw["dataset"] = {
        "group_metadata": str(root / "group_metadata.csv"),
        "manifest": str(root / "splits" / "manifest.csv"),
        "report": str(root / "splits" / "split_report.txt"),
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


class ProjectTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.template_dir = self.root / "template"
        self.masks = write_template_set(self.template_dir)
        self.config = make_config(self.root)


class TemplateDiscoveryTests(ProjectTestCase):
    def test_finds_four_templates_and_the_background(self) -> None:
        templates = find_templates(self.template_dir, CLASSES)
        self.assertEqual([t.class_name for t in templates.objects], ["metal", "plastic", "paper", "other"])
        self.assertEqual(templates.background.name, "empty.jpg")

    def test_organic_maps_to_other(self) -> None:
        templates = find_templates(self.template_dir, CLASSES)
        organic = [t for t in templates.objects if t.stem == "Organic"][0]
        self.assertEqual(organic.class_name, "other")

    def test_names_are_case_insensitive_and_any_image_extension_works(self) -> None:
        folder = self.root / "mixed"
        folder.mkdir()
        cv2.imwrite(str(folder / "metal.PNG"), platform())
        cv2.imwrite(str(folder / "PLASTIC.bmp"), platform())
        cv2.imwrite(str(folder / "paper.jpeg"), platform())
        cv2.imwrite(str(folder / "ORGANIC.jpg"), platform())
        cv2.imwrite(str(folder / "platform.jpg"), platform())
        templates = find_templates(folder, CLASSES)
        self.assertEqual(templates.background.name, "platform.jpg")

    def test_missing_template_is_named_in_the_error(self) -> None:
        (self.template_dir / "Paper.jpg").unlink()
        with self.assertRaisesRegex(SynthesisError, "missing template image for Paper"):
            find_templates(self.template_dir, CLASSES)

    def test_extra_faces_of_one_object_are_grouped_under_its_class(self) -> None:
        for name in ("Metal_2.jpg", "metal 3.png", "Metal-side.jpeg", "Organic2.jpg"):
            cv2.imwrite(str(self.template_dir / name), platform())
        templates = find_templates(self.template_dir, CLASSES)
        self.assertEqual(len(templates.objects), 4)  # still one object per class
        by_class = {t.class_name: t for t in templates.objects}
        self.assertEqual(
            sorted(f.name for f in by_class["metal"].files),
            ["Metal-side.jpeg", "Metal.jpg", "Metal_2.jpg", "metal 3.png"],
        )
        self.assertEqual(sorted(f.name for f in by_class["other"].files), ["Organic.jpg", "Organic2.jpg"])
        self.assertEqual(len(by_class["plastic"].files), 1)
        self.assertEqual(templates.background.name, "empty.jpg")

    def test_a_longer_word_starting_with_a_class_name_is_not_a_face(self) -> None:
        # "metallic_platform" is not "Metal" + separator, so it is the extra photo, not a face.
        cv2.imwrite(str(self.template_dir / "metallic_platform.jpg"), platform())
        with self.assertRaisesRegex(SynthesisError, "exactly one empty platform photo"):
            find_templates(self.template_dir, CLASSES)

    def test_faces_do_not_hide_a_missing_class(self) -> None:
        (self.template_dir / "Paper.jpg").unlink()
        cv2.imwrite(str(self.template_dir / "Metal_2.jpg"), platform())
        with self.assertRaisesRegex(SynthesisError, "missing template image for Paper"):
            find_templates(self.template_dir, CLASSES)

    def test_missing_empty_platform_is_rejected(self) -> None:
        (self.template_dir / "empty.jpg").unlink()
        with self.assertRaisesRegex(SynthesisError, "exactly one empty platform photo"):
            find_templates(self.template_dir, CLASSES)

    def test_two_extra_photos_are_rejected_as_background(self) -> None:
        cv2.imwrite(str(self.template_dir / "second_empty.jpg"), platform())
        with self.assertRaisesRegex(SynthesisError, "exactly one empty platform photo"):
            find_templates(self.template_dir, CLASSES)

    def test_missing_folder_is_rejected(self) -> None:
        with self.assertRaisesRegex(SynthesisError, "template folder not found"):
            find_templates(self.root / "nowhere", CLASSES)

    def test_class_missing_from_config_is_rejected(self) -> None:
        with self.assertRaisesRegex(SynthesisError, "'other', which is not in config classes"):
            find_templates(self.template_dir, ("metal", "plastic", "paper", "glass"))

    def test_non_image_files_are_ignored(self) -> None:
        (self.template_dir / "notes.txt").write_text("hello", encoding="utf-8")
        templates = find_templates(self.template_dir, CLASSES)
        self.assertEqual(len(templates.objects), 4)


class IsolationTests(ProjectTestCase):
    def test_recovers_the_object_with_high_overlap(self) -> None:
        background = read_bgr(self.template_dir / "empty.jpg")
        for name, truth in self.masks.items():
            with self.subTest(template=name):
                photo = read_bgr(self.template_dir / f"{name}.jpg")
                mask = isolate_object(photo, background, name=name)
                predicted = mask > 0
                actual = truth > 0
                iou = (predicted & actual).sum() / (predicted | actual).sum()
                self.assertGreaterEqual(iou, 0.9, f"IoU {iou:.3f}")

    def test_exposure_difference_does_not_break_isolation(self) -> None:
        background = platform()
        dark_photo = object_photo(self.masks["Metal"], brighten=0.85)
        mask = isolate_object(dark_photo, background, name="dark")
        self.assertGreater(float(mask.mean()) / 255, 0.05)

    def test_photo_without_an_object_is_refused(self) -> None:
        with self.assertRaisesRegex(SynthesisError, "could not find an object"):
            isolate_object(platform(), platform(), name="blank")

    def test_size_mismatch_is_refused(self) -> None:
        small = np.zeros((H // 2, W // 2, 3), np.uint8)
        with self.assertRaisesRegex(SynthesisError, "same camera and framing"):
            isolate_object(small, platform(), name="small")

    def test_object_larger_than_half_the_frame_is_refused(self) -> None:
        # Covers about 56% of the frame and does not touch the border, so the platform colour is still readable.
        big = object_photo(rectangle(40, 30, 280, 210))
        with self.assertRaisesRegex(SynthesisError, "limit is 50%"):
            isolate_object(big, platform(), name="big")

    def test_object_touching_the_border_still_isolates_correctly(self) -> None:
        mask = rectangle(0, 0, 60, 50)  # in the corner, touching the frame edge
        photo = object_photo(mask, brighten=1.05)
        found = isolate_object(photo, platform(), name="corner") > 0
        iou = (found & (mask > 0)).sum() / (found | (mask > 0)).sum()
        self.assertGreater(iou, 0.85)

    def test_tiny_object_is_refused(self) -> None:
        tiny = object_photo(rectangle(100, 100, 106, 106))
        with self.assertRaisesRegex(SynthesisError, "too small"):
            isolate_object(tiny, platform(), name="tiny")

    def test_cutout_is_cropped_with_alpha_from_the_mask(self) -> None:
        photo = read_bgr(self.template_dir / "Plastic.jpg")
        mask = isolate_object(photo, platform(), name="Plastic")
        cutout = make_cutout(photo, mask)
        self.assertEqual(cutout.shape[2], 4)
        self.assertLess(cutout.shape[0] * cutout.shape[1], H * W / 4)
        self.assertEqual(int(cutout[..., 3].max()), 255)
        self.assertEqual(int(cutout[..., 3].min()), 0)


class CompositingTests(unittest.TestCase):
    def setUp(self) -> None:
        mask = rectangle(100, 100, 150, 140)
        photo = object_photo(mask)
        self.cutout = make_cutout(photo, mask)
        self.background = platform()

    def test_output_matches_the_camera_frame(self) -> None:
        scene = compose_scene(self.background, self.cutout, np.random.default_rng(0), FAST)
        self.assertEqual(scene.shape, self.background.shape)
        self.assertEqual(scene.dtype, np.uint8)

    def test_same_seed_gives_identical_pixels(self) -> None:
        first = compose_scene(self.background, self.cutout, np.random.default_rng([4, 2]), FAST)
        second = compose_scene(self.background, self.cutout, np.random.default_rng([4, 2]), FAST)
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_give_different_scenes(self) -> None:
        scenes = [compose_scene(self.background, self.cutout, np.random.default_rng(s), FAST) for s in range(4)]
        for other in scenes[1:]:
            self.assertFalse(np.array_equal(scenes[0], other))

    def test_zero_hue_saturation_and_value_jitter_leaves_colour_nearly_unchanged(self) -> None:
        still = SynthesisSettings(max_hue_shift_degrees=0.0, saturation_jitter=0.0, value_jitter=0.0)
        patch = np.full((20, 20, 3), (40, 60, 220), np.uint8)
        out = _jitter_colour(patch, np.random.default_rng(0), still)
        self.assertLessEqual(int(np.abs(out.astype(int) - patch.astype(int)).max()), 3)

    def test_hue_shift_changes_object_colour_when_allowed(self) -> None:
        patch = np.full((20, 20, 3), (40, 60, 220), np.uint8)
        wide = SynthesisSettings(max_hue_shift_degrees=60.0, saturation_jitter=0.0, value_jitter=0.0)
        outs = [_jitter_colour(patch, np.random.default_rng(s), wide) for s in range(6)]
        self.assertTrue(any(not np.array_equal(outs[0], o) for o in outs[1:]))

    def test_rotation_enlarges_the_canvas_so_nothing_is_cropped(self) -> None:
        rotated = _rotate_expand(np.full((10, 20, 4), 255, np.uint8), 45.0)
        self.assertGreater(rotated.shape[0], 20)
        self.assertGreater(rotated.shape[1], 20)
        self.assertGreater(int(rotated[..., 3].sum()), 0)  # the object survives the rotation

    def test_settings_reject_bad_values(self) -> None:
        bad = [
            dict(per_template=0),
            dict(seed=-1),
            dict(min_scale=1.2, max_scale=0.8),
            dict(blur_probability=2.0),
            dict(max_hue_shift_degrees=-5.0),
        ]
        for kwargs in bad:
            with self.subTest(**kwargs):
                with self.assertRaises(SynthesisError):
                    SynthesisSettings(**kwargs)


class GenerationTests(ProjectTestCase):
    def read_csv(self) -> dict[str, tuple[str, str]]:
        known = {p.relative_to(self.config.paths.dataset_dir).as_posix() for p in self.config.paths.dataset_dir.rglob("*.jpg")}
        mapping, _ = read_group_metadata(self.config.dataset.group_metadata, known)
        return mapping

    def test_writes_the_requested_count_into_each_class_folder(self) -> None:
        result, _ = run_synthesis(self.config, self.template_dir, FAST, stamp="t")
        for name in CLASSES:
            files = list((self.config.paths.dataset_dir / name).glob("*.jpg"))
            self.assertEqual(len(files), FAST.per_template, name)
        self.assertEqual(len(result.items), 4 * FAST.per_template)

    def test_generated_dataset_passes_the_existing_validator(self) -> None:
        run_synthesis(self.config, self.template_dir, FAST, stamp="t")
        report = validate_dataset(self.config.paths.dataset_dir, CLASSES, min_recommended_images=0)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_valid_images, 4 * FAST.per_template)

    def test_group_file_gives_every_variant_its_template_object_id(self) -> None:
        run_synthesis(self.config, self.template_dir, FAST, stamp="t")
        mapping = self.read_csv()
        self.assertEqual(len(mapping), 4 * FAST.per_template)
        ids_by_class: dict[str, set[str]] = {}
        for rel, (object_id, session_id) in mapping.items():
            class_name = rel.split("/", 1)[0]
            ids_by_class.setdefault(class_name, set()).add(object_id)
            self.assertEqual(session_id, f"synthetic-seed-{FAST.seed}")
        for name in CLASSES:
            self.assertEqual(len(ids_by_class[name]), 1, f"{name} variants must share one group")

    def write_extra_faces(self) -> None:
        """Metal gets 3 faces in total, Organic gets 2; each face shows the object somewhere else."""
        for name, layout in (
            ("Metal_2", rectangle(150, 30, 215, 95)),
            ("Metal_3", rectangle(20, 140, 90, 200)),
            ("Organic_2", rectangle(30, 30, 100, 90)),
        ):
            cv2.imwrite(str(self.template_dir / f"{name}.jpg"), object_photo(layout, brighten=1.06), [cv2.IMWRITE_JPEG_QUALITY, 95])

    def test_per_template_is_split_across_faces_so_class_totals_stay_equal(self) -> None:
        self.write_extra_faces()
        settings = SynthesisSettings(per_template=7, seed=1)
        result, _ = run_synthesis(self.config, self.template_dir, settings, stamp="t")
        for name in CLASSES:
            files = list((self.config.paths.dataset_dir / name).glob("*.jpg"))
            self.assertEqual(len(files), 7, name)
        self.assertEqual(len(result.items), 4 * 7)
        metal_names = sorted(p.name for p in (self.config.paths.dataset_dir / "metal").glob("*.jpg"))
        for stem, count in (("Metal", 3), ("Metal_2", 2), ("Metal_3", 2)):  # remainder goes to the first faces
            self.assertEqual(len([n for n in metal_names if re.fullmatch(rf"synth_{stem}_\d{{4}}\.jpg", n)]), count, stem)

    def test_all_faces_of_a_class_share_one_group_id(self) -> None:
        self.write_extra_faces()
        run_synthesis(self.config, self.template_dir, SynthesisSettings(per_template=6, seed=1), stamp="t")
        ids_by_class: dict[str, set[str]] = {}
        for rel, (object_id, _) in self.read_csv().items():
            ids_by_class.setdefault(rel.split("/", 1)[0], set()).add(object_id)
        for name in CLASSES:
            self.assertEqual(len(ids_by_class[name]), 1, f"{name}: every face must be in the same split group")
        self.assertEqual(len({next(iter(v)) for v in ids_by_class.values()}), 4)

    def test_faces_look_different_from_each_other(self) -> None:
        self.write_extra_faces()
        run_synthesis(self.config, self.template_dir, SynthesisSettings(per_template=3, seed=1), stamp="t")
        folder = self.config.paths.dataset_dir / "other"
        self.assertEqual(len({p.read_bytes() for p in folder.glob("*.jpg")}), 3)

    def test_same_stem_with_different_extensions_does_not_collide(self) -> None:
        cv2.imwrite(str(self.template_dir / "metal.png"), object_photo(rectangle(150, 30, 215, 95)))
        result, _ = run_synthesis(self.config, self.template_dir, SynthesisSettings(per_template=4, seed=1), stamp="t")
        files = list((self.config.paths.dataset_dir / "metal").glob("*.jpg"))
        self.assertEqual(len(files), 4)
        self.assertEqual(len({p.name.lower() for p in files}), 4)

    def test_too_few_images_for_the_number_of_faces_is_refused(self) -> None:
        self.write_extra_faces()
        with self.assertRaisesRegex(SynthesisError, "Metal.*3 faces"):
            run_synthesis(self.config, self.template_dir, SynthesisSettings(per_template=2, seed=1), stamp="t")
        self.assertFalse(self.config.paths.dataset_dir.exists())

    def test_report_lists_every_face(self) -> None:
        self.write_extra_faces()
        _, report_path = run_synthesis(self.config, self.template_dir, SynthesisSettings(per_template=6, seed=1), stamp="t")
        objects = json.loads(report_path.read_text(encoding="utf-8"))["objects"]
        metal = objects["synthetic-metal-Metal"]
        self.assertEqual(sorted(metal["template_files"]), ["Metal.jpg", "Metal_2.jpg", "Metal_3.jpg"])
        self.assertEqual(metal["images"], 6)
        self.assertEqual(metal["images_per_face"], {"Metal.jpg": 2, "Metal_2.jpg": 2, "Metal_3.jpg": 2})

    def test_file_names_are_ascii_and_do_not_touch_the_templates(self) -> None:
        before = sorted(p.name for p in self.template_dir.iterdir())
        run_synthesis(self.config, self.template_dir, FAST, stamp="t")
        after = sorted(p.name for p in self.template_dir.iterdir())
        self.assertEqual(before, after)
        for path in self.config.paths.dataset_dir.rglob("*.jpg"):
            path.name.encode("ascii")

    def test_same_seed_gives_byte_identical_images(self) -> None:
        run_synthesis(self.config, self.template_dir, FAST, stamp="t")
        first = {p.name: p.read_bytes() for p in (self.config.paths.dataset_dir / "metal").glob("*.jpg")}
        again = self.root / "again"
        again.mkdir()
        other = make_config(again)
        run_synthesis(other, self.template_dir, FAST, stamp="t")
        second = {p.name: p.read_bytes() for p in (other.paths.dataset_dir / "metal").glob("*.jpg")}
        self.assertEqual(first, second)

    def test_report_is_written_outside_the_dataset_without_absolute_paths(self) -> None:
        _, report_path = run_synthesis(self.config, self.template_dir, FAST, stamp="t")
        self.assertFalse(str(report_path).startswith(str(self.config.paths.dataset_dir)))
        text = report_path.read_text(encoding="utf-8")
        report = json.loads(text)
        self.assertEqual(report["seed"], FAST.seed)
        self.assertEqual(report["total_images"], 4 * FAST.per_template)
        self.assertEqual(report["background_file"], "empty.jpg")
        self.assertNotIn(str(self.root), text)

    def test_refuses_existing_output_without_overwrite(self) -> None:
        run_synthesis(self.config, self.template_dir, FAST, stamp="t")
        keep = sorted((self.config.paths.dataset_dir / "metal").glob("*.jpg"))
        with self.assertRaisesRegex(SynthesisError, "Use --overwrite"):
            run_synthesis(self.config, self.template_dir, FAST, stamp="t2")
        self.assertEqual(sorted((self.config.paths.dataset_dir / "metal").glob("*.jpg")), keep)

    def test_overwrite_moves_the_old_dataset_to_a_backup(self) -> None:
        run_synthesis(self.config, self.template_dir, FAST, stamp="first")
        old_dir = self.config.paths.dataset_dir
        marker = old_dir / "metal" / "old_photo.jpg"
        marker.write_bytes(b"old")
        run_synthesis(self.config, self.template_dir, FAST, overwrite=True, stamp="second")
        backup = old_dir.with_name(old_dir.name + ".second.bak")
        self.assertTrue((backup / "metal" / "old_photo.jpg").exists())
        self.assertFalse((old_dir / "metal" / "old_photo.jpg").exists())
        self.assertEqual(len(list((old_dir / "metal").glob("*.jpg"))), FAST.per_template)

    def test_failed_generation_leaves_the_existing_dataset_untouched(self) -> None:
        run_synthesis(self.config, self.template_dir, FAST, stamp="t")
        before = sorted(p.name for p in (self.config.paths.dataset_dir / "metal").glob("*.jpg"))
        (self.template_dir / "Paper.jpg").write_bytes(b"not an image")
        with self.assertRaises(SynthesisError):
            run_synthesis(self.config, self.template_dir, FAST, overwrite=True, stamp="t2")
        self.assertEqual(sorted(p.name for p in (self.config.paths.dataset_dir / "metal").glob("*.jpg")), before)
        self.assertFalse(self.config.paths.dataset_dir.with_name(self.config.paths.dataset_dir.name + ".partial").exists())
        self.assertFalse(self.config.paths.dataset_dir.with_name(self.config.paths.dataset_dir.name + ".t2.bak").exists())


class CommandLineTests(ProjectTestCase):
    def run_cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--config", str(self.root / "config.yaml"), *argv])
        return code, out.getvalue()

    def test_success_prints_counts_and_exits_zero(self) -> None:
        code, out = self.run_cli("--template-dir", str(self.template_dir), "--per-template", "2")
        self.assertEqual(code, 0)
        self.assertIn("Wrote 8 images", out)
        self.assertIn("metal=2", out)

    def test_bad_per_template_exits_2(self) -> None:
        code, out = self.run_cli("--template-dir", str(self.template_dir), "--per-template", "0")
        self.assertEqual(code, 2)
        self.assertIn("per_template", out)

    def test_missing_templates_exit_1(self) -> None:
        code, out = self.run_cli("--template-dir", str(self.root / "missing"))
        self.assertEqual(code, 1)
        self.assertIn("template folder not found", out)

    def test_bad_config_exits_2(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--config", str(self.root / "nope.yaml")])
        self.assertEqual(code, 2)

    def test_default_template_folder_sits_next_to_the_dataset(self) -> None:
        # setUp writes the templates to <root>/template, the same place the default resolves to
        # when the dataset is at <root>/dataset.
        self.assertEqual(self.config.paths.dataset_dir.parent / "template", self.template_dir)
        code, _ = self.run_cli("--per-template", "1")
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
