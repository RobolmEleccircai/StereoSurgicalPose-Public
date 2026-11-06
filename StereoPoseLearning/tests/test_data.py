"""Reader contract tests using real-schema tiny files, without a simulator."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from stereo_pose.data import (StereoDataset, build_manifest, collate_samples,
                              project_points, resize_intrinsics)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_label(root, frame, *, seq="seq_000", split="train", group="trajectory_a",
               source_frame=None, isolated=False, job=None, pitch=0.1,
               scene_hash="asset-version-a"):
    """Use the actual TemporalStereo_v2 label field names and coordinate units."""
    job = job or f"{frame:06d}"
    label_dir = root / "sequences" / seq / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    source_frame = frame if source_frame is None else source_frame
    label = {"schema_version": 5, "sequence_id": seq, "job_id": job,
             "mode": "validate" if isolated else "generate", "fps": 60,
             "frame_index": frame, "source_frame_index": source_frame,
             "timestamp_ns": round(source_frame * 1e9 / 60), "time_s": frame / 60,
             "source_time_s": source_frame / 60, "position_unit": "mm",
             "category": "occlusion", "phase": "free_motion", "split": split,
             "trajectory_group_id": group, "feature_profile_id": "example/occlusion",
             "isolated_validation_pose_override": isolated,
             "body_pose_truth": {"source_scene_sha256": scene_hash},
             "quality_flags": {"overexposed_any_eye": True},
             "joint_positions": {"psm": {"psm_tool_pitch_joint": pitch,
                "psm_tool_yaw_joint": 0.2, "psm_tool_gripper1_joint": -0.15,
                "psm_tool_gripper2_joint": 0.15}},
             # This nonexistent file must never be opened by the model reader.
             "camera_truth": "not_a_model_input.json", "eyes": {}}
    for eye, offset in (("left", 0.0), ("right", -5.0)):
        images = label_dir.parent / "images" / eye
        masks = label_dir.parent / "masks" / eye
        images.mkdir(parents=True, exist_ok=True)
        masks.mkdir(parents=True, exist_ok=True)
        image_path, mask_path = images / f"{job}.png", masks / f"{job}.png"
        Image.new("RGB", (64, 32), (100 + frame % 50, 120, 140)).save(image_path)
        Image.new("L", (64, 32), 255).save(mask_path)
        origins = {"shaft": [offset, 0, 100], "jaw1": [offset - 1, 0, 110],
                   "jaw2": [offset + 1, 0, 110]}
        frames = {}
        for name, position in origins.items():
            transform = np.eye(4)
            transform[:3, 3] = position
            frames[name] = {"T_optical_from_local_mm": transform.tolist()}
        tips = [[offset - 2, 0, 120], [offset + 2, 0, 120]]
        label["eyes"][eye] = {"image": f"../images/{eye}/{job}.png",
             "image_sha256": sha(image_path), "mask": f"../masks/{eye}/{job}.png",
             "mask_sha256": sha(mask_path),
             "K": [[60, 0, 31.5], [0, 60, 15.5], [0, 0, 1]],
             "D": [-0.05, 0.003, 0.001, -0.001, 0.0001],
             "local_frames": frames,
             "tool": {"shaft_reference_position_mm": origins["shaft"],
                      "tip_position_mm": [offset, 0, 120],
                      "jaw1_tip_position_mm": tips[0], "jaw2_tip_position_mm": tips[1]},
             "fixed_tip_visibility": [{"visible": False}, {"visible": False}],
             "jaw_tip_region_visibility": [{"region_visible": True}, {"region_visible": False}]}
    label_path = label_dir / f"{job}.json"
    label_path.write_text(json.dumps(label), encoding="utf8")
    return label_path


def save_plan(root, n=3, *, seq="seq_000", group="trajectory_a", split="train"):
    path = root / "trajectories" / f"{seq}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sequence_id": seq, "trajectory_group_id": group,
                               "frame_count": n, "split": split, "category": "occlusion"}))


class DataContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "run"
        self.root.mkdir()
        self.manifest_path = Path(self.temp.name) / "manifest.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_full_plan_and_shapes_occluded_targets_are_kept(self):
        save_plan(self.root)
        for frame in range(3):
            save_label(self.root, frame)
        manifest = build_manifest([self.root], self.manifest_path)
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["split_counts"]["train"], 3)
        self.assertEqual(manifest["network_input_keys"], ["images", "K", "D"])
        dataset = StereoDataset(manifest, "train", window=2, image_size=(16, 32))
        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        shapes = {"images": (2, 2, 3, 16, 32), "K": (2, 2, 3, 3),
                  "D": (2, 2, 5), "C": (2, 2, 3), "R": (2, 2, 3, 3),
                  "joints": (2, 3), "points": (2, 2, 6, 3),
                  "R_jaws": (2, 2, 2, 3, 3), "masks": (2, 2, 1, 4, 8),
                  "heatmaps": (2, 2, 5, 4, 8), "heatmap_valid": (2, 2, 5)}
        for key, shape in shapes.items():
            self.assertEqual(tuple(sample[key].shape), shape, key)
            self.assertTrue(torch.isfinite(sample[key]).all(), key)
        self.assertTrue(sample["heatmap_valid"].bool().all())
        self.assertTrue((sample["heatmaps"].flatten(-2).sum(-1) > 0).all())
        self.assertTrue(sample["metadata"][0]["overexposed"])
        self.assertFalse(sample["metadata"][0]["fixed_tip_visibility"]["left"][0]["visible"])
        self.assertNotIn("camera_truth", sample)
        self.assertNotIn("baseline", sample)
        self.assertEqual(tuple(collate_samples([sample, sample])["images"].shape), (2, 2, 2, 3, 16, 32))
        # C is the shaft origin; H1/H2 are the jaw transform origins, not J1/J2.
        np.testing.assert_allclose(sample["points"][0, 0, 1].numpy(), [-1, 0, 110])
        np.testing.assert_allclose(sample["points"][0, 0, 3].numpy(), [-2, 0, 120])
        frame = dataset.load_frame(dataset.frame_records()[0])
        self.assertEqual(tuple(frame["images"].shape), (2, 3, 16, 32))

    def test_resize_projection_pixel_centres(self):
        K = np.asarray([[1000, 0, 639.2], [0, 990, 358.7], [0, 0, 1.]])
        D = np.asarray([-0.2, 0.01, 0.002, -0.003, 0.001])
        points = np.asarray([[2, -3, 100], [15, 8, 70], [0, 0, -1.]])
        resized = resize_intrinsics(K, (720, 1280), (360, 640))
        p0, front0 = project_points(points, K, D)
        p1, front1 = project_points(points, resized, D)
        np.testing.assert_allclose(p1, (p0 + 0.5) * 0.5 - 0.5, atol=1e-10)
        np.testing.assert_array_equal(front0, [True, True, False])
        np.testing.assert_array_equal(front0, front1)

    def test_windows_never_cross_gaps_split_or_isolated(self):
        for frame in (0, 1, 3, 4):
            save_label(self.root, frame)
        save_label(self.root, 1, isolated=True, job="boundary", pitch=0.9)
        for frame in range(2):
            save_label(self.root, frame, seq="seq_001", group="trajectory_b", split="validation", pitch=1.0)
        manifest = build_manifest([self.root], self.manifest_path)
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["isolated_diagnostic_pairs"], 1)
        self.assertEqual(len(manifest["temporal_gaps"]), 1)
        self.assertEqual(len(StereoDataset(manifest, "train", window=1)), 5)
        train = StereoDataset(manifest, "train", window=2)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(StereoDataset(manifest, "train", window=3)), 0)
        validation = StereoDataset(manifest, "validation", window=1)
        self.assertEqual(len(validation), 2)
        self.assertTrue(validation.frame_records()[0]["outside_train_joint_range"][0])
        self.assertEqual([r["split"] for r in validation.frame_records()], ["validation"] * 2)

    def test_unrendered_indices_do_not_create_frames(self):
        save_plan(self.root, n=180)
        save_label(self.root, 42)
        (self.root / "indices").mkdir()
        (self.root / "indices" / "train.jsonl").write_text('{"frame_index":0,"rendered_and_verified":false}\n')
        manifest = build_manifest([self.root], self.manifest_path)
        self.assertFalse(manifest["complete"])
        self.assertEqual(len(manifest["records"]), 1)
        self.assertEqual(manifest["planned_split_counts"]["train"], 180)

    def test_legacy_boundary_reason_overrides_false_flag(self):
        for frame in (0, 1):
            save_label(self.root, frame)
        path = save_label(self.root, 2, job="validation_029")
        label = json.loads(path.read_text())
        label.update({"mode": "validate", "reason": "maximum_opening",
                      "isolated_validation_pose_override": False})
        path.write_text(json.dumps(label))
        manifest = build_manifest([self.root], self.manifest_path)
        self.assertEqual(manifest["isolated_diagnostic_pairs"], 1)
        self.assertEqual(len(StereoDataset(manifest, "train", window=1)), 3)
        self.assertEqual(len(StereoDataset(manifest, "train", window=2)), 1)
        self.assertIn("Legacy validate", " ".join(manifest["warnings"]))

    def test_matching_dedup_and_conflicting_duplicate_rejected(self):
        save_label(self.root, 0)
        clone = Path(self.temp.name) / "clone"
        shutil.copytree(self.root, clone)
        manifest = build_manifest([self.root, clone], self.manifest_path)
        self.assertEqual(len(manifest["records"]), 1)
        self.assertEqual(len(manifest["deduplicated"]), 1)
        p = clone / "sequences/seq_000/labels/000000.json"
        label = json.loads(p.read_text())
        label["joint_positions"]["psm"]["psm_tool_pitch_joint"] += 0.1
        p.write_text(json.dumps(label))
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            build_manifest([self.root, clone], Path(self.temp.name) / "conflict.json")

    def test_split_leak_and_asset_version_conflict_are_rejected(self):
        save_label(self.root, 0)
        p = save_label(self.root, 0, seq="seq_001", split="test")
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            build_manifest([self.root], self.manifest_path)
        label = json.loads(p.read_text())
        label["trajectory_group_id"] = "different_trajectory"
        label["body_pose_truth"]["source_scene_sha256"] = "asset-version-b"
        p.write_text(json.dumps(label))
        with self.assertRaisesRegex(ValueError, "Different USD"):
            build_manifest([self.root], self.manifest_path)

    def test_changed_asset_after_snapshot_and_missing_pair_are_rejected(self):
        save_label(self.root, 0)
        manifest = build_manifest([self.root], self.manifest_path)
        image = Path(manifest["records"][0]["image_paths"]["left"])
        Image.new("RGB", (64, 32), (0, 0, 0)).save(image)
        with self.assertRaisesRegex(ValueError, "changed after manifest"):
            StereoDataset(manifest, "train", window=1)[0]
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            build_manifest([self.root], Path(self.temp.name) / "changed.json")
        image.unlink()
        with self.assertRaises(FileNotFoundError):
            build_manifest([self.root], Path(self.temp.name) / "missing.json")


if __name__ == "__main__":
    unittest.main()
