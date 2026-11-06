"""Read immutable TemporalStereo labels without leaking truth into model inputs.

Only ``images``, ``K`` and ``D`` are network inputs. All other tensors returned
by this module are supervision targets. The reader never opens camera_truth.
Occlusion and overexposure are retained; keypoint heatmaps are *amodal* targets.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


EYES = ("left", "right")
POINT_NAMES = ("C", "H1", "H2", "J1", "J2", "T")
HEATMAP_NAMES = POINT_NAMES[:5]
SPLITS = ("train", "validation", "test")
ISOLATED_VALIDATION_REASONS = {"closed_jaw", "maximum_opening", "contact_keyframe", "withdraw_keyframe"}
TENSOR_KEYS = (
    "images", "K", "D", "C", "R", "joints", "points", "R_jaws",
    "masks", "heatmaps", "heatmap_valid", "image_size",
)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf8") as handle:
        return json.load(handle)


def _joint_values(label: dict) -> np.ndarray:
    q = label["joint_positions"]["psm"]
    half = float(q["psm_tool_gripper2_joint"])
    if abs(float(q["psm_tool_gripper1_joint"]) + half) > 1e-7:
        raise ValueError("Unsupported independently driven/asymmetric jaws")
    result = np.asarray([q["psm_tool_pitch_joint"], q["psm_tool_yaw_joint"], half], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite joint target")
    return result


def _pose_truth(label: dict) -> dict[str, np.ndarray]:
    cs, rs, points, jaws = [], [], [], []
    for eye in EYES:
        e = label["eyes"][eye]
        shaft = np.asarray(e["local_frames"]["shaft"]["T_optical_from_local_mm"], dtype=np.float64)
        j1 = np.asarray(e["local_frames"]["jaw1"]["T_optical_from_local_mm"], dtype=np.float64)
        j2 = np.asarray(e["local_frames"]["jaw2"]["T_optical_from_local_mm"], dtype=np.float64)
        if any(x.shape != (4, 4) for x in (shaft, j1, j2)):
            raise ValueError("Expected 4x4 optical-from-local transforms")
        c = np.asarray(e["tool"]["shaft_reference_position_mm"], dtype=np.float64)
        if not np.allclose(c, shaft[:3, 3], atol=1e-4, rtol=0):
            raise ValueError("C does not match the shaft transform origin")
        tips = [np.asarray(e["tool"][f"jaw{i}_tip_position_mm"], dtype=np.float64) for i in (1, 2)]
        midpoint = (tips[0] + tips[1]) * 0.5
        if not np.allclose(midpoint, e["tool"]["tip_position_mm"], atol=1e-4, rtol=0):
            raise ValueError("T is inconsistent with the two fixed tip identities")
        for matrix in (shaft, j1, j2):
            r = matrix[:3, :3]
            if not np.allclose(r.T @ r, np.eye(3), atol=2e-4) or np.linalg.det(r) < 0.999:
                raise ValueError("Invalid rotation target")
        cs.append(c)
        rs.append(shaft[:3, :3])
        points.append(np.stack([c, j1[:3, 3], j2[:3, 3], *tips, midpoint]))
        jaws.append(np.stack([j1[:3, :3], j2[:3, :3]]))
    result = {"C": np.stack(cs), "R": np.stack(rs), "points": np.stack(points),
              "R_jaws": np.stack(jaws), "joints": _joint_values(label)}
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError("Nonfinite pose target")
    return result


def _image_metadata(path: Path, expected_hash: str | None, verify_hashes: bool) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if verify_hashes and expected_hash and _hash_file(path) != expected_hash:
        raise ValueError(f"Image/mask SHA256 mismatch: {path}")
    with Image.open(path) as image:
        size = image.size
        image.verify()
    return size


def _planned_sequences(root: Path) -> dict[str, dict]:
    result = {}
    for path in sorted((root / "trajectories").glob("seq_*.json")):
        plan = _read_json(path)
        if "split" not in plan or "frame_count" not in plan:
            continue
        key = plan.get("trajectory_group_id", plan["sequence_id"])
        result[key] = {"sequence_id": plan["sequence_id"], "split": plan["split"],
                       "category": plan.get("category", "unknown"), "frame_count": int(plan["frame_count"])}
    return result


def build_manifest(run_roots: Iterable[str | Path], output: str | Path,
                   *, verify_hashes: bool = True) -> dict:
    """Index actual committed RGB+mask+label pairs, never unrendered plan rows.

    Matching duplicate source frames are explicitly deduplicated by immutable
    label and image hashes. Conflicting duplicates fail. Isolated diagnostic
    states remain in the manifest as one-frame groups; they cannot join a video.
    The supplied run roots can be disjoint worker outputs for a partial dataset.
    """
    roots = [Path(p).expanduser().resolve() for p in run_roots]
    if not roots:
        raise ValueError("At least one run root is required")
    records, deduplicated = [], []
    seen, groups, plans = {}, {}, {}
    source_hashes = set()
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for key, plan in _planned_sequences(root).items():
            if key in plans and plans[key] != plan:
                raise ValueError(f"Conflicting trajectory plan {key}")
            plans[key] = plan
        label_paths = sorted(root.glob("sequences/*/labels/*.json"))
        if not label_paths:
            raise ValueError(f"No generated frame labels under {root}/sequences/*/labels")
        for path in label_paths:
            label = _read_json(path)
            if "eyes" not in label or "joint_positions" not in label:
                continue  # Non-frame receipts/summary files are not image data.
            split = label["split"]
            if split not in SPLITS:
                raise ValueError(f"Unknown split {split!r} in {path}")
            if label.get("position_unit") != "mm":
                raise ValueError(f"Unsupported position unit in {path}")
            if abs(float(label.get("fps", 0)) - 60.0) > 1e-7:
                raise ValueError(f"Expected 60 Hz source timestamps: {path}")
            truth = _pose_truth(label)
            group = str(label.get("trajectory_group_id", label["sequence_id"]))
            if group in groups and groups[group] != split:
                raise ValueError(f"Trajectory group crosses splits: {group}")
            groups[group] = split
            legacy_boundary = (label.get("mode") == "validate"
                               and label.get("reason") in ISOLATED_VALIDATION_REASONS)
            isolated = bool(label.get("isolated_validation_pose_override", False)) or legacy_boundary
            # Preserve the parent identity for split-leakage checks and report the
            # independent temporal identity for explicit diagnostic overrides.
            temporal_group = group + ("/isolated/" + str(label["job_id"]) if isolated else "")
            scene_hash = label.get("body_pose_truth", {}).get("source_scene_sha256")
            if not scene_hash:
                raise ValueError(f"Missing USD source identity: {path}")
            source_hashes.add(scene_hash)
            images, masks, image_hashes, mask_hashes = {}, {}, {}, {}
            sizes = []
            for eye in EYES:
                e = label["eyes"][eye]
                image = (path.parent / e["image"]).resolve()
                mask = (path.parent / e["mask"]).resolve()
                size = _image_metadata(image, e.get("image_sha256"), verify_hashes)
                mask_size = _image_metadata(mask, e.get("mask_sha256"), verify_hashes)
                if size != mask_size:
                    raise ValueError(f"RGB/mask dimensions differ: {path} {eye}")
                K = np.asarray(e["K"], dtype=np.float64)
                D = np.asarray(e["D"], dtype=np.float64)
                if K.shape != (3, 3) or D.shape != (5,) or not np.isfinite(K).all() or not np.isfinite(D).all():
                    raise ValueError(f"Invalid K/D: {path} {eye}")
                if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
                    raise ValueError(f"Invalid intrinsics: {path} {eye}")
                images[eye], masks[eye] = str(image), str(mask)
                image_hashes[eye] = e.get("image_sha256") or _hash_file(image)
                mask_hashes[eye] = e.get("mask_sha256") or _hash_file(mask)
                sizes.append(size)
            if sizes[0] != sizes[1]:
                raise ValueError(f"Stereo image dimensions differ: {path}")
            frame_index = int(label["frame_index"])
            source_index = int(label.get("source_frame_index", frame_index))
            record = {
                "label_path": str(path.resolve()), "label_sha256": _hash_file(path),
                "run_root": str(root), "sequence_id": str(label["sequence_id"]),
                "trajectory_group_id": group, "temporal_group_id": temporal_group,
                "feature_profile_id": label.get("feature_profile_id", ""),
                "split": split, "category": label["category"], "phase": label.get("phase", "unknown"),
                "job_id": str(label["job_id"]), "frame_index": frame_index,
                "source_frame_index": source_index, "timestamp_ns": int(label["timestamp_ns"]),
                "source_time_s": float(label.get("source_time_s", label["time_s"])),
                "fps": float(label["fps"]), "image_paths": images, "mask_paths": masks,
                "image_sha256": image_hashes, "mask_sha256": mask_hashes,
                "original_image_size": [sizes[0][1], sizes[0][0]],
                "source_scene_sha256": scene_hash, "isolated_diagnostic": isolated,
                "isolation_reason": ("legacy_validation_boundary_reason" if legacy_boundary
                                     else "explicit_validation_override" if isolated else None),
                "overexposed": bool(label.get("quality_flags", {}).get("overexposed_any_eye", False)),
                "joint_angles_rad": truth["joints"].tolist(),
                "fixed_tip_visibility": {eye: label["eyes"][eye].get("fixed_tip_visibility", []) for eye in EYES},
                "tip_region_visibility": {eye: [bool(v.get("region_visible", False)) for v in label["eyes"][eye].get("jaw_tip_region_visibility", [])] for eye in EYES},
            }
            identity = (temporal_group, source_index)
            if identity in seen:
                old = seen[identity]
                signatures = ("label_sha256", "image_sha256", "mask_sha256", "split", "timestamp_ns")
                if any(old[k] != record[k] for k in signatures):
                    raise ValueError(f"Conflicting duplicate source frame {identity}: {old['label_path']} vs {path}")
                deduplicated.append({"kept": old["label_path"], "duplicate": str(path.resolve())})
                continue
            seen[identity] = record
            records.append(record)
    if not records:
        raise ValueError("No actual stereo pairs found")
    if len(source_hashes) != 1:
        raise ValueError("Different USD source versions cannot be mixed in one training manifest")
    for record in records:
        plan = plans.get(record["trajectory_group_id"])
        if plan and any(record[key] != plan[key] for key in ("sequence_id", "split", "category")):
            raise ValueError(f"Frame identity/split contradicts trajectory plan: {record['label_path']}")
    records.sort(key=lambda r: (r["split"], r["sequence_id"], r["temporal_group_id"], r["frame_index"], r["timestamp_ns"]))
    observed_groups = defaultdict(set)
    temporal_records = defaultdict(list)
    for record in records:
        if not record["isolated_diagnostic"]:
            observed_groups[record["trajectory_group_id"]].add(record["frame_index"])
            temporal_records[record["temporal_group_id"]].append(record)
    planned_counts = Counter()
    for plan in plans.values():
        planned_counts[plan["split"]] += plan["frame_count"]
    complete = bool(plans) and all(observed_groups[group] == set(range(plan["frame_count"])) for group, plan in plans.items())
    complete = complete and all(group in plans for group in observed_groups)
    temporal_gaps = []
    for group, rows in temporal_records.items():
        rows.sort(key=lambda r: r["frame_index"])
        for a, b in zip(rows, rows[1:]):
            if not _continuous(a, b):
                temporal_gaps.append({"temporal_group_id": group,
                                      "after_frame_index": a["frame_index"],
                                      "before_frame_index": b["frame_index"]})
    complete = complete and not temporal_gaps
    warnings = []
    if not complete:
        warnings.append("Partial/diagnostic data: this manifest is not a completed full-dataset validation.")
    isolated_count = sum(r["isolated_diagnostic"] for r in records)
    if isolated_count:
        warnings.append(f"{isolated_count} isolated diagnostic pairs retained as independent one-frame groups.")
    if any(r["isolation_reason"] == "legacy_validation_boundary_reason" for r in records):
        warnings.append("Legacy validate boundary reasons override an inconsistent isolated_validation_pose_override=false flag; these states are not temporal video frames.")
    if temporal_gaps:
        warnings.append(f"{len(temporal_gaps)} missing/discontinuous intervals: temporal windows never cross these gaps.")
    # Generalization ranges are descriptions only; they do not reject/reassign
    # user-approved held-out poses or remove occluded examples.
    train_joints = [r["joint_angles_rad"] for r in records if r["split"] == "train" and not r["isolated_diagnostic"]]
    ranges = None
    if train_joints:
        a = np.asarray(train_joints)
        low, high = a.min(0), a.max(0)
        ranges = {"minimum_rad": low.tolist(), "maximum_rad": high.tolist(),
                  "scope": "complete_train" if complete else "observed_train_only"}
        for record in records:
            a = np.asarray(record["joint_angles_rad"])
            flags = ((a < low - 1e-7) | (a > high + 1e-7)).tolist()
            record["outside_train_joint_range"] = flags
    counts = Counter(r["split"] for r in records)
    trajectory_counts = Counter(r["split"] for r in records if not r["isolated_diagnostic"])
    manifest = {
        "schema_version": 1, "roots": [str(x) for x in roots], "records": records,
        "complete": complete, "warnings": warnings,
        "split_counts": {s: counts[s] for s in SPLITS},
        "trajectory_split_counts": {s: trajectory_counts[s] for s in SPLITS},
        "planned_split_counts": {s: planned_counts[s] for s in SPLITS},
        "isolated_diagnostic_pairs": isolated_count, "deduplicated": deduplicated,
        "temporal_gaps": temporal_gaps,
        "source_scene_sha256": next(iter(source_hashes)), "train_joint_ranges": ranges,
        "network_input_keys": ["images", "K", "D"],
        "target_point_order": list(POINT_NAMES), "heatmap_point_order": list(HEATMAP_NAMES),
        "heatmap_semantics": "amodal projected reference points; visibility is not implied",
        "fps": 60, "hashes_verified": verify_hashes,
    }
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Manifest output already exists; choose a new snapshot path: {destination}")
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return manifest


def resize_intrinsics(K: np.ndarray, source_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    """Pixel-center convention; both sizes are (height,width), D is unchanged."""
    sh, sw = source_size
    th, tw = target_size
    sx, sy = tw / sw, th / sh
    A = np.asarray([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float64)
    return A @ np.asarray(K, dtype=np.float64)


def project_points(points: np.ndarray, K: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Brown-Conrady OpenCV 5 coefficient projection used for target maps."""
    points = np.asarray(points, dtype=np.float64)
    z = points[..., 2]
    positive = z > 1e-8
    xy = points[..., :2] / np.where(positive, z, 1.0)[..., None]
    x, y = xy[..., 0], xy[..., 1]
    k1, k2, p1, p2, k3 = np.asarray(D)
    r2 = x*x + y*y
    radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
    uvh = np.stack([xd, yd, np.ones_like(x)], -1) @ np.asarray(K).T
    uv = uvh[..., :2] / uvh[..., 2:3]
    return uv, positive & np.isfinite(uv).all(-1)


def _heatmaps(uv: np.ndarray, valid: np.ndarray, image_size: tuple[int, int], stride: int,
              sigma: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_size
    out_h, out_w = height // stride, width // stride
    # Resize pixel centres, rather than simply dividing uv by stride.
    centre = (uv + 0.5) / stride - 0.5
    valid = valid & (uv[:, 0] >= -0.5) & (uv[:, 0] < width - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < height - 0.5)
    centre = np.where(valid[:, None], centre, 0.0)
    yy, xx = np.mgrid[:out_h, :out_w]
    dist2 = (xx[None] - centre[:, 0, None, None]) ** 2 + (yy[None] - centre[:, 1, None, None]) ** 2
    values = np.exp(-dist2 / (2 * sigma * sigma)) * valid[:, None, None]
    return values.astype(np.float32), valid


def _continuous(a: dict, b: dict) -> bool:
    return (a["temporal_group_id"] == b["temporal_group_id"]
            and a["sequence_id"] == b["sequence_id"] and a["split"] == b["split"]
            and not a["isolated_diagnostic"] and not b["isolated_diagnostic"]
            and b["frame_index"] == a["frame_index"] + 1
            and b["source_frame_index"] == a["source_frame_index"] + 1
            and abs((b["timestamp_ns"] - a["timestamp_ns"]) - 1e9/60) <= 2)


class StereoDataset(Dataset):
    """Fixed-length causal windows within a single trajectory and split.

    Evaluation uses window=1 plus model-state caching to include sequence starts.
    No fabricated padding, no cross-gap windows and no future frame is appended.
    """
    def __init__(self, manifest: str | Path | dict, split: str, window: int = 8,
                 image_size: tuple[int, int] = (360, 640), heatmap_stride: int = 4,
                 heatmap_sigma: float = 1.5, *, verify_labels: bool = True):
        if split not in SPLITS:
            raise ValueError(f"Invalid split {split!r}")
        if not 1 <= window <= 32:
            raise ValueError("Window must be between 1 and 32, without crossing trajectories")
        if heatmap_stride <= 0 or heatmap_sigma <= 0:
            raise ValueError("Heatmap stride and sigma must be positive")
        if len(image_size) != 2 or min(image_size) <= 0 or any(x % heatmap_stride for x in image_size):
            raise ValueError("Image height and width must be positive multiples of heatmap_stride")
        self.manifest = _read_json(Path(manifest).expanduser()) if isinstance(manifest, (str, Path)) else manifest
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported model dataset manifest schema")
        self.split, self.window = split, int(window)
        self.image_size = tuple(map(int, image_size))
        self.heatmap_stride, self.heatmap_sigma = int(heatmap_stride), float(heatmap_sigma)
        self.verify_labels = verify_labels
        self.records = self.frame_records(split)
        self.windows = []
        start = 0
        for end, record in enumerate(self.records):
            if end and not _continuous(self.records[end-1], record):
                start = end
            if end - start + 1 >= self.window:
                self.windows.append(tuple(range(end - self.window + 1, end + 1)))
        self._label_cache = {}
        self._verified_images = set()

    def __len__(self) -> int:
        return len(self.windows)

    def frame_records(self, split: str | None = None) -> list[dict]:
        chosen = self.split if split is None else split
        return sorted([r for r in self.manifest["records"] if r["split"] == chosen],
                      key=lambda r: (r["sequence_id"], r["temporal_group_id"], r["frame_index"], r["timestamp_ns"]))

    def load_frame(self, record: dict) -> dict[str, Any]:
        path = Path(record["label_path"])
        # Cache decoded labels only; RGB arrays remain bounded by the loader batch.
        if str(path) not in self._label_cache:
            if self.verify_labels and _hash_file(path) != record["label_sha256"]:
                raise ValueError(f"Label changed after manifest snapshot: {path}")
            self._label_cache[str(path)] = _read_json(path)
        label = self._label_cache[str(path)]
        truth = _pose_truth(label)
        height, width = self.image_size
        images, masks, cameras, distortions, heatmaps, valid_maps = [], [], [], [], [], []
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        for i, eye in enumerate(EYES):
            if self.verify_labels:
                for paths, hashes in (("image_paths", "image_sha256"), ("mask_paths", "mask_sha256")):
                    image_path = Path(record[paths][eye])
                    if str(image_path) not in self._verified_images:
                        if _hash_file(image_path) != record[hashes][eye]:
                            raise ValueError(f"Image/mask changed after manifest snapshot: {image_path}")
                        self._verified_images.add(str(image_path))
            with Image.open(record["image_paths"][eye]) as im:
                source_size = (im.height, im.width)
                rgb = np.asarray(im.convert("RGB").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255
            with Image.open(record["mask_paths"][eye]) as im:
                mask = np.asarray(im.convert("L").resize((width // self.heatmap_stride, height // self.heatmap_stride), Image.Resampling.NEAREST))
            images.append((rgb.transpose(2, 0, 1) - mean) / std)
            masks.append((mask > 0)[None].astype(np.float32))
            K = resize_intrinsics(np.asarray(label["eyes"][eye]["K"]), source_size, self.image_size)
            D = np.asarray(label["eyes"][eye]["D"], dtype=np.float64)
            uv, front = project_points(truth["points"][i, :5], K, D)
            hm, valid = _heatmaps(uv, front, self.image_size, self.heatmap_stride, self.heatmap_sigma)
            cameras.append(K)
            distortions.append(D)
            heatmaps.append(hm)
            valid_maps.append(valid)
        arrays = {**truth, "images": np.stack(images), "K": np.stack(cameras), "D": np.stack(distortions),
                  "masks": np.stack(masks), "heatmaps": np.stack(heatmaps),
                  "heatmap_valid": np.stack(valid_maps), "image_size": np.asarray(self.image_size)}
        result = {key: torch.as_tensor(np.array(value, copy=True), dtype=torch.float32) for key, value in arrays.items()}
        result["metadata"] = {key: record[key] for key in (
            "sequence_id", "trajectory_group_id", "temporal_group_id", "split", "category", "phase", "frame_index",
            "source_frame_index", "timestamp_ns", "source_time_s", "label_path", "job_id", "isolated_diagnostic",
            "overexposed", "fixed_tip_visibility", "tip_region_visibility")}
        result["metadata"]["image_paths"] = record["image_paths"]
        result["metadata"]["mask_paths"] = record["mask_paths"]
        result["metadata"]["outside_train_joint_range"] = record.get("outside_train_joint_range", [False]*3)
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        frames = [self.load_frame(self.records[i]) for i in self.windows[index]]
        result = {key: torch.stack([frame[key] for frame in frames]) for key in TENSOR_KEYS}
        result["metadata"] = [frame["metadata"] for frame in frames]
        return result


def collate_samples(samples: list[dict]) -> dict[str, Any]:
    """Stack tensors into [B,T,...], preserving metadata as [B][T] records."""
    result = {key: torch.stack([sample[key] for sample in samples]) for key in TENSOR_KEYS}
    result["metadata"] = [sample["metadata"] for sample in samples]
    return result
