"""Read-only public training split checks; no sample reassignment or filtering."""
from collections import Counter
import hashlib
import json
from pathlib import Path

SPLITS = ("train", "validation", "test")
LINEAGE_KEYS = ("trajectory_group_id", "source_trajectory_group_id",
                "parent_trajectory_group_id", "original_trajectory_group_id")


def verify_disjoint_splits(manifest):
    """Reject cross-split trajectories, duplicate images and inconsistent counts.

    A group identifies a physical trajectory including its derived variants.
    Sequence IDs may repeat in different capture runs; group IDs must be global.
    This check does not infer independence from similar pixels or model scores.
    """
    records = manifest.get("records", [])
    if not records:
        raise ValueError("Empty dataset manifest")
    owners = {}
    counts, production_counts = Counter(), Counter()
    groups = {split: set() for split in SPLITS}

    def register(kind, value, split):
        key = (kind, value)
        previous = owners.setdefault(key, split)
        if previous != split:
            raise ValueError(f"Cross-split {kind}: {previous} / {split}")

    for record in records:
        split = record.get("split")
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split!r}")
        group = record.get("trajectory_group_id")
        if not isinstance(group, str) or not group:
            raise ValueError("Every frame requires a nonempty trajectory_group_id")
        for key in LINEAGE_KEYS:
            if record.get(key):
                register("trajectory lineage", str(record[key]), split)
        for key in ("original_split", "source_split"):
            if key in record and record[key] != split:
                raise ValueError("Source assignment differs from manifest split")
        if record.get("temporal_group_id"):
            register("temporal group", record["temporal_group_id"], split)
        # Renaming a group cannot make copied stereo images independent.
        hashes = record.get("image_sha256", {})
        if set(hashes) != {"left", "right"} or not all(hashes.values()):
            raise ValueError("Both stereo image hashes are required")
        register("stereo image pair", (hashes["left"], hashes["right"]), split)
        if record.get("label_sha256"):
            register("label content", record["label_sha256"], split)
        if record.get("label_path"):
            register("label path", str(Path(record["label_path"]).expanduser().resolve()), split)
        counts[split] += 1
        if not record.get("isolated_diagnostic", False):
            production_counts[split] += 1
        groups[split].add(group)
    for key, actual in (("split_counts", counts), ("trajectory_split_counts", production_counts)):
        if key in manifest and manifest[key] != {s: actual[s] for s in SPLITS}:
            raise ValueError(f"Recorded {key} does not match frame assignments")
    return {"passed": True, "unit": "whole trajectory including derived variants",
            "pair_counts": {s: counts[s] for s in SPLITS},
            "trajectory_counts": {s: len(groups[s]) for s in SPLITS}}


def verify_source_assignments(manifest):
    """Match each record to its immutable, hashed capture label before training.

    The caller must retain independently generated, immutable source labels.
    Checks cannot establish authenticity if both the labels and their lineage
    have been deliberately rewritten; preserve capture plans and hashes too.
    """
    report = verify_disjoint_splits(manifest)
    for record in manifest["records"]:
        path = Path(record["label_path"]).expanduser()
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["label_sha256"]:
            raise ValueError(f"Source label hash mismatch: {path}")
        label = json.loads(data)
        for key in ("split", "sequence_id", "trajectory_group_id", "frame_index", "timestamp_ns"):
            if key not in label or label[key] != record.get(key):
                raise ValueError(f"Source label {key} differs from manifest: {path}")
        for key in ("original_split", "source_split"):
            if key in label and label[key] != record["split"]:
                raise ValueError(f"Source label assignment differs: {path}")
        for key in LINEAGE_KEYS[1:]:
            if key in label and label[key] != record.get(key):
                raise ValueError(f"Source lineage missing or changed: {path}")
        source_index = label.get("source_frame_index", label["frame_index"])
        if source_index != record.get("source_frame_index", record["frame_index"]):
            raise ValueError(f"Source frame identity differs: {path}")
        for eye in ("left", "right"):
            source_image = label["eyes"][eye]
            for field, paths_key, hash_key in (("image", "image_paths", "image_sha256"),
                                              ("mask", "mask_paths", "mask_sha256")):
                if (path.parent / source_image[field]).resolve() != Path(record[paths_key][eye]).expanduser().resolve():
                    raise ValueError(f"Source {field} path differs: {path}")
                expected = source_image.get(field + "_sha256")
                if expected is not None and expected != record[hash_key][eye]:
                    raise ValueError(f"Source {field} identity differs: {path}")
    report["source_labels_verified"] = len(manifest["records"])
    return report
