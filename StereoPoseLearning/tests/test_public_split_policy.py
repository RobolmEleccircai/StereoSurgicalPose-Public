"""Public source contracts, with no model weights or private dataset required."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from stereo_pose.split_policy import verify_disjoint_splits, verify_source_assignments


def record(split, group, index=0):
    return {"split": split, "trajectory_group_id": group, "sequence_id": "seq_" + group,
            "temporal_group_id": group, "frame_index": index, "source_frame_index": index,
            "timestamp_ns": round(index * 1e9 / 60),
            "image_sha256": {eye: f"{group}/{index}/{eye}" for eye in ("left", "right")}}


class PublicSplitPolicyTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"records": [record(s, f"group_{s}") for s in ("train", "validation", "test")]}

    def test_independent_trajectories_and_same_group_neighbors(self):
        self.manifest["records"].append(record("train", "group_train", 1))
        result = verify_disjoint_splits(self.manifest)
        self.assertEqual(result["pair_counts"], {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(result["trajectory_counts"]["train"], 1)

    def test_frame_reassignment_rejected_even_when_nonoverlapping(self):
        self.manifest["records"].append(record("train", "group_validation", 30))
        with self.assertRaisesRegex(ValueError, "trajectory lineage"):
            verify_disjoint_splits(self.manifest)

    def test_derived_trajectory_cannot_cross_split(self):
        self.manifest["records"][0]["parent_trajectory_group_id"] = "group_test"
        with self.assertRaisesRegex(ValueError, "trajectory lineage"):
            verify_disjoint_splits(self.manifest)

    def test_renaming_copied_stereo_pair_does_not_bypass(self):
        self.manifest["records"][0]["image_sha256"] = copy.deepcopy(self.manifest["records"][1]["image_sha256"])
        with self.assertRaisesRegex(ValueError, "stereo image pair"):
            verify_disjoint_splits(self.manifest)

    def test_source_split_and_wrong_counts_rejected(self):
        self.manifest["records"][0]["source_split"] = "validation"
        with self.assertRaisesRegex(ValueError, "Source assignment"):
            verify_disjoint_splits(self.manifest)
        del self.manifest["records"][0]["source_split"]
        self.manifest["trajectory_split_counts"] = {"train": 3, "validation": 0, "test": 0}
        with self.assertRaisesRegex(ValueError, "trajectory_split_counts"):
            verify_disjoint_splits(self.manifest)

    def test_actual_source_label_assignment_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "label.json"
            r = self.manifest["records"][0]
            label = {k: r[k] for k in ("split", "sequence_id", "trajectory_group_id", "frame_index", "timestamp_ns")}
            label["eyes"] = {eye: {"image": eye + ".png", "mask": eye + "_mask.png"} for eye in ("left", "right")}
            path.write_text(json.dumps(label))
            r.update(label_path=str(path), label_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     image_paths={e: str(path.parent / (e + ".png")) for e in ("left", "right")},
                     mask_paths={e: str(path.parent / (e + "_mask.png")) for e in ("left", "right")},
                     mask_sha256={e: "unused" for e in ("left", "right")})
            single = {"records": [r]}
            self.assertEqual(verify_source_assignments(single)["source_labels_verified"], 1)
            # Re-signing the manifest would not change the source label assignment.
            r["split"] = "validation"
            with self.assertRaisesRegex(ValueError, "Source label split"):
                verify_source_assignments(single)
            r["split"] = "train"
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "label hash mismatch"):
                verify_source_assignments(single)

    def test_missing_identity_and_empty_dataset_rejected(self):
        del self.manifest["records"][0]["trajectory_group_id"]
        with self.assertRaisesRegex(ValueError, "trajectory_group_id"):
            verify_disjoint_splits(self.manifest)
        with self.assertRaisesRegex(ValueError, "Empty"):
            verify_disjoint_splits({"records": []})


if __name__ == "__main__":
    unittest.main()
