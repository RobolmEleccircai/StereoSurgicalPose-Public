"""CPU-only, read-only C projection-domain audit; never starts training/rendering.

Example: python tools/check_unified_c.py --manifest RUN/manifest.json
         --config RUN/config.json --output evidence/c_projection_domain.json
"""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo_pose.data import _pose_truth, project_points, resize_intrinsics


def audit(manifest_path, config_path, stride=4):
    torch.set_num_threads(2)
    manifest_path, config_path = Path(manifest_path), Path(config_path)
    manifest_bytes = manifest_path.read_bytes()
    config_bytes = config_path.read_bytes()
    manifest, config = json.loads(manifest_bytes), json.loads(config_bytes)
    target_size = tuple(config["image_size"])
    per_eye = defaultdict(list)
    exceptional = []
    hashes = hashlib.sha256()
    label_projection_max_error = 0.0
    for record in manifest["records"]:
        content = Path(record["label_path"]).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != record["label_sha256"]:
            raise ValueError(f"Manifest hash differs: {record['label_path']}")
        hashes.update(record["label_path"].encode() + digest.encode())
        label = json.loads(content)
        truth = _pose_truth(label)
        source_size = tuple(record["original_image_size"])
        for index, eye in enumerate(("left", "right")):
            e = label["eyes"][eye]
            K, D = np.asarray(e["K"], dtype=np.float64), np.asarray(e["D"], dtype=np.float64)
            C = truth["C"][index:index+1]
            raw_uv, positive = project_points(C, K, D)
            uv, _ = project_points(C, resize_intrinsics(K, source_size, target_size), D)
            raw_uv, uv = raw_uv[0], uv[0]
            h, w = target_size
            sh, sw = source_size
            in_raw = bool(positive[0] and -.5 <= raw_uv[0] < sw-.5 and -.5 <= raw_uv[1] < sh-.5)
            in_network = bool(positive[0] and -.5 <= uv[0] < w-.5 and -.5 <= uv[1] < h-.5)
            # A positive heatmap expectation lies inside the feature-pixel-centre hull.
            low = (stride - 1) / 2
            high_x, high_y = w - (stride + 1) / 2, h - (stride + 1) / 2
            in_hull = bool(positive[0] and low <= uv[0] <= high_x and low <= uv[1] <= high_y)
            margin = float(min(uv[0]-low, uv[1]-low, high_x-uv[0], high_y-uv[1]))
            entry = {"raw_uv": raw_uv.tolist(), "network_uv": uv.tolist(),
                     "positive_depth": bool(positive[0]), "inside_raw_image": in_raw,
                     "inside_network_image": in_network, "inside_heatmap_centre_hull": in_hull,
                     "heatmap_centre_hull_margin_px": margin}
            if "shaft_reference_pixel" in e["tool"]:
                reference = np.asarray(e["tool"]["shaft_reference_pixel"], dtype=np.float64)
                label_projection_max_error = max(label_projection_max_error, float(np.linalg.norm(raw_uv-reference)))
            for key in (f"all/{eye}", f"{record['split']}/{eye}", f"category:{record['category']}/{eye}"):
                per_eye[key].append(entry)
            if not (in_raw and in_network and in_hull):
                exceptional.append({"sequence_id": record["sequence_id"], "frame_index": record["frame_index"],
                                    "split": record["split"], "eye": eye, **entry})
    stats = {}
    for key, rows in per_eye.items():
        raw, uv = np.array([r["raw_uv"] for r in rows]), np.array([r["network_uv"] for r in rows])
        stats[key] = {"observations": len(rows),
                      **{name: sum(r[name] for r in rows) for name in
                         ("positive_depth", "inside_raw_image", "inside_network_image", "inside_heatmap_centre_hull")},
                      "raw_uv_min": raw.min(0).tolist(), "raw_uv_max": raw.max(0).tolist(),
                      "network_uv_min": uv.min(0).tolist(), "network_uv_max": uv.max(0).tolist(),
                      "minimum_heatmap_hull_margin_px": min(r["heatmap_centre_hull_margin_px"] for r in rows)}
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU audit unexpectedly initialized CUDA")
    if manifest_path.read_bytes() != manifest_bytes:
        raise RuntimeError("Manifest changed during audit")
    return {
        "schema_version": "unified_C_projection_domain_v1",
        "manifest_path": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "label_path_and_hash_digest": hashes.hexdigest(), "all_label_hashes_verified": True,
        "frames": len(manifest["records"]), "split_counts": dict(Counter(r["split"] for r in manifest["records"])),
        "network_image_size": list(target_size), "heatmap_stride": stride,
        "camera_model": "per-eye stored K, Brown D[k1,k2,p1,p2,k3], optical C in mm; pixel-centre resize",
        "reference_C_definition": "_pose_truth C = shaft local-frame origin, not a visible surface point",
        "label_saved_C_pixel_max_discrepancy_px": label_projection_max_error,
        "no_visibility_filter": True, "no_training_or_data_modified": True, "CPU_threads": 2,
        "cuda_initialized": False, "images_opened": False,
        "all_C_inside_image_and_heatmap_domain": not exceptional,
        "statistics": stats, "outside_domain_observations": exceptional,
        "interpretation": "Domain compatibility check only; in-frame does not imply surface visibility or accurate learned localization. No sample is filtered."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    result = audit(args.manifest, args.config, args.stride)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("frames", "split_counts", "all_C_inside_image_and_heatmap_domain", "label_saved_C_pixel_max_discrepancy_px")}))


if __name__ == "__main__":
    main()
