"""CPU-only audit of saved predictions, with one-factor oracle substitutions.

Reads saved validation values17 and immutable label JSONs. Never runs a neural
network, touches training state, creates images, or changes dataset membership.
Oracle improvements are sensitivity experiments, not additive causal attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_document(path):
    content = path.read_bytes()
    return json.loads(content), digest(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--reference-epoch", type=int, default=29)
    parser.add_argument("--precision-evidence", type=Path,
                        help="Independent source-GPU TF32 FK reconstruction evidence for this exact saved report")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--instrument-json", type=Path)
    source.add_argument("--checkpoint", type=Path,
                        help="Only instrument_model is used; torch.load maps everything to CPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-tolerance-mm", type=float, default=0.001)
    args = parser.parse_args()
    if args.numeric_tolerance_mm <= 0:
        parser.error("Numeric tolerance must be positive")
    inputs = [args.manifest, args.report, args.history, args.instrument_json, args.checkpoint,
              args.precision_evidence]
    if args.output.resolve() in {p.resolve() for p in inputs if p is not None}:
        raise ValueError("The output must not overwrite an input")
    sys.path.insert(0, str(args.code_root.resolve()))
    import numpy as np
    import torch
    from stereo_pose.data import _pose_truth
    from stereo_pose.geometry import FixedKinematics, quaternion_xyzw_to_matrix

    torch.set_num_threads(2)
    manifest, manifest_file_hash = read_document(args.manifest)
    report, report_file_hash = read_document(args.report)
    if report.get("split") != "validation":
        raise ValueError("Expected the saved validation report")
    content = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if digest(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()) != manifest["manifest_sha256"]:
        raise ValueError("Manifest changed after indexing")
    if args.instrument_json:
        instrument, instrument_hash = read_document(args.instrument_json)
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        instrument = checkpoint["instrument_model"]
        instrument_hash = digest(json.dumps(instrument, sort_keys=True).encode())
        del checkpoint
    if instrument["source_scene_sha256"] != manifest["source_scene_sha256"]:
        raise ValueError("Instrument model and labels refer to different source assets")

    records = [r for r in manifest["records"] if r["split"] == "validation"]
    def identity(row):
        return row["sequence_id"], row["frame_index"], row["timestamp_ns"]
    lookup = {identity(row): row for row in records}
    rows = report["frames_detail"]
    if (len(lookup) != len(records) or len(rows) != len(records)
            or len({identity(row) for row in rows}) != len(rows)
            or {identity(row) for row in rows} != set(lookup)):
        raise ValueError("Report must contain each current validation frame exactly once")
    if report.get("frames") != len(rows):
        raise ValueError("Report frame count disagrees with saved predictions")
    categories = sorted({r["category"] for r in rows})
    if len(categories) != 7:
        raise ValueError("Expected all seven validation categories")
    predictions, truths, label_hashes = [], [], []
    for row in rows:
        record = lookup[identity(row)]
        if record["category"] != row["category"]:
            raise ValueError("Category mismatch")
        path = Path(record["label_path"])
        label, label_hash = read_document(path)
        if label_hash != record["label_sha256"]:
            raise ValueError("Immutable label hash mismatch: " + str(path))
        values = np.asarray(row["prediction_values17"], dtype=np.float64)
        if values.shape != (17,) or not np.isfinite(values).all():
            raise ValueError("Malformed saved prediction")
        predictions.append(values)
        truths.append(_pose_truth(label))
        label_hashes.append({"path": str(path), "sha256": label_hash})

    values = torch.tensor(np.stack(predictions), dtype=torch.float64)
    pred_C = torch.stack((values[:, :3], values[:, 7:10]), dim=1)
    pred_R = quaternion_xyzw_to_matrix(torch.stack((values[:, 3:7], values[:, 10:14]), dim=1))
    pred_q = values[:, 14:17]
    true_C = torch.tensor(np.stack([t["C"] for t in truths]), dtype=torch.float64)
    true_R = torch.tensor(np.stack([t["R"] for t in truths]), dtype=torch.float64)
    true_q = torch.tensor(np.stack([t["joints"] for t in truths]), dtype=torch.float64)
    true_points = torch.tensor(np.stack([t["points"] for t in truths]), dtype=torch.float64)
    kinematics = FixedKinematics(instrument).cpu()
    if ((pred_q < kinematics.joint_lower - 1e-6).any()
            or (pred_q > kinematics.joint_upper + 1e-6).any()):
        raise ValueError("Saved prediction exceeds physical joint limits")
    if (not torch.isfinite(true_C).all() or (pred_C[..., 2] <= 0).any()
            or (true_C[..., 2] <= 0).any()):
        raise ValueError("C ray/depth substitutions require finite positive optical depth")
    C_variants = {
        "prediction": pred_C,
        "truth": true_C,
        "truth_depth": pred_C / pred_C[..., 2:] * true_C[..., 2:],
        "truth_ray": true_C / true_C[..., 2:] * pred_C[..., 2:],
    }
    C_descriptions = {
        "prediction": "Keep predicted C unchanged.",
        "truth": "Replace all XYZ components of C with truth.",
        "truth_depth": "C'=Z_true*[X_pred/Z_pred,Y_pred/Z_pred,1]: replace optical Z depth, retain the predicted undistorted normalized ray; XYZ may all change.",
        "truth_ray": "C'=Z_pred*[X_true/Z_true,Y_true/Z_true,1]: replace the undistorted normalized ray, retain predicted optical Z depth.",
    }
    variants = {
        "prediction": ("prediction", False, ()),
        "oracle_C": ("truth", False, ()),
        "oracle_C_depth_only": ("truth_depth", False, ()),
        "oracle_C_ray_only": ("truth_ray", False, ()),
        "oracle_R": ("prediction", True, ()),
        "oracle_wrist_pitch": ("prediction", False, (0,)),
        "oracle_wrist_yaw": ("prediction", False, (1,)),
        "oracle_wrist_pitch_yaw": ("prediction", False, (0, 1)),
        "oracle_jaw_half": ("prediction", False, (2,)),
        "oracle_C_R": ("truth", True, ()),
        "oracle_all_joints": ("prediction", False, (0, 1, 2)),
        "oracle_C_R_wrist": ("truth", True, (0, 1)),
        "oracle_all_state": ("truth", True, (0, 1, 2)),
    }
    points, tip_norms = {}, {}
    with torch.inference_mode():
        for name, (C_mode, replace_R, replace_q) in variants.items():
            q = pred_q.clone()
            for index in replace_q:
                q[:, index] = true_q[:, index]
            points[name] = kinematics(C_variants[C_mode],
                                      true_R if replace_R else pred_R, q)["points"]
            tip_norms[name] = (points[name][:, :, 3:5] - true_points[:, :, 3:5]).norm(dim=-1).numpy()

    baseline_per_frame = tip_norms["prediction"].mean(axis=(1, 2))
    saved_tip = np.asarray([r["tip_mm"] for r in rows])
    reconstruction_delta = float(np.max(np.abs(baseline_per_frame - saved_tip)))
    all_truth_max = float(tip_norms["oracle_all_state"].max())
    if all_truth_max > args.numeric_tolerance_mm:
        raise ValueError("All-truth FK consistency failed: max %.9g mm" % all_truth_max)
    precision_evidence, precision_hash = None, None
    if args.precision_evidence:
        precision_evidence, precision_hash = read_document(args.precision_evidence)
        if (precision_evidence.get("report_sha256") != report_file_hash
                or precision_evidence.get("frames") != len(rows)
                or precision_evidence.get("geometry_code_sha256") != digest(
                    (args.code_root/"stereo_pose/geometry.py").read_bytes())
                or precision_evidence.get("method") != "source_gpu_per_frame_saved_values17_FK_TF32_on_off"
                or precision_evidence["gpu_tf32_true"]["saved_delta_abs_max_mm"] > args.numeric_tolerance_mm
                or abs(precision_evidence["cpu64"]["saved_delta_abs_max_mm"]-reconstruction_delta) > 1e-7):
            raise ValueError("Independent precision evidence does not reproduce this report within the original tolerance")
    if reconstruction_delta > args.numeric_tolerance_mm and precision_evidence is None:
        raise ValueError("Saved-tip max difference %.9g mm requires independent source-GPU precision evidence; tolerance was not relaxed"
                         % reconstruction_delta)

    masks = {category: np.asarray([r["category"] == category for r in rows]) for category in categories}
    def statistics(array):
        return {"mean_mm": float(np.mean(array)), "median_mm": float(np.median(array)),
                "p90_mm": float(np.quantile(array, .9)),
                "left_mean_mm": float(array[:, 0].mean()), "right_mean_mm": float(array[:, 1].mean()),
                "J1_mean_mm": float(array[:, :, 0].mean()), "J2_mean_mm": float(array[:, :, 1].mean())}
    oracle_results = {}
    for name, distances in tip_norms.items():
        per_category = {category: statistics(distances[mask]) for category, mask in masks.items()}
        category_equal_mean = float(np.mean([s["mean_mm"] for s in per_category.values()]))
        reference = float(np.mean([tip_norms["prediction"][m].mean() for m in masks.values()]))
        for category, result in per_category.items():
            base = float(tip_norms["prediction"][masks[category]].mean())
            result["reduction_from_prediction_mm"] = base - result["mean_mm"]
        oracle_results[name] = {"replaced_components": {
            "C": variants[name][0], "C_description": C_descriptions[variants[name][0]],
            "R": variants[name][1],
            "joints": [["wrist_pitch", "wrist_yaw", "jaw_half"][i] for i in variants[name][2]]},
            "all_frames": statistics(distances), "category_equal_mean_mm": category_equal_mean,
            "category_equal_reduction_mm": reference - category_equal_mean,
            "per_category": per_category}

    tip_error = (points["prediction"][:, :, 3:5] - true_points[:, :, 3:5]).numpy()
    C_error = (pred_C - true_C).numpy()
    C_broadcast = np.broadcast_to(C_error[:, :, None], tip_error.shape)
    relative_error = tip_error - C_broadcast
    def vector_decomposition(mask):
        e, c, r = tip_error[mask], C_broadcast[mask], relative_error[mask]
        e_norm, c_norm, r_norm = [np.linalg.norm(x, axis=-1) for x in (e, c, r)]
        valid = (e_norm > 1e-10) & (c_norm > 1e-10)
        cosine = (e * c).sum(axis=-1)[valid] / (e_norm * c_norm)[valid]
        mean_e2, mean_c2, mean_r2 = [float(np.mean(x*x)) for x in (e_norm, c_norm, r_norm)]
        cross = float(np.mean(2 * (c*r).sum(axis=-1)))
        return {"tip_error_mean_mm": float(e_norm.mean()),
                "C_error_mean_mm": float(c_norm.mean()),
                "relative_tip_error_mean_mm": float(r_norm.mean()),
                "C_bias_left_right_xyz_mm": c.mean(axis=(0, 2)).tolist(),
                "C_mae_left_right_xyz_mm": np.abs(c).mean(axis=(0, 2)).tolist(),
                "C_rmse_left_right_xyz_mm": np.sqrt((c*c).mean(axis=(0, 2))).tolist(),
                "tip_bias_left_right_xyz_mm": e.mean(axis=(0, 2)).tolist(),
                "tip_mae_left_right_xyz_mm": np.abs(e).mean(axis=(0, 2)).tolist(),
                "tip_rmse_left_right_xyz_mm": np.sqrt((e*e).mean(axis=(0, 2))).tolist(),
                "J1_mae_left_right_xyz_mm": np.abs(e[:, :, 0]).mean(axis=0).tolist(),
                "J1_rmse_left_right_xyz_mm": np.sqrt((e[:, :, 0]**2).mean(axis=0)).tolist(),
                "J2_mae_left_right_xyz_mm": np.abs(e[:, :, 1]).mean(axis=0).tolist(),
                "J2_rmse_left_right_xyz_mm": np.sqrt((e[:, :, 1]**2).mean(axis=0)).tolist(),
                "tip_C_error_vector_cosine_mean": float(cosine.mean()) if cosine.size else None,
                "tip_mse_mm2": mean_e2, "C_mse_mm2": mean_c2,
                "relative_tip_mse_mm2": mean_r2, "cross_term_mm2": cross,
                "mse_identity_residual_mm2": mean_e2 - mean_c2 - mean_r2 - cross,
                "opposing_C_relative_error_fraction": float(np.mean((c*r).sum(axis=-1) < 0)),
                "meaning": "e_J = e_C + e_relative. Negative cross term indicates cancellation; norm contributions are not additive."}
    decomposition = {"all_frames": vector_decomposition(np.ones(len(rows), dtype=bool)),
                     "per_category": {c: vector_decomposition(m) for c, m in masks.items()}}

    history_comparison = {"reference_epoch": args.reference_epoch,
                          "per_frame_reference_predictions_available": False,
                          "note": "History supports aggregate trends only; no historical per-frame oracle result is invented."}
    history_hash = None
    if args.history:
        history_bytes = args.history.read_bytes()
        history_hash = digest(history_bytes)
        history = [json.loads(line) for line in history_bytes.splitlines() if line.strip()]
        matched = [h for h in history if all(abs(h.get("validation_metrics", {}).get(k, 1e9)-v) < 1e-9
                   for k, v in report["category_equal_mean"].items())]
        history_comparison["report_matching_epochs"] = [h["epoch"]+1 for h in matched]
        wanted = [h for h in history if h["epoch"]+1 == args.reference_epoch or h in matched]
        history_comparison["aggregates"] = [{"epoch": h["epoch"]+1,
            "training_loss": h.get("training_loss"), "validation_loss": h.get("validation_loss"),
            "training_metrics": {k: h.get("training_metrics", {}).get(k) for k in
                ("position_mm", "tip_mm", "wrist_pitch_deg", "wrist_yaw_deg", "jaw_half_deg")},
            "validation_metrics": h.get("validation_metrics", {})} for h in wanted]

    per_frame = []
    for i, row in enumerate(rows):
        per_frame.append({"sequence_id": row["sequence_id"], "frame_index": row["frame_index"],
            "category": row["category"], "timestamp_ns": row["timestamp_ns"],
            "tip_error_mm": {name: float(value[i].mean()) for name, value in tip_norms.items()},
            "tip_error_vectors_mm": tip_error[i].tolist(), "C_error_vectors_mm": C_error[i].tolist()})
    output = {"schema_version": 1, "analysis": "saved_prediction_tip_oracle_cpu",
        "frames": len(rows), "per_category_counts": {c: int(m.sum()) for c, m in masks.items()},
        "coordinate_frame": "each_camera_optical_x_right_y_down_z_forward", "unit": "mm",
        "predictions_recomputed_by_model": False, "training_modified": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "provenance": {"manifest_file_sha256": manifest_file_hash,
            "manifest_identity": manifest["manifest_sha256"] , "report_sha256": report_file_hash,
            "instrument_sha256": instrument_hash, "history_sha256": history_hash,
            "precision_evidence_sha256": precision_hash,
            "script_sha256": digest(Path(__file__).read_bytes()), "labels": label_hashes,
            "geometry_code_sha256": digest((args.code_root/"stereo_pose/geometry.py").read_bytes())},
        "numeric_consistency": {"saved_tip_max_difference_mm": reconstruction_delta,
            "all_truth_tip_max_error_mm": all_truth_max, "tolerance_mm": args.numeric_tolerance_mm,
            "saved_metric_reconstruction": "independently_verified_source_GPU_TF32" if precision_evidence else "CPU_within_tolerance",
            "original_saved_GPU_category_equal_tip_mm": report["category_equal_mean"]["tip_mm"],
            "exported_state_FP64_category_equal_tip_mm": oracle_results["prediction"]["category_equal_mean_mm"],
            "oracle_comparison_baseline": "all variants use exported saved state decoded with CPU FP64 FK",
            "independent_precision_evidence": precision_evidence},
        "oracles": oracle_results, "translation_decomposition": decomposition,
        "history_comparison": history_comparison, "frames_detail": per_frame,
        "interpretation_limits": [
            "Truth substitution is an oracle sensitivity test, not deployable prediction.",
            "Original GPU metrics may include TF32 FK rounding. All oracle reductions use the separately reported FP64 baseline, never mixed with historical GPU metrics.",
            "C depth-only substitution preserves the predicted normalized ray; ray-only substitution preserves predicted optical Z. Neither substitutes rotation or joints.",
            "Substitution effects can compensate or oppose each other; do not sum reductions as causal contributions.",
            "This identifies state components associated with remaining tip error; it alone cannot distinguish dataset distribution from architecture or optimization causes.",
            "Class-equal means match validation reporting; pooled-frame summaries weight longer sequences more."]}
    if output["cuda_initialized"]:
        raise AssertionError("The audit must not initialize CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "frames": len(rows),
        "report_matching_epochs": history_comparison.get("report_matching_epochs", []),
        "category_equal_tip_mm": {k: v["category_equal_mean_mm"] for k, v in oracle_results.items()},
        "cuda_initialized": False}, allow_nan=False))


if __name__ == "__main__":
    main()
