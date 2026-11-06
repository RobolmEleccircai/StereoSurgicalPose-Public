"""Read-only acceptance of an explicitly converted unified-C checkpoint.

No optimizer is created, no optimizer step is taken, and no weights are saved.
The optional complete validation uses the existing runtime evaluation functions.
Only small JSON reports are written to --output; input datasets are referenced.

Example:
  python tools/check_unified_c_checkpoint.py --new-run NEW_RUN \
    --source-code FROZEN_SOURCE_CODE --source-run FROZEN_SOURCE_RUN \
    --output NEW_RUN/evidence/checkpoint_acceptance --full-validation
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stereo_pose.data import StereoDataset, collate_samples
from stereo_pose.geometry import FixedKinematics, project_brown
from stereo_pose.losses import PoseLoss
from stereo_pose.model import StereoPoseNet, joint_bounds_from_model
from stereo_pose.runtime import (evaluate, evaluate_loss, forward_batch, identity,
                                 move, read_json, verify_manifest, write_json)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_source_class(root):
    # Load the unchanged source under a distinct package, preserving its relative imports.
    name = "_unified_c_checkpoint_frozen_source"
    spec = importlib.util.spec_from_file_location(name, root / "stereo_pose/__init__.py",
        submodule_search_locations=[str(root / "stereo_pose")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(name + ".model").StereoPoseNet


def numeric_comparison(actual, expected, *, atol, rtol=0.0, unit):
    actual, expected = actual.detach().double().cpu(), expected.detach().double().cpu()
    error = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {"passed": finite and bool(torch.all(error <= atol + rtol * expected.abs())),
            "finite": finite, "maximum_absolute_error": float(error.max()),
            "mean_absolute_error": float(error.mean()), "atol": atol, "rtol": rtol,
            "unit": unit, "actual": actual.tolist(), "expected": expected.tolist()}


def gradient_summary(parameters):
    parameters = list(parameters)
    present = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not present:
        return {"passed": False, "parameters": len(parameters), "gradient_tensors": 0,
                "finite": True, "absolute_sum": 0.0, "maximum_absolute": 0.0}
    # Aggregate on the device, then transfer a compact result once for this group.
    summaries = torch.stack([torch.stack((torch.isfinite(g).all().float(), g.abs().sum(), g.abs().max()))
                             for g in present]).double().cpu()
    finite = bool((summaries[:, 0] == 1).all())
    total, maximum = float(summaries[:, 1].sum()), float(summaries[:, 2].max())
    return {"passed": finite and math.isfinite(total) and total > 0,
            "parameters": len(parameters), "gradient_tensors": len(present),
            "finite": finite, "absolute_sum": total, "maximum_absolute": maximum}


def tensor_gradient_summary(tensor):
    if tensor.grad is None:
        return {"passed": False, "finite": True, "absolute_sum": 0.0}
    grad = tensor.grad.detach().double().cpu()
    finite = bool(torch.isfinite(grad).all())
    total = float(grad.abs().sum())
    return {"passed": finite and total > 0, "finite": finite, "absolute_sum": total}


def metadata_only(saved):
    return {key: saved[key] for key in ("config", "instrument_model", "identity", "epoch")}


def run_checks(args, report):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("This actual-checkpoint check requires an explicitly available CUDA device")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This comparison requires BF16 support; do not silently change precision")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(20261111)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    new_run = args.new_run.expanduser().resolve()
    source_run = args.source_run.expanduser().resolve()
    source_code = args.source_code.expanduser().resolve()
    if new_run == source_run:
        raise ValueError("New and frozen source runs must differ")
    paths = {"new_checkpoint": new_run / "last.pt", "source_checkpoint": source_run / "last.pt",
             "new_manifest": new_run / "manifest.json", "source_manifest": source_run / "manifest.json"}
    hashes = {name: digest(path) for name, path in paths.items()}
    report["provenance"] = {"paths": {k: str(p) for k, p in paths.items()}, "input_sha256": hashes,
                            "source_code": str(source_code), "new_code": str(ROOT)}
    manifest, source_manifest = read_json(paths["new_manifest"]), read_json(paths["source_manifest"])
    if hashes["new_manifest"] != hashes["source_manifest"]:
        raise ValueError("Architecture transition must preserve the entire manifest byte-for-byte")
    expected_validation_frames = sum(r["split"] == "validation" for r in manifest["records"])
    if expected_validation_frames == 0:
        raise ValueError("Validation split is empty")
    saved = torch.load(paths["new_checkpoint"], map_location="cpu", weights_only=True)
    if saved.get("diagnostic") or saved.get("trial") or "architecture_transition" not in saved:
        raise ValueError("Expected an explicitly converted full-data checkpoint")
    new_meta = metadata_only(saved)
    config, instrument = new_meta["config"], new_meta["instrument_model"]
    verify_manifest(manifest, instrument)
    if identity(config, instrument, manifest) != new_meta["identity"]:
        raise ValueError("New checkpoint identity differs from current code/config/dataset")
    model = StereoPoseNet(config["model"], joint_bounds_from_model(instrument), pretrained=False)
    model.load_state_dict(saved["model"], strict=True)
    transition = saved["architecture_transition"]
    if transition["source_checkpoint_sha256"] != hashes["source_checkpoint"]:
        raise ValueError("Frozen source checkpoint differs from the conversion provenance")
    del saved
    gc.collect()
    saved = torch.load(paths["source_checkpoint"], map_location="cpu", weights_only=True)
    source_meta = metadata_only(saved)
    for path, expected in source_meta["identity"]["code"].items():
        if digest(source_code / path) != expected:
            raise ValueError("Frozen source code mismatch: " + path)
    if source_meta["identity"]["manifest"] != source_manifest["manifest_sha256"]:
        raise ValueError("Source checkpoint manifest identity differs")
    if source_meta["identity"]["instrument"] != new_meta["identity"]["instrument"]:
        raise ValueError("Source and new instrument geometry differ")
    if source_meta["epoch"] != new_meta["epoch"]:
        raise ValueError("Acceptance must run before the converted checkpoint resumes optimization")
    for field in ("image_size", "window", "loss_weights", "loss_scales"):
        if config.get(field) != source_meta["config"].get(field):
            raise ValueError("Transition unexpectedly changed " + field)
    if not (config["training"]["amp"] and source_meta["config"]["training"]["amp"]):
        raise ValueError("The requested comparison requires BF16/AMP in both saved configurations")
    source_class = load_source_class(source_code)
    source = source_class(source_meta["config"]["model"], joint_bounds_from_model(instrument), pretrained=False)
    source.load_state_dict(saved["model"], strict=True)
    del saved
    gc.collect()
    model, source = model.to(device).eval(), source.to(device).eval()
    if config["model"].get("channels_last", False):
        model.to(memory_format=torch.channels_last)
    if source_meta["config"]["model"].get("channels_last", False):
        source.to(memory_format=torch.channels_last)
    report["execution"] = {"device": str(device), "gpu": torch.cuda.get_device_name(device),
                           "autocast_dtype": "bfloat16", "cpu_threads": args.cpu_threads,
                           "tf32_matmul": True, "tf32_cudnn": True, "cudnn_benchmark": True,
                           "comparison_mode": "eval, same real frame and device, no hidden-state history",
                           "source_epoch_one_based": source_meta["epoch"] + 1,
                           "new_checkpoint_epoch_one_based": new_meta["epoch"] + 1}

    dataset = StereoDataset(manifest, "validation", window=1, image_size=tuple(config["image_size"]),
                            include_hires=True)
    first = dataset[0]
    batch = move(collate_samples([first]), device)
    with torch.no_grad(), amp():
        old_pred, _ = forward_batch(source, batch)
        new_pred, _ = forward_batch(model, batch)
    with torch.no_grad():
        old_uv = old_pred["landmark_uv"][..., 0, :]
        projection = project_brown(new_pred["C"].double().unsqueeze(-2),
                                   batch["K"].double(), batch["D"].double()).squeeze(-2)
        checks = {
            "rotation_matrix_preserved": numeric_comparison(new_pred["R"], old_pred["R"], atol=3e-3, rtol=2e-3, unit="matrix_entry"),
            "joint_angles_preserved": numeric_comparison(new_pred["joints"], old_pred["joints"], atol=2e-4, rtol=0, unit="rad"),
            "C_detector_uv_preserved": numeric_comparison(new_pred["c_uv"], old_uv, atol=1e-3, rtol=0, unit="pixel"),
            "optical_Z_preserved": numeric_comparison(new_pred["C"][..., 2:3], old_pred["C"][..., 2:3], atol=.05, rtol=.002, unit="mm"),
            "C_reprojects_to_its_only_uv": numeric_comparison(projection, new_pred["c_uv"], atol=1e-4, rtol=0, unit="pixel"),
            "C_uv_is_landmark_channel_zero": numeric_comparison(new_pred["c_uv"], new_pred["landmark_uv"][..., 0, :], atol=0, rtol=0, unit="pixel")}
        relative = new_pred["R"].double().transpose(-1, -2) @ old_pred["R"].double()
        angle = torch.rad2deg(torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)))
        checks["rotation_matrix_preserved"]["relative_rotation_deg"] = angle.cpu().tolist()
    report["first_validation_frame"] = {
        "metadata": {key: first["metadata"][0][key] for key in ("sequence_id", "frame_index", "trajectory_group_id", "category")},
        "checks": checks, "passed": all(row["passed"] for row in checks.values()),
        "tolerance_rationale": "The heads use changed GEMM shapes under BF16. UV shares the unchanged detector; rotation and depth tolerate small arithmetic differences. Decoder is independently reprojected in FP64 after its float32 C output.",
        "C_XY_preservation_not_expected": "C_XY now derives from the existing detected landmark, so it intentionally need not equal the old independent ray-XY output."}
    print(json.dumps({"phase": "checkpoint_comparison", "passed": report["first_validation_frame"]["passed"],
                      "maximum_absolute_differences": {k: v["maximum_absolute_error"] for k, v in checks.items()}}), flush=True)
    del source, old_pred, new_pred, batch, first, dataset
    gc.collect()
    torch.cuda.empty_cache()

    # A genuine contiguous training window, with the normal input and PoseLoss interface.
    train_set = StereoDataset(manifest, "train", window=8, image_size=tuple(config["image_size"]), include_hires=True)
    if not 0 <= args.train_window_index < len(train_set):
        raise ValueError("Requested training window index does not exist")
    example = train_set[args.train_window_index]
    batch = move(collate_samples([example]), device)
    model.train()
    model.zero_grad(set_to_none=True)
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    kinematics = FixedKinematics(instrument).to(device)
    criterion = PoseLoss(kinematics, config["loss_weights"], **config.get("loss_scales", {})).to(device)
    with amp():
        pred, _ = forward_batch(model, batch)
    pred = {key: value.float() for key, value in pred.items()}
    pred["c_uv"].retain_grad()
    pred["c_depth"].retain_grad()
    loss, components = criterion(pred, batch)
    if not torch.isfinite(loss):
        raise FloatingPointError("Real-window backward loss is nonfinite")
    loss.backward()
    groups = {
        "C_heatmap_head": model.heatmap_head,
        "left_depth_head": model.depth_heads[0], "right_depth_head": model.depth_heads[1],
        "left_rotation_head": model.rotation_heads[0], "right_rotation_head": model.rotation_heads[1],
        "wrist_parts_branch": model.foreground_parts_net.wrist_branch,
        "jaw_parts_branch": model.foreground_parts_net.jaw_branch}
    gradients = {name: gradient_summary(module.parameters()) for name, module in groups.items()}
    gradients["C_uv_tensor"] = tensor_gradient_summary(pred["c_uv"])
    gradients["C_depth_tensor"] = tensor_gradient_summary(pred["c_depth"])
    weights_unchanged = all(parameter._version == versions[name] for name, parameter in model.named_parameters())
    window_meta = example["metadata"]
    report["real_training_window_backward"] = {
        "window_index": args.train_window_index, "frames": len(window_meta),
        "sequence_id": window_meta[0]["sequence_id"], "trajectory_group_id": window_meta[0]["trajectory_group_id"],
        "frame_indices": [m["frame_index"] for m in window_meta],
        "timestamps_ns": [m["timestamp_ns"] for m in window_meta],
        "single_trajectory": len({m["trajectory_group_id"] for m in window_meta}) == 1,
        "loss": float(loss.detach().cpu()),
        "loss_components": {k: float(v.detach().cpu()) for k, v in components.items()},
        "gradients": gradients, "model_mode": "train", "online_augmentation_applied": False,
        "optimizer_constructed": False, "optimizer_steps": 0, "weights_saved": False,
        "parameter_version_counters_unchanged": weights_unchanged,
        "passed": weights_unchanged and all(g["passed"] for g in gradients.values())}
    print(json.dumps({"phase": "real_window_backward", "passed": report["real_training_window_backward"]["passed"],
                      "loss": report["real_training_window_backward"]["loss"],
                      "gradient_absolute_sums": {k: v["absolute_sum"] for k, v in gradients.items()}}), flush=True)
    model.zero_grad(set_to_none=True)
    model.eval()
    del batch, example, train_set, pred, loss, components, criterion
    gc.collect()
    torch.cuda.empty_cache()

    accepted = report["first_validation_frame"]["passed"] and report["real_training_window_backward"]["passed"]
    if args.full_validation and accepted:
        started = time.perf_counter()
        validation = evaluate(model, kinematics, manifest, "validation", config, device)
        if validation["frames"] != expected_validation_frames:
            raise AssertionError("Full validation did not evaluate every fixed frame")
        validation["checkpoint_sha256"] = hashes["new_checkpoint"]
        validation["checkpoint_epoch_one_based"] = new_meta["epoch"] + 1
        validation["meaning"] = "Unified-C architecture before any new optimizer step; fixed full validation split"
        write_json(args.output / "validation_initial.json", validation)
        initial_loss = evaluate_loss(model, kinematics, manifest, config, device)
        initial_loss.update(checkpoint_sha256=hashes["new_checkpoint"],
                            checkpoint_epoch_one_based=new_meta["epoch"] + 1,
                            split="validation", source_pairs=expected_validation_frames, optimizer_steps_in_this_tool=0)
        write_json(args.output / "initial_loss.json", initial_loss)
        report["full_validation"] = {
            "completed": True, "frames": validation["frames"],
            "category_equal_mean": validation["category_equal_mean"],
            "loss": initial_loss, "seconds": time.perf_counter() - started,
            "validation_report": str(args.output / "validation_initial.json"),
            "loss_report": str(args.output / "initial_loss.json")}
        print(json.dumps({"phase": "full_validation", "frames": expected_validation_frames,
                          "validation_loss": initial_loss["validation_loss"],
                          "category_equal_mean": validation["category_equal_mean"]}), flush=True)
    else:
        report["full_validation"] = {"completed": False,
            "reason": "not_requested" if not args.full_validation else "acceptance_checks_failed"}
    final_hashes = {name: digest(path) for name, path in paths.items()}
    report["all_input_checkpoints_and_manifests_unchanged"] = final_hashes == hashes
    report["passed"] = accepted and final_hashes == hashes
    if args.full_validation:
        report["passed"] = report["passed"] and report["full_validation"]["completed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-run", "--run", dest="new_run", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for small JSON reports only")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--train-window-index", type=int, default=0)
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "unified_C_actual_checkpoint_acceptance_v1", "passed": False,
              "no_optimizer_steps": True, "no_weights_written": True, "no_dataset_changes": True}
    started = time.perf_counter()
    try:
        run_checks(args, report)
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "checkpoint_acceptance.json", report)
        raise
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.output / "checkpoint_acceptance.json", report)
    print(json.dumps({"passed": report["passed"], "report": str(args.output / "checkpoint_acceptance.json")}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
