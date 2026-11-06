"""Explicit commands; invoking the module without a mode never trains."""
import argparse
from pathlib import Path
import torch
from .data import build_manifest
from .geometry import FixedKinematics
from .runtime import ROOT, read_json, write_json, train, load_model, evaluate


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="mode", required=True)
    index = subs.add_parser("index")
    index.add_argument("--run-root", action="append", required=True)
    index.add_argument("--output", required=True)
    for command in ("train", "smoke", "trial"):
        p = subs.add_parser(command)
        p.add_argument("--manifest", required=True)
        p.add_argument("--config", default=str(ROOT / "configs/default.json"))
        p.add_argument("--instrument-model", default=str(ROOT / "assets/instrument_model.json"))
        p.add_argument("--output", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--resume")
        p.add_argument("--extend-training", action="store_true", help="Extend epochs in a fresh directory, preserving optimizer state and initial resumed learning rate")
        p.add_argument("--loader-upgrade", action="store_true", help="Allow audited data-loading-only changes when resuming")
    for command in ("validate", "test"):
        p = subs.add_parser(command)
        p.add_argument("--manifest", required=True)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.mode == "index":
        result = build_manifest(args.run_root, args.output)
        print({k: result[k] for k in ("split_counts", "complete", "warnings")})
    elif args.mode in ("train", "smoke", "trial"):
        train(read_json(args.config), read_json(args.instrument_model), read_json(args.manifest), args.output,
              torch.device(args.device), args.resume, args.mode == "smoke", args.mode == "trial", args.loader_upgrade, args.extend_training)
    else:
        device = torch.device(args.device)
        model, saved = load_model(args.checkpoint, device, args.allow_diagnostic)
        manifest = read_json(args.manifest)
        report = evaluate(model, FixedKinematics(saved["instrument_model"]).to(device), manifest,
                          "validation" if args.mode == "validate" else "test", saved["config"], device)
        report["diagnostic_checkpoint"] = saved["diagnostic"]
        write_json(args.output, report)
        print({k: report[k] for k in ("split", "frames", "categories_present", "category_equal_mean")})


if __name__ == "__main__":
    main()
