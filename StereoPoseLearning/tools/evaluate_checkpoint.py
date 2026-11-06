"""Evaluate a fixed checkpoint with provenance; never update or select weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from stereo_pose.geometry import FixedKinematics
from stereo_pose.runtime import evaluate, evaluate_loss, load_model, read_json, write_json, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--with-validation-loss', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new result filename')
    if args.threads < 1:
        raise ValueError('CPU threads must be positive')
    if args.with_validation_loss and args.split != 'validation':
        raise ValueError('Validation loss is only defined for --split validation')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.time()
    # One open descriptor fixes the inode even if a running trainer replaces last.pt.
    with args.checkpoint.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        handle.seek(0)
        model, saved = load_model(handle, torch.device(args.device))
    manifest = read_json(args.manifest)
    verify_manifest(manifest, saved["instrument_model"])
    kin = FixedKinematics(saved['instrument_model']).to(args.device)
    report = evaluate(model, kin, manifest, args.split, saved['config'], torch.device(args.device))
    if args.with_validation_loss:
        report.update(evaluate_loss(model, kin, manifest, saved['config'], torch.device(args.device)))
    report.update(checkpoint_epoch=saved['epoch'] + 1, checkpoint_sha256=digest,
                  manifest_sha256=manifest['manifest_sha256'], device=args.device,
                  cpu_threads=args.threads, elapsed_seconds=time.time() - started,
                  checkpoint_selection='explicit caller selection; no score-based selection here',
                  checkpoint_manifest_sha256=saved['identity']['manifest'])
    write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ('checkpoint_epoch', 'frames', 'device', 'category_equal_mean')}))


if __name__ == '__main__':
    main()
