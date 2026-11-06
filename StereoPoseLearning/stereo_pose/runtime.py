"""Training and causal evaluation. Test data never selects a checkpoint."""
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import StereoDataset, collate_samples, _continuous
from .geometry import FixedKinematics, project_brown
from .losses import PoseLoss, rotation_error_degrees
from .model import StereoPoseNet, joint_bounds_from_model
from .loading import make_loader, device_batches, category_weighted_loss
from .augmentation import augment_batch
from .training_policy import (validate_training_policy, training_epochs, should_early_stop,
    learning_rate_factor, restore_loss_bests, update_loss_bests, atomic_checkpoint_link,
    update_epoch_archive)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).expanduser().read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def forward_batch(model, batch, hidden=None):
    """Pass only image data and individual camera calibration into the model."""
    if getattr(model, "config", {}).get("foreground_parts", False):
        if "hires_images" not in batch:
            raise ValueError("Foreground-parts model requires original RGB frames")
        return model(batch["images"], batch["K"], batch["D"], hidden,
                     hires_images=batch["hires_images"])
    return model(batch["images"], batch["K"], batch["D"], hidden)


def initialize_parts(model, checkpoint, signature, output):
    """Warm-start compatible local4 weights; train new local branches afresh."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("diagnostic") or saved.get("trial"):
        raise ValueError("Cannot warm-start from an incomplete-data/diagnostic model")
    for key in ("instrument", "manifest"):
        if saved["identity"][key] != signature[key]:
            raise ValueError("Warm-start asset/dataset identity differs: " + key)
    if not model.config.get("foreground_parts") or saved["config"]["model"].get("foreground_parts"):
        raise ValueError("Expected local4-to-foreground-parts transfer")
    incompatible = model.load_state_dict(saved["model"], strict=False)
    if incompatible.unexpected_keys or any(not k.startswith("foreground_parts_net.") for k in incompatible.missing_keys):
        raise ValueError("Unexpected architecture transfer mismatch: " + str(incompatible))
    identical = all(torch.equal(model.state_dict()[k].detach().cpu(), value) for k, value in saved["model"].items())
    if not identical:
        raise AssertionError("Common weights did not transfer exactly")
    write_json(output/"initialization.json", {
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "source_epoch": saved["epoch"]+1, "common_weights_bit_identical": True,
        "new_parameters": incompatible.missing_keys, "optimizer_reset": True,
        "protocol": "Warm-start local4 best; new wrist/jaw branches. Epoch counts are additional fine-tuning, not matched-budget ablation.",
        "source_identity": saved["identity"]})


def initialize_guided(model, checkpoint, signature, output):
    """Transfer only compatible learned weights; optimizer/epoch start afresh."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('diagnostic') or saved.get('trial'):
        raise ValueError('Cannot initialize from diagnostic/partial-data checkpoint')
    for key in ('instrument', 'manifest'):
        if saved['identity'][key] != signature[key]:
            raise ValueError('Warm start asset/dataset identity mismatch: '+key)
    if not model.guided or saved['config']['model'].get('feature_guidance', False):
        raise ValueError('Expected legacy-to-guided architecture transfer')
    incompatible = model.load_state_dict(saved['model'], strict=False)
    allowed = ('guided_regions.', 'guided_pose_pool.', 'guided_joint_pool.', 'guided_joint_temporal.')
    if incompatible.unexpected_keys or any(not k.startswith(allowed) for k in incompatible.missing_keys):
        raise ValueError('Unexpected checkpoint mismatch: '+str(incompatible))
    model.guided_joint_temporal.load_state_dict(model.temporal.state_dict())
    identical = all(torch.equal(model.state_dict()[k].detach().cpu(), v) for k,v in saved['model'].items())
    if not identical: raise AssertionError('Common weights did not transfer exactly')
    write_json(output/'initialization.json', {
        'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'source_epoch': saved['epoch']+1, 'common_weights_bit_identical': identical,
        'new_parameters': incompatible.missing_keys, 'joint_gru_initialized_from_pose_gru': True,
        'optimizer_reset': True, 'source_identity': saved['identity']})


def identity(config, instrument, manifest):
    code = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "stereo_pose").glob("*.py"))}
    return {"config": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "instrument": hashlib.sha256(json.dumps(instrument, sort_keys=True).encode()).hexdigest(),
            "manifest": manifest["manifest_sha256"], "code": code}


def move(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def verify_manifest(manifest, instrument):
    content = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != manifest["manifest_sha256"]:
        raise ValueError("Manifest changed after indexing; rebuild a fresh snapshot")
    if manifest["source_scene_sha256"] != instrument["source_scene_sha256"]:
        raise ValueError("The instrument geometry and dataset came from different USD assets")


def amp_context(device, enabled):
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_model(checkpoint, device, allow_diagnostic=False):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("diagnostic", False) and not allow_diagnostic:
        raise ValueError("This is a diagnostic smoke checkpoint, not a trained model. Use --allow-diagnostic only for pipeline checks.")
    model = StereoPoseNet(saved["config"]["model"], joint_bounds_from_model(saved["instrument_model"]), pretrained=False)
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), saved


def require_complete(manifest, config):
    expected = config["expected_pairs"]
    sealed = manifest.get('training_supplement', {}).get('snapshot_sealed', False)
    if not manifest["hashes_verified"] or not (manifest["complete"] or sealed) or manifest["trajectory_split_counts"] != expected:
        raise ValueError(f"Full training requires the completed 2100-pair manifest {expected}; observed {manifest['trajectory_split_counts']}. Use smoke to check partial data.")
    if manifest["isolated_diagnostic_pairs"]:
        raise ValueError("Full training manifest must contain production sequences, not isolated diagnostic captures.")


def validate_training_append(base, current, instrument):
    """Permit only extra training records; preserve every original record."""
    verify_manifest(base, instrument)
    verify_manifest(current, instrument)
    if not base['complete'] or not base['hashes_verified'] or not current['hashes_verified']:
        raise ValueError('Training append requires a verified original dataset and verified new frames')
    old = {r['label_path']: r for r in base['records']}
    new = {r['label_path']: r for r in current['records']}
    if len(old) != len(base['records']) or len(new) != len(current['records']):
        raise ValueError('Duplicate frame paths')
    if any(new.get(path) != record for path, record in old.items()):
        raise ValueError('An original frame, label, or split was changed')
    added = [r for path, r in new.items() if path not in old]
    if not added or any(r['split'] != 'train' or r['isolated_diagnostic'] for r in added):
        raise ValueError('Only new production training records may be appended')
    old_groups = {r['trajectory_group_id'] for r in old.values()}
    if any(r['trajectory_group_id'] in old_groups for r in added):
        raise ValueError('Supplement must use independently identified trajectories')
    if current.get('temporal_gaps'):
        raise ValueError('Supplement snapshot contains a temporal gap')
    metadata = current.get('training_supplement', {})
    if not metadata.get('snapshot_sealed') or metadata.get('parent_manifest_sha256') != base['manifest_sha256']:
        raise ValueError('Supplement snapshot provenance differs')
    return {'added_pairs':len(added), 'heldout_records_unchanged':True,
            'original_records_unchanged':True, 'source_generation_complete':False,
            'base_manifest_sha256':base['manifest_sha256'],
            'new_manifest_sha256':current['manifest_sha256']}


@torch.no_grad()
def evaluate_loss(model, kinematics, manifest, config, device):
    """Validation loss with the same windows, targets and weights as training."""
    verify_manifest(manifest, kinematics.model)
    dataset = StereoDataset(manifest, 'validation', window=config['window'], image_size=tuple(config['image_size']), include_hires=config.get('model', {}).get('foreground_parts', False))
    if config['training'].get('shared_frame_cache',False):dataset.preload_shared()
    if not len(dataset):
        raise ValueError('No validation windows')
    criterion = PoseLoss(kinematics, config['loss_weights'], **config.get('loss_scales', {})).to(device)
    model.eval()
    loader = make_loader(dataset, config['training'], shuffle=False)
    total, count = 0., 0
    for batch in device_batches(loader, device, config['training'].get('async_transfer', False)):
        with amp_context(device, config['training']['amp']):
            pred, _ = forward_batch(model, batch)
        pred = {k:v.float() for k,v in pred.items()}
        loss, _ = criterion(pred, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite validation loss')
        size = batch['images'].shape[0]
        total += float(loss.cpu())*size
        count += size
    return {'validation_loss':total/count, 'validation_windows':count,
            'validation_loss_protocol':'same_training_window_and_PoseLoss_eval_mode'}


@torch.no_grad()
def evaluate(model, kinematics, manifest, split, config, device, limit=None):
    """Stream each sequence; state resets at a sequence change or missing frame."""
    verify_manifest(manifest, kinematics.model)
    dataset = StereoDataset(manifest, split, window=1, image_size=tuple(config["image_size"]), include_hires=config.get("model", {}).get("foreground_parts", False))
    if config['training'].get('shared_frame_cache',False):dataset.preload_shared()
    if not len(dataset):
        raise ValueError(f"No actual {split} frames")
    model.eval()
    rows, hidden, previous = [], None, None
    for index in range(min(len(dataset), limit or len(dataset))):
        record = dataset.records[index]
        if previous is None or not _continuous(previous, record):
            hidden = None
        batch = move(collate_samples([dataset[index]]), device)
        with amp_context(device, config["training"]["amp"]):
            pred, hidden = forward_batch(model, batch, hidden)
        geom = kinematics(pred["C"], pred["R"], pred["joints"])
        delta = (geom["points"] - batch["points"]).norm(dim=-1)
        uv = project_brown(geom["points"], batch["K"], batch["D"])
        uv_true = project_brown(batch["points"], batch["K"], batch["D"])
        valid = (geom["points"][..., 2] > 0) & (batch["points"][..., 2] > 0) & torch.isfinite(uv).all(-1)
        pixel_error = (uv - uv_true).norm(dim=-1)
        d = geom["d"]
        cosine = (d * batch["R"][..., :, 2]).sum(-1).clamp(-1, 1)
        scalar = lambda x: float(x.float().mean().cpu())
        # Evaluate the unique C ray in high precision, independently of AMP/TF32.
        c_uv = project_brown(pred["C"].double().unsqueeze(-2),
                             batch["K"].double(), batch["D"].double()).squeeze(-2)
        c_truth_uv = project_brown(batch["points"][..., :1, :].double(),
                                   batch["K"].double(), batch["D"].double()).squeeze(-2)
        c_detected_uv = pred["c_uv"].double()
        row = {"sequence_id": record["sequence_id"], "frame_index": record["frame_index"],
               "category": record["category"], "timestamp_ns": record["timestamp_ns"],
               "outside_train_joint_range": record.get("outside_train_joint_range", [False]*3),
               "position_mm": scalar(delta[..., 0]), "hinge_mm": scalar(delta[..., 1:3]),
               "tip_mm": scalar(delta[..., 3:5]), "midpoint_mm": scalar(delta[..., 5]),
               "axis_deg": scalar(torch.rad2deg(torch.acos(cosine))),
               "rotation_deg": scalar(rotation_error_degrees(pred["R"], batch["R"])),
               "jaw_rotation_deg": scalar(rotation_error_degrees(geom["R_jaws"], batch["R_jaws"])),
               "joint_mae_deg": scalar(torch.rad2deg((pred["joints"]-batch["joints"]).abs())),
               "projection_px_network_resolution": scalar(pixel_error[valid]) if valid.any() else None,
               "positive_depth_fraction": scalar((geom["points"][..., 2] > 0).float()),
               "overexposed": record["overexposed"]}
        row.update({
            "c_detector_error_px": scalar((c_detected_uv-c_truth_uv).norm(dim=-1)),
            "c_reprojection_error_px": scalar((c_uv-c_truth_uv).norm(dim=-1)),
            "c_decoder_detector_gap_px": scalar((c_uv-c_detected_uv).norm(dim=-1)),
            "c_depth_error_mm": scalar((pred["C"][..., 2]-batch["points"][..., 0, 2]).abs()),
        })
        row['prediction_values17'] = model.export_values(pred)[0,0].cpu().tolist()
        angle_errors = torch.rad2deg((pred['joints']-batch['joints']).abs())[0,0].cpu().tolist()
        row.update(dict(zip(('wrist_pitch_deg','wrist_yaw_deg','jaw_half_deg'),angle_errors)))
        row['truth_joint_angles_deg'] = torch.rad2deg(batch['joints'][0,0]).cpu().tolist()
        if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise FloatingPointError("Nonfinite evaluation result")
        rows.append(row)
        previous = record
    metrics = sorted({k for row in rows for k, v in row.items() if isinstance(v, float)})
    def summarize(selected):
        return {k: float(np.mean([r[k] for r in selected if r[k] is not None]))
                for k in metrics if any(r[k] is not None for r in selected)}
    categories = {c: summarize([r for r in rows if r["category"] == c]) for c in sorted({r["category"] for r in rows})}
    strata = {}
    for name, selected in (("within_observed_train_joint_range", [r for r in rows if not any(r["outside_train_joint_range"])]),
                           ("outside_observed_train_joint_range", [r for r in rows if any(r["outside_train_joint_range"])])):
        strata[name] = {"frames": len(selected), "metrics": summarize(selected) if selected else {}}
    equal = {k: float(np.mean([v[k] for v in categories.values() if k in v])) for k in metrics}
    return {"split": split, "frames": len(rows), "categories_present": len(categories),
            "category_equal_mean": equal, "per_category": categories, "generalization": strata,
            "complete_split_evaluation": limit is None and manifest.get('split_complete', {}).get(split, manifest["complete"]),
            "network_image_size": config["image_size"], "frames_detail": rows}


def train(config, instrument, manifest, output, device, resume=None, smoke=False, trial=False, loader_upgrade=False, extend_training=False, append_train_manifest=None):
    verify_manifest(manifest, instrument)
    settings = config["training"]
    torch.set_num_threads(settings.get('cpu_threads',4))
    if device.type=='cuda':
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    seed = settings["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    validate_training_policy(settings)
    if not smoke and not trial:
        require_complete(manifest, config)
    output = Path(output).expanduser().resolve()
    append_evidence = None
    if append_train_manifest:
        if not resume or smoke or trial or loader_upgrade or extend_training or (output.exists() and any(output.iterdir())):
            raise ValueError('Appending training data requires a previous checkpoint and a fresh run')
        base_manifest = read_json(append_train_manifest)
        append_evidence = validate_training_append(base_manifest, manifest, instrument)
    if extend_training and (not resume or loader_upgrade or (output.exists() and any(output.iterdir()))):
        raise ValueError('Training extension needs a checkpoint and a fresh output directory')
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Choose a fresh run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    signature = identity(config, instrument, manifest)
    dataset = StereoDataset(manifest, "train", window=config["window"], image_size=tuple(config["image_size"]), include_hires=config.get("model", {}).get("foreground_parts", False))
    if config['training'].get('shared_frame_cache',False):dataset.preload_shared()
    if not len(dataset):
        raise ValueError(f"No real contiguous {config['window']}-frame training windows; do not fabricate adjacency.")
    initialization = settings.get('initialize_from')
    model = StereoPoseNet(config["model"], joint_bounds_from_model(instrument),
                          pretrained=False if resume or initialization else None).to(device)
    if initialization and not resume:
        (initialize_parts if config["model"].get("foreground_parts") else initialize_guided)(model, initialization, signature, output)
    if config["model"].get("channels_last",False):model.to(memory_format=torch.channels_last)
    kin = FixedKinematics(instrument).to(device)
    criterion = PoseLoss(kin, config["loss_weights"], **config.get("loss_scales", {})).to(device)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": list(model.backbone.parameters()), "lr": settings["backbone_lr"]},
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": settings["head_lr"]}],
        weight_decay=settings["weight_decay"], fused=settings.get("fused_optimizer",False))
    epochs = settings["epochs"]
    def rate(epoch):
        return learning_rate_factor(epoch, settings)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and settings["amp"] and not torch.cuda.is_bf16_supported())
    start, best, stale = 0, float("inf"), 0
    best_tip = float('inf')
    loss_bests = restore_loss_bests()
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        same_identity = saved["identity"] == signature
        if append_train_manifest:
            old_config = json.loads(json.dumps(saved['config']))
            new_config = json.loads(json.dumps(config))
            for c in (old_config, new_config):
                c.pop('expected_pairs', None)
                c.pop('notes', None)
                for key in ('accumulate','comparison_report','comparison_history','comparison_label'):
                    c['training'].pop(key, None)
            changed = {k for k in set(saved['identity']['code']) | set(signature['code'])
                       if saved['identity']['code'].get(k) != signature['code'].get(k)}
            same_identity = (old_config == new_config
                and changed <= {'stereo_pose/runtime.py','stereo_pose/__main__.py'}
                and saved['identity']['instrument'] == signature['instrument']
                and saved['identity']['manifest'] == base_manifest['manifest_sha256'])
            if same_identity:
                write_json(output/'training_data_addition.json', dict(append_evidence,
                    checkpoint=str(resume), source_epoch=saved['epoch']+1,
                    source_checkpoint_sha256=hashlib.sha256(Path(resume).read_bytes()).hexdigest(),
                    optimizer_scheduler_rng_preserved=True, old_identity=saved['identity'],
                    new_identity=signature, architecture_and_loss_unchanged=True,
                    old_accumulate=saved['config']['training']['accumulate'],
                    new_accumulate=settings['accumulate']))
        if extend_training:
            old_config = json.loads(json.dumps(saved['config']))
            new_config = json.loads(json.dumps(config))
            for c in (old_config, new_config):
                for key in ('epochs','patience','extension_reset_lr','decay_anchor_epoch','decay_anchor_factor','workers','pin_memory','prefetch_factor','persistent_workers','async_transfer'):
                    c['training'].pop(key, None)
            changed = {k for k in set(saved['identity']['code']) | set(signature['code'])
                       if saved['identity']['code'].get(k) != signature['code'].get(k)}
            factors = [g['lr']/g['initial_lr'] for g in saved['optimizer']['param_groups']]
            same_identity = (old_config == new_config and changed <= {'stereo_pose/runtime.py','stereo_pose/__main__.py','stereo_pose/loading.py'}
                and saved['identity']['instrument'] == signature['instrument']
                and saved['identity']['manifest'] == signature['manifest']
                and epochs >= saved['config']['training']['epochs']
                and epochs > saved['epoch']+1
                and settings['decay_anchor_epoch'] == saved['epoch']+1
                and (all(abs(f-settings['decay_anchor_factor']) < 1e-12 and f > 0 for f in factors)
                     or (settings.get('extension_reset_lr',False) and 0 < settings['decay_anchor_factor'] <= 1)))
            if same_identity:
                write_json(output/'training_extension.json', {'source_checkpoint':str(resume),
                    'source_epoch':saved['epoch']+1,'total_epochs':epochs,'old_identity':saved['identity'],
                    'new_identity':signature,'old_lr_factors':factors,
                    'new_lr_factor':settings['decay_anchor_factor'],
                    'old_patience':saved['config']['training']['patience'],'new_patience':settings['patience'],
                    'schedule':'Cosine decay; optional explicit learning rate restart; optimizer moments and RNG preserved'})
        if loader_upgrade:
            old_config = json.loads(json.dumps(saved['config']))
            new_config = json.loads(json.dumps(config))
            for c in (old_config, new_config):
                for key in ('workers','pin_memory','prefetch_factor','persistent_workers','async_transfer'):
                    c['training'].pop(key, None)
            allowed = {'stereo_pose/runtime.py', 'stereo_pose/loading.py', 'stereo_pose/__main__.py'}
            old_code, new_code = saved['identity']['code'], signature['code']
            changed = {key for key in set(old_code)|set(new_code) if old_code.get(key)!=new_code.get(key)}
            same_identity = (old_config == new_config and changed <= allowed
                             and saved['identity']['instrument'] == signature['instrument']
                             and saved['identity']['manifest'] == signature['manifest'])
            if same_identity:
                write_json(output/'loader_upgrade.json', {'checkpoint': str(resume), 'completed_epochs': saved['epoch']+1,
                    'old_identity': saved['identity'], 'new_identity': signature, 'changed_code': sorted(changed),
                    'old_loading': saved['config']['training'], 'new_loading': settings})
        if not same_identity or bool(saved["diagnostic"]) != smoke or saved.get("trial", False) != trial:
            raise ValueError("Resume code/config/asset/data or diagnostic status differs; start a new run.")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"]); scaler.load_state_dict(saved["scaler"])
        start, best, stale = saved["epoch"] + 1, saved["best"], saved["stale"]
        best_tip = saved.get('best_tip', float('inf'))
        loss_bests = restore_loss_bests(saved)
        if append_train_manifest:
            # Keep the original baseline immutable and keep all new epochs separate.
            preserve_names = ['best_train.pt','best_val.pt','training_loss_best.json','validation_loss_best.json']
            if settings.get('save_metric_best_checkpoints', True):
                preserve_names += ['best.pt','best_tip.pt','validation_best.json','validation_best_tip.json']
            for name in preserve_names:
                source = Path(resume).parent/name
                if source.exists():shutil.copy2(source, output/name)
            source_rows = [json.loads(s) for s in (Path(resume).parent/'history.jsonl').read_text().splitlines() if s.strip()]
            baseline = next(r for r in source_rows if r['epoch'] == saved['epoch'])
            write_json(output/'baseline_before_addition.json', baseline)
        if extend_training:
            if settings.get('extension_reset_lr',False):
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr']*settings['decay_anchor_factor']
                scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
            # The latest checkpoint may be worse than the retained best checkpoint.
            if settings.get('save_metric_best_checkpoints', True):
                source_best = Path(resume).parent/'best.pt'
                if not source_best.exists():
                    raise FileNotFoundError('Cannot extend without preserving the previous best checkpoint')
                shutil.copy2(source_best, output/'best.pt')
            for name in ('best_train.pt','best_val.pt','training_loss_best.json','validation_loss_best.json'):
                source_loss_best = Path(resume).parent/name
                if source_loss_best.exists():
                    shutil.copy2(source_loss_best, output/name)
            source_history = Path(resume).parent/'history.jsonl'
            if source_history.exists():
                rows = [json.loads(line) for line in source_history.read_text().splitlines() if line.strip()]
                (output/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch'] < start))
            supplement = Path(resume).parent/'validation_loss.jsonl'
            if supplement.exists():
                rows = [json.loads(line) for line in supplement.read_text().splitlines() if line.strip()]
                (output/'validation_loss.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch'] < start))
        torch.set_rng_state(saved["rng_torch"])
        nr = saved["rng_numpy"]
        np.random.set_state((nr[0], np.asarray(nr[1], dtype=np.uint32), *nr[2:]))
        random.setstate(saved["rng_python"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
        weights_equal = all(torch.equal(value.detach().cpu(), saved['model'][key])
                            for key,value in model.state_dict().items())
        if not weights_equal:
            raise RuntimeError('Restored weights differ from checkpoint')
        optimizer_steps = [float(state['step']) for state in optimizer.state.values() if 'step' in state]
        write_json(output/'resume_validation.json', {'checkpoint': str(resume), 'weights_bit_identical': weights_equal,
            'completed_epochs': start, 'next_epoch': start+1, 'optimizer_steps_min': min(optimizer_steps),
            'optimizer_steps_max': max(optimizer_steps), 'scheduler_last_epoch': scheduler.last_epoch,
            'learning_rates': [g['lr'] for g in optimizer.param_groups], 'rng_states_restored': True})
    loader = make_loader(dataset, settings, shuffle=True)
    balanced_sampler = loader.batch_sampler if settings.get('trajectory_balanced', False) else None
    if balanced_sampler is not None:
        write_json(output/'sampling.json', balanced_sampler.coverage_report())
    write_json(output / "config.json", config)
    write_json(output / "identity.json", signature)
    report = {"diagnostic": smoke, "trial": trial, "training_windows": len(dataset),
              "parameters": sum(p.numel() for p in model.parameters()), "gradient_groups": {}}
    software = {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "timm", "numpy", "scipy", "Pillow")}
    software.update({"cuda_runtime": torch.version.cuda, "device": str(device),
                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                     "backbone_weight_id": config["model"]["backbone"]})
    write_json(output / "software.json", software)
    weight_hash = hashlib.sha256()
    for key, value in model.backbone.state_dict().items():
        weight_hash.update(key.encode()); weight_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    write_json(output / ("resumed_backbone.json" if resume else "initial_backbone.json"), {"sha256_state_tensors": weight_hash.hexdigest(), "resumed": bool(resume)})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history_path = output / "history.jsonl"
    print(json.dumps({"event": "training_started", "trial": trial, "epochs": settings["epochs"],
                      "training_windows": len(dataset), "device": str(device)}), flush=True)
    for epoch in training_epochs(start, settings, smoke=smoke):
        epoch_started = time.perf_counter()
        if balanced_sampler is not None:
            balanced_sampler.set_epoch(epoch)
            write_json(output/'sampling.json', balanced_sampler.coverage_report())
        model.train(); optimizer.zero_grad(set_to_none=True)
        steps = min(len(loader), settings["accumulate"]) if smoke else len(loader)
        totals = []
        component_sums = {}; sample_count=0; loss_mass=0.0
        for step, batch in enumerate(device_batches(loader, device, settings.get('async_transfer', False))):
            if step >= steps:
                break
            batch = augment_batch(batch, config.get("augmentation", {}))
            with amp_context(device, settings["amp"]):
                pred, _ = forward_batch(model, batch)
            # Geometry and powers in the distortion model remain float32.
            pred = {k: v.float() for k, v in pred.items()}
            batch_count=batch["images"].shape[0]; sample_count+=batch_count
            category_balanced = balanced_sampler is not None and balanced_sampler.category_balanced
            if category_balanced:
                loss, metrics, metric_mass = category_weighted_loss(criterion, pred, batch, balanced_sampler)
            else:
                loss, metrics = criterion(pred, batch)
                metric_mass = float(batch_count)
            loss_mass += metric_mass
            for key, value in metrics.items():
                component_sums[key] = component_sums.get(key, 0) + value.detach()*metric_mass
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; checkpoint not marked successful")
            group_start = (step // settings["accumulate"]) * settings["accumulate"]
            divisor = min(settings["accumulate"], steps-group_start)
            # The sampler emits one window per trajectory across a complete
            # logical update. Its category weights sum to one across 4+4+1
            # microbatches; adding a trajectory does not increase its class's
            # total optimization weight.
            backward_loss = (loss * metric_mass if category_balanced else
                             loss * batch_count / balanced_sampler.logical_batch_size
                             if balanced_sampler is not None else loss/divisor)
            scaler.scale(backward_loss).backward()
            if (step+1) % settings["accumulate"] == 0 or step+1 == steps:
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad_norm"], error_if_nonfinite=True)
                if smoke:
                    names = ["backbone", "fusion", "temporal", "depth_heads", "rotation_heads",
                             "mask_head", "heatmap_head", "guided_regions", "guided_pose_pool",
                             "foreground_parts_net"]
                    for name in names:
                        values = [p.grad.detach().float().abs().sum() for p in getattr(model, name).parameters() if p.grad is not None]
                        total = float(torch.stack(values).sum().cpu()) if values else 0.
                        if not math.isfinite(total) or total <= 0:
                            raise AssertionError(f"No finite gradient through {name}")
                        report["gradient_groups"][name] = total
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            totals.append(float(loss.detach().cpu())*metric_mass)
            if (step+1) % 20 == 0:
                progress_row = {"event": "progress", "epoch": epoch+1, "step": step+1,
                                  "steps": steps, "mean_loss": float(sum(totals)/loss_mass),
                                  "elapsed_epoch_seconds": time.perf_counter()-epoch_started}
                with (output/'progress.jsonl').open('a') as progress_file:
                    progress_file.write(json.dumps(progress_row)+'\n')
                print(json.dumps(progress_row), flush=True)
        validation = evaluate(model, kin, manifest, "validation", config, device, limit=2 if smoke else None)
        validation_loss = evaluate_loss(model, kin, manifest, config, device) if not smoke else {}
        if not smoke and not trial and validation["categories_present"] != 7:
            raise ValueError("Validation is missing categories")
        tip_score = validation['category_equal_mean']['tip_mm']
        score = validation['category_equal_mean'][settings.get('selection_metric', 'tip_mm')]
        improved = score < best
        tip_improved = tip_score < best_tip
        best_tip = min(best_tip, tip_score)
        best, stale = (score, 0) if improved else (best, stale+1)
        epoch_training_loss = float(sum(totals)/loss_mass)
        loss_bests, loss_best_updates = update_loss_bests(
            loss_bests, epoch_training_loss, validation_loss.get('validation_loss'), epoch=epoch+1)
        scheduler.step()
        nr = np.random.get_state()
        saved = {"schema_version": 1, "model": model.state_dict(), "config": config, "instrument_model": instrument,
                 "identity": signature, "diagnostic": smoke, "trial": trial, "epoch": epoch, "best": best, "stale": stale,
                 "best_tip": best_tip, **loss_bests,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "rng_torch": torch.get_rng_state(), "rng_numpy": (nr[0], nr[1].tolist(), *nr[2:]), "rng_python": random.getstate(),
                 "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        temp = output / "checkpoint.tmp"
        torch.save(saved, temp); temp.replace(output / "last.pt")
        update_epoch_archive(output/'last.pt', epoch+1, settings.get('keep_epoch_checkpoints'))
        if improved and not smoke and settings.get('save_metric_best_checkpoints', True):
            atomic_checkpoint_link(output/'last.pt', output/'best.pt')
            write_json(output/'validation_best.json', validation)
        if tip_improved and not smoke and settings.get('save_metric_best_checkpoints', True):
            atomic_checkpoint_link(output/'last.pt', output/'best_tip.pt')
            write_json(output/'validation_best_tip.json', validation)
        if not smoke:
            for filename in loss_best_updates:
                atomic_checkpoint_link(output/'last.pt', output/filename)
                if filename == 'best_train.pt':
                    write_json(output/'training_loss_best.json', {
                        'epoch': epoch+1, 'selection_metric': 'epoch_training_loss',
                        'training_loss': epoch_training_loss,
                        'scope': 'category_balanced_complete_training_epoch',
                        'checkpoint': filename, 'manifest_sha256': manifest['manifest_sha256']})
                else:
                    write_json(output/'validation_loss_best.json', dict(validation_loss,
                        epoch=epoch+1, selection_metric='validation_loss',
                        scope='complete_validation_loss_on_current_manifest',
                        checkpoint=filename, manifest_sha256=manifest['manifest_sha256']))
        summary = {"epoch": epoch, "training_loss": epoch_training_loss, "validation_tip_mm_equal_category": tip_score,
                   "selection_metric": settings.get('selection_metric', 'tip_mm'), 'selection_score':score,
                   "diagnostic": smoke, "trial": trial, "optimizer_steps_this_epoch": math.ceil(steps/settings["accumulate"]),
                   **loss_bests, "loss_best_checkpoints_updated": list(loss_best_updates) if not smoke else []}
        if balanced_sampler is not None:
            summary['sampling'] = balanced_sampler.coverage_report()
            summary['sampled_windows'] = sample_count
        summary['epoch_seconds'] = time.perf_counter()-epoch_started
        summary['training_metrics'] = {k:float((v/loss_mass).cpu()) for k,v in component_sums.items()}
        summary['training_loss_normalization_mass'] = loss_mass
        summary['validation_metrics'] = validation['category_equal_mean']
        summary.update(validation_loss)
        with history_path.open("a") as file:
            file.write(json.dumps(summary)+"\n")
        print(json.dumps(summary), flush=True)
        write_json(output / "validation_latest.json", validation)
        from tools.plot_training import plot
        plot(output, [output/'progress.jsonl'])
        if trial:
            report["status"] = "trial_running"
            report["latest_epoch"] = epoch
            write_json(output / "report.json", report)
        if (output/'STOP_AFTER_EPOCH').exists():
            report['status'] = 'stopped_after_checkpoint'
            report['completed_epochs'] = epoch+1
            write_json(output/'report.json', report)
            return report
        if should_early_stop(stale, settings['patience']):
            break
    report["peak_allocated_gpu_mb"] = torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else 0
    report["status"] = "smoke_passed_not_trained" if smoke else ("partial_data_trial_completed" if trial else "training_completed")
    write_json(output / "report.json", report)
    return report
