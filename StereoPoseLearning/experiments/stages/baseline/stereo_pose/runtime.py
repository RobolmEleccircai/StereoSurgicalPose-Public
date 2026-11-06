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
from .loading import make_loader, device_batches


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).expanduser().read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


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
    if not manifest["hashes_verified"] or not manifest["complete"] or manifest["trajectory_split_counts"] != expected:
        raise ValueError(f"Full training requires the completed 2100-pair manifest {expected}; observed {manifest['trajectory_split_counts']}. Use smoke to check partial data.")
    if manifest["isolated_diagnostic_pairs"]:
        raise ValueError("Full training manifest must contain production sequences, not isolated diagnostic captures.")


@torch.no_grad()
def evaluate_loss(model, kinematics, manifest, config, device):
    """Validation loss with the same windows, targets and weights as training."""
    verify_manifest(manifest, kinematics.model)
    dataset = StereoDataset(manifest, 'validation', window=config['window'], image_size=tuple(config['image_size']))
    if config['training'].get('shared_frame_cache',False):dataset.preload_shared()
    if not len(dataset):
        raise ValueError('No validation windows')
    criterion = PoseLoss(kinematics, config['loss_weights'], **config.get('loss_scales', {})).to(device)
    model.eval()
    loader = make_loader(dataset, config['training'], shuffle=False)
    total, count = 0., 0
    for batch in device_batches(loader, device, config['training'].get('async_transfer', False)):
        with amp_context(device, config['training']['amp']):
            pred, _ = model(batch['images'], batch['K'], batch['D'])
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
    dataset = StereoDataset(manifest, split, window=1, image_size=tuple(config["image_size"]))
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
            pred, hidden = model(batch["images"], batch["K"], batch["D"], hidden)
        geom = kinematics(pred["C"], pred["R"], pred["joints"])
        delta = (geom["points"] - batch["points"]).norm(dim=-1)
        uv = project_brown(geom["points"], batch["K"], batch["D"])
        uv_true = project_brown(batch["points"], batch["K"], batch["D"])
        valid = (geom["points"][..., 2] > 0) & (batch["points"][..., 2] > 0) & torch.isfinite(uv).all(-1)
        pixel_error = (uv - uv_true).norm(dim=-1)
        d = geom["d"]
        cosine = (d * batch["R"][..., :, 2]).sum(-1).clamp(-1, 1)
        scalar = lambda x: float(x.float().mean().cpu())
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
        row['prediction_values17'] = model.export_values(pred)[0,0].cpu().tolist()
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
            "complete_split_evaluation": limit is None and manifest["complete"],
            "network_image_size": config["image_size"], "frames_detail": rows}


def train(config, instrument, manifest, output, device, resume=None, smoke=False, trial=False, loader_upgrade=False, extend_training=False):
    verify_manifest(manifest, instrument)
    settings = config["training"]
    torch.set_num_threads(settings.get('cpu_threads',4))
    if device.type=='cuda':
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    seed = settings["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if not isinstance(settings['epochs'], int) or settings['epochs'] < 1:
        raise ValueError('Epoch count must be a positive integer')
    if not smoke and not trial:
        require_complete(manifest, config)
    output = Path(output).expanduser().resolve()
    if extend_training and (not resume or loader_upgrade or (output.exists() and any(output.iterdir()))):
        raise ValueError('Training extension needs a checkpoint and a fresh output directory')
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Choose a fresh run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    signature = identity(config, instrument, manifest)
    dataset = StereoDataset(manifest, "train", window=config["window"], image_size=tuple(config["image_size"]))
    if config['training'].get('shared_frame_cache',False):dataset.preload_shared()
    if not len(dataset):
        raise ValueError(f"No real contiguous {config['window']}-frame training windows; do not fabricate adjacency.")
    model = StereoPoseNet(config["model"], joint_bounds_from_model(instrument), pretrained=False if resume else None).to(device)
    if config["model"].get("channels_last",False):model.to(memory_format=torch.channels_last)
    kin = FixedKinematics(instrument).to(device)
    criterion = PoseLoss(kin, config["loss_weights"], **config.get("loss_scales", {})).to(device)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": list(model.backbone.parameters()), "lr": settings["backbone_lr"]},
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": settings["head_lr"]}],
        weight_decay=settings["weight_decay"], fused=settings.get("fused_optimizer",False))
    epochs, warmup = settings["epochs"], settings["warmup_epochs"]
    def rate(epoch):
        if 'decay_anchor_epoch' in settings:
            anchor = settings['decay_anchor_epoch']
            progress = max(0, min(1, (epoch-anchor)/max(epochs-anchor, 1)))
            return settings['decay_anchor_factor'] * .5 * (1 + math.cos(math.pi*progress))
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        return .5 * (1 + math.cos(math.pi * (epoch-warmup) / max(epochs-warmup, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and settings["amp"] and not torch.cuda.is_bf16_supported())
    start, best, stale = 0, float("inf"), 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        same_identity = saved["identity"] == signature
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
        if extend_training:
            if settings.get('extension_reset_lr',False):
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr']*settings['decay_anchor_factor']
                scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
            # The latest checkpoint may be worse than the retained best checkpoint.
            source_best = Path(resume).parent/'best.pt'
            if not source_best.exists():
                raise FileNotFoundError('Cannot extend without preserving the previous best checkpoint')
            shutil.copy2(source_best, output/'best.pt')
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
    for epoch in range(start, start + 1 if smoke else epochs):
        epoch_started = time.perf_counter()
        model.train(); optimizer.zero_grad(set_to_none=True)
        steps = min(len(loader), settings["accumulate"]) if smoke else len(loader)
        totals = []
        component_sums = {}; sample_count=0
        for step, batch in enumerate(device_batches(loader, device, settings.get('async_transfer', False))):
            if step >= steps:
                break
            with amp_context(device, settings["amp"]):
                pred, _ = model(batch["images"], batch["K"], batch["D"])
            # Geometry and powers in the distortion model remain float32.
            pred = {k: v.float() for k, v in pred.items()}
            loss, metrics = criterion(pred, batch)
            batch_count=batch["images"].shape[0]; sample_count+=batch_count
            for key, value in metrics.items():
                component_sums[key] = component_sums.get(key, 0) + value.detach()*batch_count
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; checkpoint not marked successful")
            group_start = (step // settings["accumulate"]) * settings["accumulate"]
            divisor = min(settings["accumulate"], steps-group_start)
            scaler.scale(loss/divisor).backward()
            if (step+1) % settings["accumulate"] == 0 or step+1 == steps:
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad_norm"], error_if_nonfinite=True)
                if smoke:
                    for name in ("backbone", "fusion", "temporal", "pose_heads", "joint_head", "mask_head", "heatmap_head"):
                        values = [p.grad.detach().float().abs().sum() for p in getattr(model, name).parameters() if p.grad is not None]
                        total = float(torch.stack(values).sum().cpu()) if values else 0.
                        if not math.isfinite(total) or total <= 0:
                            raise AssertionError(f"No finite gradient through {name}")
                        report["gradient_groups"][name] = total
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            totals.extend([float(loss.detach().cpu())]*batch_count)
            if (step+1) % 20 == 0:
                progress_row = {"event": "progress", "epoch": epoch+1, "step": step+1,
                                  "steps": steps, "mean_loss": float(np.mean(totals)),
                                  "elapsed_epoch_seconds": time.perf_counter()-epoch_started}
                with (output/'progress.jsonl').open('a') as progress_file:
                    progress_file.write(json.dumps(progress_row)+'\n')
                print(json.dumps(progress_row), flush=True)
        validation = evaluate(model, kin, manifest, "validation", config, device, limit=2 if smoke else None)
        validation_loss = evaluate_loss(model, kin, manifest, config, device) if not smoke else {}
        if not smoke and not trial and validation["categories_present"] != 7:
            raise ValueError("Validation is missing categories")
        score = validation["category_equal_mean"]["tip_mm"]
        improved = score < best
        best, stale = (score, 0) if improved else (best, stale+1)
        scheduler.step()
        nr = np.random.get_state()
        saved = {"schema_version": 1, "model": model.state_dict(), "config": config, "instrument_model": instrument,
                 "identity": signature, "diagnostic": smoke, "trial": trial, "epoch": epoch, "best": best, "stale": stale,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "rng_torch": torch.get_rng_state(), "rng_numpy": (nr[0], nr[1].tolist(), *nr[2:]), "rng_python": random.getstate(),
                 "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        temp = output / "checkpoint.tmp"
        torch.save(saved, temp); temp.replace(output / "last.pt")
        archive = output/'checkpoints'
        archive.mkdir(exist_ok=True)
        epoch_checkpoint = archive/f'epoch_{epoch+1:03d}.pt'
        epoch_checkpoint.unlink(missing_ok=True)
        epoch_checkpoint.hardlink_to(output/'last.pt')
        keep=settings.get('keep_epoch_checkpoints')
        if keep is not None:
            for old in sorted(archive.glob('epoch_*.pt'))[:-int(keep)]:old.unlink()
        if improved and not smoke:
            torch.save(saved, temp); temp.replace(output / "best.pt")
        summary = {"epoch": epoch, "training_loss": float(np.mean(totals)), "validation_tip_mm_equal_category": score,
                   "diagnostic": smoke, "trial": trial, "optimizer_steps_this_epoch": math.ceil(steps/settings["accumulate"])}
        summary['epoch_seconds'] = time.perf_counter()-epoch_started
        summary['training_metrics'] = {k:float((v/sample_count).cpu()) for k,v in component_sums.items()}
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
        if stale >= settings["patience"]:
            break
    report["peak_allocated_gpu_mb"] = torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else 0
    report["status"] = "smoke_passed_not_trained" if smoke else ("partial_data_trial_completed" if trial else "training_completed")
    write_json(output / "report.json", report)
    return report
