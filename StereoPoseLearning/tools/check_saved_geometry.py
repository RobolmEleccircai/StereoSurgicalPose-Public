"""Check actual saved state targets against the differentiable CAD decoder."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from stereo_pose.data import _pose_truth, project_points
from stereo_pose.geometry import FixedKinematics, project_brown
from stereo_pose.runtime import ROOT, read_json, write_json, verify_manifest

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
m = read_json(a.manifest); instrument = read_json(ROOT / "assets/instrument_model.json")
verify_manifest(m, instrument)
fk = FixedKinematics(instrument).double()
max_point, max_rotation, max_projection = 0., 0., 0.
for record in m["records"]:
    label = read_json(record["label_path"])
    true = _pose_truth(label)
    geom = fk(*[torch.tensor(true[k], dtype=torch.float64) for k in ("C", "R", "joints")])
    max_point = max(max_point, float(np.max(np.abs(geom["points"].numpy()-true["points"]))))
    max_rotation = max(max_rotation, float(np.max(np.abs(geom["R_jaws"].numpy()-true["R_jaws"]))))
    for i, eye in enumerate(("left", "right")):
        K = np.asarray(label["eyes"][eye]["K"]); D = np.asarray(label["eyes"][eye]["D"])
        uv, front = project_points(true["points"][i], K, D)
        got = project_brown(torch.tensor(true["points"][i]), torch.tensor(K), torch.tensor(D)).numpy()
        max_projection = max(max_projection, float(np.max(np.abs(got[front]-uv[front]))))
report = {"frames_checked": len(m["records"]), "max_point_component_error_mm": max_point,
          "max_rotation_component_error": max_rotation, "independent_projection_error_px": max_projection,
          "scope": "Saved truth conversion and numerical consistency, not image-based prediction accuracy"}
report["passed"] = max_point < 1e-3 and max_rotation < 1e-5 and max_projection < 1e-6
write_json(a.output, report)
print(json.dumps(report))
if not report["passed"]:
    raise SystemExit(1)
