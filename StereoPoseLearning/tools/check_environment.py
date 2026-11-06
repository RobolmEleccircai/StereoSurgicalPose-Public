"""Verify the model environment does not borrow packages from another project."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import site
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
a = p.parse_args()
prefix = Path(sys.prefix).resolve()
modules = {}
for name in ("torch", "torchvision", "timm", "numpy", "scipy", "PIL", "cv2", "yaml", "pytest"):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(prefix):
        raise RuntimeError(f"{name} imported from outside this environment: {path}")
    modules[name] = {"file": str(path), "version": getattr(module, "__version__", None)}
if site.ENABLE_USER_SITE:
    raise RuntimeError("User-site is enabled; activate the project Conda environment with PYTHONNOUSERSITE=1")
result = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
if result.returncode:
    raise RuntimeError(result.stdout + result.stderr)
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is unavailable")
report = {"passed": True, "python": sys.version, "executable": sys.executable,
          "prefix": str(prefix), "user_site_enabled": site.ENABLE_USER_SITE,
          "pip_check": result.stdout.strip(), "modules": modules,
          "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda}
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k:v for k,v in report.items() if k != "modules"}, indent=2))
