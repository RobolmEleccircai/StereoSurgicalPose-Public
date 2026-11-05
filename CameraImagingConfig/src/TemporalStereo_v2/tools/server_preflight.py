"""Audit a relocated server bundle and native dependencies; never start rendering."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import read, write, sha, source_hash, load_machine, validate_settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('machine', 'settings', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--worker', action='store_true'); a = p.parse_args()
    if a.output.exists(): raise ValueError('Use a new preflight report path')
    try:
        m = load_machine(a.machine); c = read(a.settings); validate_settings(c)
        isaac = Path(m['isaac_root']); version = (isaac/'VERSION').read_text().strip()
        if not version.startswith('5.1.0'): raise ValueError('This release requires Isaac Sim 5.1.0')
        if platform.system() != 'Linux' or platform.machine() != 'x86_64':
            raise ValueError('This transfer profile is verified for Linux x86_64')
        if not a.worker:
            usd = list((isaac/'extscache').glob('omni.usd.libs-*'))
            if len(usd) != 1: raise ValueError('Expected one bundled USD extension')
            env = os.environ.copy()
            for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[key] = '1'
            env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(usd[0]), m['deps_root']])
            env['LD_LIBRARY_PATH'] = os.pathsep.join([str(usd[0]/'bin'), env.get('LD_LIBRARY_PATH', '')])
            subprocess.run([str(isaac/'python.sh'), str(Path(__file__).resolve()),
                '--machine', str(a.machine.resolve()), '--settings', str(a.settings.resolve()),
                '--output', str(a.output.resolve()), '--worker'], env=env, check=True)
            return
        expected = read(ROOT/'deployment/runtime_versions.json')
        if list(sys.version_info[:2]) != expected['python_major_minor']: raise ValueError('Python ABI mismatch')
        versions = {name: importlib.metadata.version(name) for name in expected['packages']}
        if versions != expected['packages']: raise ValueError('Dependency version mismatch: '+json.dumps(versions))
        import cv2
        import numpy as np
        from PIL import ImageFont
        from runtime import dependency_audit, Scene
        from safety import Safety, make_shape
        from live_preview import preview_font
        import trimesh
        if cv2.__version__ != expected['opencv_module_version']: raise ValueError('Bundled OpenCV version mismatch')
        bundle = Path(m['asset_root']); manifest = read(bundle/'bundle_manifest.json')
        if not manifest['bundle_files']: raise ValueError('Empty asset manifest')
        for name, checksum in manifest['bundle_files'].items():
            path = (bundle/name).resolve()
            if not path.is_relative_to(bundle) or not path.is_file() or sha(path) != checksum:
                raise ValueError('Missing or changed bundle file: '+name)
        audit = dependency_audit(m, c)
        for item in audit['files']:
            path = Path(item['path']).resolve()
            if not (path.is_relative_to(bundle) or path.is_relative_to(isaac)):
                raise ValueError('Dependency escaped the transferred bundle and SDK: '+str(path))
        if audit['asset_overrides']: raise ValueError('Bundle still needs runtime network asset overrides')
        scene = Scene(m, c); safety = Safety(scene)
        if not any(x.group == 'psm' for x in safety.moving) or not any(x.group == 'ecm' for x in safety.moving):
            raise ValueError('Missing PSM or ECM collision geometry')
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        shape1 = make_shape('native_check_a', '', 'anatomy', mesh.vertices, mesh.faces)
        shape2 = make_shape('native_check_b', '', 'anatomy', mesh.vertices, mesh.faces)
        if Safety.pair(shape1, shape2)['safe']: raise ValueError('Native FCL collision query failed')
        font = preview_font(m); ImageFont.truetype(str(font), 25).getbbox('钳口 XYZ')
        output = Path(m['output_root']); output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='preflight_', dir=output) as tmp:
            path = str(Path(tmp)/'codec.mp4')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (128, 72))
            if not writer.isOpened(): raise ValueError('MP4 encoder unavailable')
            for i in (0, 1): writer.write(np.full((72, 128, 3), i*100, np.uint8))
            writer.release(); cap = cv2.VideoCapture(path)
            ok = round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2 and abs(cap.get(cv2.CAP_PROP_FPS)-60) < 1e-6
            cap.release()
            if not ok: raise ValueError('MP4 readback failed')
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True).stdout.strip()
        if not gpu: raise ValueError('No NVIDIA GPU reported')
        report = {'passed': True, 'scope': 'relocated assets, CPU/native dependencies, geometry, font and MP4; no GPU rendering',
            'new_rgb_renders': 0, 'full_generation_started': False, 'server_render_validation_pending': True,
            'machine_role': m['role'], 'isaac_version': version, 'versions': versions, 'opencv': cv2.__version__,
            'source_sha256': source_hash(), 'settings_sha256': sha(a.settings), 'gpu_inventory': gpu,
            'bundle_files_checked': len(manifest['bundle_files']), 'active_dependency_files': len(audit['files']),
            'unresolved': audit['unresolved'], 'anatomy_meshes': len(safety.static),
            'moving_meshes': len(safety.moving), 'contact_surfaces': len(safety.contact_surfaces),
            'font_sha256': sha(font), 'output_free_bytes': shutil.disk_usage(output).free,
            'expected_sequences': c['sequence_count'],
            'expected_pairs': sum(s['frame_count'] for s in c['sequence_manifest']) if 'sequence_manifest' in c else c['sequence_count']*c['frames_per_sequence'],
            'next_step': 'Check renderer/GPU compatibility on the actual server before starting the dataset job.'}
        write(a.output, report); print(json.dumps(report, ensure_ascii=False, indent=2))
    except BaseException as error:
        if not a.output.exists(): write(a.output, {'passed': False, 'error': str(error), 'new_rgb_renders': 0, 'full_generation_started': False})
        raise


if __name__ == '__main__': main()
