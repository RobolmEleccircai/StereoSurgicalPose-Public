"""Create an isolated historical architecture workspace from verified source files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('baseline', 'guided', 'local4', 'foreground_parts', 'legacy_c'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage = ROOT / 'experiments/stages' / args.stage
    source = json.loads((stage / 'source.json').read_text())
    for filename, expected in source['sha256'].items():
        if hashlib.sha256((stage / filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Archived stage was modified: ' + filename)
    if hashlib.sha256((stage / 'config.json').read_bytes()).hexdigest() != source['config_template_sha256']:
        raise ValueError('Archived configuration template was modified')
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError('Historical workspaces must be outside the active code directory')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    for folder in ('stereo_pose',):
        shutil.copytree(stage / folder, output / folder)
    for folder in ('assets', 'reference'):
        shutil.copytree(ROOT / folder, output / folder, ignore=shutil.ignore_patterns('__pycache__'))
    (output / 'configs').mkdir()
    shutil.copy2(stage / 'config.json', output / 'configs/default.json')
    shutil.copy2(ROOT / 'run.sh', output / 'run.sh')
    (output / 'tools').mkdir()
    for filename in ('reproduction.py', 'prepare_run.py', 'evaluate_checkpoint.py', 'plot_training.py'):
        shutil.copy2(ROOT / 'tools' / filename, output / 'tools' / filename)
    shutil.copy2(stage / 'source.json', output / 'source.json')
    print(json.dumps({'stage': args.stage, 'workspace': str(output), 'training_started': False}))


if __name__ == '__main__':
    main()
