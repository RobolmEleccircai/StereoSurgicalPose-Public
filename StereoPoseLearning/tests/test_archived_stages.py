"""Archived architecture stages are runnable snapshots, not configuration labels."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('stage', ['baseline','guided','local4','foreground_parts','legacy_c'])
def test_archived_source_identity_and_stereo_output(stage):
    folder=ROOT/'experiments/stages'/stage
    record=json.loads((folder/'source.json').read_text())
    for name,sha in record['sha256'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((folder/'config.json').read_bytes()).hexdigest()==record['config_template_sha256']
    name='_test_archived_'+stage
    spec=importlib.util.spec_from_file_location(name,folder/'stereo_pose/__init__.py',
        submodule_search_locations=[str(folder/'stereo_pose')])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    module=importlib.import_module(name+'.model')
    config=json.loads((folder/'config.json').read_text())['model']
    instrument=json.loads((ROOT/'assets/instrument_model.json').read_text())
    torch.set_num_threads(2)
    model=module.StereoPoseNet(config,module.joint_bounds_from_model(instrument),pretrained=False).eval()
    image=torch.randn(1,2,2,3,64,96)
    K=torch.tensor([[90.,0.,47.5],[0.,90.,31.5],[0.,0.,1.]]).expand(1,2,2,3,3)
    D=torch.zeros(1,2,2,5)
    extra={'hires_images':torch.randint(0,256,(1,2,2,3,128,192),dtype=torch.uint8)} if config.get('foreground_parts') else {}
    with torch.no_grad():
        pred,_=model(image,K,D,**extra)
        values=model.export_values(pred)
    assert values.shape==(1,2,17) and torch.isfinite(values).all()
