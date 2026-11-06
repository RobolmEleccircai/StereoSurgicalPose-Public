from types import SimpleNamespace
from unittest.mock import Mock

import torch

from stereo_pose import runtime


def test_validation_loss_uses_training_protocol_and_weights(monkeypatch):
    config = {'window':8, 'image_size':[360,640], 'training':{'amp':False},
              'loss_weights':{'position':1., 'mask':.1}, 'loss_scales':{'position_scale_mm':10.}}
    dataset_factory = Mock(return_value=[0,1,2])
    criterion = Mock(side_effect=[(torch.tensor(2.), {}), (torch.tensor(5.), {})])
    criterion_factory = Mock(return_value=SimpleNamespace(to=lambda device:criterion))
    monkeypatch.setattr(runtime, 'verify_manifest', lambda *args:None)
    monkeypatch.setattr(runtime, 'StereoDataset', dataset_factory)
    monkeypatch.setattr(runtime, 'PoseLoss', criterion_factory)
    monkeypatch.setattr(runtime, 'make_loader', lambda *args,**kwargs:None)
    batches = [{'images':torch.zeros(n,8,2,3,1,1),'K':torch.zeros(n),'D':torch.zeros(n)} for n in (2,1)]
    monkeypatch.setattr(runtime, 'device_batches', lambda *args:iter(batches))
    class Model:
        def eval(self):self.evaluating=True
        def __call__(self,*args):
            assert self.evaluating and not torch.is_grad_enabled()
            return {'C':torch.zeros(1)},None
    kin=SimpleNamespace(model={})
    result=runtime.evaluate_loss(Model(),kin,{},config,torch.device('cpu'))
    assert result['validation_windows']==3
    assert result['validation_loss']==3.  # Window-weighted, not mean of unequal batch means.
    assert dataset_factory.call_args.kwargs['window']==8
    criterion_factory.assert_called_once_with(kin,config['loss_weights'],position_scale_mm=10.)
