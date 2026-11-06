"""Native RGB routing, separate branches and causal 17-value contracts."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from stereo_pose.parts import foreground_crops, part_regions
from stereo_pose.model import StereoPoseNet, joint_bounds_from_model

ROOT = Path(__file__).resolve().parents[1]


def inputs(time=3):
    images = torch.randn(1,time,2,3,64,96)
    raw = torch.randint(0,256,(1,time,2,3,128,192),dtype=torch.uint8)
    K = torch.tensor([[90.,0.,47.5],[0.,90.,31.5],[0.,0.,1.]]).expand(1,time,2,3,3)
    D = torch.zeros(1,time,2,5)
    return images,K,D,raw


def model():
    torch.set_num_threads(2)
    config = json.loads((ROOT/'configs/default.json').read_text())['model']
    config.update(feature_guidance=True,foreground_parts=True,roi_pool_size=4,
                  roi_local_encoder=True,part_crop_size=32,part_mask_dilation_px=0)
    instrument = json.loads((ROOT/'assets/instrument_model.json').read_text())
    return StereoPoseNet(config,joint_bounds_from_model(instrument),pretrained=False)


def test_foreground_mask_applied_before_raw_rgb_crop():
    torch.manual_seed(11)
    rgb = torch.randint(0,256,(2,3,64,96),dtype=torch.uint8)
    mask = torch.zeros(2,1,64,96); mask[:,:,20:44,30:66]=1
    centres = torch.tensor([47.5,31.5]).expand(2,2,2)
    extents = torch.tensor([48.,32.]).expand(2,2,2)
    first = foreground_crops(rgb,mask,centres,extents,(64,96),32,dilation_px=0)
    changed = rgb.clone()
    background = ~mask.bool().expand_as(rgb)
    changed[background] = 255-changed[background]
    second = foreground_crops(changed,mask,centres,extents,(64,96),32,dilation_px=0)
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    assert first[:,:,:3].abs().sum()>0
    blank = foreground_crops(rgb,mask*0,centres,extents,(64,96),32)
    assert torch.isfinite(blank).all() and torch.count_nonzero(blank)==0


def test_pixel_centre_mapping_native_and_localization_sizes():
    # Output pixel centres reproduce native RGB at doubled source resolution.
    rgb = torch.arange(16,dtype=torch.uint8).reshape(1,1,4,4).expand(1,3,4,4)*10
    centres = torch.tensor([[[.5,.5]]])
    extents = torch.tensor([[[1.,1.]]])
    crop = foreground_crops(rgb,torch.ones(1,1,4,4),centres,extents,(2,2),16,cutoff=0,dilation_px=0)
    assert crop.shape==(1,1,4,16,16)
    # Grid centre is invariant to pixel-centre resize (2x and 1x conventions).
    same = foreground_crops(rgb,torch.ones(1,1,4,4),torch.tensor([[[1.5,1.5]]]),
                            torch.tensor([[[2.,2.]]]),(4,4),16,cutoff=0,dilation_px=0)
    torch.testing.assert_close(crop,same,atol=0,rtol=0)


def test_roi_hinge_pair_order_does_not_change_boxes():
    uv = torch.tensor([[[30.,40.],[50.,40.],[50.,40.],[65.,35.],[64.,45.]]],requires_grad=True)
    centres,extents = part_regions(uv,(360,640),{})
    other = uv.detach().clone(); other[:,[1,2]]=other[:,[2,1]];other[:,[3,4]]=other[:,[4,3]]
    c2,e2=part_regions(other,(360,640),{})
    torch.testing.assert_close(centres,c2);torch.testing.assert_close(extents,e2)
    assert not centres.requires_grad and not extents.requires_grad
    assert centres.shape==(1,2,2)


def test_parts_require_raw_rgb_and_preserve_causal_contract():
    torch.manual_seed(16)
    net=model().eval();images,K,D,raw=inputs()
    with pytest.raises(ValueError,match='hires_images'):
        net(images,K,D)
    with torch.no_grad():
        prediction,state=net(images,K,D,hires_images=raw)
        expected=net.export_values(prediction)
        stream=[];hidden=None
        for t in range(3):
            p,hidden=net(images[:,t:t+1],K[:,t:t+1],D[:,t:t+1],hidden,raw[:,t:t+1])
            stream.append(net.export_values(p))
        torch.testing.assert_close(expected,torch.cat(stream,1),atol=3e-5,rtol=1e-5)
        changed=raw.clone();changed[:,2]=255-changed[:,2]
        p,_=net(images,K,D,hires_images=changed)
        torch.testing.assert_close(expected[:,:2],net.export_values(p)[:,:2],atol=1e-6,rtol=1e-6)
    assert expected.shape==(1,3,17) and torch.isfinite(expected).all()
    assert state.shape==(3,2,net.temporal.hidden_size)
    assert prediction['part_heatmap_logits'].shape==(1,3,2,2,3,8,8)
    assert prediction['part_roi_centres'].shape==(1,3,2,2,2)
    assert torch.all(prediction['joints']>=net.joint_lower) and torch.all(prediction['joints']<=net.joint_upper)


def test_separate_local_branches_receive_gradients_without_mask_backdoor():
    torch.manual_seed(21)
    net=model().train();images,K,D,raw=inputs(1)
    p,_=net(images,K,D,hires_images=raw)
    (p['joints'].square().mean()+.001*p['part_heatmap_logits'].square().mean()).backward()
    for branch in (net.foreground_parts_net.wrist_branch,net.foreground_parts_net.jaw_branch):
        for name in ('encoder','cross_view','temporal','head'):
            grads=[v.grad for v in getattr(branch,name).parameters() if v.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum() for g in grads)>0,name
    for module in (net.mask_head,net.heatmap_head,net.backbone):
        assert all(v.grad is None or torch.count_nonzero(v.grad)==0 for v in module.parameters())
    for name in ("joint_head", "guided_joint_pool", "guided_joint_temporal"):
        assert not hasattr(net, name)


class FixedMap(nn.Module):
    def __init__(self,value):
        super().__init__();self.register_buffer('value',value)
    def forward(self,x):
        return self.value.expand(x.shape[0],-1,-1,-1)


def test_no_global_image_feature_bypass_into_angle_heads():
    torch.manual_seed(35)
    net=model().eval();images,K,D,raw=inputs(1)
    mask=torch.full((1,1,16,24),-100.);mask[:,:,5:11,8:16]=100.
    heat=torch.full((1,5,16,24),-100.);heat[:,:,8,12]=100.
    net.mask_head=FixedMap(mask);net.heatmap_head=FixedMap(heat)
    with torch.no_grad():
        first,_=net(images,K,D,hires_images=raw)
        # With routing fixed, arbitrary global FPN changes cannot alter q.
        second,_=net(images*-3,K,D,hires_images=raw)
    torch.testing.assert_close(first['joints'],second['joints'],atol=0,rtol=0)
