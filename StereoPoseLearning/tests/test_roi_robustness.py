"""ROI augmentation keeps temporal/geometry contracts; evaluation is unchanged."""
import torch
import pytest

from stereo_pose.parts import jitter_part_regions, ForegroundParts
from stereo_pose.losses import local_heatmap_targets


def test_roi_jitter_is_bounded_and_shared_across_time():
    torch.manual_seed(41)
    batch, time = 3, 8
    centres = torch.arange(batch*time*2*2*2, dtype=torch.float32).reshape(-1, 2, 2)
    extents = torch.linspace(20, 80, centres.numel()).reshape_as(centres)
    c, e = jitter_part_regions(centres, extents, (360, 640), batch, time, 1., .03, True)
    offset = (c-centres).reshape(batch, time, 2, 2, 2)
    factor = (e/extents).reshape(batch, time, 2, 2, 2)
    assert offset.abs().max() <= 1.00001
    assert factor.min() >= .97 and factor.max() <= 1.03
    torch.testing.assert_close(offset, offset[:, :1].expand_as(offset), atol=2e-5, rtol=0)
    torch.testing.assert_close(factor, factor[:, :1].expand_as(factor), atol=1e-6, rtol=0)
    torch.testing.assert_close(factor[..., 0], factor[..., 1], atol=1e-6, rtol=0)
    assert not torch.equal(offset[0, 0, 0], offset[0, 0, 1])


def test_roi_jitter_off_or_eval_preserves_outputs_and_rng():
    centres = torch.randn(12, 2, 2)
    extents = torch.ones_like(centres)*25
    for bound, scale, training in ((0., 0., True), (1., .03, False)):
        state = torch.get_rng_state().clone()
        c, e = jitter_part_regions(centres, extents, (360, 640), 2, 3, bound, scale, training)
        assert c is centres and e is extents
        assert torch.equal(torch.get_rng_state(), state)


def test_roi_jitter_updates_heatmap_coordinate_frame_without_changing_points():
    torch.manual_seed(3)
    centres = torch.tensor([39.5, 39.5]).expand(2, 2, 2).clone()
    extents = torch.ones_like(centres)*20
    c, e = jitter_part_regions(centres, extents, (360, 640), 1, 1, 1., .03, True)
    points = torch.tensor([40., 39., 1.]).expand(1, 1, 2, 3, 3).clone()
    K = torch.eye(3).expand(1, 1, 2, 3, 3)
    D = torch.zeros(1, 1, 2, 5)
    maps, valid = local_heatmap_targets(points, K, D,
        c.reshape(1, 1, 2, 2, 2), e.reshape(1, 1, 2, 2, 2),
        (40, 40), torch.tensor([[[360., 640.]]]))
    desired = ((points[..., :2].unsqueeze(-3)-c.reshape(1, 1, 2, 2, 1, 2))
               / e.reshape(1, 1, 2, 2, 1, 2)+1)*20-.5
    peak = maps.flatten(-2).argmax(-1)
    assert valid.all()
    assert torch.equal(peak, desired[..., 1].round().long()*40+desired[..., 0].round().long())


def test_jitter_adds_no_state_and_is_disabled_in_evaluation():
    torch.set_num_threads(2)
    config = {'part_crop_size': 32, 'part_mask_dilation_px': 0}
    old = ForegroundParts(config, 16, 16).eval()
    new = ForegroundParts({**config, 'part_roi_jitter_center_px': 1.,
                          'part_roi_jitter_scale_fraction': .03}, 16, 16).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    images = torch.randint(0, 256, (4, 3, 72, 128), dtype=torch.uint8)
    masks = torch.ones(4, 1, 18, 32)
    uv = torch.tensor([[30., 30.], [40., 30.], [40., 30.], [50., 27.], [50., 33.]]).expand(4, 5, 2)
    args = (images, masks, uv, torch.ones(4, 5), torch.zeros(1, 2, 2, 9), (72, 128), 1, 2)
    with torch.no_grad():
        expected = old(*args)
        actual = new(*args)
    torch.testing.assert_close(expected[0], actual[0], atol=0, rtol=0)
    torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)
    for name in expected[1]:
        torch.testing.assert_close(expected[1][name], actual[1][name], atol=0, rtol=0)
    new.train()
    _, details, _ = new(*args)
    assert not torch.equal(expected[1]['part_roi_centres'], details['part_roi_centres'])


@pytest.mark.parametrize('centre,scale', [(-1., .03), (1., 1.), (float('nan'), .03)])
def test_invalid_jitter_rejected(centre, scale):
    with pytest.raises(ValueError, match='jitter'):
        ForegroundParts({'part_roi_jitter_center_px': centre,
                         'part_roi_jitter_scale_fraction': scale}, 16, 16)
