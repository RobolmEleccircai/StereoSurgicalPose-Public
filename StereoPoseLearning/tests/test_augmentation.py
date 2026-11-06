"""Online appearance changes preserve geometry and cached source tensors."""
import unittest
import torch
import torch.nn.functional as F
from stereo_pose.augmentation import augment_batch


def normalize(raw, size=(8, 12)):
    shape = raw.shape
    rgb = F.interpolate(raw.reshape(-1, 3, *shape[-2:]).float() / 255,
                        size=size, mode='bilinear', align_corners=False, antialias=True)
    mean = rgb.new_tensor([.485, .456, .406])[None, :, None, None]
    std = rgb.new_tensor([.229, .224, .225])[None, :, None, None]
    return ((rgb - mean) / std).reshape(*shape[:4], *size)


def example(device='cpu'):
    torch.manual_seed(8)
    raw = torch.randint(20, 210, (1, 1, 1, 3, 16, 24), dtype=torch.uint8, device=device)
    raw = raw.expand(2, 3, 2, 3, 16, 24).clone()
    return {'images': normalize(raw), 'hires_images': raw,
            'K': torch.eye(3, device=device).expand(2, 3, 2, 3, 3).clone(),
            'D': torch.zeros(2, 3, 2, 5, device=device),
            'joints': torch.rand(2, 3, 3, device=device),
            'masks': torch.ones(2, 3, 2, 1, 2, 3, device=device),
            'metadata': [[{'split': 'train', 'timestamp_ns': t * 16666667} for t in range(3)] for _ in range(2)]}


class AugmentationTest(unittest.TestCase):
    def test_parameters_shared_across_eyes_and_window_no_source_mutation(self):
        batch = example()
        before = {k: v.clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        config = {'probability': 1, 'brightness': [1.1, 1.1], 'contrast': [1., 1.],
                  'gamma': [1., 1.], 'channel_gain': [1., 1.],
                  'noise_probability': 0, 'blur_probability': 0}
        result = augment_batch(batch, config)
        self.assertIsNot(result, batch)
        self.assertEqual(result['hires_images'].dtype, torch.uint8)
        self.assertFalse(torch.equal(result['hires_images'], before['hires_images']))
        for key, value in before.items():
            torch.testing.assert_close(batch[key], value, atol=0, rtol=0)
            if key not in ('images', 'hires_images'):
                self.assertIs(result[key], batch[key])
        self.assertIs(result['metadata'], batch['metadata'])
        for b in range(2):
            for t in range(3):
                for eye in range(2):
                    torch.testing.assert_close(result['hires_images'][b, t, eye],
                                               result['hires_images'][b, 0, 0], atol=0, rtol=0)
        torch.testing.assert_close(result['images'], normalize(result['hires_images']), atol=0, rtol=0)

    def test_empty_disabled_zero_probability_and_eval_protection(self):
        batch = example()
        for config in (None, {}, {'enabled': False}, {'probability': 0}):
            self.assertIs(augment_batch(batch, config), batch)
        batch['metadata'][0][0]['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'training-only'):
            augment_batch(batch, {'probability': 1})

    def test_noise_blur_finite_and_legacy_supported(self):
        batch = example()
        config = {'probability': 1, 'noise_probability': 1, 'noise_std': [.01, .01],
                  'blur_probability': 1, 'blur_sigma': [.7, .7]}
        result = augment_batch(batch, config)
        self.assertTrue(torch.isfinite(result['images']).all())
        self.assertEqual(result['hires_images'].shape, batch['hires_images'].shape)
        legacy = {k: v for k, v in batch.items() if k != 'hires_images'}
        result = augment_batch(legacy, config)
        self.assertNotIn('hires_images', result)
        self.assertTrue(torch.isfinite(result['images']).all())
        with self.assertRaisesRegex(ValueError, 'Invalid augmentation'):
            augment_batch(batch, {'probability': 2})
        with self.assertRaisesRegex(ValueError, 'Unknown augmentation'):
            augment_batch(batch, {'rotation': 15})

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_dtype_device_and_no_mutation(self):
        batch = example('cuda')
        before = batch['hires_images'].clone()
        result = augment_batch(batch, {'probability': 1, 'blur_probability': 0})
        self.assertEqual(result['images'].device.type, 'cuda')
        self.assertEqual(result['hires_images'].dtype, torch.uint8)
        torch.testing.assert_close(batch['hires_images'], before, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(result['images']).all())


if __name__ == '__main__':
    unittest.main()
