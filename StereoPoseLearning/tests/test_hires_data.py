"""The raw branch preserves source pixels; legacy datasets stay unchanged."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from test_data import save_label, save_plan, sha
from stereo_pose.data import StereoDataset, build_manifest, collate_samples, TENSOR_KEYS


class HighResolutionDataTest(unittest.TestCase):
    def test_raw_pixels_targets_cache_and_optional_collation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'; root.mkdir()
            save_plan(root, 2)
            source = np.arange(32 * 64 * 3, dtype=np.uint16).reshape(32, 64, 3).astype(np.uint8)
            for frame in range(2):
                p = save_label(root, frame)
                label = json.loads(p.read_text())
                for eye in ('left', 'right'):
                    image = (p.parent / label['eyes'][eye]['image']).resolve()
                    Image.fromarray(source).save(image)
                    label['eyes'][eye]['image_sha256'] = sha(image)
                p.write_text(json.dumps(label))
            manifest = build_manifest([root], Path(tmp) / 'manifest.json')
            legacy = StereoDataset(manifest, 'train', window=2, image_size=(16, 32)).preload_shared()
            hires = StereoDataset(manifest, 'train', window=2, image_size=(16, 32), include_hires=True)
            sample = hires[0]
            self.assertEqual(sample['hires_images'].dtype, torch.uint8)
            self.assertEqual(tuple(sample['hires_images'].shape), (2, 2, 3, 32, 64))
            np.testing.assert_array_equal(sample['hires_images'][0, 0].numpy(), source.transpose(2, 0, 1))
            for key in TENSOR_KEYS:
                torch.testing.assert_close(sample[key], legacy[0][key], atol=0, rtol=0)
            self.assertNotIn('hires_images', legacy[0])
            hires.preload_shared()
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(hires[0][key], value, atol=0, rtol=0)
            self.assertIsNot(hires._shared_cache, legacy._shared_cache)
            combined = collate_samples([hires[0], hires[0]])
            self.assertEqual(tuple(combined['hires_images'].shape), (2, 2, 2, 3, 32, 64))
            self.assertEqual(combined['hires_images'].dtype, torch.uint8)
            with self.assertRaisesRegex(ValueError, 'mix samples'):
                collate_samples([legacy[0], hires[0]])
            with self.assertRaisesRegex(ValueError, 'empty'):
                collate_samples([])


if __name__ == '__main__':
    unittest.main()
