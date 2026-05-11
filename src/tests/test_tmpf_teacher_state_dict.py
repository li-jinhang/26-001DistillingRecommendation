import unittest
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from models.tmpf import TeacherEnsemble


class TeacherStateDictCompatibilityTest(unittest.TestCase):
    def test_extract_state_dict_supports_wrapped_checkpoint_and_module_prefix(self):
        payload = {
            'state_dict': OrderedDict({
                'module.linear.weight': torch.ones(2, 2),
                'module.linear.bias': torch.zeros(2),
            })
        }
        extracted = TeacherEnsemble._extract_state_dict(payload)
        self.assertIn('linear.weight', extracted)
        self.assertIn('linear.bias', extracted)
        self.assertNotIn('module.linear.weight', extracted)

    def test_filter_compatible_state_dict_drops_unexpected_and_shape_mismatch(self):
        model_state = {
            'weight': torch.zeros(2, 2),
            'bias': torch.zeros(2),
        }
        loaded_state = {
            'weight': torch.ones(2, 2),
            'bias': torch.ones(3),
            'v_preference': torch.ones(2, 2),
        }
        compatible, unexpected_keys, shape_mismatch_keys = TeacherEnsemble._filter_compatible_state_dict(
            model_state,
            loaded_state
        )
        self.assertEqual(set(compatible.keys()), {'weight'})
        self.assertEqual(unexpected_keys, ['v_preference'])
        self.assertEqual(shape_mismatch_keys, ['bias'])

    def test_strict_load_still_passes_when_only_extra_keys_exist(self):
        model = nn.Linear(2, 1)
        checkpoint = OrderedDict(model.state_dict())
        checkpoint['v_preference'] = torch.randn(3, 4)
        compatible, unexpected_keys, shape_mismatch_keys = TeacherEnsemble._filter_compatible_state_dict(
            model.state_dict(),
            checkpoint
        )
        self.assertEqual(shape_mismatch_keys, [])
        self.assertEqual(unexpected_keys, ['v_preference'])
        model.load_state_dict(compatible, strict=True)

    def test_real_tmpa_checkpoint_compatibility(self):
        repo_root = Path(__file__).resolve().parents[2]
        checkpoint_candidates = sorted((repo_root / 'src' / 'saved').glob('TMPA-baby-*.pth'))
        if len(checkpoint_candidates) == 0:
            self.skipTest('未找到 TMPA 真实 checkpoint，跳过集成兼容性测试')
        raw_state = torch.load(checkpoint_candidates[-1], map_location='cpu')
        extracted = TeacherEnsemble._extract_state_dict(raw_state)
        model_state = {k: v for k, v in extracted.items() if k not in {'v_preference', 't_preference'}}
        compatible, unexpected_keys, shape_mismatch_keys = TeacherEnsemble._filter_compatible_state_dict(
            model_state,
            extracted
        )
        self.assertIn('v_preference', unexpected_keys)
        self.assertIn('t_preference', unexpected_keys)
        self.assertEqual(shape_mismatch_keys, [])
        self.assertEqual(set(compatible.keys()), set(model_state.keys()))


if __name__ == '__main__':
    unittest.main()
