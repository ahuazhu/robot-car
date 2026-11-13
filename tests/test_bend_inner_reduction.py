"""Retained baseline, physical limits and cache contracts for paired allocation.

Direct pair-inversion behavior is covered by test_bend_pair_inverse; these
regressions no longer prescribe one-way changes to either wheel.
"""
import unittest

import numpy as np

from tests.test_bend_response import logical
from robot.control.controller import compensate_pwm
from robot.navigation.bend_response import BendResponse


class BendInnerReductionTests(unittest.TestCase):
    def setUp(self):
        self.model = BendResponse(.15, 37)

    def allocate(self, pair, target, **changes):
        options = dict(allowed=True, ceiling=.30, allow_inner_reduction=True,
                       reference_speed=.22)
        options.update(changes)
        return self.model.allocate(logical(pair), target, **options)

    def test_disabled_is_identical_and_does_not_build_cache(self):
        for pair, target in (((180, 92), -2.247942), ((140, 70), -3.),
                             ((70, 140), 2.2), ((140, 40), -3.), ((90, 90), 0.)):
            with self.subTest(pair=pair, target=target):
                old = self.model._allocate_outside_only(logical(pair), target,
                                                       allowed=True, ceiling=.30)
                old_reason = self.model.last_reason
                self.assertEqual(self.allocate(pair, target, allow_inner_reduction=False), old)
                self.assertEqual(self.model.last_reason, old_reason)
                self.assertEqual(self.model._joint_cache, {})

    def test_frame3274_joint_solution_keeps_reference_and_delivery_contracts(self):
        options = dict(ceiling=.256732, lateral_acceleration=.15,
                       effort_limits=dict(peak=200.88623868957572, sum=331.55582923074917))
        base, target = (180, 92), -2.247942
        old = self.allocate(base, target, allow_inner_reduction=False, **options)
        result = self.allocate(base, target, **options)
        self.assertTrue(old['limited'])
        self.assertFalse(result['limited'])
        self.assertEqual(result['allocation_mode'], 'PAIRED_MODEL_INVERSE')
        self.assertEqual(result['base_wheels'], logical(base))
        self.assertEqual(result['base_mean'], sum(logical(base))/2)
        self.assertEqual(result['target_speed_m_s'], .22)
        self.assertEqual(compensate_pwm(result['wheels']), tuple(result['wire']))
        self.assertLess(abs(result['estimated_curvature_m_inv']-target), .01)
        self.assertIsNotNone(self.model.predict(result['wire']))
        self.assertLessEqual(max(result['wire']), 255)

    def test_permission_and_stop_remain_guards_while_current_intent_selects_pair(self):
        self.assertIsNone(self.allocate((180, 92), -3., allowed=False))
        self.assertEqual(self.model.last_reason, 'BASELINE_GATE')
        self.assertIsNone(self.allocate((180, 92), -3., motion_requested=False))
        self.assertEqual(self.model.last_reason, 'STOP_REQUESTED')
        self.assertEqual(self.model._joint_cache, {})
        straight = self.allocate((90, 90), 0.)
        self.assertIsNotNone(straight)
        self.assertLess(abs(straight['estimated_curvature_m_inv']), .02)
        opposite_baseline = self.allocate((180, 92), 3.)
        self.assertIsNotNone(opposite_baseline)
        self.assertGreater(opposite_baseline['estimated_curvature_m_inv'], 0.)
        self.assertLessEqual(opposite_baseline['estimated_curvature_m_inv'], 3.)

    def test_speed_lateral_and_effort_caps_remain_authoritative(self):
        base = (180, 92)
        for options in (dict(ceiling=.10), dict(lateral_acceleration=.01),
                        dict(effort_limits=dict(peak=10, sum=10))):
            with self.subTest(options=options):
                result = self.allocate(base, -3., **options)
                if result is None:
                    self.assertEqual(self.model.last_reason, 'NO_FEASIBLE_PAIRED_RESPONSE')
                    continue
                # Joint inversion may find a weaker pair that the one-sided
                # allocator could not. Its limitation and actual caps matter.
                self.assertTrue(result['limited'])
                self.assertEqual(result['reason'], 'BEST_FEASIBLE_PAIRED_INVERSE')
                if 'ceiling' in options:
                    self.assertLessEqual(result['estimated_speed_m_s'], options['ceiling'])
                if 'lateral_acceleration' in options:
                    self.assertLessEqual(result['estimated_speed_m_s']*abs(result['estimated_yaw_rad_s']),
                                         options['lateral_acceleration'])
                if 'effort_limits' in options:
                    self.assertLessEqual(max(result['wheels']), options['effort_limits']['peak'])
                    self.assertLessEqual(sum(result['wheels']), options['effort_limits']['sum'])
        options = dict(ceiling=.23, lateral_acceleration=.11,
                       effort_limits=dict(peak=163, sum=250))
        result = self.allocate(base, -2.5, **options)
        self.assertIsNotNone(result)
        self.assertLessEqual(result['estimated_speed_m_s'], options['ceiling'])
        self.assertLessEqual(result['estimated_speed_m_s']*abs(result['estimated_yaw_rad_s']),
                             options['lateral_acceleration'])
        self.assertLessEqual(max(result['wheels']), options['effort_limits']['peak'])
        self.assertLessEqual(sum(result['wheels']), options['effort_limits']['sum'])

    def test_cache_respects_asymmetric_and_sloping_model_support(self):
        self.model.prepare_inner_reduction()
        right = self.model._joint_cache[-1]['pairs']
        left = self.model._joint_cache[1]['pairs']
        self.assertEqual(int(right[:, 1].min()), 50)
        self.assertEqual(int(left[:, 0].min()), 60)
        self.assertEqual(int(right[right[:, 0] == 240, 1].min()), 70)
        self.assertFalse(np.any((right == [180, 50]).all(axis=1)))
        self.assertFalse(np.any((right == [255, 70]).all(axis=1)))
        self.assertTrue(np.any((left == [70, 255]).all(axis=1)))

    def test_prewarm_is_idempotent_across_allocation(self):
        self.model.prepare_inner_reduction()
        identities = {k: id(v) for k, v in self.model._joint_cache.items()}
        self.model.prepare_inner_reduction()
        self.allocate((180, 92), -2.247942)
        self.assertEqual(identities, {k: id(v) for k, v in self.model._joint_cache.items()})


if __name__ == '__main__':
    unittest.main()
