"""Opt-in reachable curvature without changing baseline speed-control semantics."""
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_bend_response import logical
from robot.control.controller import compensate_pwm
from robot.navigation.bend_response import BendResponse


class BendInnerReductionTests(unittest.TestCase):
    def setUp(self):
        self.model = BendResponse(.15, 37)

    def allocate(self, pair, target, **changes):
        options = dict(allowed=True, ceiling=.30, allow_inner_reduction=True)
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

    def test_frame3274_reaches_target_with_least_required_inside_reduction(self):
        options = dict(ceiling=.256732, lateral_acceleration=.15,
                       effort_limits=dict(peak=200.88623868957572, sum=331.55582923074917))
        base, target = (180, 92), -2.247942
        old = self.allocate(base, target, allow_inner_reduction=False, **options)
        result = self.allocate(base, target, **options)
        self.assertEqual(old['wire'], [217, 92])
        self.assertTrue(old['limited'])
        self.assertFalse(result['limited'])
        self.assertEqual(result['wire'], [208, 67])
        self.assertEqual(result['inside_reduced_pwm'], 25)
        self.assertEqual(result['allocation_mode'], 'OUTSIDE_PLUS_INNER_REDUCTION')
        self.assertFalse(result['inside_preserved'])
        self.assertEqual(result['base_wheels'], logical(base))
        self.assertEqual(result['base_mean'], sum(logical(base))/2)
        self.assertEqual(result['target_speed_m_s'], old['target_speed_m_s'])
        self.assertEqual(compensate_pwm(result['wheels']), tuple(result['wire']))
        self.assertLess(abs(result['estimated_curvature_m_inv']-target), .01)
        self.assertGreater(result['estimated_speed_m_s'], .22)
        table = self.model._joint_cache[-1]
        # Even without the other limits, a smaller inner reduction cannot reach
        # this target in the accepted model. No guessed minimum duty is used.
        higher_inside = ((table['pairs'][:, 1] > result['wire'][1])
                         & (table['pairs'][:, 1] <= base[1])
                         & (table['pairs'][:, 0] >= base[0]))
        self.assertTrue(np.all(table['curvature'][higher_inside] > target))

    def test_existing_solved_outside_pair_is_unchanged_and_needs_no_cache(self):
        base, target = (70, 140), 2.2
        old = self.allocate(base, target, allow_inner_reduction=False)
        self.assertFalse(old['limited'])
        self.assertEqual(self.allocate(base, target), old)
        self.assertEqual(self.model._joint_cache, {})

    def test_joint_candidate_must_improve_previous_result_and_preserve_turn(self):
        for base, target in (((180, 92), -2.5), ((100, 180), 3.1),
                             ((140, 70), -3.), ((120, 180), 2.4)):
            with self.subTest(base=base, target=target):
                old = self.allocate(base, target, allow_inner_reduction=False)
                result = self.allocate(base, target)
                self.assertIsNotNone(result)
                outside = int(target > 0)
                self.assertGreaterEqual(result['wire'][outside], base[outside])
                self.assertLessEqual(result['wire'][1-outside], base[1-outside])
                self.assertGreater(result['estimated_curvature_m_inv']*target, 0)
                previous = old['estimated_curvature_m_inv'] if old else self.model.curvature(base)
                if result != old:
                    self.assertLess(abs(result['estimated_curvature_m_inv']-target), abs(previous-target))
                self.assertIsNotNone(self.model.predict(result['wire']))
                self.assertLessEqual(max(result['wire']), 255)

    def test_unreachable_result_stays_explicit_instead_of_wrong_direction_approximation(self):
        result = self.allocate((180, 92), -6.)
        self.assertTrue(result['limited'])
        self.assertEqual(result['reason'], 'BEST_FEASIBLE_WITH_INNER_REDUCTION')
        self.assertLess(result['estimated_curvature_m_inv'], 0)
        self.assertLess(result['wire'][1], 92)
        self.assertGreater(result['estimated_curvature_m_inv'], -6.)

    def test_speed_limit_gap_does_not_turn_large_overshoot_into_solved(self):
        # Found by the independent review: the old joint draft jumped to
        # 193/64 (k=-2.27727), 34% stronger than this request, because all
        # milder points on that inner row were rejected by the speed ceiling.
        result = self.allocate((192, 117), -1.699029, ceiling=.2232667075,
            effort_limits=dict(peak=161.85928, sum=263.77813), lateral_acceleration=.15)
        self.assertIsNone(result)
        self.assertEqual(self.model.last_reason, 'NO_FEASIBLE_OUTSIDE_IMPROVEMENT')

    def test_joint_overshoot_is_only_adjacent_feasible_pwm_quantization(self):
        checked = 0
        for base in ((180, 92), (192, 117), (140, 70), (100, 180)):
            outside = 0 if base[0] > base[1] else 1
            sign = -1 if outside == 0 else 1
            for magnitude in (1.7, 2.247942, 2.5, 3.):
                for ceiling in (.2232667075, .26, .30):
                    result = self.allocate(base, sign*magnitude, ceiling=ceiling,
                        effort_limits=dict(peak=205., sum=340.), lateral_acceleration=.15)
                    if result is None or result.get('inside_preserved', True):
                        continue
                    checked += 1
                    k = sign*result['estimated_curvature_m_inv']
                    if k <= magnitude:
                        continue
                    self.assertFalse(result['limited'])
                    previous = result['wire'].copy()
                    previous[outside] -= 1
                    self.assertGreaterEqual(previous[outside], base[outside])
                    motion = self.model.predict(previous)
                    self.assertIsNotNone(motion)
                    v, w = motion
                    self.assertLessEqual(sign*w/v, magnitude)
                    self.assertLessEqual(v, ceiling+1e-9)
                    self.assertLessEqual(v*abs(w), .15+1e-9)
                    self.assertLessEqual(max(logical(previous)), 205.+1e-9)
                    self.assertLessEqual(sum(logical(previous)), 340.+1e-9)
        self.assertGreater(checked, 10)

    def test_permissions_straight_wrong_direction_and_unknown_baseline_are_unchanged(self):
        for pair, target, allowed in (((180, 92), -3., False), ((90, 90), 0., True),
                                      ((180, 92), 3., True), ((140, 40), -3., True),
                                      ((50, 140), 3., True), ((255, 70), -3., True)):
            with self.subTest(pair=pair, target=target, allowed=allowed):
                old = self.allocate(pair, target, allowed=allowed, allow_inner_reduction=False)
                reason = self.model.last_reason
                self.assertEqual(self.allocate(pair, target, allowed=allowed), old)
                self.assertEqual(self.model.last_reason, reason)
                self.assertEqual(self.model._joint_cache, {})

    def test_speed_lateral_and_effort_caps_remain_authoritative(self):
        base = (180, 92)
        for options in (dict(ceiling=.10), dict(lateral_acceleration=.01),
                        dict(effort_limits=dict(peak=10, sum=10))):
            with self.subTest(options=options):
                self.assertIsNone(self.allocate(base, -3., **options))
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

    def test_prewarm_is_idempotent_and_per_frame_prediction_is_not_a_full_grid(self):
        self.model.prepare_inner_reduction()
        identities = {k: id(v) for k, v in self.model._joint_cache.items()}
        with patch.object(self.model, '_predict', wraps=self.model._predict) as predict:
            self.model.prepare_inner_reduction()
            self.allocate((180, 92), -2.247942)
        self.assertEqual(identities, {k: id(v) for k, v in self.model._joint_cache.items()})
        # The original outside-only evaluation remains at most one 255-duty row;
        # the new two-wheel part uses only cached arithmetic and masks.
        self.assertLessEqual(max(len(call.args[0]) for call in predict.call_args_list), 255)


if __name__ == '__main__':
    unittest.main()
