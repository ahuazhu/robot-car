"""Shared measured mapping does not infer driving permission from PWM ratios."""
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_bend_response import logical
from robot.control.controller import compensate_pwm
from robot.navigation.bend_response import BendResponse


class BendSharedAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = BendResponse(.15, 37)
        cls.model.prepare_inner_reduction()

    def allocate(self, pair, curvature, **changes):
        options = dict(allowed=True, allow_inner_reduction=True,
                       reference_speed=.22, ceiling=.30, lateral_acceleration=.15)
        options.update(changes)
        return self.model.allocate((0., 0.) if not any(pair) else logical(pair),
                                   curvature, **options)

    def test_explicit_stop_never_energizes_either_switch_position(self):
        for enabled in (False, True):
            for pair in ((0, 0), (140, 70)):
                with self.subTest(enabled=enabled, pair=pair):
                    self.assertIsNone(self.allocate(pair, -2.,
                        allow_inner_reduction=enabled, motion_requested=False))
                    self.assertEqual(self.model.last_reason, 'STOP_REQUESTED')
        self.assertIsNone(self.allocate((0, 0), -2.))
        self.assertEqual(self.model.last_reason, 'STOP_REQUESTED')

    def test_explicit_forward_intent_does_not_need_a_nonzero_baseline(self):
        zero = self.allocate((0, 0), -2., motion_requested=True)
        one_wheel_zero = self.allocate((140, 0), -2., motion_requested=True)
        ordinary = self.allocate((140, 70), -2., motion_requested=True)
        for result in (zero, one_wheel_zero):
            self.assertEqual(result['wire'], ordinary['wire'])
            self.assertFalse(result['base_model_available'])
            self.assertIsNone(result['base_curvature_m_inv'])
        self.assertEqual(zero['base_mean'], 0.)

    def test_request_direction_controls_model_even_when_baseline_points_the_other_way(self):
        for curvature in (-2., 2.):
            with self.subTest(curvature=curvature):
                a = self.allocate((140, 70), curvature)
                b = self.allocate((70, 140), curvature)
                self.assertEqual(a['wire'], b['wire'])
                self.assertGreater(a['estimated_curvature_m_inv']*curvature, 0.)
                self.assertLessEqual(abs(a['estimated_curvature_m_inv']), abs(curvature))

    def test_shallow_left_can_use_the_right_pwm_bank_motor_asymmetry(self):
        result = self.allocate((140, 70), .05)
        self.assertIsNotNone(result)
        # Right-turn PWM ordering can yield a slight left body yaw. Selecting
        # the model bank by requested turn sign would miss this valid pair.
        self.assertGreater(result['wire'][0], result['wire'][1])
        self.assertGreater(result['estimated_curvature_m_inv'], 0.)
        self.assertLessEqual(result['estimated_curvature_m_inv'], .05)
        self.assertLess(.05-result['estimated_curvature_m_inv'], .005)

    def test_straight_tracks_zero_yaw_using_a_real_adjacent_feasible_bracket(self):
        result = self.allocate((140, 70), 0.)
        self.assertIsNotNone(result)
        self.assertFalse(result['limited_curvature'])
        self.assertLess(abs(result['estimated_curvature_m_inv']), .02)
        self.assertLess(abs(result['speed_error_m_s']), .002)
        wire = result['wire']
        current = result['estimated_curvature_m_inv']
        witnesses = []
        for axis in (0, 1):
            for step in (-1, 1):
                other = wire.copy()
                other[axis] += step
                prediction = self.model.predict(other)
                if prediction is None:
                    continue
                v, w = prediction
                if current*w <= 0 and abs(current) <= abs(w/v)+1e-12:
                    witnesses.append(other)
        self.assertTrue(witnesses)

    def test_equal_pwm_anchors_are_in_shared_domain_without_duplicates(self):
        pairs = self.model._pair_cache[0]['pairs']
        for pwm in (90, 120, 150, 180, 200):
            self.assertEqual(int(np.sum(np.all(pairs == (pwm, pwm), axis=1))), 1)

    def test_straight_below_its_supported_speed_cannot_be_replaced_by_a_sharp_turn(self):
        self.assertIsNone(self.allocate((90, 90), 0., reference_speed=.10, ceiling=.18))
        self.assertEqual(self.model.last_reason, 'NO_FEASIBLE_STRAIGHT_RESPONSE')
        # The left sharp-turn anchor is under this cap, but is not an answer
        # to a straight request.
        self.assertLess(self.model.predict((60, 120))[0], .18)

    def test_unreachable_low_speed_is_explicit_without_extrapolation(self):
        result = self.allocate((90, 90), 0., reference_speed=.05)
        self.assertTrue(result['limited_speed'])
        self.assertTrue(result['limited_reference'])
        self.assertEqual(result['reason'], 'REFERENCE_LIMITED_PAIRED_INVERSE')
        self.assertEqual(result['target_speed_m_s'], .05)
        self.assertGreater(result['estimated_speed_m_s'], .18)
        self.assertGreater(result['feasible_reference_range'][0], .18)
        self.assertIsNotNone(self.model.predict(result['wire']))
        self.assertIsNone(self.allocate((90, 90), 0., reference_speed=.05, ceiling=.05))

    def test_no_speed_measurement_uses_original_effort_without_fabricated_velocity(self):
        baseline = (54, 121)
        result = self.allocate(baseline, 2.6, reference_speed=None)
        self.assertFalse(result['base_model_available'])
        self.assertEqual(result['reference_kind'], 'EFFORT')
        self.assertAlmostEqual(result['target_effort_mean_pwm'], sum(logical(baseline))/2)
        self.assertIsNone(result['target_speed_m_s'])
        self.assertIsNone(result['target_yaw_rad_s'])
        self.assertIsNone(result['speed_error_m_s'])
        self.assertIsNone(result['limited_speed'])
        self.assertLessEqual(result['estimated_curvature_m_inv'], 2.6)
        higher = self.allocate(baseline, 2.6, reference_speed=None, reference_effort=120.)
        self.assertGreater(sum(higher['wheels']), sum(result['wheels']))

    def test_cold_request_excludes_only_predictions_that_require_rolling_nodes(self):
        rolling = self.allocate((140, 70), -3., reference_speed=.20)
        cold = self.allocate((140, 70), -3., reference_speed=.20, allow_warmup=False)
        self.assertEqual(rolling['wire'], [140, 50])
        self.assertTrue(rolling['requires_warmup'])
        self.assertIsNotNone(cold)
        self.assertNotEqual(cold['wire'], rolling['wire'])
        self.assertFalse(cold['requires_warmup'])
        self.assertTrue(cold['limited_curvature'])
        # A shared triangle boundary also has independent non-warm support;
        # merely touching a warm triangle must not incorrectly exclude it.
        table = self.model._pair_cache[0]
        for pair, expected in (((140, 50), True), ((140, 70), False), ((120, 60), False)):
            matches = np.flatnonzero(np.all(table['pairs'] == pair, axis=1))
            self.assertEqual(len(matches), 1)
            self.assertEqual(bool(table['requires_warmup'][matches[0]]), expected)

    def test_cold_filter_is_cached_and_preserves_straight_and_physical_limits(self):
        with patch.object(self.model, '_cold_coverage', side_effect=AssertionError('recomputed')):
            for curvature in (0., -.05, .05, -2., 2.):
                result = self.allocate((140, 70), curvature, allow_warmup=False,
                    ceiling=.23, effort_limits=dict(peak=170., sum=280.))
                if result is None:
                    continue
                self.assertFalse(result['requires_warmup'])
                self.assertLessEqual(result['estimated_speed_m_s'], .23+1e-9)
                self.assertLessEqual(max(result['wheels']), 170.+1e-9)
                self.assertLessEqual(sum(result['wheels']), 280.+1e-9)

    def test_invalid_reference_and_permission_cannot_authorize_motion(self):
        for speed in (0., -.1, float('nan'), float('inf'), True):
            with self.subTest(speed=speed):
                self.assertIsNone(self.allocate((140, 70), -2., reference_speed=speed))
                self.assertEqual(self.model.last_reason, 'REFERENCE_SPEED_UNAVAILABLE')
        for effort in (-1., float('nan'), float('inf'), True):
            self.assertIsNone(self.allocate((140, 70), -2.,
                             reference_speed=None, reference_effort=effort))
            self.assertEqual(self.model.last_reason, 'REFERENCE_EFFORT_UNAVAILABLE')
        self.assertIsNone(self.allocate((140, 70), 0., allowed=False))

    def test_shared_domain_outputs_keep_hard_limits_and_nonzero_turn_sign(self):
        for curvature in (-2., -.2, -.02, 0., .02, .2, 2.):
            for reference in (None, .10, .22):
                with self.subTest(curvature=curvature, reference=reference):
                    result = self.allocate((100, 150), curvature, reference_speed=reference,
                        ceiling=.23, lateral_acceleration=.12,
                        effort_limits=dict(peak=170., sum=280.))
                    if result is None:
                        continue
                    v, w = result['estimated_speed_m_s'], result['estimated_yaw_rad_s']
                    self.assertLessEqual(v, .23+1e-9)
                    self.assertLessEqual(v*abs(w), .12+1e-9)
                    self.assertLessEqual(max(result['wheels']), 170.+1e-9)
                    self.assertLessEqual(sum(result['wheels']), 280.+1e-9)
                    self.assertEqual(compensate_pwm(result['wheels']), tuple(result['wire']))
                    if curvature:
                        self.assertGreater(w*curvature, 0.)
                        self.assertLessEqual(abs(w/v), abs(curvature)+1e-12)


if __name__ == '__main__':
    unittest.main()
