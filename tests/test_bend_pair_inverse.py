"""Current speed/curvature inversion within the measured paired-PWM domain."""
import unittest
from unittest.mock import patch

from tests.test_bend_response import logical
from robot.control.controller import compensate_pwm
from robot.navigation.bend_response import BendResponse


class BendPairInverseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = BendResponse(.15, 37)
        cls.model.prepare_inner_reduction()

    def allocate(self, pair, curvature, **changes):
        options = dict(allowed=True, allow_inner_reduction=True,
                       reference_speed=.20, ceiling=.30, lateral_acceleration=.15)
        options.update(changes)
        return self.model.allocate(logical(pair), curvature, **options)

    def test_disabled_retains_original_outputs_reasons_and_lazy_cache(self):
        model = BendResponse(.15, 37)
        for pair, curvature in (((180, 92), -2.247942), ((140, 70), -3.),
                                ((70, 140), 2.2), ((54, 121), 2.66), ((90, 90), 0.)):
            with self.subTest(pair=pair, curvature=curvature):
                expected = model._allocate_outside_only(logical(pair), curvature,
                                                       allowed=True, ceiling=.30)
                reason = model.last_reason
                actual = model.allocate(logical(pair), curvature, allowed=True,
                    ceiling=.30, allow_inner_reduction=False, reference_speed=.05)
                self.assertEqual(actual, expected)
                self.assertEqual(model.last_reason, reason)
                self.assertEqual(model._joint_cache, {})

    def test_left18022_can_increase_inner_and_match_speed_and_curvature(self):
        result = self.allocate((65, 152), 2.799098,
            reference_speed=.20132332762843433, ceiling=.214255,
            effort_limits=dict(peak=167.80488528754137, sum=269.0009189661297))
        self.assertEqual(result['wire'], [67, 203])
        self.assertEqual(result['inside_reduced_pwm'], -2)
        self.assertFalse(result['limited'])
        self.assertLess(abs(result['speed_error_m_s']), .001)
        self.assertLess(2.799098-result['estimated_curvature_m_inv'], .007)

    def test_right18823_can_reduce_outer_without_trading_away_curvature(self):
        result = self.allocate((178, 83), -2.558447,
            reference_speed=.2197009688542004, ceiling=.293293,
            effort_limits=dict(peak=192.66560039992325, sum=314.83004422041336))
        self.assertEqual(result['wire'], [165, 55])
        self.assertEqual(result['outside_added_pwm'], -13)
        self.assertFalse(result['limited'])
        self.assertLess(abs(result['estimated_curvature_m_inv']+2.558447), .002)
        # A steep inner-PWM edge also brackets this target at159/56, but that
        # endpoint understeers4%. Its slightly closer speed must not win.
        self.assertNotEqual(result['wire'], [159, 56])

    def test_outside_domain_is_inverted_without_inventing_baseline_motion(self):
        for pair, curvature in (((54, 121), 2.66743365), ((112, 63), -1.9452279),
                                ((119, 36), -3.74630445), ((255, 70), -2.2)):
            with self.subTest(pair=pair):
                self.assertIsNone(self.model.predict(pair))
                result = self.allocate(pair, curvature)
                self.assertIsNotNone(result)
                self.assertFalse(result['base_model_available'])
                self.assertIsNone(result['base_curvature_m_inv'])
                self.assertEqual(result['target_speed_m_s'], .20)
                self.assertEqual(result['base_wheels'], logical(pair))
                self.assertEqual(result['base_mean'], sum(logical(pair))/2)
                self.assertIsNotNone(self.model.predict(result['wire']))
                self.assertLessEqual(abs(result['estimated_curvature_m_inv']), abs(curvature))

    def test_invalid_explicit_speed_is_rejected_and_missing_speed_uses_effort(self):
        for speed in (0., -1., float('nan'), float('inf')):
            with self.subTest(speed=speed):
                self.assertIsNone(self.allocate((54, 121), 2.6, reference_speed=speed))
                self.assertEqual(self.model.last_reason, 'REFERENCE_SPEED_UNAVAILABLE')
        result = self.allocate((140, 70), -2.2, reference_speed=None)
        self.assertIsNone(result['target_speed_m_s'])
        self.assertIsNone(result['speed_error_m_s'])
        self.assertEqual(result['reference_kind'], 'EFFORT')
        self.assertAlmostEqual(result['target_effort_mean_pwm'], sum(logical((140, 70)))/2)

    def test_reference_is_a_target_while_physical_ceiling_remains_a_limit(self):
        slow = self.allocate((180, 92), -2., reference_speed=.18)
        fast = self.allocate((180, 92), -2., reference_speed=.27)
        self.assertLess(slow['estimated_speed_m_s'], fast['estimated_speed_m_s'])
        self.assertGreater(slow['estimated_speed_m_s'], .18)
        bounded = self.allocate((180, 92), -2., reference_speed=.27, ceiling=.21)
        self.assertLessEqual(bounded['estimated_speed_m_s'], .21+1e-9)

    def test_speed_constraint_hole_never_accepts_the_previous34percent_overshoot(self):
        result = self.allocate((192, 117), -1.699029, ceiling=.2232667075,
            effort_limits=dict(peak=161.85928, sum=263.77813))
        self.assertIsNotNone(result)
        self.assertLessEqual(abs(result['estimated_curvature_m_inv']), 1.699029)
        self.assertNotEqual(result['wire'], [193, 64])

    def test_unreachable_target_is_reported_without_overshoot(self):
        result = self.allocate((119, 36), -4.)
        self.assertTrue(result['limited'])
        self.assertEqual(result['reason'], 'BEST_FEASIBLE_PAIRED_INVERSE')
        self.assertEqual(result['wire'], [140, 50])
        self.assertLess(abs(result['estimated_curvature_m_inv']), 4.)

    def test_permission_still_required_for_shared_pair_inversion(self):
        for pair, curvature, changes in (((140, 70), -2., dict(allowed=False)),
                ((90, 90), 0., dict(allowed=False)),
                ((140, 70), 2., dict(allowed=False)),
                ((140, 0), -3., dict(allowed=False))):
            with self.subTest(pair=pair, curvature=curvature):
                self.assertIsNone(self.allocate(pair, curvature, **changes))
                self.assertEqual(self.model.last_reason, 'BASELINE_GATE')

    def test_no_measured_pair_under_current_limits_keeps_fallback_explicit(self):
        for options in (dict(ceiling=.10), dict(lateral_acceleration=0.),
                        dict(effort_limits=dict(peak=10., sum=10.))):
            with self.subTest(options=options):
                self.assertIsNone(self.allocate((54, 121), 2.6, **options))
                self.assertEqual(self.model.last_reason, 'NO_FEASIBLE_PAIRED_RESPONSE')

    def test_final_pwm_satisfies_model_direction_and_every_original_physical_limit(self):
        for pair, sign in (((180, 92), -1), ((70, 140), 1),
                           ((112, 63), -1), ((54, 121), 1)):
            for magnitude in (1.2, 2., 2.6, 4.):
                for ceiling in (.20, .24, .30):
                    with self.subTest(pair=pair, k=sign*magnitude, ceiling=ceiling):
                        result = self.allocate(pair, sign*magnitude, ceiling=ceiling,
                            effort_limits=dict(peak=190., sum=300.), lateral_acceleration=.12)
                        if result is None:
                            continue
                        wire, v, w = result['wire'], result['estimated_speed_m_s'], result['estimated_yaw_rad_s']
                        self.assertEqual(compensate_pwm(result['wheels']), tuple(wire))
                        self.assertEqual(self.model.predict(wire), (v, w))
                        self.assertLessEqual(max(wire), 255)
                        self.assertGreater(min(wire), 0)
                        self.assertGreater(w*sign, 0)
                        self.assertLessEqual(abs(w/v), magnitude+1e-12)
                        self.assertLessEqual(v, ceiling+1e-9)
                        self.assertLessEqual(v*abs(w), .12+1e-9)
                        self.assertLessEqual(max(result['wheels']), 190.+1e-9)
                        self.assertLessEqual(sum(result['wheels']), 300.+1e-9)

    def test_hot_path_does_not_recompute_candidate_models_or_run_outside_only_first(self):
        with patch.object(self.model, '_predict', wraps=self.model._predict) as predict, \
                patch.object(self.model, '_allocate_outside_only', side_effect=AssertionError('old scan')):
            result = self.allocate((180, 92), -2.247942)
        self.assertIsNotNone(result)
        self.assertEqual(len(predict.call_args_list), 1)
        self.assertEqual(len(predict.call_args_list[0].args[0]), 1)


if __name__ == '__main__':
    unittest.main()
