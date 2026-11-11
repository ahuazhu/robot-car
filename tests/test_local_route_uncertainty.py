"""A stale distant tail must not change the current footprint's error bound."""
import unittest
from unittest.mock import patch
import numpy as np

from tests.support import APP_ROOT
from tests.support.driving import road
from robot.config import RuntimeSettings
from robot.core.geometry import fit_curve
from robot.navigation.tracker import TargetTracker
from robot.navigation.steering_control import SteeringControl
from robot.navigation.profile_speed_control import ProfileSpeedControl


class LocalRouteUncertaintyTests(unittest.TestCase):
    def evaluate(self, far_sigma, *, remove_profile=False):
        settings = RuntimeSettings.load(APP_ROOT, {})
        frame, cp = settings.runtime['control_frame'], settings.crosswalk_policy
        tracker = TargetTracker(settings.road_calibration)

        def path(end):
            return fit_curve(np.c_[np.linspace(.1, end, 60), np.zeros(60)])['path'] + frame['origin_m']

        old = path(.8)
        tracker.model.initialize(old, 0., .9, 'TWO EDGES')
        tracker.model.variance[:] = .006 ** 2
        tracker.model.variance[old[:, 0] - frame['origin_m'][0] > .6] = far_sigma ** 2
        motion = dict(valid=True, rotation=np.eye(2), translation=np.zeros(2),
                      sigma_m=0., interval_s=.1, travel_m=0., speed_m_s=0., yaw_rad_s=0.)
        measured = dict(path=path(.45), confidence=.9, mode='TWO EDGES')
        with patch.object(tracker.motion, 'update', return_value=motion), \
             patch.object(tracker.pixels, 'measure', return_value=(measured, {})):
            result, details = tracker.update(None, 10.)
        self.assertTrue(result['observed'])
        self.assertFalse(result['target_inferred'])
        self.assertEqual(len(result['path_sigma_m']), len(result['control_path_m']))
        if remove_profile:
            result.pop('path_sigma_m')
        c = SteeringControl(settings.road_policy, frame['origin_m'], frame['body_m'], .3,
            lever=settings.turn_lever, gain=settings.steering_gain, cruise=settings.cruise_pwm,
            deceleration=cp.brake_deceleration_m_s2, calibration=settings.calibration,
            straight_cruise=settings.straight_cruise_pwm,
            confirmed_cruise=settings.confirmed_cruise_pwm,
            speed_control=ProfileSpeedControl.from_settings(settings))
        c.observe(details['ground_motion'], 10., False)
        self.assertIsNotNone(c.command(result, .1))
        return result, c.status

    def test_far_unseen_tail_is_retained_without_penalizing_fresh_driving_prefix(self):
        normal, before = self.evaluate(.006)
        stale, after = self.evaluate(.20)
        self.assertEqual(normal['path'], stale['path'])
        self.assertAlmostEqual(stale['road_model']['sigma_m'], .20)
        self.assertEqual(after['support_uncertainty']['scope'], 'FORWARD_PROFILE')
        self.assertAlmostEqual(before['envelope']['minimum_clearance_m'],
                               after['envelope']['minimum_clearance_m'])
        self.assertGreater(after['envelope']['minimum_clearance_m'], 0.)
        self.assertAlmostEqual(before['speed_profile_control']['ceiling_m_s'],
                               after['speed_profile_control']['ceiling_m_s'])

    def test_missing_pointwise_evidence_retains_conservative_legacy_bound(self):
        _, known = self.evaluate(.20)
        _, unknown = self.evaluate(.20, remove_profile=True)
        self.assertEqual(unknown['support_uncertainty']['scope'], 'GLOBAL')
        self.assertLess(unknown['speed_profile_control']['ceiling_m_s'],
                        known['speed_profile_control']['ceiling_m_s'])

    def test_uncertainty_on_current_swept_segment_still_limits_speed(self):
        c = SteeringControl(RuntimeSettings.load(APP_ROOT, {}).road_policy,
                            [0., 0.], [.075, .105, .085], .3)
        path = road()['control_path_m']
        error = np.full(len(path), .006)
        normal = c.envelope(path, 0., .20, .3, support_sigma_m=error, speed_m_s=.2)
        error[(path[:, 0] >= .1) & (path[:, 0] < .3)] = .20
        uncertain = c.envelope(path, 0., .006, .3, support_sigma_m=error, speed_m_s=.2)
        self.assertGreater(normal['minimum_clearance_m'], 0.)
        self.assertLess(uncertain['minimum_clearance_m'], 0.)
        self.assertLess(uncertain['speed_scale'], 1.)

    def test_invalid_profile_cannot_silently_reduce_uncertainty(self):
        c = SteeringControl(RuntimeSettings.load(APP_ROOT, {}).road_policy,
                            [0., 0.], [.075, .105, .085], .3)
        path = road()['control_path_m']
        for profile in ([.001], np.full(len(path), np.nan), np.full(len(path), -.01)):
            with self.subTest(profile=str(profile)[:30]), self.assertRaises(ValueError):
                c.envelope(path, 0., .20, .3, support_sigma_m=profile)


if __name__ == '__main__':
    unittest.main()
