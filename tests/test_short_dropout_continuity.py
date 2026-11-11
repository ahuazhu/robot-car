"""Old measured geometry bridges a short gap without adopting a new candidate."""

from tests.support import ROOT
from dataclasses import replace
import json
import math
import unittest
from unittest.mock import patch
import numpy as np

from robot.config import RuntimeSettings
from robot.core.ground_motion import GroundMotion, command_motion, apply_ground_motion
from robot.core.road_policy import RoadPolicy
from robot.navigation.tracker import TargetTracker
from robot.navigation.steering_control import SteeringControl
from robot.navigation.road_recovery import RoadSearch
from tests.test_curve_handoff import CAL, measurement, path, scene


class ShortDropoutTests(unittest.TestCase):
    def policy(self):
        return RoadPolicy(maximum_blind_distance_m=.15, maximum_prediction_command_s=.25)

    def estimator(self):
        return GroundMotion(dict(CAL, road_search=dict(finish_priority=True)), self.policy())

    def test_recent_measured_translation_is_used_without_pwm_scale_or_measured_label(self):
        estimator = self.estimator()
        previous = dict(scene(), interval_s=.2, translation=np.array([-.04, .01]))
        estimator.last_visual = (1., previous)
        fallback = command_motion(.5, 1., .2, estimator.policy)
        predicted = estimator.recent_prediction(1.2, .2, fallback)
        np.testing.assert_allclose(predicted['translation'], [-.04, .01])
        self.assertTrue(predicted['prediction_valid'])
        self.assertFalse(predicted['valid'])
        self.assertIsNone(predicted['speed_m_s'])
        self.assertGreater(predicted['sigma_m'], previous['sigma_m'])
        self.assertEqual(estimator.last_visual[0], 1.)
        self.assertIs(estimator.recent_prediction(1.4, .2, fallback), fallback)

    def test_fractional_turn_transform_composes_to_the_measured_turn(self):
        for sign in (-1, 1):
            estimator = self.estimator()
            original = command_motion(.2, sign * .8, .2, estimator.policy)
            original.update(valid=True, sigma_m=.002)
            estimator.last_visual = (1., original)
            half = estimator.recent_prediction(1.1, .1,
                                               command_motion(0, 0, .1, estimator.policy))
            points = np.array([[.1, .02], [.4, -.03]])
            twice = apply_ground_motion(apply_ground_motion(points, half), half)
            np.testing.assert_allclose(twice, apply_ground_motion(points, original), atol=1e-12)

    def test_missing_camera_or_overlong_gap_cannot_renew_recent_motion(self):
        estimator = self.estimator()
        estimator.last_visual = (1., scene())
        fallback = command_motion(.2, 0, .3, estimator.policy)
        self.assertIs(estimator.recent_prediction(1.3, .3, fallback), fallback)
        estimator.update(None, 1.1)
        self.assertIsNone(estimator.last_visual)

    def test_real_flow_failure_uses_recent_twist_for_only_the_original_window(self):
        estimator = self.estimator()
        estimator.previous = np.zeros((240, 320), np.uint8)
        estimator.previous_mask = np.zeros((240, 320), np.uint8)
        estimator.previous_time = 1.
        estimator.last_visual = (1., dict(scene(), interval_s=.2,
                                         translation=np.array([-.03, 0.])))
        image = np.zeros((240, 320, 3), np.uint8)
        with patch('robot.core.ground_motion.cv2.goodFeaturesToTrack', return_value=None):
            first = estimator.update(image, 1.2, .5, 0.)
            second = estimator.update(image, 1.4, .5, 0.)
        self.assertTrue(first['prediction_valid'])
        self.assertEqual(first['reason'], 'INSUFFICIENT FEATURES')
        self.assertNotIn('prediction_valid', second)
        self.assertFalse(second['valid'])

    def tracker(self, *, enabled=True):
        tracker = TargetTracker(dict(CAL, road_search=dict(finish_priority=enabled),
            road_model=dict(maximum_blind_distance_m=.15, maximum_prediction_command_s=.25)))
        tracker.motion.update = lambda *args: scene()
        tracker.pixels.measure = lambda *args: (measurement(path(.1, .65)), {})
        self.assertTrue(tracker.update(None, 0, require_current=True)[0]['observed'])
        return tracker

    def test_candidate_confirmation_does_not_revoke_the_old_usable_curve(self):
        tracker = self.tracker()
        before = tracker.model.path.copy()
        tracker.pixels.measure = lambda *args: (measurement(path(.1, .65, .2)), {})
        result, _ = tracker.update(None, .2)
        self.assertEqual(result['state'], 'PREDICTED')
        self.assertTrue(result['history_prediction_only'])
        self.assertTrue(result['transition_pending'])
        self.assertFalse(result['observed'])
        self.assertGreater(result['pending_frames'], 0)
        np.testing.assert_array_equal(tracker.model.path, before)
        self.assertLess(abs(np.asarray(result['control_path_m'])[:, 1]).max(), 1e-10)
        self.assertGreater(result['age_s'], 0.)
        self.assertEqual(result['road_model']['fresh_fraction'], 0.)

    def test_one_missed_frame_at_five_fps_keeps_forward_output_and_recovers(self):
        tracker = self.tracker()
        estimator = self.estimator()
        actual = dict(scene(), interval_s=.2, translation=np.array([-.03, 0.]))
        estimator.last_visual = (0., actual)
        predicted = estimator.recent_prediction(.2, .2,
                                                command_motion(.5, 0, .2, self.policy()))
        tracker.motion.update = lambda *args: predicted
        tracker.pixels.measure = lambda *args: (None, {})
        result, _ = tracker.update(None, .2)
        self.assertEqual(result['state'], 'PREDICTED')
        steering = SteeringControl(self.policy(), [0., 0.], [.075, .105, .085], .3)
        steering.observe(predicted, .2, False)
        command = steering.command(result, age_s=.12)
        self.assertIsNotNone(command)
        search = RoadSearch(self.policy(), [0., 0.], [.075, .105, .085], .3,
                            allow_visual_scan=True, finish_priority=True)
        # A preceding unavailable target may already have started look-around.
        search.command(dict(valid=False, state='LOST'), None, predicted, .1,
                       active=True, stopped=True)
        self.assertIsNotNone(search.scan.began)
        output = search.command(result, command, predicted, .2, active=True)
        self.assertGreater(min(output), 0)
        self.assertGreaterEqual(steering.status['prediction']['lease_s'], .2)
        self.assertEqual(search.snapshot()['phase'], 'CAUTIOUS')
        self.assertIsNone(search.scan.began)
        self.assertFalse(steering.status['prediction']['measured_motion'])
        tracker.motion.update = lambda *args: actual
        tracker.pixels.measure = lambda *args: (measurement(path(.1, .65)), {})
        recovered, _ = tracker.update(None, .4)
        self.assertEqual(recovered['state'], 'OBSERVED')

    def test_startup_and_conservative_mode_still_wait_for_candidate_confirmation(self):
        for enabled, require_current in ((False, False), (True, True)):
            tracker = self.tracker(enabled=enabled)
            tracker.pixels.measure = lambda *args: (measurement(path(.1, .65, .2)), {})
            result, _ = tracker.update(None, .2, require_current=require_current)
            self.assertFalse(result['transition_pending'])
            self.assertFalse(result['observed'])

    def test_green_counterevidence_and_expired_memory_never_authorize_history(self):
        for kind in ('green', 'expired'):
            tracker = self.tracker()
            surface = type('Surface', (), {'contradiction': lambda *args:
                        dict(contradicted=True, visible_samples=20, green_samples=20,
                             longest_green_m=.2)})() if kind == 'green' else None
            tracker.pixels.measure = lambda *args: (None, dict(surface=surface))
            result, _ = tracker.update(None, .2 if kind == 'green' else .8)
            self.assertFalse(result['valid'])
            self.assertFalse(result['history_prediction_only'])

    def test_prediction_bridges_a_camera_interval_and_still_expires_at_distance(self):
        control = SteeringControl(self.policy(), [0., 0.], [.075, .105, .085], .3)
        moving = dict(scene(), interval_s=.18, translation=np.array([-.036, 0.]))
        control.observe(moving, .2, False)
        target = dict(valid=True, observed=False, state='PREDICTED', confidence=.65,
            control_target_m=[.2, 0.], control_path_m=path(-.15, .65),
            path=path(-.15, .65), path_observed_mask=[False]*len(path(-.15, .65)),
            path_curvature_m_inv=0., road_model=dict(sigma_m=.008,
                blind_distance_m=.036, age_s=.18, length_m=.8))
        command = control.command(target, age_s=.13)
        self.assertIsNotNone(command)
        budget = control.status['prediction']
        self.assertGreaterEqual(budget['lease_s'], .18)
        self.assertLessEqual(budget['lease_s'], .25)
        self.assertLessEqual(budget['lease_s'] * control.status['speed_upper_m_s'],
                             budget['remaining_distance_m'] + 1e-12)
        target['road_model']['blind_distance_m'] = .15
        self.assertIsNone(control.command(target, age_s=.13))

    def test_estimated_motion_never_enables_speed_boost_or_trim_identification(self):
        control = SteeringControl(self.policy(), [0., 0.], [.075, .105, .085], .3)
        control.applied(.9, (100, 100))
        predicted = dict(scene(), valid=False, prediction_valid=True,
                         translation=np.array([-.02, 0.]), source='RECENT VISUAL PREDICTION')
        control.observe(predicted, 1., True)
        self.assertFalse(control.status['motion_measured'])
        self.assertFalse(control.status['feedback_valid'])
        self.assertEqual(control.trim, 0.)


if __name__ == '__main__':
    unittest.main()
