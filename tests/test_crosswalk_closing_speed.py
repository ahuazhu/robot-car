"""Whole-body approach rate and recovery from a real, unserved far stop."""

from tests.support import ROOT
import math
import unittest
from unittest.mock import patch
import numpy as np

from robot.tasks.crosswalk import CrosswalkTask, crosswalk_body_support, crosswalk_closing_speed, draw_crosswalk_overlay
from tests.support.crosswalk import configured, motion, observation


class ClosingSpeedTests(unittest.TestCase):
    def setUp(self):
        self.policy = configured(body_front_m=.02, body_rear_m=.16, body_half_width_m=.085)

    def scene(self, shift, yaw=0., dt=.1):
        r = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
        return dict(valid=True, rotation=r, translation=-r @ np.asarray(shift),
                    interval_s=dt, sigma_m=.001, travel_m=float(np.linalg.norm(shift)),
                    speed_m_s=float(np.linalg.norm(shift)) / dt, yaw_rad_s=yaw / dt)

    def test_matches_independent_world_line_clearance_in_both_directions(self):
        for normal in ([1., 0.], [.8, .6], [.8, -.6], [-.8, .6]):
            for shift, yaw in (([.02, .03], .08), ([0., .04], 0.), ([0., 0.], .1)):
                scene = self.scene(shift, yaw)
                current = scene['rotation'] @ normal
                old_gap = .5 - crosswalk_body_support(normal, self.policy)
                new_gap = (.5 - float(np.asarray(normal) @ shift)
                           - crosswalk_body_support(current, self.policy))
                actual, source = crosswalk_closing_speed(current, scene, self.policy)
                self.assertAlmostEqual(actual, max(0., (old_gap - new_gap) / .1))
                self.assertEqual(source, 'BODY_CLEARANCE_RATE')

    def test_parallel_translation_is_not_approach_but_a_turning_corner_is(self):
        parallel = self.scene([0., .04])
        self.assertEqual(crosswalk_closing_speed([1., 0.], parallel, self.policy)[0], 0.)
        turning = self.scene([0., 0.], .1)
        normal = turning['rotation'] @ [1., 0.]
        self.assertGreater(crosswalk_closing_speed(normal, turning, self.policy)[0], .08)

    def test_missing_or_invalid_timed_motion_cannot_authorize_blind_speed(self):
        for scene in (dict(valid=False), self.scene([.01, 0.], dt=.1)):
            if scene.get('valid'):scene['interval_s'] = 0.
            self.assertIsNone(crosswalk_closing_speed([1., 0.], scene, self.policy)[0])
        value, source = crosswalk_closing_speed([1., 0.], motion(.2), self.policy)
        self.assertEqual(value, .2)
        self.assertEqual(source, 'SCENE_SPEED_BOUND')

    def test_large_total_motion_cannot_trigger_a_far_stop_when_closure_is_small(self):
        task = CrosswalkTask(configured(body_front_m=.02, body_rear_m=.16,
                                       body_half_width_m=.085, brake_deceleration_m_s2=.08,
                                       stop_clearance_m=.02))
        scene = self.scene([.01, .035])
        for now in (0., .1, .2):
            state = task.update(observation(now, .42), scene, now, True, False, False, True)
        self.assertGreater(scene['speed_m_s'], .36)
        self.assertAlmostEqual(state['approach_speed_m_s'], .1)
        self.assertEqual(state['phase'], 'APPROACH')
        self.assertLess(state['predicted_stopping_distance_m'], .1)
        self.assertNotEqual(task.limit_command((60, 60), state, scene), (0, 0))


class FarStopLifecycleTests(unittest.TestCase):
    def task(self):
        task = CrosswalkTask(configured(body_front_m=.02, body_rear_m=.16,
                                       body_half_width_m=.085, brake_deceleration_m_s2=.08))
        for now in (0., .1, .2):
            task.update(observation(now, .405), motion(.3), now, True, False, False, True)
        self.assertEqual(task.phase, 'STOPPING')
        return task

    def test_recorded_38cm_stop_reapproaches_instead_of_serving_the_yield(self):
        task = self.task()
        for now in (.3, .4, .5, .6):
            state = task.update(observation(now, .405), motion(), now, True, True, True, True)
        self.assertEqual(state['phase'], 'APPROACH')
        self.assertFalse(state['yield_completed'])
        self.assertEqual(state['hold_elapsed_s'], 0)
        self.assertNotEqual(task.limit_command((60, 60), state, motion()), (0, 0))
        # Only the subsequent actual legal stop earns the dwell.
        for i in range(5):
            now = .8 + i * .2
            task.update(observation(now, .405 - (i + 1) * .051),
                        motion(.255, .051), now, True, False, False, True)
        for i in range(45):
            now = 1.7 + i * .1
            state = task.update(observation(now, .15), motion(), now, True, True, True, True)
            if state['yield_completed']:break
        self.assertTrue(state['yield_verified'])
        self.assertGreaterEqual(state['hold_elapsed_s'], 3.2)
        self.assertEqual(task.identity, 1)

    def test_far_stop_without_current_road_does_not_time_out_into_passing(self):
        task = self.task()
        for i in range(60):
            now = .3 + i * .1
            state = task.update(observation(now, .405), motion(), now, True, True, True, False)
            self.assertFalse(state['yield_completed'])
            self.assertEqual(task.limit_command((60, 60), state, motion()), (0, 0))
        self.assertEqual(state['phase'], 'RECOVERY_HOLD')

    def test_legal_stop_is_locked_against_later_far_measurement_nudges(self):
        task = CrosswalkTask(configured(body_front_m=.02, body_rear_m=.16,
                                       body_half_width_m=.085, brake_deceleration_m_s2=.08))
        for now in (0., .1, .2, .3, .4, .5, .6):
            state = task.update(observation(now, .15), motion(), now, True, True, True, True)
        self.assertTrue(task.legal_stop_acquired)
        task.track = observation(.7, .405)
        for now in (.7, .8, .9, 1.):
            state = task.update(observation(now, .405), motion(), now, True, True, True, True)
            self.assertNotEqual(state['phase'], 'APPROACH')
            self.assertEqual(task.limit_command((60, 60), state, motion()), (0, 0))


class CrosswalkOverlayTests(unittest.TestCase):
    def test_preview_and_sensing_pause_cannot_look_like_an_active_approach(self):
        image = np.zeros((240, 320, 3), np.uint8)
        for active, reason, expected in (
            (False, None, 'PREVIEW'),
            (True, 'CROSSWALK POSITION UNCERTAIN', 'WAIT POSITION'),
            (True, 'CROSSWALK SPEED UNVERIFIED', 'WAIT MOTION'),
        ):
            state = dict(mode='yield', phase='APPROACH', distance_to_body_m=.38,
                         approach_wait_reason=reason)
            with patch('robot.tasks.crosswalk.cv2.putText') as text:
                draw_crosswalk_overlay(image, state, {}, active=active)
                self.assertEqual(text.call_args.args[1], 'ZEBRA YIELD %s 38cm' % expected)


if __name__ == '__main__':
    unittest.main()
