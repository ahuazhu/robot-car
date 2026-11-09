"""A crossing uses one physical dwell and a moving braking envelope."""
from tests.support import ROOT
import json
import unittest
import numpy as np
from robot.core.stationarity import StationaryEvidence
from robot.tasks.crosswalk import CrosswalkTask
from tests.support.crosswalk import configured, motion, observation


class SingleCrosswalkDwellTests(unittest.TestCase):
    def task(self):
        config = json.loads((ROOT/'maixcam2/curve_follow/runtime.json').read_text())['crosswalk']
        return CrosswalkTask(configured(**config))

    def update(self, task, now, distance, speed=0., sigma=.01, scene=None):
        obs = observation(now, near=distance+task.policy.body_front_m)
        obs['sigma_m'] = sigma
        return task.update(obs, motion(speed) if scene is None else scene,
                           now, True, speed == 0., speed == 0., True)

    def test_normal_approach_reaches_ten_to_fifteen_cm_then_stops_once(self):
        task = self.task()
        for now in (0., .1, .2):
            state = self.update(task, now, .30, speed=.04)
        self.assertEqual(state['phase'], 'APPROACH')
        self.assertGreater(task.limit_command((120, 140), state, motion(.04))[0], 0)
        self.assertGreater(task.speed_ceiling, .06)
        for now, distance in ((.3, .25), (.4, .20), (.5, .14)):
            scene = motion(.04, .30-distance if now == .3 else previous-distance)
            state = self.update(task, now, distance, speed=.04, scene=scene)
            previous = distance
        self.assertEqual(state['phase'], 'STOPPING')
        self.assertEqual(task.limit_command((120, 140), state, motion(.04)), (0, 0))
        for step in range(35):
            now = .6+step*.1
            state = self.update(task, now, .13)
            if task.phase == 'PASSING':
                break
            self.assertNotEqual(task.phase, 'APPROACH')
            self.assertEqual(task.limit_command((120, 140), state, motion()), (0, 0))
        self.assertEqual(task.phase, 'PASSING')
        self.assertTrue(state['yield_verified'])
        self.assertGreaterEqual(state['hold_elapsed_s'], 3.2)
        self.assertLess(now-.6, 3.4)
        self.assertEqual(task.limit_command((120, 140), state, motion()), (120, 140))

    def test_uncertain_position_does_not_add_a_second_full_hold(self):
        task = self.task()
        for step in range(45):
            now = step*.1
            state = self.update(task, now, .18, speed=.2 if step < 3 else 0., sigma=.03)
            if task.phase == 'PASSING':
                break
        self.assertEqual(task.phase, 'PASSING')
        self.assertLess(now, 4.)
        self.assertTrue(state['yield_completed'])
        self.assertFalse(state['yield_verified'])
        self.assertIn('CROSSWALK STOP UNVERIFIED', state['issues'])
        self.assertGreaterEqual(state['hold_elapsed_s'], task.policy.hold_s)

    def test_small_alternating_flow_jitter_does_not_restart_dwell_for_eleven_seconds(self):
        task = self.task()
        for now in (0., .1, .2):
            state = self.update(task, now, .14)
        for step in range(1, 30):
            now = .2+step*.16
            delta = .001 if step % 2 else -.001
            scene = motion(abs(delta)/.16, abs(delta))
            scene['translation'] = np.array([delta, 0.])
            scene['sigma_m'] = .0016
            state = self.update(task, now, .14, scene=scene)
            if task.phase == 'PASSING':
                break
        self.assertEqual(task.phase, 'PASSING')
        self.assertLess(now-.2, 4.)
        self.assertTrue(state['yield_verified'])
        self.assertGreaterEqual(state['hold_elapsed_s'], task.policy.hold_s)

    def test_definite_line_overlap_is_an_exception_after_the_same_physical_dwell(self):
        task = self.task()
        for step in range(38):
            now = step * .1
            state = self.update(task, now, -.02, sigma=.005)
            if task.phase == 'PASSING':
                break
            self.assertEqual(task.limit_command((120, 140), state, motion()), (0, 0))
        self.assertEqual(task.phase, 'PASSING')
        self.assertTrue(state['yield_completed'])
        self.assertFalse(state['yield_verified'])
        self.assertEqual(state['position_outcome'], 'AT_OR_PAST_LINE')
        self.assertGreaterEqual(state['hold_elapsed_s'], task.policy.hold_s)
        self.assertIn('CROSSWALK STOP WINDOW MISSED', state['issues'])

    def test_exception_outcome_is_about_the_stop_not_the_later_passing_position(self):
        task = self.task()
        for now in (0., .1, .2):
            state = self.update(task, now, .30, speed=.24)
        self.assertEqual(task.phase, 'STOPPING')
        for step in range(36):
            now = .3 + step * .1
            state = self.update(task, now, .29)
            if task.phase == 'PASSING':
                break
        self.assertEqual(task.phase, 'PASSING')
        self.assertEqual(state['position_outcome'], 'TOO_FAR')
        self.assertFalse(state['yield_verified'])
        state = task.update(None, motion(.1, .7), now+.1, True)
        self.assertEqual(state['position_outcome'], 'TOO_FAR')
        self.assertFalse(state['yield_verified'])

        state = task.update(observation(now+.2, .7), motion(), now+.2, True)
        self.assertFalse(state['yield_completed'])
        self.assertEqual(state['position_outcome'], 'NOT_VERIFIED')

    def test_steady_slow_motion_cannot_be_labelled_flow_jitter(self):
        evidence = StationaryEvidence(.006, .025, .25, .45)
        for step in range(30):
            scene = motion(.01, .0016)
            self.assertFalse(evidence.observe(scene, step*.16, True))

    def test_zero_ack_loss_and_missing_flow_break_the_rest_window(self):
        evidence = StationaryEvidence(.006, .025, .25, .45)
        for step in range(10):
            self.assertTrue(evidence.observe(motion(), step*.1, True))
        self.assertFalse(evidence.observe(motion(), 1., False))
        self.assertEqual(evidence.samples, [])
        self.assertFalse(evidence.observe(motion(valid=False), 1.1, True))

    def test_long_vision_gap_cannot_backfill_unobserved_rest_time(self):
        task = self.task()
        for now in (0., .1, .2, .3, .4, .5):
            state = self.update(task, now, .14)
        self.assertEqual(task.phase, 'HOLD')
        state = self.update(task, 4., .14)
        self.assertFalse(state['yield_completed'])
        self.assertEqual(state['hold_elapsed_s'], 0.)
        self.assertEqual(task.phase, 'STOPPING')

    def test_far_away_normal_command_is_not_limited_to_old_creep_speed(self):
        task = self.task()
        state = self.update(task, 0., 2., speed=.1)
        self.assertEqual(task.limit_command((60, 70), state, motion(.1)), (60, 70))


if __name__ == '__main__':
    unittest.main()
