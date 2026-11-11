"""A missed zebra detection changes a distance envelope, not motor permission."""

from tests.support import ROOT
import json
import unittest
import numpy as np

from robot.tasks.crosswalk import CrosswalkTask, crosswalk_body_support
from tests.support.crosswalk import configured, motion, observation


class CrosswalkDistanceAuthorityTests(unittest.TestCase):
    def task(self):
        config = json.loads((ROOT / "maixcam2/curve_follow/runtime.json").read_text())["crosswalk"]
        return CrosswalkTask(configured(**config))

    def start(self, task, distance=.40, sigma=.0225, normal=(1., 0.)):
        near = distance + crosswalk_body_support(normal, task.policy)
        for now in (0., .15, .30):
            obs = dict(observation(now, near), normal=list(normal), sigma_m=sigma)
            state = task.update(obs, motion(.03), now, True, False, False, True)
        self.assertEqual(state["phase"], "APPROACH")
        return state

    def test_measured_approach_survives_old_detection_and_ten_cm_prediction(self):
        task = self.task()
        self.start(task, .60)
        for step in range(1, 51):
            now = .30 + step * .15
            scene = dict(motion(.03, .0045), sigma_m=.0018, interval_s=.15)
            state = task.update(None, scene, now, True, False, False, True)
            command = task.limit_command((120, 140), state, scene)
            self.assertEqual(state["phase"], "APPROACH")
            self.assertGreater(min(command), 0)
            self.assertIsNone(state["approach_wait_reason"])
            self.assertFalse(state["yield_completed"])
        self.assertGreater(task.prediction_travel, task.policy.maximum_prediction_travel_m)
        self.assertGreater(state["observation_age_s"], task.policy.observation_max_age_s)
        self.assertTrue(state["position_available"])
        self.assertEqual(state["position_source"], "MEASURED_MOTION")
        self.assertEqual(task.track["observed_at"], .30)

    def test_recorded_oblique_far_stop_does_not_expire_while_still(self):
        task = self.task()
        self.start(task, .2577, normal=(.530565, -.847644))
        for step in range(1, 801):
            now = .30 + step * .15
            # Alternating subpixel displacement cancels; its per-frame noise
            # floor must not accumulate into metres of stationary uncertainty.
            delta = .00002 if step % 2 else -.00002
            scene = dict(motion(abs(delta) / .15), sigma_m=.002, interval_s=.15,
                         translation=np.array([delta, 0.]), travel_m=abs(delta))
            state = task.update(None, scene, now, True, True, True, True)
            self.assertEqual(state["phase"], "APPROACH")
            self.assertTrue(state["position_available"])
            self.assertGreater(min(task.limit_command((120, 140), state, scene)), 0)
        self.assertAlmostEqual(state["position_error_m"], .0225)
        self.assertAlmostEqual(state["distance_to_body_m"], .2577)
        self.assertEqual(state["position_source"], "STATIONARY")
        self.assertEqual(state["hold_elapsed_s"], 0.)

    def test_wider_position_bound_reduces_speed_instead_of_requesting_zero(self):
        ceilings = []
        for sigma in (.01, .06):
            task = self.task()
            state = self.start(task, .40, sigma)
            command = task.limit_command((120, 140), state, motion(.03))
            self.assertGreater(min(command), 0)
            self.assertIsNone(state["approach_wait_reason"])
            ceilings.append(task.speed_ceiling)
            delay = task.policy.command_latency_s + task.policy.speed_response_s
            stopping = task.speed_ceiling * delay + task.speed_ceiling ** 2 / (
                2 * task.policy.approach_deceleration
            )
            self.assertLessEqual(stopping, .40 - sigma - task.policy.stop_clearance_m + 1e-12)
        self.assertLess(ceilings[1], ceilings[0])

    def test_line_overlap_bound_still_requests_braking_and_never_serves_dwell(self):
        task = self.task()
        for step in range(45):
            now = step * .15
            obs = dict(observation(now, .13), sigma_m=.12)
            state = task.update(obs, motion(), now, True, True, True, True)
            if step >= 2:
                self.assertEqual(task.limit_command((120, 140), state, motion()), (0, 0))
                self.assertFalse(state["yield_completed"])
        self.assertEqual(state["phase"], "RECOVERY_HOLD")

    def test_short_motion_gap_resumes_with_conservative_bounds_not_a_false_fresh_pose(self):
        task = self.task()
        self.start(task)
        state = task.update(None, motion(valid=False), .45, True)
        self.assertEqual(task.limit_command((120, 140), state, motion(valid=False)), (0, 0))
        state = task.update(None, motion(), .60, True, True, True, True)
        self.assertTrue(state["position_available"])
        self.assertGreater(state["position_error_m"], .15)
        self.assertEqual(state["position_source"], "BOUNDED_MOTION")
        self.assertEqual(state["approach_speed_source"], "SCENE_SPEED_BOUND_AFTER_GAP")
        self.assertGreater(min(task.limit_command((120, 140), state, motion())), 0)
        obs = observation(.75, .40 + task.policy.body_front_m)
        state = task.update(obs, motion(), .75, True, True, True, True)
        self.assertTrue(state["position_available"])
        self.assertGreater(min(task.limit_command((120, 140), state, motion())), 0)

    def test_motion_gap_that_consumes_remaining_distance_cannot_resume_blindly(self):
        task = self.task()
        self.start(task, .26)
        task.update(None, motion(valid=False), .60, True)
        state = task.update(None, motion(), .75, True, True, True, True)
        self.assertLess(state["clearance_interval_m"][0], 0.)
        self.assertEqual(task.limit_command((120, 140), state, motion()), (0, 0))
        self.assertFalse(state["yield_completed"])

    def test_actual_motion_after_rest_updates_pose_and_its_uncertainty(self):
        task = self.task()
        self.start(task)
        for step in range(1, 10):
            state = task.update(None, motion(), .30 + step * .15, True, True, True, True)
        before = state["distance_to_body_m"]
        sigma = state["position_error_m"]
        scene = dict(motion(.10, .015), sigma_m=.002, interval_s=.15)
        state = task.update(None, scene, 1.80, True, True, True, True)
        self.assertAlmostEqual(state["distance_to_body_m"], before - .015)
        self.assertGreater(state["position_error_m"], sigma)
        self.assertEqual(state["position_source"], "MEASURED_MOTION")

    def test_far_braking_stop_resumes_from_continuous_pose_without_new_full_detection(self):
        task = self.task()
        for now in (0., .15, .30):
            state = task.update(observation(now, .40), motion(.35), now, True)
        self.assertEqual(state["phase"], "STOPPING")
        for now in (.45, .60, .75):
            state = task.update(None, motion(), now, True, True, True, True)
        self.assertEqual(state["phase"], "APPROACH")
        self.assertFalse(state["yield_completed"])
        self.assertEqual(state["hold_elapsed_s"], 0.)
        self.assertGreater(min(task.limit_command((120, 140), state, motion())), 0)

    def test_occluded_approach_stops_once_then_serves_one_verified_dwell(self):
        task = self.task()
        self.start(task, .26, sigma=.01)
        now = .30
        for step in range(1, 25):
            now += .15
            scene = dict(motion(.04, .006), sigma_m=.0002, interval_s=.15)
            state = task.update(None, scene, now, True, False, False, True)
            if state["phase"] == "STOPPING":
                break
            self.assertGreater(min(task.limit_command((120, 140), state, scene)), 0)
        self.assertEqual(state["phase"], "STOPPING")
        for step in range(1, 26):
            now += .15
            state = task.update(None, motion(), now, True, True, True, True)
            if state["phase"] == "PASSING":
                break
            self.assertEqual(task.limit_command((120, 140), state, motion()), (0, 0))
            self.assertNotEqual(state["phase"], "APPROACH")
        self.assertEqual(state["phase"], "PASSING")
        self.assertTrue(state["yield_verified"])
        self.assertGreaterEqual(state["hold_elapsed_s"], 3.2)
        self.assertEqual(task.limit_command((120, 140), state, motion()), (120, 140))


if __name__ == "__main__":
    unittest.main()
