"""Recorded zero/46-duty cycling must not turn a stopped stale pose into motion."""

from tests.support import ROOT
import unittest

from robot.tasks.crosswalk import CrosswalkTask
from tests.support.crosswalk import configured, motion, observation


class StaleCrosswalkResumeTests(unittest.TestCase):
    def task(self):
        t = CrosswalkTask(
            configured(
                body_front_m=0.02, body_rear_m=0.16, body_half_width_m=0.085, stop_clearance_m=0.08
            )
        )
        t.track = observation(0.0, near=0.60)
        t.track["sigma_m"] = 0.145
        t.hits = 3
        t.identity = 18
        t.phase = "STOPPING"
        t.last_update = 95.0
        t.stopping_since = 95.0
        t.stop_deadline = 96.0
        return t

    def test_recorded_stale_far_pose_never_restarts_approach(self):
        t = self.task()
        for i in range(20):
            now = 95.1 + i * 0.15
            s = t.update(None, motion(), now, True, True, True, True)
            self.assertNotEqual(s["phase"], "APPROACH")
            self.assertEqual(t.limit_command((60, 80), s, motion()), (0, 0))
        self.assertEqual(s["phase"], "RECOVERY_HOLD")

    def test_expired_but_precise_pose_cannot_authorize_forward_reapproach(self):
        t = self.task()
        t.track["sigma_m"] = 0.01
        s = t.update(None, motion(), 95.1, True, True, True, True)
        self.assertNotEqual(s["phase"], "APPROACH")
        self.assertEqual(t.limit_command((60, 80), s, motion()), (0, 0))

    def test_current_but_coarse_far_pose_is_not_a_movement_authorization(self):
        t = self.task()
        obs = observation(95.1, 0.60)
        obs["sigma_m"] = 0.05
        s = t.update(obs, motion(), 95.1, True, True, True, True)
        self.assertEqual(s["phase"], "STOPPING")
        self.assertEqual(t.limit_command((60, 80), s, motion()), (0, 0))

    def test_current_precise_far_observation_does_not_restart_approach(self):
        t = self.task()
        obs = observation(95.1, 0.60)
        s = t.update(obs, motion(), 95.1, True, True, True, True)
        self.assertEqual(s["phase"], "STOPPING")
        self.assertEqual(s["position_outcome"], "TOO_FAR")
        self.assertEqual(t.limit_command((60, 80), s, motion()), (0, 0))

    def test_coarse_legal_rest_still_earns_dwell_without_creeping(self):
        t = self.task()
        t.track = observation(95.0, 0.12)
        t.track["sigma_m"] = 0.05
        for i in range(6):
            now = 95.1 + i * 0.1
            obs = observation(now, 0.12)
            obs["sigma_m"] = 0.05
            s = t.update(obs, motion(), now, True, True, True, True)
            self.assertEqual(t.limit_command((60, 80), s, motion()), (0, 0))
        self.assertEqual(s["phase"], "HOLD")
        self.assertTrue(s["stop_verified"])


if __name__ == "__main__":
    unittest.main()
