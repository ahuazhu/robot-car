"""Race IV.1(3) constrains a stop interval, never one exact setpoint."""

from tests.support import ROOT
import unittest

from robot.tasks.crosswalk import CrosswalkTask, crosswalk_body_support
from tests.support.crosswalk import configured, motion, observation
from robot.tasks.crosswalk import legal_stop_interval


class CrosswalkStopWindowTests(unittest.TestCase):
    def task(self):
        return CrosswalkTask(
            configured(
                body_front_m=0.02,
                body_rear_m=0.16,
                body_half_width_m=0.085,
                stop_clearance_m=0.08,
                brake_deceleration_m_s2=0.08,
            )
        )

    def update(self, t, now, distance, sigma=0.01, speed=0.0):
        obs = observation(now, distance + 0.02)
        obs["sigma_m"] = sigma
        state = t.update(obs, motion(speed), now, True, True, True, True)
        return state, t.limit_command((60, 60), state, motion(speed))

    def test_moving_inside_stop_window_brakes_without_approaching_buffer(self):
        for normal in ([1.0, 0.0], [0.8, 0.6]):
            for distance in (0.11, 0.14):
                with self.subTest(normal=normal, distance=distance):
                    t = self.task()
                    for now in (0.0, 0.1, 0.2):
                        near = distance + crosswalk_body_support(normal, t.policy)
                        obs = dict(observation(now, near), normal=normal)
                        s = t.update(obs, motion(0.04), now, True, False, False, True)
                        if now < 0.2:
                            self.assertFalse(s["confirmed"])
                            self.assertEqual(s["phase"], "SEARCH")
                    self.assertTrue(s["confirmed"])
                    self.assertEqual(s["phase"], "STOPPING")
                    self.assertEqual(t.limit_command((60, 60), s, motion(0.04)), (0, 0))
                    self.assertFalse(s["stop_verified"])
                    self.assertEqual(s["hold_elapsed_s"], 0.0)
                    # Once stopped anywhere in the window, serve the yield at
                    # that pose rather than rolling closer to the lower buffer.
                    for step in range(1, 43):
                        now = 0.2 + step * 0.1
                        near = distance - 0.01 + crosswalk_body_support(normal, t.policy)
                        obs = dict(observation(now, near), normal=normal)
                        s = t.update(obs, motion(), now, True, True, True, True)
                        if s["phase"] == "PASSING":
                            break
                        self.assertNotEqual(s["phase"], "APPROACH")
                        self.assertEqual(t.limit_command((60, 60), s, motion()), (0, 0))
                    self.assertEqual(s["phase"], "PASSING")
                    self.assertTrue(s["yield_verified"])
                    self.assertGreaterEqual(s["hold_elapsed_s"], t.policy.hold_s)

    def test_window_entry_requires_entire_interval_below_twenty_centimetres(self):
        for distance, sigma in ((0.24, 0.01), (0.19, 0.01), (0.18, 0.03)):
            with self.subTest(distance=distance, sigma=sigma):
                t = self.task()
                for now in (0.0, 0.1, 0.2):
                    s, cmd = self.update(t, now, distance, sigma=sigma, speed=0.02)
                self.assertEqual(s["phase"], "APPROACH")
                self.assertNotEqual(cmd, (0, 0))
                self.assertFalse(s["stop_verified"])

    def test_already_stationary_at_any_legal_distance_does_not_creep_to_buffer(self):
        for distance in (0.05, 0.10, 0.15, 0.18):
            with self.subTest(distance=distance):
                t = self.task()
                for now in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                    s, cmd = self.update(t, now, distance)
                self.assertEqual(s["phase"], "HOLD")
                self.assertEqual(cmd, (0, 0))
                self.assertEqual(s["legal_stop_window_m"], [0.0, 0.2])

    def test_coarse_position_is_sufficient_if_entire_interval_is_legal(self):
        t = self.task()
        for now in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            s, cmd = self.update(t, now, 0.10, sigma=0.05)
        self.assertEqual(s["phase"], "HOLD")
        self.assertTrue(s["stop_verified"])
        self.assertEqual(cmd, (0, 0))

    def test_legal_range_changes_do_not_restart_the_dwell(self):
        t = self.task()
        for now in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            s, cmd = self.update(t, now, 0.15)
        start = t.hold_since
        for i in range(1, 34):
            now = 0.6 + i * 0.1
            distance = 0.15 if i % 2 == 0 else 0.095
            s, cmd = self.update(t, now, distance)
            if s["phase"] != "PASSING":
                self.assertEqual(cmd, (0, 0))
        self.assertEqual(t.hold_since, start)
        self.assertEqual(s["phase"], "PASSING")
        self.assertTrue(s["yield_verified"])
        self.assertGreaterEqual(s["hold_elapsed_s"], t.policy.hold_s)

    def test_uncertainty_crossing_either_rule_boundary_is_not_legal(self):
        for interval in ((-0.01, 0.05), (0.17, 0.21), (0.0, 0.10), (0.1, 0.20)):
            self.assertFalse(legal_stop_interval(interval))
        self.assertTrue(legal_stop_interval((0.05, 0.15)))

    def test_initial_approach_enters_window_and_serves_one_stop(self):
        t = self.task()
        for now in (0.0, 0.1, 0.2, 0.3):
            s, cmd = self.update(t, now, 0.30)
        self.assertEqual(s["phase"], "APPROACH")
        self.assertNotEqual(cmd, (0, 0))
        self.update(t, 0.4, 0.24)
        for now in (0.5, 0.6, 0.7, 0.8, 0.9):
            s, cmd = self.update(t, now, 0.18)
        self.assertEqual(s["phase"], "HOLD")
        self.assertEqual(cmd, (0, 0))


if __name__ == "__main__":
    unittest.main()
