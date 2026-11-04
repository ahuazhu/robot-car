"""Race IV.1(3) constrains a stop interval, never one exact setpoint."""

from tests.support import ROOT
import unittest

from robot.tasks.crosswalk import CrosswalkTask
from tests.support.crosswalk import configured, motion, observation
from robot.tasks.crosswalk import legal_stop_interval


class CrosswalkStopWindowTests(unittest.TestCase):
    def task(self):
        return CrosswalkTask(
            configured(
                body_front_m=0.02, body_rear_m=0.16, body_half_width_m=0.085, stop_clearance_m=0.08
            )
        )

    def update(self, t, now, distance, sigma=0.01, speed=0.0):
        obs = observation(now, distance + 0.02)
        obs["sigma_m"] = sigma
        state = t.update(obs, motion(speed), now, True, True, True, True)
        return state, t.limit_command((60, 60), state, motion(speed))

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
        self.assertGreaterEqual(s["hold_elapsed_s"], 3.3)

    def test_uncertainty_crossing_either_rule_boundary_is_not_legal(self):
        for interval in ((-0.01, 0.05), (0.17, 0.21), (0.0, 0.10), (0.1, 0.20)):
            self.assertFalse(legal_stop_interval(interval))
        self.assertTrue(legal_stop_interval((0.05, 0.15)))

    def test_conclusively_outside_window_restarts_approach_but_legal_stop_does_not(self):
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
