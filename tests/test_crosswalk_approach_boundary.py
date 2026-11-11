"""Crosswalk braking regression from the 2026-10-25 recorded approach.

An uncertainty interval touching 20 cm is not evidence of an early stop.
"""

from tests.support import ROOT
import unittest


from robot.tasks.crosswalk import CrosswalkTask
from tests.support.crosswalk import configured, motion, observation


class CrosswalkApproachBoundaryTests(unittest.TestCase):
    def make_task(self):
        return CrosswalkTask(
            configured(
                body_front_m=0.02,
                body_rear_m=0.16,
                body_half_width_m=0.085,
                brake_deceleration_m_s2=0.08,
                stop_clearance_m=0.08,
                approach_speed_m_s=0.06,
                speed_per_pwm_m_s=0.006705385281385282,
            )
        )

    def observe(self, task, now, clearance, speed=0.0, sigma=0.0225):
        obs = observation(now, near=clearance + task.policy.body_front_m)
        obs["sigma_m"] = sigma
        state = task.update(obs, motion(speed), now, True, speed == 0.0, speed == 0.0, True)
        command = task.limit_command((53, 53), state, motion(speed))
        return state, command

    def brake(self, task, clearance):
        for now in (0.0, 0.15, 0.30):
            state, _ = self.observe(task, now, clearance, speed=0.24)
        self.assertEqual(state["phase"], "STOPPING")

    def test_unserved_ambiguous_stops_resume_if_the_nearest_line_leaves_braking_room(self):
        # Live frames 3404 and 3414 stopped at 20.72 and 17.96 cm,
        # both with a 2.25 cm position bound. Neither proves >20 cm.
        for clearance in (0.2071874987, 0.1796386638):
            with self.subTest(clearance=clearance):
                task = self.make_task()
                self.brake(task, clearance + 0.006)
                for now in (0.45, 0.60, 0.75):
                    state, command = self.observe(task, now, clearance)
                    self.assertEqual(state["phase"], "APPROACH")
                    self.assertGreater(min(command), 0)
                    self.assertEqual(state["hold_elapsed_s"], 0.)
                    self.assertFalse(state["yield_verified"])

    def test_precisely_reobserved_far_stop_resumes_approach_without_serving_dwell(self):
        task = self.make_task()
        self.brake(task, 0.26)
        for step in range(24):
            now = 0.45 + step * 0.15
            state, command = self.observe(task, now, (0.25, 0.24, 0.26)[step % 3], sigma=0.01)
            self.assertFalse(state["yield_verified"])
            self.assertFalse(state["yield_completed"])
            if state["phase"] != "APPROACH":
                self.assertEqual(command, (0, 0))
        self.assertEqual(state["phase"], "APPROACH")
        self.assertEqual(state["hold_elapsed_s"], 0)
        self.assertNotEqual(command, (0, 0))

    def test_fresh_legal_position_confirms_and_completes_normal_dwell(self):
        task = self.make_task()
        self.brake(task, 0.19)
        self.observe(task, 0.45, 0.18)
        # A fresh, more precise observation resolves the boundary ambiguity.
        for now in (0.60, 0.75, 0.90):
            state, command = self.observe(task, now, 0.18, sigma=0.01)
        self.assertEqual(state["phase"], "HOLD")
        self.assertEqual(command, (0, 0))
        hold_start = task.hold_since
        for step in range(1, 24):
            now = 0.90 + step * 0.15
            state, _ = self.observe(task, now, 0.18, sigma=0.01)
        self.assertGreaterEqual(now - hold_start, task.policy.hold_s)
        self.assertEqual(state["phase"], "PASSING")
        self.assertTrue(state["yield_completed"])
        self.assertTrue(state["yield_verified"])

    def test_ambiguity_never_earns_passage_by_waiting_in_an_unserved_far_pose(self):
        task = self.make_task()
        self.brake(task, 0.19)
        for step in range(1, 55):
            state, command = self.observe(task, 0.30 + step * 0.15, 0.18)
            if state["yield_completed"]:
                break
            self.assertGreater(min(command), 0)
        self.assertFalse(state["yield_completed"])
        self.assertEqual(state["phase"], "APPROACH")
        self.assertFalse(state["yield_verified"])
        self.assertEqual(state["hold_elapsed_s"], 0.)


if __name__ == "__main__":
    unittest.main()
