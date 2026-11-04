from tests.support import ROOT
from tests.support.driving import flow

"""Full-speed experiment: accepted local evidence, actual cap, baseline fallback."""
import json, unittest
import numpy as np

from tests.support.driving import road, control
from robot.navigation.tracker import TargetTracker
from robot.core.geometry import fit_curve
from robot.control.controller import compensate_pwm
from robot.navigation.steering_control import SteeringControl
from robot.core.road_policy import RoadPolicy

CAL = json.loads((ROOT / "tests/fixtures/camera_pose_20261031_191327.json").read_text())


class FullSpeedTests(unittest.TestCase):
    def test_clear_straight_can_reach_existing_120_cap_and_turns_keep_baseline(self):
        c, command = control(road(), fast=83.0)
        self.assertEqual(command, (83, 83))
        self.assertEqual(compensate_pwm(command), (120, 120))
        for k in (-4.0, -3.0, 3.0, 4.0):
            _, base = control(road(k), fast=60.0)
            _, fast = control(road(k), fast=83.0)
            self.assertEqual(fast, base)

    def test_weak_old_tail_cannot_veto_a_fresh_supported_goal(self):
        r = road(confidence=0.3, observation_confidence=0.9)
        _, base = control(r, fast=60.0)
        c, fast = control(r, fast=83.0)
        self.assertGreater(sum(fast), sum(base))
        self.assertTrue(c.status["speed_plan"]["boost_eligible"])
        self.assertLessEqual(max(compensate_pwm(fast)), 120)
        self.assertEqual(r["confidence"], 0.3)  # History safety quality is not overwritten.

    def test_unconfirmed_or_unobserved_segment_cannot_supply_boost_quality(self):
        for r in (
            road(state="CONFIRMING", observed=False, observation_confidence=0.99),
            road(state="PREDICTED", observed=False, observation_confidence=0.99),
            road(observation_confidence=0.0, candidate_confidence=0.99),
            road(observation_confidence=0.99, path_observed_mask=[False] * 100),
        ):
            _, base = control(r, fast=60.0)
            c, fast = control(r, fast=83.0)
            self.assertEqual(fast, base)
            self.assertFalse(c.status.get("speed_plan", {}).get("boost_eligible", False))

    def test_flow_dropout_cannot_drive_boost_but_does_not_erase_observed_road_time(self):
        c = SteeringControl(RoadPolicy(), [0, 0], (0.075, 0.105, 0.085), 0.3, straight_cruise=83.0)
        r = road()
        for now in (0.0, 0.15, 0.3, 0.45):
            c.observe(flow(), now, False)
            c.command(r, 0.1)
        self.assertTrue(c.status["speed_plan"]["boost_eligible"])
        c.observe(flow(False), 0.6, False)
        c.command(r, 0.1)
        self.assertFalse(c.status["speed_plan"]["boost_eligible"])
        self.assertEqual(c.status["speed_plan"]["selected_cruise_pwm"], 60.0)
        c.observe(flow(), 0.75, False)
        c.command(r, 0.1)
        self.assertTrue(c.status["speed_plan"]["boost_eligible"])

    def test_actual_road_loss_restarts_observation_horizon(self):
        c = SteeringControl(RoadPolicy(), [0, 0], (0.075, 0.105, 0.085), 0.3, straight_cruise=83.0)
        for now in (0.0, 0.15, 0.3, 0.45):
            c.observe(flow(), now, False)
            c.command(road(), 0.1)
        c.observe(flow(), 0.6, False)
        self.assertIsNone(c.command(road(valid=False, state="LOST", observed=False), 0.1))
        c.observe(flow(), 2.0, False)
        c.command(road(), 0.1)
        self.assertEqual(c.status["speed_plan"]["road_confirmed_s"], 0.0)
        self.assertFalse(c.status["speed_plan"]["boost_eligible"])

    def test_tracker_exports_quality_only_from_accepted_measurement(self):
        t = TargetTracker(CAL)

        def measure(y=0.0):
            return dict(
                path=fit_curve(np.c_[np.linspace(0.12, 0.65, 50), np.full(50, y)])["path"],
                confidence=0.9,
                mode="ONE LEFT",
            )

        t.pixels.measure = lambda *args: (measure(), {})
        r, _ = t.update(None, 0.0)
        self.assertTrue(r["observed"])
        self.assertEqual(r["observation_confidence"], 0.9)
        t.pixels.measure = lambda *args: (measure(0.25), {})
        r, _ = t.update(None, 0.1)
        self.assertFalse(r["observed"])
        self.assertEqual(r["candidate_confidence"], 0.9)
        self.assertEqual(r["observation_confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
