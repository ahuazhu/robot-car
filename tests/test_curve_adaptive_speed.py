from tests.support import ROOT
from tests.support.driving import road, control

"""Straight effort increases only with current measured road support.

These geometry checks preserve curvature and baseline recovery behavior;
actual speed/track performance still requires a recorded road run.
"""
import math
import unittest
import numpy as np


from robot.navigation.steering_control import SteeringControl
from robot.core.road_policy import RoadPolicy
from robot.control.controller import CommandRamp, compensate_pwm
from robot.tasks.crosswalk import CrosswalkPolicy, CrosswalkTask


class AdaptiveSpeedTests(unittest.TestCase):
    def test_clear_straight_increases_effort_with_same_steering_ratio_and_hardware_cap(self):
        for y in (0.0, 0.006, -0.006):
            r = road()
            r["path"][:, 1] += y
            old, a = control(r, 60.0)
            new, b = control(r)
            self.assertGreater(sum(b), sum(a))
            self.assertTrue(new.status["speed_plan"]["boost_eligible"])
            self.assertAlmostEqual((a[1] - a[0]) / sum(a), (b[1] - b[0]) / sum(b), delta=0.02)
            self.assertLessEqual(max(compensate_pwm(b)), 109)

    def test_left_right_bends_retain_the_baseline_commands(self):
        for curvature in (-4.0, -2.0, 2.0, 4.0):
            r = road(curvature)
            old, a = control(r, 60.0)
            new, b = control(r)
            self.assertEqual(b, a)
            self.assertEqual(new.status["speed_plan"]["selected_cruise_pwm"], 60.0)

    def test_upcoming_bend_removes_boost_even_before_the_target_turns(self):
        r = road(path_curvature_m_inv=3.0)
        _, a = control(r, 60.0)
        new, b = control(r)
        self.assertEqual(a, b)
        self.assertEqual(new.status["speed_plan"]["selected_cruise_pwm"], 60.0)

    def test_curve_load_reduces_extra_effort_continuously(self):
        selected = []
        for curvature in (0.0, 0.5, 1.0, 1.5, 2.0):
            new, _ = control(road(path_curvature_m_inv=curvature))
            selected.append(new.status["speed_plan"]["selected_cruise_pwm"])
        self.assertEqual(selected[0], 72.0)
        self.assertEqual(selected[-1], 60.0)
        self.assertTrue(all(a >= b for a, b in zip(selected, selected[1:])))

    def test_prediction_missing_flow_or_unobserved_goal_cannot_enable_extra_effort(self):
        cases = [
            (road(state="PREDICTED", observed=False), True),
            (road(path_observed_mask=[False] * 100), True),
            (road(), False),
        ]
        for r, measured in cases:
            _, a = control(r, 60.0, measured)
            new, b = control(r, 72.0, measured)
            self.assertEqual(a, b)
            self.assertFalse(new.status["speed_plan"]["boost_eligible"])

    def test_short_far_or_unverified_body_support_keeps_baseline(self):
        for start, end in ((0.06, 0.12), (0.4, 0.7)):
            r = road(start=start, end=end)
            _, a = control(r, 60.0)
            new, b = control(r)
            self.assertEqual(a, b)
            self.assertFalse(new.status["speed_plan"]["boost_eligible"])
        _, a = control(road(), 60.0, body=None)
        new, b = control(road(), body=None)
        self.assertEqual(a, b)
        self.assertFalse(new.status["speed_plan"]["boost_eligible"])

    def test_pending_handoff_and_blocked_body_cannot_gain_speed(self):
        for r in (road(transition_pending=True), road()):
            if not r.get("transition_pending"):
                r["path"] = r["path"] + np.array([0.0, 0.13])
            _, a = control(r, 60.0)
            new, b = control(r)
            self.assertEqual(a, b)

    def test_crosswalk_ceiling_remains_the_same_for_both_cruise_levels(self):
        p = CrosswalkPolicy.from_config(
            dict(
                mode="yield",
                body_front_m=0.02,
                body_rear_m=0.16,
                body_half_width_m=0.085,
                geometry_verified=True,
                brake_lamp_verified=True,
                brake_deceleration_m_s2=0.08,
                approach_speed_m_s=0.06,
                stop_clearance_m=0.08,
            )
        )
        outputs = []
        for cruise in (60.0, 72.0):
            task = CrosswalkTask(p)
            obs = dict(normal=[1.0, 0.0], near_m=0.6, far_m=0.7, sigma_m=0.02, observed_at=0.0)
            motion = dict(
                valid=True,
                speed_m_s=0.05,
                yaw_rad_s=0.0,
                rotation=np.eye(2),
                translation=np.zeros(2),
                sigma_m=0.001,
                travel_m=0.0,
            )
            state = task.update(obs, motion, 0.0, True)
            outputs.append(task.limit_command((cruise, cruise), state, motion))
            self.assertGreater(task.speed_ceiling, 0.06)
        self.assertEqual(outputs[0], outputs[1])

    def test_existing_ramp_still_limits_start_and_honors_zero_immediately(self):
        _, command = control(road())
        ramp = CommandRamp()
        samples = [ramp.apply(command, t * 0.1) for t in range(8)]
        self.assertLess(max(samples[0]), max(command))
        self.assertTrue(all(max(b) - max(a) <= 18.01 for a, b in zip(samples, samples[1:])))
        self.assertEqual(ramp.apply((0, 0), 0.9), (0, 0))

    def test_production_speed_step_is_bounded_and_preserves_bend_commands(self):
        for fast in (72.0, 78.0):
            _, straight = control(road(), fast)
            self.assertGreater(sum(straight), 120)
            self.assertLessEqual(max(compensate_pwm(straight)), fast + 37)
            for curvature in (-3.0, -2.0, 2.0, 3.0):
                _, old = control(road(curvature), 60.0)
                _, new = control(road(curvature), fast)
                self.assertEqual(new, old)

    def test_a_single_good_frame_does_not_enable_boost_and_weak_frame_keeps_baseline(self):
        c = SteeringControl(
            RoadPolicy(), [0.0, 0.0], (0.075, 0.105, 0.085), 0.3, cruise=60.0, straight_cruise=78.0
        )
        m = dict(
            valid=True,
            interval_s=0.15,
            rotation=np.eye(2),
            translation=np.array([-0.015, 0.0]),
            sigma_m=0.0002,
        )
        for now in (0.0, 0.15, 0.30):
            c.observe(m, now, False)
            command = c.command(road(), age_s=0.1)
            self.assertEqual(c.status["speed_plan"]["selected_cruise_pwm"], 60.0)
            self.assertTrue(any(command))
        c.observe(m, 0.45, False)
        c.command(road(), age_s=0.1)
        self.assertGreater(c.status["speed_plan"]["selected_cruise_pwm"], 60.0)
        weak = road(confidence=0.6)
        c.observe(m, 0.60, False)
        command = c.command(weak, age_s=0.1)
        _, baseline = control(weak, fast=60.0)
        self.assertEqual(command, baseline)
        self.assertTrue(any(command))
        self.assertFalse(c.status["speed_plan"]["boost_eligible"])
        c.observe(m, 0.75, False)
        c.command(road(), age_s=0.1)
        self.assertEqual(c.status["speed_plan"]["selected_cruise_pwm"], 60.0)

    def test_repeated_command_evaluation_cannot_replace_elapsed_observation_time(self):
        c = SteeringControl(
            RoadPolicy(), [0.0, 0.0], (0.075, 0.105, 0.085), 0.3, cruise=60.0, straight_cruise=78.0
        )
        m = dict(
            valid=True,
            interval_s=0.15,
            rotation=np.eye(2),
            translation=np.array([-0.015, 0.0]),
            sigma_m=0.0002,
        )
        c.observe(m, 1.0, False)
        for _ in range(20):
            c.command(road(), age_s=0.1)
            self.assertEqual(c.status["speed_plan"]["selected_cruise_pwm"], 60.0)

    def test_invalid_speed_settings_are_rejected(self):
        for fast in (float("nan"), float("inf"), 59.0, -1.0):
            with self.assertRaises(ValueError):
                control(road(), fast)


if __name__ == "__main__":
    unittest.main()
