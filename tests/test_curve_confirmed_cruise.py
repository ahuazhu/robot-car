"""Configured faster cruise needs the complete measured-road gate."""

from tests.support import ROOT
import json, unittest
import numpy as np

from tests.support.driving import road
from tests.support.driving import flow
from robot.navigation.steering_control import SteeringControl
from robot.core.road_policy import RoadPolicy
from robot.control.controller import compensate_pwm, MAX_MOTOR_PWM

CONFIG = json.loads((ROOT / "maixcam2/curve_follow/runtime.json").read_text())


def controller(confirmed=None):
    return SteeringControl(
        RoadPolicy(),
        [0.0, 0.0],
        (0.075, 0.105, 0.085),
        0.3,
        lever=CONFIG["turn_lever_m"],
        gain=CONFIG["steering_gain"],
        cruise=CONFIG["cruise_pwm"],
        straight_cruise=CONFIG["straight_cruise_pwm"],
        confirmed_cruise=CONFIG["confirmed_cruise_pwm"] if confirmed is None else confirmed,
    )


def replay(result, confirmed=None, measured=True):
    c = controller(confirmed)
    for now in (0.0, 0.15, 0.3, 0.45, 0.6):
        c.observe(flow(measured), now, False)
        command = c.command(result, 0.1)
    return c, command


class ConfirmedCruiseTests(unittest.TestCase):
    def test_configured_clear_straight_reaches_same_cap(self):
        _, old = replay(road(), CONFIG["cruise_pwm"])
        c, new = replay(road())
        self.assertEqual(new, old)
        self.assertEqual(compensate_pwm(new), (MAX_MOTOR_PWM, MAX_MOTOR_PWM))
        self.assertEqual(c.status["speed_plan"]["confirmed_cruise_pwm"], 163)

    def test_supported_bends_gain_effort_without_changing_curvature_or_quality_source(self):
        for k in (-3.0, 3.0):
            for quality in (1.0, 0.3):
                with self.subTest(k=k, quality=quality):
                    r = road(k, confidence=quality, observation_confidence=0.9)
                    _, old = replay(r, CONFIG["cruise_pwm"])
                    c, new = replay(r)
                    self.assertTrue(c.status["speed_plan"]["boost_eligible"])
                    self.assertGreater(sum(new), sum(old))
                    self.assertLessEqual(sum(new), sum(old) * 140 / 120 + 2)
                    self.assertAlmostEqual(
                        (new[1] - new[0]) / sum(new), (old[1] - old[0]) / sum(old), delta=0.02
                    )
                    self.assertLessEqual(max(compensate_pwm(new)), MAX_MOTOR_PWM)

    def test_uncertain_road_and_motion_keep_previous_effort(self):
        for r, measured in (
            (road(3.0, state="PREDICTED", observed=False), True),
            (road(3.0, confidence=0.4, observation_confidence=0.4), True),
            (road(3.0), False),
            (road(3.0, path_observed_mask=[False] * 100), True),
            (road(3.0, transition_pending=True), True),
            (road(valid=False, state="LOST", observed=False), True),
        ):
            _, old = replay(r, CONFIG["cruise_pwm"], measured)
            c, new = replay(r, measured=measured)
            self.assertEqual(new, old)
            self.assertFalse(c.status.get("speed_plan", {}).get("boost_eligible", False))

    def test_body_limit_and_short_support_cannot_enable_faster_cruise(self):
        blocked = road()
        blocked["path"] = blocked["path"] + np.array([0.0, 0.13])
        for r in (blocked, road(start=0.06, end=0.12)):
            _, old = replay(r, CONFIG["cruise_pwm"])
            c, new = replay(r)
            self.assertEqual(new, old)
            self.assertFalse(c.status["speed_plan"]["boost_eligible"])

    def test_faster_cruise_waits_for_history_and_drops_immediately_on_lost_flow(self):
        c = controller()
        old = controller(CONFIG["cruise_pwm"])
        r = road(3.0)
        for now in (0.0, 0.15, 0.3):
            c.observe(flow(), now, False)
            old.observe(flow(), now, False)
            self.assertEqual(c.command(r, 0.1), old.command(r, 0.1))
        c.observe(flow(), 0.45, False)
        old.observe(flow(), 0.45, False)
        self.assertGreater(sum(c.command(r, 0.1)), sum(old.command(r, 0.1)))
        c.observe(flow(False), 0.6, False)
        old.observe(flow(False), 0.6, False)
        self.assertEqual(c.command(r, 0.1), old.command(r, 0.1))

    def test_invalid_confirmed_cruise_is_rejected(self):
        for value in (float("nan"), float("inf"), 119.0, 164.0):
            with self.assertRaises(ValueError):
                controller(value)


if __name__ == "__main__":
    unittest.main()
