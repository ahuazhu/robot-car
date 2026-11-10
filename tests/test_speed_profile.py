"""Physical invariants and recorded-input boundaries for the offline planner."""

from dataclasses import replace
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

from tests.support import ROOT
from robot.navigation.speed_profile import SpeedLimits, plan_speed_profile
from maixcam2.curve_follow.tools.drive_reference import DriveReference
from maixcam2.curve_follow.tools.replay_speed_profile import load_frames, replay, supported_segment


LIMITS = SpeedLimits(0.30, 0.15, 0.15, 0.08, 0.75, 0.15)


class SpeedProfileTests(unittest.TestCase):
    def test_straight_stopping_bound_charges_delay_once(self):
        expected = math.sqrt((0.08 * 0.75) ** 2 + 2 * 0.08 * 0.4) - 0.08 * 0.75
        for count in (2, 11, 101):
            s = [0.4 * i / (count - 1) for i in range(count)]
            plan = plan_speed_profile(s, [0.0] * count, LIMITS)
            self.assertAlmostEqual(plan.entry_limit_m_s, expected)
            self.assertEqual(plan.entry_reason, "SUPPORT_END")
            self.assertEqual(plan.speeds_m_s[-1], 0.0)

    def test_long_straight_reaches_ceiling(self):
        plan = plan_speed_profile([0, 0.5, 1, 2], [0] * 4, LIMITS)
        self.assertEqual(plan.entry_limit_m_s, 0.30)
        self.assertEqual(plan.entry_reason, "MAXIMUM_SPEED")

    def test_future_turn_slows_before_turn(self):
        limits = replace(LIMITS, maximum_m_s=1.0, delay_s=0.0, turn_lever_m=0.05)
        plan = plan_speed_profile([0, 0.1, 0.2, 1.5], [0, 6, 6, 0], limits)
        turn_speed = math.sqrt(0.15 / 6)
        self.assertAlmostEqual(plan.entry_limit_m_s, math.sqrt(turn_speed ** 2 + 2 * 0.08 * 0.1))
        self.assertEqual(plan.entry_reason, "CURVATURE")
        self.assertEqual(plan.limiting_station_m, 0.1)
        self.assertLess(plan.entry_limit_m_s, limits.maximum_m_s)

    def test_nominal_outer_wheel_limit_and_forward_only_boundary(self):
        plan = plan_speed_profile([0, 2, 3], [2] * 3, LIMITS)
        self.assertAlmostEqual(plan.entry_limit_m_s * (1 + 0.15 * 2), LIMITS.maximum_m_s)
        self.assertEqual(plan.entry_reason, "OUTER_WHEEL")
        blocked = plan_speed_profile([0, 2, 3], [7] * 3, LIMITS)
        self.assertEqual(blocked.entry_limit_m_s, 0)
        self.assertEqual(blocked.entry_reason, "FORWARD_KINEMATICS")

    def test_stop_inserted_exactly_and_margin_not_added_as_extra_travel(self):
        plan = plan_speed_profile([0, 0.3, 0.6, 1], [0] * 4, LIMITS,
                                  stop_at_m=0.415, distance_margin_m=0.015)
        self.assertAlmostEqual(plan.stations_m[-1], 0.4)
        self.assertEqual(plan.entry_reason, "STOP_POINT")
        self.assertEqual(plan.speeds_m_s[-1], 0)
        immediate = plan_speed_profile([0, 0.3], [0, 0], LIMITS, distance_margin_m=0.4,
                                       initial_speed_m_s=0.1)
        self.assertEqual(immediate.stations_m, (0.0,))
        self.assertEqual(immediate.entry_limit_m_s, 0)
        self.assertTrue(immediate.initial_overspeed)

    def test_distant_stop_never_extends_support(self):
        plan = plan_speed_profile([0, 0.3], [0, 0], LIMITS, stop_at_m=2)
        self.assertEqual(plan.stations_m[-1], 0.3)
        self.assertEqual(plan.entry_reason, "SUPPORT_END")

    def test_higher_delay_or_margin_never_increases_entry_speed(self):
        base = plan_speed_profile([0, 0.3, 0.7], [0, 2, 0], LIMITS)
        for delay in (0.8, 1.5, 3.0):
            plan = plan_speed_profile([0, 0.3, 0.7], [0, 2, 0], replace(LIMITS, delay_s=delay))
            self.assertLessEqual(plan.entry_limit_m_s, base.entry_limit_m_s)
        shorter = plan_speed_profile([0, 0.3, 0.7], [0, 2, 0], LIMITS, distance_margin_m=0.1)
        self.assertLessEqual(shorter.entry_limit_m_s, base.entry_limit_m_s)

    def test_forward_acceleration_and_initial_overspeed_are_distinct(self):
        plan = plan_speed_profile([0, 0.1, 0.2, 1, 2], [0] * 5, LIMITS, initial_speed_m_s=0)
        self.assertEqual(plan.speeds_m_s[0], 0)
        self.assertAlmostEqual(plan.speeds_m_s[1] ** 2, 2 * LIMITS.acceleration_m_s2 * 0.1)
        self.assertFalse(plan.initial_overspeed)
        overspeed = plan_speed_profile([0, 0.1], [0, 0], LIMITS, initial_speed_m_s=0.3)
        self.assertTrue(overspeed.initial_overspeed)
        self.assertLess(overspeed.speeds_m_s[0], 0.3)

    def test_random_profiles_satisfy_point_and_two_pass_physics(self):
        rng = random.Random(780)
        for _ in range(100):
            s = [0.0]
            for _ in range(30):
                s.append(s[-1] + rng.uniform(0.005, 0.08))
            k = [rng.uniform(-6, 6) for _ in s]
            plan = plan_speed_profile(s, k, LIMITS, initial_speed_m_s=rng.uniform(0, 0.3))
            for i, speed in enumerate(plan.speeds_m_s):
                self.assertGreaterEqual(speed, 0)
                self.assertLessEqual(speed, plan.point_limits_m_s[i] + 1e-12)
                self.assertLessEqual(speed * speed * abs(k[i]), LIMITS.lateral_acceleration_m_s2 + 1e-12)
                if i:
                    ds = s[i] - s[i - 1]
                    delta = speed ** 2 - plan.speeds_m_s[i - 1] ** 2
                    self.assertLessEqual(delta, 2 * LIMITS.acceleration_m_s2 * ds + 1e-12)
                    self.assertGreaterEqual(delta, -2 * LIMITS.deceleration_m_s2 * ds - 1e-12)
            # Every future cap has enough distance to coast through latency
            # and brake from the entry ceiling, assuming constant delay speed.
            for distance, cap in zip(s, plan.point_limits_m_s):
                v = plan.entry_limit_m_s
                if v > cap + 1e-12:
                    self.assertLessEqual(v * LIMITS.delay_s + (v * v - cap * cap) /
                                         (2 * LIMITS.deceleration_m_s2), distance + 1e-12)

    def test_bad_input_cannot_produce_a_driving_profile(self):
        for s, k in (([0, 0.1], [0]), ([0, 0], [0, 0]), ([1, 2], [0, 0]),
                     ([0, math.nan], [0, 0]), ([0, 1], [0, math.inf])):
            with self.assertRaises(ValueError):
                plan_speed_profile(s, k, LIMITS)
        for kwargs in ({"distance_margin_m": -1}, {"stop_at_m": math.nan}, {"initial_speed_m_s": -1}):
            with self.assertRaises(ValueError):
                plan_speed_profile([0, 1], [0, 0], LIMITS, **kwargs)
        with self.assertRaises(ValueError):
            replace(LIMITS, deceleration_m_s2=0)


class DriveReferenceTests(unittest.TestCase):
    def setUp(self):
        self.reference = DriveReference(json.loads((ROOT / "docs/动力测量参考-20261110.json").read_text()))

    def test_actual_pwm_plateau_chooses_lowest_equal_speed_duty(self):
        speed = self.reference.speed_for_pwm(150)
        self.assertEqual(speed, self.reference.speed_for_pwm(180))
        self.assertEqual(self.reference.pwm_for_speed(speed), 150)
        self.assertAlmostEqual(self.reference.speed_for_pwm(200), 0.3049522161776844)
        self.assertAlmostEqual(self.reference.pwm_for_speed(self.reference.speed_for_pwm(190)), 190)

    def test_interpolation_round_trip_and_unsupported_domain(self):
        for pwm in (90, 105, 120, 135, 149, 181, 199, 200):
            self.assertAlmostEqual(self.reference.pwm_for_speed(self.reference.speed_for_pwm(pwm)), pwm)
        for pwm in (-90, 0, 53, 89.9, 201, math.nan):
            self.assertIsNone(self.reference.speed_for_pwm(pwm))
        for speed in (0, 0.1, 0.4, math.nan):
            self.assertIsNone(self.reference.pwm_for_speed(speed))


class ReplayTests(unittest.TestCase):
    def test_history_and_gaps_never_become_observed_road(self):
        target = dict(valid=True, observed=True, state="OBSERVED", control_path_m=[[i * 0.01, 0] for i in range(15)],
                      path_observed_mask=[True] * 7 + [False] + [True] * 7)
        path, reason = supported_segment(target)
        self.assertEqual(reason, "")
        self.assertEqual(len(path), 6)
        self.assertAlmostEqual(path[-1, 0], 0.06)
        for state in ("PREDICTED", "CONFIRMING"):
            self.assertIsNone(supported_segment(dict(target, state=state))[0])
        for mask in (["false"] * 15, [True] * 2, None):
            self.assertIsNone(supported_segment(dict(target, path_observed_mask=mask))[0])
        self.assertIsNone(supported_segment(dict(target, observed=False))[0])
        self.assertIsNone(supported_segment(dict(target, control_path_m=[["bad", 0]]))[0])
        broken = dict(target, control_path_m=[[i * 0.1, 0] for i in range(15)], path_observed_mask=[True] * 15)
        self.assertIsNone(supported_segment(broken)[0])

    def test_recording_replay_is_finite_and_explicitly_offline(self):
        reference = DriveReference(json.loads((ROOT / "docs/动力测量参考-20261110.json").read_text()))
        report = replay([ROOT / "tests/fixtures/curve_oscillation.json"], LIMITS, reference,
                        front_margin_m=0.075, include_frames=True)
        self.assertFalse(report["production_commands_changed"])
        self.assertGreater(report["counts"]["segments_evaluated"], 0)
        self.assertGreater(report["counts"]["NOT_CURRENT_OBSERVATION"], 0)
        for segment in report["segments"]:
            self.assertGreater(segment["path_first_point_m"][0], 0)
            self.assertTrue(0 <= segment["entry_limit_m_s"] <= LIMITS.maximum_m_s)
        json.dumps(report, allow_nan=False)

    def test_control_log_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "app.log"
            path.write_text('noise\nCONTROL {"active": false}\n')
            self.assertEqual(load_frames(path), [{"active": False}])
            path.write_text('{"arbitrary": "dictionary"}')
            with self.assertRaises(ValueError):
                load_frames(path)


if __name__ == "__main__":
    unittest.main()
