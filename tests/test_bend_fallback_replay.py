"""Recorded fallback boundaries retain current intent and motor supervision.

These are single-step control regressions, not full camera/vehicle replays.
The fixture documents missing previous CONTROL frames and body-envelope
oracles. Geometry, current requested turns and prior published PWM are real.
"""
import unittest

from tests.support.fallback_replay import command_step
from tests.support.driving import road
from tests import test_session as session_tests
from tests import test_bend_response as bend_tests
from robot.control.controller import compensate_pwm


class BendFallbackReplayTests(unittest.TestCase):
    def assert_model_delivery(self, replay, ceiling):
        profile = replay["profile"]
        allocation = profile.status["bend_allocation"]
        self.assertTrue(allocation["active"])
        self.assertEqual(list(replay["wire"]), allocation["wire"])
        speed, yaw = profile.bend_response.predict(replay["wire"])
        self.assertAlmostEqual(speed, allocation["estimated_speed_m_s"])
        self.assertAlmostEqual(yaw, allocation["estimated_yaw_rad_s"])
        self.assertLessEqual(speed, ceiling + 1e-9)
        self.assertLessEqual(speed * abs(yaw), profile.limits.lateral_acceleration_m_s2 + 1e-9)
        self.assertLessEqual(max(replay["wire"]), 255)
        self.assertAlmostEqual(allocation["requested_curvature_m_inv"],
                               replay["steering"].status["requested_curvature_m_inv"])
        self.assertAlmostEqual(replay["checked_curvatures"][-1], yaw / speed)
        return allocation

    def test_recorded_geometry_reconstructs_the_current_turn_not_previous_turn(self):
        for name in ("short_support", "reference_drop", "body_withdrawal"):
            with self.subTest(case=name):
                replay = command_step(name)
                recorded = replay["case"]["steering"]
                current = replay["steering"].status
                self.assertAlmostEqual(current["requested_curvature_m_inv"],
                                       recorded["requested_curvature_m_inv"], places=12)
                self.assertEqual(current["target_mode"], recorded["target_mode"])
                self.assertEqual(replay["case"]["reconstruction"]["complete_previous_control"], False)

    def test_short_support_keeps_current_turn_through_pwm_model_and_real_ramp(self):
        replay = command_step("short_support")
        profile = replay["profile"]
        self.assertFalse(profile.status["selected"])
        self.assertEqual(profile.status["actuator_model_source"], "PWM")
        allocation = self.assert_model_delivery(replay,
            replay["case"]["steering"]["envelope"]["speed_limit_m_s"])
        self.assertEqual(allocation["reference_kind"], "EFFORT")
        self.assertIsNone(allocation["target_speed_m_s"])
        old_speed, old_yaw = profile.bend_response.predict(replay["case"]["recorded_output"]["pwm"])
        desired = allocation["target_curvature_m_inv"]
        self.assertLess(abs(desired-allocation["estimated_curvature_m_inv"]),
                        abs(desired-old_yaw/old_speed))

    def test_reference_drop_replans_current_turn_and_reports_unreachable_speed(self):
        replay = command_step("reference_drop")
        profile = replay["profile"]
        allocation = self.assert_model_delivery(replay,
            replay["case"]["steering"]["dwpp_envelope"]["speed_limit_m_s"])
        self.assertTrue(profile.status["selected"])
        self.assertFalse(profile.status["motion_feedback"])
        self.assertIsNone(profile.status["measured_speed_m_s"])
        self.assertLess(profile.status["target_speed_m_s"],
                        replay["case"]["previous_trace"]["model"]["reference_v"])
        self.assertNotEqual(list(replay["wire"]), replay["case"]["previous_trace"]["pwm"])
        self.assertTrue(allocation["limited_reference"])
        self.assertTrue(allocation["limited_speed"])
        self.assertGreater(allocation["feasible_reference_range"][0],
                           allocation["target_speed_m_s"])

    def test_tighter_actual_arc_resolves_a_model_pair_and_checks_its_new_curvature(self):
        for cap in (.19, .18):
            with self.subTest(cap=cap):
                replay = command_step("short_support", envelopes=lambda index, curvature: dict(
                    speed_scale=1., speed_limit_m_s=.3 if index == 0 else cap,
                    minimum_clearance_m=.01))
                self.assert_model_delivery(replay, cap)
                self.assertGreaterEqual(len(replay["checked_curvatures"]), 3)

    def test_unmodeled_braking_is_explicit_and_zero_cap_never_launches(self):
        replay = command_step("body_withdrawal")
        self.assertFalse(replay["profile"].status["bend_allocation"]["active"])
        self.assertEqual(replay["profile"].status["actuator_model_reason"], "UNCALIBRATED_BRAKING")
        self.assertIsNone(replay["profile"].bend_proposal)
        # No model-speed claim is made for the retained low-range brake.
        stopped = command_step("short_support", envelopes=lambda index, curvature: dict(
            speed_scale=1., speed_limit_m_s=.3 if index == 0 else 0.,
            minimum_clearance_m=.01))
        self.assertEqual(stopped["wire"], (0, 0))
        self.assertFalse(stopped["profile"].status["bend_allocation"]["active"])

    def test_profile_low_speed_and_old_bend_gate_still_reach_the_shared_model(self):
        for speed, allowed in ((.20, False), (.08, True), (.08, False)):
            with self.subTest(speed=speed, allowed=allowed):
                profile = bend_tests.BendResponseTests().control(gain=1.05)
                profile.applied(.1, (80., 80.))
                target = road(-2., reference_sigma_m=.005)
                output = profile.command(target, target["control_path_m"], .3, speed,
                    -2., .005, .075, True, measured_yaw_rate=-2.*speed,
                    bend_allowed=allowed, bend_inner_reduction=True)
                allocation = profile.status["bend_allocation"]
                self.assertTrue(allocation["active"])
                self.assertEqual(list(compensate_pwm(output)), allocation["wire"])
                self.assertEqual(profile.status["actuator_model_source"], "DWPP")
                self.assertIsNotNone(profile.bend_response.predict(allocation["wire"]))
                self.assertLessEqual(allocation["estimated_speed_m_s"],
                                     profile.status["bend_speed_ceiling_m_s"]+1e-9)
        # A zero requested effort cannot obtain a calibrated launch pulse even
        # when the curvature and requested reference would have table support.
        profile = bend_tests.BendResponseTests().control(gain=1.05)
        profile.applied(.1, (0., 0.))
        target = road(-2., reference_sigma_m=.005)
        output = profile.command(target, target["control_path_m"], .3, .4,
            -2., .005, .075, True, measured_yaw_rate=-.8,
            bend_allowed=True, bend_inner_reduction=True)
        self.assertEqual(output, (0., 0.))
        self.assertFalse(profile.status["bend_allocation"]["active"])
        self.assertEqual(profile.status["actuator_model_reason"], "STOP_REQUESTED")

    def test_manual_stop_overrides_the_replayed_model_and_cannot_restart_it(self):
        replay = command_step("reference_drop")
        self.assertIsNotNone(replay["output"])
        helper = session_tests.DrivingSessionTests()
        helper.setUp()
        try:
            session = helper.session()
            helper.start(session)
            session.steering, session.ramp = replay["steering"], replay["ramp"]
            before = len(session.link.calls)
            helper.clock.now += .1
            frame = helper.frame(session, command=replay["output"], follow_command=replay["output"])
            session.update(frame, ("stop",))
            self.assertEqual(session.pwm, (0, 0))
            self.assertFalse(session.active)
            self.assertIsNone(replay["profile"].bend_proposal)
            self.assertIsNone(replay["profile"].dwpp_proposal)
            helper.clock.now += .1
            session.update(helper.frame(session, command=replay["output"],
                                        follow_command=replay["output"]))
            self.assertEqual(session.pwm, (0, 0))
            self.assertFalse(session.active)
            self.assertFalse(any(op == 2 and (left or right)
                                 for op, left, right, _ in session.link.calls[before:]))
        finally:
            helper.tearDown()


if __name__ == "__main__":
    unittest.main()
