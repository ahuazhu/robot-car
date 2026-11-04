"""Physical horizon, safe branch handover, body origin and measured yaw feedback."""

from tests.support import ROOT
import unittest, math, json
from pathlib import Path
import numpy as np

from robot.core.geometry import choose_target, arc, fit_curve
from robot.core.road_policy import RoadPolicy
from robot.navigation.steering_control import SteeringControl


def line(start=-0.25, end=0.75, y=0.0):
    return np.c_[np.linspace(start, end, 90), np.full(90, y)]


def motion(distance=0.0, yaw=0.0, dt=0.1, valid=True):
    c, s = math.cos(yaw), math.sin(yaw)
    return dict(
        valid=valid,
        rotation=np.array([[c, s], [-s, c]]),
        translation=np.array([-distance, 0.0]),
        interval_s=dt,
        travel_m=distance,
        sigma_m=0.0002,
        speed_m_s=distance / dt,
        yaw_rad_s=yaw / dt,
    )


class ReferenceTests(unittest.TestCase):
    def test_nearest_rule_applies_to_near_and_far_curves_without_radius_modes(self):
        from robot.core.geometry import nearest_forward_point

        for start in (0.12, 0.25, 0.38, 0.5):
            p = line(start, 0.75, 0.01)
            chosen = choose_target(p)
            self.assertIsNotNone(chosen)
            expected, _ = nearest_forward_point(chosen["reference_path"])
            np.testing.assert_allclose(chosen["point"], expected, atol=1e-10)
            self.assertEqual(chosen["target_selection"], "NEAREST_CURVE")

    def test_reference_is_chassis_center_not_front_axle(self):
        origin = np.array([-0.055, 0.0])
        p = line(0.1, 0.75)
        chosen = choose_target(p, reference_origin_m=origin)
        self.assertAlmostEqual(chosen["reference_origin_m"][0], -0.055)
        self.assertGreaterEqual(chosen["point"][0], 0.06)

    def test_far_straight_does_not_invent_near_approach_geometry(self):
        p = line(0.35, 0.7)
        copy = p.copy()
        chosen = choose_target(p)
        np.testing.assert_array_equal(p, copy)
        np.testing.assert_array_equal(chosen["reference_path"], p)
        self.assertGreaterEqual(chosen["point"][0], 0.35 - 1e-8)

    def test_occlusion_uses_measured_motion_history_without_refreshing_observation(self):
        from robot.navigation.tracker import TargetTracker

        cal = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "tests/fixtures/camera_pose_20261031_191327.json"
            ).read_text()
        )
        tracker = TargetTracker(cal)
        observed = line(0.25, 0.7)
        tracker.pixels.measure = lambda *args: (
            dict(path=observed, confidence=0.85, mode="ONE RIGHT"),
            {},
        )
        first, _ = tracker.update(None, 0.0)
        self.assertFalse(first["target_inferred"])
        tracker.pixels.measure = lambda *args: (None, {})
        measured = motion(0.01, 0.0, dt=0.2, valid=True)
        tracker.motion.update = lambda *args: measured
        predicted, _ = tracker.update(None, 0.2)
        self.assertTrue(predicted["valid"])
        self.assertEqual(predicted["state"], "PREDICTED")
        self.assertTrue(predicted["target_inferred"])
        self.assertEqual(predicted["control_target_source"], "HISTORY PREDICTION")
        self.assertFalse(any(predicted["path_observed_mask"]))
        np.testing.assert_allclose(
            np.asarray(predicted["path"]), observed - [0.01, 0.0], atol=1e-10
        )
        self.assertTrue(np.all(tracker.model.seen == 0.0))
        self.assertEqual(tracker.model.last_seen, 0.0)

    def test_empty_or_behind_curve_cannot_generate_target(self):
        self.assertIsNone(choose_target(line(-0.8, -0.4)))
        self.assertIsNone(choose_target([]))

    def test_prediction_and_nearest_rule_are_left_right_symmetric(self):
        p = line(0.2, 0.7, 0.08)
        a = choose_target(p)
        b = choose_target(p * [1, -1])
        np.testing.assert_allclose(a["point"] * [1, -1], b["point"], atol=1e-10)


class FeedbackTests(unittest.TestCase):
    def control(self, body=None):
        return SteeringControl(RoadPolicy(), [0.0, 0.0], body, 0.3)

    def test_yaw_bias_converges_using_time_aligned_past_setpoints(self):
        c = self.control()
        c.requested = 0.0
        c.applied(0, (40, 40))
        for i in range(1, 151):
            # A leftward drivetrain bias of 0.8/m under a straight setpoint.
            observed = c.trim + 0.8
            c.observe(motion(0.015, observed * 0.015), i * 0.1)
            c.applied(i * 0.1, (40, 40))
        self.assertAlmostEqual(c.trim, -0.8, delta=0.025)

    def test_invalid_motion_and_dwell_cannot_train_a_false_trim(self):
        c = self.control()
        c.requested = 1.0
        c.applied(0, (40, 40))
        c.observe(motion(0.02, 0, valid=False), 0.1)
        self.assertEqual(c.trim, 0.0)
        c.applied(0.1, (0, 0))
        c.observe(motion(0.02, 0), 0.2)
        self.assertEqual(c.trim, 0.0)

    def test_saturated_command_cannot_wind_up_integrator(self):
        c = self.control()
        c.requested = 4.0
        c.applied(0, (0, 60))
        for i in range(1, 10):
            c.observe(motion(0.02), i * 0.1)
        self.assertEqual(c.trim, 0.0)

    def test_body_envelope_blocks_a_path_already_under_a_corner(self):
        c = self.control([0.075, 0.105, 0.085])
        bad = c.envelope(line(y=0.10), 0.0, 0.005, 0.3)
        self.assertEqual(bad["speed_scale"], 0.0)
        self.assertEqual(bad["first_unsafe_m"], 0.0)
        good = c.envelope(line(), 0.0, 0.005, 0.3)
        self.assertGreater(good["speed_scale"], 0.9)
        self.assertGreater(good["minimum_clearance_m"], 0.05)

    def test_body_sweep_is_symmetric_and_partial_support_is_reported(self):
        c = self.control([0.075, 0.105, 0.085])
        x = np.linspace(-0.2, 0.7, 100)
        p = np.c_[x, 0.6 * x * x]
        a = c.envelope(p, 1.0, 0.005, 0.3)
        b = c.envelope(p * [1, -1], -1.0, 0.005, 0.3)
        self.assertAlmostEqual(a["speed_scale"], b["speed_scale"])
        partial = c.envelope(line(0.2, 0.7), 0.0, 0.005, 0.3)
        self.assertEqual(partial["state"], "PARTIAL")
        self.assertLess(partial["coverage"], 1.0)

    def test_inferred_prefix_cannot_trigger_a_false_current_body_collision(self):
        c = self.control([0.075, 0.105, 0.085])
        inferred = line(y=0.1)
        observed = line(0.3, 0.7, y=0.1)
        result = c.envelope(inferred, 0.0, 0.005, 0.3, observed_path=observed)
        self.assertNotEqual(result["first_unsafe_m"], 0.0)
        self.assertEqual(result["state"], "PARTIAL")

    def test_configuration_rejects_nonphysical_geometry(self):
        for body in ([0.075, -0.1, 0.085], [float("nan"), 0.1, 0.085]):
            with self.assertRaises(ValueError):
                self.control(body)


class RecordedStopTests(unittest.TestCase):
    def test_recorded_unseen_body_corners_are_not_current_collisions(self):
        root = Path(__file__).resolve().parents[1]
        cal = json.loads((root / "tests/fixtures/camera_pose_20261031_191327.json").read_text())
        fixture = json.loads((root / "tests/fixtures/curve_body_visibility_stop.json").read_text())
        target = fixture["target"]
        origin = np.array([-0.055, 0.0])
        point = np.array(target["control_target_m"])
        curvature = 1.2 * 2 * point[1] / float(point @ point) + fixture["steering"]["trim_m_inv"]
        args = (
            np.array(target["control_path_m"]),
            curvature,
            target["road_model"]["sigma_m"],
            target["lookahead_arc_m"],
        )
        observed = np.array(target["road_path_m"]) - origin
        unbounded = SteeringControl(RoadPolicy(), origin, [0.075, 0.105, 0.085], 0.3)
        self.assertEqual(unbounded.envelope(*args, observed_path=observed)["first_unsafe_m"], 0.0)
        visible = SteeringControl(RoadPolicy(), origin, [0.075, 0.105, 0.085], 0.3, calibration=cal)
        checked = visible.envelope(*args, observed_path=observed)
        self.assertNotEqual(checked["first_unsafe_m"], 0.0)
        self.assertEqual(checked["state"], "PARTIAL")

    def test_speed_constraint_uses_measured_speed_not_pwm_proxy(self):
        c = SteeringControl(RoadPolicy(), [0.0, 0.0], [0.075, 0.105, 0.085], 0.3)
        args = (line(), 0.0, 0.005, 0.3)
        conservative = c.envelope(*args, observed_path=line(0.25, 0.7, y=0.1))
        measured = c.envelope(*args, observed_path=line(0.25, 0.7, y=0.1), speed_m_s=0.03)
        self.assertLess(conservative["speed_scale"], 1.0)
        self.assertGreater(measured["speed_scale"], conservative["speed_scale"])

    def test_changing_wheel_commands_cannot_train_a_false_bias(self):
        c = SteeringControl(RoadPolicy(), [0.0, 0.0], None, 0.3)
        c.requested = 1.0
        for i in range(10):
            c.applied(i * 0.1, (20 + 3 * i, 20 + 3 * i))
            c.observe(motion(0.015), i * 0.1 + 0.1)
        self.assertEqual(c.trim, 0.0)

    def test_inactive_observation_measures_speed_without_integrating(self):
        c = SteeringControl(RoadPolicy(), [0.0, 0.0], None, 0.3)
        c.requested = 1.0
        c.applied(0.0, (40, 40))
        c.observe(motion(0.015), 0.1, allow_feedback=False)
        self.assertEqual(c.trim, 0.0)
        self.assertGreater(c.status["speed_upper_m_s"], 0.15)


class PreviewContractTests(unittest.TestCase):
    def test_valid_road_with_unavailable_control_goal_remains_renderable(self):
        from robot.interface.road_overlay import render_target

        cal = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "tests/fixtures/camera_pose_20261031_191327.json"
            ).read_text()
        )
        image = np.zeros((240, 320, 3), np.uint8)
        details = dict(fits={}, bird=image)
        for state in ("CONFIRMING", "OBSERVED", "LOST"):
            result = dict(
                valid=state != "LOST", state=state, path=line(0.12, 0.6).tolist(), target_m=None
            )
            out, bird = render_target(image, result, details, cal)
            self.assertEqual(out.shape, image.shape)
            self.assertEqual(bird.shape, image.shape)


if __name__ == "__main__":
    unittest.main()


class ControlTimeTests(unittest.TestCase):
    def control(self):
        return SteeringControl(RoadPolicy(), [0.0, 0.0], None, 0.3, gain=1.0)

    def result(self, point):
        return dict(
            valid=True,
            state="OBSERVED",
            confidence=1.0,
            control_target_m=point,
            path_curvature_m_inv=0.0,
            control_path_m=line(point[0], point[0] + 0.5, point[1]),
            path=[],
        )

    def measured(self, c, speed, yaw, dt=0.2):
        c.observe(motion(speed * dt, yaw * dt, dt=dt), 2.0, allow_feedback=False)
        # origin zero; set exact body twist so this test isolates command-time SE(2).
        c.status.update(body_speed_m_s=speed, body_yaw_rad_s=yaw)

    def test_zero_age_keeps_geometric_pure_pursuit(self):
        c = self.control()
        r = self.result([0.25, -0.04])
        c.command(r)
        q = np.asarray(c.status["command_target_m"])
        self.assertAlmostEqual(c.requested, 2 * q[1] / float(q @ q))

    def test_turn_exit_accounts_for_rotation_during_processing_delay(self):
        c = self.control()
        self.measured(c, 0.13, -0.36)
        r = self.result([0.26, -0.02])
        before = 2 * (-0.02) / (0.26**2 + 0.02**2)
        self.assertIsNotNone(c.command(r, age_s=0.2))
        self.assertGreater(c.requested, before)
        self.assertTrue(c.status["latency_prediction"]["applied"])
        self.assertAlmostEqual(c.status["latency_prediction"]["yaw_rad"], -0.072)
        self.assertEqual(r["control_target_m"], [0.26, -0.02])

    def test_delay_prediction_is_left_right_symmetric(self):
        a, b = self.control(), self.control()
        self.measured(a, 0.13, -0.36)
        self.measured(b, 0.13, 0.36)
        ca = a.command(self.result([0.26, -0.02]), age_s=0.2)
        cb = b.command(self.result([0.26, 0.02]), age_s=0.2)
        self.assertEqual(ca, cb[::-1])
        self.assertAlmostEqual(a.requested, -b.requested)

    def test_invalid_flow_does_not_invent_motion_or_reuse_previous_twist(self):
        c = self.control()
        self.measured(c, 0.13, -0.36)
        c.observe(motion(valid=False), 2.2, allow_feedback=False)
        c.command(self.result([0.26, -0.02]), age_s=0.2)
        self.assertFalse(c.status["latency_prediction"]["applied"])
        q = np.asarray(c.status["command_target_m"])
        self.assertAlmostEqual(c.requested, 2 * q[1] / float(q @ q))

    def test_expired_frame_cannot_be_made_fresh_by_prediction(self):
        c = self.control()
        self.measured(c, 0.13, -0.36)
        self.assertIsNone(
            c.command(self.result([0.26, -0.02]), age_s=c.policy.missing_command_s + 0.01)
        )

    def test_passed_target_is_not_pursued_behind_chassis(self):
        c = self.control()
        self.measured(c, 0.5, 0.0)
        self.assertIsNone(
            c.command(
                dict(self.result([0.1, 0.0]), control_path_m=[[0.1, 0.0], [0.11, 0.0]]), age_s=0.25
            )
        )

    def test_motion_alignment_reselects_lookahead_point_on_same_curve(self):
        from robot.core.geometry import pursuit_point

        c = self.control()
        self.measured(c, 0.2, 0.0)
        p = line(-0.2, 0.8, 0.02)
        point = choose_target(p)["point"]
        r = dict(self.result(point), control_path_m=p)
        c.command(r, age_s=0.25)
        q = c.status["latency_prediction"]["point_m"]
        expected = pursuit_point(p - [0.05, 0.0], c.status["lookahead_m"])["point"]
        np.testing.assert_allclose(q, expected, atol=1e-10)

    def test_constant_curvature_tracking_is_invariant_to_processing_delay(self):
        from robot.core.geometry import nearest_forward_point

        c = self.control()
        self.measured(c, 0.15, 0.3)
        angle = np.linspace(-0.5, 1.5, 2400)
        p = np.c_[0.5 * np.sin(angle), 0.5 * (1 - np.cos(angle))]
        point, _ = nearest_forward_point(p)
        r = dict(self.result(point), control_path_m=p)
        c.command(r, age_s=0.2)
        self.assertAlmostEqual(c.requested, 2.0, delta=0.002)
