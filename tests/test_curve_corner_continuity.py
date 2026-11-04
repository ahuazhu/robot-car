"""Short-overlap continuation uses old evidence, never an unconfirmed road."""

from tests.support import ROOT
import json, unittest
import numpy as np

from robot.navigation.tracker import TargetTracker
from robot.control.controller import CommandRamp
from robot.navigation.steering_control import SteeringControl
from robot.core.road_policy import RoadPolicy


def path(start, end, y=0.0):
    return np.c_[np.linspace(start, end, 18), np.full(18, y)]


def motion(valid=True):
    return dict(
        valid=valid,
        rotation=np.eye(2),
        translation=np.zeros(2),
        sigma_m=0.001,
        interval_s=0.1,
        travel_m=0.0,
        speed_m_s=0.06,
        yaw_rad_s=0.0,
        source="GROUND FLOW",
        reason="",
        inliers=20,
        residual_m=0.001,
    )


class ContinuityTests(unittest.TestCase):
    def tracker(self):
        c = json.loads((ROOT / "tests/fixtures/camera_pose_20261031_191327.json").read_text())
        c["control_frame"] = dict(
            origin_m=[-0.055, 0.0], body_m=[0.075, 0.105, 0.085], geometry_verified=True
        )
        t = TargetTracker(c)
        t.motion.update = lambda *args: motion()
        t.pixels.measure = lambda *args: (
            dict(path=path(0.12, 0.25), confidence=0.9, mode="ONE LEFT"),
            {},
        )
        t.update(None, 0.0, require_current=True)
        return t

    def candidate(self, t, y=0.02):
        t.pixels.measure = lambda *args: (
            dict(path=path(0.24, 0.65, y), confidence=0.9, mode="ONE LEFT"),
            {},
        )

    def test_small_compatible_successor_does_not_pause_or_refresh_old_support(self):
        for sign in (-1, 1):
            t = self.tracker()
            old = t.model.path.copy()
            seen = t.model.seen.copy()
            self.candidate(t, sign * 0.02)
            r, _ = t.update(None, 0.1)
            self.assertTrue(r["transition_pending"])
            self.assertEqual(r["state"], "PREDICTED")
            self.assertFalse(r["observed"])
            self.assertIsNotNone(r["history_continuity_evidence"])
            np.testing.assert_allclose(r["path"], old)
            np.testing.assert_array_equal(t.model.seen, seen)
            self.assertIsNotNone(r["candidate_path_m"])
            self.assertFalse(any(r["path_observed_mask"]))

    def test_repeated_compatible_candidate_can_be_confirmed_without_stopping(self):
        t = self.tracker()
        self.candidate(t)
        first, _ = t.update(None, 0.1)
        self.assertFalse(first["observed"])
        second, _ = t.update(None, 0.2)
        self.assertTrue(second["observed"])
        self.assertFalse(second["transition_pending"])
        self.assertEqual(t.model.pending_count, 0)
        self.assertTrue(np.all(t.model.seen == 0.2))
        self.assertIsNone(second["history_continuity_evidence"])
        controller = SteeringControl(RoadPolicy(), [-0.055, 0.0], [0.075, 0.105, 0.085], 0.3)
        self.assertIsNotNone(controller.command(second))

    def test_startup_expiry_dropout_and_wide_shift_cannot_continue(self):
        for case in ("startup", "expired", "motion", "shift"):
            t = self.tracker()
            self.candidate(t, 0.12 if case == "shift" else 0.02)
            if case == "motion":
                t.motion.update = lambda *args: motion(False)
            r, _ = t.update(
                None, 0.7 if case == "expired" else 0.1, require_current=case == "startup"
            )
            self.assertFalse(r["transition_pending"], case)
            self.assertIsNone(r["history_continuity_evidence"], case)

    def test_physical_corridor_width_controls_continuation_not_camera_side(self):
        t = self.tracker()
        t.control_body = [0.075, 0.105, 0.145]
        self.candidate(t)
        r, _ = t.update(None, 0.1)
        self.assertFalse(r["transition_pending"])


class RollingRestartTests(unittest.TestCase):
    def ramp(self):
        r = CommandRamp()
        for i in range(10):
            r.apply((100, 60), i * 0.1)
        return r

    def test_short_rolling_pause_outputs_zero_but_does_not_restart_at_first_step(self):
        r = self.ramp()
        self.assertEqual(r.pause(1.0, rolling=True), (0, 0))
        out = r.apply((100, 60), 1.1)
        cold = CommandRamp().apply((100, 60), 1.1)
        self.assertGreater(max(out), max(cold))
        self.assertLessEqual(max(out), 100)
        self.assertAlmostEqual(out[1] / out[0], 0.6, delta=0.02)

    def test_explicit_stop_no_motion_and_long_pause_reset_resume_effort(self):
        for how in ("zero", "notrolling", "long"):
            r = self.ramp()
            if how == "zero":
                self.assertEqual(r.apply((0, 0), 1.0), (0, 0))
            else:
                self.assertEqual(
                    r.pause(2.0 if how == "long" else 1.0, rolling=how == "long"), (0, 0)
                )
            self.assertLessEqual(max(r.apply((100, 60), 2.1)), 18)


if __name__ == "__main__":
    unittest.main()
