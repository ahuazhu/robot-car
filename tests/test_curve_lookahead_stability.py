"""Recorded same-curve reversal due to abrupt adaptive lookahead changes."""

from tests.support import ROOT
import json, math, unittest
import numpy as np
from unittest.mock import patch

from robot.navigation.steering_control import SteeringControl
from robot.core.road_policy import RoadPolicy
from robot.core import geometry
from robot.core.geometry import support_projection

CAL = json.loads((ROOT / "tests/fixtures/camera_pose_20261031_191327.json").read_text())


def controller():
    return SteeringControl(RoadPolicy(), [-0.055, 0.0], [0.075, 0.105, 0.085], 0.3, calibration=CAL)


class LookaheadStabilityTests(unittest.TestCase):
    def test_same_image_repeated_evaluations_do_not_accumulate_smoothing_time(self):
        c = controller()
        c.observed_at = 1.0
        self.assertEqual(c.smooth_lookahead(0.18), 0.18)
        c.observed_at = 1.1
        first = c.smooth_lookahead(0.36)
        for _ in range(20):
            self.assertEqual(c.smooth_lookahead(0.36), first)
        self.assertGreater(first, 0.18)
        self.assertLess(first, 0.36)

    def test_response_is_time_based_and_reset_removes_previous_radius(self):
        a = controller()
        b = controller()
        for c in (a, b):
            c.observed_at = 0.0
            c.smooth_lookahead(0.18)
        a.observed_at = 0.4
        one = a.smooth_lookahead(0.36)
        for now in (0.1, 0.2, 0.3, 0.4):
            b.observed_at = now
            many = b.smooth_lookahead(0.36)
        self.assertAlmostEqual(one, many)
        self.assertAlmostEqual(one, 0.18 + (0.36 - 0.18) * (1 - math.exp(-1)))
        a.reset()
        a.observed_at = 0.5
        self.assertEqual(a.smooth_lookahead(0.18), 0.18)

    def test_explicit_alternate_probes_do_not_change_radius_state(self):
        c = controller()
        c.observed_at = 1.0
        c.smooth_lookahead(0.18)
        x = np.linspace(0.06, 0.7, 60)
        p = np.c_[x, np.zeros(len(x))]
        r = dict(
            valid=True,
            state="OBSERVED",
            observed=True,
            confidence=0.9,
            control_target_m=p[0],
            control_path_m=p,
            path=p,
            path_observed_mask=[True] * len(p),
            path_curvature_m_inv=0.0,
        )
        c.command(r, age_s=0.0, lookahead_m=0.30, alternatives=False)
        self.assertEqual(c.filtered_lookahead, 0.18)

    def test_recorded_reversal_is_removed_without_replacing_or_extending_the_curve(self):
        cases = json.loads(
            (ROOT / "tests/fixtures/curve_crosswalk_lookahead_jump.json").read_text()
        )["cases"]
        old = controller()
        new = controller()
        before = []
        after = []
        at_reversal = {}
        for case in cases:
            s = case["steering"]
            target = case["target"]
            age = s["latency_prediction"]["age_s"]
            for c in (old, new):
                c.observed_at = case["received_at"]
                c.trim = s.get("trim_m_inv", 0.0)
                c.status = {
                    k: s[k]
                    for k in (
                        "body_speed_m_s",
                        "body_yaw_rad_s",
                        "speed_upper_m_s",
                        "motion_sigma_m",
                        "motion_interval_s",
                    )
                    if k in s
                }
                c.preview_speed = s["horizon_speed_m_s"]
                c.preview_interval = s["lookahead_time_s"] / 2 - c.response_s - age
            # Reproduce the recorded pre-fix controller, including its nearest
            # far-support fallback. The new controller uses the current rule.
            with patch.object(
                geometry, "_preview_far_support", side_effect=lambda path, selected, *args: selected
            ):
                old.command(target, age_s=age, lookahead_m=s["lookahead_m"], alternatives=False)
            new.command(target, age_s=age, alternatives=False)
            self.assertAlmostEqual(old.requested, s["requested_curvature_m_inv"])
            before.append(old.requested)
            after.append(new.requested)
            selected = np.asarray(new.status["receive_target_m"])
            distance, _ = support_projection([selected], np.asarray(target["control_path_m"]))
            self.assertLess(distance[0], 1e-8)
            if case["frame"] == 870:
                at_reversal = dict(old=old.requested, new=new.requested)
        self.assertLess(at_reversal["old"], 0.0)
        self.assertGreater(at_reversal["new"], 0.0)
        self.assertLess(
            max(abs(b - a) for a, b in zip(after, after[1:])),
            max(abs(b - a) for a, b in zip(before, before[1:])),
        )


if __name__ == "__main__":
    unittest.main()
