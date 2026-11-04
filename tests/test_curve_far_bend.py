"""A far transverse road must convey its direction before it leaves view."""

from tests.support import ROOT
import json, unittest
from unittest.mock import patch
import numpy as np

from robot.core import geometry
from robot.core.geometry import pursuit_point, support_projection, arc
from robot.navigation.steering_control import SteeringControl
from robot.core.road_policy import RoadPolicy

FIXTURE = json.loads((ROOT / "tests/fixtures/bend_far_support_20261101.json").read_text())


class FarBendTests(unittest.TestCase):
    def test_recorded_transverse_bends_turn_early_on_supported_curve(self):
        for row in FIXTURE:
            for mirror in (1, -1):
                path = np.asarray(row["path"]) * [1, mirror]
                r = pursuit_point(path, row["lookahead_m"])
                q = r["point"]
                k = row["gain"] * 2 * q[1] / float(q @ q)
                self.assertEqual(r["mode"], "FAR_SUPPORT")
                self.assertGreater(k * mirror, 0.5)
                self.assertLess(abs(row["old_curvature"]), 0.25)
                self.assertLess(support_projection([q], path)[0][0], 1e-10)
                self.assertLessEqual(r["position_m"], arc(path)[-1] + 1e-10)
                self.assertLess(r["speed_scale"], 1.0)

    def test_far_rule_converges_to_circle_rule_without_target_jump(self):
        # Near and far sides of the threshold both approach the same anchor.
        L = 0.25
        results = []
        for d in (L - 1e-8, L, L + 1e-8):
            p = np.array([[d, -0.1], [d, 0], [d, 0.3]])
            results.append(pursuit_point(p, L)["point"])
        self.assertLess(np.linalg.norm(results[0] - results[1]), 0.0001)
        self.assertLess(np.linalg.norm(results[2] - results[1]), 0.0001)

    def test_far_preview_stops_at_forward_domain_break(self):
        p = np.array([[0.3, 0], [0.3, 0.08], [0.04, 0.12], [0.6, 0.5]])
        r = pursuit_point(p, 0.06)
        # The minimum is on the first forward branch and the advance cannot jump
        # over the intervening behind-domain segment to the remote final branch.
        self.assertLess(r["position_m"], arc(p)[2])
        self.assertGreaterEqual(r["point"][0], 0.06 - 1e-12)

    def test_far_preview_is_clipped_at_maximum_range(self):
        p = np.array([[0.6, 0], [0.8, 0.3]])
        r = pursuit_point(p, 0.18)
        self.assertAlmostEqual(r["distance_m"], 0.65)
        self.assertLess(support_projection([r["point"]], p)[0][0], 1e-10)
        self.assertIsNone(pursuit_point(p + [0.1, 0], 0.18))

    def test_short_remaining_far_branch_ends_without_extrapolation(self):
        p = np.array([[0.35, 0], [0.35, 0.025]])
        r = pursuit_point(p, 0.18)
        np.testing.assert_allclose(r["point"], p[-1])

    def test_native_result_also_receives_far_preview(self):
        p = np.array([[0.3, 0], [0.3, 0.2]])
        native = dict(
            point=np.array([0.3, 0]),
            position_m=0.0,
            distance_m=0.3,
            mode="FAR_SUPPORT",
            speed_scale=0.6,
        )
        with patch.object(geometry, "native_pursuit_selection", return_value=(True, native)):
            r = pursuit_point(p, 0.18)
        self.assertGreater(r["point"][1], 0.1)
        self.assertEqual(r["mode"], "FAR_SUPPORT")

    def test_far_bend_stays_ineligible_for_high_speed(self):
        row = FIXTURE[0]
        p = np.asarray(row["path"])
        c = SteeringControl(
            RoadPolicy(),
            [0, 0],
            (0.075, 0.105, 0.085),
            0.3,
            gain=0.85,
            cruise=120,
            straight_cruise=163,
            confirmed_cruise=140,
        )
        motion = dict(
            valid=True,
            interval_s=0.15,
            rotation=np.eye(2),
            translation=np.array([-0.01, 0]),
            sigma_m=0.001,
        )
        result = dict(
            valid=True,
            state="OBSERVED",
            observed=True,
            confidence=0.8,
            observation_confidence=0.8,
            control_target_m=p[0],
            control_path_m=p,
            path=p,
            path_observed_mask=[True] * len(p),
            path_curvature_m_inv=3.45,
        )
        for now in (0.0, 0.2, 0.4, 0.6):
            c.observe(motion, now, False)
            command = c.command(result, 0.0)
        self.assertIsNotNone(command)
        self.assertGreater(c.requested, 0.5)
        self.assertEqual(c.status["target_mode"], "FAR_SUPPORT")
        self.assertFalse(c.status["speed_plan"]["boost_eligible"])
        self.assertEqual(c.status["speed_plan"]["selected_cruise_pwm"], 120)


if __name__ == "__main__":
    unittest.main()
