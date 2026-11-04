"""Geometry/freshness/motion invariants, including both recorded false stops."""

from tests.support import ROOT
import json, unittest
import cv2
import numpy as np

from robot.core.geometry import fit_curve, merge_centers, choose_target, curve_agreement
from robot.navigation.road_model import LocalRoadModel, BoundaryTrackBank
from robot.core.road_policy import RoadPolicy
from robot.core.ground_motion import GroundMotion, command_motion, apply_ground_motion
from robot.navigation.tracker import TargetTracker
from robot.perception.boundaries import CurvePixels

CAL = json.loads((ROOT / "maixcam2/curve_follow/calibration.json").read_text())
# Historical recordings retain their capture-time camera pose.
RECORDED_CAL = json.loads(
    (ROOT / "tests/fixtures/road_model_recording_calibration.json").read_text()
)


def line(start=0.12, end=0.65, y=0.0):
    return fit_curve(np.column_stack((np.linspace(start, end, 60), np.full(60, y))))


def measure(curve, quality=0.85, mode="ONE LEFT"):
    return dict(path=curve["path"], confidence=quality, mode=mode)


class SupportUnionTests(unittest.TestCase):
    def test_far_second_edge_never_discards_near_single_support(self):
        a, b = line(0.20, 0.55, 0.003), line(0.38, 0.70, -0.003)
        for left, right in ((a, b), (b, a)):
            result = merge_centers(left, right)
            self.assertIsNotNone(result)
            self.assertLess(result["path"][0, 0], 0.21)
            self.assertGreater(result["path"][-1, 0], 0.69)
            self.assertIsNotNone(choose_target(result["path"]))

    def test_divergent_far_branch_retains_consistent_near_component(self):
        a = line(0.12, 0.7)
        x = np.linspace(0.3, 0.7, 60)
        b = fit_curve(np.column_stack((x, 3 * np.maximum(x - 0.45, 0) ** 2)))
        result = merge_centers(a, b)
        self.assertIsNotNone(result)
        self.assertLess(result["path"][0, 0], 0.13)
        self.assertLess(result["path"][-1, 0], 0.7)

    def test_crossed_reversed_or_disconnected_candidates_do_not_merge(self):
        a = line()
        for b in (line(y=0.2), line(0.8, 1.2), fit_curve(a["path"][::-1])):
            self.assertIsNone(merge_centers(a, b))

    def test_handedness_and_argument_order_do_not_change_geometry(self):
        a, b = line(0.2, 0.55, 0.008), line(0.38, 0.7, -0.004)
        result = merge_centers(a, b)
        reverse = merge_centers(b, a)
        mirror = merge_centers(fit_curve(a["path"] * [1, -1]), fit_curve(b["path"] * [1, -1]))
        np.testing.assert_allclose(result["path"], reverse["path"], atol=1e-9)
        np.testing.assert_allclose(result["path"] * [1, -1], mirror["path"], atol=1e-6)

    def test_recorded_fused_far_start_now_preserves_near_observation(self):
        folder = ROOT / "tests/fixtures/road_model_sequence"
        for frame in (862, 863, 864):
            raw = cv2.imread(str(folder / ("frame-%06d.png" % frame)))
            pixels = CurvePixels(RECORDED_CAL)
            m, details = pixels.measure(raw)
            self.assertIsNotNone(m)
            self.assertLess(np.linalg.norm(m["path"], axis=1).min(), 0.30)
            self.assertIsNotNone(choose_target(m["path"]))


class PersistentRoadTests(unittest.TestCase):
    def test_far_refresh_does_not_refresh_unseen_near_points(self):
        model = LocalRoadModel()
        model.update(measure(line()), 0)
        for now in (0.2, 0.4):
            event = model.update(measure(line(0.36, 0.70), mode="ONE RIGHT"), now)
            self.assertTrue(event["observed"])
        near = model.seen[model.path[:, 0] < 0.30]
        self.assertTrue(len(near) > 0)
        self.assertTrue(np.all(near == 0.0))
        support, info = model.support(0.65)
        self.assertIsNotNone(support)
        self.assertGreater(support[0, 0], 0.34)
        self.assertLess(info["age_s"], 0.3)

    def test_no_target_does_not_prevent_geometry_update(self):
        t = TargetTracker(CAL)
        t.pixels.measure = lambda *args: (measure(line(0.75, 1.0)), {})
        result, _ = t.update(None, 0.0)
        self.assertTrue(result["observed"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["invalid_reason"], "NO FORWARD CURVE POINT")
        self.assertEqual(t.model.last_seen, 0.0)
        self.assertIsNotNone(t.model.path)

    def test_global_model_refresh_does_not_hide_stale_support(self):
        model = LocalRoadModel()
        model.update(measure(line()), 0)
        model.update(measure(line(0.4, 0.7)), 0.4)
        _, info = model.support(0.61)
        self.assertLess(info["oldest_seen"], 0.401)
        self.assertGreater(info["oldest_seen"], 0.399)

    def test_prediction_cannot_create_a_new_observation(self):
        model = LocalRoadModel()
        model.update(measure(line()), 0)
        old = model.seen.copy()
        model.predict(command_motion(0.05, 0.2, 0.1, RoadPolicy()))
        np.testing.assert_array_equal(model.seen, old)
        self.assertEqual(model.last_seen, 0.0)
        self.assertGreater(model.variance.min(), RoadPolicy().observation_sigma_m ** 2)

    def test_long_unknown_motion_is_not_silently_clamped_to_half_second(self):
        t = TargetTracker(CAL)
        t.pixels.measure = lambda *args: (measure(line()), {})
        t.update(None, 0)
        t.pixels.measure = lambda *args: (None, {})
        result, _ = t.update(None, 1.0, 0.1, 0)
        self.assertFalse(result["valid"])
        self.assertGreater(result["blind_distance_m"], 0.09)

    def test_single_boundary_track_keeps_identity_and_missing_age(self):
        bank = BoundaryTrackBank()
        f = dict(line(), confidence=0.9, score=0.9)
        bank.update({"left": [f]}, 0)
        identity = bank.summary(0)["left"][0]["id"]
        bank.predict(command_motion(0, 0, 0.1, RoadPolicy()), 0.1)
        bank.update({"left": [dict(line(0.2, 0.7), confidence=0.9, score=0.9)]}, 0.1)
        self.assertEqual(bank.summary(0.1)["left"][0]["id"], identity)
        self.assertTrue(bank.summary(0.1)["left"][0]["confirmed"])
        bank.predict(command_motion(0, 0, 0.2, RoadPolicy()), 0.3)
        self.assertAlmostEqual(bank.summary(0.3)["left"][0]["age_s"], 0.2)
        self.assertEqual(bank.summary(0.3)["right"], [])


class GroundMotionTests(unittest.TestCase):
    def test_metric_flow_sees_motion_even_when_motor_command_is_zero(self):
        folder = ROOT / "tests/fixtures/road_model_sequence"
        old = cv2.imread(str(folder / "frame-000862.png"))
        new = cv2.imread(str(folder / "frame-000863.png"))
        motion = GroundMotion(RECORDED_CAL)
        motion.update(old, 0.0)
        result = motion.update(new, 0.27, 0.0, 0.0)
        self.assertTrue(result["valid"], result)
        self.assertGreater(result["travel_m"], 0.035)
        self.assertLess(result["travel_m"], 0.075)
        self.assertGreaterEqual(result["inliers"], 8)
        pix = CurvePixels(RECORDED_CAL)
        a, _ = pix.measure(old)
        b, _ = pix.measure(new)
        before = curve_agreement(a["path"], b["path"])["distance_m"]
        after = curve_agreement(apply_ground_motion(a["path"], result), b["path"])["distance_m"]
        self.assertLess(after, 0.015)
        self.assertLess(after, before * 0.5)

    def test_textureless_frame_uses_explicit_uncertain_prior(self):
        motion = GroundMotion(CAL)
        raw = np.full((240, 320, 3), 120, np.uint8)
        motion.update(raw, 0.0)
        result = motion.update(raw, 0.2, 0.05, 0)
        self.assertFalse(result["valid"])
        self.assertEqual(result["source"], "COMMAND PRIOR")
        self.assertGreater(result["sigma_m"], 0.0)

    def test_out_of_order_and_long_interval_do_not_supply_visual_velocity(self):
        raw = np.full((240, 320, 3), 120, np.uint8)
        motion = GroundMotion(CAL)
        motion.update(raw, 1.0)
        self.assertFalse(motion.update(raw, 1.0)["valid"])
        result = motion.update(raw, 3.0)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "FRAME INTERVAL")

    def test_measured_transform_is_rigid_without_scale(self):
        folder = ROOT / "tests/fixtures/road_model_sequence"
        motion = GroundMotion(RECORDED_CAL)
        motion.update(cv2.imread(str(folder / "frame-000862.png")), 0.0)
        result = motion.update(cv2.imread(str(folder / "frame-000863.png")), 0.27)
        np.testing.assert_allclose(result["rotation"].T @ result["rotation"], np.eye(2), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(result["rotation"]), 1.0)


class PolicyTests(unittest.TestCase):
    def test_bad_or_unknown_limits_fail_before_driving(self):
        for values in (
            {"typo": 1},
            {"maximum_prediction_s": -1},
            {"measurement_quality": float("nan")},
            {"lookahead_min_m": 0.5},
            {"motion_min_features": 4.5},
            {"motion_min_features": True},
        ):
            with self.assertRaises(ValueError):
                RoadPolicy.from_config({"road_model": values})


class CoupledRampTests(unittest.TestCase):
    def test_acceleration_preserves_turn_ratio_instead_of_straightening(self):
        from robot.control.controller import CommandRamp

        for desired in ((60, 0), (12, 60), (60, 12), (40, 40)):
            ramp = CommandRamp()
            for i in range(8):
                result = ramp.apply(desired, i * 0.1)
                self.assertLess(abs(result[0] * desired[1] - result[1] * desired[0]), sum(desired))
                self.assertLessEqual(max(result), max(desired))
            self.assertEqual(ramp.apply(None, 1.0), (0, 0))

    def test_stopped_wheel_is_never_raised_by_acceleration_shaping(self):
        from robot.control.controller import CommandRamp, compensate_pwm

        ramp = CommandRamp()
        for i in range(8):
            self.assertEqual(compensate_pwm(ramp.apply((60, 0), i * 0.1))[1], 0)
        self.assertEqual(ramp.apply((0, 0), 1.0), (0, 0))


class CandidateAssociationTests(unittest.TestCase):
    def candidate(self, side, curve, q=0.9, score=0.9):
        return dict(side=side, center=curve, path=curve["path"], confidence=q, score=score)

    def test_unique_prior_association_preserves_good_edge_beside_false_far_edge(self):
        pixels = CurvePixels(CAL)
        left = self.candidate("left", line())
        right = self.candidate("right", line(0.5, 0.7, 0.3))
        result, details = pixels.measure_fits(dict(left=left, right=right), {}, line()["path"])
        self.assertIsNotNone(result)
        self.assertEqual(result["mode"], "ONE LEFT")
        self.assertEqual(details["boundary_fallback"], "ASSOCIATED LEFT")
        # Without association evidence do not choose one arbitrarily.
        self.assertIsNone(pixels.measure_fits(dict(left=left, right=right), {})[0])

    def test_conflicting_best_pair_can_use_an_alternative_compatible_boundary(self):
        pixels = CurvePixels(CAL)
        left = self.candidate("left", line())
        false_right = self.candidate("right", line(y=0.20), score=1.0)
        true_right = self.candidate("right", line(0.2, 0.7), score=0.8)
        result, details = pixels.measure_fits(
            dict(left=left, right=false_right),
            dict(candidates=dict(left=[left], right=[false_right, true_right])),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["mode"], "TWO EDGES")
        self.assertIs(details["fits"]["right"], true_right)

    def test_association_does_not_arbitrarily_choose_between_two_conflicting_hypotheses(self):
        pixels = CurvePixels(CAL)
        left = self.candidate("left", line(y=0.08))
        right = self.candidate("right", line(y=-0.08))
        result, _ = pixels.measure_fits(dict(left=left, right=right), {}, line()["path"])
        self.assertIsNone(result)

    def test_continuous_lossless_recording_has_no_false_support_expiry(self):
        folder = ROOT / "tests/fixtures/road_model_sequence"
        t = TargetTracker(RECORDED_CAL)
        seen = {}
        telemetry = json.loads((folder / "telemetry.json").read_text())
        for frame in range(794, 879):
            trace = telemetry[str(frame)]
            result, _ = t.update(
                cv2.imread(str(folder / ("frame-%06d.png" % frame))),
                trace["captured"],
                trace["v"],
                trace["omega"],
            )
            self.assertNotEqual(result["invalid_reason"], "ROAD SUPPORT EXPIRED", (frame, result))
            if frame in (864, 865, 866, 867):
                self.assertTrue(result["observed"], (frame, result))
            seen[frame] = result["observed"]
        self.assertGreaterEqual(sum(seen.values()), 80)


class SymmetricFusionTests(unittest.TestCase):
    def test_equal_range_inputs_do_not_give_left_or_right_endpoint_bias(self):
        a, b = line(y=0.01), line(y=-0.01)
        result = merge_centers(a, b)
        reverse = merge_centers(b, a)
        self.assertLess(abs(result["path"][:, 1]).max(), 1e-8)
        np.testing.assert_allclose(result["path"], reverse["path"], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
