"""Current green can refute transported history; paint and unknown cannot refresh it.

The shifted-history test uses an unchanged real image with deliberately misplaced
old geometry. It isolates the policy boundary, not a claimed real driving event.
"""

from tests.support import ROOT
import copy
import json
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from robot.control.controller import target_command
from robot.core.road_policy import RoadPolicy
from robot.navigation.road_model import LocalRoadModel, BoundaryTrackBank
from robot.navigation.road_recovery import RoadSearch
from robot.navigation.tracker import TargetTracker
from robot.perception.boundaries import CurvePixels
from robot.perception.surface import RoadSurfaceEvidence

FIXTURE = ROOT / "tests/fixtures/crosswalk_20261030"
CALIBRATION = json.loads((FIXTURE / "calibration.json").read_text())


def path(start=0.15, end=0.70, y=0.0, count=50):
    return np.column_stack((np.linspace(start, end, count), np.full(count, y)))


def motion(translation=(0.0, 0.0), sigma=0.0):
    return dict(
        valid=True,
        rotation=np.eye(2),
        translation=np.array(translation),
        sigma_m=sigma,
        interval_s=0.1,
        travel_m=float(np.linalg.norm(translation)),
        speed_m_s=0.0,
        yaw_rad_s=0.0,
        source="GROUND FLOW",
        reason="",
        inliers=20,
        residual_m=0.0,
    )


def measured(points):
    return dict(path=points, confidence=0.9, mode="TWO EDGES")


def search_road(points):
    return dict(
        observed=True,
        valid=True,
        path=points,
        path_observed_mask=[True] * len(points),
        road_model=dict(sigma_m=0.005),
    )


def synthetic_surface(*, saturation=200, unknown=False):
    pixels = CurvePixels(CALIBRATION)
    green = (
        (pixels.xs[:, None] > 0.29) & (pixels.xs[:, None] < 0.51) & (abs(pixels.ys[None, :]) < 0.22)
    )
    if unknown:
        pixels.visible[:] = False
    return RoadSurfaceEvidence(pixels, green, np.full(green.shape, saturation, np.uint8))


class SurfacePredictionTests(unittest.TestCase):
    def model(self):
        model = LocalRoadModel()
        model.initialize(path(), 0.0, 0.9, "TWO EDGES")
        return model

    def tracker(self, surface=None, measurement=None):
        tracker = TargetTracker(CALIBRATION)
        tracker.model.initialize(path(), 0.0, 0.9, "TWO EDGES")
        tracker.motion.update = Mock(return_value=motion())
        details = {} if surface is None else dict(surface=surface)
        tracker.pixels.measure = Mock(return_value=(measurement, details))
        return tracker

    def search(self, origin=(0.0, 0.0)):
        return RoadSearch(RoadPolicy(), origin, [0.075, 0.105, 0.085], 0.3)

    def test_absent_surface_keeps_existing_prediction_behavior(self):
        tracker = self.tracker()
        result, _ = tracker.update(None, 0.1)
        self.assertTrue(result["valid"])
        self.assertEqual(result["state"], "PREDICTED")
        self.assertIsNone(result["surface_rejection"])
        self.assertIsNotNone(target_command(result))

    def test_white_gray_unknown_and_absent_evidence_never_refresh_history(self):
        for surface in (
            None,
            synthetic_surface(saturation=0),
            synthetic_surface(saturation=100),
            synthetic_surface(unknown=True),
        ):
            with self.subTest(surface=surface):
                model = self.model()
                model.confirm(measured(path(y=0.05)))
                before = copy.deepcopy(model.__dict__)
                report = model.reject_surface(surface)
                self.assertFalse(report and report["contradicted"])
                for key, value in before.items():
                    actual = model.__dict__[key]
                    if isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(actual, value)
                    else:
                        self.assertEqual(actual, value, key)
                self.assertEqual(model.support(0.1)[1]["oldest_seen"], 0.0)
                self.assertIsNone(model.support(0.61)[0])

    def test_middle_green_revokes_entire_old_curve_and_pending_not_just_gap(self):
        model = self.model()
        model.confirm(measured(path(y=0.05)))
        report = model.reject_surface(synthetic_surface())
        self.assertTrue(report["contradicted"])
        self.assertGreater(report["longest_green_m"], 0.075)
        # Both the safe near prefix and far island disappear with this hypothesis.
        for field in ("path", "seen", "quality", "variance", "travel", "pending", "last_seen"):
            self.assertIsNone(getattr(model, field), field)
        self.assertEqual(model.pending_count, 0)
        self.assertEqual(model.mode, "LOST")
        self.assertEqual(model.confidence, 0.0)
        self.assertIsNone(model.support(0.1)[0])
        json.dumps(report)

    def test_per_sample_uncertainty_does_not_hide_a_precise_contradicted_segment(self):
        model = self.model()
        model.variance[model.path[:, 0] > 0.6] = 0.2**2
        self.assertTrue(model.reject_surface(synthetic_surface())["contradicted"])

    def test_tracker_refutes_history_before_update_and_cannot_select_far_island(self):
        tracker = self.tracker(synthetic_surface())
        result, _ = tracker.update(None, 0.1)
        self.assertTrue(result["surface_rejection"]["contradicted"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["path"], [])
        self.assertIsNone(result["target_m"])
        self.assertIsNone(target_command(result))
        json.dumps(result)

    def test_same_frame_fresh_measurement_can_replace_refuted_history(self):
        fresh = path(0.15, 0.26, count=12)
        tracker = self.tracker(synthetic_surface(), measured(fresh))
        result, _ = tracker.update(None, 0.1)
        self.assertTrue(result["surface_rejection"]["contradicted"])
        self.assertTrue(result["observed"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["state"], "OBSERVED")
        self.assertEqual(result["pending_frames"], 0)
        np.testing.assert_array_equal(tracker.model.path, fresh)
        np.testing.assert_array_equal(tracker.model.seen, np.full(len(fresh), 0.1))

    def test_old_boundary_tracks_are_revoked_without_refreshing_others(self):
        bank = BoundaryTrackBank()

        def track(identity, points):
            return dict(id=identity, path=points, seen=0.0, sigma_m=0.005, quality=0.9, hits=3)

        bank.tracks["left"] = [track(1, path()), track(2, path(y=0.3))]
        self.assertEqual(bank.reject_surface(synthetic_surface()), 1)
        self.assertEqual([t["id"] for t in bank.tracks["left"]], [2])
        self.assertEqual(bank.tracks["left"][0]["seen"], 0.0)
        self.assertEqual(bank.tracks["left"][0]["hits"], 3)
        self.assertEqual(bank.reject_surface(None), 0)

    def test_search_refutes_transported_camera_coordinates_and_revokes_map_and_pose(self):
        search = self.search(origin=(0.06, -0.02))
        old = path()
        search.observe(search_road(old), motion(), 0.0)
        surface = Mock()
        surface.contradiction.return_value = dict(contradicted=True)
        shift = np.array([-0.02, 0.03])
        search.observe(dict(observed=False), motion(shift, 0.002), 0.1, surface=surface)
        self.assertEqual(surface.contradiction.call_count, 2)
        for call in surface.contradiction.call_args_list:
            np.testing.assert_allclose(call.args[0], old + shift, atol=1e-15)
            self.assertAlmostEqual(call.args[1], np.hypot(0.005, 0.002))
        self.assertEqual(search.patches, [])
        self.assertIsNone(search.pose_hint)
        self.assertFalse(search.swept_clear(0.0))
        self.assertIsNone(search.space_source)
        self.assertIsNone(search.command(dict(observed=False), None, motion(), 0.1, active=True))

    def test_search_preserves_unknown_history_without_refreshing_timestamps(self):
        search = self.search()
        search.observe(search_road(path()), motion(), 0.0)
        old = search.patches[0]["path"].copy()
        search.observe(dict(observed=False), motion(), 0.1, surface=synthetic_surface(saturation=0))
        np.testing.assert_array_equal(search.patches[0]["path"], old)
        self.assertEqual(search.patches[0]["seen"], 0.0)
        self.assertEqual(search.pose_hint["seen"], 0.0)

    def test_search_can_add_current_patch_after_revoking_old_map(self):
        search = self.search()
        search.observe(search_road(path()), motion(), 0.0)
        fresh = path(0.15, 0.26, count=12)
        search.observe(search_road(fresh), motion(), 0.1, surface=synthetic_surface())
        self.assertEqual(len(search.patches), 1)
        np.testing.assert_array_equal(search.patches[0]["path"], fresh)
        self.assertEqual(search.patches[0]["seen"], 0.1)
        np.testing.assert_array_equal(search.pose_hint["path"], fresh)

    def test_real_white_crossings_preserve_old_road_and_original_seen_time(self):
        specs = [
            (FIXTURE / "seven_bars.png", CALIBRATION),
            (
                ROOT / "tests/fixtures/lap_marker_20261031/start.png",
                json.loads(
                    (ROOT / "tests/fixtures/lap_marker_20261031/calibration.json").read_text()
                ),
            ),
            (
                ROOT / "tests/fixtures/curve_white_markings.png",
                json.loads((ROOT / "tests/fixtures/curve_memory_actual.json").read_text())[
                    "calibration"
                ],
            ),
        ]
        for filename, calibration in specs:
            for gain in (1.0, 0.55):
                with self.subTest(file=filename.name, gain=gain):
                    raw = cv2.imread(str(filename))
                    self.assertIsNotNone(raw)
                    raw = np.uint8(raw.astype(float) * gain)
                    pixels = CurvePixels(calibration)
                    prepared = pixels.prepare(raw)
                    surface = RoadSurfaceEvidence(pixels, prepared[4], prepared[1])
                    road, _ = pixels.measure(raw)
                    self.assertIsNotNone(road)
                    model = LocalRoadModel()
                    model.initialize(road["path"], 0.0, road["confidence"], road["mode"])
                    before = model.path.copy()
                    self.assertFalse(model.reject_surface(surface)["contradicted"])
                    np.testing.assert_array_equal(model.path, before)
                    np.testing.assert_array_equal(model.seen, np.zeros(len(before)))

    def test_real_image_with_constructed_shifted_history_loses_prediction(self):
        raw = cv2.imread(str(FIXTURE / "seven_bars.png"))
        self.assertIsNotNone(raw)
        pixels = CurvePixels(CALIBRATION)
        road, details = pixels.measure(raw)
        prepared = pixels.prepare(raw)
        surface = RoadSurfaceEvidence(pixels, prepared[4], prepared[1])
        # Keep camera pixels unchanged; deliberately misplace only old geometry.
        old = road["path"] + [0.0, 0.25]
        self.assertTrue(surface.contradiction(old, 0.005 / 0.9)["contradicted"])
        results = []
        for evidence in (None, surface):
            tracker = TargetTracker(CALIBRATION)
            tracker.model.initialize(old, 0.0, 0.9, "TWO EDGES")
            tracker.motion.update = Mock(return_value=motion())
            current = {k: v for k, v in details.items() if k not in ("surface", "candidates")}
            if evidence is not None:
                current["surface"] = evidence
            tracker.pixels.measure = Mock(return_value=(None, current))
            result, _ = tracker.update(raw, 0.1)
            results.append(result)
        self.assertTrue(results[0]["valid"])
        self.assertEqual(results[0]["state"], "PREDICTED")
        self.assertIsNotNone(target_command(results[0]))
        self.assertTrue(results[1]["surface_rejection"]["contradicted"])
        self.assertFalse(results[1]["valid"])
        self.assertIsNone(target_command(results[1]))


if __name__ == "__main__":
    unittest.main()
