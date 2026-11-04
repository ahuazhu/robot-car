"""Compiled kernels preserve reference geometry, rejection and bounded fallback."""

from tests.support import ROOT
import unittest
from unittest.mock import patch
import numpy as np
from robot.core import geometry
from robot.core.native_geometry import native_fit_status
import json
import cv2
from robot.perception.boundaries import CurvePixels
from robot.core.native_geometry import native_boundary_chains
from robot.core.native_geometry import native_local_fit
from robot.core.native_geometry import native_support_projection


@unittest.skipUnless(
    native_fit_status().get("agreement_enabled"),
    "compile native kernel to run agreement equivalence",
)
class NativeAgreementTests(unittest.TestCase):
    def test_recorded_shaped_and_random_curves_keep_same_decisions(self):
        rng = np.random.default_rng(20261024)
        for i in range(300):
            n = int(rng.integers(12, 130))
            x = np.linspace(0.1, 0.8, n)
            a = np.c_[x, 0.03 * np.sin(x * rng.uniform(1, 8))]
            b = a.copy() + [rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12)]
            if i % 5 == 0:
                b = b[::-1]
            if i % 7 == 0:
                b = b[n // 2 :]
            if i % 11 == 0:
                b = b + [1.0, 0.0]
            with (
                patch.object(geometry, "native_curve_agree_paths", return_value=(False, None)),
                patch.object(geometry, "native_curve_agreement", return_value=(False, None)),
            ):
                reference = geometry.curve_agreement(a, b)
            result = geometry.curve_agreement(a, b)
            self.assertEqual(reference is None, result is None, i)
            if result:
                for key in result:
                    self.assertAlmostEqual(reference[key], result[key], places=12, msg=(i, key))

    def test_oversized_support_falls_back_without_changing_result(self):
        a = np.c_[np.linspace(0.1, 0.8, 220), np.zeros(220)]
        with (
            patch.object(geometry, "native_curve_agree_paths", return_value=(False, None)),
            patch.object(geometry, "native_curve_agreement", return_value=(False, None)),
        ):
            reference = geometry.curve_agreement(a, a)
        self.assertEqual(reference, geometry.curve_agreement(a, a))

    def test_frame_cache_does_not_leak_across_observations(self):
        a = np.c_[np.linspace(0.1, 0.8, 40), np.zeros(40)]
        with geometry.geometry_frame():
            first = geometry.curve_agreement(a, a)
            shifted = geometry.curve_agreement(a, a + [0, 0.2])
            self.assertGreater(shifted["distance_m"], first["distance_m"] + 0.1)
        with geometry.geometry_frame():
            self.assertAlmostEqual(geometry.curve_agreement(a, a + [0, 0.1])["distance_m"], 0.1)


@unittest.skipUnless(native_fit_status().get("boundary_enabled"), "compile the contour kernel")
class BoundaryKernelTests(unittest.TestCase):
    def test_real_contours_all_thresholds_seams_and_reversal_match_reference(self):
        cal = json.loads((ROOT / "maixcam2/curve_follow/calibration.json").read_text())
        pixel = CurvePixels(cal)
        accepted = 0
        names = (
            "curve_shadow_glare.jpg",
            "curve_zebra_turn.jpg",
            "right_bend.jpg",
            "curve_white_markings.png",
            "curve_unfounded_prediction.png",
        )
        for name in names:
            img = cv2.imread(str(ROOT / "tests/fixtures" / name))[:240, :320]
            prepared = pixel.prepare(img)
            _, sat, raw, rawgray, green, contrast, ngray, _ = prepared
            for normalized in (False, True):
                gray = ngray if normalized else rawgray
                value = contrast if normalized else raw
                for cutoff in pixel.white_value_levels:
                    white = (sat <= 105) & (value >= cutoff) & pixel.visible
                    codes = (
                        gray.astype(np.uint8)
                        + 2 * green.astype(np.uint8)
                        + 4 * white.astype(np.uint8)
                    )
                    contours, _ = cv2.findContours(
                        white.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
                    )
                    minimum = max(5, int(np.ceil(pixel.policy.minimum_segment_m / pixel.step)))
                    for contour in contours:
                        points = contour[:, 0, :]
                        if len(points) < 2 * minimum:
                            continue
                        for points in (points, np.roll(points, 7, axis=0), points[::-1]):
                            actual = native_boundary_chains(
                                points,
                                codes,
                                gray,
                                pixel.visible,
                                pixel.xs,
                                pixel.ys,
                                pixel.step,
                                pixel.exterior_distances,
                                minimum,
                                pixel.policy.minimum_segment_m,
                            )
                            expected = pixel.contour_chains_reference(points, codes, gray, minimum)
                            self.assertIsNotNone(actual)
                            self.assertEqual(len(actual), len(expected))
                            accepted += len(expected)
                            for a, b in zip(actual, expected):
                                for x, y in zip(a, b):
                                    np.testing.assert_allclose(x, y, atol=1e-12, rtol=1e-10)
        self.assertGreater(accepted, 30)

    def test_entire_extraction_keeps_candidate_geometry_and_scores(self):
        cal = json.loads((ROOT / "maixcam2/curve_follow/calibration.json").read_text())
        pixel = CurvePixels(cal)
        image = cv2.imread(str(ROOT / "tests/fixtures/curve_zebra_turn.jpg"))[:240, :320]
        for normalized in (False, True):
            actual, da = pixel.extract(image, normalized=normalized)
            with patch("robot.perception.boundaries.native_boundary_chains", return_value=None):
                expected, de = pixel.extract(image, normalized=normalized)
            self.assertEqual(da["counts"], de["counts"])
            for side in ("left", "right"):
                for a, b in zip(da["candidates"][side], de["candidates"][side]):
                    np.testing.assert_allclose(a["path"], b["path"], atol=1e-12)
                    np.testing.assert_allclose(a["center"]["path"], b["center"]["path"], atol=1e-12)
                    self.assertAlmostEqual(a["score"], b["score"])


@unittest.skipUnless(
    native_fit_status().get("chain_enabled"), "compile chain kernel for equivalence checks"
)
class NativeChainTests(unittest.TestCase):
    def compare(self, points):
        with patch.object(geometry, "native_chain_fit", return_value=(False, None)):
            reference = geometry.fit_curve(points)
        actual = geometry.fit_curve(points)
        self.assertEqual(reference is None, actual is None)
        if actual:
            for key in ("path", "tangent", "curvature"):
                np.testing.assert_allclose(
                    actual[key], reference[key], atol=1e-8, rtol=1e-8, err_msg=key
                )
            self.assertAlmostEqual(actual["length"], reference["length"], places=10)
            self.assertAlmostEqual(actual["residual"], reference["residual"], places=10)

    def test_random_noisy_reversed_and_duplicate_chains(self):
        rng = np.random.default_rng(20261024)
        for i in range(500):
            n = int(rng.integers(8, 500))
            x = np.linspace(0.05, rng.uniform(0.12, 1), n)
            points = np.c_[x, rng.uniform(0.01, 0.1) * np.sin(x * rng.uniform(1, 12))]
            points += rng.normal(0, rng.uniform(0, 0.001), points.shape)
            if i % 5 == 0:
                points = points[::-1]
            if i % 7 == 0:
                points = np.repeat(points, 2, axis=0)
            self.compare(points)

    def test_invalid_chains_and_bounded_fallback(self):
        for points in (
            np.zeros((0, 2)),
            np.zeros((20, 2)),
            np.c_[np.linspace(0, 0.04, 10), np.zeros(10)],
            np.full((5, 2), np.nan),
            np.c_[np.linspace(0, 3, 100), np.zeros(100)],
            np.c_[np.linspace(0.1, 0.8, 1700), np.zeros(1700)],
        ):
            self.compare(points)


@unittest.skipUnless(
    native_fit_status()["enabled"], "compile native kernel to run equivalence checks"
)
class NativeFitTests(unittest.TestCase):
    def test_quadratic_core_preserves_curved_noisy_and_invalid_fits(self):
        rng = np.random.default_rng(232)
        cases = [np.zeros((10, 2)), np.array([[0.0, 0.0], [float("nan"), 0.0], [1.0, 0.0]])]
        for i in range(80):
            s = np.linspace(0, rng.uniform(0.09, 1.5), rng.integers(15, 190))
            points = np.column_stack((s, 0.12 * np.sin(rng.uniform(-5, 5) * s)))
            cases.append(points + rng.normal(0, 0.0008, points.shape))
        for points in cases:
            with patch.object(geometry, "native_local_fit", return_value=None):
                reference = geometry.fit_curve(points)
            actual = geometry.fit_curve(points)
            self.assertEqual(reference is None, actual is None)
            if actual is not None:
                for key in ("path", "tangent", "curvature"):
                    np.testing.assert_allclose(actual[key], reference[key], atol=1e-7, rtol=1e-8)
                self.assertAlmostEqual(actual["residual"], reference["residual"], places=10)

    def test_sparse_input_returns_to_reference_support_selection(self):
        points = np.column_stack((np.arange(10.0), np.zeros(10)))
        self.assertIsNone(native_local_fit(points, np.arange(10.0)))

    def test_wrong_shapes_never_reach_the_c_buffer(self):
        with self.assertRaises(ValueError):
            native_local_fit(np.zeros((10, 3)), np.arange(10.0))


@unittest.skipUnless(native_fit_status().get("projection_enabled"), "compile the projection kernel")
class ProjectionTests(unittest.TestCase):
    def test_random_curves_queries_duplicate_points_and_endpoints_match_reference(self):
        rng = np.random.default_rng(928)
        for i in range(160):
            r = np.cumsum(rng.normal(0, 0.02, (int(rng.integers(3, 160)), 2)), axis=0)
            if i % 3 == 0:
                r[1] = r[0]
            q = np.vstack((rng.normal(0, 0.3, (25, 2)), r[[0, -1]]))
            with patch.object(geometry, "native_support_projection", return_value=None):
                expected = geometry.support_projection(q, r)
            actual = native_support_projection(q, r)
            self.assertIsNotNone(actual)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, atol=2e-12, rtol=1e-10)

    def test_bad_shapes_and_nonfinite_input_fall_back_without_c_buffer_reads(self):
        for q, r in (
            (np.zeros((2, 3)), np.zeros((5, 2))),
            (np.zeros((2, 2)), np.zeros((1, 2))),
            (np.array([[np.nan, 0.0]]), np.zeros((5, 2))),
        ):
            self.assertIsNone(native_support_projection(q, r))


@unittest.skipUnless(native_fit_status().get("pursuit_enabled"), "compile pursuit kernel")
class PursuitKernelTests(unittest.TestCase):
    def test_random_walks_degenerate_segments_and_both_turn_directions(self):
        rng = np.random.default_rng(2458)
        for i in range(300):
            p = np.cumsum(rng.normal(0, 0.03, (int(rng.integers(2, 100)), 2)), axis=0) + [0.15, 0.0]
            if i % 3 == 0:
                p[1] = p[0]
            for radius in (0.12, 0.25, 0.45):
                actual = geometry.pursuit_point(p, radius)
                with patch.object(geometry, "native_pursuit_selection", return_value=(False, None)):
                    expected = geometry.pursuit_point(p, radius)
                    anchor = geometry.nearest_forward_point(p)
                actual_anchor = geometry.nearest_forward_point(p)
                self.assertEqual(actual is None, expected is None)
                self.assertEqual(actual_anchor is None, anchor is None)
                if anchor is not None:
                    for a, b in zip(actual_anchor, anchor):
                        np.testing.assert_allclose(a, b, atol=1e-12)
                if actual is None:
                    continue
                self.assertEqual(actual["mode"], expected["mode"])
                for k in ("point", "position_m", "distance_m", "speed_scale"):
                    np.testing.assert_allclose(actual[k], expected[k], atol=1e-11, rtol=1e-9)

    def test_nonfinite_and_malformed_path_is_rejected(self):
        for p in ([], [[0.1, 0]], [[np.nan, 0], [0.2, 0]], [[0.1, 0, 0], [0.2, 0, 0]]):
            self.assertIsNone(geometry.pursuit_point(p, 0.2))


if __name__ == "__main__":
    unittest.main()
