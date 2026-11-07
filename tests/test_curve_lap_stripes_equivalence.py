"""Frozen scalar equivalence for batched transverse-line geometry."""

from tests.support import ROOT
import json
import unittest
from types import MethodType
from unittest.mock import patch
import cv2
import numpy as np


from robot.tasks.lap_counter import LapMarkerDetector
from tests.reference.lap_stripes_scalar_reference import stripes as scalar_stripes
from robot.perception.boundaries import CurvePixels
from robot.core.geometry import geometry_frame


class StripeEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = []
        fixtures = ROOT / "tests/fixtures"
        for folder, names in (
            ("lap_marker_20261031", ["start.png"]),
            (
                "lap_line_20261031",
                ["line-0.png", "line-1.png", "line-2.png", "line-3.png", "bend.png"],
            ),
            ("crosswalk_20261030", ["seven_bars.png", "bend_without_crossing.png"]),
        ):
            calibration = json.loads((fixtures / folder / "calibration.json").read_text())
            for name in names:
                pixels = CurvePixels(calibration)
                im = cv2.imread(str(fixtures / folder / name))
                with geometry_frame():
                    road, details = pixels.measure(im)
                if road is None:
                    raise AssertionError("Fixture has no road: " + folder + "/" + name)
                cls.cases.append((folder + "/" + name, pixels, im, road, details))

    def compare(self, pixels, details, road):
        scalar = LapMarkerDetector(pixels, {"enabled": True})
        batch = LapMarkerDetector(pixels, {"enabled": True})
        calls = []
        original = pixels.sample

        def sample(mask, points):
            calls.append(np.array(points, copy=True))
            return original(mask, points)

        pixels.sample = sample
        try:
            expected = scalar_stripes(scalar, details, road)
            before = calls[:]
            calls.clear()
            actual = batch.stripes(details, road)
            after = calls[:]
        finally:
            pixels.sample = original
        self.assertLessEqual(len(after), 4, "sampling must be batched per threshold pass")
        self.assertEqual(bool(after), bool(before))
        if before:
            # Every old value/saturation probe remains present, in the same
            # candidate order; batching changes calls, not sampled evidence.
            for channel in (0, 1):
                np.testing.assert_allclose(
                    np.vstack(after[channel::2]), np.vstack(before[channel::2]),
                    rtol=0, atol=1e-14,
                )
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                if key in ("center", "normal", "distance_m", "sigma_m", "thickness_m"):
                    np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-14)
                else:
                    self.assertEqual(a[key], b[key], key)
        return batch

    def test_real_line_bend_and_crosswalk_candidates_and_evaluation_order_match(self):
        for name, pixels, im, road, details in self.cases:
            with self.subTest(name=name):
                self.compare(pixels, details, road)

    def test_real_exposure_variants_match_scalar(self):
        name, pixels, im, road, details = self.cases[1]
        for gain in (0.55, 0.75, 1.15):
            altered = np.clip(im.astype(float) * gain, 0, 255).astype(np.uint8)
            with geometry_frame():
                road, details = pixels.measure(altered)
            self.assertIsNotNone(road)
            with self.subTest(gain=gain):
                self.compare(pixels, details, road)

    def test_three_frame_confirmation_results_match(self):
        for name, pixels, im, road, details in self.cases:
            scalar = LapMarkerDetector(pixels, {"enabled": True})
            batch = LapMarkerDetector(pixels, {"enabled": True})
            scalar.stripes = MethodType(scalar_stripes, scalar)
            for now in (0.0, 0.2, 0.4):
                expected = scalar.detect(im, details, road, now)
                actual = batch.detect(im, details, road, now)
                for key in ("marker_visible", "confirmed", "confirmation_frames", "reason"):
                    self.assertEqual(actual[key], expected[key], (name, now, key))

    def test_cache_reuses_profile_arrays_and_refreshes_for_geometry_change(self):
        _, pixels, _, road, details = self.cases[1]
        batch = self.compare(pixels, details, road)
        first = batch._profile_cache
        batch.stripes(details, road)
        self.assertIs(batch._profile_cache, first)
        width = pixels.width
        try:
            pixels.width = width * 0.9
            expected = scalar_stripes(batch, details, road)
            actual = batch.stripes(details, road)
            self.assertEqual(actual, expected)
            self.assertIsNot(batch._profile_cache, first)
        finally:
            pixels.width = width

    def test_scalar_acceptance_boundary_keeps_identical_candidate_decisions(self):
        _, pixels, _, _, details = self.cases[1]
        row = min(40, len(pixels.xs) - 1)
        c1 = 5
        c2 = min(25, len(pixels.ys) - 1)
        lines = np.array([[[c1, row, c2, row]], [[c2, row, c1, row]]], np.int32)
        mid = np.array([pixels.xs[row], (pixels.ys[c1] + pixels.ys[c2]) / 2])
        for alignment in (0.35 - 2e-15, 0.35, 0.35 + 2e-15):
            forward = np.array([np.sqrt(1 - alignment * alignment), alignment])
            road = dict(path=mid + np.linspace(-0.1, 0.1, 17)[:, None] * forward)
            with self.subTest(alignment=alignment), patch("cv2.HoughLinesP", return_value=lines):
                self.compare(pixels, details, road)

    def test_stable_ties_duplicates_and_dense_hough_candidates_keep_order(self):
        _, pixels, _, road, details = self.cases[1]
        rows, cols = details["value"].shape
        rng = np.random.default_rng(8141)
        segments = np.column_stack(
            (
                rng.integers(0, cols, 160),
                rng.integers(0, rows, 160),
                rng.integers(0, cols, 160),
                rng.integers(0, rows, 160),
            )
        ).astype(np.int32)
        segments = segments[np.any(segments[:, :2] != segments[:, 2:], axis=1)]
        segments = np.concatenate((segments, segments[:40], segments[:40, [2, 3, 0, 1]]))[
            :, None, :
        ]
        with patch("cv2.HoughLinesP", return_value=segments):
            self.compare(pixels, details, road)


if __name__ == "__main__":
    unittest.main()
