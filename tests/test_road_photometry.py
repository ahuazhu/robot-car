"""Photometric fusion preserves exact pixel masks and measured road geometry."""

from tests.support import ROOT
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from robot.core.geometry import geometry_frame
from robot.core.native_geometry import native_fit_status, native_road_regions
from robot.perception.boundaries import CurvePixels
from robot.perception.photometry import road_regions_reference


@unittest.skipUnless(native_fit_status().get("road_regions_enabled"), "compile the photometric kernel")
class RoadPhotometryTests(unittest.TestCase):
    def test_threshold_edges_and_random_pixels_are_bit_exact(self):
        rng = np.random.default_rng(20261109)
        for kind in range(20):
            h = rng.integers(0, 180, (93, 117), dtype=np.uint8)
            s = rng.integers(0, 256, h.shape, dtype=np.uint8)
            v = rng.integers(0, 256, h.shape, dtype=np.uint8)
            # Include constant windows and every relevant brightness/color seam.
            if kind < 8:
                v[:] = (0, 25, 26, 34, 35, 209, 210, 255)[kind]
            h[:, :4] = (31, 32, 95, 96)
            s[:, :6] = (55, 56, 104, 105, 109, 110)
            seen = rng.choice(np.array([0, 1, 2, 128, 255], np.uint8), h.shape)
            vf = v.astype(np.float32)
            mean = cv2.blur(vf, (15, 15))
            second = cv2.blur(vf * vf, (15, 15))
            actual = native_road_regions(h, s, v, seen, mean, second)
            expected = road_regions_reference(h, s, v, seen, mean, second)
            self.assertIsNotNone(actual)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_mismatched_shapes_are_rejected_before_native_access(self):
        a = np.zeros((9, 11), np.uint8)
        f = a.astype(np.float32)
        self.assertIsNone(native_road_regions(a, a, a, a[:-1], f, f))
        self.assertIsNone(native_road_regions(a, a, a, a, f[:-1], f))
        self.assertIsNone(native_road_regions(a[:0], a[:0], a[:0], a[:0], f[:0], f[:0]))

    def test_historical_sequence_has_identical_regions_and_measured_roads(self):
        cal = json.loads((ROOT / "tests/fixtures/road_model_recording_calibration.json").read_text())
        pixels = CurvePixels(cal)
        files = sorted((ROOT / "tests/fixtures/road_model_sequence").glob("*.png"))
        self.assertEqual(len(files), 85)
        for name in files:
            with self.subTest(frame=name.name):
                image = cv2.imread(str(name))[:240, :320]
                actual_regions = pixels.prepare(image)
                with geometry_frame():
                    actual, ad = pixels.measure(image)
                with patch("robot.perception.photometry.native_road_regions", return_value=None):
                    expected_regions = pixels.prepare(image)
                    with geometry_frame():
                        expected, ed = pixels.measure(image)
                for a, b in zip(actual_regions, expected_regions):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(ad["counts"], ed["counts"])
                self.assertEqual(ad["photometric_passes"], ed["photometric_passes"])
                self.assertEqual(actual is None, expected is None)
                if actual is not None:
                    self.assertEqual(actual["mode"], expected["mode"])
                    self.assertEqual(actual["confidence"], expected["confidence"])
                    np.testing.assert_array_equal(actual["path"], expected["path"])
                    for side in ("left", "right"):
                        for a, b in zip(actual["boundaries"][side], expected["boundaries"][side]):
                            np.testing.assert_array_equal(a["path"], b["path"])
                            self.assertEqual(a["confidence"], b["confidence"])

    def test_returned_planes_remain_owned_by_their_frame(self):
        cal = json.loads((ROOT / "tests/fixtures/camera_pose_20261031_191327.json").read_text())
        pixels = CurvePixels(cal)
        image = cv2.imread(str(ROOT / "tests/fixtures/curve_shadow_glare.jpg"))[:240, :320]
        first = pixels.prepare(image)
        saved = [a.copy() for a in first]
        pixels.prepare(np.zeros_like(image))
        for a, b in zip(first, saved):
            np.testing.assert_array_equal(a, b)
