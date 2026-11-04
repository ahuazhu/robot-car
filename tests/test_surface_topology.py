"""Fixed green exterior, white paint and gray road retain distinct evidence."""

import json
import unittest

import cv2
import numpy as np

from tests.support import ROOT
from robot.core.geometry import arc
from robot.perception.boundaries import CurvePixels
from robot.tasks.crosswalk import CrosswalkDetector, CrosswalkPolicy


FIXTURES = ROOT / "tests/fixtures"


def ground_coordinates(calibration):
    """Render known ground-space shapes independently of the detector masks."""
    yy, xx = np.indices((240, 320))
    points = np.stack((xx / 320.0, yy / 240.0, np.ones_like(xx)), axis=-1)
    ground = points @ np.asarray(calibration["H_normalized"]).T
    return ground[:, :, 0] / ground[:, :, 2], ground[:, :, 1] / ground[:, :, 2]


def green_reflection_scene(calibration, lateral=0.08):
    """Only green ground and a 5cm pale reflection; there is no road."""
    x, y = ground_coordinates(calibration)
    hsv = np.empty((240, 320, 3), np.uint8)
    hsv[:] = (60, 60, 150)
    reflection = (x > 0.08) & (x < 1.0) & (abs(y - lateral) < 0.025)
    hsv[reflection] = (60, 30, 185)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def road_with_transverse_line(calibration):
    """A straight 30cm gray road with 2cm white borders and one joined line."""
    x, y = ground_coordinates(calibration)
    visible_ground = (x > 0.08) & (x < 1.0)
    image = np.full((240, 320, 3), 120, np.uint8)
    image[(abs(y) > 0.15) & visible_ground] = (35, 180, 35)
    image[(abs(y) >= 0.15) & (abs(y) < 0.17) & visible_ground] = 240
    image[(abs(y) <= 0.17) & (x >= 0.35) & (x <= 0.37)] = 240
    return image


class SurfaceTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.synthetic_calibration = json.loads(
            (FIXTURES / "synthetic_road_calibration.json").read_text()
        )

    def test_pale_reflection_on_green_ground_is_not_a_road(self):
        for lateral in (-0.08, 0.08):
            with self.subTest(lateral=lateral):
                pixels = CurvePixels(self.synthetic_calibration)
                measured, _ = pixels.measure(
                    green_reflection_scene(self.synthetic_calibration, lateral)
                )
                self.assertIsNone(measured)

    def test_white_transverse_line_does_not_turn_or_remove_the_real_gray_road(self):
        pixels = CurvePixels(self.synthetic_calibration)
        measured, _ = pixels.measure(road_with_transverse_line(self.synthetic_calibration))
        self.assertIsNotNone(measured)
        self.assertGreater(measured["confidence"], 0.65)
        self.assertGreater(arc(measured["path"])[-1], 0.30)
        self.assertLess(abs(measured["path"][:, 1]).max(), 0.015)

    def test_recorded_weak_fragment_retains_both_boundaries_and_near_support(self):
        folder = FIXTURES / "curve_recordings_20261023"
        pixels = CurvePixels(json.loads((folder / "calibration.json").read_text()))
        image = cv2.imread(str(folder / "weak-fragment.jpg"))[:240, :320]
        measured, _ = pixels.measure(image)
        self.assertIsNotNone(measured)
        self.assertEqual(measured["mode"], "TWO EDGES")
        self.assertGreater(measured["confidence"], 0.94)
        self.assertGreater(arc(measured["path"])[-1], 0.19)
        self.assertLess(np.linalg.norm(measured["path"], axis=1).min(), 0.29)

    def test_recorded_white_markings_retain_the_long_observed_boundary(self):
        calibration = json.loads((FIXTURES / "curve_memory_actual.json").read_text())["calibration"]
        pixels = CurvePixels(calibration)
        image = cv2.imread(str(FIXTURES / "curve_white_markings.png"))
        measured, details = pixels.measure(image)
        self.assertIsNotNone(measured)
        self.assertGreater(details["fits"]["left"]["length"], 0.60)
        self.assertLess(np.linalg.norm(measured["path"], axis=1).min(), 0.20)

    def test_recorded_crosswalk_keeps_white_bars_and_original_luminance(self):
        calibration = json.loads((FIXTURES / "crosswalk_20261024_calibration.json").read_text())
        pixels = CurvePixels(calibration)
        image = cv2.imread(str(FIXTURES / "curve_zebra_turn.jpg"))[:240, :320]
        measured, details = pixels.measure(image)
        self.assertIsNotNone(measured)
        crossing = CrosswalkDetector(pixels, CrosswalkPolicy()).detect(details, measured, 1.0)
        expected = json.loads((FIXTURES / "crosswalk_zebra_turn_expected.json").read_text())
        self.assertIsNotNone(crossing)
        self.assertEqual(crossing["bar_count"], expected["bar_count"])
        for name in ("near_m", "far_m", "normal", "corners"):
            np.testing.assert_allclose(crossing[name], expected[name], atol=1e-10, rtol=1e-10)
        raw_value = cv2.cvtColor(details["bird"], cv2.COLOR_BGR2HSV)[:, :, 2]
        np.testing.assert_array_equal(details["value"], raw_value)


if __name__ == "__main__":
    unittest.main()
