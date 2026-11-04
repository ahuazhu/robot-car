"""Coordinate caching and common masks must preserve every detection/status."""

from tests.support import ROOT
import json, unittest
import cv2, numpy as np

from robot.tasks.crosswalk import CrosswalkDetector, CrosswalkPolicy
from tests.reference.crosswalk_prelookup_reference import ScalarCrosswalkDetector
from robot.perception.boundaries import CurvePixels
from robot.core.geometry import geometry_frame


class CrosswalkLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = []
        fixtures = ROOT / "tests/fixtures"
        specs = [
            (
                fixtures / "crosswalk_20261030" / name,
                fixtures / "crosswalk_20261030/calibration.json",
            )
            for name in ("seven_bars.png", "bend_without_crossing.png")
        ]
        specs += [
            (p, fixtures / "crosswalk_20261024_calibration.json")
            for p in sorted((fixtures / "crosswalk_recordings_20261024").glob("frame-*.png"))
        ]
        specs.append(
            (fixtures / "curve_zebra_turn.jpg", fixtures / "crosswalk_20261024_calibration.json")
        )
        for file, calibration in specs:
            pixels = CurvePixels(json.loads(calibration.read_text()))
            image = cv2.imread(str(file))[:240, :320]
            with geometry_frame():
                road, details = pixels.measure(image)
            cls.cases.append((str(file.relative_to(ROOT)), pixels, image, road or {}, details))

    def compare(self, pixels, details, road):
        old = ScalarCrosswalkDetector(pixels, CrosswalkPolicy())
        new = CrosswalkDetector(pixels, CrosswalkPolicy())
        for now in (1.0, 1.2):
            self.assertEqual(new.detect(details, road, now), old.detect(details, road, now))
            self.assertEqual(new.status, old.status)

    def test_all_real_frames_preserve_complete_observation_and_each_pass(self):
        for name, pixels, image, road, details in self.cases:
            with self.subTest(name=name):
                self.compare(pixels, details, road)
                if name.endswith("curve_zebra_turn.jpg"):
                    actual = CrosswalkDetector(pixels, CrosswalkPolicy()).detect(details, road, 1.0)
                    expected = json.loads(
                        (ROOT / "tests/fixtures/crosswalk_zebra_turn_expected.json").read_text()
                    )
                    self.assertEqual(actual["bar_count"], expected["bar_count"])
                    for key in (
                        "near_m",
                        "far_m",
                        "confidence",
                        "normal",
                        "corners",
                        "bar_length_m",
                        "bar_width_m",
                    ):
                        np.testing.assert_allclose(
                            actual[key], expected[key], atol=1e-10, rtol=1e-10
                        )

    def test_exposure_changes_preserve_raw_and_contrast_fallback(self):
        _, pixels, image, _, _ = self.cases[0]
        for gain in (0.55, 0.75, 1.15):
            altered = np.clip(image.astype(float) * gain, 0, 255).astype(np.uint8)
            with geometry_frame():
                road, details = pixels.measure(altered)
            with self.subTest(gain=gain):
                self.compare(pixels, details, road or {})

    def test_missing_precomputed_hsv_preserves_fallback(self):
        _, pixels, image, road, details = self.cases[0]
        self.compare(
            pixels, {k: v for k, v in details.items() if k not in ("saturation", "value")}, road
        )

    def test_cached_coordinates_are_bit_identical_float32(self):
        for _, pixels, _, _, _ in self.cases:
            d = CrosswalkDetector(pixels, CrosswalkPolicy())
            rows, cols = np.indices(pixels.visible.shape)
            expected = np.c_[pixels.xs[rows.ravel()], pixels.ys[cols.ravel()]].astype(np.float32)
            self.assertTrue(np.array_equal(d._metric_points.reshape(-1, 2), expected))


if __name__ == "__main__":
    unittest.main()
