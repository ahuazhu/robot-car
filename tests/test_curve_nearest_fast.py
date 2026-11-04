"""Native nearest samples preserve overlap, fusion and distance decisions."""

from tests.support import ROOT
import unittest
from unittest.mock import patch
import numpy as np

from robot.core import geometry
from robot.core.native_geometry import native_fit_status, native_nearest_samples


@unittest.skipUnless(
    native_fit_status().get("nearest_samples_enabled"), "compile the nearest-sample kernel"
)
class NearestSampleTests(unittest.TestCase):
    def compare(self, q, r):
        delta = q[:, None] - r[None, :]
        distances = np.hypot(delta[..., 0], delta[..., 1])
        indices = distances.argmin(axis=1)
        actual = native_nearest_samples(q, r)
        self.assertIsNotNone(actual)
        np.testing.assert_array_equal(actual[0], indices)
        np.testing.assert_allclose(
            actual[1], distances[np.arange(len(q)), indices], rtol=2e-15, atol=1e-15
        )

    def test_random_reversed_strided_and_duplicate_paths(self):
        rng = np.random.default_rng(20261101)
        for _ in range(100):
            q = rng.normal(size=(int(rng.integers(1, 160)), 2))
            r = rng.normal(size=(int(rng.integers(3, 160)), 2))
            r[1] = r[0]
            self.compare(q, r)
            self.compare(q[::-2], r[::-1])

    def test_ties_keep_first_index_and_endpoint_samples(self):
        q = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        r = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
        self.compare(q, r)
        np.testing.assert_array_equal(native_nearest_samples(q, r)[0], [0, 0, 1])

    def test_unsupported_dtype_shape_and_nonfinite_use_reference(self):
        for q, r in (
            (np.zeros((2, 3)), np.zeros((2, 2))),
            (np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32)),
            (np.array([[np.nan, 0.0]]), np.zeros((2, 2))),
            (np.zeros((2, 2)), np.array([[np.inf, 0.0]])),
            (np.zeros((0, 2)), np.zeros((2, 2))),
        ):
            self.assertIsNone(native_nearest_samples(q, r))
        a = np.array([[0, 0], [1, 0], [2, 0]], np.float32)
        with patch.object(geometry, "native_nearest_samples", return_value=None):
            expected = geometry._nearest_samples(a, a)
        actual = geometry._nearest_samples(a, a)
        for x, y in zip(actual, expected):
            np.testing.assert_array_equal(x, y)

    def test_distance_quantile_and_endpoint_exclusion_match_original(self):
        rng = np.random.default_rng(173)
        for _ in range(100):
            a = np.c_[np.linspace(0.1, 0.7, 50), rng.normal(0, 0.01, 50)]
            b = np.c_[
                np.linspace(rng.uniform(-0.1, 0.4), rng.uniform(0.5, 0.9), 60),
                rng.normal(0, 0.04, 60),
            ]
            for left, right in ((a, b), (a[::-1], b), (a, b[:4])):
                with patch.object(geometry, "native_nearest_samples", return_value=None):
                    expected = geometry.curve_distance(left, right)
                actual = geometry.curve_distance(left, right)
                self.assertEqual(actual is None, expected is None)
                if actual is not None:
                    self.assertAlmostEqual(actual, expected, places=14)

    def test_merge_curves_crossings_gaps_reversed_paths_and_weights_match(self):
        rng = np.random.default_rng(81)
        for _ in range(70):
            x = np.linspace(0.08, 0.7, 55)
            a = geometry.fit_curve(np.c_[x, rng.uniform(-0.06, 0.06) * np.sin(x * 4)])
            b = geometry.fit_curve(
                np.c_[
                    x + rng.uniform(-0.07, 0.07),
                    a["path"][0, 1] + rng.uniform(-0.1, 0.1) * np.sin(x * 4),
                ]
            )
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)
            for left, right in (
                (a, b),
                (b, a),
                (a, dict(b, path=b["path"][::-1], tangent=-b["tangent"][::-1])),
            ):
                weights = tuple(rng.uniform(0.1, 1.0, 2))
                with patch.object(geometry, "native_nearest_samples", return_value=None):
                    expected = geometry.merge_centers(left, right, *weights)
                actual = geometry.merge_centers(left, right, *weights)
                self.assertEqual(actual is None, expected is None)
                if actual is not None:
                    self.assertEqual(actual.keys(), expected.keys())
                    for key in actual:
                        np.testing.assert_allclose(
                            actual[key], expected[key], atol=1e-12, rtol=1e-12
                        )


if __name__ == "__main__":
    unittest.main()
