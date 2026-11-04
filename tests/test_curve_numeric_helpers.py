"""Numeric equivalence for specialized geometry operations used on the camera."""

from tests.support import ROOT
import unittest
import numpy as np

from robot.core.geometry import differential, arc, _quantile_1d


class NumericGeometryTests(unittest.TestCase):
    def test_nonuniform_arc_derivatives_match_numpy(self):
        rng = np.random.default_rng(4024)
        for count in (5, 17, 60, 130):
            s = np.cumsum(rng.uniform(0.002, 0.03, count))
            path = np.c_[np.sin(s), np.cos(s)]
            distance = arc(path)
            derivative = np.gradient(path, distance, axis=0)
            tangent = derivative / np.maximum(np.linalg.norm(derivative, axis=1)[:, None], 1e-8)
            second = np.gradient(tangent, distance, axis=0)
            expected = tangent[:, 0] * second[:, 1] - tangent[:, 1] * second[:, 0]
            actual_t, actual_k = differential(path)
            np.testing.assert_allclose(actual_t, tangent, atol=1e-12)
            np.testing.assert_allclose(actual_k, expected, atol=1e-12)

    def test_linear_quantile_matches_numpy_on_small_finite_samples(self):
        rng = np.random.default_rng(4024)
        for count in (1, 2, 3, 7, 40, 130):
            for values in (rng.normal(size=count), np.zeros(count), np.arange(count, dtype=float)):
                for q in (0, 0.2, 0.5, 0.75, 0.8, 0.9, 1):
                    self.assertAlmostEqual(
                        float(_quantile_1d(values, q)), float(np.quantile(values, q)), places=12
                    )


if __name__ == "__main__":
    unittest.main()
