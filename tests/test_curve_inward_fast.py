"""Complete-offset fast path preserves scalar segment reference geometry."""

from tests.support import ROOT
import unittest
import numpy as np

from robot.core.geometry import arc, fit_curve, inward_center


def scalar_reference(edge, side, width=0.3):
    offset = width / 2 if side == "right" else -width / 2
    runs = []
    run = []
    for point, tangent, curvature in zip(edge["path"], edge["tangent"], edge["curvature"]):
        if 1 - offset * curvature > 0.2:
            run.append([point[0] - offset * tangent[1], point[1] + offset * tangent[0]])
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    options = []
    for run in runs:
        points = np.array(run)
        if len(points) < 8 or arc(points)[-1] < 0.075:
            continue
        fitted = fit_curve(points)
        if fitted is not None:
            options.append(fitted)
    return (
        min(options, key=lambda c: np.hypot(c["path"][:, 0], c["path"][:, 1]).min())
        if options
        else None
    )


class InwardFastTests(unittest.TestCase):
    def compare(self, edge):
        for side in ("left", "right"):
            expected = scalar_reference(edge, side)
            actual = inward_center(edge, side)
            self.assertEqual(actual is None, expected is None)
            if expected is not None:
                for name in ("path", "tangent", "curvature"):
                    np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=1e-12)
                self.assertAlmostEqual(actual["length"], expected["length"], places=12)
                self.assertAlmostEqual(actual["residual"], expected["residual"], places=12)

    def test_regular_curves_and_reversed_order_match_scalar_reference(self):
        rng = np.random.default_rng(20261101)
        for _ in range(40):
            x = np.linspace(0.1, rng.uniform(0.3, 1.0), int(rng.integers(12, 80)))
            points = np.c_[x, rng.uniform(-0.08, 0.08) * np.sin(x * rng.uniform(0.5, 3.0))]
            for ordered in (points, points[::-1]):
                edge = fit_curve(ordered)
                self.assertIsNotNone(edge)
                self.compare(edge)

    def test_cusp_splits_and_invalid_parts_still_use_only_supported_runs(self):
        edge = fit_curve(np.c_[np.linspace(0.1, 0.7, 40), np.zeros(40)])
        for sign in (-1, 1):
            for begin, end in ((0, 10), (9, 20), (0, len(edge["path"]))):
                changed = dict(edge, curvature=edge["curvature"].copy())
                changed["curvature"][begin:end] = sign * 6.0
                self.compare(changed)

    def test_short_regular_components_remain_rejected(self):
        for count, length in ((7, 0.2), (10, 0.05)):
            path = np.c_[np.linspace(0.1, 0.1 + length, count), np.zeros(count)]
            edge = dict(
                path=path, tangent=np.tile([1.0, 0.0], (count, 1)), curvature=np.zeros(count)
            )
            self.compare(edge)
            self.assertIsNone(inward_center(edge, "right"))


if __name__ == "__main__":
    unittest.main()
