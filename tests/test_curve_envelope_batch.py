"""Batch 28 corners without changing future poses, coverage or stop envelopes."""

from tests.support import ROOT
import json, math, unittest
import numpy as np

from robot.navigation.steering_control import SteeringControl
from robot.core.road_policy import RoadPolicy
from tests.reference.curve_envelope_reference import envelope_reference


class EnvelopeBatchTests(unittest.TestCase):
    def test_random_curves_partial_support_visibility_and_command_time_transform(self):
        rng = np.random.default_rng(178)
        cal = json.loads((ROOT / "tests/fixtures/camera_pose_20261031_191327.json").read_text())
        for i in range(120):
            x = np.linspace(-0.15, 0.7, 80)
            k = rng.uniform(-4, 4)
            path = np.c_[x, 0.5 * k * x * x + rng.uniform(-0.08, 0.08)]
            c = SteeringControl(
                RoadPolicy(),
                [-0.055, 0.0],
                [0.075, 0.105, 0.085],
                0.3,
                calibration=cal if i % 2 else None,
            )
            theta = rng.uniform(-0.1, 0.1)
            co, si = math.cos(theta), math.sin(theta)
            transform = (
                (np.array([[co, si], [-si, co]]), np.array([-0.02, 0.01])) if i % 3 == 0 else None
            )
            args = (path, k, 0.01, 0.3)
            kwargs = dict(
                observed_path=path[i % 25 :],
                speed_m_s=0.15,
                sensor_transform=transform,
                lookahead_m=0.24,
            )
            a = c.envelope(*args, **kwargs)
            b = envelope_reference(c, *args, **kwargs)
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                if isinstance(a[key], (str, type(None))):
                    self.assertEqual(a[key], b[key])
                else:
                    self.assertAlmostEqual(a[key], b[key], delta=1e-10)
