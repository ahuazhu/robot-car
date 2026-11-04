"""Same-frame geometry reuse must not alias mutable scores or fresh observations."""

from tests.support import ROOT
import unittest
from unittest.mock import patch
import numpy as np

from robot.core import geometry


class FitCacheTests(unittest.TestCase):
    def test_same_geometry_fits_once_and_view_scores_remain_independent(self):
        p = np.c_[np.linspace(0.1, 0.7, 50), np.zeros(50)]
        with patch.object(geometry, "native_chain_fit", wraps=geometry.native_chain_fit) as calls:
            with geometry.geometry_frame():
                a = geometry.fit_curve(p)
                a["score"] = 7.0
                b = geometry.fit_curve(p.copy())
                self.assertNotIn("score", b)
                self.assertEqual(calls.call_count, 1)
                np.testing.assert_array_equal(a["path"], b["path"])
            with geometry.geometry_frame():
                geometry.fit_curve(p)
            self.assertEqual(calls.call_count, 2)

    def test_normal_offset_cache_keeps_independent_metadata(self):
        p = np.c_[np.linspace(0.1, 0.7, 50), np.zeros(50)]
        with geometry.geometry_frame():
            edge = geometry.fit_curve(p)
            a = geometry.inward_center(edge, "left")
            a["overlap_m"] = 123
            b = geometry.inward_center(edge, "left")
            self.assertNotIn("overlap_m", b)
            mirrored = geometry.inward_center(edge, "right")
            np.testing.assert_allclose(a["path"] * [1, -1], mirrored["path"], atol=1e-12)
