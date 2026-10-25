"""The optional numerical kernel must preserve the reference fit and rejection."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/curve_follow'))
import geometry
from native_geometry import native_fit_status,native_local_fit

@unittest.skipUnless(native_fit_status()['enabled'],'compile native kernel to run equivalence checks')
class NativeFitTests(unittest.TestCase):
    def test_quadratic_core_preserves_curved_noisy_and_invalid_fits(self):
        rng=np.random.default_rng(232)
        cases=[np.zeros((10,2)),np.array([[0.,0.],[float('nan'),0.],[1.,0.]])]
        for i in range(80):
            s=np.linspace(0,rng.uniform(.09,1.5),rng.integers(15,190))
            points=np.column_stack((s,.12*np.sin(rng.uniform(-5,5)*s)))
            cases.append(points+rng.normal(0,.0008,points.shape))
        for points in cases:
            with patch.object(geometry,'native_local_fit',return_value=None):reference=geometry.fit_curve(points)
            actual=geometry.fit_curve(points)
            self.assertEqual(reference is None,actual is None)
            if actual is not None:
                for key in ('path','tangent','curvature'):
                    np.testing.assert_allclose(actual[key],reference[key],atol=1e-7,rtol=1e-8)
                self.assertAlmostEqual(actual['residual'],reference['residual'],places=10)
    def test_sparse_input_returns_to_reference_support_selection(self):
        points=np.column_stack((np.arange(10.),np.zeros(10)))
        self.assertIsNone(native_local_fit(points,np.arange(10.)))
    def test_wrong_shapes_never_reach_the_c_buffer(self):
        with self.assertRaises(ValueError):native_local_fit(np.zeros((10,3)),np.arange(10.))

if __name__=='__main__':unittest.main()
