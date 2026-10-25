"""Numerical batching must keep curve geometry and rejection decisions."""
import sys, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/curve_follow'))
import geometry
from native_geometry import native_fit_status
@unittest.skipUnless(native_fit_status().get('chain_enabled'),'compile chain kernel for equivalence checks')
class NativeChainTests(unittest.TestCase):
 def compare(self,points):
  with patch.object(geometry,'native_chain_fit',return_value=(False,None)):
   reference=geometry.fit_curve(points)
  actual=geometry.fit_curve(points)
  self.assertEqual(reference is None,actual is None)
  if actual:
   for key in ('path','tangent','curvature'):
    np.testing.assert_allclose(actual[key],reference[key],atol=1e-8,rtol=1e-8,err_msg=key)
   self.assertAlmostEqual(actual['length'],reference['length'],places=10)
   self.assertAlmostEqual(actual['residual'],reference['residual'],places=10)
 def test_random_noisy_reversed_and_duplicate_chains(self):
  rng=np.random.default_rng(20261024)
  for i in range(500):
   n=int(rng.integers(8,500));x=np.linspace(.05,rng.uniform(.12,1),n)
   points=np.c_[x,rng.uniform(.01,.1)*np.sin(x*rng.uniform(1,12))]
   points+=rng.normal(0,rng.uniform(0,.001),points.shape)
   if i%5==0:points=points[::-1]
   if i%7==0:points=np.repeat(points,2,axis=0)
   self.compare(points)
 def test_invalid_chains_and_bounded_fallback(self):
  for points in (np.zeros((0,2)),np.zeros((20,2)),np.c_[np.linspace(0,.04,10),np.zeros(10)],np.full((5,2),np.nan),np.c_[np.linspace(0,3,100),np.zeros(100)],np.c_[np.linspace(.1,.8,1700),np.zeros(1700)]):self.compare(points)
