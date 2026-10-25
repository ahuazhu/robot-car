"""Compiled point projection must preserve the independent NumPy reference."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/curve_follow'))
import geometry
from native_geometry import native_fit_status,native_support_projection
@unittest.skipUnless(native_fit_status().get('projection_enabled'),'compile the projection kernel')
class ProjectionTests(unittest.TestCase):
 def test_random_curves_queries_duplicate_points_and_endpoints_match_reference(self):
  rng=np.random.default_rng(928)
  for i in range(160):
   r=np.cumsum(rng.normal(0,.02,(int(rng.integers(3,160)),2)),axis=0)
   if i%3==0:r[1]=r[0]
   q=np.vstack((rng.normal(0,.3,(25,2)),r[[0,-1]]))
   with patch.object(geometry,'native_support_projection',return_value=None):expected=geometry.support_projection(q,r)
   actual=native_support_projection(q,r)
   self.assertIsNotNone(actual)
   for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=2e-12,rtol=1e-10)
 def test_bad_shapes_and_nonfinite_input_fall_back_without_c_buffer_reads(self):
  for q,r in ((np.zeros((2,3)),np.zeros((5,2))),(np.zeros((2,2)),np.zeros((1,2))),(np.array([[np.nan,0.]]),np.zeros((5,2)))):
   self.assertIsNone(native_support_projection(q,r))
if __name__=='__main__':unittest.main()
