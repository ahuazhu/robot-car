"""First-circle and closest-segment native geometry match NumPy search."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/curve_follow'))
import geometry
from native_geometry import native_fit_status
@unittest.skipUnless(native_fit_status().get('pursuit_enabled'),'compile pursuit kernel')
class PursuitKernelTests(unittest.TestCase):
 def test_random_walks_degenerate_segments_and_both_turn_directions(self):
  rng=np.random.default_rng(2458)
  for i in range(300):
   p=np.cumsum(rng.normal(0,.03,(int(rng.integers(2,100)),2)),axis=0)+[.15,0.]
   if i%3==0:p[1]=p[0]
   for radius in (.12,.25,.45):
    actual=geometry.pursuit_point(p,radius)
    with patch.object(geometry,'native_pursuit_selection',return_value=(False,None)):
     expected=geometry.pursuit_point(p,radius)
     anchor=geometry.nearest_forward_point(p)
    actual_anchor=geometry.nearest_forward_point(p)
    self.assertEqual(actual is None,expected is None)
    self.assertEqual(actual_anchor is None,anchor is None)
    if anchor is not None:
     for a,b in zip(actual_anchor,anchor):np.testing.assert_allclose(a,b,atol=1e-12)
    if actual is None:continue
    self.assertEqual(actual['mode'],expected['mode'])
    for k in ('point','position_m','distance_m','speed_scale'):np.testing.assert_allclose(actual[k],expected[k],atol=1e-11,rtol=1e-9)
 def test_nonfinite_and_malformed_path_is_rejected(self):
  for p in ([],[[.1,0]],[[np.nan,0],[.2,0]],[[.1,0,0],[.2,0,0]]):
   self.assertIsNone(geometry.pursuit_point(p,.2))
