"""Compiled association must preserve reference geometry and rejection decisions."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/curve_follow'))
import geometry
from native_geometry import native_fit_status

@unittest.skipUnless(native_fit_status().get('agreement_enabled'),'compile native kernel to run agreement equivalence')
class NativeAgreementTests(unittest.TestCase):
 def test_recorded_shaped_and_random_curves_keep_same_decisions(self):
  rng=np.random.default_rng(20261024)
  for i in range(300):
   n=int(rng.integers(12,130));x=np.linspace(.1,.8,n)
   a=np.c_[x,.03*np.sin(x*rng.uniform(1,8))]
   b=a.copy()+[rng.uniform(-.12,.12),rng.uniform(-.12,.12)]
   if i%5==0:b=b[::-1]
   if i%7==0:b=b[n//2:]
   if i%11==0:b=b+[1.,0.]
   with patch.object(geometry,'native_curve_agree_paths',return_value=(False,None)), patch.object(geometry,'native_curve_agreement',return_value=(False,None)):
    reference=geometry.curve_agreement(a,b)
   result=geometry.curve_agreement(a,b)
   self.assertEqual(reference is None,result is None,i)
   if result:
    for key in result:self.assertAlmostEqual(reference[key],result[key],places=12,msg=(i,key))
 def test_oversized_support_falls_back_without_changing_result(self):
  a=np.c_[np.linspace(.1,.8,220),np.zeros(220)]
  with patch.object(geometry,'native_curve_agree_paths',return_value=(False,None)), patch.object(geometry,'native_curve_agreement',return_value=(False,None)):reference=geometry.curve_agreement(a,a)
  self.assertEqual(reference,geometry.curve_agreement(a,a))
 def test_frame_cache_does_not_leak_across_observations(self):
  a=np.c_[np.linspace(.1,.8,40),np.zeros(40)]
  with geometry.geometry_frame():
   first=geometry.curve_agreement(a,a)
   shifted=geometry.curve_agreement(a,a+[0,.2])
   self.assertGreater(shifted['distance_m'],first['distance_m']+.1)
  with geometry.geometry_frame():
   self.assertAlmostEqual(geometry.curve_agreement(a,a+[0,.1])['distance_m'],.1)
if __name__=='__main__':unittest.main()
