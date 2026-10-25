"""Batching must preserve the original detector on a real six-bar crossing."""
import sys,json,unittest
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'));sys.path.insert(0,str(ROOT/'tests/fixtures'))
from tracker import CurvePixels
from crosswalk import CrosswalkDetector,CrosswalkPolicy
from crosswalk_scalar_reference import detect_reference
class BatchTests(unittest.TestCase):
 def test_actual_crossing_geometry_matches_scalar_detector(self):
  cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text());p=CurvePixels(cal)
  image=cv2.imread(str(ROOT/'tests/fixtures/curve_zebra_turn.jpg'))[:240,:320]
  road,details=p.measure(image);self.assertIsNotNone(road)
  detector=CrosswalkDetector(p,CrosswalkPolicy());expected=detect_reference(detector,details,road,1.)
  actual=detector.detect(details,road,1.);self.assertIsNotNone(expected);self.assertIsNotNone(actual)
  self.assertEqual(actual['bar_count'],expected['bar_count'])
  for key in ('near_m','far_m','confidence','normal','corners','bar_length_m','bar_width_m'):
   np.testing.assert_allclose(actual[key],expected[key],atol=1e-10,rtol=1e-10)
if __name__=='__main__':unittest.main()
