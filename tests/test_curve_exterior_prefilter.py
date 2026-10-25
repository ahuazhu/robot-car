"""Conservative contour rejection must preserve accepted road geometry."""
import sys,json,unittest
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
from tracker import CurvePixels
class ExteriorPrefilterTests(unittest.TestCase):
 def test_real_glare_crosswalk_and_bends_match_unfiltered_contours(self):
  cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
  for name in ('curve_shadow_glare.jpg','curve_zebra_turn.jpg','right_bend.jpg','curve_white_markings.png'):
   with self.subTest(frame=name):
    bgr=cv2.imread(str(ROOT/'tests/fixtures'/name))[:240,:320]
    self.assertEqual(bgr.shape[:2],(240,320))
    pixels=CurvePixels(cal);prepared=pixels.prepare(bgr)
    for normalized in (False,True):
     for cutoff in pixels.white_value_levels:
      a,da=pixels.extract(bgr,prepared=prepared,normalized=normalized,white_value_min=cutoff)
      unfiltered=(*prepared[:-1],np.ones_like(prepared[-1]))
      b,db=pixels.extract(bgr,prepared=unfiltered,normalized=normalized,white_value_min=cutoff)
      self.assertEqual(da['counts'],db['counts'])
      for side in ('left','right'):
       self.assertEqual(a[side] is None,b[side] is None)
       if a[side] is not None:np.testing.assert_array_equal(a[side]['path'],b[side]['path'])
