"""Accelerated contour probes preserve the independent NumPy reference."""
import sys,json,unittest
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
from tracker import CurvePixels
from native_geometry import native_boundary_chains,native_fit_status
@unittest.skipUnless(native_fit_status().get('boundary_enabled'),'compile the contour kernel')
class BoundaryKernelTests(unittest.TestCase):
 def test_real_contours_all_thresholds_seams_and_reversal_match_reference(self):
  cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text());pixel=CurvePixels(cal);accepted=0
  names=('curve_shadow_glare.jpg','curve_zebra_turn.jpg','right_bend.jpg','curve_white_markings.png','curve_unfounded_prediction.png')
  for name in names:
   img=cv2.imread(str(ROOT/'tests/fixtures'/name))[:240,:320];prepared=pixel.prepare(img)
   _,sat,raw,rawgray,green,contrast,ngray,_=prepared
   for normalized in (False,True):
    gray=ngray if normalized else rawgray;value=contrast if normalized else raw
    for cutoff in pixel.white_value_levels:
     white=(sat<=105)&(value>=cutoff)&pixel.visible
     codes=gray.astype(np.uint8)+2*green.astype(np.uint8)+4*white.astype(np.uint8)
     contours,_=cv2.findContours(white.astype(np.uint8)*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
     minimum=max(5,int(np.ceil(pixel.policy.minimum_segment_m/pixel.step)))
     for contour in contours:
      points=contour[:,0,:]
      if len(points)<2*minimum:continue
      for points in (points,np.roll(points,7,axis=0),points[::-1]):
       actual=native_boundary_chains(points,codes,gray,pixel.visible,pixel.xs,pixel.ys,pixel.step,pixel.exterior_distances,minimum,pixel.policy.minimum_segment_m)
       expected=pixel.contour_chains_reference(points,codes,gray,minimum)
       self.assertIsNotNone(actual);self.assertEqual(len(actual),len(expected))
       accepted+=len(expected)
       for a,b in zip(actual,expected):
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-12,rtol=1e-10)
  self.assertGreater(accepted,30)
 def test_entire_extraction_keeps_candidate_geometry_and_scores(self):
  cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text());pixel=CurvePixels(cal)
  image=cv2.imread(str(ROOT/'tests/fixtures/curve_zebra_turn.jpg'))[:240,:320]
  for normalized in (False,True):
   actual,da=pixel.extract(image,normalized=normalized)
   with patch('tracker.native_boundary_chains',return_value=None):expected,de=pixel.extract(image,normalized=normalized)
   self.assertEqual(da['counts'],de['counts'])
   for side in ('left','right'):
    for a,b in zip(da['candidates'][side],de['candidates'][side]):
     np.testing.assert_allclose(a['path'],b['path'],atol=1e-12)
     np.testing.assert_allclose(a['center']['path'],b['center']['path'],atol=1e-12)
     self.assertAlmostEqual(a['score'],b['score'])
