"""Exposure regression with a real crossing and its frozen camera geometry."""
import json,sys,unittest
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
from tracker import CurvePixels
from crosswalk import CrosswalkDetector,CrosswalkPolicy,CrosswalkTask

class CrosswalkContrastTests(unittest.TestCase):
 def setUp(self):
  self.fixture=ROOT/'tests/fixtures/crosswalk_20261030'
  self.pixels=CurvePixels(json.loads((self.fixture/'calibration.json').read_text()))
  self.detector=CrosswalkDetector(self.pixels,CrosswalkPolicy())
  self.raw=cv2.imread(str(self.fixture/'seven_bars.png'))
 def detect(self,image):
  road,details=self.pixels.measure(image)
  self.assertIsNotNone(road,'The photographed road must remain measurable')
  return self.detector.detect(details,road,1.)
 def test_real_crossing_survives_exposure_changes(self):
  for gain in (1.,.85,.7,.55,.4):
   with self.subTest(gain=gain):
    obs=self.detect(np.uint8(self.raw.astype(float)*gain))
    self.assertIsNotNone(obs);self.assertGreaterEqual(obs['bar_count'],4)
    self.assertAlmostEqual(obs['near_m'],.45,delta=.02)
    self.assertEqual(self.detector.status['reason'],'DETECTED')
    self.assertEqual(self.detector.status['source'],'raw' if gain==1. else 'contrast')
 def test_real_crossing_survives_a_local_shadow_and_mild_blur(self):
  shade=np.ones(self.raw.shape[:2],float);shade[:,:160]=.6
  shade=cv2.GaussianBlur(shade,(61,61),0)
  for image in (np.uint8(self.raw*shade[:,:,None]),
                cv2.GaussianBlur(np.uint8(self.raw.astype(float)*.7),(3,3),0)):
   obs=self.detect(image)
   self.assertIsNotNone(obs);self.assertGreaterEqual(obs['bar_count'],4)
   self.assertAlmostEqual(obs['near_m'],.45,delta=.02)
 def test_raw_fast_path_remains_identical_without_contrast(self):
  road,details=self.pixels.measure(self.raw)
  expected=self.detector.detect({k:v for k,v in details.items() if k!='contrast_value'},road,1.)
  actual=self.detector.detect(details,road,1.)
  self.assertEqual(actual,expected)
  self.assertTrue(all(p['source']=='raw' for p in self.detector.status['passes']))
 def test_bend_and_shadow_are_not_crosswalks(self):
  image=cv2.imread(str(self.fixture/'bend_without_crossing.png'))
  for gain in (1.,.7,.4):
   for blur in (1,3):
    modified=np.uint8(image.astype(float)*gain)
    if blur>1:modified=cv2.GaussianBlur(modified,(blur,blur),0)
    road,details=self.pixels.measure(modified)
    self.assertIsNone(self.detector.detect(details,road or {},1.),(gain,blur))
 def test_diagnostics_distinguish_missing_road_and_missing_pattern(self):
  self.assertIsNone(self.detector.detect({}, {}, 1.))
  self.assertEqual(self.detector.status['reason'],'NO_ROAD_SUPPORT')
  road,details=self.pixels.measure(self.raw)
  self.detector.detect(details,dict(road,confidence=.3),1.)
  self.assertEqual(self.detector.status['reason'],'ROAD_CONFIDENCE_LOW')
 def test_detections_do_not_bypass_multi_frame_confirmation(self):
  task=CrosswalkTask(CrosswalkPolicy())
  road,details=self.pixels.measure(np.uint8(self.raw.astype(float)*.7))
  for i in range(3):
   obs=self.detector.detect(details,road,i*.1)
   state=task.update(obs,None,i*.1)
   self.assertEqual(state['confirmed'],i==2)
   self.assertEqual(state['observation_association'],'NEW' if i==0 else 'MATCHED')
 def test_unmatched_observation_is_explicit_without_discarding_confirmed_event(self):
  task=CrosswalkTask(CrosswalkPolicy());road,details=self.pixels.measure(self.raw)
  for i in range(3):task.update(self.detector.detect(details,road,i*.1),None,i*.1)
  old_id=task.identity;obs=self.detector.detect(details,road,.3)
  obs['near_m']+=.3;obs['far_m']+=.3
  state=task.update(obs,None,.3)
  self.assertEqual(state['observation_association'],'UNMATCHED_CONFIRMED')
  self.assertEqual(state['event_id'],old_id)

if __name__=='__main__':unittest.main()
