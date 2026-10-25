"""Single nearest-point control over observed and predicted centre geometry."""
import json,math,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
from geometry import choose_target,nearest_forward_point,support_projection
from steering_control import SteeringControl
from road_policy import RoadPolicy
from tracker import TargetTracker
from ground_motion import command_motion
class NearestTargetTests(unittest.TestCase):
 def fixture(self):return json.loads((ROOT/'tests/fixtures/curve_nearest_path_stop.json').read_text())
 def test_actual_stop_has_supported_forward_target_instead_of_none(self):
  f=self.fixture();p=np.asarray(f['path']);r=choose_target(p,reference_origin_m=f['origin_m'],observation_sigma_m=f['sigma_m'])
  self.assertIsNotNone(r);self.assertEqual(r['target_selection'],'NEAREST_CURVE')
  expected,position=nearest_forward_point(r['reference_path'])
  np.testing.assert_allclose(r['point'],expected,atol=1e-10)
  self.assertGreater(r['distance_m'],.4);self.assertLess(r['distance_m'],.45)
  self.assertLessEqual(r['reference_sigma_m'],.05)
  self.assertLess(r['target_speed_scale'],1.)
  np.testing.assert_array_equal(r['reference_path'],p-np.asarray(f['origin_m']))
 def test_projection_selects_nearest_point_inside_segment_not_vertex(self):
  point,s=nearest_forward_point(np.array([[.4,-.1],[.4,.1]]))
  np.testing.assert_allclose(point,[.4,0.]);self.assertAlmostEqual(s,.1)
 def test_forward_part_is_available_even_when_global_minimum_is_behind_car(self):
  point,_=nearest_forward_point(np.array([[-.2,.2],[.2,.2],[.4,.2]]))
  self.assertAlmostEqual(point[0],.06);self.assertAlmostEqual(point[1],.2)
  self.assertIsNone(nearest_forward_point(np.array([[-.2,0.],[-.1,0.]])))
 def test_nearest_control_is_symmetric_and_curve_provenance_remains_observed(self):
  f=self.fixture();points=[]
  for sign in (1.,-1.):
   path=np.asarray(f['path'])*[1,sign]
   r=choose_target(path,reference_origin_m=f['origin_m'])
   self.assertEqual(r['target_selection'],'NEAREST_CURVE');points.append(r['point'])
  np.testing.assert_allclose(points[0]*[1,-1],points[1],atol=1e-12)
 def test_large_distance_and_uncertainty_still_reject_target(self):
  path=np.c_[np.linspace(.8,1.,15),np.zeros(15)]
  self.assertIsNone(choose_target(path))
  f=self.fixture();self.assertIsNone(choose_target(f['path'],observation_sigma_m=.06))
 def test_latency_alignment_keeps_nearest_target_on_fitted_path(self):
  f=self.fixture();path=np.asarray(f['path']);target=choose_target(path,reference_origin_m=f['origin_m'])
  r=dict(valid=True,state='OBSERVED',confidence=.8,control_target_m=target['point'],
         control_path_m=target['reference_path'],path=path,reference_sigma_m=target['reference_sigma_m'],
         target_selection='NEAREST_CURVE',target_speed_scale=target['target_speed_scale'],path_curvature_m_inv=target['curvature'])
  c=SteeringControl(RoadPolicy(),f['origin_m'],None,.3)
  # Valid zero-twist estimate: even infinitesimal rounding at tangency used to
  # make a preview-circle re-selection reject an otherwise valid nearest goal.
  m=command_motion(0.,0.,.2,RoadPolicy());m['valid']=True
  c.observe(m,2.,allow_feedback=False)
  cmd=c.command(r,age_s=.2);self.assertIsNotNone(cmd)
  np.testing.assert_allclose(c.status['latency_prediction']['point_m'],target['point'],atol=1e-12)
  self.assertEqual(c.status['target_selection'],'ADAPTIVE_PURSUIT')
 def test_complete_tracker_can_control_actual_accepted_single_edge(self):
  f=self.fixture();cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
  cal['control_frame']={'origin_m':f['origin_m']}
  t=TargetTracker(cal);t.pixels.measure=lambda *args:(dict(path=np.asarray(f['path']),mode='ONE RIGHT',confidence=.78),{})
  result,_=t.update(None,0.)
  self.assertTrue(result['valid']);self.assertEqual(result['target_selection'],'NEAREST_CURVE')
  self.assertIsNotNone(result['control_target_m']);self.assertTrue(result['observed'])

 def test_complete_tracker_prediction_metadata_serializes_for_live_web_and_logs(self):
  f=self.fixture();cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
  for path in (np.asarray(f['path']),np.c_[np.linspace(.2,.6,40),np.zeros(40)]):
   tracker=TargetTracker(cal)
   tracker.pixels.measure=lambda *args:(dict(path=path,mode='ONE RIGHT',confidence=.78),{})
   result,_=tracker.update(None,0.)
   decoded=json.loads(json.dumps(result))
   self.assertIsInstance(decoded['target_inferred'],bool)
   self.assertIsNotNone(decoded['target_m'])

 def test_user_reported_orange_extension_is_absent_without_observation_history(self):
  import cv2
  cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
  cal['control_frame']=json.loads((ROOT/'maixcam2/curve_follow/runtime.json').read_text())['control_frame']
  raw=cv2.imread(str(ROOT/'tests/fixtures/curve_unfounded_prediction.png'))
  tracker=TargetTracker(cal);result,details=tracker.update(raw,0.)
  self.assertTrue(result['valid']);self.assertTrue(result['observed'])
  self.assertEqual(result['control_target_source'],'OBSERVED')
  self.assertFalse(result['target_inferred']);self.assertFalse(result['reference_inferred'])
  self.assertTrue(all(result['path_observed_mask']))
  # There is no generated prefix that the old far-only quadratic could turn
  # outside the road. Every displayed/control sample has measured support.
  np.testing.assert_allclose(np.asarray(result['control_path_m'])+cal['control_frame']['origin_m'],result['path'],atol=1e-12)
  nearest,_=nearest_forward_point(result['control_path_m'])
  np.testing.assert_allclose(nearest,result['control_target_m'],atol=1e-10)
