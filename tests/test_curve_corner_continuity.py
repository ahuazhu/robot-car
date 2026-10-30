"""Short-overlap continuation uses old evidence, never an unconfirmed road."""
import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
from tracker import TargetTracker
from controller import CommandRamp
from steering_control import SteeringControl
from road_policy import RoadPolicy

def path(start,end,y=0.):return np.c_[np.linspace(start,end,18),np.full(18,y)]
def motion(valid=True):return dict(valid=valid,rotation=np.eye(2),translation=np.zeros(2),sigma_m=.001,
 interval_s=.1,travel_m=0.,speed_m_s=.06,yaw_rad_s=0.,source='GROUND FLOW',reason='',inliers=20,residual_m=.001)

class ContinuityTests(unittest.TestCase):
 def tracker(self):
  c=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
  c['control_frame']=dict(origin_m=[-.055,0.],body_m=[.075,.105,.085],geometry_verified=True)
  t=TargetTracker(c);t.motion.update=lambda *args:motion()
  t.pixels.measure=lambda *args:(dict(path=path(.12,.25),confidence=.9,mode='ONE LEFT'),{})
  t.update(None,0.,require_current=True);return t
 def candidate(self,t,y=.02):
  t.pixels.measure=lambda *args:(dict(path=path(.24,.65,y),confidence=.9,mode='ONE LEFT'),{})
 def test_small_compatible_successor_does_not_pause_or_refresh_old_support(self):
  for sign in (-1,1):
   t=self.tracker();old=t.model.path.copy();seen=t.model.seen.copy();self.candidate(t,sign*.02)
   r,_=t.update(None,.1)
   self.assertTrue(r['transition_pending']);self.assertEqual(r['state'],'PREDICTED')
   self.assertFalse(r['observed']);self.assertIsNotNone(r['history_continuity_evidence'])
   np.testing.assert_allclose(r['path'],old);np.testing.assert_array_equal(t.model.seen,seen)
   self.assertIsNotNone(r['candidate_path_m']);self.assertFalse(any(r['path_observed_mask']))
 def test_repeated_compatible_candidate_can_be_confirmed_without_stopping(self):
  t=self.tracker();self.candidate(t)
  first,_=t.update(None,.1);self.assertFalse(first['observed'])
  second,_=t.update(None,.2)
  self.assertTrue(second['observed']);self.assertFalse(second['transition_pending'])
  self.assertEqual(t.model.pending_count,0);self.assertTrue(np.all(t.model.seen==.2))
  self.assertIsNone(second['history_continuity_evidence'])
  controller=SteeringControl(RoadPolicy(),[-.055,0.],[.075,.105,.085],.3)
  self.assertIsNotNone(controller.command(second))
 def test_startup_expiry_dropout_and_wide_shift_cannot_continue(self):
  for case in ('startup','expired','motion','shift'):
   t=self.tracker();self.candidate(t,.12 if case=='shift' else .02)
   if case=='motion':t.motion.update=lambda *args:motion(False)
   r,_=t.update(None,.7 if case=='expired' else .1,require_current=case=='startup')
   self.assertFalse(r['transition_pending'],case);self.assertIsNone(r['history_continuity_evidence'],case)
 def test_physical_corridor_width_controls_continuation_not_camera_side(self):
  t=self.tracker();t.control_body=[.075,.105,.145];self.candidate(t)
  r,_=t.update(None,.1);self.assertFalse(r['transition_pending'])

class RollingRestartTests(unittest.TestCase):
 def ramp(self):
  r=CommandRamp()
  for i in range(10):r.apply((100,60),i*.1)
  return r
 def test_short_rolling_pause_outputs_zero_but_does_not_restart_at_first_step(self):
  r=self.ramp();self.assertEqual(r.pause(1.,rolling=True),(0,0))
  out=r.apply((100,60),1.1);cold=CommandRamp().apply((100,60),1.1)
  self.assertGreater(max(out),max(cold));self.assertLessEqual(max(out),100)
  self.assertAlmostEqual(out[1]/out[0],.6,delta=.02)
 def test_explicit_stop_no_motion_and_long_pause_reset_resume_effort(self):
  for how in ('zero','notrolling','long'):
   r=self.ramp()
   if how=='zero':self.assertEqual(r.apply((0,0),1.),(0,0))
   else:self.assertEqual(r.pause(2. if how=='long' else 1.,rolling=how=='long'),(0,0))
   self.assertLessEqual(max(r.apply((100,60),2.1)),18)

if __name__=='__main__':unittest.main()
