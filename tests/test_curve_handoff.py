"""Short-overlap continuity does not authorize unrelated or expired roads."""
import sys,json,math,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
from geometry import fit_curve,curve_handoff,curve_agreement
from road_model import LocalRoadModel
from road_policy import RoadPolicy
from tracker import TargetTracker
from steering_control import SteeringControl

CAL=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())

def path(start,end,y=0.):return fit_curve(np.c_[np.linspace(start,end,60),np.full(60,y)])['path']
def measurement(p,q=.9):return dict(path=p,confidence=q,mode='ONE RIGHT')
def scene(valid=True):return dict(valid=valid,rotation=np.eye(2),translation=np.zeros(2),sigma_m=.0015,
 interval_s=.1,travel_m=0.,speed_m_s=.02,yaw_rad_s=0.,source='GROUND FLOW',reason='',inliers=20,residual_m=.001)

class HandoffGeometryTests(unittest.TestCase):
 def test_proves_short_ordered_overlap_without_creating_curve_points(self):
  old=path(.12,.24);new=path(.22,.65)
  self.assertIsNone(curve_agreement(old,new))
  result=curve_handoff(old,new,.009,.75,.06)
  self.assertIsNotNone(result);self.assertLess(result['distance_m'],1e-8)
  self.assertLess(result['old_exit_overlap_m'],.06)
 def test_rejects_crossing_reversal_offset_far_gap_and_contained_path(self):
  old=path(.12,.24)
  crossing=np.c_[np.full(40,.23),np.linspace(-.2,.2,40)]
  for new in (path(.22,.65,.04),path(.3,.65),path(.22,.65)[::-1],crossing,path(.14,.23)):
   self.assertIsNone(curve_handoff(old,new,.009,.75,.06))
 def test_curve_and_mirror_have_same_proof(self):
  angle=np.linspace(-.3,.3,80)
  curve=np.c_[.2+np.sin(angle)*.4,(1-np.cos(angle))*.4]
  a,b=curve[:45],curve[42:]
  left=curve_handoff(a,b,.01,.75,.06);right=curve_handoff(a*[1,-1],b*[1,-1],.01,.75,.06)
  self.assertIsNotNone(left)
  self.assertEqual(left,right)
 def test_nan_and_tiny_inputs_fail_closed(self):
  old=path(.12,.24)
  for new in (np.full((20,2),np.nan),old[:3],np.zeros((8,2))):
   self.assertIsNone(curve_handoff(old,new,.01,.75,.06))

class HandoffModelTests(unittest.TestCase):
 def model(self):
  m=LocalRoadModel();m.update(measurement(path(.12,.24)),0.);return m
 def test_continuity_uses_existing_evidence_count_while_moving(self):
  m=self.model();old_seen=m.seen.copy();new=path(.22,.65)
  e=m.update(measurement(new),.1,allow_handoff=True)
  self.assertEqual(e['association_kind'],'COVERAGE_HANDOFF');self.assertTrue(e['continuation_allowed'])
  self.assertFalse(e['observed']);self.assertEqual(e['required'],math.ceil(m.policy.confirmation_evidence/.9))
  self.assertEqual(e['reject'],'COVERAGE HANDOFF PENDING')
  np.testing.assert_array_equal(m.seen,old_seen)
  e=m.update(measurement(new),.2,allow_handoff=True)
  self.assertTrue(e['reacquired']);self.assertTrue(e['observed'])
  np.testing.assert_array_equal(m.path,new)
  self.assertTrue(np.all(m.seen==.2))
 def test_missing_motion_or_old_support_never_authorizes_moving_handoff(self):
  for now,allow in ((.1,False),(.7,True)):
   m=self.model();e=m.update(measurement(path(.22,.65)),now,allow_handoff=allow)
   self.assertFalse(e['observed']);self.assertFalse(e['continuation_allowed'])
   self.assertEqual(e['required'],m.policy.stopped_confirmation_frames)
 def test_true_geometric_conflict_is_still_confirmed_while_stopped(self):
  m=self.model();e=m.update(measurement(path(.12,.65,.20)),.1,allow_handoff=True)
  self.assertFalse(e['observed']);self.assertFalse(e['continuation_allowed'])
  self.assertIsNone(e['handoff_evidence'])
 def test_conflicting_followup_cannot_inherit_a_continuity_proof(self):
  m=self.model();m.update(measurement(path(.22,.65)),.1,allow_handoff=True)
  e=m.update(measurement(path(.22,.65,.2)),.2,allow_handoff=True)
  self.assertFalse(e['continuation_allowed']);self.assertFalse(e['observed']);self.assertEqual(m.pending_count,1)

class HandoffTrackerTests(unittest.TestCase):
 def tracker(self):
  t=TargetTracker(CAL);t.motion.update=lambda *args:scene()
  t.pixels.measure=lambda *args:(measurement(path(.12,.24)),{})
  t.update(None,0.,require_current=True);return t
 def test_pending_reference_uses_only_old_supported_path_and_no_freshness_credit(self):
  t=self.tracker();old=t.model.path.copy()
  t.pixels.measure=lambda *args:(measurement(path(.22,.65)),{})
  r,_=t.update(None,.1)
  self.assertTrue(r['transition_pending']);self.assertEqual(r['state'],'PREDICTED')
  self.assertFalse(r['observed']);self.assertTrue(r['target_inferred'])
  np.testing.assert_array_equal(r['path'],old)
  self.assertIsNotNone(r['candidate_path_m']);self.assertIsNotNone(r['association_prior_path_m'])
 def test_startup_never_arms_from_a_pending_reference(self):
  t=self.tracker();t.pixels.measure=lambda *args:(measurement(path(.22,.65)),{})
  r,_=t.update(None,.1,require_current=True)
  self.assertFalse(r['transition_pending']);self.assertEqual(r['state'],'CONFIRMING');self.assertFalse(r['valid'])
 def test_dropout_or_expiry_cannot_continue_old_reference(self):
  for valid,now in ((False,.1),(True,.7)):
   t=self.tracker();t.motion.update=lambda *args:scene(valid)
   t.pixels.measure=lambda *args:(measurement(path(.22,.65)),{})
   r,_=t.update(None,now)
   self.assertFalse(r['transition_pending']);self.assertEqual(r['state'],'CONFIRMING')

class HandoffControlTests(unittest.TestCase):
 def control(self,result,speed=.04):
  c=SteeringControl(RoadPolicy(),[0.,0.],[.075,.105,.085],.3)
  c.status=dict(body_speed_m_s=speed,body_yaw_rad_s=0.,speed_upper_m_s=speed+.002,
                motion_sigma_m=.0015,motion_interval_s=.1)
  c.preview_speed=speed+.002;c.preview_interval=.1
  p=path(.1,.5,.01)
  r=dict(valid=True,state='PREDICTED',confidence=.8,control_target_m=p[0],control_path_m=p,path=p,
         path_curvature_m_inv=0.,road_model=dict(sigma_m=.005,blind_distance_m=.005,age_s=.05,length_m=.4))
  r.update(result);return c,c.command(r,age_s=.01)
 def test_pending_keeps_useful_duty_but_gets_a_bounded_motor_lease(self):
  normal,base=self.control({});pending,slow=self.control(dict(transition_pending=True))
  self.assertIsNotNone(base);self.assertIsNotNone(slow)
  self.assertEqual(slow,base) # Quality already decays in the shared drive law.
  lease=pending.status['handoff']['lease_s']
  self.assertGreater(lease,0.);self.assertLessEqual(lease,.15)
  self.assertLessEqual(lease,pending.status['handoff']['remaining_time_s'])
  self.assertLessEqual(lease*(.04+.002),pending.status['handoff']['remaining_distance_m'])
 def test_no_motion_or_remaining_support_blocks_pending_command(self):
  for model in (dict(sigma_m=.005,blind_distance_m=.06,age_s=.05,length_m=.4),
                dict(sigma_m=.005,blind_distance_m=.005,age_s=.65,length_m=.4)):
   c,command=self.control(dict(transition_pending=True,road_model=model))
   self.assertIsNone(command)
  c,command=self.control(dict(transition_pending=True),speed=0.)
  self.assertIsNone(command)

if __name__=='__main__':unittest.main()
