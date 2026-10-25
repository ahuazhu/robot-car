"""Search may rotate only inside actual observed space, with bounded pulses."""
import math,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/curve_follow'))
from road_policy import RoadPolicy
from road_recovery import RoadSearch

def motion(yaw=0.,travel=0.,valid=True):
 c,s=math.cos(yaw),math.sin(yaw)
 return dict(valid=valid,rotation=np.array([[c,s],[-s,c]]),translation=np.array([-travel,0.]),sigma_m=.0001,interval_s=.1,travel_m=abs(travel),speed_m_s=travel/.1,yaw_rad_s=yaw/.1)
def road(start=-.4,end=.8,y=0.):
 p=np.c_[np.linspace(start,end,90),np.full(90,y)]
 return dict(observed=True,valid=True,path=p,path_observed_mask=[True]*len(p),road_model=dict(sigma_m=.005))
def missing():return dict(observed=False,valid=False)
class ActiveSearchTests(unittest.TestCase):
 def ready(self):
  r=RoadSearch(RoadPolicy(),[0.,0.],[.075,.105,.085],.3)
  r.observe(road(),motion(),0.);return r
 def test_scan_is_a_small_leased_counterrotation_inside_known_road(self):
  r=self.ready();cmd=r.command(missing(),None,motion(),.1,active=True)
  self.assertIsNotNone(cmd);self.assertEqual(sum(cmd),0);self.assertGreater(cmd[1],0)
  self.assertLessEqual(max(abs(v) for v in cmd),30)
  self.assertTrue(r.last_action);self.assertEqual(r.status['lease_s'],.08)
  self.assertLessEqual(abs(r.status['goal_rad']),math.radians(10.))
 def test_missing_map_partial_coverage_invalid_flow_and_outside_body_cannot_turn(self):
  for kind in ('empty','far','flow','outside'):
   r=self.ready()
   if kind=='empty':r.patches=[];r.pose_hint=None
   if kind=='far':r.patches=[];r.pose_hint=None;r.observe(road(.5,.8),motion(),0.)
   if kind=='outside':r.patches=[];r.observe(road(y=.12),motion(),0.)
   cmd=r.command(missing(),None,motion(valid=kind!='flow'),.1,active=True)
   self.assertIsNone(cmd,kind);self.assertFalse(r.last_action)
 def test_no_motion_without_an_active_mission_or_during_yield(self):
  for active,allowed in ((False,True),(True,False)):
   r=self.ready();self.assertIsNone(r.command(missing(),None,motion(),.1,active=active,allowed=allowed));self.assertFalse(r.last_action)
 def test_pulse_requires_an_explicit_zero_frame_and_measured_rest(self):
  r=self.ready();r.command(missing(),None,motion(),.1,active=True)
  for now,m in ((.2,motion(yaw=.01)),(.3,motion())):
   r.observe(missing(),m,now);self.assertEqual(r.command(missing(),None,m,now,active=True),(0,0));self.assertFalse(r.last_action)
  self.assertIsNotNone(r.command(missing(),None,motion(),.4,active=True));self.assertTrue(r.last_action)
 def test_recovered_follow_command_ends_search_without_rearming_anything(self):
  r=self.ready();r.command(missing(),None,motion(),.1,active=True)
  self.assertEqual(r.command(road(),(35,40),motion(),.2,active=True),(35,40))
  self.assertEqual(r.status['phase'],'IDLE');self.assertFalse(r.last_action)
 def test_drift_or_total_search_duration_prevents_further_pulses(self):
  for kind in ('drift','time'):
   r=self.ready();r.command(missing(),None,motion(),.1,active=True)
   if kind=='drift':r.observe(missing(),motion(travel=.02),.2)
   self.assertIsNone(r.command(missing(),None,motion(),10. if kind=='time' else .3,active=True))
   self.assertTrue(r.finished);self.assertFalse(r.last_action)
 def test_near_pose_inference_is_explicit_and_does_not_create_a_path(self):
  r=self.ready();r.patches=[];r.pose_hint=None;r.observe(road(.2,.7),motion(),0.)
  old=r.patches[0]['path'].copy()
  self.assertTrue(r.swept_clear(.1));self.assertEqual(r.space_source,'LOCAL_POSE_ESTIMATE')
  np.testing.assert_array_equal(r.patches[0]['path'],old)
  r.now=.7;self.assertFalse(r.swept_clear(.1))
 def test_expired_driving_history_can_start_bounded_search_without_becoming_fresh(self):
  r=self.ready();r.patches=[];r.pose_hint=None;r.observe(road(.2,.7),motion(),0.)
  r.observe(missing(),motion(),.7)
  command=r.command(missing(),None,motion(),.7,active=True)
  self.assertIsNotNone(command);self.assertTrue(r.last_action)
  self.assertEqual(r.pose_hint['seen'],0.)
  r.observe(missing(),motion(),10.)
  self.assertIsNone(r.command(missing(),None,motion(),10.,active=True))
 def test_very_old_pose_cannot_be_revived_by_requesting_search(self):
  r=self.ready();r.patches=[];r.pose_hint=None;r.observe(road(.2,.7),motion(),0.)
  r.observe(missing(),motion(),2.)
  self.assertIsNone(r.command(missing(),None,motion(),2.,active=True))
  self.assertIsNone(r.requested_at)
 def test_search_shrinks_angle_to_available_clearance(self):
  from unittest.mock import patch
  r=self.ready();original=r.swept_clear
  with patch.object(r,'swept_clear',side_effect=lambda angle:abs(angle)<=.05 and original(angle)):
   command=r.command(missing(),None,motion(),.1,active=True)
  self.assertIsNotNone(command);self.assertLessEqual(abs(r.status['goal_rad']),.05)
 def test_curvature_uncertainty_rejects_blind_pose_inference(self):
  r=self.ready();r.patches=[];r.pose_hint=None;r.observe(road(.25,.7),motion(),0.)
  r.pose_hint['curvature']=4.
  self.assertFalse(r.swept_clear(.1))
 def test_contradictory_reinitialized_road_clears_old_corridor(self):
  r=self.ready();r.observe(road(y=.15),motion(),.1)
  self.assertEqual(len(r.patches),1);self.assertFalse(r.swept_clear(0.))
 def test_uncertainty_growth_never_refreshes_observation_timestamps(self):
  r=self.ready();seen=r.patches[0]['seen'];m=motion();m['sigma_m']=.06
  r.observe(missing(),m,.1);self.assertFalse(r.patches)
  self.assertIsNone(r.command(missing(),None,motion(),.2,active=True))
 def test_unmeasured_transform_revokes_old_map_even_when_next_flow_is_valid(self):
  r=self.ready();r.observe(missing(),motion(valid=False),.1)
  r.observe(missing(),motion(),.2)
  self.assertFalse(r.patches)
  self.assertIsNone(r.command(missing(),None,motion(),.3,active=True))
 def test_complete_measured_sweep_is_bounded(self):
  r=self.ready();r.command(missing(),None,motion(),.1,active=True)
  for now,increment in ((.2,r.limits.maximum_yaw_rad),(.3,-2*r.limits.maximum_yaw_rad),(.4,r.limits.maximum_yaw_rad)):
   r.observe(missing(),motion(yaw=increment),now)
   r.command(missing(),None,motion(),now,active=True)
  self.assertTrue(r.finished);self.assertEqual(r.status['phase'],'EXHAUSTED')
 def test_observation_storage_is_bounded_and_covers_only_real_samples(self):
  r=self.ready()
  for i in range(60):r.observe(road(),motion(),i*.1)
  self.assertLessEqual(len(r.patches),2)
  self.assertEqual(r.patches[0]['seen'],5.9)

if __name__=='__main__':unittest.main()
