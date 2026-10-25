"""Task exceptions never masquerade as legally verified yields."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_crosswalk import CrosswalkTask,configured,motion,observation
class CrosswalkRecoveryTests(unittest.TestCase):
 def test_verified_dwell_completes_without_a_lane_target_but_authorizes_no_blind_drive(self):
  t=CrosswalkTask(configured())
  for now in np.arange(0,5.1,.1):s=t.update(observation(float(now)),motion(),float(now),True,True,True,False)
  self.assertEqual(s['phase'],'PASSING');self.assertTrue(s['yield_verified']);self.assertTrue(s['yield_completed'])
  self.assertIsNone(s['failure']);self.assertIsNone(t.limit_command(None,s,motion()))
 def test_missed_window_is_recorded_but_only_real_dwell_can_complete_recovery(self):
  t=CrosswalkTask(configured())
  for now in np.arange(0,5.2,.1):s=t.update(observation(float(now),near=.125),motion(),float(now),True,True,True,True)
  self.assertEqual(s['phase'],'PASSING');self.assertTrue(s['yield_completed']);self.assertFalse(s['yield_verified'])
  self.assertIn('CROSSWALK STOP WINDOW MISSED',s['issues']);self.assertIsNone(s['failure'])
  self.assertGreaterEqual(s['hold_elapsed_s'],t.policy.hold_s)
 def test_unknown_stop_without_motion_ack_or_lamp_cannot_earn_recovery_dwell(self):
  for missing in ('motion','ack','lamp'):
   t=CrosswalkTask(configured())
   for now in np.arange(0,5.2,.1):
    s=t.update(observation(float(now),near=.125),motion(valid=missing!='motion'),float(now),True,missing!='ack',missing!='lamp',True)
   self.assertFalse(s['yield_completed'],missing);self.assertFalse(s['yield_verified'])
   self.assertEqual(t.limit_command((50,50),s,motion()),(0,0))
 def test_speed_sensor_can_recover_without_resetting_task_or_start(self):
  t=CrosswalkTask(configured())
  for now in (0.,.1,.2):s=t.update(observation(now,near=.6),motion(),now,True,True,True,True)
  for now in (.3,.8,1.4):
   s=t.update(observation(now,near=.6),motion(valid=False),now,True,True,True,True)
   self.assertEqual(t.limit_command((50,60),s,motion(valid=False)),(0,0))
  self.assertIn('CROSSWALK SPEED UNVERIFIED',t.issues);self.assertIsNone(t.failure)
  s=t.update(observation(1.5,near=.6),motion(.02),1.5,True,True,True,True)
  self.assertNotEqual(t.limit_command((50,60),s,motion(.02)),(0,0));self.assertIsNone(t.recovery_reason)
if __name__=='__main__':unittest.main()
