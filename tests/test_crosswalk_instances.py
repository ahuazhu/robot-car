"""A completed but unlocalized crossing cannot suppress fresh forward hazards."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_crosswalk import CrosswalkTask,configured,motion,observation

class CrosswalkInstanceTests(unittest.TestCase):
 def served(self):
  t=CrosswalkTask(configured(body_front_m=.02,body_rear_m=.16,body_half_width_m=.085))
  for now in (0.,.1,.2):t.update(observation(now,near=.16),motion(),now,True,True,True,True)
  t.phase='PASSING';t.yield_completed=t.yield_verified=True
  return t
 def stale(self):
  t=self.served()
  # Recorded failure: old mean behind car while uncertainty grows to metres.
  t.track.update(near_m=-2.173,far_m=-2.073,sigma_m=6.062)
  t.last_update=10.
  return t
 def test_logged_49_to_31cm_approach_is_confirmed_as_a_fresh_hazard(self):
  t=self.stale()
  for i,near in enumerate((.48799,.47965,.44295)):
   now=10.1+i*.15;s=t.update(observation(now,near),motion(valid=False),now,True)
   self.assertEqual(s['phase'],'PASSING' if i<2 else 'APPROACH')
  self.assertEqual(t.identity,2);self.assertGreaterEqual(t.hits,3)
  self.assertEqual(s['retired_event']['event_id'],1)
  self.assertFalse(s['retired_event']['passage_proven'])
  self.assertFalse(s['yield_completed']);self.assertEqual(s['observation_association'],'REOBSERVED_CROSSWALK')
 def test_new_candidate_limits_speed_before_confirmation(self):
  t=self.stale();s=t.update(observation(10.1,.48),motion(.25),10.1,True)
  self.assertEqual(s['pending_hits'],1)
  cmd=t.limit_command((100,120),s,motion(.25))
  self.assertLess(cmd[1],120)
  self.assertLess(abs(cmd[0]*120-cmd[1]*100),220)
 def test_one_flash_cannot_replace_a_served_event(self):
  t=self.stale();s=t.update(observation(10.1,.48),motion(),10.1,True)
  self.assertEqual(t.identity,1)
  s=t.update(None,motion(),10.7,True)
  self.assertIsNone(s['pending_observation']);self.assertEqual(t.identity,1)
  self.assertEqual(t.limit_command((100,120),s,motion()),(100,120))
 def test_inconsistent_candidate_cannot_accumulate_confirmation(self):
  t=self.stale()
  for now,near in ((10.1,.48),(10.2,.65),(10.3,.48),(10.4,.65)):
   s=t.update(observation(now,near),motion(),now,True)
  self.assertEqual(t.identity,1);self.assertEqual(s['pending_hits'],1)
 def test_visible_same_crossing_and_short_flow_gap_do_not_repeat_yield(self):
  t=self.served()
  t.update(None,motion(valid=False),.3,True)
  for now,near in ((.4,.15),(.5,.14),(.6,.13)):
   s=t.update(observation(now,near),motion(.1,.01),now,True)
   self.assertEqual(t.identity,1);self.assertEqual(s['phase'],'PASSING')
   self.assertIsNone(s['pending_observation'])
 def test_trusted_old_instance_and_already_under_body_pattern_do_not_rearm(self):
  t=self.served()
  s=t.update(observation(.21,.48),motion(),.21,True)
  self.assertIsNone(s['pending_observation']);self.assertEqual(t.identity,1)
  t=self.stale()
  for i in range(4):s=t.update(observation(10.1+i*.1,.015),motion(),10.1+i*.1,True)
  self.assertIsNone(s['pending_observation']);self.assertEqual(t.identity,1)
 def test_each_new_confirmed_event_requires_its_own_dwell(self):
  t=self.stale()
  for now in (10.1,10.2,10.3):s=t.update(observation(now,.16),motion(),now,True,True,True,True)
  self.assertEqual(t.identity,2);self.assertFalse(s['yield_completed'])
  for i in range(1,40):
   now=10.3+i*.1;s=t.update(observation(now,.16),motion(),now,True,True,True,True)
  self.assertEqual(s['phase'],'PASSING');self.assertTrue(s['yield_verified'])
  self.assertGreaterEqual(s['hold_elapsed_s'],3.3)

if __name__=='__main__':unittest.main()
