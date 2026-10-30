"""Race IV.1(3) constrains a stop interval, never one exact setpoint."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_crosswalk import CrosswalkTask,configured,motion,observation
from crosswalk import legal_stop_interval

class CrosswalkStopWindowTests(unittest.TestCase):
 def task(self):
  return CrosswalkTask(configured(body_front_m=.02,body_rear_m=.16,body_half_width_m=.085,stop_clearance_m=.08))
 def update(self,t,now,distance,sigma=.01,speed=0.):
  obs=observation(now,distance+.02);obs['sigma_m']=sigma
  state=t.update(obs,motion(speed),now,True,True,True,True)
  return state,t.limit_command((60,60),state,motion(speed))
 def test_already_stationary_at_any_legal_distance_does_not_creep_to_buffer(self):
  for distance in (.05,.10,.15,.18):
   with self.subTest(distance=distance):
    t=self.task()
    for now in (0.,.1,.2,.3,.4,.5,.6):s,cmd=self.update(t,now,distance)
    self.assertEqual(s['phase'],'HOLD');self.assertEqual(cmd,(0,0))
    self.assertEqual(s['legal_stop_window_m'],[0.,.2])
 def test_coarse_position_is_sufficient_if_entire_interval_is_legal(self):
  t=self.task()
  for now in (0.,.1,.2,.3,.4,.5,.6):s,cmd=self.update(t,now,.10,sigma=.05)
  self.assertEqual(s['phase'],'HOLD');self.assertTrue(s['stop_verified']);self.assertEqual(cmd,(0,0))
 def test_legal_range_changes_do_not_restart_the_dwell(self):
  t=self.task()
  for now in (0.,.1,.2,.3,.4,.5,.6):s,cmd=self.update(t,now,.15)
  start=t.hold_since
  for i in range(1,34):
   now=.6+i*.1;distance=.15 if i%2==0 else .095
   s,cmd=self.update(t,now,distance)
   if s['phase']!='PASSING':self.assertEqual(cmd,(0,0))
  self.assertEqual(t.hold_since,start);self.assertEqual(s['phase'],'PASSING')
  self.assertTrue(s['yield_verified']);self.assertGreaterEqual(s['hold_elapsed_s'],3.3)
 def test_uncertainty_crossing_either_rule_boundary_is_not_legal(self):
  for interval in ((-.01,.05),(.17,.21),(0.,.10),(.1,.20)):
   self.assertFalse(legal_stop_interval(interval))
  self.assertTrue(legal_stop_interval((.05,.15)))
 def test_conclusively_outside_window_restarts_approach_but_legal_stop_does_not(self):
  t=self.task()
  for now in (0.,.1,.2,.3):s,cmd=self.update(t,now,.30)
  self.assertEqual(s['phase'],'APPROACH');self.assertNotEqual(cmd,(0,0))
  self.update(t,.4,.24)
  for now in (.5,.6,.7,.8,.9):s,cmd=self.update(t,now,.18)
  self.assertEqual(s['phase'],'HOLD');self.assertEqual(cmd,(0,0))

if __name__=='__main__':unittest.main()
