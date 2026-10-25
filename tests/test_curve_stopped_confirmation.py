"""A running mission recovers from uncertainty without rearming a fault."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
from controller import RoadSupervisor

def candidate():return dict(state='CONFIRMING',confirmation_required_frames=4,road_evidence='CANDIDATE')
def absent():return dict(state='LOST',valid=False,road_evidence='ABSENT')

class StoppedConfirmationTests(unittest.TestCase):
 def test_real_stop_trace_resumes_after_four_stable_observations(self):
  rows=json.loads((ROOT/'tests/fixtures/curve_stop_confirmation.json').read_text())['rows']
  guard=RoadSupervisor();origin=rows[0]['frame_received_at']
  for row in rows:
   t=dict(row['target'],road_evidence='CANDIDATE');m=t['motion']
   rest=bool(m['valid'] and m['speed_m_s'] is not None and abs(m['speed_m_s'])<=.006 and abs(m['yaw_rad_s'])<=.025)
   command=(0,0) if t['observed'] else None
   self.assertIsNone(guard.update(t,command,row['frame_received_at']-origin,row['detection_ms'],280,stop_verified=rest),row['frame'])
   if row['frame']==4092:
    self.assertEqual(guard.phase,'CONFIRMING');self.assertEqual(t['pending_frames'],3)
   if row['frame']==4093:self.assertTrue(t['reacquired'])
  self.assertEqual(guard.phase,'IDLE')

 def test_candidate_attempt_count_alone_never_disarms(self):
  g=RoadSupervisor()
  for i in range(50):
   self.assertIsNone(g.update(candidate(),None,i*.1,150,280,stop_verified=True))
  self.assertEqual(g.frames,50)
  self.assertIsNone(g.update(dict(state='OBSERVED'),(20,20),5.,100,280))
  self.assertEqual(g.phase,'IDLE')

 def test_zero_request_alone_does_not_claim_verified_physical_rest(self):
  g=RoadSupervisor()
  for now in (0.,.2,.4,.6,1.):
   self.assertIsNone(g.update(candidate(),None,now,150,280))
   self.assertEqual(g.phase,'SETTLING')
  self.assertIsNone(g.update(candidate(),None,1.1,150,280,stop_verified=True))
  self.assertEqual(g.phase,'CONFIRMING')

 def test_continuous_negative_evidence_after_history_expiry_ends_mission(self):
  g=RoadSupervisor()
  self.assertIsNone(g.update(absent(),None,0.,100,280))
  self.assertIsNone(g.update(absent(),None,.2,100,280))
  self.assertEqual(g.update(absent(),None,.36,100,280),'ROAD NOT FOUND')
  self.assertEqual(g.update(dict(state='OBSERVED'),(20,20),.5,100,280),'ROAD NOT FOUND')
  g.reset()
  self.assertIsNone(g.update(dict(state='OBSERVED'),(20,20),.6,100,280))

 def test_ambiguous_edges_or_low_quality_candidate_do_not_prove_road_absence(self):
  g=RoadSupervisor()
  for i in range(30):
   r=dict(state='LOST',valid=False,road_evidence='AMBIGUOUS' if i%2 else 'CANDIDATE')
   self.assertIsNone(g.update(r,None,i*.2,100,280))
  self.assertEqual(g.phase,'REACQUIRING')

 def test_old_support_can_remain_usable_after_current_edges_disappear(self):
  g=RoadSupervisor()
  for i in range(10):self.assertIsNone(g.update(dict(state='PREDICTED',valid=True,road_evidence='ABSENT'),None,i*.1,100,280))
  self.assertIsNone(g.missing_since)
  self.assertIsNone(g.update(absent(),None,1.,100,280))
  self.assertEqual(g.update(absent(),None,1.36,100,280),'ROAD NOT FOUND')

 def test_slow_frames_only_authorize_wait_and_cannot_prove_road_absence(self):
  g=RoadSupervisor()
  for now in (0.,.3,.6,1.):self.assertIsNone(g.update(absent(),None,now,300,280))
  self.assertEqual(g.phase,'VISION_WAIT')
  self.assertIsNone(g.update(absent(),None,1.1,150,280))
  self.assertEqual(g.update(absent(),None,1.46,150,280),'ROAD NOT FOUND')

 def test_confirmed_road_can_resume_soft_loss_without_rearming_hardware(self):
  g=RoadSupervisor()
  g.update(absent(),None,0.,100,280)
  self.assertEqual(g.update(absent(),None,.36,100,280),'ROAD NOT FOUND')
  self.assertEqual(g.update(dict(state='OBSERVED'),(20,20),.5,100,280),'ROAD NOT FOUND')
  self.assertIsNone(g.update(dict(state='OBSERVED'),(20,20),.6,100,280,road_ready=True))
  self.assertEqual(g.phase,'IDLE')
 def test_only_approved_search_can_operate_during_soft_loss(self):
  g=RoadSupervisor();g.update(absent(),None,0.,100,280)
  self.assertEqual(g.update(absent(),None,.36,100,280),'ROAD NOT FOUND')
  self.assertIsNone(g.update(absent(),(-10,10),.5,100,280,search_pending=True))
  self.assertEqual(g.failed,'ROAD NOT FOUND')
  self.assertEqual(g.update(absent(),(30,30),.6,100,280),'ROAD NOT FOUND')
 def test_confirmed_road_never_clears_a_hard_fault(self):
  g=RoadSupervisor()
  g.update(dict(state='LOST'),None,0.,100,280)
  self.assertEqual(g.update(dict(state='OBSERVED'),(20,20),.6,100,280,road_ready=True),'INVALID ROAD EVIDENCE')
 def test_invalid_evidence_contract_fails_closed(self):
  g=RoadSupervisor()
  self.assertEqual(g.update(dict(state='LOST'),None,0.,100,280),'INVALID ROAD EVIDENCE')

if __name__=='__main__':unittest.main()
