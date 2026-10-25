import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/curve_follow'))
from controller import assess_vision_command

class StartupReadinessTests(unittest.TestCase):
 def test_interleaved_slow_frames_do_not_erase_observed_road_stability(self):
  result=dict(observed=True,confidence=.8);count=0
  for n,latency in enumerate((225,310,240,301,222),1):
   command,count=assess_vision_command(result,(40,40),latency,280,count,.65)
   self.assertEqual(count,n)
   self.assertEqual(command,None if latency>280 else (40,40))
  self.assertGreaterEqual(count,5)
  self.assertIsNotNone(command)
 def test_slow_current_frame_never_allows_start_or_drive(self):
  command,count=assess_vision_command(dict(observed=True,confidence=.8),(40,40),281,280,9,.65)
  self.assertGreaterEqual(count,5)
  self.assertIsNone(command)
 def test_predicted_weak_or_missing_geometry_resets_stability(self):
  for result,proposed in ((dict(observed=False,confidence=.9),(40,40)),(dict(observed=True,confidence=.5),(40,40)),(dict(observed=True,confidence=.9),None)):
   _,count=assess_vision_command(result,proposed,225,280,9,.65)
   self.assertEqual(count,0)
 def test_five_observations_are_still_required(self):
  count=0
  for _ in range(4):_,count=assess_vision_command(dict(observed=True,confidence=.8),(40,40),225,280,count,.65)
  self.assertLess(count,5)
if __name__=='__main__':unittest.main()
