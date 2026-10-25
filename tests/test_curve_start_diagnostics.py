import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/curve_follow'))
from controller import startup_blockers

class StartDiagnosticsTests(unittest.TestCase):
 def check(self,**changes):
  args=dict(result=dict(observed=True,confidence=.8,state='OBSERVED'),command=(40,40),consecutive=5,detection_ms=200,budget_ms=280,link_ok=True,preview_only=False,minimum_confidence=.65)
  args.update(changes);return startup_blockers(**args)
 def test_healthy_start_has_no_blockers(self):self.assertEqual(self.check(),[])
 def test_slow_processing_is_reported_separately_from_road_geometry(self):
  self.assertEqual(self.check(command=None,detection_ms=300),['VISION_TOO_SLOW'])
 def test_stability_and_serial_are_independent_blockers(self):
  self.assertEqual(self.check(consecutive=2,link_ok=False),['UNO_DISCONNECTED','ROAD_STABILIZING'])
 def test_unobserved_path_confirmation_is_not_reported_as_ready(self):
  reasons=self.check(result=dict(observed=False,confidence=.4,state='CONFIRMING'),command=None,consecutive=0)
  self.assertIn('PATH_CONFIRMING',reasons);self.assertIn('ROAD_UNOBSERVED',reasons)

if __name__=='__main__':unittest.main()
