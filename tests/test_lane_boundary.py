import importlib.util
from pathlib import Path
import unittest
import cv2
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("lane",root/"maixcam2/lane_follow/main.py")
lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
class BoundaryTests(unittest.TestCase):
    def test_actual_bend_uses_near_edge_then_stops_outside(self):
        for mirrored in (False,True):
            tracker=lane.RoadTracker()
            for n in (2,10,11,12,13):
                im=cv2.imread(str(root/f"tests/fixtures/near_boundary/{n}.jpg"))
                if mirrored:im=cv2.flip(im,1)
                result=tracker.update(lane.detect_road(im),n*.1)
                cmd=lane.lane_command(result)
                if n in (11,12):
                    self.assertIsNotNone(cmd)
                    self.assertTrue(result["mode"].startswith("EDGE"))
                    self.assertEqual(cmd[0]>cmd[1],mirrored)
                if n==13:self.assertIsNone(cmd)
if __name__=="__main__":unittest.main()
