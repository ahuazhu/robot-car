"""Regression for the real right bend where the lower road edges leave the frame."""
import importlib.util
from pathlib import Path
import unittest
import cv2
import numpy as np
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('lane',root/'maixcam2/lane_follow/main.py')
lane=importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)
class BendTests(unittest.TestCase):
    def test_real_bend_and_mirror(self):
        source=cv2.imread(str(root/'tests/fixtures/right_bend.jpg'))
        for frame,right in [(source,True),(cv2.flip(source,1),False)]:
            result=lane.detect_road(frame)
            self.assertTrue(result['valid'])
            self.assertGreaterEqual(len(result['points']),4)
            # Do not bridge the distant unrelated patch into the near band.
            self.assertTrue(all(a[0]-b[0]<=12 for a,b in zip(result['points'],result['points'][1:])))
            left_pwm,right_pwm,_=lane.lane_command(result)
            self.assertEqual(left_pwm>right_pwm,right)
            self.assertTrue(0<=left_pwm<=120 and 0<=right_pwm<=120)
    def test_narrow_visible_band(self):
        frame=cv2.imread(str(root/'tests/fixtures/right_bend_narrow_band.jpg'))
        result=lane.detect_road(frame)
        self.assertTrue(result['valid'])
        self.assertGreater(lane.lane_command(result)[0],lane.lane_command(result)[1])
    def test_sharp_corner_steers_without_reversing(self):
        frame=cv2.imread(str(root/'tests/fixtures/sharp_bend_stop.jpg'))
        for picture,right in [(frame,True),(cv2.flip(frame,1),False)]:
            result=lane.detect_road(picture)
            self.assertTrue(result['valid'])
            left_pwm,right_pwm,_=lane.lane_command(result)
            self.assertTrue(0<=left_pwm<=110 and 0<=right_pwm<=110)
            self.assertEqual(left_pwm>right_pwm,right)
    def test_absent_boundaries_stop(self):
        for color in [(30,170,30),(100,100,100),(0,0,0)]:
            self.assertIsNone(lane.lane_command(lane.detect_road(np.full((240,320,3),color,np.uint8))))
if __name__=='__main__': unittest.main()
