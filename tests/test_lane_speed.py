import importlib.util
from pathlib import Path
import unittest
import cv2
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('lane',root/'maixcam2/lane_follow/main.py')
lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
class SpeedTests(unittest.TestCase):
    def test_recorded_sharp_bend_slows_both_motion_and_rotation(self):
        frame=cv2.imread(str(root/'tests/fixtures/sharp_bend_stop.jpg'))
        for image in (frame,cv2.flip(frame,1)):
            command=lane.lane_command(lane.detect_road(image))
            self.assertIsNotNone(command)
            left,right,_=command
            self.assertLessEqual((left+right)/2,60)
            self.assertLessEqual(abs(left-right),50)
            self.assertGreaterEqual(min(left,right),25)
    def test_stop_is_immediate_and_cannot_resume_old_steering(self):
        ramp=lane.SteeringRamp()
        for i in range(20):ramp.apply((75,25,1),i*.05)
        self.assertEqual(ramp.apply((0,0,0),1),(0,0))
        self.assertEqual(ramp.apply(None,1.1),(0,0))
        left,right=ramp.apply((80,80,0),2)
        self.assertEqual(left,right)
        self.assertLess(left,20)
    def test_direction_change_never_keeps_old_direction(self):
        ramp=lane.SteeringRamp()
        for i in range(20):ramp.apply((75,25,1),i*.05)
        left,right=ramp.apply((25,75,-1),1)
        self.assertLess(left,right)
        self.assertLessEqual(abs(left-right),7)
    def test_delayed_frame_cannot_cause_large_steering_jump(self):
        ramp=lane.SteeringRamp();ramp.apply((80,80,0),0)
        left,right=ramp.apply((75,25,1),10)
        self.assertLessEqual(abs(left-right),12)
if __name__=='__main__':unittest.main()
