"""Coordinate-sign, missing-road, and calibration guards for perception test."""
import sys
from pathlib import Path
import unittest
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/duckietown_test'))
from vision import LaneVision, make_calibration
from duckietown_core import NormalizedImagePoint

class DuckietownTests(unittest.TestCase):
    def test_measured_corners_project_to_metres(self):
        pts=[[40,220],[280,220],[220,150],[100,150]]
        cfg=make_calibration(pts,.3,.6)
        v=LaneVision(cfg)
        for pixel,expected in zip(pts,[(.3,.15),(.3,-.15),(.6,-.15),(.6,.15)]):
            p=v.projector.vector2ground(NormalizedImagePoint(pixel[0]/320,pixel[1]/240))
            np.testing.assert_allclose([p.x,p.y],expected,atol=1e-6)
        self.assertFalse(cfg['drive_validated'])
        self.assertFalse(cfg['lens_rectified'])
    def test_wrong_order_and_degenerate_points_rejected(self):
        for pts in ([[40,150],[280,150],[220,220],[100,220]],[[40,220]]*4):
            with self.assertRaises(ValueError):make_calibration(pts,.3,.6)
    def test_empty_images_do_not_create_boundaries(self):
        for color in [(0,0,0),(255,255,255),(0,180,0)]:
            im=np.full((240,320,3),color,dtype=np.uint8)
            r=LaneVision().detect(im)
            self.assertEqual(r['lines'],[])
            self.assertFalse(r['motor_enabled'])
    def test_stop_bar_alone_is_not_a_side_boundary(self):
        im=np.full((240,320,3),80,np.uint8)
        cv2.rectangle(im,(0,175),(319,190),(255,255,255),-1)
        self.assertEqual(LaneVision().detect(im)['lines'],[])
    def test_left_and_right_labels_on_synthetic_straight(self):
        im=np.full((240,320,3),(0,160,0),dtype=np.uint8)
        cv2.rectangle(im,(75,120),(245,239),(80,80,80),-1)
        for x in (75,240):cv2.rectangle(im,(x,120),(x+5,239),(240,240,240),-1)
        r=LaneVision().detect(im)
        self.assertGreater(r['counts']['left'],0);self.assertGreater(r['counts']['right'],0)
        for l in r['lines']:
            self.assertEqual(l['side'],'left' if l['pixels'][0]<160 else 'right')
    def test_resolution_mismatch_rejected(self):
        with self.assertRaises(ValueError):LaneVision().detect(np.zeros((480,640,3),np.uint8))

if __name__=='__main__':unittest.main()
