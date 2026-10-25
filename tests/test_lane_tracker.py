import importlib.util
from pathlib import Path
import unittest
import numpy as np
import cv2
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('lane',root/'maixcam2/lane_follow/main.py')
lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
def road(shift=0):
    img=np.full((240,320,3),(30,170,30),np.uint8)
    for y in range(240):
        half=int(25+y*.45);center=160+shift
        left,right=max(0,center-half),min(319,center+half)
        if left<=right:img[y,left:right+1]=(110,110,110)
    return img
class TrackerTests(unittest.TestCase):
    def test_visible_both_edges_refresh_single_edge_timeout(self):
        tracker=lane.RoadTracker()
        pts=[(y,160-(y*.5+20),160+(y*.5+20),160) for y in range(208,155,-4)]
        partial=[(y,'left',160-(y*.5+20)) for y in (224,220,216,212)]
        for i in range(21):
            raw=dict(valid=True,points=pts,rows=[(p[0],p) for p in pts],partial=partial,mode='BOTH',offset=0)
            out=tracker.update(raw,i*.5)
            self.assertTrue(out['valid'])
            self.assertIsNotNone(lane.lane_command(out))

    def test_three_clipped_rows_do_not_discard_valid_full_road(self):
        frame=cv2.imread(str(root/'tests/fixtures/three_near_edges.png'))
        for picture in (frame,cv2.flip(frame,1)):
            tracker=lane.RoadTracker()
            for index in range(12):
                detected=lane.detect_road(picture)
                self.assertTrue(detected['valid'])
                tracked=tracker.update(detected,index*.1)
                self.assertTrue(tracked['valid'])
                self.assertIsNotNone(lane.lane_command(tracked))

    def test_short_dropout_pauses_then_recovers(self):
        guard=lane.MotionGuard()
        self.assertEqual(guard.resolve((80,90,0),True,0,5),((80,90,0),False))
        self.assertEqual(guard.resolve(None,True,1,0),((0,0,0.0),True))
        self.assertEqual(guard.resolve((80,90,0),True,1.1,1),((80,90,0),False))
        self.assertEqual(guard.resolve((80,90,0),True,1.2,2),((80,90,0),False))
    def test_fresh_valid_frame_wins_at_dropout_deadline(self):
        guard=lane.MotionGuard();guard.resolve(None,True,0,0)
        self.assertEqual(guard.resolve((86,94,0),True,.36,1),((86,94,0),False))
        self.assertIsNone(guard.lost_at)
    def test_extended_dropout_stops_and_inactive_resets(self):
        guard=lane.MotionGuard();guard.resolve(None,True,0,0)
        self.assertEqual(guard.resolve(None,True,.36,0),(None,False))
        guard.resolve(None,False,1,0)
        self.assertEqual(guard.resolve((80,80,0),True,2,5),((80,80,0),False))
    def test_fallback_requires_calibration_and_expires(self):
        tracker=lane.RoadTracker();partial=lane.detect_road(road(-110))
        self.assertFalse(partial['valid'])
        self.assertFalse(tracker.update(partial,0)['valid'])
        self.assertTrue(tracker.update(lane.detect_road(road()),1)['valid'])
        self.assertIsNotNone(tracker.width_fit)
        tracked=tracker.update(partial,2)
        self.assertTrue(tracked['valid']);self.assertEqual(tracked['mode'],'EDGE RIGHT')
        command=lane.lane_command(tracked);self.assertIsNotNone(command)
        self.assertLess(command[0],command[1])
        self.assertFalse(tracker.update(partial,4.1)['valid'])
    def test_missing_both_edges_does_not_continue(self):
        tracker=lane.RoadTracker();tracker.update(lane.detect_road(road()),0)
        for color in [(30,170,30),(110,110,110),(0,0,0)]:
            result=tracker.update(lane.detect_road(np.full((240,320,3),color,np.uint8)),0.1)
            self.assertFalse(result['valid']);self.assertIsNone(lane.lane_command(result))
    def test_left_edge_recovery(self):
        tracker=lane.RoadTracker();tracker.update(lane.detect_road(road()),0)
        result=tracker.update(lane.detect_road(road(110)),1)
        self.assertTrue(result['valid']);self.assertEqual(result['mode'],'EDGE LEFT')
        command=lane.lane_command(result);self.assertGreater(command[0],command[1])
if __name__=='__main__':unittest.main()
