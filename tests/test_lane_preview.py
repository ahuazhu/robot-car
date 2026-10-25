"""Offline checks: python test_lane_preview.py /path/to/lane_preview/main.py"""
import importlib.util
import sys
import unittest
import cv2
import numpy as np

spec = importlib.util.spec_from_file_location("lane_preview", sys.argv.pop(1))
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)


def road(shift=0, curve=0):
    img = np.full((240, 320, 3), (30, 170, 30), dtype=np.uint8)
    for y in range(240):
        center = int(160 + shift + curve * (1 - y / 240))
        half = int(25 + y * .25)
        l, r = max(0, center-half), min(319, center+half)
        img[y, l:r+1] = (110, 110, 110)
        img[y, l:l+3] = (240, 240, 240)
        img[y, r-2:r+1] = (240, 240, 240)
    return img


class LaneTests(unittest.TestCase):
    def test_straight_center(self):
        out = lane.detect_road(road())
        self.assertTrue(out['valid'])
        self.assertLess(abs(out['offset']), 2)

    def test_right_and_left_offset(self):
        self.assertGreater(lane.detect_road(road(30))['offset'], 25)
        self.assertLess(lane.detect_road(road(-30))['offset'], -25)

    def test_curve(self):
        out = lane.detect_road(road(curve=70))
        self.assertTrue(out['valid'])
        self.assertGreater(out['points'][-1][3], out['points'][0][3] + 20)

    def test_white_crossing_does_not_shift_center(self):
        original = road()
        marked = original.copy()
        road_pixels = (marked[:, :, 0] == marked[:, :, 1])
        marked[175:220][road_pixels[175:220]] = 255
        a, b = lane.detect_road(original), lane.detect_road(marked)
        self.assertTrue(b['valid'])
        self.assertEqual(a['points'], b['points'])

    def test_no_road_and_no_green(self):
        for color in [(30, 170, 30), (100, 100, 100), (0, 0, 0)]:
            out = lane.detect_road(np.full((240, 320, 3), color, np.uint8))
            self.assertFalse(out['valid'])
            self.assertIsNone(out['offset'])

    def test_clipped_road_is_uncertain(self):
        out = lane.detect_road(road(shift=-145))
        self.assertFalse(out['valid'])

    def test_small_glare_patch_not_selected(self):
        img = road()
        img[145:230, 10:20] = 240
        out = lane.detect_road(img)
        self.assertTrue(out['valid'])
        self.assertLess(abs(out['offset']), 2)


if __name__ == '__main__':
    unittest.main()
