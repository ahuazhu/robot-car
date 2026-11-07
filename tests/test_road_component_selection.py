"""Separate visible road pieces cannot veto the current near lane."""
from tests.support import ROOT
import json
import unittest
import cv2
import numpy as np
from robot.core.road_policy import RoadPolicy
from robot.navigation.tracker import TargetTracker
from robot.navigation.steering_control import SteeringControl
from robot.perception.boundaries import CurvePixels
from robot.perception.surface import RoadSurfaceEvidence


class RoadComponentSelectionTests(unittest.TestCase):
    def scene(self, mirror=False):
        root = ROOT / 'tests/fixtures/right_road_stall_20261107'
        image = cv2.imread(str(root / 'scene.png'))
        cal = json.loads((root / 'calibration.json').read_text())
        if mirror:
            image = image[:, ::-1].copy()
            flip = np.array([[-1., 0., 319/320], [0., 1., 0.], [0., 0., 1.]])
            cal['H_normalized'] = (np.diag([1., -1., 1.]) @ np.array(cal['H_normalized']) @ flip).tolist()
        return image, cal

    def test_recorded_scene_recovers_toward_near_road_with_mirrored_commands(self):
        for mirror in (False, True):
            with self.subTest(mirror=mirror):
                image, cal = self.scene(mirror)
                result, details = TargetTracker(cal).update(image, 0., require_current=True)
                side = 'right' if mirror else 'left'
                proof = result['road_component_selection']
                self.assertTrue(result['valid'])
                self.assertTrue(result['observed'])
                self.assertEqual(result['mode'], 'ONE ' + side.upper())
                self.assertEqual(proof['selected_side'], side)
                self.assertNotEqual(*proof['components'])
                self.assertGreater(proof['green_separator_m'], 0.075)
                self.assertIsNone(details['fits']['left' if mirror else 'right'])
                control = SteeringControl(RoadPolicy(), [-.055, 0.], [.075, .105, .085], .3,
                                          lever=.15, gain=.85, cruise=120, calibration=cal)
                command = control.command(result)
                self.assertIsNotNone(command)
                self.assertGreater(min(command), 0)
                self.assertGreater(command[1 if mirror else 0], command[0 if mirror else 1])
                self.assertIsNone(control.status['envelope']['first_unsafe_m'])

    def test_no_definite_green_cannot_resolve_conflicting_boundaries(self):
        image, cal = self.scene()
        pixels = CurvePixels(cal)
        _, details = pixels.measure(image)
        fits = {side: max(options, key=lambda edge: edge['score'])
                for side, options in details['candidates'].items()}
        details['surface'] = RoadSurfaceEvidence(
            pixels, np.zeros_like(pixels.visible), details['saturation'],
            road_value=details['contrast_value'], reference_origin=[-.055, 0.],
        )
        result, _ = pixels.measure_fits(fits, details)
        self.assertIsNone(result)
        self.assertEqual(details['measurement_reason'], 'EDGES DISAGREE')

    def test_equally_near_components_remain_ambiguous_despite_green_separator(self):
        _, cal = self.scene()
        pixels = CurvePixels(cal)
        pixels.visible[:] = True
        xx, yy = np.meshgrid(pixels.xs, pixels.ys, indexing='ij')
        road = (xx >= .2) & (xx <= .7) & ((abs(yy-.3) <= .15) | (abs(yy+.3) <= .15))
        surface = RoadSurfaceEvidence(
            pixels, ~road, np.where(road, 0, 220).astype(np.uint8),
            road_value=np.full(road.shape, 180, np.uint8),
        )
        x = np.linspace(.22, .60, 40)
        a, b = np.c_[x, np.full(40, .15)], np.c_[x, np.full(40, -.15)]
        self.assertIsNone(surface.nearer_separated_component(
            [(a, np.tile([0., 1.], (40, 1))), (b, np.tile([0., -1.], (40, 1)))], .005,
        ))


if __name__ == '__main__':
    unittest.main()
