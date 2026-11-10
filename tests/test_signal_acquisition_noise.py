"""Exercise acquisition noise through the real RGB565 embedded detector."""
import unittest

import numpy as np

from esp32s3.video_preview.replay import NativeDetector


class SignalAcquisitionNoiseTests(unittest.TestCase):
    BOX = [100, 40, 80, 64]

    @classmethod
    def setUpClass(cls):
        cls.detector = NativeDetector(full_frame=True)

    @classmethod
    def tearDownClass(cls):
        cls.detector.close()

    def setUp(self):
        self.reset_detector()

    def reset_detector(self):
        self.d = self.detector
        self.d.cancel()
        self.assertTrue(self.d.manual(self.BOX))
        self.d.cycle()
        self.d.arm()
        self.now = 0

    def scene(self, lamp, background=200):
        image = np.full((240, 320, 3), background, np.uint8)
        x, y, width, height = self.BOX
        image[y:y + height, x:x + width] = lamp
        return image

    def frame(self, image, step_ms=100):
        state = self.d.process(image, self.now)
        self.now += step_ms
        return state

    def establish_on(self, values=(190, 215, 240, 240), backgrounds=None):
        self.frame(self.scene(170))
        backgrounds = backgrounds or (200,) * len(values)
        for level, background in zip(values, backgrounds):
            state = self.frame(self.scene(level, background))
            self.assertFalse(state['confirmed'])
        self.assertTrue(state['locked'], self.d.diagnostics())
        return state

    def off_frames(self, count=4, image=None):
        image = self.scene(170) if image is None else image
        return [self.frame(image, 210) for _ in range(count)]

    def test_on_rise_does_not_permanently_prevent_normal_off(self):
        self.establish_on()
        for _ in range(12):
            self.assertFalse(self.frame(self.scene(240))['confirmed'])
        self.assertTrue(self.off_frames()[-1]['confirmed'], self.d.diagnostics())

    def test_on_rise_while_exposure_settles_still_accepts_real_off(self):
        self.establish_on(backgrounds=(180, 190, 200, 200))
        self.assertTrue(self.off_frames()[-1]['confirmed'], self.d.diagnostics())

    def test_sustained_on_never_becomes_an_off_event(self):
        self.establish_on()
        states = [self.frame(self.scene(240)) for _ in range(40)]
        self.assertFalse(any(s['confirmed'] for s in states), self.d.diagnostics())

    def test_global_exposure_darkening_does_not_become_an_off_event(self):
        for transform in ('offset', 'gain'):
            with self.subTest(transform=transform):
                self.reset_detector()
                self.establish_on()
                on = self.scene(240)
                darker = (np.maximum(on.astype(int) - 30, 0).astype(np.uint8)
                          if transform == 'offset' else (on * .75).astype(np.uint8))
                states = self.off_frames(8, darker)
                self.assertFalse(any(s['confirmed'] for s in states), self.d.diagnostics())

    def test_structural_background_change_does_not_become_an_off_event(self):
        self.establish_on()
        changed = self.scene(170)
        changed[:] = 120
        for x in range(0, 320, 32):
            changed[:, x:x + 16] = 230
        x, y, width, height = self.BOX
        changed[y:y + height, x:x + width] = 170
        states = self.off_frames(8, changed)
        self.assertFalse(any(s['confirmed'] for s in states), self.d.diagnostics())
        self.assertEqual(self.d.diagnostics()['reason'], 'BACKGROUND_STRUCTURE_CHANGED')

    def test_short_off_does_not_cumulate_across_an_on_frame(self):
        self.establish_on()
        self.assertFalse(any(s['confirmed'] for s in self.off_frames(2)))
        for _ in range(4):
            self.assertFalse(self.frame(self.scene(240))['confirmed'])
        self.assertEqual(self.d.state()['hits'], 0)
        self.assertTrue(self.off_frames()[-1]['confirmed'], self.d.diagnostics())

    def test_peak_to_peak_rejection_then_stable_window_resets_statistics(self):
        self.frame(self.scene(170))
        for value in (180, 220, 255, 255):
            state = self.frame(self.scene(value))
        self.assertFalse(state['locked'], self.d.diagnostics())
        self.assertEqual(self.d.diagnostics()['reason'], 'UNSTABLE_BRIGHT_BASELINE')
        for _ in range(4):
            state = self.frame(self.scene(255))
        self.assertTrue(state['locked'], self.d.diagnostics())
        self.assertTrue(self.off_frames()[-1]['confirmed'], self.d.diagnostics())

    def test_last_acquisition_frame_flash_is_not_the_reference(self):
        # At 50 Hz, the 300 ms lock boundary can land on one anomalous frame.
        self.frame(self.scene(100), 20)
        for _ in range(15):
            state = self.frame(self.scene(180), 20)
            self.assertFalse(state['locked'])
        state = self.frame(self.scene(223), 180)
        self.assertFalse(state['locked'], self.d.diagnostics())
        for _ in range(4):
            state = self.frame(self.scene(180), 150)
            self.assertFalse(state['confirmed'], self.d.diagnostics())
        self.assertTrue(state['locked'], self.d.diagnostics())
        for _ in range(4):
            self.assertFalse(self.frame(self.scene(180), 150)['confirmed'])
        self.assertTrue(self.off_frames(image=self.scene(100))[-1]['confirmed'],
                        self.d.diagnostics())


if __name__ == '__main__':
    unittest.main()
