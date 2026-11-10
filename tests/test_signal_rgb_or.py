"""Replay the actual RGB565 detector; synthetic controls, no device/motor I/O.

Physical emission, camera transforms and corrupt frames are separate controls.
These tests protect implementation behavior, not field accuracy or the ability
to distinguish two physical situations with identical recorded RGB pixels.
"""
import ctypes
import unittest

import numpy as np

from tests.support import ROOT
from esp32s3.video_preview.replay import NativeDetector


class SignalRGBOrTests(unittest.TestCase):
    BOX = [100, 40, 80, 64]
    MODES = ("OFF", "ON")

    @classmethod
    def setUpClass(cls):
        cls.detector = NativeDetector(full_frame=True)

    @classmethod
    def tearDownClass(cls):
        cls.detector.close()

    def setUp(self):
        self.d = self.detector
        self.d.cancel()
        self.assertTrue(self.workspace(True))
        self.now = 0

    def workspace(self, enabled):
        owned = self.d._rgb_reference
        self.assertIsNotNone(owned, "RGB replay workspace support is required")
        return bool(self.d.lib.rgbReferenceBuffer(
            self.d.handle, owned if enabled else None,
            ctypes.sizeof(owned) if enabled else 0))

    def scene(self, pixel, background=None):
        if background is None:
            # Several independent background levels identify affine WB changes.
            image = np.repeat(np.repeat(
                np.linspace(40, 160, 320, dtype=np.uint8)[None, :, None], 240, 0), 3, 2)
        else:
            image = np.full((240, 320, 3), background, np.uint8)
        x, y, width, height = self.BOX
        image[y:y + height, x:x + width] = pixel
        return image

    def strong_red(self, background=None):
        return [self.scene((60, 60, red), background) for red in (180, 180, 130, 130)]

    def sample(self, image, step=100):
        state = self.d.process(image, self.now)
        self.now += step
        return state

    def establish(self, mode, on=None, off=None):
        self.d.cancel()
        self.assertTrue(self.d.manual(self.BOX))
        self.d.cycle(mode == "OFF")
        self.d.arm()
        self.now = 0
        off = self.scene(60) if off is None else off
        on = self.strong_red() if on is None else on
        on = on if isinstance(on, list) else [on] * 4
        if mode == "OFF":
            for _ in range(4):
                state = self.sample(off)
                self.assertFalse(state["locked"])
                self.assertFalse(state["confirmed"])
        for index, image in enumerate(on):
            state = self.sample(image)
            self.assertEqual(bool(state["locked"]), index == 3, self.d.diagnostics())
            self.assertFalse(state["confirmed"])
        return state

    def confirm_rgb(self, off):
        states = [self.sample(off, 200) for _ in range(3)]
        self.assertEqual([bool(s["confirmed"]) for s in states], [False, False, True])
        evidence = self.d.rgb_diagnostics()
        self.assertFalse(evidence["luma_off"])
        self.assertTrue(evidence["rgb_off"], evidence)
        self.assertEqual(states[-1]["elapsed"], 400)
        return evidence

    def test_rgb_rescues_a_real_red_off_rejected_by_the_noisy_luma_baseline(self):
        # Fixed ambient60 + varying red emission; OFF removes emission.
        for mode in self.MODES:
            with self.subTest(mode=mode):
                self.establish(mode)
                evidence = self.confirm_rgb(self.scene(60))
                self.assertTrue(evidence["background_fitted"])
                self.assertEqual(evidence["channel"], 0)
                self.assertGreater(evidence["joint_fraction"], .9)

    def test_rgb_uses_the_same_frame_count_and_elapsed_time_confirmation(self):
        off = self.scene(60)
        for mode in self.MODES:
            with self.subTest(mode=mode, boundary="two_frames"):
                self.establish(mode)
                self.assertFalse(self.sample(off, 600)["confirmed"])
                self.assertFalse(self.sample(off, 100)["confirmed"])
                self.assertTrue(self.d.rgb_diagnostics()["rgb_off"])
                self.assertTrue(self.sample(off)["confirmed"])
            with self.subTest(mode=mode, boundary="three_frames_too_short"):
                self.establish(mode)
                for _ in range(3):
                    self.assertFalse(self.sample(off, 100)["confirmed"])
                self.now += 100
                self.assertTrue(self.sample(off)["confirmed"])
            with self.subTest(mode=mode, boundary="three_frames_400ms"):
                self.establish(mode)
                self.confirm_rgb(off)

    def test_low_colored_light_and_weak_red_remainder_are_not_black_gray(self):
        for mode in self.MODES:
            for green in (65, 80, 95):
                with self.subTest(mode=mode, still_green=green):
                    self.establish(mode)
                    for _ in range(4):
                        self.assertFalse(self.sample(self.scene((12, green, 12)), 200)["confirmed"])
                    self.assertFalse(self.d.rgb_diagnostics()["rgb_off"])
                    self.assertEqual(self.d.rgb_diagnostics()["reason"], "RGB_NOT_DARK")
            with self.subTest(mode=mode, still_red=True):
                on = [self.scene((50, 50, red)) for red in (175, 175, 128, 128)]
                self.establish(mode, on, self.scene(50))
                for _ in range(4):
                    self.assertFalse(self.sample(self.scene((50, 50, 64)), 200)["confirmed"])
                self.assertEqual(self.d.rgb_diagnostics()["reason"], "RGB_NOT_DARK")

    def test_background_observability_is_required_for_the_channel_that_fades(self):
        def cyan(pixel):
            image = self.scene(pixel)
            image[:, :, 2] = 0  # No R evidence outside the target.
            image[40:104, 100:180] = pixel
            return image
        on = [cyan((60, 60, red)) for red in (180, 180, 130, 130)]
        after = on[-1].copy()
        after[:, :, 2] = (after[:, :, 2].astype(float) * (60 / 130)).astype(np.uint8)
        for mode in self.MODES:
            with self.subTest(mode=mode):
                self.establish(mode, on, cyan((60, 60, 60)))
                for _ in range(4):
                    self.assertFalse(self.sample(after, 200)["confirmed"])
                evidence = self.d.rgb_diagnostics()
                self.assertFalse(evidence["rgb_off"])
                self.assertNotEqual(evidence["channel"], 0)

    def test_observable_channel_white_balance_is_compensated_before_rgb_votes(self):
        on = self.strong_red()
        after = on[-1].copy()
        after[:, :, 2] = (after[:, :, 2].astype(float) * (60 / 130)).astype(np.uint8)
        for mode in self.MODES:
            with self.subTest(mode=mode):
                self.establish(mode, on)
                for _ in range(4):
                    self.assertFalse(self.sample(after, 200)["confirmed"])
                evidence = self.d.rgb_diagnostics()
                self.assertTrue(evidence["background_fitted"])
                self.assertFalse(evidence["rgb_off"])
                self.assertLess(evidence["joint_fraction"], .1)

    def test_flat_background_cannot_certify_a_hidden_affine_white_balance_change(self):
        on = self.strong_red((50, 50, 50))
        after = on[-1].copy()
        after[:, :, 2] = np.clip(after[:, :, 2].astype(float) * .1 + 45, 0, 255).astype(np.uint8)
        for mode in self.MODES:
            with self.subTest(mode=mode):
                self.establish(mode, on, self.scene(60, (50, 50, 50)))
                for _ in range(4):
                    self.assertFalse(self.sample(after, 200)["confirmed"])
                evidence = self.d.rgb_diagnostics()
                self.assertFalse(evidence["luma_off"])
                self.assertFalse(evidence["background_fitted"])
                self.assertEqual(evidence["reason"], "BACKGROUND_UNOBSERVABLE")

    def test_global_darkening_and_all_black_camera_frames_do_not_add_a_start(self):
        on = self.scene(240)
        transforms = {
            "gain": (on.astype(float) * .3).astype(np.uint8),
            "offset": np.maximum(on.astype(int) - 170, 0).astype(np.uint8),
            "bad_black_frame": np.zeros_like(on),
        }
        for mode in self.MODES:
            for kind, after in transforms.items():
                with self.subTest(mode=mode, camera_change=kind):
                    self.establish(mode, on, self.scene(20))
                    for _ in range(4):
                        self.assertFalse(self.sample(after, 200)["confirmed"])
                    evidence = self.d.rgb_diagnostics()
                    self.assertFalse(evidence["luma_off"])
                    self.assertFalse(evidence["rgb_off"])

    def test_color_drop_and_black_consensus_must_share_the_same_pixels(self):
        on, after, off = self.scene(20), self.scene(20), self.scene(20)
        on[40:104, 100:156] = (12, 12, 240)  # About70% emits red.
        after[40:104, 100:132] = (12, 160, 12)  # About40% remains visibly ON.
        for mode in self.MODES:
            with self.subTest(mode=mode):
                self.establish(mode, on, off)
                for _ in range(4):
                    self.assertFalse(self.sample(after, 200)["confirmed"])
                evidence = self.d.rgb_diagnostics()
                self.assertGreater(evidence["dark_fraction"], .6)
                self.assertLess(evidence["joint_fraction"], .5)
                self.assertFalse(evidence["rgb_off"])

    def test_missing_rgb_workspace_preserves_the_original_luma_path(self):
        self.assertFalse(self.workspace(False))
        for mode in self.MODES:
            with self.subTest(mode=mode, path="RGB_unavailable"):
                self.establish(mode)
                for _ in range(4):
                    self.assertFalse(self.sample(self.scene(60), 200)["confirmed"])
                evidence = self.d.rgb_diagnostics()
                self.assertFalse(evidence["workspace_available"])
                self.assertEqual(evidence["reason"], "NO_WORKSPACE")
            with self.subTest(mode=mode, path="luma_still_available"):
                self.establish(mode, self.scene(240), self.scene(20))
                states = [self.sample(self.scene(20), 200) for _ in range(3)]
                self.assertTrue(states[-1]["confirmed"])
                self.assertTrue(self.d.rgb_diagnostics()["luma_off"])

    def test_an_armed_session_rejects_buffer_replacement_without_losing_its_reference(self):
        for mode in self.MODES:
            with self.subTest(mode=mode):
                self.establish(mode)
                replacement = ctypes.create_string_buffer(ctypes.sizeof(self.d._rgb_reference))
                self.assertFalse(self.d.lib.rgbReferenceBuffer(
                    self.d.handle, replacement, ctypes.sizeof(replacement)))
                self.assertTrue(self.d.rgb_diagnostics()["reference_ready"])
                self.confirm_rgb(self.scene(60))

    def test_cancel_clears_the_rgb_reference_and_pending_confirmation(self):
        for mode in self.MODES:
            with self.subTest(mode=mode):
                self.establish(mode)
                for _ in range(2):
                    self.assertFalse(self.sample(self.scene(60), 200)["confirmed"])
                self.d.cancel()
                evidence = self.d.rgb_diagnostics()
                self.assertFalse(evidence["reference_ready"])
                self.assertTrue(evidence["workspace_available"])
                self.assertFalse(evidence["rgb_off"])
                self.d.arm()
                self.assertFalse(self.sample(self.scene(60))["confirmed"])
                self.assertFalse(self.d.rgb_diagnostics()["reference_ready"])
                self.establish(mode)
                self.confirm_rgb(self.scene(60))

    def test_camera_gap_cannot_finish_confirmation_using_the_old_rgb_reference(self):
        for mode in self.MODES:
            with self.subTest(mode=mode):
                self.establish(mode)
                for _ in range(2):
                    self.assertFalse(self.sample(self.scene(60), 200)["confirmed"])
                self.now += 1000
                state = self.sample(self.scene(60))
                self.assertFalse(state["confirmed"])
                self.assertEqual(state["hits"], 0)
                self.assertFalse(self.d.rgb_diagnostics()["reference_ready"])
                self.establish(mode)
                self.confirm_rgb(self.scene(60))

    def test_fixed_red_reflection_does_not_veto_real_off_in_the_luma_path(self):
        pairs = [((80, 180, 200), (80, 60, 200)),
                 ((100, 180, 255), (30, 60, 255)),
                 ((255, 255, 255), (30, 60, 255))]
        for mode in self.MODES:
            for lit, dark in pairs:
                with self.subTest(mode=mode, lit=lit):
                    self.establish(mode, self.scene(lit), self.scene(dark))
                    states = [self.sample(self.scene(dark), 200) for _ in range(3)]
                    self.assertTrue(states[-1]["confirmed"])
                    evidence = self.d.rgb_diagnostics()
                    self.assertTrue(evidence["luma_off"])
                    self.assertFalse(evidence["rgb_off"])
                    self.assertEqual(evidence["reason"], "LUMA_PATH_USED")


if __name__ == "__main__":
    unittest.main()
