"""Complete camera field of view and native touch controls remain separate."""

from tests.support import ROOT
import unittest
import cv2, numpy as np

from robot.interface.screen_view import DeviceScreen

SIZES = ((552, 368), (640, 480), (480, 640), (320, 240))


def picture():
    im = np.full((240, 320, 3), 60, np.uint8)
    im[:8] = [0, 0, 255]
    im[-8:] = [255, 0, 0]
    im[:, :8] = [0, 255, 0]
    im[:, -8:] = [0, 255, 255]
    return im


class DeviceScreenTests(unittest.TestCase):
    def test_all_camera_edges_visible_on_landscape_and_portrait_screens(self):
        for size in SIZES:
            with self.subTest(size=size):
                d = DeviceScreen(*size)
                out = d.compose(picture(), "WAIT START", (0, 0), 1, False)
                x, y, w, h = d.frame_rect
                np.testing.assert_array_equal(out[y, x + w // 2], [0, 0, 255])
                np.testing.assert_array_equal(out[y + h - 1, x + w // 2], [255, 0, 0])
                np.testing.assert_array_equal(out[y + h // 2, x], [0, 255, 0])
                np.testing.assert_array_equal(out[y + h // 2, x + w - 1], [0, 255, 255])

    def test_status_and_buttons_cannot_obscure_any_viewport_pixels(self):
        for size in SIZES:
            d = DeviceScreen(*size)
            im = picture()
            a = d.compose(im, "STOPPED", (0, 0), 0, False)
            b = d.compose(im, "CROSSWALK POSITION UNCERTAIN", (200, 180), 123456, True)
            x, y, w, h = d.frame_rect
            np.testing.assert_array_equal(a[y : y + h, x : x + w], b[y : y + h, x : x + w])
            ex, ey, ew, eh = d.exit_rect
            np.testing.assert_array_equal(b[ey + 1, ex + 1], [90, 50, 30])

    def test_frame_shape_aspect_and_non_overlapping_screen_regions(self):
        for sw, sh in SIZES:
            d = DeviceScreen(sw, sh)
            x, y, w, h = d.frame_rect
            self.assertEqual(d.compose(picture(), "READY", (0, 0), 1, True).shape, (sh, sw, 3))
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, d.header_height)
            self.assertLessEqual(x + w, sw)
            self.assertLessEqual(y + h, d.footer_top)
            self.assertLessEqual(abs(w - 4 * h / 3), 1.0)

    def test_composing_screen_never_modifies_vision_input(self):
        im = picture()
        before = im.copy()
        DeviceScreen(552, 368).compose(im, "RUNNING", (160, 150), 1, True)
        np.testing.assert_array_equal(im, before)

    def test_native_menu_has_bottom_arm_stop_and_top_exit_actions(self):
        for sw, sh in SIZES:
            d = DeviceScreen(sw, sh)
            cy = (d.footer_top + sh) // 2
            for fraction, expected in ((1 / 6, "start"), (0.5, "stop"), (5 / 6, "stop")):
                self.assertEqual(d.action_at(sw * fraction, cy), expected)
            self.assertEqual(d.action_at(0, sh - 1), "start")
            self.assertEqual(d.action_at(sw - 1, sh - 1), "stop")
            self.assertEqual([b["label"] for b in d.buttons], ["ARM", "STOP", "EXIT"])
            ex, ey, ew, eh = d.exit_rect
            self.assertEqual(d.action_at(ex + ew / 2, ey + eh / 2), "exit")

    def test_camera_and_status_taps_cannot_trigger_buttons(self):
        for sw, sh in SIZES:
            d = DeviceScreen(sw, sh)
            x, y, w, h = d.frame_rect
            for tx, ty in (
                (0, 0),
                (sw / 2, d.header_height / 2),
                (x, y),
                (x + w - 1, y + h - 1),
                (sw / 2, d.footer_top - 1),
            ):
                self.assertIsNone(d.action_at(tx, ty))

    def test_outside_screen_taps_never_start_or_exit(self):
        d = DeviceScreen(552, 368)
        for x, y in ((-1, 367), (552, 367), (1, 368), (1, -1), (float("nan"), 367)):
            self.assertIsNone(d.action_at(x, y))

    def test_touch_rectangles_match_the_visible_button_backgrounds(self):
        d = DeviceScreen(552, 368)
        out = d.compose(picture(), "STOPPED", (0, 0), 1, arm_ready=True)
        for b, color in zip(d.buttons, ((170, 80, 0), (0, 0, 180), (90, 50, 30))):
            x, y, w, h = b["rect"]
            np.testing.assert_array_equal(out[y + 1, x + 1], color)
            self.assertEqual(d.action_at(x + 1, y + 1), b["action"])

    def test_stop_and_exit_remain_available_when_arm_disabled(self):
        d = DeviceScreen(552, 368)
        out = d.compose(picture(), "ROAD NOT READY", (0, 0), 1, False)
        x, y, w, h = d.buttons[0]["rect"]
        np.testing.assert_array_equal(out[y + 1, x + 1], [55, 55, 55])
        stop = next(b for b in d.buttons if b["action"] == "stop")
        sx, sy, sw, sh = stop["rect"]
        np.testing.assert_array_equal(out[sy + 1, sx + 1], [0, 0, 180])
        self.assertEqual(d.action_at(sx + sw / 2, sy + sh / 2), "stop")
        ex, ey, ew, eh = d.exit_rect
        self.assertEqual(d.action_at(ex + ew / 2, ey + eh / 2), "exit")

    def test_exit_is_in_top_right_header_outside_camera_view(self):
        for sw, sh in SIZES:
            d = DeviceScreen(sw, sh)
            x, y, w, h = d.exit_rect
            self.assertGreater(x, sw / 2)
            self.assertLessEqual(y + h, d.header_height)
            self.assertLessEqual(x + w, sw)
            self.assertEqual(d.action_at(x + w / 2, y + h / 2), "exit")

    def test_arm_label_action_and_waiting_colour(self):
        d = DeviceScreen(640, 480)
        arm = next(b for b in d.buttons if b["label"] == "ARM")
        x, y, w, h = arm["rect"]
        self.assertEqual(d.action_at(x + w / 2, y + h / 2), "start")
        waiting = d.compose(picture(), "WAIT START LIGHT", (0, 0), 1, arm_ready=False, armed=True)
        np.testing.assert_array_equal(waiting[y + 1, x + 1], [0, 120, 180])

    def test_telemetry_describes_the_same_display_layout(self):
        d = DeviceScreen(552, 368)
        s = d.snapshot()
        self.assertEqual(s["frame_rect"], list(d.frame_rect))
        self.assertTrue(s["camera_frame_complete"])
        self.assertEqual(s["fit"], "CONTAIN")

    def test_invalid_display_and_frame_sizes_are_rejected(self):
        for size in ((0, 0), (80, 240), (320, 80)):
            with self.assertRaises(ValueError):
                DeviceScreen(*size)
        d = DeviceScreen(552, 368)
        with self.assertRaises(ValueError):
            d.compose(np.zeros((240, 320), np.uint8), "STOPPED", (0, 0), 1, False)


if __name__ == "__main__":
    unittest.main()
