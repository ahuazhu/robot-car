"""Brightness-only signal, compact region and one-shot start semantics."""

from tests.support import ROOT
import json, unittest
from unittest.mock import patch
import cv2, numpy as np

from robot.tasks.start_light import StartLight

CONFIG = json.loads((ROOT / "maixcam2/curve_follow/runtime.json").read_text())["start_light"]


def gate(**changes):
    c = dict(
        CONFIG,
        trigger_direction="BOTH",
        delta_luma=35,
        auto_arm_on_launch=False,
        search_roi_fraction=[0, 0, 1, 1 / 3],
        maximum_region_fraction=0.2,
        confirm_s=0.4,
        confirm_frames=3,
    )
    c.update(changes)
    return StartLight(c)


def scene():
    return np.full((240, 320, 3), 30, np.uint8)


def lamp(im, value=(0, 0, 230), x=140, y=12, w=32, h=24):
    out = im.copy()
    out[y : y + h, x : x + w] = value
    return out


def event(direction="BRIGHTER", x=140):
    return dict(state="CHANGE", direction=direction, roi=[x, 12, 32, 24])


class BrightnessDetectorTests(unittest.TestCase):
    def test_red_light_up_and_off_both_confirm(self):
        off = scene()
        on = lamp(off)
        for before, after, direction in ((off, on, "BRIGHTER"), (on, off, "DARKER")):
            with self.subTest(direction=direction):
                d = gate()
                d.arm(before)
                o = d.detect(after)
                self.assertEqual(o["state"], "CHANGE")
                self.assertEqual(o["direction"], direction)
                self.assertEqual(o["kind"], "BRIGHTNESS")
                self.assertGreaterEqual(o["magnitude"], CONFIG["delta_luma"])
                self.assertNotIn("color_magnitude", o)
                self.assertGreaterEqual(o["changed_pixels"], 512)
                self.assertFalse(d.update(o, 0.0))
                self.assertFalse(d.update(d.detect(after), 0.21))
                self.assertTrue(d.update(d.detect(after), 0.42))

    def test_brightness_alone_confirms_both_light_up_and_off(self):
        off = scene()
        on = lamp(off, 230)
        for before, after in ((off, on), (on, off)):
            d = gate()
            d.arm(before)
            self.assertFalse(d.update(d.detect(after), 0.0))
            self.assertFalse(d.update(d.detect(after), 0.21))
            self.assertTrue(d.update(d.detect(after), 0.42))

    def test_colour_alone_at_equal_lightness_is_rejected(self):
        a = np.full((240, 320, 3), [160, 128, 128], np.uint8)
        b = a.copy()
        b[12:36, 140:172] = [160, 190, 155]
        before = cv2.cvtColor(a, cv2.COLOR_LAB2BGR)
        after = cv2.cvtColor(b, cv2.COLOR_LAB2BGR)
        d = gate()
        d.arm(before)
        self.assertEqual(d.detect(after)["state"], "STEADY")

    def test_brightness_region_is_independent_of_colour_elsewhere(self):
        before_lab = np.full((240, 320, 3), [100, 128, 128], np.uint8)
        after_lab = before_lab.copy()
        after_lab[10:40, 20:60, 0] = 230
        after_lab[10:40, 200:240, 1] = 185
        d = gate()
        d.arm(cv2.cvtColor(before_lab, cv2.COLOR_LAB2BGR))
        self.assertEqual(d.detect(cv2.cvtColor(after_lab, cv2.COLOR_LAB2BGR))["state"], "CHANGE")

    def test_interrupted_brightness_change_does_not_confirm(self):
        d = gate()
        d.arm(scene())
        for i, im in enumerate((lamp(scene(), 230), scene(), lamp(scene(), 230))):
            self.assertFalse(d.update(d.detect(im), i * 0.21))

    def test_no_signal_from_unchanged_lit_lamp(self):
        im = lamp(scene())
        d = gate()
        d.arm(im)
        for t in (0.0, 0.21, 0.42, 1.0):
            self.assertFalse(d.update(d.detect(im), t))

    def test_below_upper_third_is_ignored(self):
        before = scene()
        after = before.copy()
        after[80:] = [0, 0, 230]
        d = gate()
        d.arm(before)
        self.assertEqual(d.detect(after)["state"], "STEADY")

    def test_all_horizontal_locations_in_upper_band_are_eligible(self):
        for x, y in ((0, 0), (288, 0), (0, 56), (288, 56)):
            d = gate()
            d.arm(scene())
            o = d.detect(lamp(scene(), x=x, y=y))
            self.assertEqual(o["state"], "CHANGE")
            self.assertLessEqual(o["roi"][1] + o["roi"][3], 80)

    def test_not_restricted_to_red_or_a_template(self):
        for value in ((0, 230, 0), (0, 0, 230), (230, 0, 0)):
            d = gate()
            d.arm(scene())
            self.assertEqual(d.detect(lamp(scene(), value))["state"], "CHANGE")

    def test_tiny_joint_patch_is_rejected(self):
        d = gate()
        d.arm(scene())
        self.assertEqual(d.detect(lamp(scene(), w=10, h=10))["state"], "STEADY")

    def test_disconnected_small_patches_are_not_summed(self):
        after = scene()
        for x in (10, 60, 110, 160, 210, 260):
            after = lamp(after, x=x, w=10, h=10)
        d = gate()
        d.arm(scene())
        self.assertEqual(d.detect(after)["state"], "STEADY")

    def test_widespread_joint_change_is_rejected(self):
        d = gate()
        d.arm(scene())
        self.assertEqual(d.detect(np.full((240, 320, 3), [0, 0, 230], np.uint8))["state"], "STEADY")

    def test_long_thin_change_is_not_a_compact_lamp_region(self):
        d = gate()
        d.arm(scene())
        self.assertEqual(d.detect(lamp(scene(), x=0, y=20, w=280, h=10))["state"], "STEADY")

    def test_sparse_joint_ring_is_not_a_filled_compact_change(self):
        before = scene()
        after = before.copy()
        cv2.rectangle(after, (100, 5), (169, 74), (0, 0, 230), 2)
        d = gate(minimum_changed_pixels=64)
        d.arm(before)
        self.assertEqual(d.detect(after)["state"], "STEADY")

    def test_independent_lamp_regions_can_be_detected(self):
        after = lamp(lamp(scene(), x=20), x=240)
        d = gate()
        d.arm(scene())
        self.assertEqual(d.detect(after)["state"], "CHANGE")

    def test_morphological_joining_does_not_invent_change_pixels(self):
        after = scene()
        for y in range(6, 40, 4):
            for x in range(100, 130, 4):
                after[y, x] = [0, 0, 230]
        d = gate()
        d.arm(scene())
        self.assertEqual(d.detect(after)["state"], "STEADY")

    def test_no_chroma_condition_or_colour_configuration_remains(self):
        d = gate()
        self.assertNotIn("delta_color", CONFIG)
        self.assertFalse(hasattr(d, "color_delta"))
        self.assertEqual(d._prepare(scene()).shape, (40, 160))
        d.arm(scene())
        self.assertEqual(d.detect(lamp(scene(), 230))["kind"], "BRIGHTNESS")

    def test_brightness_below_raised_threshold_is_ignored(self):
        lab = np.full((240, 320, 3), [100, 128, 128], np.uint8)
        after = lab.copy()
        after[12:36, 140:172, 0] = 130
        d = gate()
        d.arm(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
        self.assertEqual(d.detect(cv2.cvtColor(after, cv2.COLOR_LAB2BGR))["state"], "STEADY")

    def test_recorded_minor_image_motion_does_not_trigger(self):
        frames = [
            cv2.imread(str(p))
            for p in sorted((ROOT / "tests/fixtures/start_signal_jitter_20261031").glob("*.png"))
        ]
        self.assertGreater(len(frames), 5)
        d = gate()
        d.arm(frames[0])
        for i, im in enumerate(frames[1:], 1):
            self.assertFalse(d.update(d.detect(im), i * 0.21))

    def test_explicit_auto_arm_records_reference_frames_first(self):
        d = StartLight(dict(CONFIG, auto_arm_on_launch=True))
        im = lamp(scene())
        for _ in range(CONFIG["confirm_frames"]):
            self.assertEqual(d.detect(im)["state"], "BASELINE")
        self.assertEqual(d.detect(im)["state"], "STEADY")
        self.assertTrue(d.snapshot()["baseline_ready"])

    def test_invalid_images_and_no_template_loading(self):
        with patch.object(cv2, "imread", side_effect=AssertionError("No templates")):
            d = gate()
            d.arm()
            for im in (None, np.zeros((120, 160, 3), np.uint8)):
                self.assertEqual(d.detect(im)["state"], "UNKNOWN")
            self.assertFalse(d.snapshot()["baseline_ready"])


class SignalGateTests(unittest.TestCase):
    def test_configured_launch_waits_for_manual_arm(self):
        d = StartLight(CONFIG)
        self.assertFalse(d.armed)
        self.assertEqual(d.phase, "DISARMED")
        after = scene()
        before = lamp(after, 230)
        for now, image in ((0.0, before), (0.3, after), (0.6, after), (1.0, after)):
            self.assertFalse(d.update(d.detect(image), now))
        self.assertIsNone(d.reference)
        self.assertFalse(d.fired)
        d.arm(before)
        self.assertFalse(d.update(d.detect(after), 2.0))
        self.assertFalse(d.update(d.detect(after), 2.21))
        self.assertFalse(d.update(d.detect(after), 2.42))
        self.assertTrue(d.update(d.detect(after), 2.63))

    def ready_gate(self):
        d = gate()
        d.arm(scene())
        return d

    def confirm(self, d):
        self.assertFalse(d.update(event(), 0.0))
        self.assertFalse(d.update(event(), 0.21))
        self.assertTrue(d.update(event(), 0.42))

    def test_short_flash_cannot_confirm(self):
        d = self.ready_gate()
        d.update(event(), 0.0)
        self.assertFalse(d.update(event(), 0.21))
        self.assertFalse(d.update(dict(state="STEADY"), 0.42))
        self.assertFalse(d.latched)

    def test_direction_or_region_change_resets_confirmation(self):
        for second in (event("DARKER"), event(x=240)):
            d = self.ready_gate()
            d.update(event(), 0.0)
            self.assertFalse(d.update(second, 0.21))
            self.assertEqual(d.hits, 1)

    def test_stale_slow_and_nonfinite_frames_reset_unconfirmed_signal(self):
        for now, timely in ((1.0, True), (0.21, False), (float("nan"), True), (-0.1, True)):
            d = self.ready_gate()
            d.update(event(), 0.0)
            self.assertFalse(d.update(event(), now, timely=timely))
            self.assertEqual(d.hits, 0)

    def test_latched_signal_can_wait_for_shared_road_motor_gates(self):
        d = self.ready_gate()
        self.confirm(d)
        self.assertTrue(d.update(dict(state="STEADY"), 1.0, timely=False))
        self.assertEqual(d.phase, "SIGNAL_READY")

    def test_stop_clears_pending_and_latched_signal(self):
        for confirmed in (False, True):
            d = self.ready_gate()
            if confirmed:
                self.confirm(d)
            else:
                d.update(event(), 0.0)
            d.cancel()
            self.assertFalse(d.update(event(), 1.0))
            self.assertFalse(d.latched)
            self.assertIsNone(d.reference)

    def test_rearm_discards_old_signal_and_uses_current_image(self):
        d = self.ready_gate()
        self.confirm(d)
        d.cancel()
        d.arm(lamp(scene()))
        self.assertFalse(d.update(d.detect(lamp(scene())), 1.0))
        self.assertEqual(d.hits, 0)

    def test_no_detection_after_start_or_when_disarmed(self):
        d = self.ready_gate()
        d.started()
        with patch.object(d, "_prepare", side_effect=AssertionError("No post-start processing")):
            d.detect(scene())
        self.assertFalse(d.update(event(), 1.0))
        self.assertFalse(d.armed)
        self.assertTrue(d.fired)
        d.cancel()
        with patch.object(d, "_prepare", side_effect=AssertionError("No disarmed processing")):
            d.detect(scene())

    def test_latched_signal_does_not_keep_processing_images(self):
        d = self.ready_gate()
        self.confirm(d)
        with patch.object(d, "_prepare", side_effect=AssertionError("Already confirmed")):
            d.detect(scene())

    def test_disabled_never_arms(self):
        d = StartLight()
        d.arm(scene())
        self.assertFalse(d.update(event(), 0.0))
        self.assertFalse(d.armed)

    def test_compact_rule_visible_in_snapshot(self):
        s = self.ready_gate().snapshot()
        self.assertEqual(s["change_rule"], "BRIGHTNESS_ONLY")
        self.assertNotIn("delta_color_min", s)
        self.assertEqual(s["minimum_changed_pixels"], 512)

    def test_invalid_configuration_rejected(self):
        for change in (
            dict(enabled="yes"),
            dict(auto_arm_on_launch=1),
            dict(search_roi_fraction=[0, 0, 0, 1 / 6]),
            dict(delta_luma=float("nan")),
            dict(confirm_frames=1),
            dict(confirm_frames=True),
            dict(minimum_changed_pixels=0),
            dict(maximum_frame_gap_s=0),
            dict(maximum_region_fraction=0),
            dict(minimum_region_fill=2),
            dict(maximum_region_aspect=0.5),
            dict(minimum_confirmation_iou=0),
            dict(minimum_confirmation_iou=1.1),
            dict(minimum_confirmation_iou=float("nan")),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                StartLight(dict(CONFIG, **change))


class SearchRectangleTests(unittest.TestCase):
    def detector(self):
        return StartLight(
            dict(CONFIG, trigger_direction="BOTH", delta_luma=35, auto_arm_on_launch=False)
        )

    def test_actual_configuration_matches_requested_rectangle(self):
        d = self.detector()
        self.assertEqual(d.search_roi, (128, 0, 64, 32))
        self.assertEqual(d._prepare(scene()).shape, (16, 32))
        self.assertEqual(d.snapshot()["search_roi"], [128, 0, 64, 32])

    def test_changes_outside_rectangle_on_all_sides_are_ignored(self):
        for x0, y0, x1, y1 in ((0, 0, 128, 240), (192, 0, 320, 240), (0, 32, 320, 240)):
            after = scene()
            after[y0:y1, x0:x1] = 255
            d = self.detector()
            d.arm(scene())
            for t in (0.0, 0.21, 0.42):
                self.assertFalse(d.update(d.detect(after), t))
            self.assertEqual(d.observation["state"], "STEADY")

    def test_filter_cannot_leak_immediate_outside_pixels_into_rectangle(self):
        before = scene()
        after = np.full_like(before, 255)
        after[:32, 128:192] = before[:32, 128:192]
        d = self.detector()
        d.arm(before)
        self.assertEqual(d.detect(after)["state"], "STEADY")

    def test_both_brightness_polarities_still_confirm_inside_actual_rectangle(self):
        before = scene()
        after = lamp(before, 230)
        for a, b in ((before, after), (after, before)):
            d = self.detector()
            d.arm(a)
            self.assertFalse(d.update(d.detect(b), 0.0))
            self.assertFalse(d.update(d.detect(b), 0.21))
            self.assertFalse(d.update(d.detect(b), 0.42))
            self.assertTrue(d.update(d.detect(b), 0.63))

    def test_candidate_coordinates_are_in_full_image_space(self):
        for x, y in ((128, 0), (160, 8)):
            d = self.detector()
            d.arm(scene())
            o = d.detect(lamp(scene(), 230, x=x, y=y))
            self.assertEqual(o["state"], "CHANGE")
            rx, ry, w, h = o["roi"]
            self.assertGreaterEqual(rx, 128)
            self.assertGreaterEqual(ry, 0)
            self.assertLessEqual(rx + w, 192)
            self.assertLessEqual(ry + h, 32)
            self.assertLess(abs(rx - x), 4)
            self.assertLess(abs(ry - y), 4)

    def test_moderate_change_passes_new_gate_but_not_previous_threshold(self):
        lab = np.full((240, 320, 3), [100, 128, 128], np.uint8)
        changed = lab.copy()
        changed[4:28, 140:172, 0] = 140
        off = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        on = cv2.cvtColor(changed, cv2.COLOR_LAB2BGR)
        for before, after in ((off, on), (on, off)):
            old = StartLight(
                dict(CONFIG, trigger_direction="BOTH", auto_arm_on_launch=False, delta_luma=45)
            )
            old.arm(before)
            self.assertEqual(old.detect(after)["state"], "STEADY")
            d = self.detector()
            d.arm(before)
            self.assertFalse(d.update(d.detect(after), 0.0))
            self.assertFalse(d.update(d.detect(after), 0.21))
            self.assertFalse(d.update(d.detect(after), 0.42))
            self.assertTrue(d.update(d.detect(after), 0.63))

    def test_rectangle_can_fill_without_old_full_width_area_conflict(self):
        # The requested rectangle already defines a concentrated region. The old
        # 20 percent bound would fall below the512px minimum in a smaller rectangle.
        d = self.detector()
        d.arm(scene())
        after = scene()
        after[:32, 128:192] = 230
        self.assertEqual(d.detect(after)["state"], "CHANGE")

    def test_preview_rectangle_exactly_matches_detection_crop(self):
        d = self.detector()
        original = scene()
        preview = original.copy()
        d.draw_search_region(preview)
        for x, y in ((128, 0), (191, 0), (128, 31), (191, 31)):
            np.testing.assert_array_equal(preview[y, x], [255, 255, 0])
        for x, y in ((127, 10), (192, 10), (160, 32), (160, 20)):
            np.testing.assert_array_equal(preview[y, x], original[y, x])
        np.testing.assert_array_equal(original, scene())
        d.arm(original)
        self.assertEqual(d.detect(original)["state"], "STEADY")

    def test_rectangle_remains_visible_while_disarmed_and_running(self):
        d = self.detector()
        for running in (False, True):
            if running:
                d.started()
            preview = scene()
            d.draw_search_region(preview)
            np.testing.assert_array_equal(preview[0, 128], [255, 255, 0])

    def test_invalid_rectangles_are_rejected(self):
        for rect in (
            [0, 0, 1],
            [0, 0, 0, 1],
            [-0.1, 0, 0.25, 1 / 6],
            [0.9, 0, 0.25, 1 / 6],
            [0, 0, 1, float("nan")],
            [0.375, 0, 0.25, 1 / 240],
        ):
            with self.subTest(rect=rect), self.assertRaises(ValueError):
                StartLight(dict(CONFIG, search_roi_fraction=rect))


class ReflectionConfirmationTests(unittest.TestCase):
    def detector(self, before=None):
        d = StartLight(dict(CONFIG, trigger_direction="BOTH", delta_luma=35))
        d.arm(scene() if before is None else before)
        return d

    def test_sweeping_reflection_does_not_accumulate_overlapping_regions(self):
        # Every adjacent pair overlaps; the old gate latched after the third frame.
        for dark in (False, True):
            before = np.full_like(scene(), 230) if dark else scene()
            d = self.detector(before)
            for i, x in enumerate((128, 140, 152, 160, 148, 136, 128)):
                image = lamp(before, 30 if dark else 230, x=x, y=4, w=32, h=24)
                self.assertFalse(d.update(d.detect(image), i * 0.21))

    def test_slow_drift_is_compared_to_first_region_not_previous_frame(self):
        d = self.detector()
        for i, x in enumerate((128, 132, 136, 140, 144, 148, 152, 156, 160)):
            self.assertFalse(d.update(d.detect(lamp(scene(), 230, x=x, y=4)), i * 0.16))

    def test_flash_longer_than_old_confirmation_is_rejected(self):
        for dark in (False, True):
            a, b = scene(), lamp(scene(), 230, y=4)
            if dark:
                a, b = b, a
            d = self.detector(a)
            for now in (0.0, 0.15, 0.3, 0.45, 0.55):
                self.assertFalse(d.update(d.detect(b), now))
            self.assertFalse(d.update(d.detect(a), 0.7))
            self.assertEqual(d.hits, 0)

    def test_stable_both_polarities_with_small_jitter_still_confirm(self):
        for dark in (False, True):
            before = np.full_like(scene(), 230) if dark else scene()
            d = self.detector(before)
            for i, x in enumerate((140, 142, 138, 140)):
                image = lamp(before, 30 if dark else 230, x=x, y=4)
                self.assertEqual(d.update(d.detect(image), i * 0.21), i == 3)

    def test_stable_signal_after_a_sweep_can_confirm_afresh(self):
        d = self.detector()
        for now, x in ((0.0, 128), (0.21, 140), (0.42, 152)):
            self.assertFalse(d.update(d.detect(lamp(scene(), 230, x=x, y=4)), now))
        for now in (0.63, 0.84):
            self.assertFalse(d.update(d.detect(lamp(scene(), 230, x=152, y=4)), now))
        self.assertTrue(d.update(d.detect(lamp(scene(), 230, x=152, y=4)), 1.05))

    def test_region_growth_resets_and_timing_does_not_replace_frame_count(self):
        d = self.detector()
        self.assertFalse(d.update(event(), 0.0))
        self.assertFalse(d.update(dict(event(), roi=[128, 0, 64, 32]), 0.21))
        self.assertEqual(d.hits, 1)
        d = self.detector()
        for t in (0.0, 0.31, 0.62):
            self.assertFalse(d.update(event(), t))
        self.assertTrue(d.update(event(), 0.83))

    def test_fixed_anchor_is_reported_in_telemetry(self):
        s = self.detector().snapshot()
        self.assertEqual(s["confirmation_anchor"], "FIRST_REGION")
        self.assertEqual(s["minimum_confirmation_iou"], 0.6)
        self.assertEqual(s["required_frames"], 4)
        self.assertEqual(s["required_s"], 0.6)


class DarkerProductionTests(unittest.TestCase):
    def images(self, before_luma=190, after_luma=160):
        before = np.full((240, 320, 3), [80, 128, 128], np.uint8)
        before[4:28, 140:180, 0] = before_luma
        after = before.copy()
        after[4:28, 140:180, 0] = after_luma
        return cv2.cvtColor(before, cv2.COLOR_LAB2BGR), cv2.cvtColor(after, cv2.COLOR_LAB2BGR)

    def detector(self, before):
        d = StartLight(CONFIG)
        d.arm(before)
        return d

    def test_production_direction_and_confirmation_are_explicit(self):
        s = StartLight(CONFIG).snapshot()
        self.assertEqual(s["trigger_direction"], "DARKER")
        self.assertEqual(s["delta_luma_min"], 28)
        self.assertEqual(s["required_frames"], 4)
        self.assertEqual(s["required_s"], 0.6)
        self.assertEqual(s["minimum_confirmation_iou"], 0.6)
        self.assertEqual(s["minimum_changed_pixels"], 512)
        self.assertEqual(s["search_roi"], [128, 0, 64, 32])

    def test_residual_light_and_lab_drop_30_pass_28_but_not_old_35(self):
        before, after = self.images()
        self.assertGreater(int(after[12, 150].min()), 0)
        d = self.detector(before)
        o = d.detect(after)
        self.assertEqual(o["state"], "CHANGE")
        self.assertEqual(o["direction"], "DARKER")
        self.assertGreaterEqual(o["magnitude"], 28)
        self.assertLess(o["magnitude"], 35)
        old = StartLight(dict(CONFIG, delta_luma=35))
        old.arm(before)
        self.assertEqual(old.detect(after)["state"], "STEADY")
        for i, t in enumerate((0.0, 0.21, 0.42, 0.63)):
            self.assertEqual(d.update(d.detect(after), t), i == 3)

    def test_below_28_darkening_never_confirms(self):
        before, after = self.images(after_luma=164)
        d = self.detector(before)
        for t in (0.0, 0.21, 0.42, 0.63, 1.0):
            self.assertEqual(d.detect(after)["state"], "STEADY")
            self.assertFalse(d.update(d.detect(after), t))

    def test_brighter_reflection_and_unchanged_lit_lamp_never_trigger_or_shift_baseline(self):
        before, brighter = self.images(after_luma=230)
        d = self.detector(before)
        reference = d.reference.copy()
        for i, im in enumerate(
            (before, brighter, brighter, brighter, brighter, before, before, before, before)
        ):
            self.assertEqual(d.detect(im)["state"], "STEADY")
            self.assertFalse(d.update(d.detect(im), i * 0.21))
        np.testing.assert_array_equal(d.reference, reference)

    def test_update_rejects_injected_brighter_direction(self):
        before, after = self.images()
        d = self.detector(before)
        for t in (0.0, 0.21, 0.42, 0.63):
            self.assertFalse(d.update(event("BRIGHTER"), t))
        self.assertEqual(d.hits, 0)
        self.assertFalse(d.latched)
        for i, t in enumerate((0.84, 1.05, 1.26, 1.47)):
            self.assertEqual(d.update(d.detect(after), t), i == 3)

    def test_dark_signal_requires_both_four_frames_and_point_six_seconds(self):
        before, after = self.images()
        d = self.detector(before)
        for t in (0.0, 0.31, 0.62):
            self.assertFalse(d.update(d.detect(after), t))
        self.assertTrue(d.update(d.detect(after), 0.83))
        d = self.detector(before)
        for t in (0.0, 0.1, 0.2, 0.3):
            self.assertFalse(d.update(d.detect(after), t))
        self.assertTrue(d.update(d.detect(after), 0.61))

    def test_dark_sweep_and_short_flash_still_do_not_trigger(self):
        before = np.full_like(scene(), 230)
        d = self.detector(before)
        for i, x in enumerate((128, 140, 152, 160, 148, 136, 128)):
            self.assertFalse(d.update(d.detect(lamp(before, 30, x=x, y=4, w=32, h=24)), i * 0.21))
        before, after = self.images()
        d = self.detector(before)
        for t in (0.0, 0.15, 0.3, 0.45, 0.55):
            self.assertFalse(d.update(d.detect(after), t))
        self.assertFalse(d.update(d.detect(before), 0.7))
        self.assertEqual(d.hits, 0)

    def test_direction_defaults_to_both_and_invalid_values_are_rejected(self):
        self.assertEqual(StartLight().snapshot()["trigger_direction"], "BOTH")
        for direction in ("BOTH", "BRIGHTER", "DARKER"):
            self.assertEqual(
                StartLight(dict(CONFIG, trigger_direction=direction)).snapshot()[
                    "trigger_direction"
                ],
                direction,
            )
        for direction in ("darker", "OFF", "", None, True, []):
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                StartLight(dict(CONFIG, trigger_direction=direction))


if __name__ == "__main__":
    unittest.main()
