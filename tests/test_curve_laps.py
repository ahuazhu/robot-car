"""White-line identity, departure suppression, spatial crossing and stop latch."""

from tests.support import ROOT
import json, math, unittest
import cv2, numpy as np
from robot.tasks.lap_counter import LapMarkerDetector, LapCounter
from robot.perception.boundaries import CurvePixels
from robot.core.geometry import geometry_frame

FIX = ROOT / "tests/fixtures/lap_marker_20261031"


def marker(d=0.4, qualified=True):
    return dict(
        center=[d - 0.02, 0.0],
        normal=[1.0, 0.0],
        distance_m=d,
        sigma_m=0.005,
        qualified=qualified,
        support=1.0,
        contrast=60.0,
    )


def observation(d=None, gate=False, ready=False):
    m = marker(d) if d is not None else None
    return dict(
        stripes=[m] if m else [], marker_visible=gate, confirmed=ready, marker=m if ready else None
    )


def motion(travel=0.0, valid=True, sigma=0.001):
    return dict(
        rotation=np.eye(2), translation=np.array([-travel, 0.0]), valid=valid, sigma_m=sigma
    )


class LapCounterTests(unittest.TestCase):
    def setup_counter(self):
        c = LapCounter(dict(enabled=True, laps=3))
        c.begin(observation(0.4, True, True), 0.0)
        return c

    def pass_line(self, c, at):
        if c.track is None:
            c.update(observation(0.4, True, True), motion(), at, True)
        c.update(observation(0.2, True), motion(0.2), at + 0.1, True)
        return c.update(observation(), motion(0.24), at + 0.2, True)

    def clear(self, c, at):
        c.update(observation(), motion(0.6), at, True)
        c.update(observation(), motion(), at + 0.7, True)

    def test_initial_crossing_not_a_lap_and_exactly_three_returns_stop(self):
        c = self.setup_counter()
        self.assertFalse(self.pass_line(c, 0.0))
        self.assertEqual(c.count, 0)
        self.assertEqual(c.event, "DEPARTURE_SKIPPED")
        self.clear(c, 1.0)
        for n in (1, 2, 3):
            done = self.pass_line(c, 3.0 * n)
            self.assertEqual(c.count, n)
            self.assertFalse(done)
            if n == 3:
                self.assertEqual(c.phase, "FINISH_DISTANCE")
                self.assertFalse(c.complete)
                self.assertTrue(c.update(observation(), motion(0.41), 3 * n + 0.3, True))
            if n < 3:
                self.clear(c, 3 * n + 1.0)
        self.assertTrue(c.complete)
        self.assertEqual(c.phase, "FINISHED")
        for t in (11.0, 12.0, 13.0):
            self.assertFalse(c.update(observation(0.2, True, True), motion(0.3), t, True))
        self.assertEqual(c.count, 3)

    def test_current_runtime_counts_two_returns_then_measured_forty_centimetres(self):
        config = json.loads((ROOT / "maixcam2/curve_follow/runtime.json").read_text())["race"]
        self.assertEqual((config["laps"], config["finish_distance_m"]), (2, 0.4))
        self.assertNotIn("finish_runout_s", config)
        c = LapCounter(config)
        c.begin(observation(0.4, True, True), 0.0)
        self.assertFalse(self.pass_line(c, 0.0))
        self.clear(c, 1.0)
        self.assertFalse(self.pass_line(c, 3.0))
        self.clear(c, 4.0)
        self.assertFalse(self.pass_line(c, 6.0))
        self.assertEqual((c.count, c.phase, c.finish_travel), (2, "FINISH_DISTANCE", 0.0))
        self.assertFalse(c.update(observation(), motion(0.2), 6.3, True))
        self.assertFalse(c.update(observation(), motion(), 100.0, True))
        self.assertTrue(c.update(observation(), motion(0.21), 100.1, True))
        self.assertEqual((c.count, c.phase), (2, "FINISHED"))
        c.begin(observation(0.4, True, True), 101.0)
        self.assertEqual((c.count, c.finish_travel), (0, 0.0))

    def test_on_site_target_changes_actual_finish_count_and_keeps_distance_rule(self):
        for target in (1, 4):
            with self.subTest(target=target):
                c = LapCounter(dict(enabled=True, laps=2, finish_distance_m=0.4))
                c.prepare(target)
                c.begin(observation(0.4, True, True), 0.0)
                self.assertFalse(self.pass_line(c, 0.0))
                self.clear(c, 1.0)
                for lap in range(1, target + 1):
                    self.assertFalse(self.pass_line(c, lap * 3.0))
                    self.assertEqual(c.count, lap)
                    if lap < target:
                        self.assertNotEqual(c.phase, "FINISH_DISTANCE")
                        self.clear(c, lap * 3.0 + 1.0)
                self.assertEqual(c.phase, "FINISH_DISTANCE")
                self.assertFalse(c.update(observation(), motion(0.39, sigma=0), 100.0, True))
                self.assertTrue(c.update(observation(), motion(0.01, sigma=0), 100.1, True))
                self.assertEqual((c.count, c.total, c.phase), (target, target, "FINISHED"))

    def test_repeated_visible_line_or_flicker_does_not_create_more_laps(self):
        c = self.setup_counter()
        for i in range(20):
            self.assertFalse(c.update(observation(0.4, True, True), motion(), i * 0.1, True))
        self.assertEqual(c.count, 0)
        self.pass_line(c, 3.0)
        for i in range(20):
            c.update(observation(0.2, True, True), motion(), 4 + i * 0.1, True)
        self.assertEqual(c.count, 0)
        self.assertTrue(c.visit_done)

    def test_unconfirmed_white_candidates_cannot_establish_a_visit(self):
        c = self.setup_counter()
        self.pass_line(c, 0.0)
        self.clear(c, 1.0)
        for i, d in enumerate((0.5, 0.3, 0.1, 0.06)):
            c.update(observation(d), motion(0.1), 3 + i * 0.1, True)
        self.assertEqual(c.count, 0)
        self.assertIsNone(c.track)

    def test_clearance_is_not_blocked_by_unrelated_white_stripes(self):
        c = self.setup_counter()
        self.pass_line(c, 0.0)
        c.update(observation(0.5), motion(0.6), 1.0, True)
        c.update(observation(0.4), motion(), 1.7, True)
        self.assertEqual(c.phase, "SEEK_RETURN")
        self.assertFalse(c.departing)

    def test_line_occlusion_cannot_rearm_before_whole_body_passes(self):
        c = self.setup_counter()
        self.pass_line(c, 0.0)
        c.update(observation(), motion(), 1.0, True)
        c.update(observation(), motion(), 2.0, True)
        self.assertTrue(c.visit_done)
        self.assertEqual(c.phase, "WAIT_CLEAR")
        self.assertTrue(c.departing)

    def test_uncertain_initial_marker_cannot_turn_into_a_return_during_occlusion(self):
        c = self.setup_counter()
        c.update(observation(), motion(sigma=0.09), 0.1, True)
        c.update(observation(), motion(), 1.0, True)
        c.update(observation(), motion(), 2.0, True)
        self.assertEqual(c.count, 0)
        self.assertTrue(c.departing)
        self.assertTrue(c.visit_done)

    def test_current_invalid_motion_or_uncertain_plane_cannot_count(self):
        for valid, sigma in ((False, 0.001), (True, 0.09)):
            c = self.setup_counter()
            self.pass_line(c, 0.0)
            self.clear(c, 1.0)
            c.update(observation(0.2, True, True), motion(), 3.0, True)
            c.update(observation(), motion(0.3, valid, sigma), 3.1, True)
            self.assertEqual(c.count, 0)

    def test_confirmed_far_plane_can_count_after_measured_crossing_without_near_view(self):
        c = self.setup_counter()
        self.pass_line(c, 0.0)
        self.clear(c, 1.0)
        c.update(observation(0.5, True, True), motion(), 3.0, True)
        c.update(observation(), motion(0.6), 3.1, True)
        self.assertEqual(c.count, 1)
        self.assertEqual(c.missed, 0)

    def test_reverse_motion_moves_line_away_instead_of_counting(self):
        c = self.setup_counter()
        c.update(observation(), motion(-0.3), 0.1, True)
        self.assertGreater(c.track["distance"], 0.6)
        self.assertEqual(c.count, 0)

    def test_second_nearby_white_stripe_cannot_replace_the_tracked_plane(self):
        c = self.setup_counter()
        c.update(observation(0.2), motion(), 0.1, True)
        self.assertAlmostEqual(c.track["distance"], 0.4)

    def test_stop_retains_display_and_new_start_resets_a_new_race(self):
        c = self.setup_counter()
        self.pass_line(c, 0.0)
        self.clear(c, 1.0)
        self.pass_line(c, 3.0)
        self.assertEqual(c.count, 1)
        c.stop()
        self.assertFalse(c.update(observation(0.1, True, True), motion(0.2), 4.0, False))
        self.assertEqual(c.count, 1)
        c.begin(observation(0.4, True, True), 5.0)
        self.assertEqual(c.count, 0)
        self.assertTrue(c.departing)

    def test_start_without_visible_marker_seeks_departure_then_counts_three_returns(self):
        c = LapCounter(dict(enabled=True, laps=3))
        c.begin(observation(), 0.0)
        self.assertTrue(c.running)
        self.assertEqual(c.phase, "SEEK_DEPARTURE")
        self.assertEqual(c.count, 0)
        for t in (0.1, 0.2, 0.3):
            self.assertFalse(c.update(observation(), motion(), t, True))
            self.assertEqual(c.count, 0)
        self.assertFalse(c.update(observation(0.4, True, True), motion(), 0.4, True))
        self.assertEqual(c.phase, "DEPARTURE")
        self.assertIsNotNone(c.track)
        self.assertFalse(self.pass_line(c, 1.0))
        self.assertEqual(c.count, 0)
        self.assertEqual(c.event, "DEPARTURE_SKIPPED")
        self.clear(c, 2.0)
        for n in (1, 2, 3):
            self.assertFalse(self.pass_line(c, 3 * n))
            self.assertEqual(c.count, n)
            if n == 3:
                self.assertTrue(c.update(observation(), motion(0.41), 3 * n + 0.3, True))
            if n < 3:
                self.clear(c, 3 * n + 1.0)
        self.assertTrue(c.complete)

    def test_unconfirmed_lines_while_seeking_departure_never_count(self):
        c = LapCounter(dict(enabled=True, laps=3))
        c.begin(observation(), 0.0)
        for i, d in enumerate((0.5, 0.3, 0.1, 0.01)):
            c.update(observation(d), motion(0.1), i * 0.1, True)
            self.assertIsNone(c.track)
            self.assertEqual(c.count, 0)
            self.assertEqual(c.phase, "SEEK_DEPARTURE")

    def test_disabled_counter_never_requests_a_stop(self):
        c = LapCounter({})
        c.begin({}, 0.0)
        self.assertFalse(c.update(observation(0.1, True, True), motion(0.5), 0.1, True))


class FirstWhiteLineWindowTests(unittest.TestCase):
    def counter(self, laps=2, start=100.):
        c = LapCounter(dict(enabled=True, laps=laps, initial_line_ignore_s=30., finish_distance_m=.4))
        c.begin(observation(), start)
        return c

    def test_first_line_at_or_before_thirty_seconds_is_skipped_after_crossing(self):
        for delay in (0., 29.9, 30.):
            with self.subTest(delay=delay):
                c = self.counter()
                LapCounterTests().pass_line(c, 100.+delay)
                self.assertEqual(c.count, 0)
                self.assertEqual(c.event, "DEPARTURE_SKIPPED")
                self.assertTrue(c.first_line_ignored)

    def test_first_line_after_thirty_seconds_counts_first_lap_and_next_counts_second(self):
        c = self.counter()
        h = LapCounterTests()
        self.assertFalse(h.pass_line(c, 130.001))
        self.assertEqual(c.count, 1)
        self.assertFalse(c.first_line_ignored)
        self.assertEqual(c.event, "LAP_COMPLETED")
        h.clear(c, 131.)
        self.assertFalse(h.pass_line(c, 140.))
        self.assertEqual((c.count, c.phase), (2, "FINISH_DISTANCE"))
        self.assertTrue(c.update(observation(), motion(.41), 140.3, True))
        json.dumps(c.snapshot(), allow_nan=False)

    def test_first_visit_before_deadline_remains_ignored_when_crossing_after_deadline(self):
        c = self.counter()
        c.update(observation(.4, True, True), motion(), 129.9, True)
        LapCounterTests().pass_line(c, 131.)
        self.assertEqual(c.count, 0)
        self.assertEqual(c.first_line_seen_at, 129.9)
        self.assertTrue(c.first_line_ignored)

    def test_consistent_detection_before_deadline_is_retained_until_confirmation(self):
        c = self.counter()
        obs = dict(observation(.4, True, True), first_observed_at=129.9)
        c.update(obs, motion(), 130.1, True)
        LapCounterTests().pass_line(c, 131.)
        self.assertEqual(c.count, 0)
        self.assertAlmostEqual(c.snapshot()["first_line_seen_after_start_s"], 29.9)

    def test_unconfirmed_flash_and_time_alone_never_create_a_lap_or_lock_skip(self):
        c = self.counter()
        c.update(observation(.4), motion(), 120., True)
        c.update(observation(), motion(), 131., True)
        self.assertEqual(c.count, 0)
        self.assertIsNone(c.first_line_ignored)
        LapCounterTests().pass_line(c, 140.)
        self.assertEqual(c.count, 1)

    def test_late_first_line_with_single_lap_target_still_requires_forty_cm(self):
        c = self.counter(laps=1)
        self.assertFalse(LapCounterTests().pass_line(c, 131.))
        self.assertEqual((c.count, c.phase), (1, "FINISH_DISTANCE"))
        self.assertFalse(c.update(observation(), motion(.39, sigma=0), 132., True))
        self.assertTrue(c.update(observation(), motion(.01, sigma=0), 132.1, True))

    def test_pre_start_detector_history_is_clamped_and_new_race_resets_window(self):
        c = self.counter()
        obs = dict(observation(.4, True, True), first_observed_at=50.)
        c.begin(obs, 100.)
        self.assertEqual(c.first_line_seen_at, 100.)
        self.assertTrue(c.first_line_ignored)
        c.stop()
        c.begin(observation(), 300.)
        self.assertIsNone(c.first_line_seen_at)
        LapCounterTests().pass_line(c, 331.)
        self.assertEqual(c.count, 1)

    def test_invalid_window_cannot_enable_unsafe_implicit_counting(self):
        for window in (-1, 0, True, "30", float("inf"), float("nan")):
            with self.subTest(window=window), self.assertRaises(ValueError):
                LapCounter(dict(enabled=True, initial_line_ignore_s=window))



class FinishParkingTests(unittest.TestCase):
    def setup_finish(self, **kwargs):
        c = LapCounter(dict(enabled=True, laps=1, finish_distance_m=0.4), **kwargs)
        c.begin(observation(0.4, True, True), 0.0)
        h = LapCounterTests()
        h.pass_line(c, 0.0)
        h.clear(c, 1.0)
        self.assertFalse(h.pass_line(c, 3.0))
        self.assertEqual((c.phase, c.count, c.finish_travel), ("FINISH_DISTANCE", 1, 0.0))
        return c

    def test_crossing_frame_does_not_backfill_pre_line_travel(self):
        c = self.setup_finish()
        self.assertFalse(c.update(observation(), motion(0.39, sigma=0), 3.3, True))
        self.assertTrue(c.update(observation(), motion(0.01, sigma=0), 3.4, True))
        self.assertEqual(c.event, "FINISH_DISTANCE_REACHED")
        self.assertTrue(c.rear_cleared)

    def test_no_elapsed_time_pwm_or_missing_flow_can_create_distance(self):
        c = self.setup_finish()
        self.assertFalse(c.update(observation(), motion(0.2, sigma=0), 3.3, True))
        self.assertFalse(c.update(observation(), None, 300.0, True))
        self.assertFalse(c.update(observation(), motion(sigma=0), 400.0, True))
        self.assertEqual(c.finish_travel, 0.2)
        self.assertTrue(c.update(observation(), motion(0.2, sigma=0), 400.1, True))

    def test_reverse_undoes_measured_progress(self):
        c = self.setup_finish()
        c.update(observation(), motion(0.3, sigma=0), 3.3, True)
        c.update(observation(), motion(-0.1, sigma=0), 3.4, True)
        self.assertFalse(c.update(observation(), motion(0.1, sigma=0), 3.5, True))
        self.assertAlmostEqual(c.finish_travel, 0.3)
        self.assertTrue(c.update(observation(), motion(0.1, sigma=0), 3.6, True))

    def test_chassis_rotation_does_not_count_camera_arc_as_forward_travel(self):
        origin = np.array([-0.055, 0.0])
        c = self.setup_finish(reference_origin_m=origin)
        angle = 0.2
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        spin = dict(valid=True, rotation=rotation, translation=origin-rotation@origin, sigma_m=0)
        for i in range(10):
            self.assertFalse(c.update(observation(), spin, 3.3 + i * 0.1, True))
        self.assertAlmostEqual(c.finish_travel, 0.0)

    def test_distance_lower_bound_accounts_for_measurement_uncertainty(self):
        c = self.setup_finish()
        self.assertFalse(c.update(observation(), motion(0.4, sigma=0.01), 3.3, True))
        self.assertAlmostEqual(c.snapshot()["finish_remaining_m"], 0.02)
        self.assertTrue(c.update(observation(), motion(0.021, sigma=0), 3.4, True))

    def test_command_prior_never_counts_as_measured_progress(self):
        c = self.setup_finish()
        before = c.release_track["distance"]
        self.assertFalse(c.update(observation(), motion(0.8, valid=False), 3.3, True))
        self.assertEqual(c.finish_travel, 0.0)
        self.assertEqual(c.release_track["distance"], before)
        self.assertFalse(c.snapshot()["finish_motion_verified"])
        self.assertFalse(c.update(observation(), motion(0.4), 3.4, True))

    def test_other_stripes_do_not_replace_the_final_line_or_restart_distance(self):
        c = self.setup_finish()
        for i in range(3):
            self.assertFalse(c.update(observation(0.5, True, True), motion(0.1), 3.3+i*.1, True))
        self.assertEqual(c.count, 1)
        self.assertAlmostEqual(c.finish_travel, 0.3)
        self.assertTrue(c.update(observation(), motion(0.11), 3.6, True))

    def test_stop_new_start_and_finish_latch(self):
        c = self.setup_finish()
        c.update(observation(), motion(0.2), 3.3, True)
        c.stop()
        self.assertFalse(c.update(observation(), motion(1.0), 3.4, False))
        self.assertFalse(c.complete)
        c.begin(observation(0.4, True, True), 5.0)
        self.assertEqual((c.count, c.finish_travel), (0, 0.0))
        c = self.setup_finish()
        self.assertTrue(c.update(observation(), motion(0.41), 3.3, True))
        self.assertFalse(c.update(observation(0.4, True, True), motion(1.0), 3.4, True))
        self.assertEqual((c.count, c.phase), (1, "FINISHED"))

    def test_finish_configuration_rejects_invalid_distance_and_origin(self):
        for value in (-0.1, 0, float("nan"), float("inf"), True, "0.4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LapCounter(dict(enabled=True, laps=1, finish_distance_m=value))
        with self.assertRaises(ValueError):
            LapCounter({}, reference_origin_m=[float("nan"), 0])

    def test_nonfinite_motion_cannot_turn_distance_comparison_into_early_finish(self):
        for valid in (True, False):
            with self.subTest(valid=valid):
                c = self.setup_finish()
                c.update(observation(), motion(0.3, sigma=0), 3.3, True)
                self.assertFalse(c.update(observation(), motion(float("nan"), valid=valid), 3.4, True))
                self.assertFalse(c.update(observation(), motion(0.1, sigma=0), 3.5, True))
                self.assertFalse(c.complete)

    def test_every_finish_frame_is_strict_json_before_and_after_rear_clearance(self):
        c = self.setup_finish()
        seen = set()
        for i, travel in enumerate((0.02, 0.15, 0.15, 0.10)):
            done = c.update(observation(), motion(travel), 3.3+i*.1, True)
            snapshot = c.snapshot()
            self.assertIs(type(snapshot["rear_line_cleared"]), bool)
            seen.add(snapshot["rear_line_cleared"])
            decoded = json.loads(json.dumps(dict(laps=snapshot), allow_nan=False))
            self.assertEqual(decoded["laps"]["complete"], done)
            if i < 3:
                self.assertFalse(done)
        self.assertEqual(seen, {False, True})
        self.assertTrue(c.complete)
        self.assertEqual(c.phase, "FINISHED")

    def test_snapshot_exposes_distance_without_old_timer(self):
        c = self.setup_finish()
        c.update(observation(), motion(0.2, sigma=0), 3.3, True)
        s = c.snapshot()
        self.assertEqual((s["finish_distance_m"], s["finish_remaining_m"]), (0.4, 0.2))
        self.assertNotIn("finish_runout_s", s)
        self.assertNotIn("finish_elapsed_s", s)


class RecordedMarkerTests(unittest.TestCase):
    def detector(self):
        cal = json.loads((FIX / "calibration.json").read_text())
        config = json.loads((FIX / "config.json").read_text())
        return LapMarkerDetector(CurvePixels(cal), config)

    def result(self, d, im):
        with geometry_frame():
            road, details = d.pixels.measure(im)
        self.assertIsNotNone(road)
        for t in (0.0, 0.15, 0.3):
            r = d.detect(im, details, road, t)
        return r

    def test_first_detection_time_survives_confirmation_and_resets_on_loss(self):
        d = self.detector()
        candidates = iter([[marker()], [marker()], [marker()], [], [marker()], [marker()], [marker()]])
        d.stripes = lambda *args: next(candidates)
        for now in (29.8, 29.9, 30.1):
            result = d.detect(None, {}, {}, now)
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["first_observed_at"], 29.8)
        result = d.detect(None, {}, {}, 30.2)
        self.assertIsNone(result["first_observed_at"])
        for now in (30.3, 30.4, 30.5):
            result = d.detect(None, {}, {}, now)
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["first_observed_at"], 30.3)

    def test_actual_confirmed_stripe_selected_and_nearer_reflection_rejected(self):
        d = self.detector()
        r = self.result(d, cv2.imread(str(FIX / "start.png")))
        self.assertTrue(r["confirmed"])
        self.assertAlmostEqual(r["marker"]["center"][0], 0.42, delta=0.02)
        self.assertEqual(len(r["stripes"]), 1, "near reflection must not be a marker")

    def test_white_stripe_is_still_confirmed_without_the_removed_gantry(self):
        d = self.detector()
        im = cv2.imread(str(FIX / "start.png"))
        im[:148] = 190
        r = self.result(d, im)
        self.assertTrue(r["marker_visible"])
        self.assertTrue(r["confirmed"])

    def test_one_frame_line_flash_is_not_enough_to_establish_a_visit(self):
        d = self.detector()
        im = cv2.imread(str(FIX / "start.png"))
        with geometry_frame():
            road, details = d.pixels.measure(im)
        self.assertFalse(d.detect(im, details, road, 0.0)["confirmed"])


class WhiteLineRegressionTests(unittest.TestCase):
    def detector(self, cal=None):
        cal = cal or json.loads(
            (ROOT / "tests/fixtures/lap_line_20261031/calibration.json").read_text()
        )
        return LapMarkerDetector(CurvePixels(cal), dict(enabled=True, laps=3))

    def test_all_current_recorded_lines_survive_without_a_gantry_or_full_navigation_curve(self):
        d = self.detector()
        for i in range(4):
            im = cv2.imread(str(ROOT / f"tests/fixtures/lap_line_20261031/line-{i}.png"))
            with geometry_frame():
                road, details = d.pixels.measure(im)
            self.assertIsNotNone(road)
            r = d.detect(im, details, road, i * 0.2)
            self.assertTrue(r["marker_visible"])
            self.assertEqual(len(r["stripes"]), 1)
            self.assertAlmostEqual(r["stripes"][0]["center"][0], 0.48, delta=0.025)
            self.assertGreaterEqual(r["stripes"][0]["support"], 0.75)
        self.assertTrue(r["confirmed"])

    def test_real_bend_and_real_seven_bar_crosswalk_do_not_count_as_a_line(self):
        for folder, name in (
            ("lap_line_20261031", "bend.png"),
            ("crosswalk_20261030", "seven_bars.png"),
        ):
            f = ROOT / "tests/fixtures" / folder
            d = self.detector(json.loads((f / "calibration.json").read_text()))
            im = cv2.imread(str(f / name))
            with geometry_frame():
                road, details = d.pixels.measure(im)
            self.assertIsNotNone(road)
            for t in (0.0, 0.2, 0.4):
                r = d.detect(im, details, road, t)
            self.assertFalse(r["confirmed"], (folder, r))
            self.assertFalse(r["marker_visible"])

    def test_line_connected_to_longitudinal_edges_survives_and_bars_do_not(self):
        d = self.detector()
        g = d.pixels
        xx, yy = np.meshgrid(g.xs, g.ys, indexing="ij")
        road = dict(path=np.c_[np.linspace(0.25, 0.7, 40), np.zeros(40)])
        for kind in ("line", "partial", "edges", "zebra", "red", "blank"):
            value = np.full(g.visible.shape, 100, np.uint8)
            sat = np.zeros(value.shape, np.uint8)
            if kind == "line":
                paint = (abs(xx - 0.45) < 0.012) | ((abs(yy - 0.10) < 0.009) & (xx > 0.3))
            elif kind == "partial":
                paint = (abs(xx - 0.45) < 0.012) & (abs(yy) < 0.05)
            elif kind == "edges":
                paint = abs(abs(yy) - 0.10) < 0.009
            elif kind == "zebra":
                paint = (xx > 0.4) & (xx < 0.5) & (np.mod(yy + 0.14, 0.04) < 0.02)
            elif kind == "red":
                paint = abs(xx - 0.45) < 0.012
                sat[paint] = 200
            else:
                paint = np.zeros(value.shape, bool)
            value[paint] = 245
            details = dict(
                bird=np.repeat(value[:, :, None], 3, axis=2),
                value=value,
                contrast_value=value,
                saturation=sat,
            )
            result = d.stripes(details, road)
            self.assertEqual(bool(result), kind == "line", (kind, result))

    def test_consistency_compensates_motion_and_tolerates_half_second_frames(self):
        d = self.detector()
        observations = iter([marker(0.7), marker(0.52), marker(0.34)])
        d.stripes = lambda *a: [next(observations)]
        for i, t in enumerate((0.0, 0.5, 1.0)):
            r = d.detect(None, {}, dict(), t, motion=motion(0.18 if i else 0.0))
        self.assertTrue(r["confirmed"])
        self.assertEqual(r["confirmation_frames"], 3)

    def test_duplicate_lock_releases_after_body_passes_line_without_a_gantry(self):
        c = LapCounterTests().setup_counter()
        helper = LapCounterTests()
        helper.pass_line(c, 0.0)
        c.update(observation(), motion(0.2), 1.0, True)
        c.update(observation(), motion(), 1.7, True)
        self.assertEqual(c.phase, "SEEK_RETURN")
        self.assertFalse(c.departing)

    def test_shadow_and_exposure_do_not_change_current_marker_identity(self):
        raw = cv2.imread(str(ROOT / "tests/fixtures/lap_line_20261031/line-0.png"))
        for gain in (0.55, 0.75, 1.0, 1.15):
            d = self.detector()
            im = np.clip(raw.astype(float) * gain, 0, 255).astype(np.uint8)
            with geometry_frame():
                road, details = d.pixels.measure(im)
            self.assertIsNotNone(road)
            for t in (0.0, 0.2, 0.4):
                r = d.detect(im, details, road, t)
            self.assertTrue(r["confirmed"], (gain, r))
            self.assertAlmostEqual(r["marker"]["center"][0], 0.48, delta=0.03)


if __name__ == "__main__":
    unittest.main()
