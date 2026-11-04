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
                self.assertEqual(c.phase, "FINISH_CLEAR_REAR")
                self.assertFalse(c.complete)
                self.assertFalse(
                    c.update(observation(), motion(0.6), 3 * n + 0.3, True, driving=True)
                )
                self.assertTrue(
                    c.update(observation(), motion(0.3), 3 * n + 3.8, True, driving=True)
                )
            if n < 3:
                self.clear(c, 3 * n + 1.0)
        self.assertTrue(c.complete)
        self.assertEqual(c.phase, "FINISHED")
        for t in (11.0, 12.0, 13.0):
            self.assertFalse(c.update(observation(0.2, True, True), motion(0.3), t, True))
        self.assertEqual(c.count, 3)

    def test_current_runtime_two_returns_then_five_forward_seconds(self):
        config = json.loads((ROOT / "maixcam2/curve_follow/runtime.json").read_text())["race"]
        self.assertEqual((config["laps"], config["finish_runout_s"]), (2, 5.0))
        c = LapCounter(config)
        c.begin(observation(0.4, True, True), 0.0)
        self.assertFalse(self.pass_line(c, 0.0))
        self.assertEqual(c.count, 0)
        self.clear(c, 1.0)
        self.assertFalse(self.pass_line(c, 3.0))
        self.assertEqual(c.count, 1)
        self.clear(c, 4.0)
        self.assertFalse(self.pass_line(c, 6.0))
        self.assertEqual(c.count, 2)
        self.assertEqual(c.phase, "FINISH_CLEAR_REAR")
        self.assertEqual(c.finish_elapsed, 0.0)
        self.assertFalse(c.update(observation(), motion(0.08), 6.3, True, driving=True))
        self.assertEqual(c.phase, "FINISH_CLEAR_REAR")
        self.assertEqual(c.finish_elapsed, 0.0)
        self.assertFalse(c.update(observation(), motion(0.1), 6.5, True, driving=True))
        self.assertEqual(c.phase, "FINISH_RUNOUT")
        self.assertEqual(c.finish_elapsed, 0.0)
        self.assertFalse(c.update(observation(), motion(0.1), 7.5, True, driving=True))
        self.assertFalse(c.update(observation(), motion(), 17.5, True, driving=False))
        self.assertEqual(c.finish_elapsed, 1.0)
        self.assertFalse(
            c.update(observation(0.4, True, True), motion(0.1), 21.49, True, driving=True)
        )
        self.assertTrue(c.update(observation(), motion(0.01), 21.5, True, driving=True))
        self.assertEqual(c.count, 2)
        self.assertEqual(c.finish_elapsed, 5.0)
        self.assertEqual(c.phase, "FINISHED")
        self.assertFalse(
            c.update(observation(0.4, True, True), motion(0.3), 22.0, True, driving=True)
        )
        c.begin(observation(0.4, True, True), 23.0)
        self.assertEqual(c.count, 0)
        self.assertEqual(c.finish_elapsed, 0.0)

    def test_on_site_target_changes_actual_finish_count_and_keeps_five_ack_seconds(self):
        for target in (1, 4):
            with self.subTest(target=target):
                c = LapCounter(dict(enabled=True, laps=2, finish_runout_s=5.0))
                c.prepare(target)
                c.begin(observation(0.4, True, True), 0.0)
                self.assertFalse(self.pass_line(c, 0.0))
                self.assertEqual(c.count, 0)
                self.clear(c, 1.0)
                for lap in range(1, target + 1):
                    self.assertFalse(self.pass_line(c, lap * 3.0))
                    self.assertEqual(c.count, lap)
                    if lap < target:
                        self.assertNotEqual(c.phase, "FINISH_CLEAR_REAR")
                        self.clear(c, lap * 3.0 + 1.0)
                self.assertEqual(c.phase, "FINISH_CLEAR_REAR")
                now = target * 3.0 + 0.5
                self.assertFalse(
                    c.update(
                        observation(), motion(0.3), now, True, driving=True, forward_total_s=100.0
                    )
                )
                self.assertEqual(c.phase, "FINISH_RUNOUT")
                self.assertFalse(
                    c.update(
                        observation(),
                        motion(),
                        now + 50.0,
                        True,
                        driving=True,
                        forward_total_s=104.99,
                    )
                )
                self.assertTrue(
                    c.update(
                        observation(),
                        motion(),
                        now + 50.1,
                        True,
                        driving=True,
                        forward_total_s=105.0,
                    )
                )
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
                self.assertFalse(
                    c.update(observation(), motion(0.6), 3 * n + 0.3, True, driving=True)
                )
                self.assertTrue(
                    c.update(observation(), motion(0.3), 3 * n + 3.8, True, driving=True)
                )
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


class FinishParkingTests(unittest.TestCase):
    def setup_finish(self, front=0.02, rear=0.16, half=0.085):
        c = LapCounter(dict(enabled=True, laps=1, finish_runout_s=3.5), front, rear, half)
        c.begin(observation(0.4, True, True), 0.0)
        h = LapCounterTests()
        h.pass_line(c, 0.0)
        h.clear(c, 1.0)
        self.assertFalse(h.pass_line(c, 3.0))
        self.assertEqual(c.phase, "FINISH_CLEAR_REAR")
        self.assertEqual(c.count, 1)
        self.assertFalse(c.complete)
        return c

    def clear_rear(self, c):
        self.assertFalse(c.update(observation(), motion(0.2), 3.4, True, driving=True))
        self.assertEqual(c.phase, "FINISH_RUNOUT")
        self.assertTrue(c.rear_cleared)
        self.assertEqual(c.finish_elapsed, 0.0)

    def test_three_laps_rear_clearance_and_timer_are_separate(self):
        c = self.setup_finish()
        self.assertFalse(c.rear_cleared)
        self.assertFalse(c.update(observation(), motion(0.08), 3.3, True, driving=True))
        self.assertEqual(c.phase, "FINISH_CLEAR_REAR")
        self.assertEqual(c.finish_elapsed, 0.0)
        self.clear_rear(c)
        self.assertFalse(c.complete)

    def test_stops_at_first_update_after_3_5_driving_seconds(self):
        c = self.setup_finish()
        self.clear_rear(c)
        self.assertFalse(c.update(observation(), motion(0.2), 6.89, True, driving=True))
        self.assertTrue(c.update(observation(), motion(0.01), 6.91, True, driving=True))
        self.assertTrue(c.complete)
        self.assertEqual(c.phase, "FINISHED")
        self.assertEqual(c.event, "FINISH_RUNOUT_ELAPSED")

    def test_paused_time_does_not_reduce_required_driving_time(self):
        c = self.setup_finish()
        self.clear_rear(c)
        self.assertFalse(c.update(observation(), motion(0.1), 4.4, True, driving=True))
        self.assertAlmostEqual(c.finish_elapsed, 1.0)
        self.assertFalse(c.update(observation(), motion(), 20.4, True, driving=False))
        self.assertAlmostEqual(c.finish_elapsed, 1.0)
        self.assertFalse(c.update(observation(), motion(0.1), 22.4, True, driving=True))
        self.assertTrue(c.update(observation(), motion(0.1), 22.9, True, driving=True))

    def test_reverse_or_in_place_commands_do_not_count(self):
        c = self.setup_finish()
        self.clear_rear(c)
        self.assertFalse(c.update(observation(), motion(-0.1), 4.4, True, driving=False))
        self.assertFalse(c.update(observation(), motion(), 5.4, True, driving=False))
        self.assertEqual(c.finish_elapsed, 0.0)

    def test_different_body_lengths_change_rear_clearance_trigger(self):
        short = self.setup_finish(rear=0.10)
        long = self.setup_finish(rear=0.22)
        short.update(observation(), motion(0.1), 3.4, True, driving=True)
        long.update(observation(), motion(0.1), 3.4, True, driving=True)
        self.assertEqual(short.phase, "FINISH_RUNOUT")
        self.assertEqual(long.phase, "FINISH_CLEAR_REAR")

    def test_body_clearance_uncertainty_must_be_accounted_for(self):
        c = self.setup_finish()
        c.release_track["distance"] = -0.165
        c.release_track["sigma"] = 0.02
        self.assertFalse(c.update(observation(), motion(), 3.4, True, driving=True))
        self.assertEqual(c.phase, "FINISH_CLEAR_REAR")
        self.assertFalse(c.update(observation(), motion(0.05), 3.5, True, driving=True))
        self.assertEqual(c.phase, "FINISH_RUNOUT")

    def test_invalid_command_prior_cannot_falsely_prove_rear_crossing(self):
        c = self.setup_finish()
        before = c.release_track["distance"]
        self.assertFalse(c.update(observation(), motion(0.8, valid=False), 3.4, True, driving=True))
        self.assertEqual(c.release_track["distance"], before)
        self.assertFalse(c.update(observation(), motion(), 3.5, True, driving=True))
        self.assertFalse(c.rear_cleared)
        self.assertEqual(c.finish_elapsed, 0.0)

    def test_confirmed_rear_crossing_latches_no_later_odometry_requirement(self):
        c = self.setup_finish()
        self.clear_rear(c)
        before = c.release_track["distance"]
        self.assertTrue(c.update(observation(), motion(0.8, valid=False), 6.9, True, driving=True))
        self.assertEqual(c.release_track["distance"], before)
        self.assertTrue(c.complete)

    def test_uncertain_rear_plane_is_retained(self):
        c = self.setup_finish()
        c.release_track["sigma"] = 0.09
        self.assertFalse(c.update(observation(), motion(), 3.4, True))
        self.assertIsNotNone(c.release_track)
        self.assertEqual(c.phase, "FINISH_CLEAR_REAR")
        self.assertEqual(c.missed, 0)
        self.assertFalse(c.update(observation(), motion(0.9), 3.5, True, driving=True))
        self.assertEqual(c.phase, "FINISH_RUNOUT")

    def test_other_stripes_cannot_replace_final_plane_or_restart_timer(self):
        c = self.setup_finish()
        self.clear_rear(c)
        distance = c.release_track["distance"]
        for t in (3.5, 3.6, 3.7):
            self.assertFalse(
                c.update(observation(0.5, True, True), motion(0.01), t, True, driving=True)
            )
        self.assertEqual(c.release_track["distance"], distance)
        self.assertEqual(c.count, 1)
        self.assertAlmostEqual(c.finish_elapsed, 0.3)

    def test_time_before_rear_crossing_is_not_counted(self):
        c = self.setup_finish()
        c.update(observation(), motion(), 100.0, True, driving=True)
        self.assertEqual(c.finish_elapsed, 0.0)
        self.assertEqual(c.phase, "FINISH_CLEAR_REAR")
        c.update(observation(), motion(0.2), 101.0, True, driving=True)
        self.assertEqual(c.phase, "FINISH_RUNOUT")
        self.assertEqual(c.finish_elapsed, 0.0)

    def test_oblique_vehicle_width_is_used_for_rear_crossing(self):
        c = self.setup_finish()
        n = np.array([math.sqrt(0.5), math.sqrt(0.5)])
        r = c.release_track
        r["normal"] = n
        r["sigma"] = 0.005
        rear_projection = -c.rear * n[0] - c.half * abs(n[1])
        r["distance"] = rear_projection - 0.02
        self.assertFalse(c.update(observation(), motion(), 3.4, True, driving=True))
        self.assertEqual(c.phase, "FINISH_RUNOUT")

    def test_stop_and_new_start_reset_all_finish_state(self):
        c = self.setup_finish()
        self.clear_rear(c)
        c.update(observation(), motion(0.1), 4.4, True, driving=True)
        c.stop()
        self.assertFalse(c.update(observation(), motion(1.0), 9.0, False, driving=True))
        self.assertFalse(c.complete)
        c.begin(observation(0.4, True, True), 10.0)
        self.assertEqual(c.count, 0)
        self.assertFalse(c.rear_cleared)
        self.assertEqual(c.finish_elapsed, 0.0)
        self.assertIsNone(c.finish_last)

    def test_finish_latch_prevents_extra_counts(self):
        c = self.setup_finish()
        self.clear_rear(c)
        self.assertTrue(c.update(observation(), motion(0.6), 6.9, True, driving=True))
        for t in (7.0, 8.0, 9.0):
            self.assertFalse(
                c.update(observation(0.4, True, True), motion(0.4), t, True, driving=True)
            )
        self.assertEqual(c.count, 1)
        self.assertTrue(c.complete)

    def test_finish_configuration_rejects_invalid_time(self):
        for x in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                LapCounter(dict(enabled=True, laps=3, finish_runout_s=x))

    def test_snapshot_shows_driving_countdown_not_35cm_target(self):
        c = self.setup_finish()
        self.clear_rear(c)
        c.update(observation(), motion(0.1), 4.4, True, driving=True)
        s = c.snapshot()
        self.assertEqual(s["finish_runout_s"], 3.5)
        self.assertEqual(s["finish_remaining_s"], 2.5)
        self.assertNotIn("finish_rear_clearance_m", s)


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
