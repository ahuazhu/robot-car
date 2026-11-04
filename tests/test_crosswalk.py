from tests.support import ROOT
from tests.support.crosswalk import motion, observation, configured

"""Race requirements, physical-stop evidence and recorded perception regressions."""
import json, unittest
import cv2, numpy as np
from robot.tasks.crosswalk import (
    CrosswalkPolicy,
    CrosswalkDetector,
    CrosswalkTask,
    crosswalk_body_support,
)
from robot.perception.boundaries import CurvePixels

# Recorded images must use their capture-time pose, not today's editable calibration.
CAL = json.loads((ROOT / "tests/fixtures/crosswalk_20261024_calibration.json").read_text())


class CrosswalkRuleTests(unittest.TestCase):
    def test_preview_expiry_does_not_leave_old_crossing_on_screen(self):
        task = CrosswalkTask(CrosswalkPolicy())
        for now in (0, 0.1, 0.2):
            task.update(observation(now), motion(), now)
        self.assertIsNone(task.update(None, motion(), 1.3)["observation"])

    def test_rules_and_incomplete_hardware_configuration(self):
        p = CrosswalkPolicy.from_config({})
        self.assertTrue(p.blockers())
        self.assertEqual(p.mode, "observe")
        for d in (
            {"hold_s": 3},
            {"hold_s": 5},
            {"maximum_hold_s": 5},
            {"stop_clearance_m": 0.2},
            {"mode": "guess"},
            {"unknown": 1},
        ):
            with self.assertRaises(ValueError):
                CrosswalkPolicy.from_config(d)

    def test_whole_body_projection_including_turn(self):
        p = configured()
        self.assertAlmostEqual(crosswalk_body_support([1, 0], p), 0.12)
        self.assertAlmostEqual(crosswalk_body_support([-1, 0], p), 0.13)
        self.assertAlmostEqual(crosswalk_body_support([0.8, 0.6], p), 0.156)

    def test_unconfigured_task_never_authorizes_movement(self):
        task = CrosswalkTask(CrosswalkPolicy.from_config({"mode": "yield"}))
        self.assertEqual(task.limit_command((50, 60), {}, motion()), (0, 0))

    def test_single_flash_does_not_trigger_or_persist(self):
        task = CrosswalkTask(configured())
        s = task.update(observation(0), motion(), 0, True)
        self.assertFalse(s["confirmed"])
        self.assertEqual(s["phase"], "SEARCH")
        self.assertIsNone(task.update(None, motion(), 0.6, True)["observation"])

    def test_observe_mode_does_not_change_drive_command(self):
        task = CrosswalkTask(CrosswalkPolicy())
        self.assertEqual(task.limit_command((40, 60), {}, motion()), (40, 60))

    def test_bad_gap_or_different_crossing_cannot_complete_confirmation(self):
        task = CrosswalkTask(configured())
        for now, near in ((0, 0.5), (0.1, 0.5), (0.7, 0.5), (0.8, 0.7)):
            s = task.update(observation(now, near), motion(), now, True)
        self.assertFalse(s["confirmed"])

    def test_normal_hold_requires_zero_ack_lamp_and_physical_stillness(self):
        for invalid in ("lamp", "ack", "moving", "flow"):
            task = CrosswalkTask(configured())
            for now in (0, 0.1, 0.2):
                task.update(observation(now), motion(), now, True, True, True, True)
            self.assertEqual(task.phase, "STOPPING")
            for now in (0.3, 0.4, 0.5, 0.6):
                s = task.update(
                    observation(now),
                    motion(speed=0.03 if invalid == "moving" else 0, valid=invalid != "flow"),
                    now,
                    True,
                    invalid != "ack",
                    invalid != "lamp",
                    True,
                )
            self.assertNotEqual(s["phase"], "HOLD")
            self.assertEqual(s["hold_elapsed_s"], 0)

    def test_hold_strictly_over_three_seconds_and_passage_rearms_spatially(self):
        task = CrosswalkTask(configured())
        for now in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            s = task.update(observation(now), motion(), now, True, True, True, True)
        self.assertEqual(task.phase, "HOLD")
        start = task.hold_since
        for now in np.arange(0.7, start + 3.29, 0.1):
            task.update(observation(float(now)), motion(), float(now), True, True, True, True)
        self.assertEqual(task.phase, "HOLD")
        s = task.update(observation(start + 3.31), motion(), start + 3.31, True, True, True, True)
        self.assertEqual(task.phase, "PASSING")
        # Fresh frames of this same event cannot stop a second time.
        for j in range(1, 6):
            now = start + 3.31 + j * 0.1
            s = task.update(
                observation(now, 0.27 - 0.02 * j), motion(0.2, 0.02), now, True, False, False, True
            )
            self.assertEqual(task.phase, "PASSING")
            self.assertEqual(task.identity, 1)
        for j in range(6, 31):
            now = start + 3.31 + j * 0.1
            s = task.update(None, motion(0.2, 0.02), now, True, False, False, True)
        self.assertEqual(s["passed"], 1)
        self.assertIsNone(task.track)
        self.assertEqual(s["phase"], "SEARCH")

    def test_stationary_hold_survives_out_of_view_without_refreshing_observation_time(self):
        task = CrosswalkTask(configured())
        for now in (0, 0.1, 0.2):
            task.update(observation(now), motion(), now, True, True, True, True)
        last = task.track["observed_at"]
        for now in np.arange(0.3, 3.91, 0.1):
            s = task.update(None, motion(), float(now), True, True, True, True)
        self.assertEqual(task.phase, "PASSING")
        self.assertEqual(task.track["observed_at"], last)

    def test_lost_motion_cannot_be_replaced_by_pwm_or_stale_pose(self):
        task = CrosswalkTask(configured())
        for now in (0, 0.1, 0.2):
            task.update(observation(now, 0.6), motion(), now, True, False, False, True)
        for now in (0.4, 0.6, 0.8):
            s = task.update(None, motion(valid=False), now, True)
        self.assertEqual(s["phase"], "STOPPING")
        self.assertIsNone(s["failure"])
        self.assertFalse(s["stop_verified"])
        self.assertIn("CROSSWALK POSITION UNCERTAIN", s["issues"])
        self.assertEqual(task.limit_command((50, 50), s, motion(valid=False)), (0, 0))
        self.assertEqual(task.limit_command((50, 60), s, motion()), (0, 0))

    def test_missed_stop_window_and_too_long_unverified_stop_fail_closed(self):
        task = CrosswalkTask(configured())
        for now in (0, 0.1, 0.2):
            s = task.update(observation(now, 0.125), motion(), now, True, True, True, True)
        self.assertIsNone(s["failure"])
        self.assertIn("CROSSWALK STOP WINDOW MISSED", s["issues"])
        self.assertEqual(task.limit_command((50, 50), s, motion()), (0, 0))
        self.assertFalse(s["yield_verified"])
        task = CrosswalkTask(configured())
        for now in (0, 0.1, 0.2, 1.3):
            s = task.update(observation(now), motion(), now, True, False, False, True)
        self.assertIsNone(s["failure"])
        self.assertIn("CROSSWALK STOP UNVERIFIED", s["issues"])
        self.assertEqual(task.limit_command((50, 50), s, motion()), (0, 0))
        self.assertFalse(s["yield_completed"])

    def test_dwell_allows_fresh_zero_without_road_but_never_stale_vision(self):
        task = CrosswalkTask(configured())
        task.phase = "HOLD"
        self.assertEqual(task.supervision_command(None, 200, 280), (0, 0))
        self.assertIsNone(task.supervision_command(None, 281, 280))
        task.phase = "PASSING"
        self.assertIsNone(task.supervision_command(None, 200, 280))

    def test_approach_scales_wheels_together_preserving_curvature(self):
        task = CrosswalkTask(configured())
        for now in (0, 0.1, 0.2):
            s = task.update(observation(now, 0.5), motion(0.2), now, True)
        command = task.limit_command((40, 60), s, motion(0.2))
        self.assertLess(command[1], 60)
        self.assertLessEqual(abs(command[0] / 40 - command[1] / 60), 0.025)

    def test_zero_speed_feedback_never_restores_full_cruise(self):
        task = CrosswalkTask(configured(stop_clearance_m=0.08))
        for now in (0, 0.1, 0.2):
            state = task.update(observation(now, 0.5), motion(), now, True)
        first = task.limit_command((53, 53), state, motion())
        self.assertLess(first[0], 20)
        state = task.update(observation(0.3, 0.5), motion(), 0.3, True)
        second = task.limit_command((53, 53), state, motion())
        self.assertLess(second[0], 25)
        self.assertGreaterEqual(second[0], first[0])

    def test_coasting_time_is_derived_from_measured_speed(self):
        task = CrosswalkTask(configured(brake_deceleration_m_s2=0.08))
        for now in (0, 0.1, 0.2):
            task.update(observation(now, 0.35), motion(0.24), now, True)
        self.assertEqual(task.phase, "STOPPING")
        state = task.update(observation(1.3, 0.27), motion(0.02), 1.3, True, True, True, True)
        self.assertIsNone(state["failure"])
        self.assertEqual(state["hold_elapsed_s"], 0)

    def test_early_stop_is_not_counted_as_a_legal_yield(self):
        task = CrosswalkTask(configured(brake_deceleration_m_s2=0.08))
        for now in (0, 0.1, 0.2):
            task.update(observation(now, 0.5), motion(0.24), now, True)
        self.assertEqual(task.phase, "STOPPING")
        state = task.update(observation(0.4, 0.5), motion(), 0.4, True, True, True, True)
        self.assertEqual(state["phase"], "APPROACH")
        self.assertEqual(state["hold_elapsed_s"], 0)

    def test_prestart_observation_expires_but_latched_fault_does_not(self):
        task = CrosswalkTask(configured())
        for now in (0, 0.1, 0.2):
            task.update(observation(now), motion(), now)
        self.assertIsNone(task.update(None, motion(), 2.0)["observation"])
        task._fail("TEST FAULT")
        task.update(observation(2.1), motion(), 2.1)
        self.assertEqual(task.update(None, motion(), 4.0)["failure"], "TEST FAULT")

    def test_short_passage_flow_gap_does_not_trigger_another_stop(self):
        task = CrosswalkTask(configured())
        for now in (0, 0.1, 0.2):
            task.update(observation(now), motion(), now)
        task.phase = "PASSING"
        state = task.update(None, motion(valid=False), 0.3, True)
        self.assertIsNone(state["failure"])
        self.assertEqual(state["phase"], "PASSING")

    def test_served_crossing_uncertainty_blocks_rearming_not_lane_following(self):
        task = CrosswalkTask(configured())
        for now in (0, 0.1, 0.2):
            task.update(observation(now), motion(), now)
        task.phase = "PASSING"
        identity = task.identity
        for now in (0.7, 1.2, 1.7):
            state = task.update(None, motion(valid=False), now, True)
        self.assertGreater(state["position_error_m"], 2 * task.policy.maximum_position_error_m)
        self.assertEqual(state["phase"], "PASSING")
        self.assertIsNone(state["failure"])
        self.assertEqual(state["passed"], 0)
        self.assertEqual(task.identity, identity)
        self.assertEqual(task.limit_command((40, 50), state, motion()), (40, 50))
        # A fresh road frame is still required by the independent lane supervisor.
        self.assertIsNone(task.supervision_command(None, 100, 280))
        # A later visual displacement may prove conservative full-body passage even
        # with a wider interval than a legal stopping position would allow.
        state = task.update(None, motion(0.3, 0.9), 1.8, True)
        self.assertEqual(state["passed"], 1)
        self.assertIsNone(task.track)

    def test_passing_timer_records_stationary_dwell_not_elapsed_driving(self):
        task = CrosswalkTask(configured())
        for now in np.arange(0, 4.01, 0.1):
            state = task.update(
                observation(float(now)), motion(), float(now), True, True, True, True
            )
        self.assertEqual(state["phase"], "PASSING")
        duration = state["hold_elapsed_s"]
        state = task.update(None, motion(), 4.2, True)
        self.assertEqual(state["hold_elapsed_s"], duration)
        self.assertGreaterEqual(duration, 3.3)

    def test_missing_speed_requests_zero_and_requires_observed_recovery(self):
        task = CrosswalkTask(configured())
        for now in (0, 0.1, 0.2):
            state = task.update(observation(now, 0.6), motion(), now, True)
        self.assertEqual(task.limit_command((50, 60), state, motion(valid=False)), (0, 0))
        self.assertIsNone(task.failure)
        task.last_update = 0.4
        self.assertNotEqual(task.limit_command((50, 60), state, motion(0.02)), (0, 0))
        self.assertIsNone(task.speed_missing_since)
        task.last_update = 0.5
        task.limit_command((50, 60), state, motion(valid=False))
        task.last_update = 1.6
        self.assertEqual(task.limit_command((50, 60), state, motion(valid=False)), (0, 0))
        self.assertIsNone(task.failure)
        self.assertIn("CROSSWALK SPEED UNVERIFIED", task.issues)
        task.last_update = 1.7
        self.assertNotEqual(task.limit_command((50, 60), state, motion(0.02)), (0, 0))

    def test_closed_loop_with_coasting_and_motor_deadzone(self):
        for friction in (0.2, 0.4, 0.8):
            for deadzone in (0, 18, 30):
                task = CrosswalkTask(
                    configured(
                        body_front_m=0.02,
                        body_rear_m=0.16,
                        body_half_width_m=0.08,
                        brake_deceleration_m_s2=0.08,
                        stop_clearance_m=0.08,
                        approach_speed_m_s=0.06,
                        speed_per_pwm_m_s=0.006705385281385282,
                    )
                )
                speed = 0.15
                traveled = 0.0
                ack = (53, 53)
                hold = []
                dt = 0.15
                for i in range(280):
                    now = i * dt
                    near = 0.65 - traveled
                    scene = motion(speed, speed * dt)
                    scene["sigma_m"] = 0.0015
                    obs = observation(now, near) if 0.09 <= near <= 0.43 else None
                    state = task.update(obs, scene, now, True, ack == (0, 0), ack == (0, 0), True)
                    command = task.limit_command((53, 53), state, scene) or (0, 0)
                    self.assertIsNone(task.failure, (friction, deadzone, state))
                    if task.phase == "HOLD":
                        hold.append((now, near - 0.02))
                    desired = max(0.0, (command[0] - deadzone) * 0.0045) if command[0] else 0.0
                    new_speed = speed + max(-friction * dt, min(0.4 * dt, desired - speed))
                    traveled += (speed + new_speed) * dt / 2
                    speed = new_speed
                    ack = command
                    if state["passed"]:
                        break
                self.assertEqual(task.passed, 1, (friction, deadzone))
                self.assertTrue(hold)
                self.assertTrue(all(0 < distance < 0.20 for _, distance in hold))
                self.assertGreaterEqual(hold[-1][0] - hold[0][0] + dt, 3.3 - 1e-8)

    def test_completed_hold_waits_for_road_without_latching_a_task_fault(self):
        task = CrosswalkTask(configured())
        for now in np.arange(0, 4.81, 0.1):
            s = task.update(observation(float(now)), motion(), float(now), True, True, True, False)
        self.assertEqual(task.phase, "PASSING")
        self.assertTrue(s["yield_verified"])
        s = task.update(observation(5.0), motion(), 5.0, True, True, True, False)
        self.assertIsNone(s["failure"])
        self.assertIsNone(task.limit_command(None, s, motion()))


class CrosswalkPerceptionTests(unittest.TestCase):
    def setUp(self):
        self.pixels = CurvePixels(CAL)
        self.detector = CrosswalkDetector(self.pixels, CrosswalkPolicy())

    def detect_raw(self, name):
        raw = cv2.imread(str(ROOT / "tests/fixtures/crosswalk_recordings_20261024" / name))
        self.assertIsNotNone(raw, "Missing regression fixture: " + name)
        measure, details = self.pixels.measure(raw)
        return self.detector.detect(details, measure or {}, 0.0)

    def test_real_seven_bar_crossing_including_near_field(self):
        for frame in (96, 97, 98, 99, 100):
            obs = self.detect_raw("frame-%05d.png" % frame)
            self.assertIsNotNone(obs, frame)
            self.assertGreaterEqual(obs["bar_count"], 4)

    def test_actual_road_without_crossing_and_single_transverse_line(self):
        for frame in (0, 10, 40, 65, 120, 155):
            self.assertIsNone(self.detect_raw("frame-%05d.png" % frame), frame)

    def test_rotated_metric_bars_and_partial_occlusion(self):
        grid = self.pixels
        xx, yy = np.meshgrid(grid.xs, grid.ys, indexing="ij")
        for angle, count in ((0.0, 7), (0.3, 7), (-0.3, 5)):
            direction = np.array([np.cos(angle), np.sin(angle)])
            normal = np.array([-direction[1], direction[0]])
            bird = np.full((len(grid.xs), len(grid.ys), 3), 100, np.uint8)
            for shift in np.linspace(-0.12, 0.12, 7)[:count]:
                center = np.array([0.4, 0.0]) + shift * normal
                along = (xx - center[0]) * direction[0] + (yy - center[1]) * direction[1]
                across = (xx - center[0]) * normal[0] + (yy - center[1]) * normal[1]
                bird[(abs(along) < 0.05) & (abs(across) < 0.01)] = 255
            road = dict(
                path=np.array([0.4, 0.0]) + np.linspace(-0.3, 0.3, 30)[:, None] * direction,
                confidence=1.0,
            )
            obs = self.detector.detect(dict(bird=bird), road, 0)
            self.assertIsNotNone(obs, (angle, count))
            self.assertGreaterEqual(obs["bar_count"], 4)

    def test_synthetic_solid_line_is_not_zebra(self):
        grid = self.pixels
        bird = np.full((len(grid.xs), len(grid.ys), 3), 100, np.uint8)
        bird[(grid.xs > 0.4) & (grid.xs < 0.42)] = 255
        road = dict(path=np.c_[np.linspace(0.1, 0.8, 25), np.zeros(25)], confidence=1.0)
        self.assertIsNone(self.detector.detect(dict(bird=bird), road, 0))


if __name__ == "__main__":
    unittest.main()
