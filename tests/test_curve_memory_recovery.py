"""Regressions from real confirmation stops, independent of lane handedness."""

from tests.support import ROOT
import json, unittest
import numpy as np

from robot.navigation import tracker
from robot.control.controller import RoadSupervisor, target_command

CAL = json.loads((ROOT / "maixcam2/curve_follow/calibration.json").read_text())


def line(x0, x1, y):
    return np.column_stack((np.linspace(x0, x1, 60), np.full(60, y)))


class MemoryRecoveryTests(unittest.TestCase):
    def make_tracker(self):
        t = tracker.TargetTracker(CAL)
        self.measurement = dict(path=line(0.12, 0.24, 0), mode="ONE LEFT", confidence=0.81)
        t.pixels.measure = lambda *args: (self.measurement, {})
        self.assertTrue(t.update(None, 0)[0]["observed"])
        return t

    def test_live_stop_trace_recovers_after_crop_shift_without_blind_motion(self):
        # Frames 2607/2610: ~6.7cm shift resets the run; three stable frames
        # then hit the old 0.8s timer. Frame four was rejected too late.
        for sign in (-1, 1):
            for period in (0.16, 0.20, 0.27):
                with self.subTest(sign=sign, period=period):
                    t = self.make_tracker()
                    guard = RoadSupervisor()
                    results = []
                    for i, y in enumerate(
                        (0.0, 0.0, sign * 0.06655, sign * 0.06655, sign * 0.06655, sign * 0.06655),
                        1,
                    ):
                        self.measurement = dict(self.measurement, path=line(0.34, 0.75, y))
                        result, _ = t.update(None, i * period)
                        command = target_command(result)
                        self.assertIsNone(
                            guard.update(
                                result, command, i * period, period * 1000, 280, stop_verified=True
                            )
                        )
                        if not result["observed"]:
                            self.assertEqual(result["state"], "CONFIRMING")
                            self.assertEqual(result["confirmation_required_frames"], 4)
                            self.assertIsNone(command)  # Still zero while old memory expires.
                        results.append(result)
                    self.assertEqual(results[-2]["pending_frames"], 3)
                    self.assertEqual(results[-2]["invalid_reason"], "ROAD SUPPORT EXPIRED")
                    self.assertTrue(results[-1]["observed"])
                    self.assertIsNotNone(command)
                    self.assertEqual(guard.snapshot()["attempt_frames"], 0)

    def test_conflicting_candidates_request_zero_but_can_recover_without_new_start(self):
        t = self.make_tracker()
        guard = RoadSupervisor()
        for i in range(1, 31):
            self.measurement = dict(
                self.measurement, path=line(0.34, 0.75, 0.08 if i % 2 else -0.08)
            )
            r, _ = t.update(None, i * 0.2)
            cmd = target_command(r)
            self.assertIsNone(cmd)
            self.assertIsNone(guard.update(r, cmd, i * 0.2, 200, 280, stop_verified=True))
        self.assertGreater(guard.snapshot()["attempt_frames"], 7)
        for i in range(31, 35):
            self.measurement = dict(self.measurement, path=line(0.34, 0.75, 0.08))
            r, _ = t.update(None, i * 0.2)
            cmd = target_command(r)
            self.assertIsNone(guard.update(r, cmd, i * 0.2, 200, 280, stop_verified=True))
        self.assertTrue(r["observed"])
        self.assertIsNotNone(cmd)
        self.assertEqual(guard.phase, "IDLE")

    def test_true_no_edge_keeps_the_original_missing_observation_deadline(self):
        t = self.make_tracker()
        self.measurement = None
        r, _ = t.update(None, 0.7, 0.05, 0)
        self.assertFalse(r["valid"])
        self.assertEqual(r["state"], "LOST")
        self.assertIsNone(target_command(r))
        guard = RoadSupervisor()
        self.assertIsNone(guard.update(r, None, 0.70, 180, 280, stop_verified=True))
        self.assertEqual(
            guard.update(r, None, 1.06, 180, 280, stop_verified=True), "ROAD NOT FOUND"
        )

    def test_missing_frame_starts_its_own_zero_motion_deadline(self):
        guard = RoadSupervisor()
        for now in (0.0, 0.27, 0.54):
            self.assertIsNone(
                guard.update(
                    dict(
                        state="CONFIRMING",
                        confirmation_required_frames=4,
                        road_evidence="CANDIDATE",
                    ),
                    None,
                    now,
                    270,
                    280,
                    stop_verified=True,
                )
            )
        # A missing frame must not retroactively shrink the whole previous
        # confirmation episode from 0.8s to 0.35s and stop immediately.
        missing = dict(state="LOST", invalid_reason="ROAD SUPPORT EXPIRED", road_evidence="ABSENT")
        self.assertIsNone(guard.update(missing, None, 0.81, 180, 280, stop_verified=True))
        self.assertEqual(
            guard.update(missing, None, 1.17, 180, 280, stop_verified=True), "ROAD NOT FOUND"
        )

    def test_one_slow_frame_cannot_turn_old_confirmation_time_into_immediate_expiry(self):
        guard = RoadSupervisor()
        r = dict(state="CONFIRMING", confirmation_required_frames=4, road_evidence="CANDIDATE")
        for now in (0.0, 0.27, 0.54):
            self.assertIsNone(guard.update(r, None, now, 270, 280, stop_verified=True))
        self.assertIsNone(guard.update(r, None, 0.81, 300, 280, stop_verified=True))
        self.assertIsNone(guard.update(r, None, 1.17, 300, 280, stop_verified=True))
        self.assertEqual(guard.phase, "VISION_WAIT")
        self.assertIsNone(
            guard.update(dict(state="OBSERVED"), (20, 20), 1.3, 150, 280, stop_verified=True)
        )

    def test_interrupted_absence_cannot_be_called_confirmed_road_loss(self):
        guard = RoadSupervisor()
        confirm = dict(
            state="CONFIRMING", confirmation_required_frames=4, road_evidence="CANDIDATE"
        )
        missing = dict(state="LOST", invalid_reason="ROAD SUPPORT EXPIRED", road_evidence="ABSENT")
        for i in range(6):
            r = confirm if i % 2 == 0 else missing
            self.assertIsNone(guard.update(r, None, i * 0.1, 180, 280, stop_verified=True))
        self.assertIsNone(guard.update(confirm, None, 0.6, 180, 280, stop_verified=True))

    def test_invalid_confirmation_contract_fails_closed(self):
        for count in (0, 1, "4", None):
            guard = RoadSupervisor()
            self.assertEqual(
                guard.update(
                    dict(
                        state="CONFIRMING",
                        confirmation_required_frames=count,
                        road_evidence="CANDIDATE",
                    ),
                    None,
                    0,
                    180,
                    280,
                    stop_verified=True,
                ),
                "INVALID CONFIRMATION STATE",
            )

    def test_blind_path_distance_limit_is_unchanged(self):
        t = self.make_tracker()
        self.measurement = None
        r, _ = t.update(None, 0.3, 0.3, 0)
        self.assertFalse(r["valid"])
        self.assertEqual(r["invalid_reason"], "ROAD SUPPORT EXPIRED")
        self.assertGreater(r["blind_distance_m"], 0.05)


if __name__ == "__main__":
    unittest.main()
