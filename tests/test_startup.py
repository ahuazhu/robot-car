"""Startup readiness and user-facing blockers share the actual control gates."""

from tests.support import ROOT
import unittest
from robot.control.controller import startup_blockers, motor_startup_blockers
from robot.control.controller import assess_vision_command


class StartDiagnosticsTests(unittest.TestCase):
    def check(self, **changes):
        args = dict(
            result=dict(observed=True, confidence=0.8, state="OBSERVED"),
            command=(40, 40),
            consecutive=5,
            detection_ms=200,
            budget_ms=280,
            link_ok=True,
            preview_only=False,
            minimum_confidence=0.65,
        )
        args.update(changes)
        return startup_blockers(**args)

    def test_healthy_start_has_no_blockers(self):
        self.assertEqual(self.check(), [])

    def test_slow_processing_is_reported_separately_from_road_geometry(self):
        self.assertEqual(self.check(command=None, detection_ms=300), ["VISION_TOO_SLOW"])

    def test_stability_and_serial_are_independent_blockers(self):
        self.assertEqual(
            self.check(consecutive=2, link_ok=False), ["UNO_DISCONNECTED", "ROAD_STABILIZING"]
        )

    def test_unobserved_path_confirmation_is_not_reported_as_ready(self):
        reasons = self.check(
            result=dict(observed=False, confidence=0.4, state="CONFIRMING"),
            command=None,
            consecutive=0,
        )
        self.assertIn("PATH_CONFIRMING", reasons)
        self.assertIn("ROAD_UNOBSERVED", reasons)


class MotorStartupReadinessTests(unittest.TestCase):
    def motor(self, **changes):
        state = dict(ack=[8, 0, 0, 0, 0], ack_age_ms=999, armed=False,
                     worker_alive=True, error=None)
        state.update(changes)
        return state

    def test_recent_idle_ack_allows_start(self):
        self.assertEqual(motor_startup_blockers(self.motor()), [])
        self.assertEqual(motor_startup_blockers(self.motor(ack_age_ms=1499)), [])

    def test_motor_faults_and_stale_or_dead_peer_have_specific_reasons(self):
        cases = [
            ({"ack": [8, 0, 0, 0, 4]}, "UNO_EMERGENCY_STOP"),
            ({"ack": [8, 0, 0, 0, 1]}, "UNO_FAULT_1"),
            ({"ack": [8, 0, 0, 0, 2]}, "UNO_FAULT_2"),
            ({"ack": [8, 0, 0, 0, 3]}, "UNO_FAULT_3"),
            ({"ack_age_ms": 1500}, "UNO_STATUS_STALE"),
            ({"ack_age_ms": -1}, "UNO_STATUS_STALE"),
            ({"worker_alive": False}, "UNO_WORKER_STOPPED"),
            ({"error": "UART unavailable"}, "UNO_LINK_ERROR"),
            ({"armed": True}, "UNO_NOT_STOPPED"),
            ({"ack": [8, 1, 20, 20, 0]}, "UNO_NOT_STOPPED"),
        ]
        for changed, reason in cases:
            with self.subTest(changed=changed):
                self.assertEqual(motor_startup_blockers(self.motor(**changed)), [reason])

    def test_missing_or_malformed_motor_evidence_cannot_imply_ready(self):
        self.assertEqual(motor_startup_blockers(None), ["UNO_STATUS_UNAVAILABLE"])
        for changed in ({"ack": []}, {"ack": [1, 0, 0, 0, True]},
                        {"ack": [1, 0, 201, 0, 0]}, {"ack_age_ms": float("nan")}):
            with self.subTest(changed=changed):
                self.assertIn("UNO_STATUS_INVALID", motor_startup_blockers(self.motor(**changed)))


class StartupReadinessTests(unittest.TestCase):
    def test_interleaved_slow_frames_do_not_erase_observed_road_stability(self):
        result = dict(observed=True, confidence=0.8)
        count = 0
        for n, latency in enumerate((225, 310, 240, 301, 222), 1):
            command, count = assess_vision_command(result, (40, 40), latency, 280, count, 0.65)
            self.assertEqual(count, n)
            self.assertEqual(command, None if latency > 280 else (40, 40))
        self.assertGreaterEqual(count, 5)
        self.assertIsNotNone(command)

    def test_slow_current_frame_never_allows_start_or_drive(self):
        command, count = assess_vision_command(
            dict(observed=True, confidence=0.8), (40, 40), 281, 280, 9, 0.65
        )
        self.assertGreaterEqual(count, 5)
        self.assertIsNone(command)

    def test_predicted_weak_or_missing_geometry_resets_stability(self):
        for result, proposed in (
            (dict(observed=False, confidence=0.9), (40, 40)),
            (dict(observed=True, confidence=0.5), (40, 40)),
            (dict(observed=True, confidence=0.9), None),
        ):
            _, count = assess_vision_command(result, proposed, 225, 280, 9, 0.65)
            self.assertEqual(count, 0)

    def test_five_observations_are_still_required(self):
        count = 0
        for _ in range(4):
            _, count = assess_vision_command(
                dict(observed=True, confidence=0.8), (40, 40), 225, 280, count, 0.65
            )
        self.assertLess(count, 5)


if __name__ == "__main__":
    unittest.main()
