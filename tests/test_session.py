"""Lifecycle integration: real task gates, recovery and motor publication priority."""

from tests.support import ROOT
import contextlib
import io
from types import SimpleNamespace
import unittest
import numpy as np
from robot.control.session import DrivingSession, FrameDecision
from robot.control.requests import ControlRequest
from robot.control.controller import CommandRamp, RoadSupervisor
from robot.core.road_policy import RoadPolicy
from robot.interface.preview_transport import WebControl
from robot.navigation.steering_control import SteeringControl
from robot.navigation.road_recovery import RoadSearch
from robot.tasks.crosswalk import CrosswalkPolicy, CrosswalkTask
from robot.tasks.lap_counter import LapCounter
from robot.tasks.start_light import StartLight


class MotorPeer:
    """External UNO boundary; every publication and lease is observable."""

    def __init__(self):
        self.armed = False
        self.calls = []
        self.fail_drive = False

    def send(self, op, left=0, right=0, *, lease_s=None):
        self.calls.append((op, left, right, lease_s))
        if op == 2 and self.fail_drive:
            raise RuntimeError("UART unavailable")
        if op == 0:
            self.armed = False
        if op == 1:
            self.armed = True
        return [len(self.calls), int(self.armed), left, right, 0]

    def stop(self):
        return self.send(0)


class DrivingSessionTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.clock = SimpleNamespace(now=10.0)

    def tearDown(self):
        self.output.__exit__(None, None, None)

    def session(self, *, preview=False, yield_mode=False):
        policy = RoadPolicy()
        crosswalk_policy = CrosswalkPolicy(mode="yield" if yield_mode else "observe")
        settings = SimpleNamespace(
            calibrated=True,
            preview_only=preview,
            vision_budget_ms=280,
            pwm_boost=37,
            road_policy=policy,
            crosswalk_policy=crosswalk_policy,
        )
        session = DrivingSession(
            settings,
            MotorPeer(),
            WebControl(),
            SteeringControl(policy, [0.0, 0.0], [0.075, 0.105, 0.085], 0.3),
            RoadSearch(policy, [0.0, 0.0], [0.075, 0.105, 0.085], 0.3, allow_visual_scan=True),
            StartLight({"enabled": True}),
            LapCounter({"enabled": True, "laps": 2, "finish_runout_s": 5.0}),
            CrosswalkTask(crosswalk_policy),
            CommandRamp(),
            RoadSupervisor(),
            clock=lambda: self.clock.now,
        )
        session.connect()
        return session

    def frame(self, session, **changes):
        result = dict(
            valid=True,
            observed=True,
            confidence=0.8,
            observation_confidence=0.8,
            state="OBSERVED",
            mode="ONE LEFT",
            road_evidence="OBSERVED",
        )
        motion = dict(
            valid=True,
            speed_m_s=0.0,
            yaw_rad_s=0.0,
            interval_s=0.1,
            sigma_m=0.0,
            rotation=np.eye(2),
            translation=np.zeros(2),
        )
        values = dict(
            result=result,
            details={"ground_motion": motion},
            captured=self.clock.now,
            command=(40, 40),
            follow_command=(40, 40),
            processing_ms=100.0,
            lap_observation={"confirmed": False, "marker": None},
            lap_finished=False,
            motor_snapshot={"brake_lamp_supported": True, "ack": [0, int(session.active), 0, 0, 0]},
            zero_ack=True,
            light_start=False,
            crosswalk_state=session.crosswalk.snapshot(self.clock.now),
        )
        values.update(changes)
        return FrameDecision(**values)

    def warm(self, session):
        for _ in range(5):
            self.clock.now += 0.1
            session.update(self.frame(session))

    def start(self, session):
        self.warm(session)
        self.clock.now += 0.1
        session.update(self.frame(session), ("start_now",))
        self.assertTrue(session.active)

    def test_stop_wins_over_simultaneous_start_visual_signal_and_finish(self):
        s = self.session()
        self.start(s)
        before = len(s.link.calls)
        self.clock.now += 0.1
        s.update(
            self.frame(s, light_start=True, lap_finished=True),
            ("stop", "start", "start_now"),
            np.zeros((240, 320, 3), np.uint8),
        )
        self.assertFalse(s.active)
        self.assertEqual(s.pwm, (0, 0))
        self.assertEqual(s.reason, "STOPPED")
        self.assertFalse(s.start_light.armed)
        self.assertFalse(s.laps.running)
        self.assertFalse(
            any(
                op == 1 or (op == 2 and (left or right))
                for op, left, right, _ in s.link.calls[before:]
            )
        )

    def test_arm_can_observe_without_road_but_direct_start_cannot_move(self):
        s = self.session()
        result = dict(
            valid=False,
            observed=False,
            confidence=0.0,
            state="LOST",
            mode="LOST",
            road_evidence="ABSENT",
        )
        s.update(
            self.frame(s, result=result, command=None),
            ("start",),
            np.zeros((240, 320, 3), np.uint8),
        )
        self.assertTrue(s.start_light.armed)
        self.assertFalse(s.active)
        self.clock.now += 0.1
        s.update(self.frame(s, result=result, command=None), ("start_now",))
        self.assertFalse(s.active)
        self.assertFalse(any(c[0] == 1 for c in s.link.calls))

    def test_visual_and_direct_start_share_success_and_all_readiness_gates(self):
        for trigger in ("visual", "direct"):
            for blocked in (None, "weak", "slow", "disconnected", "preview", "yield_setup"):
                with self.subTest(trigger=trigger, blocked=blocked):
                    s = self.session(
                        preview=blocked == "preview", yield_mode=blocked == "yield_setup"
                    )
                    self.warm(s)
                    self.clock.now += 0.1
                    frame = self.frame(s, light_start=trigger == "visual")
                    if blocked == "weak":
                        frame.result["confidence"] = 0.1
                    if blocked == "slow":
                        frame.processing_ms = 281.0
                    if blocked == "disconnected":
                        s.link_ok = False
                    s.update(frame, ("start_now",) if trigger == "direct" else ())
                    self.assertEqual(s.active, blocked is None)
                    self.assertEqual(any(c[0] == 1 for c in s.link.calls), blocked is None)
                    if blocked is None:
                        self.assertTrue(s.laps.running)
                        self.assertTrue(s.start_light.fired)

    def test_slow_active_frame_publishes_zero_and_later_timely_frame_recovers(self):
        s = self.session()
        self.start(s)
        self.clock.now += 0.1
        s.update(self.frame(s, processing_ms=281.0))
        self.assertTrue(s.active)
        self.assertEqual(s.pwm, (0, 0))
        self.assertEqual(s.link.calls[-1][1:3], (0, 0))
        self.clock.now += 0.1
        s.update(self.frame(s))
        self.assertTrue(any(s.pwm))
        self.assertEqual(sum(c[0] == 1 for c in s.link.calls), 1)

    def test_missing_road_can_scan_and_exhaustion_waits_at_zero_without_rearming(self):
        s = self.session()
        self.start(s)
        base = self.clock.now + 0.1
        missing = dict(
            valid=False,
            observed=False,
            confidence=0.0,
            state="LOST",
            mode="LOST",
            road_evidence="ABSENT",
        )
        motion = dict(valid=False, speed_m_s=0.0, yaw_rad_s=0.0)
        for elapsed in (0.0, 0.01, 0.76, 6.1, 6.5):
            self.clock.now = base + elapsed
            command = s.search.command(
                missing, None, motion, self.clock.now, active=True, stopped=True
            )
            s.update(
                self.frame(
                    s,
                    result=missing,
                    details={"ground_motion": motion},
                    command=command,
                    follow_command=None,
                )
            )
            if elapsed == 0.76:
                self.assertTrue(any(s.pwm))
                self.assertEqual(sum(s.pwm), 0)
                self.assertEqual(s.link.calls[-1][3], 0.12)
        self.assertTrue(s.active)
        self.assertEqual(s.pwm, (0, 0))
        self.assertEqual(s.reason, "ROAD NOT FOUND")
        self.assertEqual(sum(c[0] == 1 for c in s.link.calls), 1)

    def test_motor_publication_failure_ends_session_and_cancels_waiting(self):
        s = self.session()
        self.start(s)
        s.link.fail_drive = True
        self.clock.now += 0.1
        s.update(self.frame(s))
        self.assertFalse(s.active)
        self.assertFalse(s.link_ok)
        self.assertEqual(s.pwm, (0, 0))
        self.assertEqual(s.reason, "MOTOR LINK STOP")
        self.assertFalse(s.start_light.armed)
        self.assertEqual(s.link.calls[-1][0], 0)

    def test_confirmed_finish_stops_an_active_session_and_does_not_restart(self):
        s = self.session()
        self.start(s)
        self.clock.now += 0.1
        s.update(self.frame(s, lap_finished=True))
        self.assertFalse(s.active)
        self.assertEqual(s.reason, "LAPS COMPLETE")
        self.assertEqual(s.pwm, (0, 0))
        self.clock.now += 0.1
        s.update(self.frame(s))
        self.assertFalse(s.active)
        self.assertEqual(sum(c[0] == 1 for c in s.link.calls), 1)

    def test_arm_selects_target_resets_old_race_and_does_not_rebaseline_when_repeated(self):
        s = self.session()
        s.laps.count, s.laps.complete, s.laps.phase = 2, True, "FINISHED"
        black = np.zeros((240, 320, 3), np.uint8)
        white = np.full_like(black, 255)
        s.update(self.frame(s), (ControlRequest("start", 4),), black)
        self.assertEqual(s.laps.total, 4)
        self.assertEqual(s.laps.count, 0)
        self.assertFalse(s.laps.complete)
        self.assertFalse(s.laps.running)
        self.assertEqual(s.laps.finish_seconds, 5.0)
        baseline = s.start_light.reference.copy()
        observations = s.start_light.baseline_observations
        self.clock.now += 0.1
        s.update(self.frame(s), (ControlRequest("start", 9),), white)
        self.assertEqual(s.laps.total, 4)
        self.assertTrue(s.start_light.armed)
        self.assertFalse(s.active)
        np.testing.assert_array_equal(s.start_light.reference, baseline)
        self.assertEqual(s.start_light.baseline_observations, observations)

    def test_armed_target_is_used_for_visual_start(self):
        s = self.session()
        self.warm(s)
        s.update(self.frame(s), (ControlRequest("start", 7),), np.zeros((240, 320, 3), np.uint8))
        self.clock.now += 0.1
        s.update(self.frame(s, light_start=True))
        self.assertTrue(s.active)
        self.assertTrue(s.laps.running)
        self.assertEqual(s.laps.total, 7)
        self.assertEqual(s.laps.finish_seconds, 5.0)

    def test_direct_start_changes_target_and_resets_progress_only_after_readiness(self):
        s = self.session()
        s.laps.count, s.laps.phase = 1, "STOPPED"
        frame = self.frame(s)
        frame.result["confidence"] = 0.1
        s.update(frame, (ControlRequest("start_now", 7),))
        self.assertFalse(s.active)
        self.assertEqual((s.laps.total, s.laps.count, s.laps.phase), (2, 1, "STOPPED"))
        self.warm(s)
        self.clock.now += 0.1
        s.update(self.frame(s), (ControlRequest("start_now", 7),))
        self.assertTrue(s.active)
        self.assertTrue(s.laps.running)
        self.assertEqual((s.laps.total, s.laps.count), (7, 0))

    def test_armed_direct_start_rejects_changed_target_then_accepts_same_or_unspecified(self):
        for target in (4, None):
            with self.subTest(target=target):
                s = self.session()
                self.warm(s)
                s.update(
                    self.frame(s), (ControlRequest("start", 4),), np.zeros((240, 320, 3), np.uint8)
                )
                self.clock.now += 0.1
                s.update(self.frame(s), (ControlRequest("start_now", 5),))
                self.assertFalse(s.active)
                self.assertTrue(s.start_light.armed)
                self.assertEqual(s.laps.total, 4)
                self.assertFalse(any(call[0] == 1 for call in s.link.calls))
                self.clock.now += 0.1
                s.update(self.frame(s), (ControlRequest("start_now", target),))
                self.assertTrue(s.active)
                self.assertEqual(s.laps.total, 4)

    def test_running_start_requests_cannot_change_target_or_reset_progress(self):
        s = self.session()
        self.warm(s)
        s.update(self.frame(s), (ControlRequest("start_now", 7),))
        s.laps.count = 3
        self.clock.now += 0.1
        s.update(self.frame(s), (ControlRequest("start", 4), ControlRequest("start_now", 9)))
        self.assertTrue(s.active)
        self.assertTrue(s.laps.running)
        self.assertEqual((s.laps.total, s.laps.count), (7, 3))
        self.assertEqual(sum(call[0] == 1 for call in s.link.calls), 1)

    def test_stop_unlocks_arm_target_for_a_new_selection(self):
        s = self.session()
        black = np.zeros((240, 320, 3), np.uint8)
        s.update(self.frame(s), (ControlRequest("start", 4),), black)
        self.clock.now += 0.1
        s.update(self.frame(s), (ControlRequest("start", 9), ControlRequest("stop", "bad")), black)
        self.assertFalse(s.start_light.armed)
        self.assertEqual(s.laps.total, 4)
        self.assertEqual(s.reason, "STOPPED")
        self.clock.now += 0.1
        s.update(self.frame(s), (ControlRequest("start", 9),), black)
        self.assertTrue(s.start_light.armed)
        self.assertEqual(s.laps.total, 9)

    def test_touch_start_without_target_retains_last_selected_target_after_stop(self):
        s = self.session()
        self.warm(s)
        s.update(self.frame(s), (ControlRequest("start_now", 7),))
        self.clock.now += 0.1
        s.update(self.frame(s), ("stop",))
        self.assertFalse(s.laps.running)
        self.start(s)
        self.assertEqual(s.laps.total, 7)
        self.assertEqual(s.laps.count, 0)
        self.assertEqual(s.laps.finish_seconds, 5.0)

    def test_stop_wins_over_target_change_and_preserves_finished_record(self):
        s = self.session()
        self.start(s)
        s.laps.count, s.laps.complete, s.laps.phase = 2, True, "FINISHED"
        self.clock.now += 0.1
        s.update(
            self.frame(s, light_start=True, lap_finished=True),
            (ControlRequest("start_now", 7), ControlRequest("stop", {"invalid": True})),
        )
        self.assertFalse(s.active)
        self.assertFalse(s.laps.running)
        self.assertEqual((s.laps.total, s.laps.count, s.laps.phase), (2, 2, "FINISHED"))
        self.assertTrue(s.laps.complete)
        self.assertEqual(s.reason, "STOPPED")
        self.assertEqual(s.pwm, (0, 0))


if __name__ == "__main__":
    unittest.main()
