"""Lifecycle integration: real task gates, recovery and motor publication priority."""

from tests.support import ROOT
import contextlib
import io
import json
import tempfile
from pathlib import Path
from robot.tasks.start_signal import StartSignal
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
class StartLight:
    """External sensor boundary; detector/protocol behavior is tested separately."""
    available = True

    def __init__(self, config):
        self.enabled = config["enabled"]
        self.armed = self.fired = False
        self.initial_lamp_state = "OFF"
        self.reference = None
        self.baseline_observations = 0

    def arm(self, initial_lamp_state="OFF"):
        if not self.armed:
            self.initial_lamp_state = initial_lamp_state
            self.armed = True
            self.reference = object()
            self.baseline_observations = 1
        return True

    def cancel(self, *, clear_permission=True):
        self.armed = False
        self.reference = None

    def consume(self):
        return True  # FrameDecision.light_start represents a valid peer event here.

    def started(self):
        self.armed, self.fired = False, True

    def snapshot(self):
        return dict(enabled=self.enabled, armed=self.armed, fired=self.fired)



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

    def session(self, *, preview=False, yield_mode=False, finish_priority=False):
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
            RoadSearch(policy, [0.0, 0.0], [0.075, 0.105, 0.085], 0.3,
                       allow_visual_scan=True, finish_priority=finish_priority),
            StartLight({"enabled": True}),
            LapCounter({"enabled": True, "laps": 2, "finish_distance_m": 0.4}),
            CrosswalkTask(crosswalk_policy, finish_priority=finish_priority),
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
            motor_snapshot={"brake_lamp_supported": True, "ack": [0, int(session.active), 0, 0, 0],
                            "armed": session.active, "worker_alive": True,
                            "error": None, "ack_age_ms": 0},
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

    def granted_signal(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        Path(folder.name,'start_signal.local.json').write_text(json.dumps(dict(peer_url='http://10.0.0.2',key='a'*64)))
        peer=dict(boot='a'*16,paired=True,session='',armed=False,confirmed=False,sequence=10,frame_age_ms=10,roi=[96,8,128,112],protocol=2,control_port=81,trigger_direction='DARKER',target_locked=False,target_roi=[96,8,0,0],event_age_ms=0,required_frames=3,required_ms=400,phase='DISARMED')
        def transport(op,session,boot):
            if op=='arm':peer.update(armed=True,session=session,phase='WAIT_OFF')
            if op=='cancel':peer.update(armed=False,confirmed=False,session='',target_locked=False,phase='DISARMED')
            return dict(peer)
        signal=StartSignal({'enabled':True},app_dir=Path(folder.name),clock=lambda:self.clock.now,transport=transport)
        signal._step();self.assertTrue(signal.arm());signal._step()
        peer.update(confirmed=True,sequence=11,target_locked=True,target_roi=[144,48,32,32],confirmation_frames=3,confirmed_ms=420,event_age_ms=420,phase='SIGNAL_READY')
        signal._step();self.assertTrue(signal.permission)
        return signal

    def test_granted_permission_can_start_after_road_delay_and_survive_esp_loss(self):
        s=self.session();signal=self.granted_signal();s.start_light=signal
        signal.transport=lambda *args:(_ for _ in ()).throw(OSError())
        self.clock.now+=30;signal._step()
        self.assertTrue(signal.update(self.clock.now));self.assertFalse(s.active)
        self.warm(s);self.clock.now+=.1
        s.update(self.frame(s,light_start=signal.update(self.clock.now)))
        self.assertTrue(s.active);self.assertTrue(signal.permission)
        signal._step();self.clock.now+=.1;s.update(self.frame(s))
        self.assertTrue(s.active);self.assertTrue(signal.permission)
        self.assertTrue(any(call[0]==2 and call[1]>0 and call[2]>0 for call in s.link.calls))

    def test_manual_stop_and_final_lap_clear_permission_and_cannot_restart(self):
        for action in ('stop','final_lap'):
            with self.subTest(action=action):
                s=self.session();signal=self.granted_signal();s.start_light=signal
                self.warm(s);self.clock.now+=.1;s.update(self.frame(s,light_start=signal.update(self.clock.now)))
                self.assertTrue(s.active)
                self.clock.now+=.1
                if action=='stop':s.update(self.frame(s),('stop',))
                else:s.update(self.frame(s,lap_finished=True))
                self.assertFalse(s.active);self.assertFalse(signal.permission)
                self.clock.now+=1;s.update(self.frame(s,light_start=signal.update(self.clock.now)))
                self.assertFalse(s.active);self.assertFalse(signal.permission)

    def test_motor_fault_stops_motion_but_does_not_erase_signal_permission(self):
        s=self.session();signal=self.granted_signal();s.start_light=signal
        self.warm(s);self.clock.now+=.1;s.update(self.frame(s,light_start=signal.update(self.clock.now)))
        self.assertTrue(s.active);s.stop('MOTOR LINK STOP')
        self.assertFalse(s.active);self.assertTrue(signal.permission)
        self.clock.now+=1;s.update(self.frame(s,light_start=signal.update(self.clock.now)))
        self.assertFalse(s.active);self.assertTrue(signal.permission)
        self.clock.now+=.1;s.update(self.frame(s),('stop',))
        self.assertFalse(signal.permission)

    def test_stop_clears_permission_even_if_uno_stop_ack_fails(self):
        s=self.session();signal=self.granted_signal();s.start_light=signal
        self.warm(s);self.clock.now+=.1;s.update(self.frame(s,light_start=signal.update(self.clock.now)))
        s.link.stop=lambda:(_ for _ in ()).throw(RuntimeError('UNO unavailable'))
        self.clock.now+=.1
        with self.assertRaises(RuntimeError):s.update(self.frame(s),('stop',))
        self.assertFalse(signal.permission);self.assertFalse(s.active)

    def test_recovered_follow_after_search_cancel_still_obeys_crosswalk_stop(self):
        for phase, finish_priority in (
            (phase, priority) for phase in ("STOPPING", "HOLD", "RECOVERY_HOLD")
            for priority in (False, True)
        ):
            with self.subTest(phase=phase, finish_priority=finish_priority):
                s = self.session(finish_priority=finish_priority)
                self.start(s)
                config = json.loads((ROOT / "maixcam2/curve_follow/runtime.json").read_text())
                policy = CrosswalkPolicy.from_config(config["crosswalk"])
                s.settings.crosswalk_policy = policy
                s.crosswalk = CrosswalkTask(policy)
                frame = self.frame(s)
                s.search.command(
                    dict(valid=False, observed=False, state="LOST"), None,
                    frame.details["ground_motion"], self.clock.now, active=True,
                    stopped=True,
                )
                frame.result.update(reacquired=True, pending_frames=0)
                self.clock.now += 0.1
                command = s.search.command(
                    frame.result, (40, 40), frame.details["ground_motion"],
                    self.clock.now, active=True, allowed=False, stopped=True,
                )
                self.assertEqual(command, (40, 40))
                frame.command = command
                s.crosswalk.phase = phase
                before = len(s.link.calls)
                s.update(frame)
                self.assertTrue(s.active)
                self.assertEqual(s.pwm, (0, 0))
                self.assertIsNone(s.recovery.failed)
                self.assertEqual(s.link.calls[before:], [(2, 0, 0, None)])
                self.assertFalse(s.search.is_turning())

    def test_ordinary_predicted_road_publishes_its_own_short_motor_lease(self):
        s = self.session()
        self.start(s)
        self.clock.now += 0.1
        frame = self.frame(s)
        frame.result.update(state="PREDICTED", observed=False)
        s.steering.status["prediction"] = dict(lease_s=0.07)
        s.update(frame)
        self.assertTrue(any(s.pwm))
        self.assertEqual(s.link.calls[-1][3], 0.07)

    def test_confirmed_road_resumes_running_lost_mission_without_second_startup(self):
        for change, resumed in (({}, True), ({"observed": False}, False),
                                ({"pending_frames": 1}, False),
                                ({"observation_confidence": 0.60}, False)):
            with self.subTest(change=change):
                s = self.session()
                self.start(s)
                s.recovery.failed = "ROAD NOT FOUND"
                s.recovery.phase = "LOST_WAIT"
                s.consecutive = 0
                self.clock.now += 0.1
                frame = self.frame(s)
                frame.result.update(dict({"reacquired": True, "pending_frames": 0}, **change))
                s.update(frame)
                self.assertEqual(any(s.pwm), resumed)
                self.assertTrue(s.active)
                self.assertEqual(s.recovery.failed, None if resumed else "ROAD NOT FOUND")

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

    def test_motor_health_blocks_real_start_despite_last_successful_communication(self):
        cases = [
            ({"ack": [9, 0, 0, 0, 4]}, "UNO_EMERGENCY_STOP"),
            ({"ack": [9, 0, 0, 0, 2]}, "UNO_FAULT_2"),
            ({"ack_age_ms": 1500}, "UNO_STATUS_STALE"),
            ({"worker_alive": False}, "UNO_WORKER_STOPPED"),
            ({"error": "UNO acknowledgement timeout"}, "UNO_LINK_ERROR"),
            ({"ack": [9, 1, 40, 40, 0], "armed": True}, "UNO_NOT_STOPPED"),
            ({"ack": []}, "UNO_STATUS_INVALID"),
        ]
        for trigger in ("direct", "visual"):
            for changed, reason in cases:
                with self.subTest(trigger=trigger, changed=changed):
                    s = self.session()
                    self.warm(s)
                    frame = self.frame(s, light_start=trigger == "visual")
                    frame.motor_snapshot.update(changed)
                    self.assertTrue(s.link_ok)
                    readiness = s.readiness(frame)
                    self.assertFalse(readiness["direct_start_ready"])
                    self.assertIn(reason, readiness["blocked_by"])
                    s.update(frame, ("start_now",) if trigger == "direct" else ())
                    self.assertFalse(s.active)
                    self.assertFalse(any(call[0] == 1 for call in s.link.calls))

    def test_healthy_motor_status_recovers_readiness_without_erasing_permission(self):
        s = self.session()
        signal = self.granted_signal()
        s.start_light = signal
        self.warm(s)
        frame = self.frame(s)
        frame.motor_snapshot["ack"][4] = 4
        self.assertFalse(s.readiness(frame)["direct_start_ready"])
        self.assertTrue(signal.permission)
        frame.motor_snapshot["ack"][4] = 0
        self.assertTrue(s.readiness(frame)["direct_start_ready"])
        self.assertTrue(signal.permission)

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

    def test_rolling_search_zero_does_not_restart_acceleration_from_rest(self):
        s = self.session()
        self.start(s)
        for _ in range(8):
            self.clock.now += 0.1
            s.update(self.frame(s, command=(80, 120), follow_command=(80, 120)))
        before = s.ramp.previous
        self.assertEqual(before, (80.0, 120.0))
        lost = dict(valid=False, observed=False, confidence=0.0, state="LOST",
                    mode="LOST", road_evidence="ABSENT")
        moving = dict(valid=True, speed_m_s=0.2, yaw_rad_s=0.0, interval_s=0.1, sigma_m=0.001)
        self.clock.now += 0.1
        zero = s.search.command(lost, None, moving, self.clock.now, active=True, stopped=False)
        self.assertEqual(zero, (0, 0))
        s.update(self.frame(s, result=lost, command=zero, follow_command=None,
                            details={"ground_motion": moving}))
        self.assertEqual(s.pwm, (0, 0))
        self.assertEqual(s.link.calls[-1][1:3], (0, 0), "zero must actually be published")
        self.assertTrue(0 < max(s.ramp.previous) < max(before))
        self.clock.now += 0.1
        frame = self.frame(s, command=(80, 120), follow_command=(80, 120),
                           details={"ground_motion": moving})
        frame.result.update(reacquired=True, pending_frames=0)
        frame.command = s.search.command(frame.result, frame.command, moving, self.clock.now,
                                        active=True, stopped=False)
        s.update(frame)
        self.assertEqual(s.pwm, (80, 120))

    def test_search_zero_resets_restart_effort_when_rest_uncertainty_or_gap_requires_it(self):
        for valid, speed, sigma, gap in ((True, 0, 0.001, .1), (False, .2, .001, .1),
                                         (True, .2, .02, .1), (True, .2, .001, .31)):
            with self.subTest(valid=valid, speed=speed, sigma=sigma, gap=gap):
                s = self.session()
                self.start(s)
                for _ in range(8):
                    self.clock.now += .1
                    s.update(self.frame(s, command=(80, 120), follow_command=(80, 120)))
                lost = dict(valid=False, observed=False, confidence=0.0, state="LOST",
                            mode="LOST", road_evidence="ABSENT")
                moving = dict(valid=valid, speed_m_s=speed, yaw_rad_s=0.0,
                              interval_s=.1, sigma_m=sigma)
                self.clock.now += gap
                zero = s.search.command(lost, None, moving, self.clock.now, active=True)
                s.update(self.frame(s, result=lost, command=zero, follow_command=None,
                                    details={"ground_motion": moving}))
                self.assertEqual(s.pwm, (0, 0))
                self.assertEqual(s.ramp.previous, (0.0, 0.0))

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

    def test_finish_priority_continues_scans_and_manual_stop_ends_retries(self):
        s = self.session(finish_priority=True)
        self.start(s)
        base = self.clock.now + 0.1
        before = len(s.link.calls)
        missing = dict(valid=False, observed=False, confidence=0.0, state="LOST",
                       mode="LOST", road_evidence="ABSENT")
        motion = dict(valid=False, speed_m_s=0.0, yaw_rad_s=0.0)
        for step in range(151):
            self.clock.now = base + step / 10
            command = s.search.command(missing, None, motion, self.clock.now,
                                       active=s.active, stopped=True)
            s.update(self.frame(s, result=missing, details={"ground_motion": motion},
                                command=command, follow_command=None))
        self.assertTrue(s.active)
        self.assertGreaterEqual(s.search.scan.round_index, 3)
        self.assertGreaterEqual(s.search.scan.total_pulses, 8)
        for op, left, right, lease in s.link.calls[before:]:
            if op == 2 and (left or right):
                self.assertEqual(left + right, 0)
                self.assertEqual(lease, 0.12)
        self.clock.now += 0.1
        s.update(self.frame(s), ("stop",))
        self.assertFalse(s.active)
        self.assertEqual(s.pwm, (0, 0))
        self.assertIsNone(s.search.scan.began)

    def test_cautious_path_moves_while_supervisor_rechecks_without_new_start(self):
        s = self.session(finish_priority=True)
        self.start(s)
        s.recovery.failed = "ROAD NOT FOUND"
        s.consecutive = 0
        self.clock.now += 0.1
        frame = self.frame(s)
        frame.result["observation_confidence"] = 0.6
        frame.result["confidence"] = 0.6
        frame.command = s.search.command(frame.result, (100, 120),
                                        frame.details["ground_motion"], self.clock.now,
                                        active=True, stopped=True)
        s.update(frame)
        self.assertTrue(s.active)
        self.assertEqual(s.recovery.phase, "SEARCHING")
        self.assertTrue(any(s.pwm))
        self.assertLessEqual(max(map(abs, s.pwm)), 60)
        self.assertEqual(s.reason, "CAUTIOUS FOLLOW")

    def test_accepted_prediction_exits_scan_with_its_own_lease_and_task_stop_priority(self):
        for phase in (None, 'STOPPING', 'HOLD', 'RECOVERY_HOLD'):
            with self.subTest(phase=phase):
                s = self.session(finish_priority=True)
                self.start(s)
                if phase:
                    config = json.loads((ROOT / 'maixcam2/curve_follow/runtime.json').read_text())
                    s.settings.crosswalk_policy = CrosswalkPolicy.from_config(config['crosswalk'])
                    s.crosswalk = CrosswalkTask(s.settings.crosswalk_policy)
                frame = self.frame(s)
                s.search.command(dict(valid=False, observed=False, state='LOST'), None,
                                 frame.details['ground_motion'], self.clock.now,
                                 active=True, stopped=True)
                self.clock.now += .1
                frame.result.update(state='PREDICTED', observed=False)
                s.steering.status['prediction'] = dict(lease_s=.22)
                frame.command = s.search.command(frame.result, (100, 120),
                    frame.details['ground_motion'], self.clock.now,
                    active=True, allowed=phase is None, stopped=True)
                self.assertTrue(any(frame.command))
                self.assertIsNone(s.search.scan.began)
                if phase:
                    s.crosswalk.phase = phase
                s.update(frame)
                self.assertTrue(s.active)
                if phase:
                    self.assertEqual(s.pwm, (0, 0))
                    self.assertIsNone(s.link.calls[-1][3])
                else:
                    self.assertTrue(any(s.pwm))
                    self.assertEqual(s.link.calls[-1][3], .22)
                    self.assertEqual(s.reason, 'CAUTIOUS FOLLOW')

    def test_unverified_crosswalk_returns_motor_ownership_to_follow_or_search(self):
        fixture=json.loads((ROOT/'tests/fixtures/crosswalk_unverified_20261111.json').read_text())
        config=json.loads((ROOT/'maixcam2/curve_follow/runtime.json').read_text())['crosswalk']
        for mode in ('follow','search'):
            with self.subTest(mode=mode):
                s=self.session(finish_priority=True);self.start(s)
                s.settings.crosswalk_policy=CrosswalkPolicy.from_config(config)
                s.crosswalk=CrosswalkTask(s.settings.crosswalk_policy,finish_priority=True)
                task=s.crosswalk;task.track=dict(fixture['track'],observed_at=self.clock.now)
                task.hits=fixture['hits'];task.identity=1;task.phase='RECOVERY_HOLD'
                task.position_at=task.last_update=self.clock.now
                before=len(s.link.calls);began=self.clock.now
                for step in range(1,42):
                    self.clock.now=began+step*.2
                    frame=self.frame(s)
                    if mode=='search':
                        frame.result.update(valid=False,observed=False,state='LOST',
                                            mode='LOST',road_evidence='ABSENT',confidence=0.)
                    scene=frame.details['ground_motion']
                    scene['travel_m']=0.
                    frame.crosswalk_state=task.update(None,scene,self.clock.now,True,True,True,
                                                      mode=='follow')
                    self.assertEqual(task.phase,'RECOVERY_PASSING')
                    follow=(100,120) if mode=='follow' else None
                    frame.command=s.search.command(frame.result,follow,scene,self.clock.now,
                        active=True,allowed=task.allows_road_search(),stopped=True)
                    frame.follow_command=follow
                    s.update(frame)
                    if step==1 and mode=='follow':
                        self.assertTrue(any(s.link.calls[-1][1:3]))
                self.assertTrue(s.active)
                self.assertEqual(task.phase,'RECOVERY_PASSING')
                self.assertFalse(task.yield_completed);self.assertFalse(task.yield_verified)
                moving=[call for call in s.link.calls[before:] if call[0]==2 and any(call[1:3])]
                self.assertTrue(moving)
                self.assertLessEqual(max(abs(v) for call in moving for v in call[1:3]),97)
                if mode=='search':
                    self.assertTrue(any(call[1]+call[2]==0 and call[3]==.12 for call in moving))
                self.clock.now+=.1;s.update(self.frame(s),('stop',))
                self.assertFalse(s.active);self.assertEqual(s.pwm,(0,0))
                self.assertEqual(task.phase,'SEARCH')

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

    def test_observed_partial_route_keeps_short_lease_through_motor_publication(self):
        from tests.test_persistent_centerline import recorded_calibration,FIXTURE
        from robot.navigation.tracker import TargetTracker
        import cv2
        s=self.session(finish_priority=True);self.start(s)
        cal=recorded_calibration();tracker=TargetTracker(cal)
        result,details=tracker.update(cv2.imread(str(FIXTURE/'raw.png')),self.clock.now)
        s.steering=SteeringControl(tracker.policy,[-.055,0.],[.075,.105,.085],.3,
                                   cruise=135,calibration=cal)
        self.clock.now+=.1
        scene=self.frame(s).details['ground_motion']
        s.steering.observe(scene,self.clock.now,False)
        follow=s.steering.command(result,.1)
        command=s.search.command(result,follow,scene,self.clock.now,active=True,stopped=True)
        frame=self.frame(s,result=result,details=dict(ground_motion=scene),
                         command=command,follow_command=follow)
        s.update(frame)
        self.assertEqual(result['state'],'OBSERVED')
        self.assertTrue(result['partial_boundary_inference'])
        self.assertGreater(s.link.calls[-1][1],s.link.calls[-1][2])
        self.assertGreater(s.link.calls[-1][3],0.)
        self.assertLessEqual(s.link.calls[-1][3],.25)
        self.clock.now+=.1;s.update(self.frame(s),('stop',))
        self.assertFalse(s.active);self.assertEqual(s.pwm,(0,0))

    def test_route_heading_uses_bounded_recovery_then_yields_to_constrained_follow(self):
        s=self.session(finish_priority=True);self.start(s)
        before=len(s.link.calls)
        for _ in range(20):
            self.clock.now+=.1
            frame=self.frame(s)
            frame.result.update(valid=False,observed=False,state='LOST',confidence=.01,
                route_alignment=dict(heading_rad=np.pi/2,distance_m=.1),
                road_model=dict(persistent_history=True))
            frame.follow_command=None
            frame.command=s.search.command(frame.result,None,frame.details['ground_motion'],
                self.clock.now,active=True,allowed=True,stopped=frame.zero_ack)
            s.update(frame)
        pulses=[call for call in s.link.calls[before:] if call[0]==2 and any(call[1:3])]
        self.assertTrue(pulses)
        self.assertTrue(all(call[1]+call[2]==0 and call[3]==.12 for call in pulses))
        self.assertLess(max(abs(v) for call in pulses for v in call[1:3]),97)
        self.clock.now+=.1
        frame=self.frame(s)
        frame.result.update(valid=True,observed=False,state='PREDICTED',confidence=.01,
            route_alignment=dict(heading_rad=np.pi/2,distance_m=.1),
            road_model=dict(persistent_history=True),
            speed_limit_source='CENTERLINE_CONFIDENCE',control_confidence=.01)
        frame.follow_command=(2,5)
        frame.command=s.search.command(frame.result,frame.follow_command,frame.details['ground_motion'],
            self.clock.now,active=True,allowed=True,stopped=frame.zero_ack)
        self.assertEqual(frame.command,frame.follow_command)
        self.assertFalse(s.search.is_turning())
        s.update(frame)
        self.assertGreater(sum(s.pwm),0)
        self.assertGreaterEqual(min(s.pwm),0)
        self.clock.now+=.1;s.update(self.frame(s),('stop',))
        self.assertFalse(s.active);self.assertEqual(s.pwm,(0,0))

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

    def test_measured_finish_completes_and_logs_without_numpy_serialization_crash(self):
        s = self.session()
        self.start(s)
        s.laps.count = s.laps.total
        s.laps.phase = "FINISH_DISTANCE"
        s.laps.release_track = dict(normal=np.array([1., 0.]), distance=-.082, sigma=.036)
        for i, travel in enumerate((.02, .15, .15, .10)):
            self.clock.now += .1
            frame = self.frame(s)
            measured = frame.details["ground_motion"]
            measured.update(translation=np.array([-travel, 0.]), sigma_m=.001)
            frame.lap_finished = s.laps.update({}, measured, self.clock.now, s.active)
            s.update(frame)
            json.dumps(dict(laps=s.laps.snapshot()), allow_nan=False)
            self.assertEqual(s.active, i < 3)
        self.assertEqual(s.reason, "LAPS COMPLETE")
        self.assertEqual(s.pwm, (0, 0))
        self.assertTrue(s.laps.complete)
        self.assertFalse(s.start_light.armed)

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
        self.assertEqual(s.laps.finish_distance, 0.4)
        baseline = s.start_light.reference
        observations = s.start_light.baseline_observations
        self.clock.now += 0.1
        s.update(self.frame(s), (ControlRequest("start", 9),), white)
        self.assertEqual(s.laps.total, 4)
        self.assertTrue(s.start_light.armed)
        self.assertFalse(s.active)
        self.assertIs(s.start_light.reference, baseline)
        self.assertEqual(s.start_light.baseline_observations, observations)

    def test_thirty_second_first_line_window_starts_after_arm_wait_finishes(self):
        s = self.session()
        self.warm(s)
        s.update(self.frame(s), (ControlRequest("start", 2),))
        self.assertTrue(s.start_light.armed)
        self.assertIsNone(s.laps.started_at)
        self.clock.now += 60.
        start_at = self.clock.now
        marker = dict(normal=[1., 0.], distance_m=.4, sigma_m=.005)
        observed = dict(confirmed=True, marker=marker, first_observed_at=start_at-40.)
        s.update(self.frame(s, light_start=True, lap_observation=observed))
        self.assertTrue(s.active)
        self.assertEqual(s.laps.started_at, start_at)
        self.assertEqual(s.laps.first_line_seen_at, start_at)
        self.assertTrue(s.laps.first_line_ignored)

    def test_first_line_window_starts_after_successful_zero_output_arm_ack(self):
        from tests.test_curve_laps import observation, motion
        for first_seen_after_ack, ignored in ((29.9, True), (30.001, False)):
            with self.subTest(first_seen_after_ack=first_seen_after_ack):
                s = self.session()
                self.warm(s)
                frame = self.frame(s)
                captured = frame.captured
                send = s.link.send
                accepted_at = []
                def delayed_ack(op, *positional, **keyword):
                    if op == 1:
                        self.clock.now += .25
                        accepted_at.append(self.clock.now)
                    return send(op, *positional, **keyword)
                s.link.send = delayed_ack
                s.update(frame, ("start_now",))
                self.assertTrue(s.active)
                self.assertEqual(s.laps.started_at, accepted_at[0])
                self.assertGreater(s.laps.started_at, captured)
                s.laps.update(observation(.4, True, True), motion(),
                              accepted_at[0] + first_seen_after_ack, True)
                self.assertEqual(s.laps.first_line_ignored, ignored)

    def test_armed_target_is_used_for_visual_start(self):
        s = self.session()
        self.warm(s)
        s.update(self.frame(s), (ControlRequest("start", 7),), np.zeros((240, 320, 3), np.uint8))
        self.clock.now += 0.1
        s.update(self.frame(s, light_start=True))
        self.assertTrue(s.active)
        self.assertTrue(s.laps.running)
        self.assertEqual(s.laps.total, 7)
        self.assertEqual(s.laps.finish_distance, 0.4)

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
        self.assertEqual(s.laps.finish_distance, 0.4)

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
