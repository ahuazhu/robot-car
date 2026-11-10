"""The operator's initial lamp state is immutable ARM metadata, never NVS."""
import dataclasses
import json
import unittest
from unittest.mock import patch

from tests import test_esp_start_signal as peer_tests
from tests import test_lamp_configuration as configuration_tests
from tests import test_session as session_tests
from robot.control.requests import ControlRequest


class InitialLampStateTests(unittest.TestCase):
    def setUp(self):
        self.case = peer_tests.SignalPeerTests()
        self.case.setUp()
        self.addCleanup(self.case.tearDown)
        self.client, self.remote = self.case.client, self.case.remote
        self.original = self.client.transport
        self.arm_modes = []

    def new_peer(self, op, session, boot, *, initial_lamp_state="OFF"):
        if op == "arm":
            self.arm_modes.append(initial_lamp_state)
            self.remote["initial_lamp_state"] = initial_lamp_state
        return self.original(op, session, boot)

    def test_default_off_keeps_legacy_transport_and_peer_compatible(self):
        self.assertTrue(self.client.arm())
        self.client._step()
        self.assertTrue(self.client.acknowledged)
        self.assertEqual(self.client.snapshot()["initial_lamp_state"], "OFF")
        self.assertFalse(self.client.permission)

    def test_on_is_sent_once_and_frozen_across_retries_and_permission(self):
        self.client.transport = self.new_peer
        self.assertTrue(self.client.arm("ON"))
        nonce = self.client.session
        self.assertTrue(self.client.arm("ON"))
        self.assertFalse(self.client.arm("OFF"))
        self.assertEqual(self.client.session, nonce)
        # A dropped response means the same ARM must be retried unchanged.
        def dropped(op, *args, **kwargs):
            self.new_peer(op, *args, **kwargs)
            raise OSError("lost reply")
        self.client.transport = dropped
        self.client._step()
        self.assertTrue(self.client.armed)
        self.assertEqual(self.client.session, nonce)
        self.client.transport = self.new_peer
        self.client._step()
        self.assertEqual(self.arm_modes, ["ON", "ON"])
        self.assertTrue(self.client.acknowledged)
        self.case.confirm()
        self.assertTrue(self.client.permission)
        self.client.consume()
        self.client.transport = lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
        self.client._step()
        self.assertTrue(self.client.permission)
        self.assertEqual(self.client.snapshot()["initial_lamp_state"], "ON")
        self.assertFalse(self.client.arm("OFF"))
        self.client.cancel()
        self.assertFalse(self.client.permission)

    def test_explicit_on_cannot_silently_use_legacy_off_or_mismatched_poll(self):
        self.client.transport = lambda op, session, boot, **kwargs: self.original(op, session, boot)
        self.assertTrue(self.client.arm("ON"))
        self.client._step()
        self.assertEqual(self.client.phase, "FAULT")
        self.assertEqual(self.client.fault_reason, "INITIAL_LAMP_STATE_MISMATCH")
        self.assertFalse(self.client.permission)

    def test_poll_mode_change_cancels_only_unconfirmed_session(self):
        self.client.transport = self.new_peer
        self.client.arm("ON"); self.client._step()
        self.remote["initial_lamp_state"] = "OFF"
        self.client._step()
        self.assertEqual(self.client.phase, "FAULT")
        self.assertFalse(self.client.permission)

    def test_invalid_values_cannot_change_a_session_but_stop_remains_unconditional(self):
        for value in (None, False, 0, [], {}, "on", "off", ""):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "invalid_initial_lamp_state"):
                    self.client.arm(value)
                with self.assertRaisesRegex(ValueError, "invalid_initial_lamp_state"):
                    ControlRequest("start", 2, value)
                self.assertEqual(ControlRequest("stop", initial_lamp_state=value).action, "stop")
        self.assertEqual(self.client.session, "")

    def test_late_arm_reply_after_stop_does_not_restore_permission_or_mode(self):
        def late(op, session, boot, **kwargs):
            peer = self.new_peer(op, session, boot, **kwargs)
            self.client.cancel()
            return dict(peer, confirmed=True, target_locked=True, target_roi=[144,48,32,32],
                        confirmation_frames=4, confirmed_ms=630)
        self.client.transport = late
        self.client.arm("ON"); self.client._step()
        self.assertFalse(self.client.armed)
        self.assertFalse(self.client.permission)

    def test_http_sends_mode_only_with_arm(self):
        captured = []
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size): return b"{}"
        class Opener:
            def open(self, request, **kwargs):
                captured.append(json.loads(request.data))
                return Response()
        self.client.endpoint = "http://10.0.0.2:81"
        with patch("robot.tasks.start_signal.build_opener", return_value=Opener()):
            self.client._http("arm", "a"*32, "b"*16, initial_lamp_state="ON")
            self.client._http("poll", "a"*32, "b"*16)
            self.client._http("cancel", "a"*32, "b"*16)
        self.assertEqual(captured[0]["initial_lamp_state"], "ON")
        self.assertNotIn("initial_lamp_state", captured[1])
        self.assertNotIn("initial_lamp_state", captured[2])


class PreparedInitialLampStateTests(unittest.TestCase):
    def setUp(self):
        self.case = configuration_tests.LampConfigurationTests()
        self.case.setUp()
        self.addCleanup(self.case.tearDown)
        self.client, self.control = self.case.client, self.case.control

    def test_prepare_captures_on_with_laps_and_delivers_it_to_actual_session(self):
        with patch("robot.interface.preview_transport.time.monotonic", lambda:self.case.now):
            self.assertTrue(self.control.submit("start", 0, laps=7,
                lamp_configuration=self.case.manual(), initial_lamp_state="ON")[0])
            self.assertEqual(self.control.status()["preparation"]["initial_lamp_state"], "ON")
            self.assertEqual(self.client.snapshot()["initial_lamp_state"], "OFF")
            self.client._step(); self.case.publish()
            request = self.control.take()
            self.assertEqual((request.laps, request.initial_lamp_state), (7, "ON"))
            with self.assertRaises(dataclasses.FrozenInstanceError):
                request.initial_lamp_state = "OFF"
            case = session_tests.DrivingSessionTests(); case.setUp()
            try:
                session = case.session(); session.start_light = self.client
                case.warm(session)
                session.update(case.frame(session), (request,))
                self.assertTrue(self.client.armed)
                self.assertEqual(self.client.initial_lamp_state, "ON")
                self.assertEqual(session.laps.total, 7)
            finally: case.tearDown()

    def test_stop_while_on_prepare_is_saving_cannot_be_overridden_by_late_save(self):
        with patch("robot.interface.preview_transport.time.monotonic", lambda:self.case.now):
            self.assertTrue(self.control.submit("start", 0, laps=7,
                lamp_configuration=self.case.manual(), initial_lamp_state="ON")[0])
            self.assertTrue(self.control.submit("stop", 0, initial_lamp_state={"bad":True})[0])
            self.client._step(); self.case.publish()
            self.assertEqual(self.control.take().action, "stop")
            self.assertIsNone(self.control.take())
            self.assertFalse(self.client.armed)
            self.assertFalse(self.client.permission)
            self.assertEqual(self.control.status()["preparation"]["phase"], "CANCELLED")

    def test_armed_mode_and_lap_target_cannot_be_mutated(self):
        case = session_tests.DrivingSessionTests(); case.setUp()
        try:
            session = case.session(); session.start_light = self.client
            case.warm(session)
            session.update(case.frame(session), (ControlRequest("start", 7, "ON"),))
            reference = self.client.session
            session.update(case.frame(session), (ControlRequest("start", 9, "OFF"),))
            self.assertEqual((self.client.session, self.client.initial_lamp_state), (reference, "ON"))
            self.assertEqual(session.laps.total, 7)
        finally: case.tearDown()

    def test_direct_start_does_not_apply_arm_mode_or_grant_signal_permission(self):
        case = session_tests.DrivingSessionTests(); case.setUp()
        try:
            session = case.session(); session.start_light = self.client
            case.warm(session)
            session.update(case.frame(session), (ControlRequest("start_now", 7, "ON"),))
            self.assertTrue(session.active)
            self.assertEqual(self.client.initial_lamp_state, "OFF")
            self.assertFalse(self.client.permission)
            self.assertFalse(any(call['op'] == 'arm' for call in self.case.calls))
        finally: case.tearDown()

