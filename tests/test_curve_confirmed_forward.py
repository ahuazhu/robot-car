"""Motor-output telemetry excludes stale commands and missing acknowledgements."""

from tests.support import ROOT
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from robot.control.motor_channel import MotorChannel, _motor_exchange, _motor_fail


class Transport:
    def __init__(self):
        self.armed = True

    def send(self, op, left=0, right=0):
        if op == 0:
            self.armed = False
        if op == 1:
            self.armed = True
        return [1, int(self.armed), left if self.armed else 0, right if self.armed else 0, 0]


class ConfirmedForwardClockTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "values": [0.0] * 20,
            "lock": threading.Lock(),
            "lamp": SimpleNamespace(value=0),
            "error": SimpleNamespace(value=b""),
        }
        self.transport = Transport()

    def exchange(self, at, op=2, left=50, right=50):
        with patch("robot.control.motor_channel.time.monotonic", return_value=at):
            _motor_exchange(self.transport, self.state, op, left, right)
        return self.state["values"][13]

    def test_completed_ack_keeps_its_send_even_when_new_target_arrives_in_flight(self):
        self.state['values'][0:4] = [1, 140, 70, 0.8]
        def send(op, left, right):
            # Main process publishes the next frame during the old UART round trip.
            self.state['values'][1:4] = [200, 80, 1.01]
            return [17, 1, left, right, 0]
        with patch.object(self.transport, 'send', side_effect=send), \
             patch('robot.control.motor_channel.time.monotonic', side_effect=[1., 1.02]):
            _motor_exchange(self.transport, self.state, 2, 140, 70, published_at=.8)
        channel = MotorChannel.__new__(MotorChannel)
        channel.state = self.state
        channel.worker = SimpleNamespace(is_alive=lambda: True)
        with patch('robot.control.motor_channel.time.monotonic', return_value=1.03):
            snapshot = channel.snapshot()
        self.assertEqual(snapshot['published'], dict(at=1.01, pwm=[200,80]))
        self.assertEqual(snapshot['ack'], [17,1,140,70,0])
        self.assertEqual(snapshot['exchange'], dict(sent_at=1., ack_at=1.02,
                         op=2, pwm=[140,70], target_published_at=.8))
        self.assertEqual(snapshot['ack_age_ms'], 10)

    def test_only_continuous_forward_ack_intervals_count(self):
        self.assertEqual(self.exchange(1.0), 0.0)
        self.assertAlmostEqual(self.exchange(1.1), 0.1)
        self.assertAlmostEqual(self.exchange(1.2, left=0, right=0), 0.1)
        self.assertAlmostEqual(self.exchange(1.3), 0.1)
        self.assertAlmostEqual(self.exchange(1.4), 0.2)

    def test_long_ack_gap_has_no_backfill(self):
        self.exchange(1.0)
        self.assertEqual(self.exchange(1.3), 0.0)
        self.assertAlmostEqual(self.exchange(1.4), 0.1)
        self.assertAlmostEqual(self.exchange(2.4), 0.1)
        self.assertAlmostEqual(self.exchange(2.5), 0.2)

    def test_reverse_rotation_stop_arm_and_fault_break_continuity(self):
        for kind in ("reverse", "rotation", "stop", "arm", "error"):
            with self.subTest(kind=kind):
                self.setUp()
                self.exchange(1.0)
                self.exchange(1.1)
                if kind == "reverse":
                    self.exchange(1.2, left=-50, right=-50)
                elif kind == "rotation":
                    self.exchange(1.2, left=-50, right=50)
                elif kind in ("stop", "arm"):
                    self.exchange(1.2, op=0 if kind == "stop" else 1, left=0, right=0)
                    if kind == "stop":
                        self.exchange(1.25, op=1, left=0, right=0)
                else:
                    with patch("robot.control.motor_channel.time.monotonic", return_value=1.2):
                        _motor_fail(self.transport, self.state, RuntimeError("fault"))
                    self.exchange(1.25, op=1, left=0, right=0)
                self.assertAlmostEqual(self.exchange(1.3), 0.1)
                self.assertAlmostEqual(self.exchange(1.4), 0.2)


if __name__ == "__main__":
    unittest.main()
