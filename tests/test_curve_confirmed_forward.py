"""Finish runout uses confirmed motor output time, never stale frame PWM."""

from tests.support import ROOT
import math
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from robot.control.motor_channel import _motor_exchange, _motor_fail
from robot.tasks.lap_counter import LapCounter


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
            "values": [0.0] * 15,
            "lock": threading.Lock(),
            "lamp": SimpleNamespace(value=0),
            "error": SimpleNamespace(value=b""),
        }
        self.transport = Transport()

    def exchange(self, at, op=2, left=50, right=50):
        with patch("robot.control.motor_channel.time.monotonic", return_value=at):
            _motor_exchange(self.transport, self.state, op, left, right)
        return self.state["values"][13]

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


class ConfirmedRunoutTests(unittest.TestCase):
    def counter(self, total=10.0):
        c = LapCounter(dict(enabled=True, laps=2, finish_runout_s=5.0))
        c.running = True
        c.count = 2
        c.phase = "FINISH_CLEAR_REAR"
        c.release_track = dict(normal=np.array([1.0, 0.0]), distance=-1.0, sigma=0.001)
        motion = dict(valid=True, rotation=np.eye(2), translation=np.zeros(2), sigma_m=0.0)
        self.assertFalse(c.update({}, motion, 1.0, True, driving=True, forward_total_s=total))
        self.assertEqual(c.phase, "FINISH_RUNOUT")
        self.assertEqual(c.finish_elapsed, 0.0)
        return c

    def test_long_frame_with_expired_motor_command_cannot_finish_early(self):
        c = self.counter()
        self.assertFalse(c.update({}, None, 5.6, True, driving=True, forward_total_s=14.6))
        # A .55s frame gap only contains .20s of confirmed forward output.
        self.assertFalse(c.update({}, None, 6.15, True, driving=True, forward_total_s=14.8))
        self.assertAlmostEqual(c.finish_elapsed, 4.8)
        self.assertFalse(c.update({}, None, 26.0, True, driving=True, forward_total_s=14.8))
        self.assertTrue(c.update({}, None, 26.3, True, driving=True, forward_total_s=15.0))

    def test_totals_override_stale_pwm_boolean_and_clearance_sets_baseline(self):
        c = self.counter(100.0)
        self.assertFalse(c.update({}, None, 20.0, True, driving=False, forward_total_s=101.0))
        self.assertEqual(c.finish_elapsed, 1.0)
        self.assertFalse(c.update({}, None, 40.0, True, driving=True, forward_total_s=101.0))
        self.assertEqual(c.finish_elapsed, 1.0)

    def test_missing_invalid_and_reset_totals_do_not_backfill_or_use_wall_time(self):
        for invalid in (None, float("nan"), float("inf"), -1.0, True, "bad"):
            with self.subTest(invalid=invalid):
                c = self.counter()
                c.update({}, None, 2.0, True, forward_total_s=11.0)
                self.assertFalse(
                    c.update({}, None, 100.0, True, driving=True, forward_total_s=invalid)
                )
                self.assertFalse(
                    c.update({}, None, 101.0, True, driving=True, forward_total_s=20.0)
                )
                self.assertEqual(c.finish_elapsed, 1.0)
                c.update({}, None, 102.0, True, forward_total_s=21.0)
                self.assertEqual(c.finish_elapsed, 2.0)
        c = self.counter()
        c.update({}, None, 2.0, True, forward_total_s=11.0)
        c.update({}, None, 3.0, True, forward_total_s=0.2)
        self.assertEqual(c.finish_elapsed, 1.0)
        c.update({}, None, 4.0, True, forward_total_s=0.5)
        self.assertAlmostEqual(c.finish_elapsed, 1.3)

    def test_legacy_callers_without_totals_keep_command_time_interface(self):
        c = self.counter(None)
        self.assertFalse(c.update({}, None, 3.0, True, driving=True))
        self.assertEqual(c.finish_elapsed, 2.0)
        self.assertFalse(c.update({}, None, 20.0, True, driving=False))
        self.assertTrue(c.update({}, None, 23.0, True, driving=True))


if __name__ == "__main__":
    unittest.main()
