from tests.support import ROOT
import time, os, multiprocessing, unittest

from robot.control.motor_channel import MotorChannel


class FakeTransport:
    def __init__(self):
        self.port = self
        ctx = multiprocessing.get_context("fork")
        self.records = ctx.RawArray("d", 4000)
        self.count = ctx.RawValue("i", 0)
        self.active = ctx.RawValue("b", False)
        self.reject_value = ctx.RawValue("b", False)
        self.last = 0.0
        self.seq = 0
        self.owner = None

    @property
    def events(self):
        return [tuple(self.records[i * 4 : i * 4 + 4]) for i in range(self.count.value)]

    @property
    def reject(self):
        return self.reject_value.value

    @reject.setter
    def reject(self, value):
        self.reject_value.value = value

    def send(self, op, left=0, right=0):
        ident = os.getpid()
        if self.owner is None:
            self.owner = ident
        if self.owner != ident:
            raise AssertionError("More than one serial owner")
        now = time.monotonic()
        time.sleep(0.004)
        self.seq += 1
        if op == 0:
            self.active.value = False
        elif op == 1:
            self.active.value = True
            self.last = now
        elif op == 2:
            if not self.active.value or self.reject or now - self.last >= 0.350:
                self.active.value = False
                raise RuntimeError("UNO rejected motion")
            self.last = now
        i = self.count.value
        self.records[i * 4 : i * 4 + 4] = (now, op, left, right)
        self.count.value = i + 1
        return [
            self.seq,
            int(self.active.value),
            left if self.active.value else 0,
            right if self.active.value else 0,
            0,
        ]

    def close(self):
        pass


class ChannelTests(unittest.TestCase):
    def test_serial_continues_while_vision_blocks_and_fails_closed(self):
        t = FakeTransport()
        c = MotorChannel(t, period=0.025, fresh_seconds=0.12, stop_seconds=0.28)
        try:
            c.send(0)
            c.send(1)
            c.send(2, 45, 55)
            time.sleep(0.095)  # Camera main loop blocked; worker must keep sending.
            motion = [e for e in t.events if e[1:] == (2, 45, 55)]
            self.assertGreaterEqual(len(motion), 2)
            time.sleep(0.105)
            self.assertTrue(any(e[1:] == (2, 0, 0) and e[0] > motion[-1][0] for e in t.events))
            time.sleep(0.15)
            self.assertFalse(c.snapshot()["armed"])
            self.assertEqual(c.snapshot()["error"], "VISION COMMAND EXPIRED")
            with self.assertRaises(RuntimeError):
                c.send(2, 50, 50)
        finally:
            c.close()

    def test_stop_preempts_published_target_and_no_automatic_rearm(self):
        t = FakeTransport()
        c = MotorChannel(t, period=0.02)
        try:
            c.send(1)
            c.send(2, 40, 40)
            time.sleep(0.04)
            c.send(0)
            cut = len(t.events)
            time.sleep(0.06)
            self.assertFalse(c.snapshot()["armed"])
            self.assertFalse(any(e[1] == 2 and (e[2] or e[3]) for e in t.events[cut:]))
            with self.assertRaises(RuntimeError):
                c.send(2, 40, 40)
        finally:
            c.close()

    def test_serial_fault_stops_and_retains_error(self):
        t = FakeTransport()
        c = MotorChannel(t, period=0.02)
        try:
            c.send(1)
            t.reject = True
            c.send(2, 45, 45)
            time.sleep(0.06)
            s = c.snapshot()
            self.assertFalse(s["armed"])
            self.assertIn("rejected", s["error"])
            self.assertEqual(s["ack"][1:4], [0, 0, 0])
        finally:
            c.close()


class CompensationTests(unittest.TestCase):
    def test_stopped_output_is_zero_and_limits_hold(self):
        from robot.control.controller import compensate_pwm

        self.assertEqual(compensate_pwm((0, 0)), (0, 0))
        self.assertEqual(compensate_pwm((53, 53)), (90, 90))
        self.assertEqual(compensate_pwm((-300, 300)), (-255, 255))

    def test_actual_right_bend_boost_preserves_curvature_instead_of_straightening(self):
        from robot.control.controller import compensate_pwm

        left, right = compensate_pwm((43, 12))
        self.assertEqual(left, 80)
        wanted = (12 - 43) / (0.15 * (12 + 43))
        actual = (right - left) / (0.15 * (right + left))
        self.assertAlmostEqual(wanted, actual, delta=0.10)
        self.assertEqual((left, right), (80, 22))

    def test_ratio_survives_boost_clipping_ramp_and_both_turn_directions(self):
        from robot.control.controller import compensate_pwm, CommandRamp

        for desired in ((43, 12), (12, 43), (100, 20), (20, 100), (60, 0), (0, 60), (40, 40)):
            ramp = CommandRamp()
            for i in range(20):
                shaped = ramp.apply(desired, i * 0.1)
                applied = compensate_pwm(shaped)
                self.assertLessEqual(max(applied), 150)
                self.assertGreaterEqual(min(applied), 0)
                if sum(shaped) > 0:
                    expected = (shaped[1] - shaped[0]) / sum(shaped)
                    actual = (applied[1] - applied[0]) / sum(applied)
                    self.assertAlmostEqual(expected, actual, delta=2 / max(1, sum(applied)))


if __name__ == "__main__":
    unittest.main()


class SearchLeaseTests(unittest.TestCase):
    def test_forward_prediction_bridges_150ms_then_expires_without_parent_refresh(self):
        t = FakeTransport()
        c = MotorChannel(t, period=.02)
        try:
            c.send(1)
            sent = time.monotonic()
            c.send(2, 50, 40, lease_s=.24)
            time.sleep(.18)
            self.assertTrue(any(e[0] >= sent + .15 and e[1:] == (2, 50, 40)
                                for e in t.events))
            time.sleep(.14)
            self.assertTrue(any(e[0] >= sent + .24 and e[1:] == (2, 0, 0)
                                for e in t.events))
            self.assertTrue(c.snapshot()['armed'])
            for lease in (.251, .3, float('nan')):
                with self.assertRaises(ValueError):
                    c.send(2, 50, 40, lease_s=lease)
        finally:
            c.close()

    def test_short_turn_expires_to_zero_independently_of_main_loop(self):
        t = FakeTransport()
        c = MotorChannel(t, period=0.02)
        try:
            c.send(1)
            c.send(2, -55, 55, lease_s=0.05)
            time.sleep(0.14)
            events = t.events
            moving = [e for e in events if e[1:] == (2, -55, 55)]
            self.assertTrue(moving)
            self.assertTrue(any(e[1:] == (2, 0, 0) and e[0] > moving[-1][0] for e in events))
            self.assertTrue(c.snapshot()["armed"])
            self.assertIsNone(c.snapshot()["error"])
            c.send(2, 35, 35)
            time.sleep(0.06)
            self.assertTrue(any(e[1:] == (2, 35, 35) for e in t.events))
        finally:
            c.close()

    def test_invalid_search_leases_never_publish_motion(self):
        t = FakeTransport()
        c = MotorChannel(t)
        try:
            c.send(1)
            for lease in (-1, 0, 0.2, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    c.send(2, -50, 50, lease_s=lease)
            time.sleep(0.04)
            self.assertFalse(any(e[1] == 2 and (e[2] or e[3]) for e in t.events))
        finally:
            c.close()
