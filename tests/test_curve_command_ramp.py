"""Acceleration follows elapsed fresh-command time, not camera frame count."""

from tests.support import ROOT
import unittest

from robot.control.controller import CommandRamp, compensate_pwm


class CommandRampTimingTests(unittest.TestCase):
    def settled(self, command=(80, 163)):
        ramp = CommandRamp()
        for i in range(11):
            ramp.apply(command, i / 10)
        self.assertEqual(ramp.previous, tuple(float(x) for x in command))
        return ramp

    def test_same_elapsed_time_gives_same_acceleration_at_five_ten_twenty_fps(self):
        outputs = []
        for fps in (5, 10, 20):
            ramp = CommandRamp()
            for frame in range(round(0.6 * fps) + 1):
                output = ramp.apply((80, 163), frame / fps)
            outputs.append(output)
        self.assertEqual(outputs, [(57, 117)] * 3)

    def test_five_fps_reaches_cruise_in_one_second(self):
        ramp = CommandRamp()
        outputs = [ramp.apply((163, 163), i / 5) for i in range(6)]
        self.assertEqual(outputs[0], (9, 9))
        self.assertEqual(outputs[3], (117, 117))
        self.assertEqual(outputs[-1], (163, 163))
        self.assertTrue(all(b[0] - a[0] <= 36 for a, b in zip(outputs, outputs[1:])))

    def test_expired_or_reversed_clock_restarts_instead_of_banking_acceleration(self):
        for now in (1.3, 1.31, 5.0, 0.9):
            with self.subTest(now=now):
                ramp = self.settled()
                self.assertEqual(ramp.apply((80, 163), now), (4, 9))
                self.assertEqual(ramp.apply((80, 163), now + 0.1), (13, 27))

    def test_fresh_interval_uses_elapsed_time_but_duplicate_time_adds_nothing(self):
        ramp = CommandRamp()
        ramp.apply((163, 163), 0.0)
        self.assertEqual(ramp.apply((163, 163), 0.28), (59, 59))
        self.assertEqual(ramp.apply((163, 163), 0.28), (59, 59))

    def test_zero_missing_and_explicit_reset_start_a_new_ramp(self):
        for stop in (None, (0, 0), "reset"):
            with self.subTest(stop=stop):
                ramp = self.settled()
                if stop == "reset":
                    ramp.reset()
                else:
                    self.assertEqual(ramp.apply(stop, 1.01), (0, 0))
                self.assertEqual(ramp.apply((80, 163), 1.02), (4, 9))

    def test_deceleration_is_immediate_and_turn_ratio_survives_lower_fps(self):
        for desired in ((80, 163), (163, 80), (0, 163), (163, 0)):
            with self.subTest(desired=desired):
                ramp = CommandRamp()
                for frame in range(6):
                    output = ramp.apply(desired, frame / 5)
                    error = abs(output[0] * desired[1] - output[1] * desired[0])
                    self.assertLessEqual(error, sum(desired) / 2)
                    self.assertLessEqual(max(compensate_pwm(output)), 200)
                    for actual, requested in zip(output, desired):
                        if requested == 0:
                            self.assertEqual(actual, 0)
                reduced = tuple(value // 4 for value in desired)
                self.assertEqual(ramp.apply(reduced, 1.05), reduced)

    def test_short_rolling_pause_decays_but_stale_pause_cannot_hide_expiry(self):
        ramp = self.settled()
        self.assertEqual(ramp.pause(1.2, rolling=True), (0, 0))
        self.assertEqual(ramp.apply((80, 163), 1.25), (67, 136))
        for now, rolling in ((1.3, True), (3.0, True), (1.1, False), (0.9, True)):
            with self.subTest(now=now, rolling=rolling):
                ramp = self.settled()
                self.assertEqual(ramp.pause(now, rolling=rolling), (0, 0))
                self.assertEqual(ramp.apply((80, 163), now + 0.05), (4, 9))


if __name__ == "__main__":
    unittest.main()
