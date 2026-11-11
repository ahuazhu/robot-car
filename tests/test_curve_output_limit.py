"""Match host/firmware duty ranges without losing STOP or ratio at saturation."""

from tests.support import ROOT
import json, re, unittest
from unittest.mock import patch

from robot.control.controller import MAX_MOTOR_PWM, compensate_pwm
from tests.support.uart import link, Clock
from tests.support.driving import control, road
from robot.control import drive_link
from robot.config import RuntimeSettings
from robot.navigation.profile_speed_control import ProfileSpeedControl


class OutputRangeTests(unittest.TestCase):
    def test_full_hardware_range_matches_host_without_retuning_legacy_cruise(self):
        text = (ROOT / "uno/uno_lane_drive/DriveSafety.h").read_text()
        self.assertEqual(int(re.search(r"MAX_PWM\s*=\s*(\d+)", text).group(1)), MAX_MOTOR_PWM)
        runtime = json.loads((ROOT / "maixcam2/curve_follow/runtime.json").read_text())
        self.assertEqual(MAX_MOTOR_PWM, 255)
        _, command = control(road(), fast=runtime["straight_cruise_pwm"])
        self.assertEqual(command, (163, 163))
        self.assertEqual(compensate_pwm(command), (200, 200))

    def test_old_low_commands_and_zero_are_unchanged_but_new_saturation_is_coupled(self):
        self.assertEqual(compensate_pwm((0, 0)), (0, 0))
        self.assertEqual(compensate_pwm((53, 53)), (90, 90))
        self.assertEqual(compensate_pwm((43, 12)), (80, 22))
        self.assertEqual(compensate_pwm((164, 164)), (201, 201))
        self.assertEqual(compensate_pwm((213, 213)), (250, 250))
        self.assertEqual(compensate_pwm((218, 218)), (255, 255))
        self.assertEqual(compensate_pwm((300, 100)), (255, 85))
        self.assertEqual(compensate_pwm((-300, 100)), (-255, 85))

    def test_active_regulator_can_use_duty_above_200_without_raising_speed_ceiling(self):
        settings = RuntimeSettings.load(ROOT / 'maixcam2/curve_follow')
        regulator = ProfileSpeedControl.from_settings(settings)
        target = road(reference_sigma_m=.005, road_model=dict(persistent_history=True))
        regulator.applied(0., (0., 0.))
        peak = 0
        for i in range(1, 81):
            now = i * .1
            command = regulator.command(target, target['control_path_m'], now, .1,
                0., .005, .075, True, high_confidence=True, measured_yaw_rate=0.)
            self.assertIsNotNone(command)
            wire = compensate_pwm(command, settings.pwm_boost)
            peak = max(peak, *wire)
            self.assertLessEqual(max(wire), 255)
            self.assertLessEqual(regulator.status['ceiling_m_s'], .30)
            regulator.applied(now, command, delivered_wheels=wire)
        self.assertGreater(peak, 200)
        self.assertEqual(regulator.maximum_effort + settings.pwm_boost, 255)

    def test_actual_maximum_duty_reply_is_accepted_and_larger_reply_rejected(self):
        for duty in (200, 201, 250, 255, 256):
            body = ("1,1,%d,-%d,0,5" % (duty, duty)).encode()
            obj = link(b"!" + body + b"*%02X\n" % drive_link.crc8(body))
            with patch.object(drive_link, "time", Clock()):
                if duty <= 255:
                    self.assertEqual(obj.send(2, duty, -duty)[2:4], [duty, -duty])
                else:
                    with self.assertRaisesRegex(RuntimeError, "Invalid UNO status"):
                        obj.send(2, 255, -255)

    def test_invalid_outbound_commands_never_reach_serial_port(self):
        for args in (
            (2, 256, 0),
            (2, 0, -256),
            (1, 1, 0),
            (3, 0, 1),
            (4, 0, 0),
            (2, 1.5, 0),
        ):
            obj = link(b"")
            with self.assertRaises(ValueError):
                obj.send(*args)
            self.assertIsNone(obj.port.tx)


if __name__ == "__main__":
    unittest.main()
