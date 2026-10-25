"""Match host/firmware duty ranges without losing STOP or ratio at saturation."""
import json,re,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'));sys.path.insert(0,str(ROOT/'tests'))
from controller import MAX_MOTOR_PWM,compensate_pwm
from test_curve_drive_link import link,Clock
from test_curve_adaptive_speed import control,road
import drive_link


class OutputRangeTests(unittest.TestCase):
    def test_firmware_cap_matches_host_and_configured_straight_reaches_it(self):
        text=(ROOT/'uno/uno_lane_drive/DriveSafety.h').read_text()
        self.assertEqual(int(re.search(r'MAX_PWM\s*=\s*(\d+)',text).group(1)),MAX_MOTOR_PWM)
        runtime=json.loads((ROOT/'maixcam2/curve_follow/runtime.json').read_text())
        self.assertEqual(MAX_MOTOR_PWM,200)
        _,command=control(road(),fast=runtime['straight_cruise_pwm'])
        self.assertEqual(command,(163,163));self.assertEqual(compensate_pwm(command),(200,200))
    def test_old_low_commands_and_zero_are_unchanged_but_new_saturation_is_coupled(self):
        self.assertEqual(compensate_pwm((0,0)),(0,0))
        self.assertEqual(compensate_pwm((53,53)),(90,90))
        self.assertEqual(compensate_pwm((43,12)),(80,22))
        self.assertEqual(compensate_pwm((300,100)),(200,67))
        self.assertEqual(compensate_pwm((-300,100)),(-200,67))
    def test_actual_maximum_duty_reply_is_accepted_and_larger_reply_rejected(self):
        for duty in (200,201):
            body=('1,1,%d,-%d,0,5'%(duty,duty)).encode()
            obj=link(b'!'+body+b'*%02X\n'%drive_link.crc8(body))
            with patch.object(drive_link,'time',Clock()):
                if duty==200:self.assertEqual(obj.send(2,200,-200)[2:4],[200,-200])
                else:
                    with self.assertRaisesRegex(RuntimeError,'Invalid UNO status'):obj.send(2,200,-200)
    def test_invalid_outbound_commands_never_reach_serial_port(self):
        for args in ((2,201,0),(2,0,-201),(2,256,0),(1,1,0),(3,0,1),(4,0,0),(2,1.5,0)):
            obj=link(b'')
            with self.assertRaises(ValueError):obj.send(*args)
            self.assertIsNone(obj.port.tx)


if __name__=='__main__':unittest.main()
