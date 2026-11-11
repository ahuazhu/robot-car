"""One calibrated gain change reaches both pursuit and the body prediction."""
from tests.support import ROOT
from tests.support.driving import road, flow
from dataclasses import replace
from robot.config import RuntimeSettings
from robot.control.controller import compensate_pwm
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.steering_control import SteeringControl
import unittest


class UndersteerGainTests(unittest.TestCase):
    def evaluate(self, curvature, gain):
        settings=RuntimeSettings.load(ROOT/'maixcam2/curve_follow',{})
        # Isolate the calibrated geometric law from the experimental dynamic
        # window, whose first command deliberately limits the turn rate.
        settings=replace(settings,runtime=dict(settings.runtime,speed_profile=dict(
            settings.runtime['speed_profile'],controller='PURE_PURSUIT')))
        speed=ProfileSpeedControl.from_settings(settings)
        control=SteeringControl(settings.road_policy,[0.,0.],(.075,.105,.085),.3,
            lever=settings.turn_lever,gain=gain,cruise=settings.cruise_pwm,
            deceleration=settings.crosswalk_policy.brake_deceleration_m_s2,
            confirmed_cruise=settings.confirmed_cruise_pwm,
            straight_cruise=settings.straight_cruise_pwm,speed_control=speed)
        control.observe(flow(),1.,False);control.applied(.99,(60,60))
        command=control.command(road(curvature,reference_sigma_m=.005,
            road_model=dict(persistent_history=True)),.1)
        self.assertIsNotNone(command)
        self.assertTrue(speed.status['selected'])
        self.assertLessEqual(max(compensate_pwm(command)),200)
        self.assertLessEqual(speed.status['target_speed_m_s'],speed.status['ceiling_m_s'])
        return control,command

    def test_field_gain_strengthens_both_bends_without_changing_straight(self):
        gain=RuntimeSettings.load(ROOT/'maixcam2/curve_follow',{}).steering_gain
        before_straight,before_command=self.evaluate(0.,.95)
        after_straight,after_command=self.evaluate(0.,gain)
        self.assertEqual(before_command,after_command)
        for curvature in (-4.,-2.,-1.,1.,2.,4.):
            with self.subTest(curvature=curvature):
                before,a=self.evaluate(curvature,.95)
                after,b=self.evaluate(curvature,gain)
                ratio_a=abs((a[1]-a[0])/sum(a))
                ratio_b=abs((b[1]-b[0])/sum(b))
                self.assertGreater(ratio_b/ratio_a,1.08)
                self.assertLess(ratio_b/ratio_a,1.20)
                self.assertGreater(after.status['envelope']['minimum_clearance_m'],
                                   before.status['envelope']['minimum_clearance_m'])
                # The same gain must be used in physical prediction and output.
                self.assertGreater(abs(after.status['requested_curvature_m_inv']),
                                   abs(before.status['requested_curvature_m_inv']))
        _,left=self.evaluate(2.,gain);_,right=self.evaluate(-2.,gain)
        self.assertEqual(left,tuple(reversed(right)))


if __name__=='__main__':unittest.main()
