"""Final actuator floor, calibrated limits and body-braking regressions."""
from tests.support import ROOT
from tests.support.driving import road
import unittest
from robot.navigation.steering_control import SteeringControl
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits
from robot.core.road_policy import RoadPolicy
from robot.control.controller import compensate_pwm, CommandRamp


def logical(wire):
    peak = max(wire)
    return tuple(x * (peak - 37) / peak for x in wire)


class ContinuousMotorOutputTests(unittest.TestCase):
    def control(self, k=-2., ceiling=.3):
        speed = ProfileSpeedControl(SpeedLimits(.3,.15,.4,.2,.35,.15),163,.4,
                                    controller='DWPP',bend_compensation=True)
        c = SteeringControl(RoadPolicy(),[0,0],(.075,.105,.085),.3,speed_control=speed)
        c.observe(dict(valid=False),.1,False)
        c.command(road(k,reference_sigma_m=.005),0.)
        speed.status = dict(selected=True,ceiling_m_s=ceiling,
                            effective_curvature_m_inv=k,
                            dwpp=dict(window=dict(w_min=-.6,w_max=.6)))
        speed.bend_proposal = speed.dwpp_proposal = None
        return c

    def test_recorded_low_right_commands_get_continuous_pair_both_directions(self):
        for wire in ((146,54),(104,39),(77,28)):
            for mirror in (False,True):
                c = self.control(2. if mirror else -2.)
                requested = tuple(reversed(wire)) if mirror else wire
                command = logical(requested)
                output = c.continuous_output(command)
                delivered = compensate_pwm(output)
                self.assertGreaterEqual(min(delivered),60)
                self.assertLessEqual(max(delivered),200)
                self.assertLessEqual(c.speed_control.bend_proposal['estimated_speed_m_s'],.3)
                self.assertLess((delivered[0]-delivered[1])*(1 if mirror else -1),0)
                c.applied(.2,output,delivered_wheels=delivered)
                self.assertAlmostEqual(c.speed_control.applied_mean,sum(command)/2.)

    def test_final_ramp_cannot_reintroduce_low_nonzero_inner_duty(self):
        c = self.control()
        ramp = CommandRamp(400.)
        command = ramp.apply(logical((160,70)),.1)
        self.assertLess(min(compensate_pwm(command)),60)
        output = c.continuous_output(command)
        self.assertGreaterEqual(min(compensate_pwm(output)),60)

    def test_impossible_low_speed_ceiling_stops_both_instead_of_one_stalled_wheel(self):
        c = self.control(ceiling=.03)
        self.assertEqual(c.continuous_output(logical((104,39))),(0.,0.))
        self.assertEqual(c.status['motor_floor']['wire'],[0,0])

    def test_repair_respects_dwpp_angular_window_and_unknown_motion_cannot_kick(self):
        c = self.control()
        c.speed_control.status['dwpp']['window'] = dict(w_min=-.10,w_max=.10)
        output = c.continuous_output(logical((146,54)))
        self.assertTrue(any(output))
        self.assertLessEqual(abs(c.speed_control.bend_proposal['estimated_yaw_rad_s']),.10+1e-9)
        c = self.control()
        c.speed_control.status.pop('dwpp')
        self.assertEqual(c.continuous_output(logical((146,54))),(0.,0.))
        self.assertEqual(c.status['motor_floor']['reason'],'MOTION_REQUIRED_FOR_CONTINUOUS_PAIR')

    def test_actual_body_arc_is_checked_and_braking_ceiling_wins(self):
        c = self.control()
        # Real geometry: a narrow lane offset from the car blocks the candidate
        # body's path at pose zero, regardless of available motor torque.
        ctx = c.actuator_context
        ctx['observed_path'] = ctx['observed_path'] + [0,.25]
        self.assertEqual(c.continuous_output(logical((104,39))),(0.,0.))
        self.assertEqual(c.status['motor_floor']['reason'],
                         'BODY_CLEARANCE_BELOW_CONTINUOUS_FLOOR')

    def test_stop_normal_outputs_and_task_owned_commands_stay_exact(self):
        c = self.control()
        for wire in ((0,0),(60,60),(140,70)):
            command = (0,0) if not any(wire) else logical(wire)
            self.assertEqual(c.continuous_output(command),command)
        c.command(road(-2.,reference_sigma_m=.005),0.,speed_profile_allowed=False)
        command = logical((104,39))
        self.assertEqual(c.continuous_output(command),command)

    def test_existing_allocation_feedback_is_not_integrated_a_second_time(self):
        c = self.control()
        model = c.speed_control.bend_response
        original = model.allocate((103.,51.5),-2.,allowed=True,ceiling=.3)
        self.assertIsNotNone(original)
        c.speed_control.bend_proposal = original
        command = tuple(x*.4 for x in original['wheels'])
        output = c.continuous_output(command)
        self.assertGreaterEqual(min(compensate_pwm(output)),60)
        c.applied(.2,output,delivered_wheels=compensate_pwm(output))
        self.assertAlmostEqual(c.speed_control.applied_mean,original['base_mean']*.4)


if __name__ == '__main__':
    unittest.main()
