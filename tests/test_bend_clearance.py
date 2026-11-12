"""Turning permission and the final proposed-arc clearance check."""
from tests.support.driving import road
from unittest.mock import patch
import unittest
import numpy as np

from robot.control.controller import compensate_pwm
from robot.core.road_policy import RoadPolicy
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits
from robot.navigation.steering_control import SteeringControl


class BendClearanceTests(unittest.TestCase):
    def run_bend(self, candidate_cap, candidate_clearance=.02):
        speed=ProfileSpeedControl(SpeedLimits(.3,.15,.4,.2,.35,.15),163,.4,
            controller='DWPP',bend_compensation=True,motor_rise_pwm_s=400,
            bend_curvature_gain=1.05)
        steering=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,
            cruise=135,speed_control=speed)
        baseline=dict(speed_scale=.8,speed_limit_m_s=.20,minimum_clearance_m=-.005)
        candidate=dict(speed_scale=1.,speed_limit_m_s=candidate_cap,
                       minimum_clearance_m=candidate_clearance)
        for now in (1.,1.2,1.4,1.6):
            steering.observe(dict(valid=True,interval_s=.2,rotation=np.eye(2),
                translation=np.array([-.04,0.]),sigma_m=.0002),now,False)
            steering.applied(now-.02,(80,80))
            # The geometric body checker is independent of allocation. Supply
            # its results for the original, proposed and (if needed) restored arc.
            with patch.object(steering,'envelope',side_effect=[baseline,candidate,baseline]) as checks:
                command=steering.command(road(-2,reference_sigma_m=.005),.1)
        return steering,speed,command,checks.call_count

    def test_near_boundary_turning_can_pass_without_permission_to_accelerate(self):
        steering,speed,command,checks=self.run_bend(.30)
        self.assertTrue(steering.status['bend_eligible'])
        self.assertFalse(steering.status['speed_plan']['boost_eligible'])
        allocation=speed.status['bend_allocation']
        self.assertTrue(allocation['active'])
        self.assertGreater(allocation['estimated_speed_m_s'],.20)
        self.assertEqual(list(compensate_pwm(command)),allocation['wire'])
        self.assertEqual(allocation['wire'][1],allocation['base_wire'][1])
        self.assertGreater(allocation['wire'][0],allocation['base_wire'][0])
        self.assertEqual(checks,2)

    def test_proposed_arc_still_brakes_and_restores_baseline_when_its_cap_fails(self):
        steering,speed,command,checks=self.run_bend(.10)
        self.assertTrue(steering.status['bend_eligible'])
        self.assertFalse(speed.status['bend_allocation']['active'])
        self.assertEqual(speed.status['actuator_model_reason'],'BODY_CLEARANCE_BASELINE')
        self.assertLessEqual(speed.status['target_speed_m_s'],.10)
        self.assertIsNone(speed.bend_proposal)
        self.assertEqual(checks,3)
        self.assertTrue(all(v>0 for v in command))

    def test_missing_candidate_clearance_is_not_permission_for_extra_output(self):
        for clearance in (None,float('nan')):
            with self.subTest(clearance=clearance):
                _,speed,command,checks=self.run_bend(None,clearance)
                self.assertFalse(speed.status['bend_allocation']['active'])
                self.assertEqual(speed.status['actuator_model_reason'],'BODY_CLEARANCE_UNOBSERVED')
                self.assertIsNone(speed.bend_proposal)
                self.assertEqual(checks,3)
                self.assertEqual(list(compensate_pwm(command)),speed.status['actuator_base_wire'])


if __name__=='__main__':unittest.main()
