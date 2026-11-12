"""Actuator-coordinate, closed-loop effort, braking and eligibility regressions."""
from tests.support import ROOT
from tests.support.driving import road
import unittest

from robot.control.controller import compensate_pwm
from robot.navigation.bend_response import BendResponse
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits
from robot.navigation.steering_control import SteeringControl
from robot.core.road_policy import RoadPolicy
import math
import numpy as np


class BendResponseTests(unittest.TestCase):
    def control(self, enabled=True):
        return ProfileSpeedControl(SpeedLimits(.3,.15,.4,.2,.35,.15),163,.4,
            controller='DWPP',bend_compensation=enabled)

    def step(self,c,*,now=.3,speed=.20,k=-2.,allowed=True):
        r=road(k,reference_sigma_m=.005)
        return c.command(r,r['control_path_m'],now,speed,k,.005,.075,True,
                         measured_yaw_rate=k*speed,bend_allowed=allowed)

    def test_actual_pwm_round_trip_and_inside_preserved_both_directions(self):
        model=BendResponse(.15,37)
        for base,k in (((103.,51.5),-2.2),((51.5,103.),2.2)):
            result=model.allocate(base,k,allowed=True)
            self.assertIsNotNone(result)
            wire=compensate_pwm(result['wheels'])
            self.assertEqual(list(wire),result['wire'])
            old=compensate_pwm(base);outside=0 if k<0 else 1
            self.assertEqual(wire[1-outside],old[1-outside])
            self.assertGreater(wire[outside],old[outside])
            self.assertLessEqual(wire[outside]-old[outside],20)
            self.assertLessEqual(max(wire),180)

    def test_straight_low_inside_invalid_direction_and_permission_no_change(self):
        model=BendResponse(.15,37)
        for wheels,k,allowed in (((80.,80.),0.,True),((103.,20.),-3.,True),
                                  ((103.,51.5),2.,True),((103.,51.5),-2.,False),
                                  ((170.,85.),-2.,True)):
            self.assertIsNone(model.allocate(wheels,k,allowed=allowed))

    def test_zero_straight_and_untrusted_outputs_match_baseline(self):
        for k,speed,allowed in ((0.,.2,True),(-2.,.08,True),(-2.,.2,False)):
            a,b=self.control(False),self.control(True)
            for c in (a,b):c.applied(.1,(80.,80.))
            x=self.step(a,k=k,speed=speed,allowed=allowed)
            y=self.step(b,k=k,speed=speed,allowed=allowed)
            self.assertEqual(x,y)
            self.assertNotIn('bend_allocation',b.status)

    def test_speed_reference_unchanged_and_applied_base_effort_does_not_ratchet(self):
        c=self.control();c.applied(.1,(80.,80.))
        w=self.step(c)
        self.assertTrue(c.status['bend_allocation']['active'])
        base=c.status['regulator_effort_mean_pwm']
        reference=c.status['target_speed_m_s']
        c.applied(.31,w,delivered_wheels=compensate_pwm(w))
        self.assertAlmostEqual(c.applied_mean,base)
        for index in range(8):
            now=.5+.2*index
            w=self.step(c,now=now,speed=reference)
            self.assertAlmostEqual(c.status['regulator_effort_mean_pwm'],base,places=8)
            c.applied(now+.01,w,delivered_wheels=compensate_pwm(w))
            self.assertAlmostEqual(c.applied_mean,base,places=8)

    def test_task_ramp_scale_updates_base_feedback_and_zero_resets(self):
        c=self.control();c.applied(.1,(80.,80.));w=self.step(c)
        base=c.status['regulator_effort_mean_pwm']
        scaled=tuple(x*.8 for x in w)
        c.applied(.31,scaled,delivered_wheels=compensate_pwm(scaled))
        self.assertAlmostEqual(c.applied_mean,base*.8)
        c.applied(.4,(0.,0.),delivered_wheels=(0,0))
        self.assertEqual(c.applied_mean,0)
        self.assertIsNone(c.reference)
        self.assertIsNone(c.bend_proposal)

    def test_applied_motion_retains_baseline_pwm_ratio_estimate(self):
        c=self.control();c.applied(.1,(80.,80.));w=self.step(c)
        result=c.status['bend_allocation']
        c.applied(.31,w,delivered_wheels=compensate_pwm(w))
        self.assertAlmostEqual(c.status['applied_curvature_m_inv'],result['nominal_curvature_m_inv'])
        self.assertNotAlmostEqual(c.status['applied_curvature_m_inv'],result['estimated_curvature_m_inv'])

    def test_clearance_braking_removes_extra_outside_effort(self):
        c=self.control();c.applied(.1,(80.,80.));w=self.step(c)
        original=c.bend_proposal['base_wheels'];v=c.status['target_speed_m_s']
        braked=c.constrain_clearance(w,v*.5)
        self.assertEqual(braked,tuple(x*.5 for x in original))
        self.assertFalse(c.status['bend_allocation']['active'])
        self.assertIsNone(c.bend_proposal)
        c.applied(.31,braked,delivered_wheels=compensate_pwm(braked))
        self.assertAlmostEqual(c.applied_mean,sum(braked)/2)

    def test_directional_curvature_estimate_and_extrapolation_are_explicit(self):
        model=BendResponse(.15,37)
        self.assertAlmostEqual(model.curvature((140,70)),-1.52,places=2)
        self.assertAlmostEqual(model.curvature((70,140)),1.613333333,places=2)
        c=self.control();c.applied(.1,(80.,80.));self.step(c)
        self.assertTrue(c.status['bend_allocation']['outside_training_domain'])

    def test_observed_supported_bend_not_denied_by_point76_tail_confidence(self):
        c=self.control()
        steering=SteeringControl(RoadPolicy(),[0,0],(.075,.105,.085),.3,
                                 speed_control=c,cruise=135)
        steering.applied(.8,(80,80),delivered_wheels=compensate_pwm((80,80)))
        target=road(-2.,reference_sigma_m=.005,confidence=.76,observation_confidence=.82)
        rotation=np.array([[math.cos(.08),-math.sin(.08)],
                           [math.sin(.08),math.cos(.08)]])
        motion=dict(valid=True,prediction_valid=True,interval_s=.2,rotation=rotation,
                    translation=-np.array([.04,0])@rotation.T,sigma_m=.0005,
                    source='GROUND FLOW')
        for now in (1.,1.2,1.4,1.6):
            steering.observe(motion,now,False)
            wheels=steering.command(target,.05)
            if wheels:
                steering.applied(now+.05,wheels,delivered_wheels=compensate_pwm(wheels))
        self.assertTrue(steering.status['speed_plan']['boost_eligible'])
        self.assertLess(c.status['centerline_confidence'],.8)
        allocation=c.status['bend_allocation']
        self.assertTrue(allocation['active'])
        self.assertGreater(max(allocation['wire']),180)
        self.assertLessEqual(max(allocation['wire']),200)
        self.assertLessEqual(allocation['outside_added_pwm'],20)


if __name__=='__main__':unittest.main()
