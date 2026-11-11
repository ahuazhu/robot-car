"""Measured chassis lag, noisy-flow handoff and conflict-speed regressions."""
from tests.support import ROOT
from tests.support.driving import road, flow
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits
from robot.navigation.dwpp import plan_twist
from robot.navigation.steering_control import SteeringControl
from robot.core.road_policy import RoadPolicy
import json
import math
import unittest
import numpy as np

PARAMS=dict(maximum=.3,lever=.15,acceleration=.4,deceleration=.2,
            angular_acceleration=2.,angular_deceleration=2.)


class MotionFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.control=ProfileSpeedControl(SpeedLimits(.3,.15,.4,.2,.35,.15),218,.4,controller='DWPP')
        self.target=road(reference_sigma_m=.005,road_model=dict(persistent_history=True))

    def step(self,now=1.,speed=.1,yaw=0.,k=2.,confidence=1.,**kwargs):
        target=dict(self.target,control_confidence=confidence)
        return self.control.command(target,target['control_path_m'],now,speed,k,.005,.075,True,
                                    measured_yaw_rate=yaw,**kwargs)

    def test_recorded_reversals_use_measured_turn_even_with_an_applied_reference(self):
        # Field frames1754/1766/1773: valid flow disagreed in turn direction
        # with the command-derived state. Check physical bounds, not old output.
        for v,w,old_v,old_w,k,dt in (
            (.0549005,-.436467,.03,-.0296296,5.670425,.149038),
            (.0848288,.557081,.0428014,-.204585,-9.696208,.182010),
            (.1072443,-.528241,.03,.0105263,5.667985,.226865)):
            self.control.reset();self.control.applied(1-dt,(30.,30.))
            self.control.dwpp_state=(1-dt,old_v,old_w)
            wheels=self.step(speed=v,yaw=w,k=k)
            self.assertIsNotNone(wheels)
            s=self.control.status;d=s['dwpp']
            self.assertEqual(d['source'],'MEASURED')
            self.assertAlmostEqual(d['current_angular_rad_s'],w)
            self.assertGreater(s['reference_angular_rad_s']*w,0)
            self.assertLessEqual(abs(s['reference_angular_rad_s']-w),2*dt+1e-8)
            self.assertGreaterEqual(min(wheels),0)
            self.assertLessEqual(max(wheels),218)
            json.dumps(s,allow_nan=False)

    def test_valid_yaw_is_not_thrown_away_when_forward_displacement_is_unresolved(self):
        steer=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,speed_control=self.control)
        angle=.06
        m=dict(flow(),rotation=np.array([[math.cos(angle),math.sin(angle)],
                                        [-math.sin(angle),math.cos(angle)]]),
               translation=np.zeros(2),sigma_m=.002)
        steer.applied(.9,(30.,30.));steer.observe(m,1.,False)
        self.assertIsNotNone(steer.command(self.target,.02))
        s=self.control.status
        self.assertFalse(s['motion_feedback'])  # no forward integral windup
        self.assertTrue(s['motion_measurement_valid'])
        self.assertAlmostEqual(s['dwpp']['current_angular_rad_s'],.4)
        self.assertEqual(s['motion_state_source'],'MEASURED')
        self.assertLessEqual(s['target_speed_m_s'],.075)

    def test_dropout_extrapolates_from_body_state_not_instant_command_replacement(self):
        self.control.applied(.9,(30.,30.))
        wheels=self.step(speed=.1,yaw=-.7,k=5.)
        self.control.applied(1.,wheels)
        # Force an opposite reference to expose command/sensor conflation.
        self.control.dwpp_state=(1.,.1,.5)
        self.step(now=1.1,speed=None,yaw=None,k=5.,motion_valid=False,motion_feedback=False)
        s=self.control.status
        self.assertEqual(s['motion_state_source'],'ESTIMATED')
        self.assertLess(s['dwpp']['current_angular_rad_s'],0)
        self.assertGreater(s['dwpp']['current_angular_rad_s'],-.7)
        self.assertFalse(s['motion_feedback'])
        self.assertLessEqual(s['target_speed_m_s'],.075)
        # A second probe at the same instant cannot evolve the estimate again.
        w=s['dwpp']['current_angular_rad_s']
        self.step(now=1.1,speed=None,yaw=None,k=5.,motion_valid=False,motion_feedback=False)
        self.assertEqual(self.control.status['dwpp']['current_angular_rad_s'],w)

    def test_long_command_gap_keeps_measured_turn_and_bounds_restart(self):
        self.control.applied(.4,(90.,90.))
        self.step(speed=.15,yaw=.8,k=-5.)
        s=self.control.status
        self.assertTrue(s['control_gap'])
        self.assertEqual(s['dwpp']['current_angular_rad_s'],.8)
        self.assertLessEqual(s['target_speed_m_s'],.075)
        self.assertGreater(s['reference_angular_rad_s'],0)
        self.assertFalse(s['motion_feedback'])

    def test_confidence_falls_now_and_recovers_by_elapsed_evidence_not_frame_count(self):
        histories=[]
        for dt in (.05,.1,.2):
            self.control.reset();self.control.applied(.9,(30.,30.))
            self.step(confidence=.04,speed=.03)
            for i in range(1,round(.6/dt)+1):
                now=1+i*dt
                self.control.applied(now-.01,(30.,30.))
                self.step(now=now,confidence=.65,speed=.03)
                before=self.control.status['centerline_confidence']
                self.step(now=now,confidence=.65,speed=.03)
                self.assertEqual(self.control.status['centerline_confidence'],before)
            histories.append(self.control.status['centerline_confidence'])
            self.assertLess(histories[-1],.37)
            self.step(now=1.61,confidence=.01,speed=.03)
            self.assertEqual(self.control.status['centerline_confidence'],.01)
            self.control.applied(2.5,(30.,30.))
            self.step(now=2.6,confidence=1.,speed=.03)
            self.assertEqual(self.control.status['centerline_confidence'],.01)
        self.assertAlmostEqual(min(histories),max(histories))

    def test_selected_arc_obeys_lateral_limit_even_when_path_requests_straight(self):
        self.control=ProfileSpeedControl(SpeedLimits(.3,.08,.4,.2,.35,.15),218,.4,controller='DWPP')
        self.control.applied(.95,(80.,80.))
        self.step(speed=.15,yaw=1.,k=0.)
        s=self.control.status
        self.assertIn('LATERAL_BRAKING',s['dwpp']['overrides'])
        self.assertLessEqual(s['target_speed_m_s']*abs(s['reference_angular_rad_s']),.08+1e-10)
        self.assertEqual(self.control.dwpp_proposal[1:],
                         (s['target_speed_m_s'],s['reference_angular_rad_s']))

    def test_emergency_zero_and_sensor_return_do_not_retain_old_turn_authority(self):
        self.control.applied(.9,(30.,30.));wheels=self.step(yaw=.7)
        self.control.applied(1.,wheels)
        self.assertEqual(self.step(now=1.1,yaw=.7,task_limit_m_s=0.),(0.,0.))
        self.control.applied(1.1,(0.,0.));self.control.reset()
        self.step(now=2.,speed=0.,yaw=0.)
        self.assertEqual(self.control.status['dwpp']['current_angular_rad_s'],0.)


class AdditionalWindowReviewTests(unittest.TestCase):
    def test_actual_overspeed_is_reported_and_braked_not_rejected_or_hidden(self):
        for v,w in ((.4,0.),(.15,3.),(.15,-3.),(.6,3.)):
            s=plan_twist(v,w,0.,.3,.1,**PARAMS)
            self.assertTrue(s.overrides)
            self.assertGreaterEqual(min(s.linear-.15*s.angular,s.linear+.15*s.angular),-1e-8)
            self.assertLessEqual(max(s.linear-.15*s.angular,s.linear+.15*s.angular),.3+1e-8)

    def test_reverse_acceleration_uses_only_time_after_stopping_original_turn(self):
        # .1rad/s -> rest in .1s, remaining .1s accelerates at .2rad/s².
        for sign in (-1,1):
            s=plan_twist(.1,sign*.1,-sign*4.,.3,.2,
                **dict(PARAMS,angular_acceleration=.2,angular_deceleration=1.))
            low=-.02 if sign>0 else -.14
            high=.14 if sign>0 else .02
            self.assertAlmostEqual(s.window.w_min,low)
            self.assertAlmostEqual(s.window.w_max,high)
            self.assertLessEqual(abs(s.angular),.02+1e-8)

if __name__=='__main__':unittest.main()
