"""Independent velocity-region checks and real controller/supervisor regressions."""
from tests.support import ROOT
from tests.support.driving import road, flow
from dataclasses import replace
import random
import unittest
from unittest.mock import patch
import numpy as np

from robot.config import RuntimeSettings
from robot.control.controller import CommandRamp, compensate_pwm
from robot.navigation.dwpp import closest_tracking_velocity, plan_twist
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.steering_control import SteeringControl
from robot.navigation.speed_profile import SpeedLimits
from robot.core.road_policy import RoadPolicy


PARAMS=dict(maximum=.3,lever=.15,acceleration=.4,deceleration=.2,
            angular_acceleration=2.,angular_deceleration=2.)


class VelocityRegionTests(unittest.TestCase):
    def test_exact_line_intersection_and_fastest_tie(self):
        rectangle=[(.1,-.3),(.2,-.3),(.2,.3),(.1,.3)]
        self.assertEqual(closest_tracking_velocity(rectangle,1.),(.2,.2,True))
        self.assertEqual(closest_tracking_velocity(rectangle,0.),(.2,0.,True))
        # Pursuit line lies outside: nearest edge ties select greater progress.
        self.assertEqual(closest_tracking_velocity([(.1,.1),(.2,.1)],0.),(.2,.1,False))
        self.assertEqual(closest_tracking_velocity([(.1,.2)],2.),(.1,.2,True))

    def test_wheel_polygon_changes_the_fastest_exact_tracking_solution(self):
        result=plan_twist(.15,0.,2.,.3,.25,**PARAMS)
        self.assertTrue(result.intersection)
        self.assertAlmostEqual(result.linear,.3/(1+.15*2))
        self.assertAlmostEqual(result.angular,2*result.linear)
        self.assertAlmostEqual(result.linear+.15*result.angular,.3)

    def test_left_right_symmetry_and_reversal(self):
        left=plan_twist(.2,.6,-4.,.3,.15,**PARAMS)
        right=plan_twist(.2,-.6,4.,.3,.15,**PARAMS)
        self.assertFalse(left.intersection)
        self.assertAlmostEqual(left.linear,right.linear)
        self.assertAlmostEqual(left.angular,-right.angular)
        self.assertGreaterEqual(left.angular,.6-2*.15-1e-9)
        # Do not command an impossible instantaneous turn reversal.
        self.assertGreater(left.angular,0.)

    def test_confidence_braking_overrides_reachability_but_never_the_speed_cap(self):
        result=plan_twist(.28,.8,3.,.03,.1,**PARAMS)
        self.assertEqual(set(result.overrides),{'LINEAR_BRAKING','ANGULAR_BRAKING'})
        self.assertLessEqual(result.linear,.03+1e-9)
        self.assertGreater(result.linear,0.)
        self.assertLessEqual(abs(.15*result.angular),result.linear+1e-9)
        stopped=plan_twist(.2,.5,2.,0.,.1,**PARAMS)
        self.assertEqual((stopped.linear,stopped.angular),(0.,0.))

    def test_random_windows_against_independent_dense_feasible_samples(self):
        rng=random.Random(117)
        for _ in range(250):
            v=rng.uniform(0.,.3)
            w=rng.uniform(-min(v,.3-v)/.15,min(v,.3-v)/.15)
            k=rng.uniform(-6.,6.)
            cap=rng.uniform(.01,.3);dt=rng.uniform(.04,.29)
            s=plan_twist(v,w,k,cap,dt,**PARAMS)
            self.assertLessEqual(s.linear,cap+1e-9)
            for wheel in (s.linear-.15*s.angular,s.linear+.15*s.angular):
                self.assertGreaterEqual(wheel,-1e-9)
                self.assertLessEqual(wheel,.3+1e-9)
            win=s.window
            self.assertGreaterEqual(s.linear,win.v_min-1e-9)
            self.assertLessEqual(s.linear,win.v_max+1e-9)
            self.assertGreaterEqual(s.angular,win.w_min-1e-9)
            self.assertLessEqual(s.angular,win.w_max+1e-9)
            if not s.overrides:
                self.assertLessEqual(s.linear-v,.4*dt+1e-9)
                self.assertLessEqual(v-s.linear,.2*dt+1e-9)
                self.assertLessEqual(abs(s.angular-w),2*dt+1e-9)
            # Compare to points sampled without the polygon clipping solver.
            best_error=abs(s.angular-k*s.linear)
            for i in range(21):
                x=win.v_min+(win.v_max-win.v_min)*i/20
                for j in range(21):
                    y=win.w_min+(win.w_max-win.w_min)*j/20
                    if 0<=x-.15*y<=.3 and 0<=x+.15*y<=.3:
                        error=abs(y-k*x)
                        self.assertLessEqual(best_error,error+1e-8)
                        if abs(error-best_error)<1e-9:
                            self.assertGreaterEqual(s.linear,x-1e-8)

    def test_invalid_inputs_do_not_reach_motor_control(self):
        for inputs in ((.1,0.,float('nan'),.3,.1),(.1,0.,0.,.3,0.),
                       (.4,0.,0.,.3,.1),(.1,3.,0.,.3,.1),(-.1,0.,0.,.3,.1)):
            with self.assertRaises(ValueError):plan_twist(*inputs,**PARAMS)


class ControllerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.limits=SpeedLimits(.3,.15,.4,.2,.35,.15)
        self.control=ProfileSpeedControl(self.limits,163,.4,controller='DWPP')
        self.control.applied(.9,(70,70))
        self.target=road(reference_sigma_m=.005,road_model=dict(persistent_history=True))

    def step(self,now=1.,speed=.1,k=3.,target=None,**kwargs):
        target=self.target if target is None else target
        return self.control.command(target,target['control_path_m'],now,speed,k,.005,.075,True,
                                    measured_yaw_rate=kwargs.pop('yaw',0.),**kwargs)

    def test_proposal_does_not_advance_dynamic_state_before_application(self):
        self.step();first=self.control.status.copy()
        self.step(now=1.1)
        self.assertEqual(self.control.status['dwpp']['source'],'MEASURED_SEED')
        self.assertEqual(self.control.status['dwpp']['window'],first['dwpp']['window'])
        self.assertIsNone(self.control.dwpp_state)

    def test_applied_ramp_scaling_seeds_next_window_and_stop_clears_it(self):
        wheels=self.step()
        proposal=self.control.dwpp_proposal
        applied=tuple(round(x*.5) for x in wheels)
        self.control.applied(1.01,applied)
        self.assertIsNotNone(self.control.dwpp_state)
        state=self.control.dwpp_state
        self.step(now=1.2,speed=.05)
        self.assertEqual(self.control.status['dwpp']['source'],'APPLIED_REFERENCE')
        self.assertEqual(self.control.status['dwpp']['current_linear_m_s'],state[1])
        self.assertLess(state[1],proposal[1])
        self.control.applied(1.21,(0,0))
        self.assertIsNone(self.control.dwpp_state)
        self.assertIsNone(self.control.reference)
        self.step(now=1.4,speed=0.)
        self.assertEqual(self.control.status['dwpp']['source'],'MEASURED_SEED')

    def test_search_rotation_and_task_command_do_not_become_pursuit_state(self):
        for actual in ((-60,60),(60,-60),(100,20)):
            self.control.reset();self.control.applied(.9,(70,70));self.step()
            self.control.applied(1.01,actual)
            self.assertIsNone(self.control.dwpp_state)

    def test_large_control_gap_and_missing_flow_use_existing_capped_fallback(self):
        wheels=self.step();self.control.applied(1.01,wheels)
        self.assertIsNone(self.step(now=1.5))
        self.assertEqual(self.control.status['reason'],'CONTROL_GAP')
        self.assertIsNone(self.step(yaw=None))
        self.assertEqual(self.control.status['reason'],'DWPP_MEASURED_YAW_REQUIRED')
        self.assertTrue(any(self.control.limit_fallback((80,120))))

    def test_low_confidence_lowers_speed_without_aging_out_the_path(self):
        self.control.applied(.9,(100,100))
        self.step(speed=.25,k=0.)
        high=self.control.status['target_speed_m_s']
        low=dict(self.target,control_confidence=.08,state='PREDICTED',observed=False,
                 path_observed_mask=[False]*100)
        command=self.step(now=1.2,speed=.25,k=0.,target=low)
        self.assertTrue(any(command))
        self.assertLess(self.control.status['target_speed_m_s'],high)
        self.assertLessEqual(self.control.status['target_speed_m_s'],.03+1e-9)

    def test_post_selection_body_braking_preserves_arc_and_replaces_proposal(self):
        wheels=self.step();before=self.control.dwpp_proposal
        result=self.control.constrain_clearance(wheels,.03)
        after=self.control.dwpp_proposal
        self.assertAlmostEqual(after[1],.03)
        self.assertAlmostEqual(after[2]/after[1],before[2]/before[1])
        self.assertEqual(after[0],result)
        self.assertEqual(self.control.status['requested_wheels'],list(result))
        self.assertIn('BODY_SWEEP_BRAKING',self.control.status['dwpp']['overrides'])
        self.control.applied(1.01,result)
        self.assertAlmostEqual(self.control.dwpp_state[1],.03)

    def test_selected_arc_checked_and_not_learned_as_drivetrain_error(self):
        steering=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,
            cruise=135,speed_control=self.control)
        steering.observe(flow(),1.,False);steering.applied(.99,(70,70))
        limits=[dict(speed_scale=1.,speed_limit_m_s=.3,minimum_clearance_m=.03),
                dict(speed_scale=.2,speed_limit_m_s=.03,minimum_clearance_m=-.01)]
        with patch.object(steering,'envelope',side_effect=limits) as check:
            command=steering.command(road(3.,reference_sigma_m=.005,
                road_model=dict(persistent_history=True)),.1)
        self.assertEqual(check.call_count,2)
        selected=self.control.status['effective_curvature_m_inv']
        self.assertNotAlmostEqual(selected,steering.requested)
        self.assertEqual(steering.status['speed_plan']['mode'],'DWPP')
        self.assertAlmostEqual(check.call_args_list[-1].args[1],selected)
        self.assertLessEqual(self.control.status['target_speed_m_s'],.03+1e-9)
        steering.applied(1.1,command)
        self.assertAlmostEqual(steering.commands[-1][1],selected-steering.trim)

    def test_body_sweep_really_integrates_selected_first_arc(self):
        steering=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,
                                 cruise=135,speed_control=self.control)
        path=self.target['control_path_m']
        # Later pursuit asks for straight travel. An initial turn must still
        # affect the first arc and all later footprint positions.
        with patch('robot.navigation.steering_control.pursuit_point',
                   return_value=dict(point=np.array([.2,0.]))):
            straight=steering.envelope(path,0.,.005,.3,speed_m_s=.1)
            turning=steering.envelope(path,4.,.005,.3,speed_m_s=.1)
        self.assertGreater(straight['minimum_clearance_m'],turning['minimum_clearance_m']+.02)
        self.assertIsNone(straight['first_unsafe_m'])
        self.assertIsNotNone(turning['first_unsafe_m'])

    def test_feedback_converges_with_lag_deadzone_and_original_pwm_ceiling(self):
        for k in (0.,2.,-2.):
            self.control.reset();ramp=CommandRamp(400);v=0.;w=0.;history=[]
            for i in range(100):
                now=1+i*.2
                wheels=self.step(now=now,speed=v,k=k,yaw=w)
                self.assertIsNotNone(wheels)
                actual=ramp.apply(wheels,now)
                self.control.applied(now,actual)
                delivered=compensate_pwm(actual)
                self.assertLessEqual(max(delivered),200)
                self.assertGreaterEqual(min(delivered),0)
                mean=sum(delivered)/2
                desired=max(0.,(mean-20)*.0017) if any(delivered) else 0.
                v+=max(-.2*.2,min(.4*.2,(desired-v)*.8))
                ratio=(delivered[1]-delivered[0])/sum(delivered) if sum(delivered) else 0.
                w=v*ratio/.15
                history.append(v)
            self.assertGreater(sum(history[-20:])/20,.18 if k else .25)
            self.assertLess(max(history),.34)
            self.assertAlmostEqual(self.control.status['effective_curvature_m_inv'],k,delta=.1)

    def test_trial_configuration_and_one_switch_baseline_rollback(self):
        settings=RuntimeSettings.load(ROOT/'maixcam2/curve_follow',{})
        active=ProfileSpeedControl.from_settings(settings)
        self.assertEqual(active.controller,'DWPP')
        baseline=replace(settings,runtime=dict(settings.runtime,speed_profile=dict(
            settings.runtime['speed_profile'],controller='PURE_PURSUIT')))
        self.assertEqual(ProfileSpeedControl.from_settings(baseline).controller,'PURE_PURSUIT')
        for changes in ({'controller':'DWP'}, {'angular_acceleration_rad_s2':float('nan')},
                        {'angular_deceleration_rad_s2':0.}):
            invalid=replace(settings,runtime=dict(settings.runtime,speed_profile=dict(
                settings.runtime['speed_profile'],**changes)))
            with self.assertRaises(ValueError):ProfileSpeedControl.from_settings(invalid)


if __name__=='__main__':unittest.main()
