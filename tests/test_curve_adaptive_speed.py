"""Straight effort increases only with current measured road support.

These geometry checks preserve curvature and baseline recovery behavior;
actual speed/track performance still requires a recorded road run.
"""
import math
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/curve_follow'))
from steering_control import SteeringControl
from road_policy import RoadPolicy
from controller import CommandRamp,compensate_pwm
from crosswalk import CrosswalkPolicy,CrosswalkTask


def road(curvature=0.,start=-.2,end=.8,**changes):
    s=np.linspace(start,end,100)
    p=(np.c_[s,np.zeros(len(s))] if not curvature else
       np.c_[np.sin(s*curvature)/curvature,(1-np.cos(s*curvature))/curvature])
    result=dict(valid=True,state='OBSERVED',observed=True,confidence=1.,
                control_target_m=[.18,0.],control_path_m=p,path=p,
                path_observed_mask=[True]*len(p),path_curvature_m_inv=curvature)
    result.update(changes)
    return result


def control(result,fast=72.,measured=True,body=(.075,.105,.085)):
    c=SteeringControl(RoadPolicy(),[0.,0.],body,.3,cruise=60.,straight_cruise=fast)
    for now in (1.,1.2,1.4,1.6):
        c.observe(dict(valid=measured,interval_s=.2,rotation=np.eye(2),translation=np.array([-.02,0.]),
                       sigma_m=.0002,speed_m_s=.1,yaw_rad_s=0.),now,False)
        command=c.command(result,age_s=.1)
    return c,command


class AdaptiveSpeedTests(unittest.TestCase):
    def test_clear_straight_increases_effort_with_same_steering_ratio_and_hardware_cap(self):
        for y in (0.,.006,-.006):
            r=road();r['path'][:,1]+=y
            old,a=control(r,60.);new,b=control(r)
            self.assertGreater(sum(b),sum(a))
            self.assertTrue(new.status['speed_plan']['boost_eligible'])
            self.assertAlmostEqual((a[1]-a[0])/sum(a),(b[1]-b[0])/sum(b),delta=.02)
            self.assertLessEqual(max(compensate_pwm(b)),109)
    def test_left_right_bends_retain_the_baseline_commands(self):
        for curvature in (-4.,-2.,2.,4.):
            r=road(curvature)
            old,a=control(r,60.);new,b=control(r)
            self.assertEqual(b,a)
            self.assertEqual(new.status['speed_plan']['selected_cruise_pwm'],60.)
    def test_upcoming_bend_removes_boost_even_before_the_target_turns(self):
        r=road(path_curvature_m_inv=3.)
        _,a=control(r,60.);new,b=control(r)
        self.assertEqual(a,b)
        self.assertEqual(new.status['speed_plan']['selected_cruise_pwm'],60.)
    def test_curve_load_reduces_extra_effort_continuously(self):
        selected=[]
        for curvature in (0.,.5,1.,1.5,2.):
            new,_=control(road(path_curvature_m_inv=curvature))
            selected.append(new.status['speed_plan']['selected_cruise_pwm'])
        self.assertEqual(selected[0],72.)
        self.assertEqual(selected[-1],60.)
        self.assertTrue(all(a>=b for a,b in zip(selected,selected[1:])))
    def test_prediction_missing_flow_or_unobserved_goal_cannot_enable_extra_effort(self):
        cases=[(road(state='PREDICTED',observed=False),True),
               (road(path_observed_mask=[False]*100),True),(road(),False)]
        for r,measured in cases:
            _,a=control(r,60.,measured);new,b=control(r,72.,measured)
            self.assertEqual(a,b)
            self.assertFalse(new.status['speed_plan']['boost_eligible'])
    def test_short_far_or_unverified_body_support_keeps_baseline(self):
        for start,end in ((.06,.12),(.4,.7)):
            r=road(start=start,end=end)
            _,a=control(r,60.);new,b=control(r)
            self.assertEqual(a,b)
            self.assertFalse(new.status['speed_plan']['boost_eligible'])
        _,a=control(road(),60.,body=None);new,b=control(road(),body=None)
        self.assertEqual(a,b);self.assertFalse(new.status['speed_plan']['boost_eligible'])
    def test_pending_handoff_and_blocked_body_cannot_gain_speed(self):
        for r in (road(transition_pending=True),road()):
            if not r.get('transition_pending'):r['path']=r['path']+np.array([0.,.13])
            _,a=control(r,60.);new,b=control(r)
            self.assertEqual(a,b)
    def test_crosswalk_ceiling_remains_the_same_for_both_cruise_levels(self):
        p=CrosswalkPolicy.from_config(dict(mode='yield',body_front_m=.02,body_rear_m=.16,
            body_half_width_m=.085,geometry_verified=True,brake_lamp_verified=True,
            brake_deceleration_m_s2=.08,approach_speed_m_s=.06,stop_clearance_m=.08))
        outputs=[]
        for cruise in (60.,72.):
            task=CrosswalkTask(p)
            obs=dict(normal=[1.,0.],near_m=.6,far_m=.7,sigma_m=.02,observed_at=0.)
            motion=dict(valid=True,speed_m_s=.05,yaw_rad_s=0.,rotation=np.eye(2),
                        translation=np.zeros(2),sigma_m=.001,travel_m=0.)
            state=task.update(obs,motion,0.,True)
            outputs.append(task.limit_command((cruise,cruise),state,motion))
            self.assertEqual(task.speed_ceiling,.06)
        self.assertEqual(outputs[0],outputs[1])
    def test_existing_ramp_still_limits_start_and_honors_zero_immediately(self):
        _,command=control(road());ramp=CommandRamp()
        samples=[ramp.apply(command,t*.1) for t in range(8)]
        self.assertLess(max(samples[0]),max(command))
        self.assertTrue(all(max(b)-max(a)<=18.01 for a,b in zip(samples,samples[1:])))
        self.assertEqual(ramp.apply((0,0),.9),(0,0))
    def test_production_speed_step_is_bounded_and_preserves_bend_commands(self):
        for fast in (72.,78.):
            _,straight=control(road(),fast)
            self.assertGreater(sum(straight),120)
            self.assertLessEqual(max(compensate_pwm(straight)),fast+37)
            for curvature in (-3.,-2.,2.,3.):
                _,old=control(road(curvature),60.)
                _,new=control(road(curvature),fast)
                self.assertEqual(new,old)

    def test_a_single_good_frame_does_not_enable_boost_and_weak_frame_keeps_baseline(self):
        c=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,cruise=60.,straight_cruise=78.)
        m=dict(valid=True,interval_s=.15,rotation=np.eye(2),translation=np.array([-.015,0.]),sigma_m=.0002)
        for now in (0.,.15,.30):
            c.observe(m,now,False);command=c.command(road(),age_s=.1)
            self.assertEqual(c.status['speed_plan']['selected_cruise_pwm'],60.)
            self.assertTrue(any(command))
        c.observe(m,.45,False);c.command(road(),age_s=.1)
        self.assertGreater(c.status['speed_plan']['selected_cruise_pwm'],60.)
        weak=road(confidence=.6)
        c.observe(m,.60,False);command=c.command(weak,age_s=.1)
        _,baseline=control(weak,fast=60.)
        self.assertEqual(command,baseline)
        self.assertTrue(any(command))
        self.assertFalse(c.status['speed_plan']['boost_eligible'])
        c.observe(m,.75,False);c.command(road(),age_s=.1)
        self.assertEqual(c.status['speed_plan']['selected_cruise_pwm'],60.)

    def test_repeated_command_evaluation_cannot_replace_elapsed_observation_time(self):
        c=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,cruise=60.,straight_cruise=78.)
        m=dict(valid=True,interval_s=.15,rotation=np.eye(2),translation=np.array([-.015,0.]),sigma_m=.0002)
        c.observe(m,1.,False)
        for _ in range(20):
            c.command(road(),age_s=.1)
            self.assertEqual(c.status['speed_plan']['selected_cruise_pwm'],60.)

    def test_invalid_speed_settings_are_rejected(self):
        for fast in (float('nan'),float('inf'),59.,-1.):
            with self.assertRaises(ValueError):control(road(),fast)


if __name__=='__main__':unittest.main()
