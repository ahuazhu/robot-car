"""Real speed-controller selection, geometry limits and actuator supervision."""

from tests.support import ROOT
from tests.support.driving import road
from tests.support.crosswalk import configured, motion, observation
from dataclasses import replace
import unittest
import numpy as np

from robot.config import RuntimeSettings
from robot.control.controller import CommandRamp, compensate_pwm
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits
from robot.navigation.steering_control import SteeringControl
from robot.core.road_policy import RoadPolicy
from robot.tasks.crosswalk import CrosswalkTask


class ProfileControlTests(unittest.TestCase):
    def setUp(self):
        self.limits = SpeedLimits(.3, .15, .3, .2, .35, .15)
        self.control = ProfileSpeedControl(self.limits, 163, .4)
        self.target = road(reference_sigma_m=.005)
        self.path = self.target['control_path_m']

    def step(self, now=.2, speed=.1, curvature=0., target=None, path=None, eligible=True,
             high_confidence=False):
        return self.control.command(self.target if target is None else target,
            self.path if path is None else path, now, speed, curvature, .005, .075, eligible,
            high_confidence=high_confidence)

    def test_enabled_configuration_and_explicit_legacy_rollback(self):
        settings = RuntimeSettings.load(ROOT/'maixcam2/curve_follow', {})
        active = ProfileSpeedControl.from_settings(settings)
        self.assertIsNotNone(active)
        self.assertEqual(active.limits.deceleration_m_s2, .2)
        self.assertEqual(active.limits.maximum_m_s, .3)
        self.assertEqual(active.high_confidence_gain, 1.5)
        self.assertIsNone(ProfileSpeedControl.from_settings(replace(settings,
            runtime=dict(settings.runtime, speed_profile={'mode':'LEGACY'}))))
        for config in ({'mode':'ACTVE'}, {'mode':'ACTIVE','maximum_m_s':.4},
                       {'mode':'ACTIVE','maximum_m_s':True}, {'mode':'ACTIVE','unknown':1}):
            with self.assertRaises(ValueError):
                ProfileSpeedControl.from_settings(replace(settings,
                    runtime=dict(settings.runtime, speed_profile=config)))

    def test_high_confidence_closes_speed_error_faster_with_identical_speed_envelope(self):
        outputs, statuses = [], []
        for tracking in (1., 1.5):
            self.control = ProfileSpeedControl(self.limits, 163, .4,
                                               high_confidence_gain=tracking)
            self.control.applied(.1, (70, 70))
            self.step(speed=.15, high_confidence=True)
            self.control.applied(.3, (70, 70))
            outputs.append(self.step(now=.4, speed=.15, high_confidence=True))
            statuses.append(self.control.status)
        self.assertGreater(sum(outputs[1]), sum(outputs[0]))
        self.assertEqual(statuses[0]['ceiling_m_s'], statuses[1]['ceiling_m_s'])
        self.assertGreater(statuses[1]['target_speed_m_s'], statuses[0]['target_speed_m_s'])
        self.assertTrue(statuses[1]['high_confidence_boost'])
        self.assertLessEqual(max(compensate_pwm(outputs[1])), 200)

    def test_lower_confidence_short_support_and_overspeed_get_no_extra_effort(self):
        for args in ({'high_confidence':False}, {'high_confidence':True,'speed':.4},
                     {'high_confidence':True, 'path':self.path[:30],
                      'target':dict(self.target,path_observed_mask=[True]*30)}):
            outputs=[]
            for gain in (1., 1.5):
                self.control=ProfileSpeedControl(self.limits,163,.4,high_confidence_gain=gain)
                self.control.applied(.1,(70,70))
                outputs.append(self.step(**args))
            self.assertEqual(outputs[0],outputs[1])
            self.assertFalse(self.control.status.get('high_confidence_boost',False))

    def test_invalid_high_confidence_gain_is_rejected(self):
        for gain in (0., .9, 2.1, True, '1.5', float('nan')):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                ProfileSpeedControl(self.limits,163,.4,high_confidence_gain=gain)

    def test_live_steering_only_boosts_confirmed_gentle_measured_path(self):
        for quality, curvature, measured, allowed, expected in (
            (.9,0.,True,True,True), (.7,0.,True,True,False),
            (.9,4.,True,True,False), (.9,0.,False,True,False),
            (.9,0.,True,False,False),
        ):
            with self.subTest(quality=quality,curvature=curvature,measured=measured,allowed=allowed):
                control=ProfileSpeedControl(self.limits,163,.4,high_confidence_gain=1.5)
                steering=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,
                    cruise=120,straight_cruise=163,confirmed_cruise=140,speed_control=control)
                target=road(curvature,reference_sigma_m=.005,confidence=quality,
                            observation_confidence=quality)
                for now in (1.,1.2,1.4,1.6):
                    scene=dict(motion(.1,.02),valid=measured,interval_s=.2)
                    steering.observe(scene,now,False)
                    steering.applied(now-.02,(60,60))
                    steering.command(target,.1,speed_profile_allowed=allowed)
                self.assertEqual(control.status.get('high_confidence_boost',False),expected)

    def test_future_bend_changes_actual_requested_speed_before_steering_turns(self):
        self.control.applied(.1, (70, 70))
        self.step(speed=.25)
        straight = self.control.status['ceiling_m_s']
        s = np.linspace(0, .8, 100)
        curved = np.c_[s, .6 * np.maximum(0, s - .1) ** 2]
        self.control.reset();self.control.applied(.1, (70, 70))
        command = self.step(speed=.25, path=curved)
        self.assertIsNotNone(command)
        self.assertLess(self.control.status['ceiling_m_s'], straight)
        self.assertTrue(self.control.status['selected'])

    def test_no_distance_credit_for_unseen_prefix_or_inferred_tail(self):
        path = np.c_[np.linspace(.1,.6,50),np.zeros(50)]
        target = dict(self.target,path_observed_mask=[True]*50)
        self.step(target=target,path=path)
        first = self.control.status['ceiling_m_s']
        self.control.reset()
        self.step(target=target,path=path+[.15,0])
        self.assertAlmostEqual(self.control.status['ceiling_m_s'],first)
        self.control.reset()
        self.step(target=dict(target,path_observed_mask=[True]*20+[False]*30),path=path)
        self.assertLess(self.control.status['ceiling_m_s'],first)
        self.control.reset()
        self.step(target=dict(target,path_observed_mask=[False]*10+[True]*40),path=path)
        self.assertEqual(self.control.status['first_supported_point_m'],path[10].tolist())
        self.assertLess(self.control.status['ceiling_m_s'],first)

    def test_missing_flow_uncertainty_prediction_and_body_rejection_use_legacy_fallback(self):
        for args in ({'speed':None}, {'eligible':False},
                     {'target':dict(self.target,state='PREDICTED',observed=False)},
                     {'target':dict(self.target,transition_pending=True)},
                     {'target':dict(self.target,reference_sigma_m=None)},
                     {'target':dict(self.target,path_observed_mask=[False]*100)}):
            self.control.reset()
            self.assertIsNone(self.step(**args))
            self.assertFalse(self.control.status['selected'])

    def test_applied_effort_not_unsent_requests_drives_feedback_and_zero_resets(self):
        self.control.applied(.1,(40,40))
        a=self.step(speed=.1)
        # A task or ramp delivered only10, so the next request starts there.
        self.control.applied(.3,(10,10))
        b=self.step(now=.4,speed=.1)
        self.assertLess(max(b),max(a)+10)
        self.assertEqual(self.control.status['applied_mean_pwm'],10)
        self.control.applied(.5,(0,0))
        self.assertIsNone(self.control.reference)
        self.step(now=.6,speed=0.)
        self.assertEqual(self.control.status['applied_mean_pwm'],0)
        self.assertLessEqual(self.control.status['target_speed_m_s'],.015+1e-9)

    def test_wheel_ratio_peak_cap_and_immediate_lower_speed_reference(self):
        for curvature in (-4.,0.,4.):
            self.control.reset();self.control.applied(.1,(163,163))
            wheels=self.step(speed=.29,curvature=curvature)
            self.assertLessEqual(max(compensate_pwm(wheels)),200)
            self.assertAlmostEqual((wheels[1]-wheels[0])/sum(wheels),.15*curvature,delta=.02)
            self.assertLessEqual(self.control.status['ceiling_m_s']*(1+abs(.15*curvature)),.3+1e-9)

    def test_feedback_converges_with_camera_lag_and_motor_deadzone(self):
        for dt, acceleration, rise, tracking_gain in (
            (dt, acceleration, rise, tracking_gain) for dt in (.15, .22, .28)
            for acceleration, rise, tracking_gain in ((.3,180,1.),(.4,240,1.),(.4,240,1.5))
        ):
            self.control = ProfileSpeedControl(
                replace(self.limits, acceleration_m_s2=acceleration), 163, .4,
                high_confidence_gain=tracking_gain)
            ramp=CommandRamp(rise);speed=0.;measured=0.;history=[]
            for i in range(100):
                now=i*dt
                self.control.applied(now-.02,ramp.previous)
                wheels=self.step(now=now,speed=measured,high_confidence=True)
                if wheels is None:self.fail(self.control.status)
                actual=compensate_pwm(ramp.apply(wheels,now))
                desired=max(0.,(sum(actual)/2-20)*.0017) if any(actual) else 0.
                speed+=max(-.2*dt,min(.4*dt,(desired-speed)*min(1.,dt/.25)))
                history.append(speed)
                measured=history[-2] if len(history)>1 else 0.
            self.assertGreater(np.mean(history[-20:]),.26)
            self.assertLess(max(history),.34)

    def test_steering_selects_profile_and_crosswalk_zero_remains_authoritative(self):
        steering=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,
            cruise=120,straight_cruise=163,confirmed_cruise=140,speed_control=self.control)
        for now in (1.,1.2,1.4,1.6):
            scene=dict(motion(.1,.02),interval_s=.2)
            steering.observe(scene,now,False)
            steering.applied(now-.02,(60,60))
            command=steering.command(self.target,.1)
        self.assertTrue(steering.status['speed_profile_control']['selected'])
        self.assertEqual(steering.status['speed_plan']['mode'],'PROFILE')
        self.assertTrue(any(command))
        legacy_command=steering.command(self.target,.1,speed_profile_allowed=False)
        self.assertIsNotNone(legacy_command)
        self.assertFalse(steering.status['speed_profile_control']['selected'])
        self.assertEqual(steering.status['speed_profile_control']['reason'],'TASK_OWNS_SPEED')
        task=CrosswalkTask(configured())
        for now in (0.,.1,.2,.3,.4,.5):
            state=task.update(observation(now),motion(),now,True,True,True,True)
        self.assertEqual(task.phase,'HOLD')
        self.assertEqual(task.limit_command(command,state,scene),(0,0))

    def test_stationary_or_unresolved_startup_uses_proven_baseline_effort(self):
        steering=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,
            cruise=120,straight_cruise=163,speed_control=self.control)
        for now in (1.,1.2,1.4,1.6):
            steering.observe(dict(motion(0.,0.),interval_s=.2),now,False)
            command=steering.command(self.target,.1)
        self.assertTrue(any(command))
        self.assertFalse(steering.status['speed_profile_control']['selected'])
        self.assertEqual(steering.status['speed_profile_control']['reason'],
                         'MEASURED_FORWARD_SPEED_UNRESOLVED')


if __name__=='__main__':unittest.main()
