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
             high_confidence=False, clearance_limit_m_s=None):
        return self.control.command(self.target if target is None else target,
            self.path if path is None else path, now, speed, curvature, .005, .075, eligible,
            high_confidence=high_confidence,clearance_limit_m_s=clearance_limit_m_s)

    def test_enabled_configuration_and_explicit_legacy_rollback(self):
        settings = RuntimeSettings.load(ROOT/'maixcam2/curve_follow', {})
        active = ProfileSpeedControl.from_settings(settings)
        self.assertIsNotNone(active)
        self.assertEqual(active.limits.deceleration_m_s2, .2)
        self.assertEqual(active.limits.maximum_m_s, .3)
        self.assertEqual(active.high_confidence_gain, 2.)
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
            for acceleration, rise, tracking_gain in (
                (.3,180,1.),(.4,240,1.),(.4,240,1.5),(.4,320,2.),(.4,400,2.))
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

    def test_measured_standstill_can_build_effort_under_the_same_speed_envelope(self):
        steering=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,
            cruise=120,straight_cruise=163,speed_control=self.control)
        for now in (1.,1.2,1.4,1.6):
            steering.observe(dict(motion(0.,0.),interval_s=.2),now,False)
            steering.applied(now-.02,(20,20))
            command=steering.command(self.target,.1)
        self.assertTrue(any(command))
        self.assertTrue(self.control.status['selected'])
        self.assertLessEqual(self.control.status['target_speed_m_s'],self.control.status['ceiling_m_s'])
        self.assertGreater(max(command),20)

    def test_same_confidence_envelope_limits_feedback_and_no_flow_fallback(self):
        ceilings=[];peaks=[]
        for confidence in (.9,.6,.3,.08):
            self.control.reset();self.control.applied(.1,(90,90))
            target=dict(self.target,control_confidence=confidence)
            self.step(target=target,speed=.25,curvature=1.)
            self.assertTrue(self.control.status['selected'])
            ceilings.append(self.control.status['ceiling_m_s'])
            self.assertLessEqual(self.control.status['target_speed_m_s'],ceilings[-1])
            # No flow switches regulators, but cannot discard the envelope.
            self.assertIsNone(self.step(target=target,speed=None))
            wheels=self.control.limit_fallback((100,130))
            peaks.append(max(wheels))
            self.assertGreater(min(wheels),0)
            self.assertAlmostEqual(wheels[0],wheels[1]*100/130,delta=1.)
        self.assertEqual(ceilings,sorted(ceilings,reverse=True))
        self.assertEqual(peaks,sorted(peaks,reverse=True))
        self.assertEqual(peaks[-1],16)

    def test_confidence_recovery_accelerates_without_enabling_low_confidence_boost(self):
        self.control=ProfileSpeedControl(self.limits,163,.4,high_confidence_gain=2.)
        self.control.applied(.1,(80,80))
        self.step(target=dict(self.target,control_confidence=.2),speed=.15,high_confidence=True)
        low=self.control.status.copy()
        self.assertFalse(low['high_confidence_boost'])
        self.control.applied(.3,(60,60))
        self.step(now=.4,target=dict(self.target,control_confidence=.9),speed=.07)
        self.assertGreater(self.control.status['target_speed_m_s'],low['target_speed_m_s'])
        self.assertLessEqual(self.control.status['target_speed_m_s'],
                             low['target_speed_m_s']+self.limits.acceleration_m_s2*.2+1e-9)

    def test_invalid_confidence_has_no_stale_fallback_envelope(self):
        self.step()
        self.assertTrue(any(self.control.limit_fallback((100,100))))
        self.assertIsNone(self.step(target=dict(self.target,control_confidence=float('nan'))))
        self.assertEqual(self.control.limit_fallback((100,100)),(0,0))

    def test_persistent_support_keeps_geometry_without_marking_it_observed(self):
        target=dict(self.target,state='PREDICTED',observed=False,
                    road_model=dict(persistent_history=True),path_observed_mask=[False]*100,
                    control_confidence=.3)
        self.assertIsNotNone(self.step(target=target))
        self.assertTrue(self.control.status['selected'])
        self.assertFalse(any(target['path_observed_mask']))

    def test_clearance_cap_survives_low_speed_acceleration_and_flow_dropout(self):
        for measured in (.02,.08,.2):
            self.control.reset();self.control.applied(.1,(100,100))
            self.step(speed=measured,clearance_limit_m_s=.1)
            self.assertEqual(self.control.status['reason'],'BODY_CLEARANCE')
            self.assertLessEqual(self.control.status['target_speed_m_s'],.1)
            self.step(speed=None,clearance_limit_m_s=.1)
            wheels=self.control.limit_fallback((100,130))
            self.assertLessEqual(max(wheels),round(163*.1/.3))

    def test_finish_priority_keeps_crawl_when_already_touching_boundary(self):
        target=dict(self.target,road_model=dict(persistent_history=True))
        command=self.step(speed=0.,target=target,clearance_limit_m_s=0.)
        self.assertIsNotNone(command);self.assertTrue(any(command))
        self.assertAlmostEqual(self.control.status['clearance_ceiling_m_s'],.03)
        self.assertGreater(self.control.status['target_speed_m_s'],0.)
        self.control.reset()
        self.assertIsNone(self.step(speed=0.,clearance_limit_m_s=0.))
        self.assertEqual(self.control.limit_fallback((100,100)),(0,0))

    def test_recorded_left_and_right_turns_cannot_accelerate_above_body_limit(self):
        import json,math
        fixture=json.loads((ROOT/'tests/fixtures/body_clearance_speed_20261111.json').read_text())
        basis=fixture['calculation_basis']
        for row in fixture['records']:
            s=row['steering'];target=row['target'];old=s['speed_profile_control']
            distance=s['envelope']['first_unsafe_m']
            a=basis['deceleration_m_s2'];delay=basis['delay_s']
            cap=math.sqrt((a*delay)**2+2*a*distance)-a*delay
            self.assertGreater(old['target_speed_m_s'],cap)
            target=dict(target,control_confidence=old['centerline_confidence'])
            self.control.reset();self.control.applied(.1,(old['applied_mean_pwm'],)*2)
            command=self.control.command(target,np.asarray(target['control_path_m']),.2,
                old['measured_speed_m_s'],s['corrected_curvature_m_inv'],
                target['road_model']['sigma_m'],.075,True,clearance_limit_m_s=cap)
            self.assertIsNotNone(command)
            self.assertLessEqual(self.control.status['target_speed_m_s'],cap+1e-9)
            self.assertEqual(self.control.status['reason'],'BODY_CLEARANCE')

    def test_steering_passes_clearance_to_planner_without_repeated_ratio_limits(self):
        from unittest.mock import patch
        steering=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,
            cruise=135,speed_control=self.control)
        envelope=dict(speed_scale=.5,speed_limit_m_s=.1,first_unsafe_m=.1,
                      minimum_clearance_m=-.01,coverage=1.,state='FULL')
        for measured in (True,False):
            self.control.reset()
            steering.observe(dict(motion(.1,.02),valid=measured,interval_s=.2),1.,False)
            steering.applied(.99,(80,80))
            with patch.object(steering,'envelope',return_value=envelope):
                command=steering.command(self.target,.1)
            self.assertTrue(any(command))
            if measured:
                self.assertTrue(self.control.status['selected'])
                self.assertLessEqual(self.control.status['target_speed_m_s'],.1)
            else:
                # A single common cap, rather than cap followed by another .5.
                self.assertEqual(max(command),54)

    def test_handoff_planning_uses_the_stricter_clearance_of_both_routes(self):
        from unittest.mock import patch
        steering=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,
            cruise=135,speed_control=self.control)
        target=dict(self.target,history_continuity_evidence={},candidate_path_m=self.path)
        steering.observe(dict(motion(.1,.02),interval_s=.2),1.,False)
        steering.applied(.99,(80,80))
        envelopes=[dict(speed_scale=1.,speed_limit_m_s=.15,minimum_clearance_m=.01),
                   dict(speed_scale=.4,speed_limit_m_s=.08,minimum_clearance_m=-.01)]
        with patch.object(steering,'envelope',side_effect=envelopes):
            command=steering.command(target,.1)
        self.assertTrue(any(command));self.assertTrue(self.control.status['selected'])
        self.assertAlmostEqual(self.control.status['clearance_ceiling_m_s'],.08)
        self.assertLessEqual(self.control.status['target_speed_m_s'],.08)


if __name__=='__main__':unittest.main()
