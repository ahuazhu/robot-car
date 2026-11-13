"""A short flow dropout must replan current steering without inventing feedback."""
from dataclasses import replace
from unittest.mock import patch
import unittest

import numpy as np
from tests.support.driving import road
from robot.control.controller import compensate_pwm
from robot.core.road_policy import RoadPolicy
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits
from robot.navigation.steering_control import SteeringControl


class BendPairContinuityTests(unittest.TestCase):
    clear = dict(speed_scale=1., speed_limit_m_s=.30, minimum_clearance_m=.04)

    def seed(self, *, enabled=True, legacy_hold=False, delivered=True, scale=1., speed=.20):
        profile = ProfileSpeedControl(SpeedLimits(.3,.15,.4,.2,.35,.15),163,.4,
            controller='DWPP',bend_compensation=True,motor_rise_pwm_s=400,
            bend_curvature_gain=1.05)
        steering = SteeringControl(replace(RoadPolicy(),maximum_prediction_command_s=.25),
            [0.,0.],(.075,.105,.085),.3,cruise=135,speed_control=profile)
        for now in (1.,1.2,1.4,1.6):
            steering.observe(dict(valid=True,interval_s=.2,rotation=np.eye(2),
                translation=np.array([-speed*.2,0.]),sigma_m=.0002),now,False)
            steering.applied(now-.02,(80,80))
            with patch.object(steering,'envelope',return_value=dict(self.clear)):
                command=steering.command(road(-2,reference_sigma_m=.005),.1,
                    motion_gap_hold=legacy_hold,bend_inner_reduction=enabled)
        self.assertIsNotNone(command)
        if delivered:
            actual=tuple(x*scale for x in command)
            steering.applied(1.7,actual,delivered_wheels=compensate_pwm(actual))
        return steering,profile,command

    def gap(self, steering, *, now=1.75, age=.04, prediction_age=.15,
            enabled=True, legacy_hold=False, target=None, allowed=True,
            valid_prediction=True, envelope=None):
        steering.observe(dict(valid=False,prediction_valid=valid_prediction,
            prediction_age_s=prediction_age,interval_s=.15,rotation=np.eye(2),
            translation=np.array([-.03,0.]),sigma_m=.001,
            source='RECENT VISUAL PREDICTION'),now,True)
        with patch.object(steering,'envelope',return_value=dict(self.clear if envelope is None else envelope)):
            return steering.command(road(-2,reference_sigma_m=.005) if target is None else target,
                age,bend_inner_reduction=enabled,motion_gap_hold=legacy_hold,
                speed_profile_allowed=allowed)

    def details(self, steering):
        return steering.status.get('motion_gap_replan',
            steering.speed_control.status.get('motion_gap_replan',{}))

    def test_gap_uses_current_gentler_curve_not_old_pair_or_old_curvature(self):
        steering,profile,old=self.seed()
        old_reference=profile.reference
        old_k=profile.status['effective_curvature_m_inv']
        before_mean=profile.applied_mean
        before_trim=steering.trim
        command=self.gap(steering,target=road(-1.2,reference_sigma_m=.005))
        self.assertIsNotNone(command)
        self.assertTrue(self.details(steering)['active'])
        self.assertTrue(profile.status['selected'])
        self.assertEqual(profile.status['dwpp']['source'],'PREDICTED_APPLIED_REFERENCE')
        self.assertAlmostEqual(profile.status['dwpp']['requested_curvature_m_inv'],steering.requested+steering.trim)
        self.assertLess(abs(profile.status['effective_curvature_m_inv']),abs(old_k))
        self.assertNotEqual(compensate_pwm(command),compensate_pwm(old))
        self.assertIsNone(profile.status['measured_speed_m_s'])
        self.assertFalse(profile.status['motion_feedback'])
        self.assertFalse(steering.status['motion_measured'])
        self.assertFalse(steering.status['feedback_valid'])
        self.assertEqual(steering.trim,before_trim)
        self.assertFalse(profile.status['high_confidence_boost'])
        self.assertLessEqual(profile.status['target_speed_m_s'],old_reference+1e-9)
        self.assertLessEqual(profile.status['regulator_effort_mean_pwm'],before_mean+1e-9)
        self.assertNotIn('prediction',steering.status)

    def test_new_replan_takes_precedence_over_legacy_frozen_pair_hold(self):
        steering,profile,old=self.seed(legacy_hold=True)
        with patch.object(profile,'hold_motion_gap',side_effect=AssertionError('Legacy frozen hold won')):
            command=self.gap(steering,legacy_hold=True,target=road(-1.2,reference_sigma_m=.005))
        self.assertTrue(self.details(steering)['active'])
        self.assertNotEqual(compensate_pwm(command),compensate_pwm(old))
        self.assertNotEqual(profile.status.get('reason'),'MOTION_GAP_HOLD')

    def test_reversed_visual_intent_cannot_reuse_old_turn_model_allocation(self):
        steering,profile,old=self.seed()
        command=self.gap(steering,target=road(2,reference_sigma_m=.005))
        self.assertIsNotNone(command)
        self.assertGreater(steering.requested,0.)
        self.assertNotEqual(compensate_pwm(command),compensate_pwm(old))
        plan=profile.status
        if plan.get('selected'):
            self.assertGreater(plan['dwpp']['requested_curvature_m_inv'],0.)
        allocation=plan.get('bend_allocation') or {}
        if allocation.get('active'):
            self.assertGreater(allocation['estimated_curvature_m_inv']*steering.requested,0.)

    def test_second_gap_cannot_renew_real_measurement_after_first_is_applied(self):
        steering,profile,_=self.seed()
        command=self.gap(steering)
        self.assertTrue(self.details(steering)['active'])
        steering.applied(1.79,command,delivered_wheels=compensate_pwm(command))
        self.gap(steering,now=1.81,age=.01,prediction_age=.21)
        self.assertFalse(self.details(steering)['active'])
        self.assertFalse(profile.status['selected'])

    def test_prediction_age_cannot_hide_expired_real_source(self):
        steering,profile,_=self.seed()
        self.gap(steering,now=1.86,age=.02,prediction_age=.1)
        self.assertFalse(self.details(steering)['active'])
        self.assertFalse(profile.status['selected'])

    def test_original_capture_and_processing_freshness_are_separate(self):
        steering,profile,_=self.seed()
        self.gap(steering,now=1.8,age=.1,prediction_age=.2)
        details=self.details(steering)
        self.assertTrue(details['active'])
        self.assertAlmostEqual(details['source_age_s'],.2)
        self.assertAlmostEqual(details['decision_age_s'],.3)
        steering,profile,_=self.seed()
        self.assertIsNone(self.gap(steering,age=.36))
        self.assertFalse(self.details(steering)['active'])

    def test_rejected_current_road_cannot_leave_a_source_for_a_second_gap(self):
        steering,profile,_=self.seed()
        rejected=road(-2,reference_sigma_m=.005,valid=False,state='LOST',observed=False)
        self.assertIsNone(self.gap(steering,target=rejected))
        self.gap(steering,now=1.8,age=.01,prediction_age=.2)
        self.assertFalse(self.details(steering).get('active',False))
        self.assertFalse(profile.status['selected'])

    def test_unsent_or_modified_source_cannot_authorize_continuity(self):
        for delivered,scale in ((False,1.),(True,.8)):
            with self.subTest(delivered=delivered,scale=scale):
                steering,profile,_=self.seed(delivered=delivered,scale=scale)
                self.gap(steering)
                self.assertFalse(self.details(steering)['active'])
                self.assertFalse(profile.status['selected'])

    def test_task_uncertain_road_and_invalid_prediction_do_not_replan(self):
        cases=[dict(allowed=False),dict(valid_prediction=False),dict(prediction_age=.25),
            dict(target=road(-2,reference_sigma_m=.005,transition_pending=True)),
            dict(target=road(-2,reference_sigma_m=.005,state='PREDICTED',observed=False)),
            dict(target=road(-2,reference_sigma_m=.005,partial_boundary_inference=True,
                road_model={'persistent_history':True}))]
        for options in cases:
            with self.subTest(options=options):
                steering,profile,_=self.seed()
                self.gap(steering,**options)
                self.assertFalse(self.details(steering).get('active',False))
                self.assertFalse(profile.status['selected'])

    def test_unknown_or_tight_actual_arc_cannot_publish_unchecked_model_pair(self):
        for cap,clearance in ((None,None),(.08,.005)):
            with self.subTest(cap=cap,clearance=clearance):
                steering,profile,_=self.seed()
                command=self.gap(steering,envelope=dict(speed_scale=1.,speed_limit_m_s=cap,
                    minimum_clearance_m=clearance))
                allocation=profile.status.get('bend_allocation') or {}
                self.assertFalse(allocation.get('active',False))
                if profile.status.get('selected') and cap is not None:
                    self.assertLessEqual(profile.status['target_speed_m_s'],cap+1e-9)

    def test_startup_reset_and_switch_off_cannot_leave_a_gap_source(self):
        steering,profile,_=self.seed(speed=.08)
        self.gap(steering)
        self.assertFalse(self.details(steering).get('active',False))
        self.assertFalse(profile.status['selected'])
        steering,profile,_=self.seed()
        steering.reset()
        self.gap(steering)
        self.assertFalse(self.details(steering).get('active',False))
        self.assertFalse(profile.status['selected'])
        steering,profile,_=self.seed()
        self.gap(steering,enabled=False)
        self.gap(steering,now=1.8,age=.01,prediction_age=.2)
        self.assertFalse(self.details(steering).get('active',False))
        self.assertFalse(profile.status['selected'])

    def test_old_switch_off_behavior_remains_unchanged(self):
        new_source,_,_=self.seed()
        baseline,_,_=self.seed(enabled=False)
        a=self.gap(new_source,enabled=False)
        b=self.gap(baseline,enabled=False)
        self.assertEqual(a,b)
        self.assertFalse(new_source.speed_control.status['selected'])
        self.assertFalse(baseline.speed_control.status['selected'])


if __name__=='__main__':
    unittest.main()
