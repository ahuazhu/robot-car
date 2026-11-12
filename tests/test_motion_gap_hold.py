"""An opt-in, single-frame continuation of an already delivered DWPP pair."""
from tests.support.driving import road
from unittest.mock import patch
from dataclasses import replace
import unittest
import numpy as np

from robot.control.controller import compensate_pwm
from robot.core.road_policy import RoadPolicy
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits
from robot.navigation.steering_control import SteeringControl


class MotionGapHoldTests(unittest.TestCase):
    clear=dict(speed_scale=1.,speed_limit_m_s=.30,minimum_clearance_m=.04)
    baseline=dict(speed_scale=.8,speed_limit_m_s=.20,minimum_clearance_m=-.005)

    def seed(self, enabled=True, delivered=True, scale=1.):
        speed=ProfileSpeedControl(SpeedLimits(.3,.15,.4,.2,.35,.15),163,.4,
            controller='DWPP',bend_compensation=True,motor_rise_pwm_s=400,
            bend_curvature_gain=1.05)
        steering=SteeringControl(replace(RoadPolicy(),maximum_prediction_command_s=.25),
            [0.,0.],(.075,.105,.085),.3,cruise=135,speed_control=speed)
        for now in (1.,1.2,1.4,1.6):
            steering.observe(dict(valid=True,interval_s=.2,rotation=np.eye(2),
                translation=np.array([-.04,0.]),sigma_m=.0002),now,False)
            steering.applied(now-.02,(80,80))
            with patch.object(steering,'envelope',side_effect=[self.baseline,self.clear,self.baseline]):
                command=steering.command(road(-2,reference_sigma_m=.005),.1,
                                         motion_gap_hold=enabled)
        self.assertTrue(speed.status['bend_allocation']['active'])
        if delivered:
            applied=tuple(value*scale for value in command)
            steering.applied(1.7,applied,delivered_wheels=compensate_pwm(applied))
        return steering,speed,command

    def gap(self, steering, *, now=1.75, age=.04, prediction_age=.15,
            enabled=True, target=None, clearance=None, allowed=True, valid_prediction=True):
        steering.observe(dict(valid=False,prediction_valid=valid_prediction,
            prediction_age_s=prediction_age,interval_s=.15,rotation=np.eye(2),
            translation=np.array([-.03,0.]),sigma_m=.001,
            source='RECENT VISUAL PREDICTION'),now,True)
        with patch.object(steering,'envelope',return_value=self.clear if clearance is None else clearance):
            return steering.command(road(-2,reference_sigma_m=.005) if target is None else target,
                age,motion_gap_hold=enabled,speed_profile_allowed=allowed)

    def test_one_gap_keeps_exact_delivered_pair_without_feedback_or_new_lease(self):
        steering,speed,previous=self.seed()
        before_mean=speed.applied_mean
        before_reference=speed.reference
        before_trim=steering.trim
        with patch('robot.navigation.profile_speed_control.plan_twist',
                   side_effect=AssertionError('No new dynamic window on predicted motion')):
            command=self.gap(steering)
        self.assertEqual(command,previous)
        self.assertEqual(compensate_pwm(command),compensate_pwm(previous))
        self.assertTrue(speed.status['selected'])
        self.assertTrue(speed.status['motion_gap_hold']['active'])
        self.assertEqual(speed.status['dwpp']['source'],'HELD_APPLIED_PLAN')
        self.assertIsNone(speed.status['measured_speed_m_s'])
        self.assertFalse(steering.status['motion_measured'])
        self.assertFalse(steering.status['feedback_valid'])
        self.assertFalse(speed.status['high_confidence_boost'])
        self.assertEqual(steering.trim,before_trim)
        self.assertEqual(speed.reference,before_reference)
        self.assertNotIn('prediction',steering.status)
        steering.applied(1.79,command,delivered_wheels=compensate_pwm(command))
        self.assertAlmostEqual(speed.applied_mean,before_mean)
        self.assertIsNone(speed._gap_hold)

    def test_second_predicted_frame_cannot_renew_the_measured_source(self):
        steering,speed,_=self.seed()
        command=self.gap(steering)
        steering.applied(1.79,command,delivered_wheels=compensate_pwm(command))
        self.gap(steering,now=1.81,age=.01,prediction_age=.21)
        self.assertFalse(speed.status['selected'])
        self.assertEqual(speed.status['motion_gap_hold']['reason'],'NO_APPLIED_MEASURED_PLAN')

    def test_capture_prediction_and_command_processing_keep_their_original_clocks(self):
        steering,speed,previous=self.seed()
        command=self.gap(steering,now=1.8,age=.1,prediction_age=.2)
        self.assertEqual(command,previous)
        self.assertTrue(speed.status['motion_gap_hold']['active'])
        self.assertAlmostEqual(speed.status['motion_gap_hold']['source_age_s'],.2)
        self.assertAlmostEqual(speed.status['motion_gap_hold']['decision_age_s'],.3)
        self.assertNotIn('prediction',steering.status)

    def test_expired_capture_prediction_cannot_hold(self):
        for now,age,prediction_age in ((1.86,.05,.26),(1.85,0.,.25)):
            with self.subTest(now=now):
                steering,speed,_=self.seed()
                self.gap(steering,now=now,age=age,prediction_age=prediction_age)
                self.assertFalse(speed.status['selected'])
                self.assertEqual(speed.status['motion_gap_hold']['reason'],'MOTION_PREDICTION_EXPIRED')

    def test_processing_beyond_original_command_freshness_cannot_hold(self):
        steering,speed,_=self.seed()
        self.assertIsNone(self.gap(steering,age=.36))
        self.assertFalse(steering.status['motion_gap_hold']['active'])
        self.assertEqual(steering.status['motion_gap_hold']['reason'],'COMMAND_EXPIRED')
        self.assertIsNone(speed._gap_hold)

    def test_valid_prediction_cannot_hide_an_expired_applied_source(self):
        steering,speed,_=self.seed()
        self.gap(steering,now=1.9,age=.02,prediction_age=.1)
        self.assertFalse(speed.status['selected'])
        self.assertEqual(speed.status['motion_gap_hold']['reason'],'MEASURED_PLAN_EXPIRED')

    def test_unsent_or_ramp_changed_proposal_is_not_a_hold_source(self):
        for delivered,scale in ((False,1.),(True,.8)):
            with self.subTest(delivered=delivered,scale=scale):
                steering,speed,_=self.seed(delivered=delivered,scale=scale)
                self.gap(steering)
                self.assertFalse(speed.status['selected'])
                self.assertEqual(speed.status['motion_gap_hold']['reason'],'NO_APPLIED_MEASURED_PLAN')

    def test_reverse_intent_never_holds_previous_turn(self):
        steering,speed,_=self.seed()
        self.gap(steering,target=road(2,reference_sigma_m=.005))
        self.assertFalse(speed.status['selected'])
        self.assertEqual(speed.status['motion_gap_hold']['reason'],'TURN_INTENT_CHANGED')

    def test_new_tighter_speed_envelope_rejects_hold(self):
        steering,speed,_=self.seed()
        self.gap(steering,target=road(-2,reference_sigma_m=.005,
                                     confidence=.3,observation_confidence=1.))
        self.assertFalse(speed.status['selected'])
        self.assertEqual(speed.status['motion_gap_hold']['reason'],'CURRENT_SPEED_LIMIT')
        self.assertLess(speed.status['confidence_ceiling_m_s'],.2)

    def test_new_tighter_or_unknown_body_envelope_rejects_hold(self):
        for cap,clearance,reason in ((.1,.02,'CURRENT_CLEARANCE_LIMIT'),
                                    (None,None,'CURRENT_CLEARANCE_UNOBSERVED'),
                                    (None,float('nan'),'CURRENT_CLEARANCE_UNOBSERVED')):
            with self.subTest(cap=cap,clearance=clearance):
                steering,speed,_=self.seed()
                envelope=dict(speed_scale=1.,speed_limit_m_s=cap,minimum_clearance_m=clearance)
                self.gap(steering,clearance=envelope)
                self.assertFalse(speed.status['selected'])
                self.assertEqual(speed.status['motion_gap_hold']['reason'],reason)
                self.assertIsNone(speed.bend_proposal)
                self.assertEqual(speed.status['clearance_ceiling_m_s'],cap)

    def test_tasks_transitions_and_missing_prediction_keep_existing_fallback(self):
        cases=[dict(allowed=False),dict(valid_prediction=False),
               dict(target=road(-2,reference_sigma_m=.005,transition_pending=True)),
               dict(target=road(-2,reference_sigma_m=.005,state='PREDICTED',observed=False))]
        for options in cases:
            with self.subTest(options=options):
                steering,speed,_=self.seed()
                self.gap(steering,**options)
                self.assertFalse(speed.status['selected'])
                self.assertFalse(steering.status['motion_gap_hold']['active'])

    def test_switch_off_clears_source_and_cannot_be_reenabled_during_gap(self):
        steering,speed,_=self.seed()
        self.gap(steering,enabled=False)
        self.assertNotIn('motion_gap_hold',speed.status)
        self.gap(steering,now=1.8,age=.01,prediction_age=.2)
        self.assertFalse(speed.status['selected'])
        self.assertEqual(speed.status['motion_gap_hold']['reason'],'NO_APPLIED_MEASURED_PLAN')

    def test_switch_off_matches_the_unchanged_baseline(self):
        with_source,_,_=self.seed(enabled=True)
        baseline,_,_=self.seed(enabled=False)
        a=self.gap(with_source,enabled=False)
        b=self.gap(baseline,enabled=False)
        self.assertEqual(a,b)
        self.assertEqual(with_source.speed_control.status,baseline.speed_control.status)

    def test_next_reliable_measurement_returns_to_normal_controller(self):
        steering,speed,_=self.seed()
        held=self.gap(steering)
        steering.applied(1.79,held,delivered_wheels=compensate_pwm(held))
        steering.observe(dict(valid=True,interval_s=.15,rotation=np.eye(2),
            translation=np.array([-.03,0.]),sigma_m=.0002),1.9,False)
        with patch.object(steering,'envelope',return_value=self.clear):
            command=steering.command(road(-2,reference_sigma_m=.005),.04,motion_gap_hold=True)
        self.assertIsNotNone(command)
        self.assertTrue(speed.status['selected'])
        self.assertNotEqual(speed.status['reason'],'MOTION_GAP_HOLD')
        self.assertIsNotNone(speed.status['measured_speed_m_s'])
        self.assertFalse(speed.status['motion_gap_hold']['active'])


if __name__=='__main__':unittest.main()
