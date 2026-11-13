"""Model delivery telemetry and baseline feedback survive downstream changes."""
import unittest

from tests.support import ROOT
from robot.control.controller import compensate_pwm
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits


class BendDeliveryTests(unittest.TestCase):
    def seed(self, *, dwpp=False):
        control=ProfileSpeedControl(SpeedLimits(.3,.15,.4,.2,.35,.15),163,.4,
            controller='DWPP',bend_compensation=True)
        wire=(160,70)
        wheels=tuple(value*(max(wire)-37)/max(wire) for value in wire)
        motion=control.bend_response.predict(wire)
        control.bend_proposal=dict(wheels=wheels,wire=list(wire),base_mean=60.,
            estimated_speed_m_s=motion[0])
        control.status=dict(selected=dwpp,bend_allocation=dict(active=True,wire=list(wire)))
        if dwpp:
            control.dwpp_proposal=(wheels,.20,-.40)
        return control,wheels

    def test_pwm_fallback_matches_wire_and_feeds_baseline_without_dwpp_state(self):
        control,wheels=self.seed()
        control.applied(1.,wheels,delivered_wheels=compensate_pwm(wheels))
        allocation=control.status['bend_allocation']
        self.assertTrue(allocation['delivery_matches'])
        self.assertTrue(allocation['active'])
        self.assertEqual(allocation['delivery_reason'],'MATCHED_ALLOCATION')
        self.assertEqual(allocation['delivered_wire'],[160,70])
        self.assertAlmostEqual(control.applied_mean,60.)
        self.assertFalse(control.status['selected'])
        self.assertIsNone(control.dwpp_state)
        self.assertIsNone(control._gap_hold)
        self.assertIsNone(control._bend_gap_source)

    def test_scaled_pwm_fallback_reports_actual_supported_response(self):
        control,wheels=self.seed()
        actual=tuple(.9*value for value in wheels)
        delivered=compensate_pwm(actual)
        motion=control.bend_response.predict(delivered)
        self.assertIsNotNone(motion)
        control.applied(1.,actual,delivered_wheels=delivered)
        allocation=control.status['bend_allocation']
        self.assertFalse(allocation['delivery_matches'])
        self.assertFalse(allocation['active'])
        self.assertEqual(allocation['proposed_wire'],[160,70])
        self.assertEqual(allocation['delivered_wire'],list(delivered))
        self.assertEqual(allocation['delivery_reason'],'MODIFIED_AFTER_ALLOCATION')
        self.assertAlmostEqual(allocation['delivered_model']['speed_m_s'],motion[0])
        self.assertAlmostEqual(allocation['delivered_model']['curvature_m_inv'],motion[1]/motion[0])
        self.assertAlmostEqual(control.applied_mean,54.)
        self.assertIsNone(control.dwpp_state)

    def test_ramp_outside_model_is_explicit_and_does_not_claim_selected_pair(self):
        control,wheels=self.seed(dwpp=True)
        actual=tuple(.25*value for value in wheels)
        delivered=compensate_pwm(actual)
        self.assertIsNone(control.bend_response.predict(delivered))
        control.applied(1.,actual,delivered_wheels=delivered)
        allocation=control.status['bend_allocation']
        self.assertFalse(allocation['delivery_matches'])
        self.assertFalse(allocation['active'])
        self.assertIsNone(allocation['delivered_model'])
        self.assertEqual(control.status['actuator_model_reason'],'RAMP_OUTSIDE_MODEL_DOMAIN')
        self.assertAlmostEqual(control.applied_mean,15.)

    def test_exact_dwpp_delivery_preserves_reference_and_model_curvature(self):
        control,wheels=self.seed(dwpp=True)
        control.applied(1.,wheels,delivered_wheels=compensate_pwm(wheels))
        self.assertTrue(control.status['bend_allocation']['delivery_matches'])
        self.assertAlmostEqual(control.applied_mean,60.)
        self.assertAlmostEqual(control.dwpp_state[1],.20)
        self.assertAlmostEqual(control.dwpp_state[2]/control.dwpp_state[1],
            control.bend_response.curvature((160,70)))

    def test_missing_wire_is_unknown_not_a_fabricated_confirmation(self):
        control,wheels=self.seed()
        control.applied(1.,wheels)
        allocation=control.status['bend_allocation']
        self.assertIsNone(allocation['delivery_matches'])
        self.assertIsNone(allocation['delivered_wire'])
        self.assertIsNone(allocation['delivered_model'])
        self.assertFalse(allocation['active'])
        self.assertEqual(allocation['delivery_reason'],'DELIVERY_UNKNOWN')

    def test_stop_clears_feedback_and_cannot_preserve_model_execution(self):
        control,_=self.seed(dwpp=True)
        control.applied(1.,(0.,0.),delivered_wheels=(0,0))
        self.assertEqual(control.applied_mean,0.)
        self.assertIsNone(control.dwpp_state)
        self.assertFalse(control.status['bend_allocation']['active'])
        self.assertFalse(control.status['bend_allocation']['delivery_matches'])
        self.assertEqual(control.status['bend_allocation']['delivered_wire'],[0,0])


if __name__=='__main__':
    unittest.main()
