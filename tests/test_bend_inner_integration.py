"""Paired model inversion preserves the original speed and delivery chain."""
import unittest
from unittest.mock import patch

from tests.support.driving import road
from tests import test_bend_response as bend_tests
from tests import test_motion_gap_hold as hold_tests
from robot.control.controller import CommandRamp, compensate_pwm


class InnerReductionIntegrationTests(unittest.TestCase):
    def control(self, *, enabled, speed=.2, allowed=True, curvature=-2.):
        control = bend_tests.BendResponseTests().control(gain=1.05)
        control.applied(.1, (80., 80.))
        target = road(curvature, reference_sigma_m=.005)
        command = control.command(target, target['control_path_m'], .3, speed,
            curvature, .005, .075, True, measured_yaw_rate=curvature*speed,
            bend_allowed=allowed, bend_inner_reduction=enabled)
        return control, command

    def test_joint_output_preserves_speed_reference_and_feedback_effort_through_ramp(self):
        # The left case's baseline pair is outside the model; the explicit DWPP
        # speed reference must still reach allocation and exact wire delivery.
        for curvature in (-2., 3.):
            with self.subTest(curvature=curvature):
                old, _ = self.control(enabled=False, curvature=curvature)
                joint, command = self.control(enabled=True, curvature=curvature)
                proposal = joint.bend_proposal.copy()
                self.assertEqual(proposal['allocation_mode'], 'PAIRED_MODEL_INVERSE')
                self.assertEqual(proposal['target_speed_m_s'], joint.status['target_speed_m_s'])
                self.assertEqual(proposal['source'], 'MEASURED_PAIRED_PWM_RESPONSE_20261112')
                if curvature > 0:
                    self.assertFalse(proposal['base_model_available'])
                    self.assertIsNone(proposal['base_curvature_m_inv'])
                for field in ('dwpp', 'target_speed_m_s', 'regulator_effort_mean_pwm'):
                    self.assertEqual(joint.status[field], old.status[field])
                ramp = CommandRamp(400)
                ramp.previous, ramp.last = (80., 80.), .1
                applied = ramp.apply(command, .3)
                delivered = compensate_pwm(applied)
                self.assertEqual(list(delivered), proposal['wire'])
                joint.applied(.31, applied, delivered_wheels=delivered)
                self.assertAlmostEqual(joint.applied_mean, proposal['base_mean'])
                self.assertAlmostEqual(joint.applied_mean, old.status['regulator_effort_mean_pwm'])
                self.assertAlmostEqual(joint.dwpp_state[1], old.status['target_speed_m_s'])
                self.assertAlmostEqual(joint.status['applied_curvature_m_inv'],
                                       proposal['estimated_curvature_m_inv'])

    def test_unreachable_body_cap_keeps_braking_explicit_without_claiming_a_model_pair(self):
        control, command = self.control(enabled=True)
        proposal = control.bend_proposal.copy()
        reference = control.status['target_speed_m_s']
        self.assertEqual(proposal['allocation_mode'], 'PAIRED_MODEL_INVERSE')
        self.assertNotEqual(proposal['wire'], proposal['base_wire'])
        braked = control.constrain_clearance(command, .10)
        for got, wheel in zip(braked, proposal['base_wheels']):
            self.assertLessEqual(got, wheel * .10 / max(
                reference, proposal['estimated_speed_m_s']) + 1e-9)
        self.assertGreater(min(braked), 0.)
        self.assertAlmostEqual(braked[0]/braked[1],
                               proposal['base_wheels'][0]/proposal['base_wheels'][1])
        self.assertLessEqual(control.status['target_speed_m_s'], .10)
        self.assertEqual(control.status['actuator_model_reason'], 'UNCALIBRATED_BRAKING')
        self.assertIsNone(control.bend_proposal)
        self.assertFalse(control.status['bend_allocation']['active'])

    def test_shared_mapping_changes_wheels_without_changing_dwpp_or_base_feedback(self):
        for options in (dict(speed=.08), dict(allowed=False), dict(curvature=0.)):
            with self.subTest(options=options):
                a, x = self.control(enabled=False, **options)
                b, y = self.control(enabled=True, **options)
                self.assertTrue(b.status['bend_allocation']['active'])
                self.assertEqual(list(compensate_pwm(y)), b.bend_proposal['wire'])
                for key in ('dwpp','target_speed_m_s','regulator_effort_mean_pwm'):
                    self.assertEqual(a.status[key],b.status[key])
                self.assertAlmostEqual(b.bend_proposal['base_mean'],
                                       a.status['regulator_effort_mean_pwm'])

    def test_steering_checks_actual_joint_arc_and_withdraws_if_unobserved(self):
        # Retain the same confirmed-road setup used by the existing hold tests.
        helper = hold_tests.MotionGapHoldTests()
        steering, profile, _ = helper.seed(enabled=False)
        import numpy as np
        steering.observe(dict(valid=True, interval_s=.15, rotation=np.eye(2),
            translation=np.array([-.03, 0.]), sigma_m=.0002), 1.9, False)
        target = road(-2., reference_sigma_m=.005)
        checked = []
        def envelope(path, curvature, *args, **kwargs):
            checked.append(curvature)
            if len(checked) == 1:
                return dict(helper.clear)
            return dict(speed_scale=1., speed_limit_m_s=None, minimum_clearance_m=None)
        with patch.object(steering, 'envelope', side_effect=envelope):
            result = steering.command(target, .04, bend_inner_reduction=True)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(checked), 3)
        self.assertNotEqual(checked[0], checked[1])
        self.assertEqual(profile.status['actuator_model_reason'], 'BODY_CLEARANCE_UNOBSERVED')
        self.assertIsNone(profile.bend_proposal)
        self.assertEqual(list(compensate_pwm(result)), profile.status['actuator_base_wire'])


if __name__ == '__main__':
    unittest.main()
