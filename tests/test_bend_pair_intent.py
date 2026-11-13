"""The paired actuator inverse follows current pursuit, including bend exit correction."""
import copy
import unittest

from tests.support.driving import road
from tests.test_bend_response import logical
from robot.core.geometry import arc, describe_curve
from robot.navigation.supported_path import supported_segment
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits


# Actual71748aa/test.3 CONTROL frame66 geometry and confidence, from
# road-test-review/control.json. Frame1061 has TRACE only and is deliberately
# not presented as a complete geometry replay. The test isolates this input's
# gate decision; its regulator state below is a controlled fixture.
FRAME66 = {'valid': True,
 'state': 'OBSERVED',
 'observed': True,
 'transition_pending': False,
 'confidence': 0.7512844685125487,
 'reference_sigma_m': 0.01326810886097759,
 'control_path_m': [[0.12701517065278106, -0.022572309400045523],
                    [0.1392607514219955, -0.031406230396389456],
                    [0.1514165157092063, -0.04036333525564107],
                    [0.16351297211437707, -0.049400375017314015],
                    [0.17557685604088488, -0.0584808532689595],
                    [0.18762858562033718, -0.06757745800224751],
                    [0.1996822645906036, -0.0766714797543908],
                    [0.2117460221816151, -0.08575212708073371],
                    [0.22379973743858267, -0.0948461003814609],
                    [0.2358814437094316, -0.10390285056486075],
                    [0.24798949158308062, -0.11292435435813661],
                    [0.26011569895528003, -0.12192143583059326],
                    [0.27224868844946026, -0.13090937054836535],
                    [0.2843767087867052, -0.13990400940587125],
                    [0.2964900173183816, -0.14891844999082446],
                    [0.30858274404319375, -0.15796048052709188],
                    [0.32065528165669865, -0.1670294491397464],
                    [0.33271185241089574, -0.1761196348548849],
                    [0.34475768669048845, -0.18522404418700433],
                    [0.35679551363332995, -0.19433903851751677],
                    [0.36882028864603206, -0.20347124338953246],
                    [0.3808142593891736, -0.2126438603997199],
                    [0.39274364357792596, -0.22190028614015062],
                    [0.404557521427759, -0.23130359055863425],
                    [0.416189184475485, -0.2409311766952127],
                    [0.42756044497807894, -0.25086465614087483],
                    [0.4385914022660988, -0.2611743351728261],
                    [0.44920199952300055, -0.2719157377174528],
                    [0.4593224317591506, -0.2831198312256531],
                    [0.46890305591936937, -0.2947886851094103],
                    [0.47791985348215404, -0.3068984895662663],
                    [0.48637269301399116, -0.31940855904280646],
                    [0.49420435151505243, -0.33231629069492136],
                    [0.5011572027169569, -0.3457161039541401],
                    [0.507733275828092, -0.35930786922597757],
                    [0.5140845663496297, -0.3730063434845288],
                    [0.5202364479379097, -0.38679554587618314],
                    [0.5261349770390688, -0.4006948832694309],
                    [0.5316495249676422, -0.4147506044829779],
                    [0.5366022927914812, -0.4290133422792591],
                    [0.540712077429179, -0.4435409303028888],
                    [0.5437362491593656, -0.45833247221985673],
                    [0.5453933279413811, -0.47333856861213136],
                    [0.5455826203616074, -0.4884334343007607],
                    [0.544337223872435, -0.5034774806986179],
                    [0.5417909724248126, -0.5183573978694015],
                    [0.538140213646411, -0.5330071880600787],
                    [0.533679400402995, -0.5474324584581522],
                    [0.5288731457157675, -0.5617461488118214],
                    [0.5235471720127131, -0.5758747727527092],
                    [0.5177287082149433, -0.5898078777478809],
                    [0.511446840946865, -0.6035382873499345],
                    [0.5046957605151032, -0.6170441956982259],
                    [0.4974876937964136, -0.6303118623722612],
                    [0.4898276806191507, -0.6433238610909926],
                    [0.481718893632421, -0.6560610724652587],
                    [0.47316766752822564, -0.6685056309680046],
                    [0.46418683793715876, -0.6806438807759966]],
 'path_observed_mask': [True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False],
 'control_confidence': 0.7694556518125877,
 'road_model': {'persistent_history': True, 'length_m': 0.8606317195734003}}


class BendPairIntentTests(unittest.TestCase):
    def controller(self):
        return ProfileSpeedControl(SpeedLimits(.3, .15, .4, .2, .35, .15), 163, .4,
            controller='DWPP', bend_compensation=True, motor_rise_pwm_s=400,
            bend_curvature_gain=1.05)

    def step(self, *, target, curvature, enabled, speed=.20, allowed=True):
        control = self.controller()
        control.applied(.1, (80., 80.))
        command = control.command(target, target['control_path_m'], .3, speed,
            curvature, .005, .075, True, measured_yaw_rate=curvature*speed,
            bend_allowed=allowed, bend_inner_reduction=enabled)
        return control, command

    def test_recorded_frame66_almost_straight_near_segment_still_requires_right_correction(self):
        target = copy.deepcopy(FRAME66)
        segment, _ = supported_segment(target, skip_inferred_prefix=True, include_history=True)
        curve = describe_curve(segment)
        local = sorted(float(k) for s, k in zip(arc(curve['path']), curve['curvature']) if s <= .225)
        middle = len(local)//2
        self.assertLess(abs((local[middle]+local[~middle])/2), 1.)
        outcomes = []
        for enabled in (False, True):
            control = self.controller()
            control.applied(.1, logical((154, 54)))
            control.reference, control.updated_at = .217529, .1
            control.dwpp_state = (.1, .217529, .217529*-2.571470384412174)
            control.command(target, target['control_path_m'], .3,
                0.1912998820043364, -2.682534961662939,
                target['reference_sigma_m'], .075, True,
                measured_yaw_rate=0.03644150850694386,
                bend_allowed=True, bend_inner_reduction=enabled)
            outcomes.append(control.status)
        self.assertEqual(outcomes[0]['actuator_model_reason'], 'NO_SUPPORTED_BEND')
        self.assertNotIn('bend_allocation', outcomes[0])
        allocation = outcomes[1]['bend_allocation']
        self.assertTrue(allocation['active'])
        self.assertEqual(allocation['allocation_mode'], 'PAIRED_MODEL_INVERSE')
        self.assertLess(allocation['estimated_curvature_m_inv'], -2.6)
        self.assertTrue(allocation['limited'])
        self.assertLess(abs(allocation['estimated_curvature_m_inv']),
                        abs(allocation['target_curvature_m_inv']))
        self.assertAlmostEqual(outcomes[0]['dwpp']['selected_curvature_m_inv'],
                               outcomes[1]['dwpp']['selected_curvature_m_inv'])

    def test_near_straight_road_does_not_cancel_either_current_pursuit_correction(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                target = road(0., reference_sigma_m=.005)
                off, _ = self.step(target=target, curvature=2.*sign, enabled=False)
                on, _ = self.step(target=target, curvature=2.*sign, enabled=True)
                self.assertEqual(off.status['actuator_model_reason'], 'NO_SUPPORTED_BEND')
                self.assertNotIn('bend_allocation', off.status)
                result = on.status['bend_allocation']
                self.assertTrue(result['active'])
                self.assertGreater(result['estimated_curvature_m_inv']*sign, 1.9)

    def test_opposite_near_road_curve_does_not_reverse_current_correction(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                target = road(-2.5*sign, reference_sigma_m=.005)
                off, _ = self.step(target=target, curvature=2.*sign, enabled=False)
                on, _ = self.step(target=target, curvature=2.*sign, enabled=True)
                self.assertEqual(off.status['actuator_model_reason'], 'COUNTER_STEER')
                self.assertNotIn('bend_allocation', off.status)
                result = on.status['bend_allocation']
                self.assertTrue(result['active'])
                self.assertGreater(result['estimated_curvature_m_inv']*sign, 1.9)
                self.assertLessEqual(abs(result['estimated_curvature_m_inv']),
                                     abs(result['target_curvature_m_inv']))

    def test_old_bend_gates_stay_off_only_while_shared_mapping_preserves_dwpp(self):
        for speed, allowed, reason in ((.08, True, 'STARTUP_OR_LOW_SPEED'),
                                     (.20, False, 'BASELINE_GATE')):
            with self.subTest(speed=speed, allowed=allowed):
                target = road(0., reference_sigma_m=.005)
                off, _ = self.step(target=target, curvature=2., enabled=False,
                                   speed=speed, allowed=allowed)
                on, _ = self.step(target=target, curvature=2., enabled=True,
                                  speed=speed, allowed=allowed)
                self.assertEqual(off.status['actuator_model_reason'], reason)
                self.assertNotIn('bend_allocation', off.status)
                self.assertTrue(on.status['bend_allocation']['active'])
                for key in ('dwpp', 'target_speed_m_s', 'regulator_effort_mean_pwm'):
                    self.assertEqual(off.status[key], on.status[key])

    def test_straight_request_maps_zero_yaw_without_inheriting_near_road_turn(self):
        target = road(2., reference_sigma_m=.005)
        off, _ = self.step(target=target, curvature=0., enabled=False)
        control, _ = self.step(target=target, curvature=0., enabled=True)
        allocation = control.status['bend_allocation']
        self.assertTrue(allocation['active'])
        self.assertEqual(allocation['target_curvature_m_inv'], 0.)
        self.assertEqual(allocation['target_yaw_rad_s'], 0.)
        base_speed, base_yaw = control.bend_response.predict(allocation['base_wire'])
        self.assertLess(abs(allocation['estimated_yaw_rad_s']), abs(base_yaw))
        for key in ('dwpp', 'target_speed_m_s', 'regulator_effort_mean_pwm'):
            self.assertEqual(off.status[key], control.status[key])


if __name__ == '__main__':
    unittest.main()
