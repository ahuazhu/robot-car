"""Measured outside-only allocation, command delivery and feedback regressions."""
from tests.support import ROOT
from tests.support.driving import road
import json
from pathlib import Path
import unittest

from robot.control.controller import compensate_pwm, CommandRamp
from robot.navigation.bend_response import BendResponse
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits


def logical(wire, boost=37):
    peak=max(wire)
    return tuple(p*(peak-boost)/peak for p in wire)


class BendResponseTests(unittest.TestCase):
    def control(self, enabled=True, gain=1.):
        return ProfileSpeedControl(SpeedLimits(.3,.15,.4,.2,.35,.15),163,.4,
            controller='DWPP',bend_compensation=enabled,motor_rise_pwm_s=400,
            bend_curvature_gain=gain)

    def step(self,c,*,now=.3,speed=.20,k=-2.,allowed=True,road_k=None):
        r=road(k if road_k is None else road_k,reference_sigma_m=.005)
        return c.command(r,r['control_path_m'],now,speed,k,.005,.075,True,
                         measured_yaw_rate=k*speed,bend_allowed=allowed)

    def test_model_nodes_reproduce_accepted_body_motion_and_real_straight_yaw(self):
        model=BendResponse(.15,37)
        data=json.loads((Path(ROOT)/'maixcam2/curve_follow/robot/navigation/bend_response_data.json').read_text())
        for name in ('right','left'):
            for node in data[name]['nodes']:
                outside,inside,v,w=node['value']
                pair=(outside,inside) if name=='right' else (inside,outside)
                predicted=model.predict(pair)
                self.assertIsNotNone(predicted)
                self.assertAlmostEqual(predicted[0],v,places=10)
                self.assertAlmostEqual(predicted[1],w,places=10)
        self.assertGreater(model.predict((90,90))[1],0)

    def test_both_turns_keep_inside_wire_and_improve_curvature(self):
        model=BendResponse(.15,37)
        for pair,k in (((140,70),-2.2),((70,140),2.2),((160,80),-2),((80,160),2)):
            result=model.allocate(logical(pair),k,allowed=True,ceiling=.30)
            self.assertIsNotNone(result)
            wire=compensate_pwm(result['wheels'])
            outside=0 if k<0 else 1
            self.assertEqual(list(wire),result['wire'])
            self.assertEqual(wire[1-outside],pair[1-outside])
            self.assertGreater(wire[outside],pair[outside])
            self.assertLess(abs(result['estimated_curvature_m_inv']-k),abs(model.curvature(pair)-k))
            self.assertTrue(result['inside_preserved'])

    def test_no_arbitrary_bonus_cap_and_unreachable_is_explicit(self):
        model=BendResponse(.15,37)
        result=model.allocate(logical((140,70)),-3.,allowed=True,ceiling=.30)
        self.assertGreater(result['outside_added_pwm'],40)
        self.assertTrue(result['limited'])
        self.assertEqual(result['wire'][1],70)
        self.assertEqual(result['reason'],'BEST_FEASIBLE_OUTSIDE_ONLY')
        self.assertLessEqual(result['wire'][0],240)

    def test_skipped255_right_not_used_and_weaker255_left_does_not_win(self):
        model=BendResponse(.15,37)
        self.assertIsNone(model.predict((255,70)))
        self.assertIsNotNone(model.predict((70,255)))
        self.assertLess(model.curvature((70,255)),model.curvature((70,240)))
        result=model.allocate(logical((70,200)),3.,allowed=True,ceiling=.30)
        self.assertEqual(result['wire'],[70,240])
        near=model.allocate(logical((70,200)),2.59,allowed=True,ceiling=.30)
        self.assertLess(near['wire'][1],240)

    def test_unknown_low_inside_and_no_feasible_speed_keep_original(self):
        model=BendResponse(.15,37)
        for pair in ((140,50),(50,140),(255,70)):
            self.assertIsNone(model.allocate(logical(pair),-3 if pair[0]>pair[1] else 3,allowed=True))
        self.assertIsNone(model.allocate(logical((140,70)),-3,allowed=True,ceiling=.10))
        self.assertEqual(model.last_reason,'NO_FEASIBLE_OUTSIDE_IMPROVEMENT')

    def test_straight_wrong_sign_and_permission_never_add_output(self):
        model=BendResponse(.15,37)
        for pair,k,allowed in (((90,90),0,True),((140,70),2,True),((140,70),-2,False)):
            self.assertIsNone(model.allocate(logical(pair),k,allowed=allowed))

    def test_shared_ramp_budget_preserves_inside_after_actual_boost(self):
        for k in (-2,2):
            c=self.control();c.applied(.1,(80.,80.));command=self.step(c,k=k)
            result=c.status['bend_allocation'];self.assertTrue(result['active'])
            ramp=CommandRamp(400);ramp.previous=(80.,80.);ramp.last=.1
            delivered=compensate_pwm(ramp.apply(command,.3))
            self.assertEqual(list(delivered),result['wire'])
            outside=0 if k<0 else 1
            self.assertEqual(delivered[1-outside],result['base_wire'][1-outside])
            base=c.bend_proposal['base_mean']
            c.applied(.31,command,delivered_wheels=delivered)
            self.assertAlmostEqual(c.applied_mean,base)

    def test_small_ramp_budget_rejects_extra_without_reducing_inside(self):
        model=BendResponse(.15,37);base=logical((140,70))
        self.assertIsNone(model.allocate(base,-2.2,allowed=True,ceiling=.3,
            effort_limits=dict(peak=max(base),sum=sum(base))))

    def test_reference_unchanged_and_extra_pwm_not_integrated_as_base_effort(self):
        a,b=self.control(False),self.control(True)
        for c in (a,b):c.applied(.1,(80.,80.))
        self.step(a);w=self.step(b)
        self.assertEqual(a.status['target_speed_m_s'],b.status['target_speed_m_s'])
        self.assertEqual(a.status['regulator_effort_mean_pwm'],b.status['regulator_effort_mean_pwm'])
        base=b.bend_proposal['base_mean']
        b.applied(.31,w,delivered_wheels=compensate_pwm(w))
        self.assertAlmostEqual(b.applied_mean,base)
        self.assertAlmostEqual(b.status['applied_curvature_m_inv'],b.status['bend_allocation']['estimated_curvature_m_inv'])

    def test_measured_outside_response_can_exceed_old_wheel_proxy_without_retargeting_dwpp(self):
        baseline, measured = self.control(False, gain=1.05), self.control(True, gain=1.05)
        for control in (baseline, measured):
            control.applied(.1, (80., 80.))
        original = self.step(baseline, k=-2.46)
        improved = self.step(measured, k=-2.46)
        allocation = measured.status['bend_allocation']
        self.assertEqual(measured.status['dwpp'], baseline.status['dwpp'])
        self.assertEqual(measured.status['target_speed_m_s'], baseline.status['target_speed_m_s'])
        self.assertEqual(measured.status['regulator_effort_mean_pwm'],
                         baseline.status['regulator_effort_mean_pwm'])
        self.assertEqual(allocation['base_wire'], list(compensate_pwm(original)))
        self.assertEqual(compensate_pwm(improved)[1], compensate_pwm(original)[1])
        self.assertGreater(compensate_pwm(improved)[0], compensate_pwm(original)[0])
        self.assertGreater(allocation['estimated_speed_m_s'], measured.status['ceiling_m_s'])
        self.assertLessEqual(allocation['estimated_speed_m_s'],
                             measured.status['bend_speed_ceiling_m_s'])
        self.assertLessEqual(measured.status['bend_speed_ceiling_m_s'], .30)
        self.assertLess(abs(allocation['estimated_curvature_m_inv'] - allocation['target_curvature_m_inv']),
                        abs(allocation['base_curvature_m_inv'] - allocation['target_curvature_m_inv']))

    def test_measured_outside_response_retains_candidate_lateral_acceleration_bound(self):
        model = BendResponse(.15, 37)
        unrestricted = model.allocate(logical((140, 70)), -3., allowed=True, ceiling=.30)
        bounded = model.allocate(logical((140, 70)), -3., allowed=True, ceiling=.30,
                                 lateral_acceleration=.08)
        self.assertIsNotNone(bounded)
        self.assertGreater(unrestricted['estimated_speed_m_s'] * abs(unrestricted['estimated_yaw_rad_s']), .08)
        self.assertLessEqual(bounded['estimated_speed_m_s'] * abs(bounded['estimated_yaw_rad_s']), .08)
        self.assertEqual(bounded['wire'][1], 70)
        self.assertGreater(bounded['wire'][0], 140)
        self.assertLess(bounded['wire'][0], unrestricted['wire'][0])

    def test_candidate_clearance_can_reject_added_pwm_even_above_old_proxy(self):
        control = self.control(gain=1.05)
        control.applied(.1, (80., 80.))
        command = self.step(control, k=-2.46)
        allocation = control.status['bend_allocation']
        self.assertGreater(allocation['estimated_speed_m_s'], control.status['ceiling_m_s'])
        base = control.bend_proposal['base_wheels']
        reference = control.status['target_speed_m_s']
        # This is the proposed arc's actual clearance cap, evaluated after
        # allocation; it is not the original arc's pre-allocation proxy.
        delivered = control.constrain_clearance(command, .10)
        self.assertEqual(delivered, tuple(wheel * .10 / reference for wheel in base))
        self.assertFalse(control.status['bend_allocation']['active'])
        self.assertIsNone(control.bend_proposal)
        self.assertAlmostEqual(control.status['target_speed_m_s'], .10)

    def test_startup_low_speed_untrusted_and_counter_steer_match_baseline(self):
        for k,speed,allowed,road_k in ((0,.2,True,None),(-2,.08,True,None),(-2,.2,False,None),(-2,.2,True,2)):
            a,b=self.control(False),self.control(True)
            for c in (a,b):c.applied(.1,(80.,80.))
            x=self.step(a,k=k,speed=speed,allowed=allowed,road_k=road_k)
            y=self.step(b,k=k,speed=speed,allowed=allowed,road_k=road_k)
            self.assertEqual(x,y)
            self.assertNotIn('bend_allocation',b.status)

    def test_braking_removes_bonus_uses_base_pair_and_clears_feedback_proposal(self):
        c=self.control();c.applied(.1,(80.,80.));w=self.step(c)
        original=c.bend_proposal['base_wheels'];v=c.status['target_speed_m_s']
        braked=c.constrain_clearance(w,v*.5)
        self.assertEqual(braked,tuple(x*.5 for x in original))
        self.assertFalse(c.status['bend_allocation']['active'])
        self.assertIsNone(c.bend_proposal)
        self.assertEqual(c.status['effective_curvature_m_inv'],c.status['dwpp']['selected_curvature_m_inv'])
        c.applied(.31,braked,delivered_wheels=compensate_pwm(braked))
        self.assertAlmostEqual(c.applied_mean,sum(braked)/2)

    def test_task_scale_and_stop_reset_feedback_without_floor(self):
        c=self.control();c.applied(.1,(80.,80.));w=self.step(c)
        base=c.bend_proposal['base_mean'];scaled=tuple(x*.8 for x in w)
        c.applied(.31,scaled,delivered_wheels=compensate_pwm(scaled))
        self.assertAlmostEqual(c.applied_mean,base*.8)
        c.applied(.4,(0.,0.),delivered_wheels=(0,0))
        self.assertEqual(c.applied_mean,0)
        self.assertIsNone(c.reference)
        self.assertIsNone(c.bend_proposal)

    def test_bend_gain_changes_only_outside_target_and_keeps_speed_and_base(self):
        changed=False
        for k in (-2.,2.):
            old,new=self.control(),self.control(gain=1.05)
            for c in (old,new):c.applied(.1,(80.,80.));self.step(c,k=k)
            a,b=old.status['bend_allocation'],new.status['bend_allocation']
            self.assertEqual(old.status['target_speed_m_s'],new.status['target_speed_m_s'])
            self.assertEqual(a['base_wire'],b['base_wire'])
            self.assertAlmostEqual(b['target_curvature_m_inv'],a['target_curvature_m_inv']*1.05)
            outside=0 if k<0 else 1
            self.assertEqual(a['wire'][1-outside],b['wire'][1-outside])
            self.assertGreaterEqual(b['wire'][outside],a['wire'][outside])
            changed |= b['wire'][outside]>a['wire'][outside]
        self.assertTrue(changed)

    def test_bend_gain_leaves_straight_low_speed_and_countersteer_unchanged(self):
        for k,speed,road_k in ((0.,.2,None),(-2.,.08,None),(-2.,.2,2.)):
            old,new=self.control(),self.control(gain=1.05)
            for c in (old,new):c.applied(.1,(80.,80.))
            self.assertEqual(self.step(old,k=k,speed=speed,road_k=road_k),
                             self.step(new,k=k,speed=speed,road_k=road_k))


if __name__=='__main__':unittest.main()
