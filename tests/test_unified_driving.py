"""Production ownership: no legacy wheel law and no discarded zebra obligation."""
from tests.support import ROOT
from tests.support.driving import road, flow
from tests.support.crosswalk import configured, motion, observation
from robot.config import RuntimeSettings
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.steering_control import SteeringControl
from robot.navigation.speed_profile import SpeedLimits
from robot.core.road_policy import RoadPolicy
from robot.tasks.crosswalk import CrosswalkTask
from dataclasses import replace
from unittest.mock import patch
import unittest


class UnifiedDrivingTests(unittest.TestCase):
    def setUp(self):
        self.speed = ProfileSpeedControl(SpeedLimits(.3,.15,.3,.2,.35,.15),218,.4,controller='DWPP')
        self.steer = SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,
                                     cruise=145,speed_control=self.speed)
        self.road = road(2., reference_sigma_m=.005,road_model=dict(persistent_history=True))

    def test_production_rejects_legacy_or_pure_pursuit_configuration(self):
        settings=RuntimeSettings.load(ROOT/'maixcam2/curve_follow')
        self.assertEqual(ProfileSpeedControl.for_driving(settings).controller,'DWPP')
        for config in (dict(mode='LEGACY'),dict(mode='ACTIVE',controller='PURE_PURSUIT')):
            with self.assertRaisesRegex(ValueError,'legacy entry is disabled'):
                ProfileSpeedControl.for_driving(replace(settings,runtime=dict(settings.runtime,speed_profile=config)))

    def test_continuous_dropout_uses_dwpp_crawl_without_speed_integrator_or_legacy(self):
        self.steer.applied(.9,(50,50))
        with patch('robot.navigation.steering_control.target_command',side_effect=AssertionError('legacy called')):
            for i in range(30):
                now=1.+i*.18
                self.steer.observe(flow(valid=i in (0,29)),now,False)
                wheels=self.steer.command(self.road,.05)
                self.assertIsNotNone(wheels)
                self.assertTrue(self.speed.status['selected'])
                self.assertEqual(self.steer.status['speed_plan']['mode'],'DWPP')
                if i not in (0,29):
                    self.assertFalse(self.speed.status['motion_feedback'])
                    self.assertIsNone(self.speed.status['measured_speed_m_s'])
                    self.assertLessEqual(self.speed.status['target_speed_m_s'],.075+1e-9)
                    self.assertLessEqual(max(wheels),109.+1e-8)
                self.steer.applied(now+.05,wheels)
            self.assertTrue(self.speed.status['motion_feedback'])

    def test_unresolved_forward_displacement_and_task_ceiling_do_not_change_controller(self):
        self.steer.observe(dict(flow(),sigma_m=.02),1.,False)
        with patch('robot.navigation.steering_control.target_command',side_effect=AssertionError('legacy called')):
            command=self.steer.command(self.road,.05,task_limit_m_s=.025)
        self.assertIsNotNone(command)
        self.assertTrue(self.speed.status['selected'])
        self.assertFalse(self.speed.status['motion_feedback'])
        self.assertLessEqual(self.speed.status['target_speed_m_s'],.025)

    def test_invalid_support_requests_search_never_old_wheel_law(self):
        self.steer.observe(flow(),1.,False)
        bad=dict(self.road,path_observed_mask=[])
        with patch('robot.navigation.steering_control.target_command',side_effect=AssertionError('legacy called')):
            self.assertIsNone(self.steer.command(bad,.05))
        self.assertEqual(self.steer.status['speed_plan']['mode'],'DWPP_SEARCH_REQUIRED')

    def test_sharp_turn_from_rest_cannot_latch_on_zero_intersection(self):
        for k in (-12.,12.):
            self.speed.reset()
            wheels=self.speed.command(self.road,self.road['control_path_m'],1.,0.,k,.005,.075,
                                      True,measured_yaw_rate=0.)
            self.assertGreater(sum(wheels),0)
            self.assertIn('FORWARD_CURVATURE_PROJECTION',self.speed.status['dwpp']['overrides'])
            self.assertGreater(self.speed.status['reference_angular_rad_s']*k,0)

    def test_task_zero_cannot_be_lifted_by_persistent_route_crawl_floor(self):
        self.steer.observe(flow(valid=False),1.,False)
        command=self.steer.command(self.road,.05,task_limit_m_s=0.)
        self.assertEqual(command,(0.,0.))
        self.assertEqual(self.speed.status['target_speed_m_s'],0.)

    def test_hard_zero_clears_integral_effort_even_when_sensor_reports_rest(self):
        self.speed.applied(.9,(80,80))
        wheels=self.speed.command(self.road,self.road['control_path_m'],1.,0.,2.,.005,.075,
                                  True,measured_yaw_rate=0.,task_limit_m_s=0.)
        self.assertEqual(wheels,(0.,0.))
        self.assertTrue(self.speed.status['selected'])
        self.assertEqual(self.speed.status['target_speed_m_s'],0.)


class PendingYieldTests(unittest.TestCase):
    def task(self):
        return CrosswalkTask(configured(body_front_m=.02,body_rear_m=.16,body_half_width_m=.085,
             stop_clearance_m=.02),finish_priority=True,unified_control=True)

    def begin(self):
        t=self.task()
        for now in (0.,.15,.3):
            t.update(observation(now,.47),motion(.08),now,True,False,False,True)
        self.assertEqual(t.phase,'APPROACH')
        return t

    def test_flow_gap_retains_stop_until_nominal_brake_entry_then_one_bounded_dwell(self):
        t=self.begin()
        state=t.update(None,motion(valid=False),.45,True,False,False,True)
        self.assertEqual(t.phase,'RECOVERY_APPROACH')
        self.assertTrue(state['yield_pending'])
        self.assertTrue(t.allows_speed_profile())
        self.assertTrue(t.allows_road_search())
        self.assertGreater(t.motion_speed_limit(),0)
        # Current flow resumes but the bars remain occluded, as in the run.
        now=.45
        while t.track['near_m']-.02 > .15:
            now+=.15
            state=t.update(None,motion(.1,.015),now,True,False,False,True)
            self.assertNotEqual(t.phase,'RECOVERY_PASSING')
        self.assertEqual(t.phase,'STOPPING')
        self.assertTrue(t.planned_stop)
        self.assertEqual(t.limit_command((100,120),state,motion()),(0,0))
        first=now+.15
        for i in range(30):
            now=first+i*.15
            state=t.update(None,motion(valid=False),now,True,True,True,True)
            if now-first < t.policy.hold_s:
                self.assertEqual(t.limit_command((100,120),state,motion()),(0,0))
            if t.phase=='RECOVERY_PASSING':break
        self.assertEqual(t.phase,'RECOVERY_PASSING')
        self.assertGreaterEqual(state['hold_elapsed_s'],3.2)
        self.assertFalse(state['yield_pending'])
        self.assertFalse(state['yield_verified'])
        self.assertTrue(any(t.limit_command((100,120),state,motion())))

    def test_unified_task_does_not_run_old_pwm_integrator(self):
        t=self.begin();t.speed_pwm=0.;t.speed_update=t.last_update
        command=(60.,80.)
        self.assertEqual(t.limit_command(command,t.snapshot(.3),motion(.3)),command)
        self.assertEqual(t.speed_pwm,0.)
        self.assertGreater(t.motion_speed_limit(),0)

    def test_fresh_geometry_recovers_approach_without_dropping_yield(self):
        t=self.begin()
        t.update(None,motion(valid=False),.45,True,False,False,True)
        state=t.update(observation(.6,.46),motion(),.6,True,False,False,True)
        self.assertEqual(t.phase,'APPROACH')
        self.assertTrue(state['yield_pending'])
        self.assertTrue(t.allows_speed_profile())

    def test_approach_allows_search_when_planner_cannot_use_otherwise_valid_target(self):
        t=self.begin()
        self.assertTrue(t.allows_road_search())
        t.phase='STOPPING'
        self.assertFalse(t.allows_road_search())
        t.phase='HOLD'
        self.assertFalse(t.allows_road_search())

    def test_unconfirmed_single_candidate_never_owns_dwell(self):
        t=self.task()
        t.update(observation(0.,.12),motion(valid=False),0.,True,False,False,True)
        for now in (.15,.3,.6):
            state=t.update(None,motion(valid=False),now,True,False,False,True)
            self.assertNotIn(t.phase,('STOPPING','HOLD','RECOVERY_HOLD'))
        self.assertIsNone(t.track)

if __name__=='__main__':unittest.main()
