"""Task uncertainty cannot permanently veto a finish-priority mission."""

from tests.support import ROOT
import json
import unittest
from robot.tasks.crosswalk import CrosswalkTask
from tests.support.crosswalk import configured, motion, observation


class FinishPriorityCrosswalkTests(unittest.TestCase):
    def task(self, *, finish_priority=True):
        config=json.loads((ROOT/'maixcam2/curve_follow/runtime.json').read_text())['crosswalk']
        return CrosswalkTask(configured(**config),finish_priority=finish_priority)

    def stuck(self, *, finish_priority=True):
        task=self.task(finish_priority=finish_priority)
        fixture=json.loads((ROOT/'tests/fixtures/crosswalk_unverified_20261111.json').read_text())
        task.track=dict(fixture['track']);task.hits=fixture['hits'];task.identity=1
        task.phase='RECOVERY_HOLD';task.position_at=task.last_update=0.
        task.stopping_since=0.
        return task

    def advance(self, task, now, *, scene=None, zero=True, lamp=True, road=True, active=True):
        return task.update(None,motion() if scene is None else scene,now,
                           active,zero,lamp,road)

    def test_recorded_54cm_error_moves_on_first_frame_without_claiming_legal_stop(self):
        task=self.stuck();state=None
        for step in range(1,80):
            now=step*.15;state=self.advance(task,now)
            command=task.limit_command((100,120),state,motion())
            self.assertEqual(state['phase'],'RECOVERY_PASSING')
            self.assertGreater(min(command),0)
            self.assertLessEqual(max(command),60)
            self.assertAlmostEqual(command[0]/command[1],100/120,delta=.01)
        self.assertEqual(state['phase'],'RECOVERY_PASSING')
        self.assertFalse(state['yield_verified']);self.assertFalse(state['yield_completed'])
        self.assertTrue(task.allows_road_search());self.assertFalse(task.allows_speed_profile())
        self.assertIn('CROSSWALK UNVERIFIED CONTINUATION',state['issues'])

    def test_conservative_policy_keeps_its_unverified_position_gate(self):
        task=self.stuck(finish_priority=False)
        for step in range(1,80):
            state=self.advance(task,step*.15)
            self.assertEqual(task.limit_command((100,120),state,motion()),(0,0))
        self.assertEqual(state['phase'],'RECOVERY_HOLD')
        self.assertFalse(state['yield_completed'])

    def test_missing_speed_or_road_releases_task_ownership_on_first_frame(self):
        for kind in ('speed','road'):
            with self.subTest(kind=kind):
                task=self.task()
                for now in (0.,.15,.3):
                    task.update(observation(now,.42),motion(),now,True,False,False,True)
                for step in range(1,3):
                    now=.3+step*.15
                    scene=motion(valid=False) if kind=='speed' else motion()
                    state=self.advance(task,now,scene=scene,road=kind!='road')
                    command=task.limit_command(None if kind=='road' else (100,120),state,scene)
                    if kind=='road':self.assertIsNone(command)
                    elif kind=='speed':self.assertTrue(any(command))
                    self.assertEqual(task.phase,'RECOVERY_PASSING')
                self.assertEqual(state['phase'],'RECOVERY_PASSING')
                self.assertFalse(state['yield_completed'])
                self.assertTrue(task.allows_road_search())

    def test_uncertainty_recovery_does_not_wait_for_a_zero_command_handshake(self):
        task=self.stuck()
        state=self.advance(task,.15,zero=False)
        self.assertEqual(state['phase'],'RECOVERY_PASSING')
        self.assertEqual(task.limit_command((100,120),state,motion()),(50,60))

    def test_missing_red_lamp_does_not_become_an_infinite_rule_verification_wait(self):
        task=self.stuck()
        for step in range(1,60):
            state=self.advance(task,step*.15,lamp=False)
        self.assertEqual(state['phase'],'RECOVERY_PASSING')
        self.assertFalse(state['yield_verified']);self.assertFalse(state['yield_completed'])
        self.assertGreater(min(task.limit_command((100,120),state,motion())),0)

    def test_actual_planned_stop_keeps_only_normal_dwell_when_proof_is_missing(self):
        task=self.task()
        for now in (0.,.15,.3):
            task.update(observation(now,.16),motion(),now,True,True,False,True)
        self.assertTrue(task.planned_stop)
        for step in range(1,22):
            now=.3+step*.15
            state=self.advance(task,now,scene=motion(valid=False),lamp=False,
                               zero=step%4!=0)
            self.assertEqual(task.limit_command((100,120),state,motion()),(0,0))
        state=self.advance(task,3.6,scene=motion(valid=False),lamp=False)
        self.assertEqual(state['phase'],'RECOVERY_PASSING')
        self.assertGreater(min(task.limit_command((100,120),state,motion())),0)
        self.assertFalse(state['yield_verified']);self.assertFalse(state['yield_completed'])
        self.assertIn('CROSSWALK BRAKE LAMP UNVERIFIED',state['issues'])
        # A fresh view of the same event cannot demand a second full dwell.
        for step in range(1,10):
            now=3.6+step*.15
            state=task.update(observation(now,.19),motion(),now,True,True,True,True)
            self.assertEqual(state['phase'],'RECOVERY_PASSING')

    def test_flow_gap_that_expands_position_interval_does_not_brake_good_road(self):
        task=self.task()
        for now in (0.,.15,.3):
            task.update(observation(now,.36),motion(.06),now,True,False,False,True)
        scene=motion(valid=False)
        state=task.update(None,scene,.65,True,False,False,True)
        self.assertEqual(state['phase'],'RECOVERY_PASSING')
        self.assertEqual(task.limit_command((100,120),state,scene),(50,60))

    def test_uncertain_approach_effort_and_pending_candidate_cannot_block_search(self):
        task=self.task()
        for now in (0.,.15,.3):
            state=task.update(observation(now,.42),motion(),now,True,False,False,True)
        task.speed_pwm=0.;task.speed_update=task.last_update
        self.assertTrue(any(task.limit_command((100,120),state,motion(.4))))
        self.assertEqual(task.phase,'RECOVERY_PASSING')
        task.pending_track=observation(.3,2.)
        task.speed_pwm=task.speed_update=None
        self.assertTrue(task.allows_road_search())
        self.assertLessEqual(max(task.limit_command((100,120),state,motion())),60)
        self.assertIsNone(task.limit_command(None,state,motion(valid=False)))

    def test_idle_configuration_failure_and_explicit_fault_never_use_override(self):
        for kind in ('idle','config','fault'):
            task=self.stuck()
            if kind=='config':task.policy.geometry_verified=False
            if kind=='fault':task.failure='CROSSWALK SETUP REQUIRED'
            for step in range(1,60):
                state=self.advance(task,step*.15,active=kind!='idle')
                self.assertNotEqual(state['phase'],'RECOVERY_PASSING')

    def test_legal_dwell_uses_normal_permission_instead_of_unverified_recovery(self):
        task=self.task()
        for step in range(30):
            now=step*.15
            state=task.update(observation(now,.16),motion(),now,True,True,True,True)
            if state['phase']=='PASSING':break
        self.assertEqual(state['phase'],'PASSING')
        self.assertTrue(state['yield_verified']);self.assertTrue(state['yield_completed'])
        self.assertNotIn('CROSSWALK UNVERIFIED CONTINUATION',state['issues'])
        self.assertGreater(state['hold_elapsed_s'],3.)

    def test_current_far_reobservation_restores_normal_approach(self):
        task=self.stuck()
        for step in range(1,37):state=self.advance(task,step*.15)
        self.assertEqual(state['phase'],'RECOVERY_PASSING')
        obs=dict(task.track,observed_at=5.6,sigma_m=.01)
        state=task.update(obs,motion(),5.6,True,True,True,True)
        self.assertEqual(state['phase'],'APPROACH')
        self.assertFalse(state['yield_completed'])
        self.assertGreater(min(task.limit_command((100,120),state,motion())),0)

    def test_zero_ack_while_coasting_does_not_shorten_a_proved_physical_dwell(self):
        task=self.task()
        for now in (0.,.15,.3):
            task.update(observation(now,.16),motion(.04),now,True,False,True,True)
        self.assertEqual(task.phase,'STOPPING')
        physical_start=None
        for step in range(1,40):
            now=.3+step*.15
            state=task.update(observation(now,.16),motion(.04 if now<1. else 0.),
                              now,True,True,True,True)
            if task.physical_rest_since is not None and physical_start is None:
                physical_start=task.physical_rest_since
            if physical_start is None or now-physical_start<task.policy.hold_s:
                self.assertNotIn(task.phase,('PASSING','RECOVERY_PASSING'))
            if task.phase=='PASSING':break
        self.assertIsNotNone(physical_start)
        self.assertEqual(task.phase,'PASSING')
        self.assertTrue(state['yield_verified'])
        self.assertGreaterEqual(now-physical_start,task.policy.hold_s)

    def test_at_or_past_line_recovers_immediately_without_claiming_points(self):
        task=self.task()
        for step in range(3):
            now=step*.15
            obs=observation(now,task.policy.body_front_m-.02)
            state=task.update(obs,motion(),now,True,True,True,True)
            self.assertTrue(any(task.limit_command((100,120),state,motion())))
        self.assertEqual(state['phase'],'RECOVERY_PASSING')
        self.assertFalse(state['yield_completed']);self.assertFalse(state['yield_verified'])
        self.assertGreater(min(task.limit_command((100,120),state,motion())),0)
        self.assertIn('CROSSWALK STOP WINDOW MISSED',state['issues'])


if __name__=='__main__':unittest.main()
