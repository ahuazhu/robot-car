"""Opt-in DWPP ownership only after the existing recovery dwell is complete."""
import copy
import unittest

from tests.support.crosswalk import configured, motion, observation
from robot.tasks.crosswalk import CrosswalkTask


class CrosswalkRecoveryDwppTests(unittest.TestCase):
    def completed_recovery(self):
        task = CrosswalkTask(configured(), finish_priority=True)
        for now in (0., .15, .3):
            task.update(observation(now, .16), motion(), now, True, True, False, True)
        self.assertTrue(task.planned_stop)
        for step in range(1, 30):
            now = .3 + step * .15
            task.update(None, motion(valid=False), now, True, True, False, True)
            if task.phase == 'RECOVERY_PASSING':
                break
        self.assertEqual(task.phase, 'RECOVERY_PASSING')
        self.assertGreaterEqual(task.hold_completed_s, task.policy.hold_s)
        return task

    def limit(self, task, command=(100., 120.), *, enabled=True, selected=True):
        return task.limit_command(command, task.snapshot(task.last_update), motion(.1),
                                  recovery_dwpp=enabled, speed_profile_selected=selected)

    def test_default_and_explicitly_disabled_preserve_recovery_pwm(self):
        task = self.completed_recovery()
        baseline = task.limit_command((100., 120.), task.snapshot(task.last_update), motion(.1))
        self.assertEqual(baseline, (50., 60.))
        self.assertEqual(self.limit(task, enabled=False), baseline)
        self.assertFalse(task.allows_speed_profile())
        self.assertFalse(task.allows_speed_profile(recovery_dwpp=False))

    def test_completed_recovery_can_use_only_a_selected_dwpp_command(self):
        task = self.completed_recovery()
        self.assertTrue(task.allows_speed_profile(recovery_dwpp=True))
        self.assertEqual(self.limit(task, selected=False), (50., 60.))
        self.assertEqual(self.limit(task), (100., 120.))
        # The task adds no floor, and preserves upstream reductions or stops.
        for command in ((20., 30.), (0., 0.), None):
            self.assertEqual(self.limit(task, command), command)

    def test_experiment_does_not_verify_or_retire_an_uncertain_served_event(self):
        task = self.completed_recovery()
        task.track['sigma_m'] = 2.
        before = copy.deepcopy(task.snapshot(task.last_update))
        self.assertTrue(task.allows_speed_profile(recovery_dwpp=True))
        self.assertEqual(self.limit(task), (100., 120.))
        self.assertEqual(task.snapshot(task.last_update), before)
        self.assertFalse(task.yield_verified)
        self.assertFalse(task.yield_completed)
        self.assertIsNotNone(task.track)

    def test_incomplete_dwell_and_recovery_approach_keep_original_ownership(self):
        for phase, completed in (('RECOVERY_APPROACH', 4.),
                                 ('RECOVERY_PASSING', 0.),
                                 ('RECOVERY_PASSING', 3.19)):
            with self.subTest(phase=phase, completed=completed):
                task = self.completed_recovery()
                task.phase, task.hold_completed_s = phase, completed
                self.assertFalse(task.allows_speed_profile(recovery_dwpp=True))
                self.assertEqual(self.limit(task), (50., 60.))

    def test_dwell_stop_setup_and_failure_never_gain_motion_permission(self):
        for phase in ('STOPPING', 'HOLD', 'RECOVERY_HOLD'):
            with self.subTest(phase=phase):
                task = self.completed_recovery()
                task.phase = phase
                self.assertFalse(task.allows_speed_profile(recovery_dwpp=True))
                self.assertEqual(self.limit(task), (0, 0))
        for blocker in ('setup', 'failure'):
            with self.subTest(blocker=blocker):
                task = self.completed_recovery()
                if blocker == 'setup':
                    task.policy.geometry_verified = False
                else:
                    task.failure = 'CROSSWALK SETUP REQUIRED'
                self.assertFalse(task.allows_speed_profile(recovery_dwpp=True))
                self.assertEqual(self.limit(task), (0, 0))

    def test_pending_crossing_restores_same_approach_braking_as_baseline(self):
        task = self.completed_recovery()
        task.pending_track = observation(task.last_update, .42)
        baseline = copy.deepcopy(task)
        self.assertFalse(task.allows_speed_profile(recovery_dwpp=True))
        expected = self.limit(baseline, enabled=False)
        self.assertEqual(self.limit(task), expected)
        self.assertLess(max(expected), 120.)
        self.assertEqual(task.speed_ceiling, baseline.speed_ceiling)

    def test_enabling_during_hold_does_not_advance_release_clock(self):
        task = CrosswalkTask(configured(), finish_priority=True)
        for now in (0., .15, .3):
            task.update(observation(now, .16), motion(), now, True, True, False, True)
        for step in range(1, 20):
            task.update(None, motion(valid=False), .3 + step * .15,
                        True, True, False, True)
            self.assertIn(task.phase, ('STOPPING', 'HOLD', 'RECOVERY_HOLD'))
            self.assertFalse(task.allows_speed_profile(recovery_dwpp=True))
            self.assertEqual(self.limit(task), (0, 0))
            self.assertLess(task.hold_completed_s, task.policy.hold_s)


if __name__ == '__main__':
    unittest.main()
