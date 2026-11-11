"""A crossing visit and its dwell require continuity across task interruptions."""

from tests.support import ROOT
import json
import unittest

from robot.tasks.crosswalk import CrosswalkPolicy, CrosswalkTask
from tests.support.crosswalk import motion, observation


class CrosswalkLifecycleTests(unittest.TestCase):
    def task(self):
        config = json.loads((ROOT / "maixcam2/curve_follow/runtime.json").read_text())
        return CrosswalkTask(CrosswalkPolicy.from_config(config["crosswalk"]), finish_priority=True)

    def update(self, task, now, *, near=.16, scene=None, active=True, zero=True, lamp=True):
        return task.update(
            None if near is None else observation(now, near),
            motion() if scene is None else scene,
            now, active, zero, lamp, True, road_available=True,
        )

    def served(self):
        task = self.task()
        for step in range(40):
            state = self.update(task, step * .1)
        self.assertEqual(state["phase"], "PASSING")
        self.assertTrue(state["yield_verified"])
        return task

    def unlocalized(self):
        task = self.served()
        for step in range(40, 100):
            self.update(task, step * .1, near=None, scene=motion(valid=False), zero=False, lamp=False)
        self.assertGreater(task.track["sigma_m"], task.policy.maximum_position_error_m)
        return task

    def test_matching_stale_mean_cannot_lend_previous_laps_yield_permission(self):
        task = self.unlocalized()
        identity = task.identity
        # New crossing is only visible near the stop window. Its mean happens
        # to match the old 16cm mean; old uncertainty prevents identity proof.
        for step in range(3):
            state = self.update(task, 10. + step * .1, near=.20, zero=False, lamp=False)
            if step < 2:
                self.assertEqual(task.identity, identity)
                self.assertEqual(state["pending_hits"], step + 1)
        self.assertEqual(task.identity, identity + 1)
        self.assertEqual(state["phase"], "APPROACH")
        self.assertFalse(state["yield_completed"])
        self.assertFalse(state["yield_verified"])
        self.assertEqual(state["observation_association"], "REOBSERVED_CROSSWALK")
        self.assertFalse(state["retired_event"]["passage_proven"])

    def test_one_coincident_flash_cannot_rearm_or_refresh_old_identity(self):
        task = self.unlocalized()
        old_seen = task.track["observed_at"]
        state = self.update(task, 10., near=.20, zero=False, lamp=False)
        self.assertEqual(state["pending_hits"], 1)
        self.assertEqual(task.track["observed_at"], old_seen)
        state = self.update(task, 10.6, near=None, zero=False, lamp=False)
        self.assertEqual(task.identity, 1)
        self.assertIsNone(state["pending_observation"])

    def test_unlocalized_old_event_cannot_provide_a_partial_bar_template(self):
        task = self.unlocalized()
        self.assertIsNone(task.detection_reference(motion(), 10.))
        partial = dict(observation(10., .20), tracking_only=True, event_id=task.identity, bar_count=3)
        for step in range(4):
            now = 10. + step * .1
            state = task.update(dict(partial, observed_at=now), motion(), now, True,
                                False, False, True, road_available=True)
        self.assertEqual(task.identity, 1)
        self.assertEqual(state["pending_hits"], 0)
        self.assertLess(task.track["observed_at"], 4.)

    def test_precise_measured_pose_preserves_identity_without_timer_expiry(self):
        task = self.served()
        for step in range(40, 70):
            self.update(task, step * .1, near=None)
        self.assertGreater(6.9 - task.track["observed_at"], task.policy.observation_max_age_s)
        self.assertLess(task.track["sigma_m"], task.policy.maximum_position_error_m)
        state = self.update(task, 7., near=.16)
        self.assertEqual(task.identity, 1)
        self.assertEqual(state["phase"], "PASSING")
        self.assertTrue(state["yield_completed"])
        self.assertIsNone(state["pending_observation"])

    def test_continuously_observed_same_crossing_never_repeats_dwell(self):
        task = self.served()
        for step in range(40, 400):
            state = self.update(task, step * .1, near=.16 + .002 * (step % 2))
            self.assertEqual(task.identity, 1)
            self.assertEqual(state["phase"], "PASSING")
            self.assertTrue(state["yield_completed"])

    def interrupted(self, *, lamp=True):
        task = self.task()
        for step in range(7):
            state = self.update(task, step * .1, lamp=lamp)
        self.assertEqual(state["phase"], "HOLD" if lamp else "STOPPING")
        self.assertIsNotNone(task.stop_ack_since)
        # Automatic motor-fault STOP keeps this task object. Inactive camera
        # frames still see the car moving parallel to the stripe until 5.8s.
        for step in range(7, 61):
            now = step * .1
            scene = motion(.1 if now < 5.9 else 0.)
            if now < 5.9:
                scene["translation"] = [0., .01]
            state = self.update(task, now, scene=scene, active=False, zero=False, lamp=lamp)
            self.assertIsNone(task.physical_rest_since)
            self.assertIsNone(task.stop_ack_since)
            self.assertFalse(task.legal_stop_acquired)
        self.assertEqual(task.identity, 1)
        return task

    def test_fault_rearm_requires_new_continuous_physical_dwell(self):
        task = self.interrupted()
        state = self.update(task, 6.1)
        self.assertFalse(state["yield_completed"])
        self.assertFalse(state["yield_verified"])
        self.assertEqual(task.physical_rest_since, 6.1)
        self.assertEqual(state["hold_elapsed_s"], 0.)
        for step in range(1, 35):
            now = 6.1 + step * .1
            state = self.update(task, now)
            if now - 6.1 < task.policy.hold_s - 1e-9:
                self.assertNotEqual(state["phase"], "PASSING")
        self.assertEqual(state["phase"], "PASSING")
        self.assertTrue(state["yield_verified"])
        self.assertGreaterEqual(state["hold_elapsed_s"], task.policy.hold_s)

    def test_fault_rearm_cannot_reuse_unverified_zero_ack_fallback_clock(self):
        task = self.interrupted(lamp=False)
        for step in range(35):
            now = 6.1 + step * .1
            state = self.update(task, now, lamp=False)
            if now - 6.1 < task.policy.hold_s - 1e-9:
                self.assertEqual(state["phase"], "STOPPING")
        self.assertEqual(state["phase"], "RECOVERY_PASSING")
        self.assertFalse(state["yield_verified"])

    def test_interruption_preserves_already_completed_passage(self):
        task = self.served()
        completed = task.hold_completed_s
        for step in range(40, 60):
            state = self.update(task, step * .1, active=False, zero=False)
        state = self.update(task, 6.)
        self.assertEqual(task.identity, 1)
        self.assertEqual(state["phase"], "PASSING")
        self.assertTrue(state["yield_completed"])
        self.assertEqual(task.hold_completed_s, completed)

    def test_rearm_after_repositioning_does_not_keep_old_legal_stop_proof(self):
        task = self.interrupted()
        state = self.update(task, 6.1, near=.22)
        self.assertEqual(state["phase"], "APPROACH")
        self.assertFalse(state["legal_stop_acquired"])
        self.assertFalse(state["yield_completed"])
        self.assertTrue(any(task.limit_command((100, 120), state, motion())))

    def test_unverified_recovery_release_still_avoids_a_second_dwell(self):
        # This is deliberately different from a completed compliant PASSING
        # event: the bounded fallback records no permission/score to inherit.
        task = self.task()
        for now in (0., .15, .3):
            self.update(task, now, lamp=False)
        for step in range(1, 23):
            self.update(task, .3 + step * .15, near=None, scene=motion(valid=False), lamp=False)
        self.assertEqual(task.phase, "RECOVERY_PASSING")
        self.assertFalse(task.yield_completed)
        self.assertGreaterEqual(task.hold_completed_s, task.policy.hold_s)
        for step in range(1, 7):
            state = self.update(task, 3.6 + step * .15, near=.19)
            self.assertEqual(task.identity, 1)
            self.assertEqual(state["phase"], "RECOVERY_PASSING")
            self.assertFalse(state["yield_verified"])

    def test_approach_and_recovery_scaling_preserve_fractional_wheel_ratio(self):
        command = (1.2, 120.)
        task = self.task()
        for now in (0., .15, .3):
            state = self.update(task, now, near=.5, scene=motion(.1), zero=False, lamp=False)
        approach = task.limit_command(command, state, motion(.1))
        self.assertGreater(approach[0], 0.)
        self.assertLess(approach[0], 1.)
        self.assertAlmostEqual(approach[0] / approach[1], command[0] / command[1], places=12)
        task.phase = "RECOVERY_PASSING"
        recovery = task.limit_command(command, task.snapshot(.3), motion(.1))
        self.assertAlmostEqual(recovery[0], .6)
        self.assertEqual(recovery[1], 60.)
        self.assertAlmostEqual(recovery[0] / recovery[1], command[0] / command[1], places=12)


if __name__ == "__main__":
    unittest.main()
