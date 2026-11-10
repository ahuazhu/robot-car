"""Braking starts at first evidence; only a verified stop earns dwell time."""

from tests.support import ROOT
import unittest
import numpy as np

from tests.support.crosswalk import configured, motion, observation
from robot.tasks.crosswalk import CrosswalkTask


class StopEvidenceTests(unittest.TestCase):
    def begin(self, task, near=0.27):
        for now in (0.0, 0.1, 0.2):
            task.update(observation(now, near), motion(), now, True, True, True, True)

    def test_first_tentative_observation_already_slows_but_does_not_confirm(self):
        task = CrosswalkTask(configured())
        state = task.update(observation(0.0, 0.5), motion(0.15), 0.0, True)
        self.assertFalse(state["confirmed"])
        self.assertEqual(state["phase"], "SEARCH")
        command = task.limit_command((40, 60), state, motion(0.15))
        self.assertLess(command[1], 60)
        self.assertAlmostEqual(command[0] / 40, command[1] / 60, delta=0.025)
        self.assertEqual(task.update(None, motion(), 0.6, True)["phase"], "SEARCH")
        self.assertIsNone(task.track)
        self.assertIsNone(task.failure)
        self.assertEqual(task.limit_command((40, 60), {}, motion()), (40, 60))

    def test_rest_evidence_survives_pairwise_flow_noise_without_refreshing_zebra(self):
        task = CrosswalkTask(
            configured(body_front_m=0.02, body_rear_m=0.16, body_half_width_m=0.085)
        )
        self.begin(task, 0.1)
        last_seen = task.track["observed_at"]
        for now in np.arange(0.3, 4.01, 0.1):
            scene = motion()
            scene["sigma_m"] = 0.006
            state = task.update(None, scene, float(now), True, True, True, True)
        self.assertGreater(state["position_error_m"], task.policy.maximum_position_error_m)
        self.assertEqual(state["phase"], "PASSING")
        self.assertIsNone(task.failure)
        self.assertEqual(task.track["observed_at"], last_seen)
        self.assertGreaterEqual(state["hold_elapsed_s"], task.policy.hold_s)

    def test_position_uncertain_before_rest_is_not_faked_into_a_legal_stop(self):
        task = CrosswalkTask(
            configured(body_front_m=0.02, body_rear_m=0.16, body_half_width_m=0.085)
        )
        for now in (0.0, 0.1, 0.2):
            task.update(observation(now, 0.11), motion(0.15), now, True)
        # Mirrors the recorded stop: clearance ~2.3cm, uncertainty ~5cm.
        task.track["near_m"] = 0.043
        task.track["sigma_m"] = 0.05
        for now in np.arange(0.3, 3.61, 0.1):
            state = task.update(None, motion(), float(now), True, True, True, True)
        self.assertIsNone(state["failure"])
        self.assertIn("CROSSWALK STOP UNVERIFIED", state["issues"])
        self.assertFalse(state["stop_verified"])
        self.assertFalse(state["yield_verified"])
        self.assertEqual(state["phase"], "RECOVERY_HOLD")
        self.assertIn("STOP_POSITION_UNCERTAIN", state["stop_blockers"])
        for now in np.arange(3.7, 5.11, 0.1):
            state = task.update(None, motion(), float(now), True, True, True, True)
        self.assertFalse(state["yield_completed"])
        self.assertFalse(state["yield_verified"])
        self.assertEqual(state["phase"], "RECOVERY_HOLD")
        self.assertIsNone(state["failure"])

    def test_stationary_drift_bound_includes_rotation_and_cannot_cross_stop_window(self):
        task = CrosswalkTask(configured())
        self.begin(task)
        interval = task._stop_interval(0.2)
        later = task._stop_interval(0.7)
        self.assertIsNotNone(interval)
        self.assertGreater(interval[0] - later[0], task.policy.stationary_speed_m_s * 0.5)
        task.stop_evidence = dict(lower_m=0.001, upper_m=0.019, normal=[1.0, 0.0], at=0.2)
        state = task.update(None, motion(), 0.4, True, True, True, True)
        # Fresh legal track support may establish a replacement proof, but no
        # previously elapsed dwell survives loss of the original proof.
        self.assertEqual(state["hold_elapsed_s"], 0.0)
        self.assertNotEqual(state["phase"], "PASSING")

    def test_reverse_motion_is_not_a_verified_stop(self):
        task = CrosswalkTask(configured())
        self.begin(task)
        for now in (0.3, 0.4, 0.5, 0.6):
            state = task.update(observation(now), motion(-0.05), now, True, True, True, True)
        self.assertFalse(state["stop_verified"])
        self.assertEqual(state["hold_elapsed_s"], 0.0)
        self.assertIn("PHYSICAL_STOP_UNVERIFIED", state["stop_blockers"])

    def test_motion_ack_or_lamp_loss_resets_verified_dwell(self):
        for missing in ("motion", "ack", "lamp", "flow"):
            with self.subTest(missing=missing):
                task = CrosswalkTask(configured())
                self.begin(task)
                for now in (0.3, 0.4, 0.5, 0.6):
                    task.update(observation(now), motion(), now, True, True, True, True)
                self.assertEqual(task.phase, "HOLD")
                state = task.update(
                    observation(0.7),
                    motion(0.02 if missing == "motion" else 0.0, valid=missing != "flow"),
                    0.7,
                    True,
                    missing != "ack",
                    missing != "lamp",
                    True,
                )
                self.assertEqual(state["phase"], "STOPPING")
                self.assertFalse(state["stop_verified"])
                self.assertEqual(state["hold_elapsed_s"], 0.0)
                for now in (0.8, 0.9, 1.0, 1.1):
                    state = task.update(observation(now), motion(), now, True, True, True, True)
                self.assertEqual(state["phase"], "HOLD")
                self.assertLess(state["hold_elapsed_s"], 0.31)

    def test_fresh_contradictory_position_cannot_reuse_old_hold(self):
        task = CrosswalkTask(configured())
        self.begin(task)
        for now in (0.3, 0.4, 0.5, 0.6):
            task.update(observation(now), motion(), now, True, True, True, True)
        state = task.update(observation(0.7, 0.332), motion(), 0.7, True, True, True, True)
        self.assertFalse(state["stop_verified"])
        self.assertFalse(state["yield_verified"])
        self.assertNotEqual(state["phase"], "PASSING")

    def test_new_crossing_and_latched_fault_cannot_show_previous_hold_seconds(self):
        task = CrosswalkTask(configured())
        self.begin(task)
        task.phase = "PASSING"
        task.hold_completed_s = 3.4
        task.update(None, motion(0.3, 0.9), 0.3, True)
        state = task.update(observation(0.4, 0.5), motion(), 0.4, True)
        self.assertEqual(state["event_id"], 2)
        self.assertEqual(state["hold_elapsed_s"], 0.0)
        self.assertIsNone(task.stop_evidence)
        task._fail("TEST")
        old = dict(task.track)
        state = task.update(observation(2.0, 0.2), motion(0.2, 0.1), 2.0, True)
        self.assertEqual(task.track, old)
        self.assertEqual(state["phase"], "FAULT")
        self.assertEqual(state["hold_elapsed_s"], 0.0)

    def test_delayed_braking_and_near_field_occlusion_complete_legal_yield(self):
        task = CrosswalkTask(
            configured(
                body_front_m=0.02,
                body_rear_m=0.16,
                body_half_width_m=0.085,
                brake_deceleration_m_s2=0.08,
                stop_clearance_m=0.08,
                approach_speed_m_s=0.06,
                speed_per_pwm_m_s=0.006705385281385282,
            )
        )
        dt = 0.05
        speed = 0.16
        near = 0.28
        last_near = near
        command = (55, 55)
        queue = [command] * 7
        hold = []
        for step in range(240):
            now = step * dt
            if step % 3 == 0:
                scene = motion(speed, last_near - near)
                scene["sigma_m"] = 0.0015
                obs = observation(now, near) if near >= 0.09 else None
                state = task.update(
                    obs, scene, now, True, command == (0, 0), command == (0, 0), True
                )
                command = task.limit_command((55, 55), state, scene) or (0, 0)
                last_near = near
                self.assertIsNone(task.failure, state)
                if task.phase == "HOLD":
                    hold.append((now, near - 0.02))
                if task.phase == "PASSING":
                    break
            queue.append(command)
            applied = queue.pop(0)
            desired = sum(applied) / 2 * (0.16 / 55.0)
            next_speed = max(0.0, speed + float(np.clip(desired - speed, -0.08 * dt, 0.4 * dt)))
            near -= (speed + next_speed) * dt / 2
            speed = next_speed
        self.assertEqual(task.phase, "PASSING")
        self.assertTrue(hold)
        self.assertTrue(all(0 < clearance < 0.20 for _, clearance in hold))
        self.assertGreaterEqual(state["hold_elapsed_s"], task.policy.hold_s)


if __name__ == "__main__":
    unittest.main()
