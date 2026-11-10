"""Separate zero-output braking from the unchanged smooth approach envelope."""

from tests.support import ROOT
import json
import unittest

from robot.tasks.crosswalk import CrosswalkPolicy, CrosswalkTask
from tests.support.crosswalk import configured, motion, observation


class BrakingCalibrationTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / 'maixcam2/curve_follow/runtime.json').read_text())['crosswalk']

    def update(self, task, now, gap, speed, previous=None, stopped=False):
        near = gap + task.policy.body_front_m
        obs = dict(observation(now, near), sigma_m=.0225)
        scene = motion(speed, 0 if previous is None else previous - gap)
        state = task.update(obs, scene, now, True, stopped, stopped, True)
        return state, task.limit_command((60, 60), state, scene)

    def test_first_19cm_braking_trigger_is_removed_without_changing_approach_speed(self):
        config = self.config()
        old = CrosswalkTask(configured(**dict(config, brake_deceleration_m_s2=.08)))
        new = CrosswalkTask(configured(**config))
        # Captured whole-body distances and closing rates before the first
        # 19cm stop. This prefix is observed; a later trajectory is not replayed
        # as ground truth after the control decisions diverge.
        prefix = ((.37530215, .02966875), (.35363192, .06986625),
                  (.27343194, .13339190), (.22785023, .11782031),
                  (.19899019, .13780438))
        previous = None
        for i, (gap, speed) in enumerate(prefix):
            a, ca = self.update(old, i * .2, gap, speed, previous)
            b, cb = self.update(new, i * .2, gap, speed, previous)
            if i < len(prefix) - 1:
                self.assertEqual(ca, cb)
                self.assertEqual(a['speed_ceiling_m_s'], b['speed_ceiling_m_s'])
            previous = gap
        self.assertEqual(a['phase'], 'STOPPING')
        self.assertEqual(b['phase'], 'APPROACH')
        self.assertNotEqual(cb, (0, 0))
        self.assertAlmostEqual(b['predicted_stopping_distance_m'], .09570665, places=6)

        # Independent forward scenario: brake inside the target window, then
        # serve a legally located, continuous dwell with zero output throughout.
        b, cb = self.update(new, 1., .15, .13780438, previous)
        self.assertEqual(b['phase'], 'STOPPING')
        self.assertEqual(cb, (0, 0))
        for i in range(40):
            b, cb = self.update(new, 1.2 + i * .1, .13, 0., .15 if i == 0 else .13, True)
            if b['yield_completed']:
                break
            self.assertEqual(cb, (0, 0))
            self.assertNotEqual(b['phase'], 'APPROACH')
        self.assertTrue(b['yield_verified'])
        self.assertGreaterEqual(b['hold_elapsed_s'], 3.2)
        self.assertNotEqual(cb, (0, 0))

    def test_candidate_still_bounds_observed_stop_with_geometric_uncertainty(self):
        p = configured(**self.config())
        # First logged zero output at22.785cm +/-2.284cm; at rest18.25cm +/-2.25cm.
        distance_upper = (.2278502255 + .0228430689) - (.1825000155 - .0225)
        # Use the larger observed scene speed, rather than the smaller closing
        # rate, to retain a margin in this calibration comparison.
        speed = .1578030975
        model = speed * p.command_latency_s + speed ** 2 / (2 * p.brake_deceleration_m_s2)
        self.assertGreater(model, distance_upper)
        self.assertGreater(model - distance_upper, .02)
        self.assertEqual(p.approach_deceleration, .08)
        self.assertEqual(p.command_latency_s, .35)

    def test_legacy_policy_keeps_one_bound_and_invalid_planning_is_rejected(self):
        self.assertEqual(configured(brake_deceleration_m_s2=.3).approach_deceleration, .3)
        for planned in (.31, 0., -.1, True):
            with self.subTest(planned=planned), self.assertRaises(ValueError):
                CrosswalkPolicy.from_config(dict(brake_deceleration_m_s2=.3,
                                                approach_deceleration_m_s2=planned))

    def simulate(self, dt, friction, deadzone, delay=0.):
        task = CrosswalkTask(configured(**self.config()))
        speed, traveled, last_distance = .15, 0., 0.
        ack, pending = (53, 53), []
        rest_gap = None
        was_stopping, restarts = False, 0
        for i in range(400):
            now = i * dt
            while pending and pending[0][0] <= now + 1e-9:
                _, ack = pending.pop(0)
            near = .65 - traveled
            scene = dict(motion(last_distance / dt if i else speed, last_distance),
                         sigma_m=.0015, interval_s=dt)
            obs = dict(observation(now, near), sigma_m=.0225) if .09 <= near <= .43 else None
            state = task.update(obs, scene, now, True, ack == (0, 0), ack == (0, 0), True)
            command = task.limit_command((53, 53), state, scene) or (0, 0)
            pending.append((now + delay, command))
            stopping = task.phase in ('STOPPING', 'HOLD', 'RECOVERY_HOLD')
            if stopping:
                self.assertEqual(command, (0, 0))
            if task.phase == 'APPROACH' and was_stopping:
                restarts += 1
            if stopping and speed <= .006 and rest_gap is None:
                rest_gap = near - task.policy.body_front_m
            # Deliver each command at its actual time, not at the next sampled
            # camera frame. Motion observations span the integrated interval.
            tick, end, previous_travel = now, now + dt, traveled
            while tick < end - 1e-9:
                while pending and pending[0][0] <= tick + 1e-9:
                    _, ack = pending.pop(0)
                next_tick = min(end, tick + .01, pending[0][0] if pending else end)
                step = next_tick - tick
                desired = max(0., (ack[0] - deadzone) * .0045) if ack[0] else 0.
                next_speed = speed + max(-friction * step, min(.4 * step, desired - speed))
                traveled += (speed + next_speed) * step / 2
                speed, tick = next_speed, next_tick
            last_distance, was_stopping = traveled - previous_travel, stopping
            if task.passed:
                break
        return task, state, rest_gap, restarts

    def test_single_stop_with_sample_intervals_motor_deadzones_and_delivery_delays(self):
        for dt in (.15, .22, .28):
            for delay in (0., .15, .28, .35):
                for friction in (.167, .2, .4, .8):
                    for deadzone in (0, 18, 30):
                        with self.subTest(dt=dt, delay=delay, friction=friction, deadzone=deadzone):
                            task, state, gap, restarts = self.simulate(dt, friction, deadzone, delay)
                            self.assertEqual(task.passed, 1)
                            self.assertTrue(state['yield_verified'])
                            self.assertGreater(gap, .0225)
                            self.assertLess(gap, .1775)
                            self.assertEqual(restarts, 0)
                            if not delay:
                                self.assertTrue(.10 <= gap <= .15)


if __name__ == '__main__':
    unittest.main()
