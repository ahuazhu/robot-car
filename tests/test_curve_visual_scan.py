"""An optional visual scan spends only sent pulses and never rearms a task."""

from tests.support import ROOT
import unittest
from unittest.mock import patch


from robot.core.road_policy import RoadPolicy
from robot.navigation.road_recovery import RoadSearch


def missing(**extra):
    return dict(valid=False, observed=False, state="LOST", **extra)


def fresh(**extra):
    result = dict(
        valid=True,
        observed=True,
        state="OBSERVED",
        observation_confidence=0.8,
        age_s=0.0,
        transition_pending=False,
    )
    result.update(extra)
    return result


def motion(valid=False, speed=0.0, yaw=0.0):
    return dict(valid=valid, speed_m_s=speed, yaw_rad_s=yaw)


class VisualScanTests(unittest.TestCase):
    def search(self, **kwargs):
        return RoadSearch(
            RoadPolicy(), [0.0, 0.0], [0.075, 0.105, 0.085], 0.3, allow_visual_scan=True, **kwargs
        )

    def call(
        self,
        search,
        now,
        *,
        issued=True,
        stopped=True,
        result=None,
        follow=None,
        moving=None,
        active=True,
        allowed=True,
    ):
        command = search.command(
            missing() if result is None else result,
            follow,
            motion() if moving is None else moving,
            now,
            active=active,
            allowed=allowed,
            stopped=stopped,
        )
        search.applied(now, command, issued=issued)
        return command

    def first_pulse(self, search):
        self.assertEqual(self.call(search, 0.0), (0, 0))
        self.assertEqual(self.call(search, 0.01), (0, 0))
        self.assertEqual(self.call(search, 0.759), (0, 0))
        command = self.call(search, 0.76)
        self.assertTrue(search.scan.last_action)
        return command

    def exhaust(self, search):
        pulses = [self.first_pulse(search)]
        now = 0.76
        for index in range(4):
            self.assertEqual(self.call(search, now + 0.01), (0, 0))
            self.assertEqual(self.call(search, now + 0.02), (0, 0))
            now += 0.87
            command = self.call(search, now)
            if index < 3:
                pulses.append(command)
            else:
                self.assertIsNone(command)
        self.assertTrue(search.scan.finished)
        return pulses, now

    def test_four_equal_counterrotations_with_fixed_hint_and_bounded_lease(self):
        search = self.search()
        search.last_hint = -1
        first = self.first_pulse(search)
        pulses = [first]
        now = 0.76
        self.assertEqual(search.snapshot()["lease_s"], 0.12)
        for index in range(3):
            search.last_hint = 1  # The entry hint stays fixed for this scan.
            self.call(search, now + 0.01)
            self.call(search, now + 0.02)
            now += 0.87
            pulses.append(self.call(search, now))
            self.assertEqual(search.snapshot()["lease_s"], 0.12)
            self.assertEqual(search.snapshot()["pulse_index"], index + 2)
        self.assertEqual(pulses, [(30, -30), (-30, 30), (-30, 30), (30, -30)])
        self.assertTrue(
            all(sum(command) == 0 and max(map(abs, command)) <= 30 for command in pulses)
        )
        self.assertEqual(search.snapshot()["space_source"], "BOUNDED_VISUAL_SCAN")
        self.assertFalse(search.snapshot()["space_verified"])
        self.assertTrue(search.snapshot()["search_pending"])

    def test_cruise_effort_is_never_exceeded_by_rounding(self):
        command = self.first_pulse(self.search(cruise=39.0))
        self.assertEqual(command, (-19, 19))

    def test_initial_stop_ack_is_required_and_missing_ack_times_out(self):
        search = self.search()
        for now in (0.0, 0.3, 1.0, 3.0, 5.9):
            self.assertEqual(self.call(search, now, stopped=False), (0, 0))
            self.assertFalse(search.scan.last_action)
        self.assertIsNone(self.call(search, 6.0, stopped=False))
        self.assertTrue(search.scan.finished)
        self.assertEqual(search.scan.pulse_index, 0)

    def test_successful_zero_publication_is_required_before_settling(self):
        search = self.search()
        for now in (0.0, 1.0, 2.0):
            self.assertEqual(self.call(search, now, issued=False), (0, 0))
            self.assertIsNone(search.scan.zero_sent_at)
        self.call(search, 2.1)
        self.call(search, 2.2)
        self.assertEqual(self.call(search, 2.949), (0, 0))
        self.assertEqual(self.call(search, 2.95), (-30, 30))

    def test_unsent_or_different_proposal_spends_no_pulse(self):
        search = self.search()
        self.call(search, 0.0)
        self.call(search, 0.01)
        command = self.call(search, 0.76, issued=False)
        self.assertEqual(command, (-30, 30))
        self.assertEqual(search.scan.pulse_index, 0)
        self.assertFalse(search.applied(0.77, (30, 30)))
        self.assertEqual(search.scan.pulse_index, 0)
        self.assertEqual(self.call(search, 0.8), command)
        self.assertEqual(search.scan.pulse_index, 1)
        self.assertFalse(search.applied(0.81, command))  # Duplicate notification.
        self.assertEqual(search.scan.pulse_index, 1)

    def test_each_pulse_requires_sent_zero_and_full_settling_interval(self):
        search = self.search()
        self.first_pulse(search)
        for now in (0.77, 1.0, 1.6):
            self.assertEqual(self.call(search, now, issued=False), (0, 0))
        self.assertEqual(search.scan.pulse_index, 1)
        self.call(search, 1.7)
        self.call(search, 1.71)
        self.assertEqual(self.call(search, 2.459), (0, 0))
        self.assertEqual(self.call(search, 2.46), (30, -30))

    def test_valid_motion_that_has_not_stopped_restarts_rest_wait(self):
        for moving in (motion(True, 0.1, 0.0), motion(True, 0.0, 0.1)):
            search = self.search()
            self.call(search, 0.0)
            self.call(search, 0.01)
            self.assertEqual(self.call(search, 0.75, moving=moving), (0, 0))
            self.assertIsNone(search.scan.rest_since)
            self.call(search, 0.8, moving=motion(True))
            self.assertEqual(self.call(search, 1.549, moving=motion(True)), (0, 0))
            self.assertEqual(self.call(search, 1.55, moving=motion(True)), (-30, 30))

    def test_fresh_ack_loss_restarts_rest_wait(self):
        search = self.search()
        self.call(search, 0.0)
        self.call(search, 0.01)
        self.call(search, 0.7, stopped=False)
        self.call(search, 0.8)
        self.assertEqual(self.call(search, 1.549), (0, 0))
        self.assertEqual(self.call(search, 1.55), (-30, 30))

    def test_new_map_does_not_switch_running_scan_to_measured_search(self):
        search = self.search()
        self.first_pulse(search)
        with patch.object(search, "swept_clear", side_effect=AssertionError("scan changed mode")):
            self.assertEqual(self.call(search, 0.77, moving=motion(True)), (0, 0))
        self.assertEqual(search.snapshot()["space_source"], "BOUNDED_VISUAL_SCAN")

    def test_stop_clears_scan_and_task_cancellation_does_not_resume_it(self):
        search = self.search()
        self.first_pulse(search)
        self.assertIsNone(self.call(search, 0.8, allowed=False))
        self.assertTrue(search.scan.finished)
        self.assertIsNone(self.call(search, 1.8))
        self.assertEqual(search.scan.pulse_index, 1)
        self.assertIsNone(self.call(search, 2.0, active=False))
        self.assertIsNone(search.scan.began)
        self.assertFalse(search.scan.finished)
        self.assertEqual(self.call(search, 2.1), (0, 0))
        self.assertEqual(search.scan.pulse_index, 0)

    def test_allowed_false_cannot_start_scan_or_return_recovered_follow(self):
        search = self.search()
        self.assertIsNone(self.call(search, 0.0, allowed=False))
        self.assertIsNone(search.scan.began)
        self.first_pulse(search)
        self.assertIsNone(self.call(search, 0.8, allowed=False, result=fresh(), follow=(30, 30)))
        self.assertTrue(search.scan.finished)

    def test_explicit_body_envelope_block_prevents_or_aborts_scan(self):
        search = self.search()
        blocked = missing(invalid_reason="BODY ENVELOPE BLOCKED")
        self.assertIsNone(self.call(search, 0.0, result=blocked))
        self.assertIsNone(search.scan.began)
        self.first_pulse(search)
        self.assertIsNone(self.call(search, 0.8, result=blocked))
        self.assertTrue(search.scan.finished)

    def test_all_four_pulses_exhaust_once_and_never_restart_on_missing_road(self):
        search = self.search()
        pulses, now = self.exhaust(search)
        self.assertEqual(len(pulses), 4)
        self.assertEqual(search.snapshot()["phase"], "EXHAUSTED")
        self.assertFalse(search.snapshot()["search_pending"])
        for later in (now + 1, now + 5, now + 100):
            self.assertIsNone(self.call(search, later))
            self.assertEqual(search.scan.pulse_index, 4)

    def test_wall_clock_budget_includes_settling_and_unsent_proposals(self):
        search = self.search()
        self.first_pulse(search)
        self.assertIsNone(self.call(search, 6.0))
        self.assertTrue(search.scan.finished)
        self.assertEqual(search.scan.pulse_index, 1)
        self.assertIsNone(self.call(search, 7.0))

    def test_weak_predicted_stale_transition_and_zero_follow_do_not_reset_budget(self):
        variants = [
            (fresh(state="PREDICT", observed=False), (30, 30)),
            (fresh(observation_confidence=0.1), (30, 30)),
            (fresh(observation_confidence=0.0), (30, 30)),
            (fresh(transition_pending=True), (30, 30)),
            (fresh(age_s=1.0), (30, 30)),
            (fresh(), (0, 0)),
            (fresh(), None),
        ]
        for result, follow in variants:
            search = self.search()
            self.exhaust(search)
            self.assertIsNone(self.call(search, 5.0, result=result, follow=follow))
            self.assertTrue(search.scan.finished)
            self.assertEqual(search.scan.pulse_index, 4)

    def test_confirmed_fresh_road_recovers_without_new_pulse_and_stops_flicker_reset(self):
        search = self.search()
        self.first_pulse(search)
        self.assertEqual(self.call(search, 0.8, result=fresh(), follow=(31, 33)), (0, 0))
        self.assertEqual(search.scan.pulse_index, 1)
        self.call(search, 0.9, result=fresh(state="PREDICT", observed=False), follow=(31, 33))
        self.assertEqual(search.scan.recovery_frames, 0)
        for index in range(4):
            self.assertEqual(
                self.call(search, 1.0 + index * 0.1, result=fresh(), follow=(31, 33)), (0, 0)
            )
            self.assertEqual(search.scan.pulse_index, 1)
        self.assertEqual(self.call(search, 1.4, result=fresh(), follow=(31, 33)), (31, 33))
        self.assertEqual(search.snapshot()["phase"], "IDLE")
        self.assertIsNone(search.scan.began)
        self.assertFalse(search.scan.last_action)

    def test_exhausted_scan_can_recover_on_five_fresh_road_observations(self):
        search = self.search()
        self.exhaust(search)
        for index in range(4):
            self.assertEqual(
                self.call(search, 5.0 + index * 0.1, result=fresh(), follow=(31, 33)), (0, 0)
            )
            self.assertTrue(search.scan.finished)
        self.assertEqual(self.call(search, 5.4, result=fresh(), follow=(31, 33)), (31, 33))
        self.assertFalse(search.scan.finished)
        self.assertEqual(search.snapshot()["phase"], "IDLE")

    def test_task_interruption_breaks_the_five_frame_recovery_sequence(self):
        search = self.search()
        self.first_pulse(search)
        for index in range(4):
            self.assertEqual(
                self.call(search, 0.8 + index * 0.1, result=fresh(), follow=(31, 33)), (0, 0)
            )
        self.assertIsNone(self.call(search, 1.2, allowed=False))
        for index in range(4):
            self.assertEqual(
                self.call(search, 1.3 + index * 0.1, result=fresh(), follow=(31, 33)), (0, 0)
            )
            self.assertTrue(search.scan.finished)
        self.assertEqual(self.call(search, 1.7, result=fresh(), follow=(31, 33)), (31, 33))

    def test_default_disabled_keeps_legacy_missing_evidence_behavior(self):
        search = RoadSearch(RoadPolicy(), [0.0, 0.0], [0.075, 0.105, 0.085], 0.3)
        self.assertIsNone(self.call(search, 0.0))
        self.assertIsNone(search.scan.began)
        self.assertEqual(search.snapshot()["phase"], "WAIT_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
