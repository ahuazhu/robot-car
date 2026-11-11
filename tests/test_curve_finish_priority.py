"""Finishing policy retries look-around; only a usable path permits translation."""

from tests.support import ROOT
import unittest
from unittest.mock import patch

from robot.core.road_policy import RoadPolicy
from robot.navigation.road_recovery import RoadSearch
from tests.test_curve_visual_scan import fresh, missing, motion
from tests.test_curve_active_search import motion as measured_motion


def current_path(**changes):
    result = fresh(
        road_evidence="OBSERVED", target_inferred=False, reference_sigma_m=0.01,
        control_path_m=[[0.1 + i * 0.02, 0.0] for i in range(8)],
        path_observed_mask=[True] * 8,
    )
    result.update(changes)
    return result


class FinishPriorityTests(unittest.TestCase):
    def search(self, **changes):
        return RoadSearch(
            RoadPolicy(), [0.0, 0.0], [0.075, 0.105, 0.085], 0.3,
            allow_visual_scan=True, finish_priority=True, **changes,
        )

    def call(self, search, now, *, result=None, follow=None, moving=None,
             allowed=True, active=True, stopped=True, issued=True):
        command = search.command(
            missing() if result is None else result, follow,
            motion() if moving is None else moving, now,
            active=active, allowed=allowed, stopped=stopped,
        )
        search.applied(now, command, issued=issued)
        return command

    def test_weak_and_predicted_paths_follow_at_capped_effort_with_same_ratio(self):
        for result in (fresh(observation_confidence=0.6),
                       fresh(state="PREDICTED", observed=False),
                       fresh(transition_pending=True)):
            with self.subTest(result=result):
                search = self.search()
                self.assertEqual(self.call(search, 0, result=result, follow=(60, 120)), (30, 60))
                self.assertEqual(search.snapshot()["phase"], "CAUTIOUS")
                self.assertTrue(search.snapshot()["search_pending"])
                self.assertFalse(search.is_turning())

    def test_good_path_keeps_planner_command_and_zero_is_not_turned_into_drive(self):
        search = self.search()
        self.assertEqual(self.call(search, 0, result=fresh(), follow=(100, 120)), (100, 120))
        self.assertEqual(search.snapshot()["phase"], "IDLE")
        self.assertEqual(self.call(search, 1, result=fresh(), follow=(0, 0)), (0, 0))

    def test_current_measured_steering_path_resumes_immediately_at_low_speed(self):
        search = self.search()
        self.call(search, 0)
        command = self.call(search, 0.1, result=current_path(), follow=(60, 120))
        self.assertEqual(command, (30, 60))
        self.assertIsNone(search.scan.began)
        self.assertEqual(search.snapshot()["phase"], "CAUTIOUS")

    def test_candidate_blind_prefix_or_invalid_geometry_cannot_release_scan(self):
        variants = [
            {"target_inferred": True}, {"road_evidence": "PREDICTED"},
            {"path_observed_mask": [False] * 8},
            {"path_observed_mask": [True, False] * 4},
            {"control_path_m": [[float("nan"), 0]] * 8},
            {"reference_sigma_m": 0.1}, {"transition_pending": True},
            {"age_s": 1.0}, {"observation_confidence": 0.1},
        ]
        for changes in variants:
            with self.subTest(changes=changes):
                search = self.search()
                self.call(search, 0)
                self.assertEqual(self.call(search, 0.1, result=current_path(**changes),
                                           follow=(60, 120)), (0, 0))
                self.assertIsNotNone(search.scan.began)

    def test_expired_prediction_or_body_conflict_cannot_resume_forward_after_scan(self):
        for result, follow in (
            (missing(state_hint='EXPIRED'), None),
            (fresh(state='PREDICTED', valid=False, observed=False), (60, 60)),
            (fresh(state='PREDICTED', observed=False,
                   invalid_reason='BODY ENVELOPE BLOCKED'), (60, 60)),
        ):
            with self.subTest(result=result):
                search = self.search()
                self.call(search, 0)
                output = self.call(search, .1, result=result, follow=follow)
                self.assertEqual(output, (0, 0))
                self.assertIsNotNone(search.scan.began)

    def test_rounds_keep_retrying_instead_of_permanent_exhaustion(self):
        # No usable route means minimum confidence, hence reduced pulse effort.
        search = self.search()
        pulses = []
        for step in range(251):
            command = self.call(search, step / 10)
            if search.is_turning():
                pulses.append(command)
                self.assertEqual(search.snapshot()["lease_s"], 0.12)
                self.assertEqual(sum(command), 0)
                self.assertLessEqual(max(map(abs, command)), 30)
                self.assertFalse(search.snapshot()["space_verified"])
        self.assertGreaterEqual(len(pulses), 12)
        self.assertGreaterEqual(search.scan.round_index, 3)
        self.assertEqual(search.scan.total_pulses, len(pulses))
        self.assertEqual(pulses[:4], [(-10, 10), (10, -10), (10, -10), (-10, 10)])
        self.assertEqual(pulses[4:8], [tuple(-v for v in p) for p in pulses[:4]])

    def test_timeout_retry_requires_new_zero_publication_and_fresh_ack(self):
        search = self.search()
        self.call(search, 0)
        self.call(search, 0.01)
        self.assertEqual(self.call(search, 0.76), (-10, 10))
        self.assertEqual(self.call(search, 6, issued=False), (0, 0))
        self.assertEqual(search.scan.round_index, 2)
        self.assertTrue(search.snapshot()["search_pending"])
        for now in (6.1, 7.0, 8.0):
            self.assertEqual(self.call(search, now, issued=False), (0, 0))
        self.assertIsNone(search.scan.zero_sent_at)
        self.call(search, 8.1)
        self.call(search, 8.2, stopped=False)
        self.call(search, 8.3)
        self.assertEqual(self.call(search, 9.04), (0, 0))
        self.assertEqual(self.call(search, 9.05), (10, -10))
        self.assertEqual(search.scan.total_pulses, 2)

    def test_virtual_edge_block_allows_only_short_counterrotation_not_forward(self):
        search = self.search()
        blocked = current_path(invalid_reason="BODY ENVELOPE BLOCKED")
        for now in (0, 0.01):
            self.assertEqual(self.call(search, now, result=blocked, follow=(90, 90)), (0, 0))
        command = self.call(search, 0.76, result=blocked, follow=(90, 90))
        self.assertEqual(command, (-10, 10))
        self.assertEqual(search.snapshot()["lease_s"], 0.12)
        self.assertEqual(self.call(search, 0.8, result=blocked, follow=(90, 90)), (0, 0))

    def test_exhausted_map_search_can_fall_back_to_visual_scan(self):
        search = self.search()
        search.finished = True
        with patch.object(search, "swept_clear", side_effect=AssertionError("old map reused")):
            self.assertEqual(self.call(search, 1), (0, 0))
        self.assertIsNotNone(search.scan.began)

    def test_measured_rest_shortens_settling_but_missing_flow_keeps_full_wait(self):
        for measured in (False, True):
            with self.subTest(measured=measured):
                search = self.search()
                moving = measured_motion() if measured else motion()
                for now in (0, 0.01, 0.11, 0.21):
                    self.assertEqual(self.call(search, now, moving=moving), (0, 0))
                command = self.call(search, 0.26, moving=moving)
                self.assertEqual(command, (-10, 10) if measured else (0, 0))

    def test_motion_or_ack_loss_breaks_accelerated_settling(self):
        search = self.search()
        self.call(search, 0, moving=measured_motion())
        self.call(search, 0.1, moving=measured_motion())
        self.call(search, 0.2, moving=measured_motion(), stopped=False)
        self.call(search, 0.3, moving=measured_motion())
        self.assertEqual(self.call(search, 0.4, moving=measured_motion()), (0, 0))
        self.assertEqual(self.call(search, 0.55, moving=measured_motion()), (-10, 10))

    def test_noisy_visual_motion_cannot_veto_all_turns_after_sent_zero_settles(self):
        search=self.search()
        noisy=motion(True,.05,.10)
        for now in (0.,.1,.4,.74):
            self.assertEqual(self.call(search,now,moving=noisy),(0,0))
        self.assertEqual(self.call(search,.76,moving=noisy),(-10,10))
        self.assertEqual(search.snapshot()['rest_source'],'ZERO_ACK_FALLBACK')
        self.assertEqual(search.snapshot()['lease_s'],.12)
        self.assertEqual(self.call(search,.8,moving=noisy),(0,0))
        self.assertEqual(self.call(search,1.3,moving=noisy),(0,0))

    def test_map_initial_settling_uses_bounded_scan_in_finish_priority_mode(self):
        search=self.search()
        with patch.object(search,'swept_clear',return_value=True):
            self.assertEqual(self.call(search,0.,moving=motion(True,.1,.1)),(0,0))
        self.assertIsNotNone(search.scan.began)
        self.assertEqual(self.call(search,.76,moving=motion(True,.1,.1)),(-10,10))

    def test_task_stop_and_inactive_mission_cancel_all_proposals(self):
        search = self.search()
        self.call(search, 0)
        self.call(search, 0.01)
        pulse = self.call(search, 0.76, issued=False)
        self.assertIsNone(self.call(search, 0.8, allowed=False))
        self.assertFalse(search.applied(0.81, pulse))
        for now in (1, 2, 10):
            self.assertIsNone(self.call(search, now, allowed=False))
            self.assertFalse(search.is_turning())
        self.assertIsNone(self.call(search, 11, active=False))
        self.assertIsNone(search.scan.began)
        self.assertEqual(search.scan.total_pulses, 0)

    def test_unsent_pulses_do_not_count_even_across_timeout_rounds(self):
        search = self.search()
        self.call(search, 0)
        self.call(search, 0.01)
        self.call(search, 0.76, issued=False)
        self.call(search, 6, issued=False)
        self.call(search, 6.1)
        self.assertEqual(search.scan.total_pulses, 0)
        self.assertEqual(search.scan.round_index, 2)


if __name__ == "__main__":
    unittest.main()
