"""Idle direction follows observed support without retaining a search map."""

from tests.support import ROOT
import copy
import unittest
from unittest.mock import patch
import numpy as np


from robot.core.road_policy import RoadPolicy
from robot.navigation.road_recovery import RoadSearch


def road(y=0.08, start=0.1, end=0.5, count=12, **extra):
    points = np.c_[np.linspace(start, end, count), np.full(count, y)]
    result = dict(
        observed=True,
        path=points,
        path_observed_mask=[True] * count,
        road_model=dict(sigma_m=0.005),
    )
    result.update(extra)
    return result


class SearchHintTests(unittest.TestCase):
    def search(self, origin=(0.0, 0.0), body=(0.075, 0.105, 0.085)):
        return RoadSearch(RoadPolicy(), origin, body, 0.3, allow_visual_scan=True)

    def compare(self, result, initial=1.0, origin=(0.0, 0.0), body=(0.075, 0.105, 0.085)):
        full = self.search(origin, body)
        idle = self.search(origin, body)
        full.last_hint = idle.last_hint = initial
        full.observe(copy.deepcopy(result), None, 1.0)
        idle.observe_hint(copy.deepcopy(result))
        self.assertEqual(idle.last_hint, full.last_hint)
        return idle.last_hint

    def test_matches_full_observation_for_direction_and_control_origin(self):
        for offset in (-0.08, 0.08):
            for initial in (-1.0, 1.0):
                self.assertEqual(self.compare(road(offset), initial), np.sign(offset))
        self.assertEqual(self.compare(road(0.08), origin=(0.0, 0.1)), -1.0)

    def test_preserves_observed_body_path_mask_and_support_guards(self):
        cases = [
            road(observed=False),
            road(path=[]),
            road(path=[1.0, 2.0]),
            road(path=np.zeros((12, 3))),
            road(path_observed_mask=[True]),
            road(count=4),
            road(start=0.1, end=0.17),
            road(path_observed_mask=[False] * 12),
        ]
        bad = road()
        bad["path"][3, 0] = np.nan
        cases.append(bad)
        for result in cases:
            with self.subTest(result=result):
                self.assertEqual(self.compare(result, initial=-1.0), -1.0)
        self.assertEqual(self.compare(road(), initial=-1.0, body=None), -1.0)
        # Equality with the minimum supported length remains accepted.
        self.assertEqual(self.compare(road(start=0.0, end=0.075), initial=-1.0), 1.0)

    def test_sigma_threshold_and_default_match_original(self):
        for sigma, y, expected in (
            (0.08, 0.08, -1.0),
            (0.08, 0.081, 1.0),
            (0.0, 0.004, -1.0),
            (None, 0.006, 1.0),
            (0.005, -0.006, -1.0),
        ):
            self.assertEqual(
                self.compare(road(y, road_model=dict(sigma_m=sigma)), initial=-1.0), expected
            )

    def test_last_eligible_run_wins_without_joining_gaps(self):
        result = road(count=13)
        result["path"][:6] = np.c_[np.linspace(0.1, 0.3, 6), np.full(6, 0.08)]
        result["path"][7:] = np.c_[np.linspace(0.1, 0.3, 6), np.full(6, -0.08)]
        result["path_observed_mask"][6] = False
        self.assertEqual(self.compare(result), -1.0)
        result["path"][7:, 1] = 0.003
        self.assertEqual(self.compare(result, initial=-1.0), 1.0)
        result["path"][7:, 0] = np.linspace(0.1, 0.12, 6)
        result["path"][7:, 1] = -0.08
        self.assertEqual(self.compare(result, initial=-1.0), 1.0)

    def test_hint_does_not_build_transport_or_modify_other_state(self):
        search = self.search()
        search.observe(road(), None, 1.0)
        search.began = 0.5
        search.finished = True
        before = copy.deepcopy(search.__dict__)
        with (
            patch(
                "robot.navigation.road_recovery.differential",
                side_effect=AssertionError("no differential"),
            ),
            patch(
                "robot.navigation.road_recovery.curve_agreement",
                side_effect=AssertionError("no agreement"),
            ),
            patch(
                "robot.navigation.road_recovery.support_projection",
                side_effect=AssertionError("no projection"),
            ),
            patch(
                "robot.navigation.road_recovery.apply_ground_motion",
                side_effect=AssertionError("no transport"),
            ),
        ):
            search.observe_hint(road(-0.08))
        self.assertEqual(search.last_hint, -1.0)
        for key, expected in before.items():
            if key == "last_hint":
                continue
            actual = search.__dict__[key]
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            elif key in ("patches", "pose_hint"):
                old = expected if isinstance(expected, list) else [expected]
                new = actual if isinstance(actual, list) else [actual]
                self.assertEqual(len(new), len(old))
                for a, b in zip(new, old):
                    np.testing.assert_array_equal(a["path"], b["path"])
                    self.assertEqual(
                        {k: v for k, v in a.items() if k != "path"},
                        {k: v for k, v in b.items() if k != "path"},
                    )
            elif key == "scan":
                self.assertEqual(vars(actual), vars(expected), key)
            else:
                self.assertEqual(actual, expected, key)


if __name__ == "__main__":
    unittest.main()
