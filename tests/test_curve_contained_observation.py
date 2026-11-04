"""The real 7.7cm patch must update its interval without losing the old road."""

from tests.support import ROOT
import json, unittest
from unittest.mock import patch
import numpy as np

from robot.core.geometry import fit_curve, contained_observation, curve_agreement, arc
from robot.navigation.road_model import LocalRoadModel


def path(start=0.12, end=0.6, y=0.0):
    return fit_curve(np.c_[np.linspace(start, end, 60), np.full(60, y)])["path"]


def measure(p, q=0.9):
    return dict(path=p, confidence=q, mode="ONE LEFT")


class ContainedObservationTests(unittest.TestCase):
    def test_recorded_false_pause_becomes_observed_update_without_stationary_gate(self):
        data = json.loads((ROOT / "tests/fixtures/curve_actual_contained.json").read_text())
        a = np.asarray(data["old_path"])
        b = np.asarray(data["candidate_path"])
        self.assertIsNone(curve_agreement(a, b))
        m = LocalRoadModel()
        m.initialize(a, 0.0, 0.9, "ONE LEFT")
        m.variance[:] = data["old_sigma_m"] ** 2
        event = m.update(
            dict(path=b, confidence=data["candidate_confidence"], mode=data["candidate_mode"]),
            0.1,
            False,
            allow_handoff=True,
        )
        self.assertTrue(event["observed"])
        self.assertEqual(event["association_kind"], "CONTAINED_OBSERVATION")
        self.assertEqual(m.pending_count, 0)
        self.assertIsNone(event["reject"])
        self.assertGreater(arc(m.path)[-1], arc(b)[-1] + 0.05)
        self.assertTrue(np.any(m.seen == 0.0))
        self.assertTrue(np.any(m.seen == 0.1))
        np.testing.assert_array_equal(m.path[0], a[0])
        np.testing.assert_array_equal(m.path[-1], a[-1])

    def test_short_patch_preserves_exact_old_tail_evidence(self):
        m = LocalRoadModel()
        old = path()[::2]
        m.initialize(old, 0.0, 0.9, "ONE LEFT")
        patch_path = path(0.30, 0.38, 0.001)
        old_seen = m.seen.copy()
        old_variance = m.variance.copy()
        event = m.update(measure(patch_path), 0.2)
        self.assertTrue(event["observed"])
        self.assertEqual(event["association_kind"], "CONTAINED_OBSERVATION")
        for point, seen, var in zip(old, old_seen, old_variance):
            if point[0] < 0.29 or point[0] > 0.39:
                index = np.argmin(np.linalg.norm(m.path - point, axis=1))
                np.testing.assert_array_equal(m.path[index], point)
                self.assertEqual(m.seen[index], seen)
                self.assertEqual(m.variance[index], var)
        fresh = m.path[m.seen == 0.2]
        np.testing.assert_array_equal(fresh, patch_path)
        self.assertIsNotNone(m.support(0.2, current_only=True)[0])

    def test_reverse_crossing_offset_and_too_short_overlap_fail_closed(self):
        old = path()
        for new in (
            path(0.3, 0.38, 0.04),
            path(0.3, 0.38)[::-1],
            np.c_[np.full(20, 0.35), np.linspace(-0.1, 0.1, 20)],
            np.c_[np.linspace(0.3, 0.34, 10), np.zeros(10)],
        ):
            self.assertIsNone(contained_observation(old, new, 0.01, 0.75, 0.06))

    def test_nearest_endpoint_clamping_does_not_fake_containment(self):
        old = path(0.2, 0.4)
        for new in (path(0.1, 0.3), path(0.3, 0.5), path(0.4, 0.6)):
            self.assertIsNone(contained_observation(old, new, 0.01, 0.75, 0.06))

    def test_uncertain_or_expired_prior_cannot_be_used_for_contained_update(self):
        for now, quality in ((0.7, 0.9), (0.1, 0.64)):
            m = LocalRoadModel()
            m.initialize(path(), 0.0, 0.9, "ONE LEFT")
            event = m.update(measure(path(0.3, 0.38), quality), now)
            self.assertNotEqual(event["association_kind"], "CONTAINED_OBSERVATION")

    def test_geometry_failure_leaves_model_arrays_unchanged(self):
        m = LocalRoadModel()
        m.initialize(path(), 0.0, 0.9, "ONE LEFT")
        prior = m.path.copy()
        seen = m.seen.copy()
        proof = contained_observation(prior, path(0.3, 0.38), 0.01, 0.75, 0.06)
        with patch("robot.navigation.road_model.describe_curve", return_value=None):
            self.assertFalse(
                m.accept_contained(path(0.3, 0.38), proof, prior, measure(path(0.3, 0.38)), 0.2)
            )
        np.testing.assert_array_equal(m.path, prior)
        np.testing.assert_array_equal(m.seen, seen)

    def test_repeated_same_patch_does_not_grow_model_or_refresh_far_tail(self):
        m = LocalRoadModel()
        old = path()[::2]
        m.initialize(old, 0.0, 0.9, "ONE LEFT")
        new = path(0.3, 0.38)
        sizes = []
        for now in (0.1, 0.2, 0.3, 0.4):
            event = m.update(measure(new), now)
            self.assertTrue(event["observed"])
            sizes.append(len(m.path))
            self.assertTrue(np.all(m.seen[m.path[:, 0] > 0.45] < 0.01))
        self.assertLessEqual(max(sizes), len(old) + 2 * len(new))


if __name__ == "__main__":
    unittest.main()
