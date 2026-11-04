"""Arming uses current support without erasing the road used in occlusions."""

from tests.support import ROOT
import json
import unittest
import numpy as np


from robot.core.geometry import fit_curve
from robot.navigation.road_model import LocalRoadModel
from robot.navigation.tracker import TargetTracker
from robot.control.controller import assess_vision_command

CAL = json.loads((ROOT / "tests/fixtures/camera_pose_20261031_191327.json").read_text())


def observation(start=0.12, end=0.65, y=0.0, quality=0.93):
    path = fit_curve(np.column_stack((np.linspace(start, end, 60), np.full(60, y))))["path"]
    return dict(path=path, confidence=quality, mode="TWO EDGES")


class CurrentSupportTests(unittest.TestCase):
    def test_old_prefix_does_not_veto_trusted_current_component_or_get_refreshed(self):
        model = LocalRoadModel()
        model.update(observation(), 0.0)
        self.assertTrue(model.update(observation(start=0.18), 0.4)["observed"])
        history, old_info = model.support(0.4)
        seen = model.seen.copy()
        geometry = model.path.copy()
        current, info = model.support(0.4, current_only=True)
        self.assertLess(old_info["confidence"], model.policy.confirmation_quality)
        self.assertIsNotNone(current)
        self.assertGreaterEqual(info["confidence"], model.policy.confirmation_quality)
        self.assertEqual(info["fresh_fraction"], 1.0)
        self.assertGreater(current[0, 0], history[0, 0])
        np.testing.assert_array_equal(model.seen, seen)
        np.testing.assert_array_equal(model.path, geometry)
        np.testing.assert_array_equal(model.support(0.4)[0], history)

    def test_history_still_supports_driving_but_cannot_arm(self):
        model = LocalRoadModel()
        model.update(observation(), 0.0)
        self.assertIsNotNone(model.support(0.1)[0])
        path, info = model.support(0.1, current_only=True)
        self.assertIsNone(path)
        self.assertEqual(info["reason"], "NO CURRENT ROAD SUPPORT")
        self.assertEqual(model.last_seen, 0.0)

    def test_short_or_weak_current_fragment_cannot_arm(self):
        model = LocalRoadModel()
        model.update(observation(), 0.0)
        model.seen[:] = 0.0
        model.seen[:3] = 0.1
        self.assertIsNone(model.support(0.1, current_only=True)[0])
        model.seen[:] = 0.1
        model.quality[:] = np.nextafter(model.policy.measurement_quality, 0.0)
        self.assertIsNone(model.support(0.1, current_only=True)[0])

    def test_tracker_arms_from_five_fresh_components_despite_old_prefix(self):
        tracker = TargetTracker(CAL)
        tracker.pixels.measure = lambda *args: (observation(), {})
        tracker.update(None, 0.0, require_current=True)
        tracker.pixels.measure = lambda *args: (observation(start=0.18), {})
        consecutive = 0
        for now in (0.1, 0.2, 0.3, 0.4, 0.5):
            result, _ = tracker.update(None, now, require_current=True)
            self.assertTrue(result["valid"], result)
            self.assertTrue(result["observed"])
            self.assertEqual(result["state"], "OBSERVED")
            self.assertFalse(result["reference_inferred"])
            self.assertEqual(result["road_model"]["support_scope"], "CURRENT")
            _, consecutive = assess_vision_command(
                result, (40, 40), 100, 280, consecutive, tracker.policy.confirmation_quality
            )
        self.assertEqual(consecutive, tracker.policy.startup_observations)
        self.assertTrue(np.any(tracker.model.seen < 0.01))
        self.assertLess(
            tracker.model.support(0.5)[1]["confidence"], tracker.policy.confirmation_quality
        )
        tracker.pixels.measure = lambda *args: (None, {})
        result, _ = tracker.update(None, 0.51)
        self.assertTrue(result["valid"])
        self.assertEqual(result["road_model"]["support_scope"], "HISTORY_ALLOWED")
        self.assertTrue(result["reference_inferred"])
        result, _ = tracker.update(None, 0.52, require_current=True)
        self.assertFalse(result["valid"])
        self.assertFalse(result["observed"])

    def test_stable_accepted_quality_is_confirmed_by_five_current_observations(self):
        # Actual stopped-car diagnostic: accepted q=.6483, but the old .65
        # current-support gate erased the target on every frame forever.
        tracker = TargetTracker(CAL)
        tracker.pixels.measure = lambda *args: (observation(quality=0.6482982100829959), {})
        streak = 0
        for i in range(5):
            result, _ = tracker.update(None, i * 0.1, require_current=True)
            self.assertTrue(result["valid"])
            self.assertTrue(result["observed"])
            self.assertAlmostEqual(result["confidence"], 0.6482982100829959)
            self.assertEqual(result["road_model"]["fresh_fraction"], 1.0)
            command, streak = assess_vision_command(
                result, (40, 40), 100, 280, streak, tracker.policy.measurement_quality
            )
            self.assertEqual(command, (40, 40))
            self.assertEqual(streak, i + 1)
            self.assertEqual(streak >= tracker.policy.startup_observations, i == 4)
        # Current-only startup must still reject stale geometry after one miss.
        tracker.pixels.measure = lambda *args: (None, {})
        result, _ = tracker.update(None, 0.5, require_current=True)
        self.assertFalse(result["valid"])
        self.assertEqual(
            assess_vision_command(
                result, None, 100, 280, streak, tracker.policy.measurement_quality
            )[1],
            0,
        )

    def test_below_measurement_quality_never_accumulates_startup_evidence(self):
        tracker = TargetTracker(CAL)
        tracker.pixels.measure = lambda *args: (observation(quality=0.54), {})
        streak = 0
        for i in range(10):
            result, _ = tracker.update(None, i * 0.1, require_current=True)
            self.assertFalse(result["valid"])
            _, streak = assess_vision_command(
                result, None, 100, 280, streak, tracker.policy.measurement_quality
            )
            self.assertEqual(streak, 0)

    def test_conflicting_measurement_cannot_supply_current_startup_support(self):
        tracker = TargetTracker(CAL)
        tracker.pixels.measure = lambda *args: (observation(), {})
        tracker.update(None, 0.0, require_current=True)
        tracker.pixels.measure = lambda *args: (observation(y=0.25), {})
        result, _ = tracker.update(None, 0.1, require_current=True)
        self.assertEqual(result["state"], "CONFIRMING")
        self.assertEqual(result["measurement_reject"], "ROAD CONFLICT")
        self.assertFalse(result["observed"])
        self.assertFalse(result["valid"])
        command, streak = assess_vision_command(result, None, 100, 280, 4, 0.65)
        self.assertIsNone(command)
        self.assertEqual(streak, 0)


if __name__ == "__main__":
    unittest.main()
