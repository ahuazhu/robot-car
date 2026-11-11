"""Three visible bars may track a confirmed event, never create a new one."""

from tests.support import ROOT
import json
import unittest
import cv2
import numpy as np

from robot.perception.boundaries import CurvePixels
from robot.tasks.crosswalk import CrosswalkDetector, CrosswalkTask
from tests.support.crosswalk import configured, motion


class PartialCrosswalkTrackingTests(unittest.TestCase):
    def setUp(self):
        calibration = json.loads((ROOT / "tests/fixtures/crosswalk_20261024_calibration.json").read_text())
        self.pixels = CurvePixels(calibration)
        self.policy = configured(body_front_m=.02, body_rear_m=.16, body_half_width_m=.085,
                                 stop_clearance_m=.02)
        self.detector = CrosswalkDetector(self.pixels, self.policy)
        self.road = dict(path=np.c_[np.linspace(.1, .8, 30), np.zeros(30)], confidence=1.)

    def image(self, indices, *, forward=.40, angle=0.):
        grid = self.pixels
        xx, yy = np.meshgrid(grid.xs, grid.ys, indexing="ij")
        direction = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        bird = np.full((len(grid.xs), len(grid.ys), 3), 100, np.uint8)
        for index in indices:
            center = np.array([forward, 0.]) + (-.08 + index * .04) * normal
            along = (xx - center[0]) * direction[0] + (yy - center[1]) * direction[1]
            across = (xx - center[0]) * normal[0] + (yy - center[1]) * normal[1]
            bird[(abs(along) < .05) & (abs(across) < .0125)] = 255
        return dict(bird=bird)

    def full_observation(self, now):
        result = self.detector.detect(self.image(range(4)), self.road, now)
        self.assertIsNotNone(result)
        self.assertEqual(result["bar_count"], 4)
        return result

    def begin(self):
        task = CrosswalkTask(self.policy)
        for now in (0., .15, .30):
            task.update(self.full_observation(now), motion(), now, True, False, False, True)
        self.assertEqual(task.phase, "APPROACH")
        return task

    def test_three_bars_do_not_initially_identify_a_crosswalk(self):
        self.assertIsNone(self.detector.detect(self.image(range(1, 4)), self.road, 1.))
        self.assertEqual(self.policy.minimum_bars, 4)

    def test_recorded_three_bar_image_matches_the_documented_reconstructed_template(self):
        fixture = ROOT / "tests/fixtures/crosswalk_20261111"
        pixels = CurvePixels(json.loads((fixture / "calibration.json").read_text()))
        road, details = pixels.measure(cv2.imread(str(fixture / "three_bars.png")))
        detector = CrosswalkDetector(pixels, self.policy)
        self.assertIsNone(detector.detect(details, road, 1.))
        reference = json.loads((fixture / "reference.json").read_text())["reference"]
        found = detector.detect(details, road, 1., reference)
        self.assertIsNotNone(found)
        self.assertEqual(found["bar_count"], 3)
        self.assertTrue(found["tracking_only"])
        self.assertEqual(found["event_id"], reference["event_id"])

    def test_reference_tracks_the_same_three_bars_but_rejects_unrelated_patterns(self):
        reference = dict(self.full_observation(0.), event_id=7)
        partial = self.detector.detect(self.image(range(1, 4)), self.road, .15, reference)
        self.assertIsNotNone(partial)
        self.assertEqual(partial["bar_count"], 3)
        self.assertTrue(partial["tracking_only"])
        self.assertEqual(partial["event_id"], 7)
        for indices, forward, angle in ((range(2), .40, 0.), (range(1, 4), .53, 0.),
                                        (range(4, 7), .40, 0.), (range(1, 4), .40, .45)):
            with self.subTest(indices=list(indices), forward=forward, angle=angle):
                candidate = self.detector.detect(self.image(indices, forward=forward, angle=angle),
                                                 self.road, .30, reference)
                self.assertIsNone(candidate)

    def test_partial_reacquires_only_after_multiple_matches_without_rearming_event(self):
        task = self.begin()
        identity, hits = task.identity, task.hits
        # A real motion gap increases the interval; three new image matches
        # must resolve it rather than labelling the old prediction "fresh".
        task.update(None, motion(valid=False), .45, True)
        self.assertGreater(task.track["sigma_m"], .10)
        for step in range(1, 5):
            now = .45 + step * .15
            scene = motion()
            before = dict(task.track)
            reference = task.detection_reference(scene, now)
            self.assertEqual(task.track, before)
            obs = self.detector.detect(self.image(range(1, 4)), self.road, now, reference)
            self.assertIsNotNone(obs)
            state = task.update(obs, scene, now, True, True, True, True)
            if step < self.policy.confirmation_frames:
                self.assertEqual(task.track["observed_at"], .30)
                self.assertEqual(state["observation_association"], "CONFIRMING_PARTIAL")
            else:
                self.assertEqual(state["position_source"], "TRACKED")
                self.assertLess(state["position_error_m"], .04)
                self.assertEqual(task.track["observed_at"], now)
                self.assertGreater(min(task.limit_command((100, 120), state, scene)), 0)
            self.assertEqual(task.identity, identity)
            self.assertEqual(task.hits, hits)
            self.assertFalse(state["yield_completed"])

    def test_hidden_nearest_bar_is_preserved_across_repeated_partial_updates(self):
        reference = dict(self.full_observation(0.), event_id=9)
        corners = np.asarray(reference["corners"]).reshape(-1, 4, 2)
        corners[0, :, 0] -= .03
        reference["corners"] = corners.reshape(-1, 2).tolist()
        reference["near_m"] = float(np.min(corners[:, :, 0]))
        original_near = reference["near_m"]
        for step in range(15):
            candidate = self.detector.detect(self.image(range(1, 4)), self.road,
                                             .15 + step * .15, reference)
            self.assertIsNotNone(candidate)
            self.assertLessEqual(candidate["near_m"] - candidate["sigma_m"], original_near)
            self.assertAlmostEqual(candidate["near_m"], original_near, places=5)
            self.assertEqual(len(candidate["corners"]), 16)
            self.assertEqual(len(candidate["observed_corners"]), 12)
            reference = candidate

    def test_short_occlusion_then_continuous_partial_tracking_preserves_served_event(self):
        task = self.begin()
        task.phase = "PASSING"
        task.yield_completed = task.yield_verified = True
        task.update(None, motion(valid=False), .45, True)
        identity = task.identity
        for step in range(1, 8):
            now = .45 + step * .15
            reference = task.detection_reference(motion(), now)
            self.assertIsNotNone(reference)
            obs = self.detector.detect(self.image(range(1, 4)), self.road, now, reference)
            self.assertIsNotNone(obs)
            self.assertTrue(obs["tracking_only"])
            state = task.update(obs, motion(), now, True, True, True, True)
            self.assertEqual(task.identity, identity)
            self.assertEqual(state["phase"], "PASSING")
            self.assertTrue(state["yield_completed"])
            self.assertIsNone(state["pending_observation"])

    def test_full_view_after_partial_continuity_does_not_replace_served_event(self):
        task = self.begin()
        task.phase = "PASSING"
        task.yield_completed = task.yield_verified = True
        task.update(None, motion(valid=False), .45, True)
        for now in (.60, .75):
            reference = task.detection_reference(motion(), now)
            obs = self.detector.detect(self.image(range(1, 4)), self.road, now, reference)
            self.assertIsNotNone(obs)
            task.update(obs, motion(), now, True, True, True, True)
        self.assertEqual(task.tracking_hits, 2)
        state = task.update(self.full_observation(.90), motion(), .90, True, True, True, True)
        self.assertEqual(task.identity, 1)
        self.assertEqual(state["phase"], "PASSING")
        self.assertEqual(state["observation_association"], "MATCHED")
        self.assertTrue(state["yield_completed"])
        self.assertIsNone(state["pending_observation"])

    def test_partial_from_another_event_or_without_full_confirmation_cannot_authorize(self):
        task = self.begin()
        obs = self.detector.detect(self.image(range(1, 4)), self.road, .45,
                                   task.detection_reference(motion(), .45))
        self.assertIsNotNone(obs)
        for candidate_task in (task, CrosswalkTask(self.policy)):
            for now in (.45, .60, .75):
                wrong = dict(obs, observed_at=now, event_id=task.identity + 1)
                candidate_task.update(wrong, motion(), now, True, True, True, True)
            if candidate_task is task:
                self.assertEqual(candidate_task.track["observed_at"], .30)
                self.assertEqual(candidate_task.tracking_hits, 0)
            else:
                self.assertIsNone(candidate_task.track)
                self.assertEqual(candidate_task.hits, 0)


if __name__ == "__main__":
    unittest.main()
