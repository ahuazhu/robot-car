"""Local road quality and observation time do not bypass command safety gates."""
from tests.support import ROOT
from tests.support.driving import road, flow
import json
import unittest
import numpy as np
from robot.core.road_policy import RoadPolicy
from robot.navigation.steering_control import SteeringControl
from robot.navigation.tracker import TargetTracker


def controller():
    return SteeringControl(RoadPolicy(), [0., 0.], [.075, .105, .085], .3,
                           lever=.15, gain=.85, cruise=120., straight_cruise=163., confirmed_cruise=140.)


def profiled():
    result = road(confidence=.3, observation_confidence=.9)
    quality = np.full(len(result["path"]), .9)
    quality[(result["path"][:, 0] < -.15) | (result["path"][:, 0] > .6)] = .3
    result["path_confidence"] = quality.tolist()
    result["path_observed_mask"] = (quality > .5).tolist()
    return result


def command(c, result, now, measured=True, age=.1):
    c.observe(flow(measured), now, False)
    return c.command(result, age)


class StableRoadSpeedTests(unittest.TestCase):
    def test_unused_low_quality_tails_do_not_slow_a_fresh_feasible_prefix(self):
        result = profiled()
        legacy = dict(result)
        legacy.pop("path_confidence")
        old = command(controller(), legacy, 0.)
        c = controller()
        new = command(c, result, 0.)
        self.assertGreater(sum(new), 1.6*sum(old))
        self.assertFalse(c.status["speed_plan"]["boost_eligible"])
        self.assertEqual(c.status["speed_plan"]["selected_cruise_pwm"], 120.)
        self.assertEqual(c.status["speed_plan"]["control_confidence"], .9)
        self.assertEqual(result["confidence"], .3)
        json.dumps(c.status, allow_nan=False)

    def test_low_quality_before_goal_or_in_body_rollout_remains_a_speed_limit(self):
        for x in (.15, .35):
            with self.subTest(x=x):
                result = profiled()
                index = int(np.argmin(abs(result["path"][:, 0]-x)))
                result["path_confidence"][index] = .3
                result["path_observed_mask"][index] = False
                legacy = dict(result); legacy.pop("path_confidence")
                c = controller()
                self.assertEqual(command(c, result, 0.), command(controller(), legacy, 0.))
                self.assertEqual(c.status["speed_plan"]["control_confidence"], .3)

    def test_current_goal_motion_and_body_evidence_remain_mandatory_for_local_quality(self):
        cases = [dict(state="PREDICTED", observed=False),
                 dict(transition_pending=True), dict(path_observed_mask=[False]*100)]
        for changes in cases:
            result = dict(profiled(), **changes)
            legacy = dict(result); legacy.pop("path_confidence")
            self.assertEqual(command(controller(), result, 0.), command(controller(), legacy, 0.))
        result = profiled();legacy=dict(result);legacy.pop("path_confidence")
        self.assertEqual(command(controller(), result, 0., measured=False),
                         command(controller(), legacy, 0., measured=False))

    def test_malformed_or_missing_quality_profiles_preserve_old_command(self):
        result = profiled();legacy=dict(result);legacy.pop("path_confidence")
        old = command(controller(), legacy, 0.)
        for profile in ([], [1.], [float("nan")]*100, ["bad"]*100, [0.]*100, [1.]*100, [1.1]*100):
            with self.subTest(profile=str(profile[:1])):
                self.assertEqual(command(controller(), dict(result,path_confidence=profile), 0.), old)

    def test_far_or_short_goal_suspends_boost_without_erasing_continuous_road_proof(self):
        for unavailable in (road(start=.4,end=.7),road(start=.06,end=.12)):
            with self.subTest(start=float(unavailable["path"][0,0])):
                c = controller()
                for now in (0., .15, .3, .45):
                    command(c, road(), now)
                self.assertTrue(c.status["speed_plan"]["boost_eligible"])
                command(c, unavailable, .6)
                self.assertFalse(c.status["speed_plan"]["boost_eligible"])
                command(c, road(), .75)
                self.assertTrue(c.status["speed_plan"]["boost_eligible"])
                self.assertGreaterEqual(c.status["speed_plan"]["road_confirmed_s"], .75)

    def test_new_road_adoption_long_image_gap_and_stale_frame_restart_proof(self):
        for kind in ("new_road", "image_gap", "stale"):
            with self.subTest(kind=kind):
                c = controller()
                for now in (0., .15, .3, .45): command(c, road(), now)
                if kind == "new_road":
                    command(c, road(reacquired=True), .6)
                elif kind == "image_gap":
                    command(c, road(), 1.)
                else:
                    self.assertIsNone(command(c, road(), .6, age=.36))
                    command(c, road(), .75)
                self.assertFalse(c.status["speed_plan"]["boost_eligible"])
                self.assertEqual(c.status["speed_plan"]["road_confirmed_s"], 0.)

    def test_sharp_curve_and_blocked_body_do_not_gain_from_local_quality(self):
        for result in (road(3.,confidence=.3,observation_confidence=.9), profiled()):
            if result.get("path_confidence") is not None:
                result["path"][:,1] += .13
            else:
                result["path_confidence"] = [.9]*len(result["path"])
            legacy=dict(result);legacy.pop("path_confidence")
            self.assertEqual(command(controller(),result,0.),command(controller(),legacy,0.))

    def test_tracker_profile_matches_control_path_and_preserves_global_minimum(self):
        cal=json.loads((ROOT/'tests/fixtures/camera_pose_20261031_191327.json').read_text())
        tracker=TargetTracker(cal)
        tracker.pixels.measure=lambda *args:(dict(path=road(start=.12,end=.6)["path"],confidence=.9,mode="ONE LEFT"),{})
        result,_=tracker.update(None,0.)
        self.assertEqual(len(result["path_confidence"]),len(result["control_path_m"]))
        self.assertAlmostEqual(min(result["path_confidence"]),result["confidence"])
        self.assertNotIn("confidence_profile",result["road_model"])
        json.dumps(result,allow_nan=False)


if __name__ == '__main__':unittest.main()
