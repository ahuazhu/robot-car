"""Route ownership, candidate trust, and spatial uncertainty regressions."""

from tests.support import ROOT
from tests.test_curve_handoff import path, measurement, scene
from robot.config import RuntimeSettings
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.road_model import LocalRoadModel
from robot.navigation.road_recovery import RoadSearch
from robot.navigation.steering_control import SteeringControl
from robot.navigation.tracker import TargetTracker
import math
import unittest
import numpy as np


class RouteReviewTests(unittest.TestCase):
    def setUp(self):
        self.settings=RuntimeSettings.load(ROOT/'maixcam2/curve_follow')

    def controls(self):
        s=self.settings;frame=s.runtime['control_frame'];task=s.crosswalk_policy
        steering=SteeringControl(s.road_policy,frame['origin_m'],frame['body_m'],
            s.calibration['road_width_m'],s.turn_lever,s.steering_gain,s.cruise_pwm,
            task.brake_deceleration_m_s2,task.speed_per_pwm_m_s,
            calibration=s.calibration,response_s=task.speed_response_s,
            straight_cruise=s.straight_cruise_pwm,confirmed_cruise=s.confirmed_cruise_pwm,
            speed_control=ProfileSpeedControl.from_settings(s))
        search=RoadSearch(s.road_policy,frame['origin_m'],frame['body_m'],
            s.calibration['road_width_m'],s.search_cruise_pwm,task.speed_response_s,
            allow_visual_scan=True,finish_priority=True)
        return steering,search

    def test_production_crawl_cannot_be_replaced_by_old_route_rotation(self):
        tracker=TargetTracker(self.settings.road_calibration)
        positions=np.linspace(0.,.5,45);angle=math.radians(80)
        original=np.c_[.1+positions*math.cos(angle),positions*math.sin(angle)]
        original+=self.settings.runtime['control_frame']['origin_m']
        tracker.model.initialize(original,0.,.9,'ONE RIGHT')
        tracker.model.observation_agreement=.01
        tracker.motion.update=lambda *args:scene(False)
        tracker.pixels.measure=lambda *args:(None,{})
        result,details=tracker.update(None,10.)
        steering,search=self.controls()
        steering.observe(details['ground_motion'],10.,allow_feedback=False)
        follow=steering.command(result,.1)
        self.assertIsNotNone(follow);self.assertTrue(any(follow))
        result.update(speed_limit_source='CENTERLINE_CONFIDENCE',
            control_confidence=steering.status['speed_profile_control']['centerline_confidence'])
        self.assertLess(result['control_confidence'],.01)
        self.assertLessEqual(max(map(abs,follow)),self.settings.confirmed_cruise_pwm*.1+1)
        for now in (10.,10.2,11.,20.,70.):
            self.assertEqual(search.command(result,follow,details['ground_motion'],now,
                                           active=True),follow)
            self.assertFalse(search.is_turning())

    def test_direction_hint_uses_confidence_scaled_bounded_pulses_and_task_wins(self):
        efforts=[]
        for confidence in (.01,.9):
            _,search=self.controls()
            result=dict(valid=False,confidence=confidence,
                route_alignment=dict(heading_rad=-math.pi/2,distance_m=.1))
            pulses=[];last_command=None
            for tick in range(160):
                now=tick*.05
                command=search.command(result,None,scene(False),now,active=True,stopped=True)
                if command is not None and any(command):
                    self.assertFalse(last_command is not None and any(last_command))
                    self.assertLessEqual(search.snapshot()['lease_s'],.12)
                    self.assertLessEqual(search.scan.pulse_index,3)
                    pulses.append((now,command))
                search.applied(now,command)
                last_command=command
            self.assertGreaterEqual(len(pulses),4)
            self.assertGreater(search.scan.round_index,1)
            self.assertTrue(all(sum(command)==0 for _,command in pulses))
            self.assertGreater(pulses[0][1][0],0)
            self.assertLess(pulses[1][1][0],0)
            self.assertTrue(all(b[0]-a[0]>.12 for a,b in zip(pulses,pulses[1:])))
            efforts.append(max(abs(pulses[0][1][0]),abs(pulses[0][1][1])))
            for now in (8.,9.,20.):
                command=search.command(result,None,scene(False),now,active=True,
                                       allowed=False,stopped=True)
                self.assertFalse(command is not None and any(command))
                self.assertFalse(search.is_turning())
        self.assertGreater(efforts[0],0)
        self.assertLess(efforts[0],efforts[1])
        self.assertLessEqual(efforts[1],30)

    def test_rejected_less_conflicting_candidate_does_not_restore_old_route_trust(self):
        model=LocalRoadModel(persistent_history=True)
        original=path(.1,.65);model.initialize(original,0.,.9,'ONE RIGHT')
        event=model.update(measurement(path(.1,.65,.25),.95),.1,stationary=False)
        self.assertFalse(event['observed']);trust=model.observation_agreement
        event=model.update(measurement(path(.1,.65,.08),.95),.2,stationary=False)
        self.assertFalse(event['observed']);self.assertEqual(model.pending_count,1)
        self.assertEqual(model.observation_agreement,trust)
        np.testing.assert_array_equal(model.path,original)
        event=model.update(measurement(original,.9),.3,stationary=False)
        self.assertTrue(event['observed'])
        self.assertGreater(model.observation_agreement,trust)
        self.assertLess(model.observation_agreement,1.)

    def test_tracker_publishes_sigma_at_each_retained_control_sample(self):
        tracker=TargetTracker(self.settings.road_calibration)
        origin=np.asarray(self.settings.runtime['control_frame']['origin_m'])
        original=path(.1,.8)+origin
        tracker.model.initialize(original,0.,.9,'TWO EDGES')
        tracker.model.variance[:]=.006**2
        tracker.model.variance[original[:,0]-origin[0]>.6]=.2**2
        tracker.motion.update=lambda *args:dict(scene(),sigma_m=0.)
        tracker.pixels.measure=lambda *args:(measurement(path(.1,.45)+origin),{})
        result,_=tracker.update(None,1.)
        self.assertTrue(result['observed'])
        sigma=np.asarray(result['path_sigma_m'])
        control=np.asarray(result['control_path_m'])
        self.assertEqual(sigma.shape,(len(control),))
        np.testing.assert_allclose(control+origin,result['path'])
        self.assertLess(float(sigma[control[:,0]<.4].max()),.01)
        self.assertAlmostEqual(float(sigma[control[:,0]>.6].max()),.2)
        self.assertEqual(result['road_model']['sigma_m'],float(sigma.max()))


if __name__=='__main__':unittest.main()
