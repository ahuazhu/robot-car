"""Recorded crosswalk exit: road revision trust, braking, and scan handoff."""
from tests.support import ROOT
from tests.support.driving import road, flow
from robot.core.road_policy import RoadPolicy
from robot.navigation.road_model import LocalRoadModel
from robot.navigation.road_recovery import RoadSearch
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits
from robot.navigation.steering_control import SteeringControl
import copy
import json
import unittest
import numpy as np

FIXTURE=json.loads((ROOT/'tests/fixtures/crosswalk_route_revision_20261111.json').read_text())


class CrosswalkRouteControlTests(unittest.TestCase):
    def search(self,now):
        search=RoadSearch(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,
                          allow_visual_scan=True,finish_priority=True)
        zero=search.command(dict(valid=False),None,dict(valid=False),now-.2,
                            active=True,stopped=True)
        search.applied(now-.2,zero)
        self.assertIsNotNone(search.scan.began)
        return search

    def test_recorded_current_roads_cancel_scan_despite_old_retained_tails(self):
        for row in FIXTURE['recovered']:
            with self.subTest(frame=row['frame']):
                result=row['result'];follow=tuple(row['follow'])
                self.assertGreater(result['age_s'],.35)
                self.assertEqual(result['road_model']['model_age_s'],0.)
                search=self.search(10.)
                command=search.command(result,follow,dict(valid=False),10.,
                                       active=True,allowed=True,stopped=True)
                self.assertEqual(command,follow)
                self.assertIsNone(search.scan.began)
                self.assertFalse(search.is_turning())

    def test_actual_stale_observation_is_not_refreshed_by_a_young_tail(self):
        result=copy.deepcopy(FIXTURE['recovered'][0]['result'])
        result['age_s']=0.;result['road_model']['model_age_s']=.5
        search=self.search(10.)
        command=search.command(result,(20,30),dict(valid=False),10.,
                               active=True,allowed=True,stopped=True)
        self.assertNotEqual(command,(20,30))
        self.assertIsNotNone(search.scan.began)

    def test_accepted_path_brake_cancels_scan_instead_of_rotating(self):
        speed=ProfileSpeedControl(SpeedLimits(.3,.15,.4,.2,.35,.15),163,.4,controller='DWPP')
        steering=SteeringControl(RoadPolicy(),[0.,0.],(.075,.105,.085),.3,
                                 cruise=145,speed_control=speed)
        motion=dict(flow(),translation=np.array([-.045,0.]))
        steering.observe(motion,10.,False)
        speed.applied(9.95,(0,0))
        target=road(state='PREDICTED',observed=False,confidence=.08,
                    reference_sigma_m=.005,path_observed_mask=[False]*100,
                    road_model=dict(persistent_history=True))
        command=steering.command(target,.1)
        self.assertEqual(command,(0,0))
        self.assertTrue(speed.status['selected'])
        result=dict(target,speed_limit_source='CENTERLINE_CONFIDENCE')
        search=self.search(10.1)
        self.assertEqual(search.command(result,command,motion,10.1,
                                       active=True,stopped=False),(0,0))
        self.assertIsNone(search.scan.began)

    def adopted_model(self):
        record=FIXTURE['revision'];quality=record['quality']
        model=LocalRoadModel(persistent_history=True,reference_origin_m=(-.055,0.))
        model.initialize(record['old_path'],0.,.8,'ONE RIGHT')
        observation=dict(path=np.asarray(record['candidate']),confidence=quality,mode='ONE RIGHT')
        for i in range(1,12):
            event=model.update(observation,i*.15,stationary=False)
            if event['reacquired']:return model,observation,i*.15
            np.testing.assert_allclose(model.path,record['old_path'])
        self.fail('Recorded replacement never confirmed')

    def test_large_confirmed_revision_retains_low_trust_until_new_evidence(self):
        model,observation,now=self.adopted_model()
        np.testing.assert_allclose(model.path,observation['path'])
        _,initial=model.support(now)
        self.assertLess(initial['confidence'],.3)
        confidence=[initial['confidence']]
        for i in range(1,13):
            event=model.update(observation,now+i*.15,stationary=False)
            self.assertTrue(event['observed'])
            confidence.append(model.support(now+i*.15)[1]['confidence'])
        self.assertEqual(confidence,sorted(confidence))
        self.assertGreater(confidence[-1],.7)
        self.assertLess(confidence[1],confidence[-1])

    def test_no_measurement_cannot_restore_revision_trust_or_delete_its_geometry(self):
        model,observation,now=self.adopted_model()
        trust=model.observation_agreement;path=model.path.copy()
        for i in (1.,5.,60.):model.update(None,now+i,stationary=False)
        self.assertEqual(model.observation_agreement,trust)
        np.testing.assert_array_equal(model.path,path)

    def test_explicit_reset_still_gives_new_mission_a_clean_start(self):
        model,observation,now=self.adopted_model()
        self.assertLess(model.observation_agreement,.5)
        model.reset();model.update(observation,now+1.)
        self.assertEqual(model.observation_agreement,1.)


if __name__=='__main__':unittest.main()
