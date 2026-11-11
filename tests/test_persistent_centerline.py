"""Persistent predictions, measured corrections and the recorded single edge."""
from tests.support import ROOT
from tests.support.crosswalk import motion
from tests.test_curve_handoff import CAL, measurement, path, scene
from robot.core.road_policy import RoadPolicy
from robot.navigation.road_model import LocalRoadModel
from robot.navigation.tracker import TargetTracker
from robot.navigation.steering_control import SteeringControl
from robot.navigation.road_recovery import RoadSearch
import cv2
import json
import numpy as np
import unittest

FIXTURE=ROOT/'tests/fixtures/single_edge_stall_20261111'


def recorded_calibration():
    cal=json.loads((FIXTURE/'calibration.json').read_text())
    return dict(cal,road_search=dict(finish_priority=True),
                control_frame=dict(origin_m=[-.055,0.],body_m=[.075,.105,.085],geometry_verified=True),
                road_model=dict(maximum_blind_distance_m=.15,maximum_prediction_command_s=.25))


class PersistentCenterlineTests(unittest.TestCase):
    def tracker(self):
        t=TargetTracker(dict(CAL,road_search=dict(finish_priority=True),
                             road_model=dict(maximum_prediction_command_s=.25)))
        t.motion.update=lambda *args:scene()
        t.pixels.measure=lambda *args:(measurement(path(.1,.65)),{})
        self.assertTrue(t.update(None,0.)[0]['observed'])
        t.pixels.measure=lambda *args:(None,{})
        return t

    def test_model_has_no_age_travel_or_uncertainty_expiry(self):
        m=LocalRoadModel(persistent_history=True)
        original=path(.1,.65);m.initialize(original,0.,.9,'ONE LEFT')
        m.variance[:]=.2**2;m.travel[:]=10.
        m.retire_expired(600.)
        support,state=m.support(600.)
        np.testing.assert_array_equal(m.path,original)
        np.testing.assert_array_equal(support,original)
        self.assertTrue(state['persistent_history'])
        self.assertEqual(state['age_s'],600.)
        self.assertFalse(any(state['observed_mask']))
        self.assertLess(state['confidence'],.9)
        self.assertIsNone(m.support(600.,current_only=True)[0])

    def test_late_prediction_is_used_from_standstill_with_a_fresh_short_motor_lease(self):
        t=self.tracker()
        result,_=t.update(None,8.)
        self.assertTrue(result['valid']);self.assertEqual(result['state'],'PREDICTED')
        self.assertFalse(result['observed']);self.assertTrue(result['target_inferred'])
        self.assertEqual(result['road_model']['oldest_seen'],0.)
        control=SteeringControl(t.policy,[0.,0.],[.075,.105,.085],.3,cruise=135)
        control.observe(dict(scene(),speed_m_s=0.),8.,False)
        command=control.command(result,.1)
        self.assertTrue(any(command));self.assertLessEqual(max(command),135)
        self.assertTrue(control.status['prediction']['persistent_history'])
        self.assertGreater(control.status['prediction']['lease_s'],0.)
        self.assertLessEqual(control.status['prediction']['lease_s'],.25)
        # Prediction retention never grants a stale camera/control command.
        self.assertIsNone(control.command(result,.36))
        self.assertIsNotNone(t.model.path)

    def test_motion_transports_the_same_curve_without_refreshing_seen_samples(self):
        t=self.tracker();old=t.model.path.copy()
        movement=dict(scene(),translation=np.array([-.1,.02]),travel_m=.102)
        t.motion.update=lambda *args:movement
        result,_=t.update(None,2.)
        np.testing.assert_allclose(t.model.path,old+[-.1,.02])
        self.assertTrue(result['valid']);self.assertEqual(t.model.last_seen,0.)
        self.assertFalse(any(result['path_observed_mask']))

    def test_confirmed_better_observation_corrects_prediction_while_moving(self):
        m=LocalRoadModel(persistent_history=True)
        old=path(.1,.65);new=path(.1,.65,.25)
        m.initialize(old,0.,.8,'ONE LEFT')
        for i in range(1,5):
            event=m.update(measurement(new,.95),i*.1,stationary=False)
            if i<4:np.testing.assert_array_equal(m.path,old)
        self.assertTrue(event['reacquired'])
        self.assertEqual(event['association_kind'],'CONFIRMED_PREDICTION_CORRECTION')
        np.testing.assert_array_equal(m.path,new)

    def test_weak_new_observation_cannot_replace_the_retained_prediction(self):
        m=LocalRoadModel(persistent_history=True);old=path(.1,.65)
        m.initialize(old,0.,.9,'ONE LEFT')
        for i in range(10):m.update(measurement(path(.1,.65,.25),.4),float(i),stationary=False)
        np.testing.assert_array_equal(m.path,old)
        self.assertEqual(m.last_seen,0.)

    def test_green_counterevidence_lowers_confidence_without_deleting_prediction(self):
        t=self.tracker();old=t.model.path.copy()
        surface=type('Surface',(),{'contradiction':lambda *args:
            dict(contradicted=True,visible_samples=20,green_samples=20,longest_green_m=.2)})()
        t.pixels.measure=lambda *args:(None,dict(surface=surface))
        result,_=t.update(None,1.)
        np.testing.assert_array_equal(t.model.path,old)
        self.assertTrue(result['valid']);self.assertLess(result['confidence'],.5)
        self.assertTrue(result['road_model']['surface_contradicted'])

    def test_recorded_curve_survives_the_recorded_expiry_threshold(self):
        state=json.loads((FIXTURE/'last_supported_state.json').read_text())['target']
        original=np.asarray(state['control_path_m'])+state['control_reference_origin_m']
        m=LocalRoadModel(persistent_history=True)
        m.initialize(original,0.,max(state['path_confidence']),state['mode'])
        m.variance[:]=.04395676987096308**2;m.travel[:]=.040704982446904306
        support,meta=m.support(.675475854)
        self.assertIsNotNone(support);self.assertTrue(meta['persistent_history'])
        np.testing.assert_array_equal(support,original)

    def test_consumed_curve_remains_stored_and_returns_control_to_search(self):
        t=self.tracker()
        t.motion.update=lambda *args:dict(scene(),translation=np.array([-.9,0.]),travel_m=.9)
        result,_=t.update(None,1.)
        self.assertFalse(result['valid']);self.assertEqual(result['invalid_reason'],'NO FORWARD CURVE POINT')
        self.assertIsNotNone(t.model.path)
        search=RoadSearch(t.policy,[0.,0.],[.075,.105,.085],.3,allow_visual_scan=True,finish_priority=True)
        search.command(result,None,scene(),1.,active=True,stopped=True)
        self.assertTrue(search.snapshot()['search_pending'])

    def test_recorded_left_boundary_and_mirror_produce_cautious_turns_into_road(self):
        image=cv2.imread(str(FIXTURE/'raw.png'))
        for mirrored in (False,True):
            with self.subTest(mirrored=mirrored):
                cal=recorded_calibration();picture=image
                if mirrored:
                    picture=image[:,::-1].copy()
                    reflect=np.array([[-1.,0.,319/320],[0.,1.,0.],[0.,0.,1.]])
                    cal['H_normalized']=(np.diag([1.,-1.,1.])@np.asarray(cal['H_normalized'])@reflect).tolist()
                tracker=TargetTracker(cal);result,details=tracker.update(picture,0.,require_current=True)
                self.assertTrue(result['valid']);self.assertTrue(result['partial_boundary_inference'])
                self.assertEqual(result['mode'],'ONE RIGHT' if mirrored else 'ONE LEFT')
                self.assertGreater(result['confidence'],tracker.policy.measurement_quality)
                steering=SteeringControl(tracker.policy,[-.055,0.],[.075,.105,.085],.3,
                                         cruise=135,calibration=cal)
                steering.observe(dict(motion(),interval_s=.1),.1,False)
                command=steering.command(result,.1)
                self.assertTrue(any(command));self.assertLessEqual(max(command),135)
                self.assertTrue(command[1]>command[0] if mirrored else command[0]>command[1])
                search=RoadSearch(tracker.policy,[-.055,0.],[.075,.105,.085],.3,
                                  allow_visual_scan=True,finish_priority=True)
                search.command(dict(valid=False,state='LOST'),None,motion(),0.,active=True,stopped=True)
                accepted=search.command(result,command,motion(),.1,active=True,stopped=True)
                self.assertTrue(any(accepted));self.assertIsNone(search.scan.began)


if __name__=='__main__':unittest.main()
