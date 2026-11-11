"""Retained-route identity, directional alignment and explicit counterevidence."""
from tests.support import ROOT
from tests.test_curve_handoff import path,measurement,scene
from robot.core.road_policy import RoadPolicy
from robot.navigation.road_model import LocalRoadModel
from robot.navigation.road_recovery import RoadSearch
from robot.navigation.route_reference import route_reference
from robot.navigation.steering_control import SteeringControl
from robot.navigation.tracker import TargetTracker
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits
from tests.test_persistent_centerline import recorded_calibration
import json,math,unittest
import numpy as np


class RouteAlignmentTests(unittest.TestCase):
    def search(self):
        return RoadSearch(RoadPolicy(),[0.,0.],[.075,.105,.085],.3,60.,
                          allow_visual_scan=True,finish_priority=True)

    def test_remote_confirmed_road_cannot_replace_followable_prediction(self):
        fixture=json.loads((ROOT/'tests/fixtures/persistent_bend_exit_20261111.json').read_text())
        far=fixture['remote_lateral'];candidate=np.asarray(far['road_path_m'])
        model=LocalRoadModel(persistent_history=True,reference_origin_m=[-.055,0.])
        old=path(.1,.65);model.initialize(old,0.,.9,'ONE LEFT')
        for i in range(8):
            event=model.update(measurement(candidate,.95),i*.2,stationary=bool(i%2))
            self.assertEqual(event['reject'],'CANDIDATE OUTSIDE CONTROL RANGE')
            self.assertFalse(event['observed'])
            np.testing.assert_array_equal(model.path,old)

    def test_perpendicular_route_has_heading_even_without_forward_pursuit(self):
        points=np.c_[np.full(30,.2),np.linspace(-.3,.4,30)]
        reference=route_reference(points,[0.,0.],.65)
        self.assertAlmostEqual(reference['heading_rad'],math.pi/2)
        self.assertAlmostEqual(reference['distance_m'],.2)
        search=self.search();result=dict(valid=False,route_alignment=reference)
        command=search.command(result,None,scene(),0.,active=True)
        self.assertEqual(command,(0,0));self.assertFalse(search.is_turning())
        self.assertEqual(search.last_hint,1.)
        self.assertEqual(search.snapshot()['phase'],'SETTLING')

    def test_constrained_follow_wins_over_any_retained_heading(self):
        search=self.search()
        for i,heading in enumerate((90.,45.,25.)):
            r=dict(valid=True,observed=True,state='OBSERVED',observation_confidence=.9,
                   speed_limit_source='CENTERLINE_CONFIDENCE',control_confidence=.01,
                   route_alignment=dict(heading_rad=math.radians(heading),distance_m=.1))
            self.assertEqual(search.command(r,(2,5),scene(),i*.2,active=True),(2,5))
        r['route_alignment']['heading_rad']=math.radians(10.)
        self.assertEqual(search.command(r,(40,40),scene(),.8,active=True),(40,40))
        self.assertFalse(search.is_turning())

    def test_right_heading_and_far_reference_do_not_authorize_wrong_turn(self):
        search=self.search()
        r=dict(valid=False,route_alignment=dict(heading_rad=-math.pi/2,distance_m=.1))
        self.assertEqual(search.command(r,None,scene(),0.,active=True),(0,0))
        self.assertEqual(search.last_hint,-1.)
        r['route_alignment']['distance_m']=.9
        search.reset();search.command(r,None,scene(),1.,active=True,stopped=True)
        self.assertNotEqual(search.snapshot()['phase'],'ALIGNING')

    def test_task_stop_and_idle_prevent_alignment_rotation(self):
        r=dict(valid=False,route_alignment=dict(heading_rad=math.pi/2,distance_m=.1))
        for active,allowed in ((False,True),(True,False)):
            search=self.search();command=search.command(r,None,scene(),0.,active=active,allowed=allowed)
            self.assertFalse(command is not None and any(command))
            self.assertFalse(search.is_turning())

    def test_search_preserves_the_planned_envelope_without_a_second_fixed_cap(self):
        result=dict(valid=True,observed=False,state='PREDICTED',observation_confidence=.2,
                    control_confidence=.6,speed_limit_source='CENTERLINE_CONFIDENCE')
        search=self.search()
        self.assertEqual(search.command(result,(90,100),scene(),0.,active=True),(90,100))
        self.assertEqual(search.snapshot()['reason'],'CENTERLINE CONFIDENCE')

    def test_recorded_counterevidence_reduces_unified_speed_without_erasing_route(self):
        fixture=json.loads((ROOT/'tests/fixtures/persistent_bend_exit_20261111.json').read_text())
        result=fixture['contradicted'];origin=result['control_reference_origin_m']
        candidate=np.asarray(result['candidate_path_m'])
        original=np.asarray(result['control_path_m'])+origin
        model=LocalRoadModel(persistent_history=True,reference_origin_m=origin)
        model.initialize(original,0.,fixture['before_exit']['confidence'],'ONE LEFT')
        model.variance[:]=result['road_model']['sigma_m']**2
        baseline=model.support(.1)[1]['confidence']
        model.update(measurement(candidate,.95),.1,stationary=False)
        self.assertLess(model.observation_agreement,1.)
        report=result['surface_rejection']
        surface=type('Surface',(),{'contradiction':lambda *args:report})()
        model.reject_surface(surface)
        _,support=model.support(.1)
        np.testing.assert_array_equal(model.path,original)
        self.assertLess(support['confidence'],baseline)
        ceilings=[];commands=[]
        for confidence in (baseline,support['confidence']):
            speed=ProfileSpeedControl(SpeedLimits(.3,.15,.4,.2,.35,.15),163,.4)
            target=dict(result,confidence=confidence,path_confidence=[confidence]*len(original),
                        road_model=dict(support,confidence=confidence))
            control=SteeringControl(RoadPolicy(),origin,[.075,.105,.085],.3,
                                    cruise=135,speed_control=speed)
            control.observe(scene(),1.,False)
            command=control.command(target,.1)
            self.assertIsNotNone(command);self.assertGreater(sum(command),0)
            ceilings.append(speed.status['confidence_ceiling_m_s']);commands.append(command)
            self.assertGreater(control.status['prediction']['lease_s'],0.)
        self.assertLess(ceilings[1],ceilings[0])
        self.assertLess(max(commands[1]),max(commands[0]))

    def test_contradiction_stays_until_current_road_observation_resolves_it(self):
        tracker=TargetTracker(recorded_calibration())
        tracker.motion.update=lambda *args:scene()
        original=path(.1,.65);tracker.model.initialize(original,0.,.9,'ONE LEFT')
        tracker.model.surface_contradicted=True
        tracker.model.surface_support=.2
        surface=type('Surface',(),{'contradiction':lambda *args:
            dict(contradicted=False,visible_samples=0,green_samples=0,longest_green_m=0.)})()
        tracker.pixels.measure=lambda *args:(None,dict(surface=surface))
        result,_=tracker.update(None,1.)
        self.assertTrue(result['road_model']['surface_contradicted'])
        self.assertEqual(result['road_model']['surface_support'],.2)
        tracker.pixels.measure=lambda *args:(measurement(original),dict(surface=surface))
        result,_=tracker.update(None,1.1)
        self.assertTrue(result['observed'])
        self.assertFalse(result['road_model']['surface_contradicted'])
        self.assertEqual(result['road_model']['surface_support'],1.)

    def test_recorded_turn_uses_continuous_confidence_instead_of_a_fixed_recovery_gear(self):
        fixture=json.loads((ROOT/'tests/fixtures/persistent_bend_exit_20261111.json').read_text())
        target=fixture['before_exit'];outputs=[]
        ceilings=[]
        for confidence in (.9,.6,.3,.1):
            result=dict(target,confidence=confidence,
                        path_confidence=[confidence]*len(target['control_path_m']))
            speed=ProfileSpeedControl(SpeedLimits(.3,.15,.4,.2,.35,.15),163,.4)
            control=SteeringControl(RoadPolicy(),target['control_reference_origin_m'],
                                    [.075,.105,.085],.3,cruise=135,speed_control=speed)
            control.observe(scene(),1.,False)
            outputs.append(control.command(result,.1))
            ceilings.append(speed.status['confidence_ceiling_m_s'])
        self.assertEqual(ceilings,sorted(ceilings,reverse=True))
        self.assertEqual([max(c) for c in outputs],sorted([max(c) for c in outputs],reverse=True))
        self.assertTrue(all(sum(c)>0 for c in outputs))

    def test_innovation_confidence_survives_missing_frame_and_recovers_on_accepted_line(self):
        model=LocalRoadModel(persistent_history=True)
        old=path(.1,.65);new=path(.1,.65,.25)
        model.initialize(old,0.,.8,'ONE LEFT')
        model.update(measurement(new,.95),.1,stationary=False)
        agreement=model.observation_agreement
        self.assertLess(agreement,1.)
        model.update(None,1.,stationary=False)
        self.assertEqual(model.observation_agreement,agreement)
        for i in range(4):model.update(measurement(new,.95),1.1+i*.1,stationary=False)
        self.assertLessEqual(model.observation_agreement,agreement+1e-9)
        np.testing.assert_array_equal(model.path,new)
        # Geometry adoption retains the disagreement. Independent consistent
        # observations restore trust, instead of one reset making it perfect.
        for i in range(8):model.update(measurement(new,.95),1.5+i*.1,stationary=False)
        self.assertGreater(model.observation_agreement,.9)
        self.assertLess(model.observation_agreement,1.)


if __name__=='__main__':unittest.main()
