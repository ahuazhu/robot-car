"""Arming uses current support without erasing the road used in occlusions."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
from geometry import fit_curve
from road_model import LocalRoadModel
from tracker import TargetTracker
from controller import assess_vision_command

CAL=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())


def observation(start=.12, end=.65, y=0., quality=.93):
    path=fit_curve(np.column_stack((np.linspace(start,end,60),np.full(60,y))))['path']
    return dict(path=path,confidence=quality,mode='TWO EDGES')


class CurrentSupportTests(unittest.TestCase):
    def test_old_prefix_does_not_veto_trusted_current_component_or_get_refreshed(self):
        model=LocalRoadModel()
        model.update(observation(),0.)
        self.assertTrue(model.update(observation(start=.18),.4)['observed'])
        history,old_info=model.support(.4)
        seen=model.seen.copy();geometry=model.path.copy()
        current,info=model.support(.4,current_only=True)
        self.assertLess(old_info['confidence'],model.policy.confirmation_quality)
        self.assertIsNotNone(current)
        self.assertGreaterEqual(info['confidence'],model.policy.confirmation_quality)
        self.assertEqual(info['fresh_fraction'],1.)
        self.assertGreater(current[0,0],history[0,0])
        np.testing.assert_array_equal(model.seen,seen)
        np.testing.assert_array_equal(model.path,geometry)
        np.testing.assert_array_equal(model.support(.4)[0],history)

    def test_history_still_supports_driving_but_cannot_arm(self):
        model=LocalRoadModel();model.update(observation(),0.)
        self.assertIsNotNone(model.support(.1)[0])
        path,info=model.support(.1,current_only=True)
        self.assertIsNone(path)
        self.assertEqual(info['reason'],'NO CURRENT ROAD SUPPORT')
        self.assertEqual(model.last_seen,0.)

    def test_short_or_weak_current_fragment_cannot_arm(self):
        model=LocalRoadModel();model.update(observation(),0.)
        model.seen[:]=0.;model.seen[:3]=.1
        self.assertIsNone(model.support(.1,current_only=True)[0])
        model.seen[:]=.1;model.quality[:]=np.nextafter(model.policy.measurement_quality,0.)
        self.assertIsNone(model.support(.1,current_only=True)[0])

    def test_tracker_arms_from_five_fresh_components_despite_old_prefix(self):
        tracker=TargetTracker(CAL)
        tracker.pixels.measure=lambda *args:(observation(),{})
        tracker.update(None,0.,require_current=True)
        tracker.pixels.measure=lambda *args:(observation(start=.18),{})
        consecutive=0
        for now in (.1,.2,.3,.4,.5):
            result,_=tracker.update(None,now,require_current=True)
            self.assertTrue(result['valid'],result)
            self.assertTrue(result['observed'])
            self.assertEqual(result['state'],'OBSERVED')
            self.assertFalse(result['reference_inferred'])
            self.assertEqual(result['road_model']['support_scope'],'CURRENT')
            _,consecutive=assess_vision_command(result,(40,40),100,280,consecutive,
                                               tracker.policy.confirmation_quality)
        self.assertEqual(consecutive,tracker.policy.startup_observations)
        self.assertTrue(np.any(tracker.model.seen<.01))
        self.assertLess(tracker.model.support(.5)[1]['confidence'],tracker.policy.confirmation_quality)
        tracker.pixels.measure=lambda *args:(None,{})
        result,_=tracker.update(None,.51)
        self.assertTrue(result['valid'])
        self.assertEqual(result['road_model']['support_scope'],'HISTORY_ALLOWED')
        self.assertTrue(result['reference_inferred'])
        result,_=tracker.update(None,.52,require_current=True)
        self.assertFalse(result['valid'])
        self.assertFalse(result['observed'])

    def test_stable_accepted_quality_is_confirmed_by_five_current_observations(self):
        # Actual stopped-car diagnostic: accepted q=.6483, but the old .65
        # current-support gate erased the target on every frame forever.
        tracker=TargetTracker(CAL)
        tracker.pixels.measure=lambda *args:(observation(quality=.6482982100829959),{})
        streak=0
        for i in range(5):
            result,_=tracker.update(None,i*.1,require_current=True)
            self.assertTrue(result['valid'])
            self.assertTrue(result['observed'])
            self.assertAlmostEqual(result['confidence'],.6482982100829959)
            self.assertEqual(result['road_model']['fresh_fraction'],1.)
            command,streak=assess_vision_command(result,(40,40),100,280,streak,tracker.policy.measurement_quality)
            self.assertEqual(command,(40,40))
            self.assertEqual(streak,i+1)
            self.assertEqual(streak>=tracker.policy.startup_observations,i==4)
        # Current-only startup must still reject stale geometry after one miss.
        tracker.pixels.measure=lambda *args:(None,{})
        result,_=tracker.update(None,.5,require_current=True)
        self.assertFalse(result['valid'])
        self.assertEqual(assess_vision_command(result,None,100,280,streak,tracker.policy.measurement_quality)[1],0)

    def test_below_measurement_quality_never_accumulates_startup_evidence(self):
        tracker=TargetTracker(CAL);tracker.pixels.measure=lambda *args:(observation(quality=.54),{})
        streak=0
        for i in range(10):
            result,_=tracker.update(None,i*.1,require_current=True)
            self.assertFalse(result['valid'])
            _,streak=assess_vision_command(result,None,100,280,streak,tracker.policy.measurement_quality)
            self.assertEqual(streak,0)

    def test_conflicting_measurement_cannot_supply_current_startup_support(self):
        tracker=TargetTracker(CAL)
        tracker.pixels.measure=lambda *args:(observation(),{})
        tracker.update(None,0.,require_current=True)
        tracker.pixels.measure=lambda *args:(observation(y=.25),{})
        result,_=tracker.update(None,.1,require_current=True)
        self.assertEqual(result['state'],'CONFIRMING')
        self.assertEqual(result['measurement_reject'],'ROAD CONFLICT')
        self.assertFalse(result['observed'])
        self.assertFalse(result['valid'])
        command,streak=assess_vision_command(result,None,100,280,4,.65)
        self.assertIsNone(command)
        self.assertEqual(streak,0)


if __name__=='__main__':unittest.main()
