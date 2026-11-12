"""Pending yield survives flow loss with the original v102 control ownership."""
from tests.support.crosswalk import configured, motion, observation
from robot.tasks.crosswalk import CrosswalkTask
import unittest


class PendingYieldTests(unittest.TestCase):
    def task(self):
        return CrosswalkTask(configured(body_front_m=.02,body_rear_m=.16,body_half_width_m=.085,
             stop_clearance_m=.02),finish_priority=True)

    def begin(self):
        t=self.task()
        for now in (0.,.15,.3):
            t.update(observation(now,.47),motion(.08),now,True,False,False,True)
        self.assertEqual(t.phase,'APPROACH')
        return t

    def test_flow_gap_retains_stop_until_nominal_brake_entry_then_one_bounded_dwell(self):
        t=self.begin()
        state=t.update(None,motion(valid=False),.45,True,False,False,True)
        self.assertEqual(t.phase,'RECOVERY_APPROACH')
        self.assertTrue(state['yield_pending'])
        self.assertFalse(t.allows_speed_profile())
        self.assertTrue(t.allows_road_search())
        # Current flow resumes but the bars remain occluded, as in the run.
        now=.45
        while t.track['near_m']-.02 > .15:
            now+=.15
            state=t.update(None,motion(.1,.015),now,True,False,False,True)
            self.assertNotEqual(t.phase,'RECOVERY_PASSING')
        self.assertEqual(t.phase,'STOPPING')
        self.assertTrue(t.planned_stop)
        self.assertEqual(t.limit_command((100,120),state,motion()),(0,0))
        first=now+.15
        for i in range(30):
            now=first+i*.15
            state=t.update(None,motion(valid=False),now,True,True,True,True)
            if now-first < t.policy.hold_s:
                self.assertEqual(t.limit_command((100,120),state,motion()),(0,0))
            if t.phase=='RECOVERY_PASSING':break
        self.assertEqual(t.phase,'RECOVERY_PASSING')
        self.assertGreaterEqual(state['hold_elapsed_s'],3.2)
        self.assertFalse(state['yield_pending'])
        self.assertFalse(state['yield_verified'])
        self.assertTrue(any(t.limit_command((100,120),state,motion())))

    def test_fresh_geometry_recovers_approach_without_dropping_yield(self):
        t=self.begin()
        t.update(None,motion(valid=False),.45,True,False,False,True)
        state=t.update(observation(.6,.46),motion(),.6,True,False,False,True)
        self.assertEqual(t.phase,'APPROACH')
        self.assertTrue(state['yield_pending'])
        self.assertFalse(t.allows_speed_profile())

    def test_unconfirmed_single_candidate_never_owns_dwell(self):
        t=self.task()
        t.update(observation(0.,.12),motion(valid=False),0.,True,False,False,True)
        for now in (.15,.3,.6):
            state=t.update(None,motion(valid=False),now,True,False,False,True)
            self.assertNotIn(t.phase,('STOPPING','HOLD','RECOVERY_HOLD'))
        self.assertIsNone(t.track)
