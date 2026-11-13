"""Task limits retain ownership when their changed forward pair is remapped."""
import unittest
from unittest.mock import patch

from tests import test_session as session_tests
from robot.control.controller import compensate_pwm


class TaskModelRemapTests(unittest.TestCase):
    def setUp(self):
        self.helper=session_tests.DrivingSessionTests()
        self.helper.setUp()
        self.addCleanup(self.helper.tearDown)
        self.clock=self.helper.clock

    def running(self):
        session=self.helper.session()
        self.helper.start(session)
        session.experiments['bend_inner_reduction']=True
        self.clock.now+=.1
        session.ramp.previous=(50.,50.)
        session.ramp.last=self.clock.now-.1
        return session

    def test_changed_task_pair_is_remapped_before_final_ramp_and_publication(self):
        session=self.running()
        frame=self.helper.frame(session)
        with patch.object(session.crosswalk,'limit_command',return_value=(30.,20.)), \
                patch.object(session.steering,'remap_task_command',return_value=(25.,25.)) as remap, \
                patch.object(session.ramp,'apply',wraps=session.ramp.apply) as ramp:
            session.update(frame)
        remap.assert_called_once_with((30.,20.),self.clock.now)
        ramp.assert_called_once_with((25.,25.),self.clock.now)
        self.assertEqual(session.pwm,(25.,25.))
        self.assertEqual(session.link.calls[-1][1:3],compensate_pwm((25.,25.)))

    def test_unchanged_task_output_and_disabled_switch_keep_existing_pair(self):
        for enabled,limited in ((True,(40.,40.)),(False,(30.,20.))):
            with self.subTest(enabled=enabled,limited=limited):
                session=self.running()
                session.experiments['bend_inner_reduction']=enabled
                with patch.object(session.crosswalk,'limit_command',return_value=limited), \
                        patch.object(session.steering,'remap_task_command') as remap:
                    session.update(self.helper.frame(session))
                remap.assert_not_called()
                self.assertEqual(session.pwm,limited)

    def test_task_owned_deferred_mapping_runs_even_when_task_leaves_pair_unchanged(self):
        session=self.running()
        session.steering.status['speed_profile_control']=dict(
            selected=False,reason='TASK_OWNS_SPEED')
        with patch.object(session.crosswalk,'limit_command',return_value=(40.,40.)), \
                patch.object(session.steering,'remap_task_command',return_value=(35.,40.)) as remap:
            session.update(self.helper.frame(session))
        remap.assert_called_once_with((40.,40.),self.clock.now)
        self.assertEqual(session.pwm,(35.,40.))

    def test_search_and_non_follow_sources_cannot_use_task_remapping(self):
        for provenance in ('search','different_command','no_follow'):
            with self.subTest(provenance=provenance):
                session=self.running()
                frame=self.helper.frame(session)
                if provenance=='different_command':frame.follow_command=(39.,40.)
                elif provenance=='no_follow':frame.follow_command=None
                with patch.object(session.crosswalk,'limit_command',return_value=(30.,20.)), \
                        patch.object(session.search,'is_turning',return_value=provenance=='search'), \
                        patch.object(session.steering,'remap_task_command') as remap:
                    session.update(frame)
                remap.assert_not_called()

    def test_zero_and_reverse_task_commands_are_not_remapped(self):
        for command in ((0.,0.),(-20.,20.),(-20.,-20.),None):
            with self.subTest(command=command):
                session=self.running()
                with patch.object(session.crosswalk,'limit_command',return_value=command), \
                        patch.object(session.steering,'remap_task_command') as remap:
                    session.update(self.helper.frame(session))
                remap.assert_not_called()
                if command is None or command==(0.,0.):self.assertEqual(session.pwm,(0,0))

    def test_fault_and_road_loss_take_priority_over_remapping(self):
        for failure in ('ROAD NOT FOUND','CROSSWALK UNSAFE'):
            with self.subTest(failure=failure):
                session=self.running()
                with patch.object(session.crosswalk,'limit_command',return_value=(30.,20.)), \
                        patch.object(session.recovery,'update',return_value=failure), \
                        patch.object(session.steering,'remap_task_command') as remap:
                    session.update(self.helper.frame(session))
                remap.assert_not_called()
                self.assertEqual(session.pwm,(0,0))
                self.assertEqual(session.reason,failure)

    def test_control_period_not_due_does_not_create_a_new_mapping(self):
        session=self.running()
        session.next_control=self.clock.now+1.
        before=len(session.link.calls)
        with patch.object(session.crosswalk,'limit_command',return_value=(30.,20.)), \
                patch.object(session.steering,'remap_task_command') as remap:
            session.update(self.helper.frame(session))
        remap.assert_not_called()
        self.assertEqual(len(session.link.calls),before)

    def test_explicit_stop_wins_even_with_a_pending_positive_task_output(self):
        session=self.running()
        with patch.object(session.crosswalk,'limit_command',return_value=(30.,20.)), \
                patch.object(session.steering,'remap_task_command') as remap:
            session.update(self.helper.frame(session),('stop',))
        remap.assert_not_called()
        self.assertFalse(session.active)
        self.assertEqual(session.pwm,(0,0))
        self.assertEqual(session.reason,'STOPPED')


if __name__=='__main__':
    unittest.main()
