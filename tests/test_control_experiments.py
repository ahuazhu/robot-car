"""Parked, acknowledged experiment changes must never race a motor start."""

import io
import json
import unittest
from unittest.mock import patch

from tests.support import ROOT
from robot.control.experiments import baseline_experiments
from robot.interface.preview_transport import LiveStream, WebControl


class ExperimentControlTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch('robot.interface.preview_transport.time.monotonic', return_value=10.0)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        self.control = WebControl()
        self.publish()

    def publish(self, *, active=False, armed=False, pwm=(0, 0)):
        self.control.update(active, True, 'WAIT START', pwm,
                            readiness={'start_light': {'armed': armed}})

    def flags(self, gap=True, recovery=False):
        return dict(motion_gap_hold=gap, crosswalk_recovery_dwpp=recovery)

    def submit(self, flags=None):
        return self.control.submit_experiments(
            self.flags() if flags is None else flags, self.control.generation)

    def apply(self, **state):
        return self.control.apply_experiments(**dict(active=False, armed=False, **state))

    def test_all_combinations_require_frame_ack_and_can_return_to_baseline(self):
        previous = baseline_experiments()
        for revision, flags in enumerate((self.flags(), self.flags(False, True),
                                          self.flags(True, True), self.flags(False, False)), 1):
            self.assertTrue(self.submit(flags)[0])
            pending = self.control.status()['experiments']
            self.assertTrue(pending['pending'])
            self.assertEqual({k: pending[k] for k in flags}, previous)
            self.assertFalse(self.control.submit('start_now', self.control.generation)[0])
            self.assertEqual(self.apply(), flags)
            state = self.control.status()['experiments']
            self.assertFalse(state['pending'])
            self.assertEqual(state['revision'], revision)
            self.assertIsNone(state['error'])
            previous = flags
        self.assertEqual(WebControl().status()['experiments']['motion_gap_hold'], False)
        self.assertEqual(WebControl().status()['experiments']['crosswalk_recovery_dwpp'], False)

    def test_submission_invalidates_other_tabs_and_rejects_duplicate(self):
        old_generation = self.control.generation
        self.assertTrue(self.submit()[0])
        self.assertFalse(self.control.submit_experiments(self.flags(False, True), old_generation)[0])
        self.assertFalse(self.submit()[0])
        self.apply()
        self.assertFalse(self.control.submit('start_now', old_generation)[0])
        self.assertTrue(self.control.submit('start_now', self.control.generation)[0])

    def test_active_armed_nonzero_and_stale_status_block_changes(self):
        for state in ({'active': True}, {'armed': True}, {'pwm': (1, 0)}, {'pwm': (0, -1)}):
            self.publish(**state)
            self.assertFalse(self.submit()[0], state)
        self.publish()
        self.now.return_value = 11.0
        self.assertFalse(self.submit()[0])

    def test_queued_or_in_progress_start_blocks_changes(self):
        self.assertTrue(self.control.submit('start_now', self.control.generation)[0])
        self.assertFalse(self.submit()[0])
        self.assertEqual(self.control.take().action, 'start_now')
        self.assertFalse(self.submit()[0])

    def test_recheck_actual_mission_state_at_apply_time(self):
        for state in ({'active': True, 'armed': False}, {'active': False, 'armed': True}):
            self.assertTrue(self.submit()[0])
            self.assertEqual(self.control.apply_experiments(**state), baseline_experiments())
            self.assertIsNotNone(self.control.status()['experiments']['error'])
            self.assertFalse(self.control.status()['experiments']['pending'])

    def test_expired_request_cannot_change_later_run(self):
        self.submit()
        self.now.return_value = 11.0
        self.assertEqual(self.apply(), baseline_experiments())
        self.assertIn('过期', self.control.status()['experiments']['error'])

    def test_web_and_touch_stop_cancel_pending_but_keep_applied_flags(self):
        self.submit()
        self.apply()
        for action in ('web', 'touch', 'session'):
            self.publish()
            self.assertTrue(self.submit(self.flags(False, True))[0])
            if action == 'web':
                self.control.submit('stop')
                self.assertEqual(self.control.take().action, 'stop')
            elif action == 'touch':
                request, touch = self.control.take_actions('stop')
                self.assertEqual((request, touch), (None, 'stop'))
            else:
                self.control.stopped()
            self.assertEqual(self.apply(), self.flags())
            self.assertFalse(self.control.status()['experiments']['pending'])

    def test_touch_start_waits_for_pending_configuration(self):
        self.submit()
        self.assertEqual(self.control.take_actions('start_now'), (None, None))
        self.assertEqual(self.apply(), self.flags())
        self.assertEqual(self.control.take_actions('start_now'), (None, 'start_now'))

    def test_lamp_save_and_experiments_are_serialized(self):
        class Signal:
            configuration_pending = True
        self.control.lamp_signal = Signal()
        self.assertFalse(self.submit()[0])
        self.control.lamp_signal.configuration_pending = False
        self.assertTrue(self.submit()[0])
        self.assertFalse(self.control.submit_lamp_configuration({}, self.control.generation)[0])

    def test_unavailable_motor_does_not_block_parked_configuration(self):
        self.control.update(False, False, 'UNO NOT CONNECTED', (0, 0), readiness={
            'blocked_by': ['UNO_DISCONNECTED'], 'start_light': {'armed': False}})
        self.assertTrue(self.submit()[0])
        self.assertEqual(self.apply(), self.flags())
        self.assertFalse(self.control.submit('start_now', self.control.generation)[0])

    def test_invalid_flags_cannot_partially_change_state(self):
        for flags in (None, {}, [], {'motion_gap_hold': True},
                      dict(self.flags(), unexpected=True), self.flags(1), self.flags('false')):
            with self.assertRaises(ValueError):
                self.submit(flags) if flags is not None else self.control.submit_experiments(None, 0)
            self.assertEqual(self.apply(), baseline_experiments())

    def test_returned_snapshots_cannot_mutate_applied_or_queued_settings(self):
        flags = self.flags()
        self.submit(flags)
        flags['motion_gap_hold'] = False
        returned = self.apply()
        returned['motion_gap_hold'] = False
        self.control.status()['experiments']['motion_gap_hold'] = False
        self.assertTrue(self.apply()['motion_gap_hold'])


class ExperimentHTTPTests(unittest.TestCase):
    def test_endpoint_checks_origin_token_shape_and_actual_ack(self):
        class Server:
            def __init__(self, address, handler): self.handler = handler
            def serve_forever(self, **kwargs): pass
            def shutdown(self): pass
            def server_close(self): pass
        with patch('http.server.ThreadingHTTPServer', Server):
            stream = LiveStream(0)
        self.addCleanup(stream.stop)
        stream.control.update(False, True, 'WAIT START', (0, 0))

        def post(body, **headers):
            handler = object.__new__(stream.server.handler)
            handler.path = '/experiments'
            raw = json.dumps(body).encode()
            handler.headers = {'Host': 'robot:8000', 'Origin': 'http://robot:8000',
                               'X-Control-Token': stream.token, 'Content-Length': str(len(raw)), **headers}
            handler.rfile = io.BytesIO(raw)
            handler.connection = type('Connection', (), {'settimeout': lambda *args: None})()
            returned = []
            handler.send_json = lambda code, value: returned.append((code, value))
            handler.do_POST()
            return returned[0]

        flags = dict(motion_gap_hold=True, crosswalk_recovery_dwpp=False)
        body = dict(generation=0, flags=flags)
        self.assertEqual(post(body, **{'X-Control-Token': 'bad'})[0], 403)
        self.assertEqual(post(body, Origin='http://other:8000')[0], 403)
        self.assertEqual(post(dict(generation=0, flags={'motion_gap_hold': True}))[0], 400)
        self.assertEqual(post(dict(generation=0, flags=dict(flags, motion_gap_hold=1)))[0], 400)
        self.assertEqual(post(body)[0], 202)
        self.assertTrue(stream.control.status()['experiments']['pending'])
        self.assertFalse(stream.control.status()['experiments']['motion_gap_hold'])
        self.assertEqual(post(body)[0], 409)
        self.assertEqual(stream.control.apply_experiments(active=False, armed=False), flags)
