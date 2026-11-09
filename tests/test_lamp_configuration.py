"""Lamp edits reserve the existing ESP worker and cannot race a race start."""
from tests.support import ROOT
import io
import json
import unittest
import tempfile
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from robot.tasks.start_signal import StartSignal
from robot.interface.preview_transport import WebControl, LiveStream


class LampConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.remote = dict(boot='1234567890abcdef', paired=True, session='', armed=False,
                           confirmed=False, sequence=10, frame_age_ms=10, roi=[0,0,320,240],
                           protocol=2, control_port=81, trigger_direction='DARKER',
                           target_locked=False, target_roi=[0,0,0,0], event_age_ms=0,
                           required_frames=3, required_ms=400, phase='DISARMED',
                           lamp_configuration=dict(manual=False, roi=[100,40,100,80]))
        self.calls = []
        def transport(op, session, boot, *, configuration=None):
            self.calls.append(dict(op=op, session=session, boot=boot, configuration=configuration))
            if op == 'configure':
                self.remote['lamp_configuration'] = dict(
                    manual=configuration['manual'],
                    roi=list(configuration.get('roi', self.remote['lamp_configuration']['roi'])))
            if op == 'arm':
                self.remote.update(armed=True, session=session)
            return dict(self.remote)
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root/'start_signal.local.json').write_text(json.dumps(dict(peer_url='http://10.0.0.2',key='a'*64)))
        self.client = StartSignal({'enabled':True}, app_dir=root,clock=lambda:self.now,transport=transport)
        self.client._step()
        self.calls.clear()
        self.control = WebControl(self.client)
        self.publish()

    def tearDown(self):
        self.temp.cleanup()

    def publish(self, **changes):
        state = dict(active=False, ready=True, reason='WAIT START',pwm=(0,0),
                     readiness=dict(start_light=self.client.snapshot(),direct_start_ready=True,
                                    arm_start_light_ready=True))
        state.update(changes)
        self.control.update(**state)

    def manual(self):
        return dict(manual=True,roi=[139,51,63,61])

    def test_save_queues_without_network_then_worker_applies_exact_configuration(self):
        ok, _, request = self.control.submit_lamp_configuration(self.manual(),0)
        self.assertTrue(ok)
        self.assertEqual(self.calls,[])
        self.assertFalse(self.client.arm())
        self.assertFalse(self.client.available)
        for action in ('start','start_now'):
            self.assertFalse(self.control.submit(action,0)[0])
        queued = self.client.snapshot()['configuration_update']
        self.assertEqual((queued['state'],queued['pending'],queued['request_id']),('QUEUED',True,request))
        self.client._step()
        self.assertEqual([call['op'] for call in self.calls],['configure'])
        self.assertEqual(self.calls[0]['boot'],self.remote['boot'])
        self.assertEqual(self.calls[0]['session'],'')
        self.assertEqual(self.calls[0]['configuration'],self.manual())
        saved = self.client.snapshot()
        self.assertEqual(saved['lamp_configuration'],self.manual())
        self.assertEqual(saved['configuration_update']['state'],'SAVED')
        self.assertFalse(saved['configuration_update']['pending'])
        self.assertFalse(saved['armed']);self.assertFalse(saved['permission']);self.assertFalse(saved['fired'])
        self.assertTrue(self.client.available)

    def test_auto_preserves_last_manual_roi_and_does_not_start(self):
        self.client.queue_configuration(self.manual());self.client._step()
        self.client.queue_configuration({'manual':False});self.client._step()
        self.assertEqual(self.calls[-1]['configuration'],{'manual':False})
        self.assertEqual(self.client.snapshot()['lamp_configuration'],dict(manual=False,roi=[139,51,63,61]))
        self.assertFalse(self.client.permission);self.assertFalse(self.client.armed)

    def test_invalid_or_outside_roi_is_rejected_before_queue(self):
        for roi in ([0,0,15,16],[-1,0,16,16],[305,0,16,16],[0,225,16,16],
                    [False,0,16,16],[0,0,16.0,16],[0,0,16]):
            with self.subTest(roi=roi):
                with self.assertRaises(ValueError):self.client.queue_configuration(dict(manual=True,roi=roi))
        for roi in ([0,0,16,16],[304,224,16,16],[0,0,320,240]):
            self.assertEqual(StartSignal.normalize_configuration(dict(manual=True,roi=roi))['roi'],roi)
        with self.assertRaises(ValueError):
            StartSignal.normalize_configuration(dict(manual=False,roi=[0,0,0,0]))
        self.assertEqual(self.calls,[])

    def test_edit_is_forbidden_while_active_local_remote_armed_or_permission_latched(self):
        self.publish(active=True)
        self.assertFalse(self.control.submit_lamp_configuration(self.manual(),0)[0])
        self.publish()
        for local, remote, permission in ((True,False,False),(False,True,False),(False,False,True)):
            self.client.armed = local;self.client.peer['armed'] = remote;self.client.permission = permission
            with self.subTest(local=local,remote=remote,permission=permission):
                with self.assertRaisesRegex(RuntimeError,'stop_first'):
                    self.client.queue_configuration(self.manual())
                self.assertEqual(self.client.permission,permission)
        self.assertEqual(self.calls,[])

    def test_queued_and_inflight_starts_including_touchscreen_block_configuration(self):
        self.assertTrue(self.control.submit('start_now',0)[0])
        self.assertFalse(self.control.submit_lamp_configuration(self.manual(),0)[0])
        self.assertEqual(self.control.take().action,'start_now')
        self.assertFalse(self.control.submit_lamp_configuration(self.manual(),0)[0])
        self.publish()
        self.control.take(touch_action='start')
        self.assertFalse(self.control.submit_lamp_configuration(self.manual(),0)[0])
        self.assertEqual(self.calls,[])

    def test_session_direct_start_cannot_bypass_a_pending_save(self):
        from tests.test_session import DrivingSessionTests
        case = DrivingSessionTests()
        case.setUp()
        try:
            session = case.session()
            session.start_light = self.client
            case.warm(session)
            self.client.queue_configuration(self.manual())
            frame = case.frame(session)
            ready = session.readiness(frame)
            self.assertFalse(ready['direct_start_ready'])
            self.assertFalse(ready['arm_start_light_ready'])
            self.assertIn('LAMP_CONFIGURATION_PENDING', ready['blocked_by'])
            # This is the actual session action used by touch/direct requests,
            # without passing through HTTP WebControl.submit().
            session.update(frame, ('start_now',))
            self.assertFalse(session.active)
            self.assertFalse(any(call[0] == 1 for call in session.link.calls))
            self.assertTrue(self.client.configuration_pending)
            self.assertFalse(self.client.permission)
        finally:
            case.tearDown()

    def test_worker_rechecks_arm_and_boot_before_sending(self):
        for change, value in (('armed',True),('boot','ffffffffffffffff')):
            with self.subTest(change=change):
                self.client.queue_configuration(self.manual())
                self.client.peer[change] = value
                self.client._step()
                self.assertEqual(self.calls,[])
                self.assertEqual(self.client.snapshot()['configuration_update']['state'],'ERROR')
                self.client.peer = dict(self.remote)

    def test_failed_save_requires_fresh_status_and_does_not_revoke_permission(self):
        original = self.client.transport
        def failed(op,*args,**kwargs):
            if op == 'configure':raise OSError('private endpoint details must not enter UI')
            return original(op,*args,**kwargs)
        self.client.transport = failed
        self.client.queue_configuration(self.manual());self.client._step()
        state = self.client.snapshot()
        self.assertEqual(state['configuration_update']['error'],'configuration_offline')
        self.assertFalse(self.client.available);self.assertFalse(self.client.arm())
        self.assertFalse(self.client.permission)
        self.client._step()
        self.assertTrue(self.client.available)
        self.client.permission = True
        with self.assertRaisesRegex(RuntimeError,'stop_first'):self.client.queue_configuration(self.manual())
        self.assertTrue(self.client.permission)

    def test_arm_cannot_slip_in_while_configuration_http_is_inflight(self):
        original = self.client.transport
        def during(op,*args,**kwargs):
            self.assertTrue(self.client.configuration_pending)
            self.assertFalse(self.client.arm())
            self.assertFalse(self.control.submit('start_now',0)[0])
            return original(op,*args,**kwargs)
        self.client.queue_configuration(self.manual())
        self.client.transport = during
        self.client._step()
        self.assertFalse(self.client.armed);self.assertFalse(self.client.permission)
        self.assertEqual([call['op'] for call in self.calls],['configure'])

    def test_http_error_preserves_safe_device_error_code_without_request_secrets(self):
        self.client.endpoint = 'http://10.0.0.2:81'
        opener = type('Opener',(),{'open':lambda *args,**kwargs: (_ for _ in ()).throw(
            HTTPError('http://10.0.0.2:81/signal',409,'Conflict',{},
                      io.BytesIO(b'{"error":"native_armed_stop_first"}')))})()
        with patch('robot.tasks.start_signal.build_opener',return_value=opener):
            with self.assertRaisesRegex(ValueError,'native_armed_stop_first'):
                self.client._http('configure','',self.remote['boot'],configuration=self.manual())

    def test_candidate_hint_is_separate_from_saved_manual_roi_and_permission(self):
        self.client.peer['lamp_configuration'] = self.manual()
        self.client.peer['candidate_hint'] = dict(valid=True,roi=[200,80,30,40],age_ms=20)
        state = self.client.snapshot()
        self.assertEqual(state['candidate_hint'],dict(valid=True,roi=[200,80,30,40],age_ms=20))
        self.assertEqual(state['lamp_configuration'],self.manual())
        self.assertFalse(state['permission']);self.assertFalse(state['armed'])
        state['candidate_hint']['roi'][0] = 999
        self.assertEqual(self.client.peer['candidate_hint']['roi'],[200,80,30,40])
        self.assertEqual(self.calls,[])

    def test_malformed_or_stale_hints_do_not_change_formal_signal_validity(self):
        hints = [None,dict(valid=False,roi=[0,0,0,0],age_ms=0),
                 dict(valid=1,roi=[0,0,16,16],age_ms=0),
                 dict(valid=True,roi=[0,0,15,16],age_ms=0),
                 dict(valid=True,roi=[305,0,16,16],age_ms=0)]
        hints += [dict(valid=True,roi=[0,0,16,16],age_ms=age)
                  for age in (-1,2501,True,'10',None)]
        for hint in hints:
            with self.subTest(hint=hint):
                self.client.peer['candidate_hint'] = hint
                self.assertIsNone(self.client.snapshot()['candidate_hint'])
                self.assertTrue(self.client._valid(self.client.peer))
                self.assertFalse(self.client.permission)

    def test_cached_candidate_hint_ages_out_even_without_another_esp_frame(self):
        self.client.peer['candidate_hint'] = dict(valid=True,roi=[0,0,16,16],age_ms=1000)
        self.now += 1.
        self.assertEqual(self.client.snapshot()['candidate_hint']['age_ms'],2000)
        self.now += .6
        self.assertIsNone(self.client.snapshot()['candidate_hint'])
        self.now = 100.
        self.client.received = None
        self.assertIsNone(self.client.snapshot()['candidate_hint'])

    def test_lamp_endpoint_reuses_origin_token_validation_and_never_does_network_io(self):
        class Server:
            def __init__(self,address,handler):self.handler=handler
            def serve_forever(self,**kwargs):pass
            def shutdown(self):pass
            def server_close(self):pass
        with patch('http.server.ThreadingHTTPServer',Server):
            stream = LiveStream(0,lamp_signal=self.client)
        self.addCleanup(stream.stop)
        stream.control.update(False,True,'WAIT START',(0,0),readiness={'start_light':self.client.snapshot()})
        def post(body, **headers):
            handler = object.__new__(stream.server.handler)
            handler.path='/lamp-config'
            raw=json.dumps(body).encode()
            handler.headers={'Host':'robot:8000','Origin':'http://robot:8000',
                             'X-Control-Token':stream.token,'Content-Length':str(len(raw)),**headers}
            handler.rfile=io.BytesIO(raw)
            handler.connection=type('Connection',(),{'settimeout':lambda *args:None})()
            returned=[];handler.send_json=lambda code,value:returned.append((code,value))
            handler.do_POST()
            return returned[0]
        body=dict(generation=0,configuration=self.manual())
        self.assertEqual(post(body,**{'X-Control-Token':'bad'})[0],403)
        self.assertEqual(post(body,Origin='http://other:8000')[0],403)
        self.assertEqual(post(dict(generation=0,configuration=dict(manual=True,roi=[0,0,8,8])))[0],400)
        self.assertEqual(post(body)[0],202)
        self.assertEqual(post(body)[0],409)
        self.assertEqual(self.calls,[])
