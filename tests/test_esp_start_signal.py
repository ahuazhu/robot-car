"""Run the actual embedded C++ core and the Maix network/session boundary."""
import json
import cv2
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from esp32s3.video_preview.replay import NativeDetector
from tests.support import ROOT
from robot.tasks.start_signal import StartSignal


class EmbeddedSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.detector=NativeDetector()
    @classmethod
    def tearDownClass(cls):cls.detector.close()
    def setUp(self):self.detector.cancel()
    def scene(self,value=40):return np.full((240,320,3),value,np.uint8)
    def lamp(self,frame,value=190,x=144,y=48,w=32,h=32):
        out=frame.copy();out[y:y+h,x:x+w]=value;return out
    def lock(self,before):
        self.detector.arm()
        states=[self.detector.process(before,t) for t in (0,100,200,300)]
        self.assertEqual([s['locked'] for s in states],[0,0,0,1],states)
        return states[-1]
    def off(self,after):return [self.detector.process(after,t) for t in (400,610,820,1030)]
    def test_requires_manual_arm_stable_lamp_then_dark_only(self):
        background=self.scene();on=self.lamp(background)
        self.assertFalse(self.detector.process(on,0)['armed'])
        self.lock(on)
        states=self.off(background)
        self.assertEqual([s['confirmed'] for s in states],[0,0,1,1])
        self.lock(on)
        for s in self.off(self.lamp(background,240)):self.assertFalse(s['confirmed'])
    def test_no_target_and_unframed_target_never_launch(self):
        for im in (self.scene(),self.scene(240),self.lamp(self.scene(),x=68,y=15)):
            self.detector.arm()
            for t in range(0,1600,100):self.assertFalse(self.detector.process(im,t)['confirmed'])
            self.assertFalse(self.detector.state()['locked'])
    def test_outer_box_centre_unchanged_and_new_area_can_locate(self):
        x0,y0,width,height=self.detector.roi
        self.assertEqual((x0+width/2,y0+height/2),(160,64))
        self.assertEqual((width,height),(160,128))
        for x,y in ((86,8),(210,100)):
            before=self.lamp(self.scene(),x=x,y=y,w=24,h=24)
            locked=self.lock(before)
            self.assertGreaterEqual(locked['target_x'],80)
            self.assertLessEqual(locked['target_x']+locked['target_w'],240)
            self.assertTrue(self.off(self.scene())[-1]['confirmed'])
    def test_original_small_lamp_remains_detectable_with_expanded_background(self):
        folder=ROOT/'tests/fixtures/esp_start_signal_history'
        self.detector.arm()
        for i in range(4):
            state=self.detector.process(cv2.imread(str(folder/('small-on-%d.png'%i))),i*100)
        self.assertTrue(state['locked'])
        self.assertGreaterEqual(state['target_x'],128)
        for i in range(4):
            state=self.detector.process(cv2.imread(str(folder/('small-off-%d.png'%i))),400+i*210)
        self.assertTrue(state['confirmed'])

    def test_new_outer_bright_object_cannot_displace_valid_central_lamp(self):
        image=self.lamp(self.scene(),250,x=82,y=36,w=40,h=48)
        image=self.lamp(image,190,x=152,y=56,w=16,h=16)
        state=self.lock(image)
        self.assertGreaterEqual(state['target_x'],150)
        self.assertLessEqual(state['target_x']+state['target_w'],170)

    def test_outside_search_window_cannot_affect_target(self):
        before=self.lamp(self.scene());self.lock(before)
        after=self.scene(250);after[0:128,80:240]=before[0:128,80:240]
        for s in self.off(after):self.assertFalse(s['confirmed'])
    def test_new_bright_object_inside_search_window_never_retargets(self):
        before=self.lamp(self.scene());locked=self.lock(before)
        after=before.copy();after[30:44,116:138]=250
        for s in self.off(after):
            self.assertFalse(s['confirmed']);self.assertEqual(s['target_x'],locked['target_x'])
    def test_uniform_exposure_change_and_flash_are_rejected(self):
        before=self.lamp(self.scene(60));self.lock(before)
        after=np.maximum(before.astype(int)-20,0).astype(np.uint8)
        for s in self.off(after):self.assertFalse(s['confirmed'])
        self.lock(before)
        for t in (400,500,600,700):self.assertFalse(self.detector.process(self.lamp(self.scene(60),250),t)['confirmed'])
        self.assertFalse(self.detector.process(before,800)['confirmed'])
    def test_large_exposure_gain_drop_is_uncertain_not_a_start(self):
        before=self.lamp(self.scene(60));self.lock(before)
        after=(before.astype(float)*.65).astype(np.uint8)
        for s in self.off(after):self.assertFalse(s['confirmed'])
    def test_faint_change_with_residual_light_can_confirm(self):
        background=self.scene(150);before=self.lamp(background,190);after=self.lamp(background,158)
        self.lock(before);self.assertTrue(self.off(after)[-1]['confirmed'])
    def test_background_structure_change_is_uncertain(self):
        before=self.lamp(self.scene());before[38:44,134:186]=10
        self.lock(before)
        after=self.scene()
        for x in range(128,192,16):after[30:100,x:x+8]=200
        for s in self.off(after):self.assertFalse(s['confirmed'])
    def test_target_jitter_during_acquisition_does_not_lock(self):
        self.detector.arm()
        for i,x in enumerate((120,150,170,120,170,150)):
            s=self.detector.process(self.lamp(self.scene(),x=x,w=20,h=24),i*150)
            self.assertFalse(s['locked']);self.assertFalse(s['confirmed'])
    def test_gap_and_cancellation_require_new_bright_baseline(self):
        on=self.lamp(self.scene());self.lock(on)
        for t in (400,510,620,1800):self.assertFalse(self.detector.process(self.scene(),t)['confirmed'])
        self.assertFalse(self.detector.state()['locked'])
        self.lock(on);self.assertTrue(self.off(self.scene())[-1]['confirmed'])
        self.detector.cancel();self.assertFalse(self.detector.process(on,1200)['confirmed'])
    def test_both_frame_count_and_confirmation_time_are_required(self):
        self.lock(self.lamp(self.scene()))
        for t in (400,850):self.assertFalse(self.detector.process(self.scene(),t)['confirmed'])
        self.assertTrue(self.detector.process(self.scene(),900)['confirmed'])
        self.lock(self.lamp(self.scene()))
        for t in (400,500,600):self.assertFalse(self.detector.process(self.scene(),t)['confirmed'])
        self.assertTrue(self.detector.process(self.scene(),800)['confirmed'])

    def test_timer_rollover_is_safe(self):
        self.detector.arm();on=self.lamp(self.scene())
        for offset in (0,100,200,300):self.detector.process(on,(0xffffff00+offset)&0xffffffff)
        for i,offset in enumerate((400,610,820,1030)):
            self.assertEqual(self.detector.process(self.scene(),(0xffffff00+offset)&0xffffffff)['confirmed'],int(i>=2))
    def test_white_and_different_coloured_emitters_can_start(self):
        background=self.scene(20)
        for colour in ((220,220,220),(20,20,240),(20,240,20),(190,20,20),(20,180,240),(240,240,20)):
            with self.subTest(bgr=colour):
                on=self.lamp(background,colour)
                self.lock(on);self.assertTrue(self.off(background)[-1]['confirmed'])
    def test_colour_separates_face_from_a_connected_bright_beam(self):
        background=self.scene(20);background[60:64,96:224]=220
        on=self.lamp(background,(20,20,240),x=144,y=42,w=32,h=40)
        target=self.lock(on)
        self.assertGreaterEqual(target['target_x'],140)
        self.assertLessEqual(target['target_x']+target['target_w'],180)
        self.assertTrue(self.off(background)[-1]['confirmed'])
    def test_component_holes_are_excluded_and_mask_survives_extinction(self):
        background=self.scene();on=self.lamp(background,220,x=138,y=42,w=44,h=44)
        on[56:72,152:168]=40
        self.lock(on);mask=self.detector.mask().copy()
        self.assertEqual(mask[28:36,36:44].sum(),0)
        self.assertGreater(mask.sum(),100)
        self.assertTrue(self.off(background)[-1]['confirmed'])
        np.testing.assert_array_equal(mask,self.detector.mask())
        self.detector.cancel();self.assertEqual(self.detector.mask().sum(),0)
    def test_change_inside_bbox_but_outside_lamp_pixels_is_ignored(self):
        on=self.lamp(self.scene(),220,x=138,y=42,w=44,h=44);on[56:72,152:168]=160
        self.lock(on);after=on.copy();after[56:72,152:168]=10
        for state in self.off(after):self.assertFalse(state['confirmed'])
    def test_background_brightening_with_unchanged_lamp_never_fires(self):
        self.lock(self.lamp(self.scene(),220))
        for state in self.off(self.lamp(self.scene(100),220)):self.assertFalse(state['confirmed'])
    def test_exposure_gain_and_offset_with_observable_background_are_rejected(self):
        background=np.repeat(np.repeat(np.linspace(40,140,320,dtype=np.uint8)[None,:,None],240,0),3,2)
        on=self.lamp(background,220);self.lock(on)
        after=np.clip(on.astype(float)*.65+20,0,255).astype(np.uint8)
        for state in self.off(after):self.assertFalse(state['confirmed'])
    def test_dark_gantry_background_does_not_prevent_real_extinction(self):
        background=self.scene(0);self.lock(self.lamp(background,220))
        self.assertTrue(self.off(background)[-1]['confirmed'])

    def test_recorded_negative_people_sequence_does_not_fire(self):
        self.detector.arm()
        for i,path in enumerate(sorted((ROOT/'tests/fixtures/start_signal_jitter_20261031').glob('*.png'))):
            image=cv2.imread(str(path));self.assertFalse(self.detector.process(image,i*210)['confirmed'])


class SignalPeerTests(unittest.TestCase):
    def setUp(self):
        self.now=100.
        self.base=dict(boot='1234567890abcdef',paired=True,session='',armed=False,confirmed=False,sequence=10,frame_age_ms=10,roi=[96,8,128,112],protocol=2,control_port=81,trigger_direction='DARKER',target_locked=False,target_roi=[96,8,0,0],event_age_ms=0,required_frames=4,required_ms=600,phase='DISARMED')
        self.remote=dict(self.base);self.calls=[]
        def transport(op,session,boot):
            self.calls.append((op,session,boot))
            if op=='arm':self.remote.update(session=session,armed=True,phase='WAIT_CHANGE',confirmed=False,target_locked=False,target_roi=[96,8,0,0],confirmation_frames=0,confirmed_ms=0,event_age_ms=0)
            if op=='cancel':self.remote.update(session='',armed=False,confirmed=False)
            return dict(self.remote)
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        (root/'start_signal.local.json').write_text(json.dumps(dict(peer_url='http://10.0.0.2',key='a'*64)))
        self.client=StartSignal({'enabled':True},app_dir=root,clock=lambda:self.now,transport=transport)
        self.client._step()
    def tearDown(self):self.temp.cleanup()
    def arm(self):
        self.assertTrue(self.client.arm());self.client._step()
    def confirm(self):
        self.remote.update(confirmed=True,sequence=max(20,self.remote['sequence']+1),phase='SIGNAL_READY',confirmation_frames=4,confirmed_ms=630,target_locked=True,target_roi=[144,48,32,32],event_age_ms=630);self.client._step()
    def test_fresh_matching_event_only_and_started_consumes_once(self):
        self.arm();self.assertFalse(self.client.update(self.now));self.confirm()
        self.assertFalse(self.client.update(self.now,timely=False));self.assertTrue(self.client.update(self.now))
        self.client.started();self.assertFalse(self.client.update(self.now));self.assertTrue(self.client.fired)
    def test_stop_cancels_late_inflight_response(self):
        self.arm()
        def delayed(*args):
            peer=dict(self.remote,confirmed=True);self.client.cancel();return peer
        self.client.transport=delayed;self.client._step()
        self.assertFalse(self.client.armed);self.assertFalse(self.client.update(self.now))
    def test_reboot_disconnect_stale_and_session_mismatch_require_manual_rearm(self):
        for corruption in ({'boot':'ffffffffffffffff'},{'session':'f'*32},{'sequence':1}):
            self.temp.cleanup();self.setUp();self.arm();self.remote.update(corruption);self.client._step()
            self.assertFalse(self.client.armed);self.assertFalse(self.client.update(self.now))
            self.temp.cleanup()
        self.temp.cleanup();self.setUp();self.arm();self.now+=.76
        self.assertFalse(self.client.update(self.now));self.assertTrue(self.client.armed);self.assertFalse(self.client.permission)
    def test_network_loss_after_confirmation_does_not_revoke_permission(self):
        self.arm();self.confirm();self.assertTrue(self.client.permission)
        self.client.transport=lambda *args: (_ for _ in ()).throw(OSError())
        self.now+=60;self.client._step()
        self.assertTrue(self.client.update(self.now));self.assertTrue(self.client.consume())
        self.client.started();self.assertTrue(self.client.permission)
        self.client._step();self.assertEqual(self.client.phase,'RUNNING')

    def test_repeated_arm_keeps_session_and_no_pairing_never_arms(self):
        self.arm();nonce=self.client.session;self.client.arm();self.assertEqual(self.client.session,nonce)
        client=StartSignal({'enabled':True},clock=lambda:self.now)
        self.assertFalse(client.arm());self.assertFalse(client.update(self.now))
    def test_idle_cannot_accept_remote_armed_or_confirmed_state(self):
        self.remote.update(armed=True,confirmed=True,session='f'*32);self.client._step()
        self.assertFalse(self.client.arm());self.assertFalse(self.client.update(self.now))

    def test_consumption_is_single_use_and_frozen_data_cannot_create_permission(self):
        self.arm();self.confirm();self.assertTrue(self.client.consume());self.assertFalse(self.client.consume())
        self.assertTrue(self.client.permission)
        self.client.cancel();self.client._step();self.client._step()
        self.assertTrue(self.client.arm());self.client._step()
        self.remote.update(confirmed=True,confirmation_frames=4,confirmed_ms=630,target_locked=True,target_roi=[144,48,32,32],event_age_ms=630)
        self.now+=.8;self.client._step()
        self.assertFalse(self.client.consume());self.assertFalse(self.client.permission)

    def test_confirmed_flag_without_confirmation_evidence_is_rejected(self):
        self.arm();self.remote.update(confirmed=True)
        self.client._step();self.assertFalse(self.client.update(self.now))

    def test_permission_waits_for_road_without_a_time_limit(self):
        self.arm();self.confirm();self.now+=600
        self.client._step();self.client._step()
        self.assertTrue(self.client.update(self.now));self.assertTrue(self.client.permission)
        self.assertTrue(self.client.consume())

    def test_permission_and_evidence_remain_until_explicit_stop(self):
        self.arm();self.confirm();self.client._step();self.client._step()
        snap=self.client.snapshot()
        self.assertTrue(snap['permission']);self.assertTrue(snap['event_confirmed'])
        self.assertEqual(snap['confirmation_frames'],4);self.assertEqual(snap['confirmed_s'],.63)
        self.assertEqual(snap['phase'],'SIGNAL_READY')
        self.client.cancel();self.assertFalse(self.client.permission)
        self.client._step();self.client._step();self.assertTrue(self.client.available)
        self.assertTrue(self.client.arm());snap=self.client.snapshot()
        self.assertEqual(snap['phase'],'ARMING');self.assertFalse(snap['event_confirmed'])
        self.assertEqual(snap['confirmation_frames'],0)

    def test_matching_confirmed_session_has_no_event_age_expiry(self):
        self.arm();self.remote.update(confirmed=True,sequence=21,target_locked=True,target_roi=[144,48,32,32],confirmation_frames=4,confirmed_ms=630,event_age_ms=120000)
        self.client._step();self.assertTrue(self.client.permission)
        self.assertTrue(self.client.update(self.now))

    def test_actual_fault_reason_survives_connection_recovery(self):
        self.arm();self.remote['boot']='ffffffffffffffff';self.client._step()
        self.assertEqual(self.client.phase,'FAULT');self.assertEqual(self.client.snapshot()['fault_reason'],'DEVICE_REBOOTED')
        self.client.transport=lambda *args:dict(self.base)
        self.client._step();self.client._step();snap=self.client.snapshot()
        self.assertTrue(snap['connected']);self.assertEqual(snap['phase'],'FAULT')
        self.assertEqual(snap['fault_reason'],'DEVICE_REBOOTED');self.assertEqual(snap['error'],'')
        self.assertTrue(self.client.arm());self.assertEqual(self.client.snapshot()['fault_reason'],'')

    def test_lost_confirmation_reply_can_recover_with_same_session(self):
        self.arm();nonce=self.client.session;transport=self.client.transport
        self.client.transport=lambda *args:(_ for _ in ()).throw(OSError())
        self.client._step();self.assertTrue(self.client.armed);self.assertFalse(self.client.permission)
        self.assertEqual(self.client.session,nonce);self.assertFalse(self.client.update(self.now))
        self.remote.update(confirmed=True,target_locked=True,target_roi=[144,48,32,32],confirmation_frames=4,confirmed_ms=630,event_age_ms=120000,sequence=21)
        self.client.transport=transport;self.client._step()
        self.assertTrue(self.client.permission);self.assertTrue(self.client.update(self.now))

    def test_bad_confirmation_evidence_remains_a_fault_not_expiration(self):
        self.arm();self.remote.update(confirmed=True,event_age_ms=2100,confirmation_frames=1,confirmed_ms=0)
        self.client._step();snap=self.client.snapshot()
        self.assertEqual(snap['phase'],'FAULT');self.assertFalse(snap['event_confirmed'])
        self.assertEqual(snap['fault_reason'],'INVALID_SIGNAL_RESPONSE')

    def test_late_first_response_is_rejected_but_keeps_resolved_peer_for_retry(self):
        self.client.endpoint='http://10.0.0.2:81'
        def slow(*args):
            self.now+=.7
            return dict(self.base)
        self.client.transport=slow;self.client._step()
        self.assertFalse(self.client.available)
        self.assertEqual(self.client.endpoint,'http://10.0.0.2:81')
        self.assertEqual(self.client.last_error_reason,'RESPONSE_TOO_SLOW')
        self.client.transport=lambda *args:dict(self.base,sequence=30)
        self.client._step();self.assertTrue(self.client.available)
        self.assertEqual(self.client.snapshot()['peer_url'],'http://10.0.0.2/')

    def test_control_traffic_uses_dedicated_port_and_preview_stays_on_80(self):
        from unittest.mock import Mock
        response=Mock();response.read.return_value=json.dumps({'signal':self.base}).encode()
        response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=None)
        opener=Mock();opener.open.return_value=response
        with patch('robot.tasks.start_signal.socket.gethostbyname',return_value='10.0.0.2'),patch('robot.tasks.start_signal.build_opener',return_value=opener):
            self.client._http('status','','')
        self.assertEqual(opener.open.call_args.args[0].full_url,'http://10.0.0.2:81/status')
        self.assertFalse(self.client._valid(dict(self.base,control_port=80)))

    def test_invalid_locked_target_geometry_is_rejected(self):
        for roi in ([0,0,320,240],[112,22,0,32],[96,8,128,113],[True,22,32,32]):
            peer=dict(self.base,target_locked=True,target_roi=roi)
            self.assertFalse(self.client._valid(peer))

    def test_new_search_window_is_accepted_and_bounds_locked_face(self):
        peer = dict(self.base, roi=[80, 0, 160, 128], target_locked=True, target_roi=[82, 2, 24, 24])
        self.assertTrue(self.client._valid(peer))
        for target in ([79,2,24,24], [220,110,24,24]):
            self.assertFalse(self.client._valid(dict(peer,target_roi=target)))
        for roi in ([80,0,161,128], [80,False,160,128]):
            self.assertFalse(self.client._valid(dict(peer,roi=roi)))
        self.remote.update(roi=[80,0,160,128])
        self.client._step()
        self.assertEqual(self.client.snapshot()["search_roi"], [80,0,160,128])

    def test_brightening_confirmation_does_not_grant_permission(self):
        self.arm();self.remote.update(trigger_direction='BOTH',confirmed=True,confirmation_frames=4,confirmed_ms=630,event_age_ms=630)
        self.client._step();self.assertFalse(self.client.permission)

    def test_unlocked_confirmation_does_not_grant_permission(self):
        self.arm();self.remote.update(confirmed=True,confirmation_frames=4,confirmed_ms=630,event_age_ms=630,target_locked=False)
        self.client._step();self.assertFalse(self.client.permission)

    def test_relaxed_three_frame_400ms_contract_is_accepted(self):
        self.remote.update(required_frames=3,required_ms=400)
        self.arm();self.remote.update(confirmed=True,sequence=21,target_locked=True,target_roi=[144,48,32,32],confirmation_frames=3,confirmed_ms=410,event_age_ms=410)
        self.client._step();self.assertTrue(self.client.permission)
        self.assertEqual(self.client.snapshot()['required_frames'],3)

    def test_reboot_after_permission_does_not_revoke_it(self):
        self.arm();self.confirm();self.assertTrue(self.client.consume())
        self.remote.update(boot='ffffffffffffffff',armed=False,confirmed=False,session='')
        self.client._step();self.client._step()
        self.assertTrue(self.client.permission);self.assertEqual(self.client.phase,'RUNNING')

    def test_lost_cancellation_is_retried_without_revoking_permission(self):
        self.arm();self.confirm();transport=self.client.transport
        self.client.transport=lambda *args:(_ for _ in ()).throw(OSError())
        self.client._step();self.assertTrue(self.client.cancel_queue);self.assertTrue(self.client.permission)
        self.client.cancel();self.assertFalse(self.client.permission)
        self.client.transport=transport;self.client._step();self.client._step()
        self.assertFalse(self.client.cancel_queue);self.assertTrue(self.client.available)

    def test_new_program_cancels_old_esp_session_without_accepting_old_grant(self):
        self.remote.update(armed=True,confirmed=True,session='f'*32,target_locked=True,target_roi=[144,48,32,32],confirmation_frames=4,confirmed_ms=630,event_age_ms=12000)
        client=StartSignal({'enabled':True},app_dir=Path(self.temp.name),clock=lambda:self.now,transport=self.client.transport)
        client._step();self.assertFalse(client.permission);self.assertTrue(client.cancel_queue)
        client._step();client._step();self.assertTrue(client.available);self.assertFalse(client.permission)
