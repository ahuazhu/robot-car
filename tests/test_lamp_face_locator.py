"""Native C++ locator and model-hinted temporal start contracts."""
import unittest
import numpy as np
from esp32s3.lampnet.native_locator import NativeLocator
from esp32s3.video_preview.replay import NativeDetector

class LocatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.locator=NativeLocator()
    @classmethod
    def tearDownClass(cls):cls.locator.close()
    def scene(self):return np.zeros((40,40),np.uint8),np.full((40,40,3),30,np.uint8)
    def test_faint_complete_lamp_is_retained_without_high_threshold_pixels(self):
        p,rgb=self.scene();p[10:22,12:24]=180;rgb[10:22,12:24]=140
        r,_=self.locator.locate(p,rgb);self.assertTrue(r['coarse_valid']);self.assertTrue(r['valid'])
    def test_tiny_strong_core_falls_back_to_usable_coarse_face(self):
        p,rgb=self.scene();p[10:22,12:24]=180;p[15,16]=255
        r,_=self.locator.locate(p,rgb);self.assertTrue(r['valid']);self.assertEqual(r['method'],1)
    def test_connected_glare_is_trimmed_from_complete_face(self):
        p,rgb=self.scene();p[10:28,10:30]=240;p[11:27,11:27]=255;rgb[11:27,11:27]=250
        r,_=self.locator.locate(p,rgb);self.assertTrue(r['valid']);self.assertLessEqual(r['box'][2],r['coarse_box'][2]);self.assertEqual(r['method'],3)
    def test_competing_separate_panels_remain_ambiguous(self):
        p,rgb=self.scene();p[8:18,6:16]=255;p[8:18,24:34]=255
        r,_=self.locator.locate(p,rgb);self.assertFalse(r['valid'])
    def test_blank_clears_previous_mask(self):
        p,rgb=self.scene();p[8:24,8:24]=255;self.locator.locate(p,rgb)
        r,mask=self.locator.locate(np.zeros_like(p),rgb);self.assertFalse(r['valid']);self.assertEqual(int(mask.sum()),0)

class ModelHintTests(unittest.TestCase):
    def setUp(self):self.d=NativeDetector();self.d.hint(None);self.d.arm();self.bg=np.full((240,320,3),40,np.uint8);self.on=self.bg.copy();self.on[48:80,144:176]=190
    def tearDown(self):self.d.close()
    def lock(self):
        self.d.hint([140,44,40,40]);states=[self.d.process(self.on,t) for t in [0,100,200,300]];self.assertTrue(states[-1]['locked'])
    def test_no_model_hint_cannot_lock_or_fire(self):
        for t in range(0,1000,100):self.d.process(self.on,t)
        self.assertFalse(self.d.state()['locked']);self.assertFalse(self.d.state()['confirmed'])
    def test_locked_face_survives_model_loss_and_confirms_dark(self):
        self.lock();before=self.d.mask().copy();self.d.hint(None)
        states=[self.d.process(self.bg,t) for t in [400,610,820,1030]]
        self.assertTrue(states[-1]['confirmed']);self.assertGreaterEqual(states[-1]['elapsed'],400);self.assertTrue(np.array_equal(before,self.d.mask()))
        self.d.process(self.on,20000);self.assertTrue(self.d.state()['confirmed']);self.d.cancel();self.assertFalse(self.d.state()['confirmed'])
    def test_changing_hint_after_lock_cannot_retarget(self):
        self.lock();box=[self.d.state()[k] for k in ['target_x','target_y','target_w','target_h']];self.d.hint([180,10,40,40]);self.d.process(self.on,400)
        self.assertEqual(box,[self.d.state()[k] for k in ['target_x','target_y','target_w','target_h']])
    def test_global_darkening_is_not_a_start(self):
        self.lock();darker=np.clip(self.on.astype(int)-30,0,255).astype(np.uint8)
        for t in [400,610,820,1030]:self.assertFalse(self.d.process(darker,t)['confirmed'])

class FullFrameTests(unittest.TestCase):
    def test_full_view_locator_maps_candidate_outside_old_cyan_box(self):
        locator=NativeLocator([0,0,320,240])
        try:
            p=np.zeros((40,40),np.uint8);rgb=np.full((40,40,3),40,np.uint8)
            p[24:34,2:12]=230;rgb[24:34,2:12]=220
            result,_=locator.locate(p,rgb)
            self.assertTrue(result['valid']);self.assertLess(result['box'][0],80);self.assertGreater(result['box'][1],128)
        finally:locator.close()
    def test_lamp_beyond_old_box_locks_and_confirms_without_growing_workspace(self):
        d=NativeDetector(full_frame=True)
        try:
            self.assertEqual(d.roi,[0,0,320,240]);self.assertLessEqual(d.mask().size,80*64)
            on=np.full((240,320,3),40,np.uint8);on[148:192,20:64]=200
            d.hint([16,144,52,52]);d.arm()
            for t in [0,160,320,480]:state=d.process(on,t)
            self.assertTrue(state['locked']);before=d.mask().copy();d.hint(None)
            for t in [640,800,960,1120]:state=d.process(np.full_like(on,40),t)
            self.assertTrue(state['confirmed']);self.assertTrue(np.array_equal(before,d.mask()))
        finally:d.close()

class ManualROITests(unittest.TestCase):
    def setUp(self):
        self.d=NativeDetector(full_frame=True);self.on=np.full((240,320,3),180,np.uint8);self.on[140:200,20:80]=160
        self.assertTrue(self.d.manual([20,140,60,60]));self.d.hint(None);self.d.arm()
    def tearDown(self):self.d.close()
    def lock(self):
        for t in [0,160,320,480]:state=self.d.process(self.on,t)
        self.assertTrue(state['locked'])
    def test_manual_face_need_not_be_brighter_than_background(self):
        self.lock();off=self.on.copy();off[140:200,20:80]=60
        for t in [640,800,960,1120]:state=self.d.process(off,t)
        self.assertTrue(state['confirmed']);self.d.process(self.on,20000);self.assertTrue(self.d.state()['confirmed'])
    def test_changes_outside_manual_face_do_not_fire(self):
        self.lock();other=self.on.copy();other[20:80,200:260]=30
        for t in [640,800,960,1120]:self.assertFalse(self.d.process(other,t)['confirmed'])
    def test_global_exposure_darkening_does_not_fire(self):
        self.lock();other=np.clip(self.on.astype(int)-30,0,255).astype(np.uint8)
        for t in [640,800,960,1120]:self.assertFalse(self.d.process(other,t)['confirmed'])
    def test_invalid_manual_rectangles_are_rejected(self):
        for box in [[-1,20,40,40],[0,0,321,40],[20,20,4,4],[20,239,20,20]]:self.assertFalse(self.d.manual(box))
    def test_stop_then_new_manual_position_uses_new_background(self):
        self.lock();self.d.cancel();self.assertTrue(self.d.manual([200,24,48,48]));self.d.arm()
        on=np.full_like(self.on,70);on[24:72,200:248]=220
        for t in [1000,1160,1320,1480]:state=self.d.process(on,t)
        self.assertTrue(state['locked']);self.assertEqual(state['target_x'],200)
        for t in [1640,1800,1960,2120]:state=self.d.process(np.full_like(on,70),t)
        self.assertTrue(state['confirmed'])
    def test_stop_then_automatic_mode_relocates_coloured_face(self):
        self.lock();self.d.cancel();self.d.manual(None);self.d.hint(None,required=False);self.d.arm()
        on=np.full_like(self.on,30);on[40:80,196:236]=(20,20,240)
        for t in [1000,1160,1320,1480]:state=self.d.process(on,t)
        self.assertTrue(state['locked']);self.assertGreaterEqual(state['target_x'],190)
        for t in [1640,1800,1960,2120]:state=self.d.process(np.full_like(on,30),t)
        self.assertTrue(state['confirmed'])

class OffOnOffTests(unittest.TestCase):
    def setUp(self):
        self.d=NativeDetector(full_frame=True);self.box=[100,40,80,64];self.d.manual(self.box);self.d.cycle();self.d.arm();self.t=0
        self.off=np.full((240,320,3),180,np.uint8);self.off[40:104,100:180]=60
        self.on=self.off.copy();self.on[40:104,100:180]=160
    def tearDown(self):self.d.close()
    def frames(self,image,count=6):
        states=[]
        for _ in range(count):states.append(self.d.process(image,self.t));self.t+=150
        return states
    def test_initial_grey_off_never_locks_or_fires_even_when_it_darkens_more(self):
        self.frames(self.off,30);darker=self.off.copy();darker[40:104,100:180]=20
        states=self.frames(darker,10);self.assertFalse(states[-1]['locked']);self.assertFalse(any(s['confirmed'] for s in states))
    def test_off_on_off_confirms_and_holds_until_stop(self):
        self.frames(self.off,10);self.assertTrue(self.frames(self.on)[-1]['locked']);self.assertFalse(self.d.state()['confirmed'])
        self.assertTrue(self.frames(self.off)[-1]['confirmed']);self.frames(self.on);self.assertTrue(self.d.state()['confirmed']);self.d.cancel();self.assertFalse(self.d.state()['confirmed'])
    def test_white_red_green_on_are_all_supported(self):
        for colour in [(230,230,230),(15,15,220),(15,220,15)]:
            with self.subTest(colour=colour):
                self.d.arm();self.frames(self.off);on=self.off.copy();on[40:104,100:180]=colour
                self.assertTrue(self.frames(on)[-1]['locked']);self.assertTrue(self.frames(self.off)[-1]['confirmed'])
    def test_short_flash_is_not_the_on_baseline(self):
        self.frames(self.off);self.frames(self.on,2);self.frames(self.off)
        self.assertFalse(self.d.state()['locked']);self.assertFalse(self.d.state()['confirmed']);self.assertTrue(self.frames(self.on)[-1]['locked'])
    def test_global_brightening_is_not_a_lamp_on(self):
        self.frames(self.off);brighter=np.clip(self.off.astype(int)+40,0,255).astype(np.uint8)
        self.frames(brighter);self.assertFalse(self.d.state()['locked']);self.frames(self.off);self.assertFalse(self.d.state()['confirmed'])
    def test_model_can_find_target_after_initial_off_reference(self):
        self.d.manual(None);self.d.hint(None);self.frames(self.off)
        self.d.hint(self.box);bright=self.off.copy();bright[40:104,100:180]=235
        self.assertTrue(self.frames(bright)[-1]['locked']);self.d.hint(None)
        self.assertTrue(self.frames(self.off)[-1]['confirmed'])
    def test_stop_and_rearm_require_another_on(self):
        self.frames(self.off);self.frames(self.on);self.frames(self.off);self.assertTrue(self.d.state()['confirmed']);self.d.cancel();self.d.arm()
        self.frames(self.off,10);self.assertFalse(self.d.state()['locked']);self.assertFalse(self.d.state()['confirmed'])
    def test_continuity_gap_cannot_reuse_the_old_on(self):
        self.frames(self.off);self.frames(self.on);self.t+=1000;self.frames(self.off)
        self.assertFalse(self.d.state()['locked']);self.assertFalse(self.d.state()['confirmed'])

if __name__=='__main__':unittest.main()
