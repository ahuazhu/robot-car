"""Recorded saturated gantry lamp and one-shot start/STOP timing."""
import json,sys,unittest
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
from start_light import StartLight
FIX=ROOT/'tests/fixtures/start_lamp_20261031'
DATA=json.loads((FIX/'calibration.json').read_text());CONFIG=DATA['config']
MODEL=cv2.imread(str(FIX/'template.png'))
def make_gate(config=CONFIG):return StartLight(config,template=MODEL)

def image(state,i=0):
 im=np.full((240,320,3),190,np.uint8);crop=cv2.imread(str(FIX/f'{state}-{i}.png'));x,y=DATA['crop_origin'];im[y:y+len(crop),x:x+crop.shape[1]]=crop
 return im

class DetectorTests(unittest.TestCase):
 def test_real_saturated_lamp_distinguishes_all_on_off_samples(self):
  gate=make_gate()
  for state in ('off','on'):
   for i in range(4):
    with self.subTest(state=state,image=i):self.assertEqual(gate.detect(image(state,i))['state'],state.upper())
 def test_red_clothing_outside_lamp_and_bright_uniform_sky_cannot_start(self):
  gate=make_gate();im=image('off');im[65:150,200:]=[0,0,255]
  self.assertEqual(gate.detect(im)['state'],'OFF')
  for value in (0,120,190,255):self.assertEqual(gate.detect(np.full((240,320,3),value,np.uint8))['state'],'OFF')
 def test_global_brightness_change_preserves_local_difference(self):
  gate=make_gate()
  for delta in (-15,15):
   for state in ('off','on'):
    im=np.clip(image(state).astype(int)+delta,0,255).astype(np.uint8)
    self.assertEqual(gate.detect(im)['state'],state.upper())
 def test_wrong_image_cannot_be_mistaken_for_lit_lamp(self):
  gate=make_gate()
  for im in (None,np.zeros((120,160,3),np.uint8)):self.assertEqual(gate.detect(im)['state'],'UNKNOWN')
 def test_invalid_calibration_rejected_before_drive(self):
  for changes in (dict(core_roi=[310,18,18,15]),dict(match_score_min=2),dict(minimum_luma=float('nan')),dict(minimum_contrast=-1),dict(confirm_s=-1),dict(auto_arm_on_launch='yes'),dict(minimum_halo_contrast=0),dict(minimum_halo_contrast=float('nan'))):
   with self.subTest(changes=changes),self.assertRaises(ValueError):make_gate(dict(CONFIG,**changes))

class FullFrameTests(unittest.TestCase):
 def placed(self,x,y,scale=1.):
  patch=cv2.resize(MODEL,None,fx=scale,fy=scale)
  im=np.full((240,320,3),190,np.uint8);im[y:y+len(patch),x:x+patch.shape[1]]=patch
  return im
 def test_translated_scaled_lamp_across_entire_frame(self):
  for scale in (.6,1.,1.5,2.):
   patch=cv2.resize(MODEL,None,fx=scale,fy=scale)
   for x,y in ((0,0),(10,100),(140,110),(320-patch.shape[1],240-patch.shape[0])):
    with self.subTest(scale=scale,x=x,y=y):
     observation=make_gate().detect(self.placed(x,y,scale))
     self.assertEqual(observation['state'],'ON')
     self.assertLessEqual(abs(observation['roi'][0]-x),2)
     self.assertLessEqual(abs(observation['roi'][1]-y),2)
 def test_full_recorded_scenes_reject_other_bright_regions(self):
  for state in ('off','on'):
   for i in range(4):
    with self.subTest(state=state,i=i):
     self.assertEqual(make_gate().detect(cv2.imread(str(FIX/f'scene-{state}-{i}.png')))['state'],state.upper())
 def test_two_matching_lamps_are_ambiguous(self):
  im=self.placed(20,30);im[150:169,250:278]=MODEL
  observation=make_gate().detect(im)
  self.assertEqual(observation['state'],'UNKNOWN')
  self.assertEqual(observation['reason'],'MULTIPLE_MATCHED_LAMPS')
 def test_candidate_jump_requires_new_continuous_confirmation(self):
  gate=make_gate()
  for t in (0.,.1,.2,.3):self.assertFalse(gate.update(gate.detect(self.placed(20,30)),t))
  moved=gate.detect(self.placed(250,180));self.assertEqual(moved['state'],'UNKNOWN')
  self.assertFalse(gate.update(moved,.4))
  for t in (.5,.6,.7,.8):self.assertFalse(gate.update(gate.detect(self.placed(250,180)),t))
  self.assertTrue(gate.update(gate.detect(self.placed(250,180)),.9))
 def test_flat_or_missing_appearance_is_rejected(self):
  with self.assertRaises(ValueError):StartLight(CONFIG,template=np.full((19,28,3),255,np.uint8))
  with self.assertRaises(ValueError):StartLight(dict(CONFIG,template_file='../../missing.png'))

class GateTests(unittest.TestCase):
 def test_sustained_light_requests_once_then_never_restarts_after_run(self):
  gate=make_gate()
  for t in (0.,.1,.2,.3):self.assertFalse(gate.update(dict(state='ON'),t))
  self.assertTrue(gate.update(dict(state='ON'),.4));gate.started()
  for t in (.5,.8,1.,2.):self.assertFalse(gate.update(dict(state='ON'),t,active=False))
  self.assertTrue(gate.fired);self.assertFalse(gate.armed)
 def test_off_unknown_and_flashes_do_not_accumulate_confirmation(self):
  for interruption in ('OFF','UNKNOWN'):
   gate=make_gate()
   for t in (0.,.1,.2):self.assertFalse(gate.update(dict(state='ON'),t))
   self.assertFalse(gate.update(dict(state=interruption),.3))
   for t in (.4,.5,.6,.7):self.assertFalse(gate.update(dict(state='ON'),t))
   self.assertTrue(gate.update(dict(state='ON'),.8))
 def test_missing_or_slow_frame_cannot_bridge_light_confirmation(self):
  for kind in ('gap','slow'):
   gate=make_gate()
   for t in (0.,.1,.2):gate.update(dict(state='ON'),t)
   self.assertFalse(gate.update(dict(state='ON'),1. if kind=='gap' else .3,timely=kind!='slow'))
   self.assertEqual(gate.elapsed,0.)
 def test_slow_lit_frame_is_reported_separately_and_never_confirms(self):
  gate=make_gate()
  for t in (0.,.1,.2,.3,.4):self.assertFalse(gate.update(dict(state='ON'),t,timely=False))
  self.assertEqual(gate.snapshot()['phase'],'WAIT_FRESH_FRAME')
  for t in (.5,.6,.7,.8):self.assertFalse(gate.update(dict(state='ON'),t))
  self.assertTrue(gate.update(dict(state='ON'),.9))
 def test_stop_overrides_a_confirmed_or_unconfirmed_lit_signal(self):
  for confirmed in (False,True):
   gate=make_gate()
   for t in (0.,.1,.2,.3,.4) if confirmed else (0.,.1):gate.update(dict(state='ON'),t)
   gate.cancel()
   for t in (.5,.6,.7,1.,2.):self.assertFalse(gate.update(dict(state='ON'),t))
   self.assertEqual(gate.phase,'DISARMED')
 def test_explicit_rearm_still_requires_current_confirmation(self):
  gate=make_gate();gate.cancel();gate.arm()
  self.assertFalse(gate.update(dict(state='ON'),10.))
  for t in (10.1,10.2,10.3):self.assertFalse(gate.update(dict(state='ON'),t))
  self.assertTrue(gate.update(dict(state='ON'),10.4))
 def test_cancel_discards_old_lamp_result_and_rearm_waits_for_new_frame(self):
  gate=make_gate();gate.update(dict(state='ON',roi=[20,30,28,19]),0.)
  gate.cancel();snapshot=gate.snapshot()
  self.assertFalse(snapshot['armed']);self.assertEqual(snapshot['lamp']['state'],'UNKNOWN')
  self.assertEqual(snapshot['lamp']['reason'],'DISARMED')
  gate.arm();self.assertEqual(gate.snapshot()['lamp']['reason'],'WAITING_FOR_FRAME')
  self.assertFalse(gate.update(dict(state='ON'),1.));self.assertEqual(gate.elapsed,0.)
 def test_disabled_mode_and_non_auto_arm_never_request_by_themselves(self):
  for config in ({},dict(CONFIG,auto_arm_on_launch=False)):
   gate=StartLight(config)
   for t in (0.,.1,.2,.3,.4):self.assertFalse(gate.update(dict(state='ON'),t))
 def test_nonmonotonic_or_nonfinite_time_restarts_confirmation(self):
  gate=make_gate()
  for t in (0.,.1,.2):gate.update(dict(state='ON'),t)
  for t in (.1,float('nan'),.3):self.assertFalse(gate.update(dict(state='ON'),t))


class LedAppearanceTests(unittest.TestCase):
 def detector(self):
  c=json.loads((ROOT/'tests/fixtures/start_lamp_led_20261031/config.json').read_text())
  return StartLight(c)
 def test_new_led_array_and_original_saturated_lamp_share_one_full_frame_detector(self):
  for folder,prefix in (('start_lamp_led_20261031',''),('start_lamp_20261031','scene-')):
   for state in ('on','off'):
    for i in range(4):
     with self.subTest(folder=folder,state=state,i=i):
      im=cv2.imread(str(ROOT/f'tests/fixtures/{folder}/{prefix}{state}-{i}.png'))
      self.assertEqual(self.detector().detect(im)['state'],state.upper())
 def test_led_array_works_at_new_positions_and_scales(self):
  model=cv2.imread(str(ROOT/'maixcam2/curve_follow/start_light_led_template.png'))
  for scale in (.8,1.,1.5):
   patch=cv2.resize(model,None,fx=scale,fy=scale)
   for x,y in ((0,0),(240,180),(100,110)):
    im=np.full((240,320,3),160,np.uint8);im[y:y+len(patch),x:x+patch.shape[1]]=patch
    with self.subTest(scale=scale,x=x,y=y):self.assertEqual(self.detector().detect(im)['state'],'ON')
 def test_stop_still_overrides_new_appearance_and_never_uses_old_confirmation(self):
  d=self.detector();im=cv2.imread(str(ROOT/'tests/fixtures/start_lamp_led_20261031/on-0.png'))
  for t in (0.,.1,.2,.3,.4):ready=d.update(d.detect(im),t)
  self.assertTrue(ready);d.cancel()
  for t in (.5,.6,.8):self.assertFalse(d.update(d.detect(im),t))

class EdgeAndBloomTests(unittest.TestCase):
 def detector(self):
  return StartLight(json.loads((ROOT/'maixcam2/curve_follow/runtime.json').read_text())['start_light'])
 def test_user_confirmed_edge_and_bloom_on_off_at_both_positions(self):
  for group in ('bloom','edge'):
   for state in ('on','off'):
    for i in range(4):
     with self.subTest(group=group,state=state,frame=i):
      im=cv2.imread(str(ROOT/f'tests/fixtures/start_lamp_edge_20261031/{group}-{state}-{i}.png'))
      self.assertEqual(self.detector().detect(im)['state'],state.upper())
 def test_template_background_may_extend_outside_image_at_every_edge(self):
  model=cv2.imread(str(ROOT/'maixcam2/curve_follow/start_light_led_template.png'))
  for x,y in ((-4,70),(295,70),(140,-4),(140,218)):
   im=np.full((240,320,3),160,np.uint8);h,w=model.shape[:2]
   x0=max(0,x);y0=max(0,y);x1=min(320,x+w);y1=min(240,y+h)
   im[y0:y1,x0:x1]=model[y0-y:y1-y,x0-x:x1-x]
   with self.subTest(x=x,y=y):self.assertEqual(self.detector().detect(im)['state'],'ON')
 def test_white_rectangles_and_reflections_at_image_edges_do_not_start(self):
  for x,y in ((0,0),(295,0),(0,215),(295,215),(120,70)):
   im=np.full((240,320,3),160,np.uint8);im[y:y+25,x:x+25]=255
   with self.subTest(x=x,y=y):self.assertEqual(self.detector().detect(im)['state'],'OFF')
 def test_new_edge_on_samples_confirm_but_old_off_samples_never_trigger(self):
  d=self.detector();d.arm()
  for i in range(4):
   im=cv2.imread(str(ROOT/f'tests/fixtures/start_lamp_edge_20261031/edge-off-{i}.png'))
   self.assertFalse(d.update(d.detect(im),i*.2))
  for i in range(4):
   im=cv2.imread(str(ROOT/f'tests/fixtures/start_lamp_edge_20261031/edge-on-{i}.png'))
   ready=d.update(d.detect(im),1.+i*.15)
  self.assertTrue(ready);d.cancel()
  self.assertFalse(d.update(dict(state='ON'),1.6))

if __name__=='__main__':unittest.main()
