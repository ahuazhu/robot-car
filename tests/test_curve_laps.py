"""White-line identity, departure suppression, spatial crossing and stop latch."""
import sys,json,math,unittest
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
from lap_counter import LapMarkerDetector,LapCounter
from tracker import CurvePixels
from geometry import geometry_frame
FIX=ROOT/'tests/fixtures/lap_marker_20261031'

def marker(d=.4,qualified=True):return dict(center=[d-.02,0.],normal=[1.,0.],distance_m=d,sigma_m=.005,qualified=qualified,support=1.,contrast=60.)
def observation(d=None,gate=False,ready=False):
 m=marker(d) if d is not None else None
 return dict(stripes=[m] if m else [],marker_visible=gate,confirmed=ready,marker=m if ready else None)
def motion(travel=0.,valid=True,sigma=.001):
 return dict(rotation=np.eye(2),translation=np.array([-travel,0.]),valid=valid,sigma_m=sigma)
class LapCounterTests(unittest.TestCase):
 def setup_counter(self):
  c=LapCounter(dict(enabled=True,laps=3));c.begin(observation(.4,True,True),0.);return c
 def pass_line(self,c,at):
  if c.track is None:c.update(observation(.4,True,True),motion(),at,True)
  c.update(observation(.2,True),motion(.2),at+.1,True)
  return c.update(observation(),motion(.24),at+.2,True)
 def clear(self,c,at):
  c.update(observation(),motion(.6),at,True);c.update(observation(),motion(),at+.7,True)
 def test_initial_crossing_not_a_lap_and_exactly_three_returns_stop(self):
  c=self.setup_counter();self.assertFalse(self.pass_line(c,0.));self.assertEqual(c.count,0);self.assertEqual(c.event,'DEPARTURE_SKIPPED')
  self.clear(c,1.)
  for n in (1,2,3):
   done=self.pass_line(c,3.*n)
   self.assertEqual(c.count,n);self.assertEqual(done,n==3)
   if n<3:self.clear(c,3*n+1.)
  self.assertTrue(c.complete);self.assertEqual(c.phase,'FINISHED')
  for t in (11.,12.,13.):self.assertFalse(c.update(observation(.2,True,True),motion(.3),t,True))
  self.assertEqual(c.count,3)
 def test_repeated_visible_line_or_flicker_does_not_create_more_laps(self):
  c=self.setup_counter()
  for i in range(20):self.assertFalse(c.update(observation(.4,True,True),motion(),i*.1,True))
  self.assertEqual(c.count,0);self.pass_line(c,3.)
  for i in range(20):c.update(observation(.2,True,True),motion(),4+i*.1,True)
  self.assertEqual(c.count,0);self.assertTrue(c.visit_done)
 def test_unconfirmed_white_candidates_cannot_establish_a_visit(self):
  c=self.setup_counter();self.pass_line(c,0.);self.clear(c,1.)
  for i,d in enumerate((.5,.3,.1,.06)):c.update(observation(d),motion(.1),3+i*.1,True)
  self.assertEqual(c.count,0);self.assertIsNone(c.track)
 def test_clearance_is_not_blocked_by_unrelated_white_stripes(self):
  c=self.setup_counter();self.pass_line(c,0.)
  c.update(observation(.5),motion(.6),1.,True);c.update(observation(.4),motion(),1.7,True)
  self.assertEqual(c.phase,'SEEK_RETURN');self.assertFalse(c.departing)
 def test_line_occlusion_cannot_rearm_before_whole_body_passes(self):
  c=self.setup_counter();self.pass_line(c,0.)
  c.update(observation(),motion(),1.,True);c.update(observation(),motion(),2.,True)
  self.assertTrue(c.visit_done);self.assertEqual(c.phase,'WAIT_CLEAR');self.assertTrue(c.departing)
 def test_uncertain_initial_marker_cannot_turn_into_a_return_during_occlusion(self):
  c=self.setup_counter();c.update(observation(),motion(sigma=.09),.1,True)
  c.update(observation(),motion(),1.,True);c.update(observation(),motion(),2.,True)
  self.assertEqual(c.count,0);self.assertTrue(c.departing);self.assertTrue(c.visit_done)
 def test_current_invalid_motion_or_uncertain_plane_cannot_count(self):
  for valid,sigma in ((False,.001),(True,.09)):
   c=self.setup_counter();self.pass_line(c,0.);self.clear(c,1.);c.update(observation(.2,True,True),motion(),3.,True)
   c.update(observation(),motion(.3,valid,sigma),3.1,True)
   self.assertEqual(c.count,0)
 def test_confirmed_far_plane_can_count_after_measured_crossing_without_near_view(self):
  c=self.setup_counter();self.pass_line(c,0.);self.clear(c,1.);c.update(observation(.5,True,True),motion(),3.,True)
  c.update(observation(),motion(.6),3.1,True)
  self.assertEqual(c.count,1);self.assertEqual(c.missed,0)
 def test_reverse_motion_moves_line_away_instead_of_counting(self):
  c=self.setup_counter();c.update(observation(),motion(-.3),.1,True)
  self.assertGreater(c.track['distance'],.6);self.assertEqual(c.count,0)
 def test_second_nearby_white_stripe_cannot_replace_the_tracked_plane(self):
  c=self.setup_counter();c.update(observation(.2),motion(),.1,True)
  self.assertAlmostEqual(c.track['distance'],.4)
 def test_stop_retains_display_and_new_start_resets_a_new_race(self):
  c=self.setup_counter();self.pass_line(c,0.);self.clear(c,1.);self.pass_line(c,3.);self.assertEqual(c.count,1)
  c.stop();self.assertFalse(c.update(observation(.1,True,True),motion(.2),4.,False));self.assertEqual(c.count,1)
  c.begin(observation(.4,True,True),5.);self.assertEqual(c.count,0);self.assertTrue(c.departing)
 def test_start_without_visible_marker_seeks_departure_then_counts_three_returns(self):
  c=LapCounter(dict(enabled=True,laps=3));c.begin(observation(),0.)
  self.assertTrue(c.running);self.assertEqual(c.phase,'SEEK_DEPARTURE');self.assertEqual(c.count,0)
  for t in (.1,.2,.3):
   self.assertFalse(c.update(observation(),motion(),t,True));self.assertEqual(c.count,0)
  self.assertFalse(c.update(observation(.4,True,True),motion(),.4,True))
  self.assertEqual(c.phase,'DEPARTURE');self.assertIsNotNone(c.track)
  self.assertFalse(self.pass_line(c,1.));self.assertEqual(c.count,0);self.assertEqual(c.event,'DEPARTURE_SKIPPED')
  self.clear(c,2.)
  for n in (1,2,3):
   self.assertEqual(self.pass_line(c,3*n),n==3);self.assertEqual(c.count,n)
   if n<3:self.clear(c,3*n+1.)
  self.assertTrue(c.complete)
 def test_unconfirmed_lines_while_seeking_departure_never_count(self):
  c=LapCounter(dict(enabled=True,laps=3));c.begin(observation(),0.)
  for i,d in enumerate((.5,.3,.1,.01)):
   c.update(observation(d),motion(.1),i*.1,True)
   self.assertIsNone(c.track);self.assertEqual(c.count,0);self.assertEqual(c.phase,'SEEK_DEPARTURE')
 def test_disabled_counter_never_requests_a_stop(self):
  c=LapCounter({});c.begin({},0.)
  self.assertFalse(c.update(observation(.1,True,True),motion(.5),.1,True))

class RecordedMarkerTests(unittest.TestCase):
 def detector(self):
  cal=json.loads((FIX/'calibration.json').read_text());config=json.loads((FIX/'config.json').read_text());return LapMarkerDetector(CurvePixels(cal),config)
 def result(self,d,im):
  with geometry_frame():road,details=d.pixels.measure(im)
  self.assertIsNotNone(road)
  for t in (0.,.15,.3):r=d.detect(im,details,road,t)
  return r
 def test_actual_confirmed_stripe_selected_and_nearer_reflection_rejected(self):
  d=self.detector();r=self.result(d,cv2.imread(str(FIX/'start.png')))
  self.assertTrue(r['confirmed']);self.assertAlmostEqual(r['marker']['center'][0],.42,delta=.02)
  self.assertEqual(len(r['stripes']),1,'near reflection must not be a marker')
 def test_white_stripe_is_still_confirmed_without_the_removed_gantry(self):
  d=self.detector();im=cv2.imread(str(FIX/'start.png'));im[:148]=190
  r=self.result(d,im);self.assertTrue(r['marker_visible']);self.assertTrue(r['confirmed'])
 def test_one_frame_line_flash_is_not_enough_to_establish_a_visit(self):
  d=self.detector();im=cv2.imread(str(FIX/'start.png'))
  with geometry_frame():road,details=d.pixels.measure(im)
  self.assertFalse(d.detect(im,details,road,0.)['confirmed'])


class WhiteLineRegressionTests(unittest.TestCase):
 def detector(self,cal=None):
  cal=cal or json.loads((ROOT/'tests/fixtures/lap_line_20261031/calibration.json').read_text())
  return LapMarkerDetector(CurvePixels(cal),dict(enabled=True,laps=3))
 def test_all_current_recorded_lines_survive_without_a_gantry_or_full_navigation_curve(self):
  d=self.detector()
  for i in range(4):
   im=cv2.imread(str(ROOT/f'tests/fixtures/lap_line_20261031/line-{i}.png'))
   with geometry_frame():road,details=d.pixels.measure(im)
   self.assertIsNotNone(road)
   r=d.detect(im,details,road,i*.2)
   self.assertTrue(r['marker_visible']);self.assertEqual(len(r['stripes']),1)
   self.assertAlmostEqual(r['stripes'][0]['center'][0],.48,delta=.025)
   self.assertGreaterEqual(r['stripes'][0]['support'],.75)
  self.assertTrue(r['confirmed'])
 def test_real_bend_and_real_seven_bar_crosswalk_do_not_count_as_a_line(self):
  for folder,name in (('lap_line_20261031','bend.png'),('crosswalk_20261030','seven_bars.png')):
   f=ROOT/'tests/fixtures'/folder;d=self.detector(json.loads((f/'calibration.json').read_text()));im=cv2.imread(str(f/name))
   with geometry_frame():road,details=d.pixels.measure(im)
   self.assertIsNotNone(road)
   for t in (0.,.2,.4):r=d.detect(im,details,road,t)
   self.assertFalse(r['confirmed'],(folder,r));self.assertFalse(r['marker_visible'])
 def test_line_connected_to_longitudinal_edges_survives_and_bars_do_not(self):
  d=self.detector();g=d.pixels;xx,yy=np.meshgrid(g.xs,g.ys,indexing='ij');road=dict(path=np.c_[np.linspace(.25,.7,40),np.zeros(40)])
  for kind in ('line','partial','edges','zebra','red','blank'):
   value=np.full(g.visible.shape,100,np.uint8);sat=np.zeros(value.shape,np.uint8)
   if kind=='line':paint=(abs(xx-.45)<.012)|((abs(yy-.10)<.009)&(xx>.3))
   elif kind=='partial':paint=(abs(xx-.45)<.012)&(abs(yy)<.05)
   elif kind=='edges':paint=(abs(abs(yy)-.10)<.009)
   elif kind=='zebra':paint=(xx>.4)&(xx<.5)&(np.mod(yy+.14,.04)<.02)
   elif kind=='red':paint=abs(xx-.45)<.012;sat[paint]=200
   else:paint=np.zeros(value.shape,bool)
   value[paint]=245;details=dict(bird=np.repeat(value[:,:,None],3,axis=2),value=value,contrast_value=value,saturation=sat)
   result=d.stripes(details,road)
   self.assertEqual(bool(result),kind=='line',(kind,result))
 def test_consistency_compensates_motion_and_tolerates_half_second_frames(self):
  d=self.detector();observations=iter([marker(.7),marker(.52),marker(.34)])
  d.stripes=lambda *a:[next(observations)]
  for i,t in enumerate((0.,.5,1.)):
   r=d.detect(None,{},dict(),t,motion=motion(.18 if i else 0.))
  self.assertTrue(r['confirmed']);self.assertEqual(r['confirmation_frames'],3)
 def test_duplicate_lock_releases_after_body_passes_line_without_a_gantry(self):
  c=LapCounterTests().setup_counter();helper=LapCounterTests();helper.pass_line(c,0.)
  c.update(observation(),motion(.2),1.,True);c.update(observation(),motion(),1.7,True)
  self.assertEqual(c.phase,'SEEK_RETURN');self.assertFalse(c.departing)
 def test_shadow_and_exposure_do_not_change_current_marker_identity(self):
  raw=cv2.imread(str(ROOT/'tests/fixtures/lap_line_20261031/line-0.png'))
  for gain in (.55,.75,1.,1.15):
   d=self.detector();im=np.clip(raw.astype(float)*gain,0,255).astype(np.uint8)
   with geometry_frame():road,details=d.pixels.measure(im)
   self.assertIsNotNone(road)
   for t in (0.,.2,.4):r=d.detect(im,details,road,t)
   self.assertTrue(r['confirmed'],(gain,r));self.assertAlmostEqual(r['marker']['center'][0],.48,delta=.03)

if __name__=='__main__':unittest.main()
