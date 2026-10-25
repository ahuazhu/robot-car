import sys,json,math,unittest
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/target_follow'))
from tracker import TargetTracker,RoadPixels,transform_path,target_at,target_command
CAL=json.loads((ROOT/'maixcam2/duckietown_test/calibration.json').read_text())

def scene(side='both',offset=0.):
    yy,xx=np.indices((240,320));p=np.stack((xx/320,yy/240,np.ones_like(xx)),axis=-1)@np.array(CAL['H_normalized']).T
    x=p[:,:,0]/p[:,:,2];y=p[:,:,1]/p[:,:,2]-offset
    valid=(x>.15)&(x<1)
    img=np.full((240,320,3),120,np.uint8)
    left=(y>.15)&valid;right=(y<-.15)&valid
    if side in ('both','left'):
        img[left]=(35,180,35);img[(y>=.15)&(y<.17)&valid]=240
    if side in ('both','right'):
        img[right]=(35,180,35);img[(y<=-.15)&(y>-.17)&valid]=240
    return img

class FakePixels:
    def __init__(self):self.measurement=None
    def measure(self,*args):return self.measurement,dict(candidates={'left':[],'right':[]})

def measured(y=0.,mode='TWO EDGES'):
    return dict(path=np.column_stack((np.linspace(.2,.65,30),np.full(30,y))),mode=mode,confidence=.85)

class TargetTests(unittest.TestCase):
    def test_straight_metric_center_and_one_sided_offset(self):
        for side in ('both','left','right'):
            for offset in (0.,.04,-.04):
                r,_=TargetTracker(CAL).update(scene(side,offset),0)
                self.assertTrue(r['valid'],(side,r))
                self.assertAlmostEqual(r['target_m'][1],offset,delta=.012)
    def test_blank_has_no_target(self):
        r,_=TargetTracker(CAL).update(np.zeros((240,320,3),np.uint8),0)
        self.assertFalse(r['valid'])
    def test_motion_updates_stored_ground_points(self):
        p=np.array([[.3,.1],[.4,.1],[.5,.1]])
        np.testing.assert_allclose(transform_path(p,.1,0,.5),p-[.05,0])
        q=transform_path(p,0,math.pi/2,1)
        np.testing.assert_allclose(q[:,0],p[:,1],atol=1e-9)
        np.testing.assert_allclose(q[:,1],-p[:,0],atol=1e-9)
    def test_no_extrapolation_beyond_visible_path(self):
        self.assertIsNone(target_at([[.42,0],[.5,0],[.6,0]]))
        self.assertAlmostEqual(target_at([[.1,0],[.2,0],[.25,0]])[0],.25)
        self.assertIsNone(target_at([[.1,0],[.2,0],[.25,0]],.30))
        self.assertIsNone(target_at([[.2,0],[.3,0],[.4,0]],.10))
    def test_short_prediction_expires_and_does_not_refresh_itself(self):
        t=TargetTracker(CAL);f=FakePixels();t.pixels=f;f.measurement=measured(.03)
        t.update(None,0);f.measurement=None
        r,_=t.update(None,.1,.05,.1);self.assertTrue(r['valid']);self.assertEqual(r['mode'],'PREDICT')
        self.assertLess(r['target_m'][1],.03)
        r,_=t.update(None,.7,.05,.1);self.assertFalse(r['valid'])
    def test_live_single_side_has_no_arbitrary_timeout(self):
        t=TargetTracker(CAL);f=FakePixels();t.pixels=f;f.measurement=measured(.01,'ONE RIGHT')
        for now in np.arange(0,5,.1):
            r,_=t.update(None,float(now));self.assertTrue(r['valid']);self.assertTrue(r['observed'])
    def test_moving_jump_is_rejected_and_stationary_reacquired(self):
        t=TargetTracker(CAL);f=FakePixels();t.pixels=f;f.measurement=measured(0)
        t.update(None,0);f.measurement=measured(.15)
        r,_=t.update(None,.1,.05,0);self.assertFalse(r['observed']);self.assertEqual(r['mode'],'PREDICT')
        for now in (.2,.3,.4,.5):r,_=t.update(None,now)
        self.assertTrue(r['observed'])
    def test_pursuit_sign_and_reduced_prediction_speed(self):
        r=dict(valid=True,target_m=[.3,.04],mode='TWO EDGES')
        left,right=target_command(r);self.assertLess(left,right)
        predicted=target_command(dict(r,mode='PREDICT'));self.assertLess(sum(predicted),left+right)
        self.assertIsNone(target_command(dict(r,valid=False)))
    def test_tight_turn_selects_closer_supported_point(self):
        x=np.linspace(.10,.55,40)
        straight=target_at(np.column_stack((x,x*0)))
        bend=target_at(np.column_stack((x,.9*x)))
        self.assertAlmostEqual(straight[0],.30)
        self.assertGreaterEqual(bend[0],.10)
        self.assertLess(bend[0],.20)
        self.assertAlmostEqual(bend[1],.9*bend[0])
        # No near observations: choose supported endpoint, not an invented 12 cm point.
        far=np.column_stack((x[x>=.25],.9*x[x>=.25]))
        self.assertGreaterEqual(target_at(far)[0],far[0,0])
    def test_lateral_target_beyond_old_18cm_limit_is_valid(self):
        t=TargetTracker(CAL);f=FakePixels();t.pixels=f;f.measurement=measured(.23,'ONE RIGHT')
        r,_=t.update(None,0)
        self.assertTrue(r['valid']);self.assertTrue(r['observed'])
        self.assertGreater(r['target_m'][1],.18)
        self.assertEqual(r['invalid_reason'],'')
    def test_single_edge_normal_infers_near_center_outside_image(self):
        pixels=RoadPixels(CAL)
        x=np.linspace(.22,.55,34)
        edge=np.column_stack((x,.9*x-.15,np.ones(len(x)))).tolist()
        pixels.extract=lambda *args: ({'left':[],'right':edge},None)
        m,_=pixels.measure(None)
        self.assertIsNotNone(m)
        target=target_at(m['path'])
        self.assertLess(target[0],.20)
        # Perpendicular distance to observed boundary is exactly half the road width.
        distance=(target[1]-.9*target[0]+.15)/math.sqrt(1+.9**2)
        self.assertAlmostEqual(distance,.15,places=5)
        p=np.linalg.inv(np.array(CAL['H_normalized']))@[*target,1.]
        u,v=p[:2]/p[2]*[320,240]
        self.assertFalse(0<=u<320 and 0<=v<240)
    def test_adaptive_lookahead_does_not_create_false_path_jump(self):
        t=TargetTracker(CAL);f=FakePixels();t.pixels=f
        x=np.linspace(.10,.55,40)
        f.measurement=dict(path=np.column_stack((x,.6*x)),mode='ONE RIGHT',confidence=.85)
        t.update(None,0)
        f.measurement=dict(path=np.column_stack((x,.65*x)),mode='ONE RIGHT',confidence=.85)
        r,_=t.update(None,.05,.04,.1)
        self.assertTrue(r['observed'])
    def test_bend_slows_forward_and_has_more_steering_authority(self):
        straight=target_command(dict(valid=True,target_m=[.3,0.],mode='ONE RIGHT'))
        bend=target_command(dict(valid=True,target_m=[.14,.20],mode='ONE RIGHT'))
        self.assertLess(sum(bend),sum(straight))
        self.assertGreater(bend[1]-bend[0],12)
        self.assertGreaterEqual(min(bend),0)
        right=target_command(dict(valid=True,target_m=[.14,-.20],mode='ONE LEFT'))
        self.assertEqual(right,bend[::-1])
    def test_unsupported_measurement_and_memory_expiry_have_reasons(self):
        t=TargetTracker(CAL);f=FakePixels();t.pixels=f;f.measurement=measured(0)
        t.update(None,0)
        f.measurement=dict(path=np.array([[.5,0],[.6,0],[.7,0]]),mode='ONE RIGHT',confidence=.9)
        r,_=t.update(None,.1,.04,0)
        self.assertEqual(r['measurement_reject'],'NO SUPPORTED TARGET')
        self.assertEqual(r['mode'],'PREDICT')
        r,_=t.update(None,.8,.04,0)
        self.assertFalse(r['valid']);self.assertEqual(r['invalid_reason'],'MEMORY EXPIRED')
    def test_known_failed_frames(self):
        paths=sorted((ROOT/'.maixpy/runs/20261021-211932-duck_single_stationary/artifacts').glob('raw*.jpg'))
        self.assertEqual(len(paths),3)
        targets=[]
        for p in paths:
            r,_=TargetTracker(CAL).update(cv2.imread(str(p)),0)
            self.assertTrue(r['valid']);self.assertEqual(r['mode'],'ONE RIGHT');targets.append(r['target_m'][1])
        self.assertLess(max(targets)-min(targets),.005)
if __name__=='__main__':unittest.main()
