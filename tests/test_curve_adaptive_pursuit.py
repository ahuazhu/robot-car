"""Pursuit target provenance and delayed closed-loop oscillation regression.

The plant is a stress model, not an identified drivetrain or proof of track
performance. Actual trace replay compares commands, not a counterfactual run.
"""
import sys,json,math,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
from geometry import pursuit_point,nearest_forward_point,support_projection
from steering_control import SteeringControl
from road_policy import RoadPolicy

def straight(y=0.,start=-.2,end=.8):
 return np.c_[np.linspace(start,end,100),np.full(100,y)]

class AdaptiveGeometryTests(unittest.TestCase):
 def test_target_is_on_supported_curve_at_physical_radius(self):
  p=straight(.04);r=pursuit_point(p,.24)
  self.assertEqual(r['mode'],'LOOKAHEAD')
  self.assertAlmostEqual(np.linalg.norm(r['point']),.24)
  self.assertLess(support_projection([r['point']],p)[0][0],1e-10)
  self.assertGreater(r['point'][0],.2)
 def test_short_and_far_support_do_not_generate_unobserved_points(self):
  short=straight(.02,.06,.12);a=pursuit_point(short,.25)
  self.assertEqual(a['mode'],'SHORT_SUPPORT');self.assertLess(a['speed_scale'],.5)
  np.testing.assert_allclose(a['point'],short[-1])
  far=straight(.01,.4,.7);b=pursuit_point(far,.25)
  self.assertEqual(b['mode'],'FAR_SUPPORT');self.assertLess(b['speed_scale'],1.)
  np.testing.assert_allclose(b['point'],far[0])
 def test_cannot_jump_to_another_forward_branch(self):
  p=np.array([[.08,0],[.12,.02],[-.1,.1],[.5,.1]])
  r=pursuit_point(p,.25)
  self.assertEqual(r['mode'],'SHORT_SUPPORT')
  self.assertLess(r['position_m'],.2);self.assertLess(r['point'][0],.13)
 def test_circle_target_is_left_right_symmetric(self):
  p=straight(.06);a=pursuit_point(p,.2);b=pursuit_point(p*[1,-1],.2)
  np.testing.assert_allclose(a['point']*[1,-1],b['point'])
 def control(self,speed,age):
  c=SteeringControl(RoadPolicy(),[0.,0.],[.075,.105,.085],.3)
  c.preview_speed=speed;c.preview_interval=.2
  p=straight(.01);q=nearest_forward_point(p)[0]
  c.command(dict(valid=True,state='OBSERVED',confidence=.9,control_target_m=q,control_path_m=p,
                 path=p,path_curvature_m_inv=0.),age_s=age)
  return c
 def test_speed_and_delay_raise_horizon_but_near_support_does_not_shrink_it(self):
  slow=self.control(.05,.1);fast=self.control(.2,.1);delayed=self.control(.2,.25)
  self.assertAlmostEqual(slow.status['lookahead_m'],.18)
  self.assertGreater(fast.status['lookahead_m'],slow.status['lookahead_m'])
  self.assertGreater(delayed.status['lookahead_m'],fast.status['lookahead_m'])
  self.assertGreater(slow.status['target_distance_m'],.17)
 def test_invalid_flow_retains_only_horizon_bound_not_pose_prediction(self):
  c=self.control(.2,.1);c.observe(dict(valid=False),1.)
  p=straight(.02);q=nearest_forward_point(p)[0]
  c.command(dict(valid=True,state='OBSERVED',confidence=.9,control_target_m=q,control_path_m=p,
                 path=p,path_curvature_m_inv=0.),age_s=.2)
  self.assertGreater(c.status['lookahead_m'],.18)
  self.assertFalse(c.status['latency_prediction']['applied'])
 def test_actual_saturation_is_removed_without_merely_clipping_curvature(self):
  rows=json.loads((ROOT/'tests/fixtures/curve_oscillation.json').read_text())
  row=next(r for r in rows if r['frame']==386);s=row['steering']
  c=SteeringControl(RoadPolicy(),[-.055,0.],None,.3)
  c.minimum_lookahead_m=.18
  c.status={k:v for k,v in s.items() if k in ('body_speed_m_s','body_yaw_rad_s','speed_upper_m_s','motion_sigma_m','motion_interval_s')}
  c.preview_speed=s['speed_upper_m_s'];c.preview_interval=s['motion_interval_s']
  command=c.command(row['target'],s['latency_prediction']['age_s'])
  self.assertIsNotNone(command)
  self.assertLess(s['requested_curvature_m_inv'],-15.)
  self.assertLess(abs(c.requested),4.);self.assertGreater(min(command),0)
  q=np.asarray(c.status['command_target_m'])
  self.assertAlmostEqual(c.requested,2*q[1]/float(q@q))

def sim(adaptive=True,speed=.15,curve=0.,period=.2,delay=.18,lag=.4):
 dt=.01;pose=np.array([0.,-.035,.08]);w=0.;wanted=0.;pending=[];record=[];next_sample=0.
 for i in range(1600):
  t=i*dt
  if t>=next_sample-1e-6:
   next_sample+=period
   angle=0. if curve==0 else math.atan2(curve*pose[0],1-curve*pose[1])
   if curve:angle+=2*math.pi*round((pose[2]-angle)/(2*math.pi))
   arc=np.linspace(pose[0]-.2,pose[0]+1.,300) if curve==0 else curve*np.linspace(angle/curve-.2,angle/curve+1.2,300)
   world=np.c_[arc,np.zeros(len(arc))] if curve==0 else np.c_[np.sin(arc)/curve,(1-np.cos(arc))/curve]
   c,s=math.cos(pose[2]),math.sin(pose[2]);p=(world-pose[:2])@np.array([[c,-s],[s,c]])
   # Same valid constant-twist receipt-to-command prediction as the application.
   theta=w*delay;c,s=math.cos(theta),math.sin(theta);R=np.array([[c,s],[-s,c]])
   delta=np.array([speed*delay,0]) if abs(w)<1e-8 else np.array([speed*math.sin(theta)/w,speed*(1-c)/w])
   p=(p-delta)@R.T
   if adaptive:
    radius=max(.18,2*speed*(lag+delay+period));chosen=pursuit_point(p,radius)
    q=None if chosen is None else chosen['point']
   else:
    chosen=nearest_forward_point(p);q=None if chosen is None else chosen[0]
   k=0 if q is None else float(2*q[1]/(q@q))
   pending.append((t+delay,speed*max(-1/.15,min(1/.15,k))))
  while pending and pending[0][0]<=t:wanted=pending.pop(0)[1]
  w+=(wanted-w)*dt/lag
  pose+=np.array([speed*math.cos(pose[2]),speed*math.sin(pose[2]),w])*dt
  error=pose[1] if curve==0 else math.hypot(pose[0],pose[1]-1/curve)-abs(1/curve)
  record.append([t,error,w])
 arr=np.asarray(record);tail=arr[arr[:,0]>8]
 return dict(rms=float(np.sqrt(np.mean(tail[:,1]**2))),max_error=float(np.max(abs(arr[:,1]))),yaw_reversals=int(np.sum(tail[:-1,2]*tail[1:,2]<0)))

class DelayedClosedLoopTests(unittest.TestCase):
 def test_bounded_tracking_on_left_right_and_straight_with_lag(self):
  for speed in (.1,.15,.2):
   for curvature in (0.,2.,-2.):
    with self.subTest(speed=speed,curvature=curvature):
     r=sim(True,speed,curvature)
     self.assertLess(r['rms'],.01)
     self.assertLess(r['max_error'],.05)
 def test_nearest_controller_reproduces_instability_on_same_delayed_plant(self):
  old=sim(False,.15,0.);new=sim(True,.15,0.)
  self.assertGreater(old['rms'],.05)
  self.assertLess(new['rms'],old['rms']/10.)

if __name__=='__main__':unittest.main()
