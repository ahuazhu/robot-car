"""Batch 28 corners without changing future poses, coverage or stop envelopes."""
import sys,json,math,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'));sys.path.insert(0,str(ROOT/'tests/fixtures'))
from steering_control import SteeringControl
from road_policy import RoadPolicy
from curve_envelope_reference import envelope_reference
class EnvelopeBatchTests(unittest.TestCase):
 def test_random_curves_partial_support_visibility_and_command_time_transform(self):
  rng=np.random.default_rng(178);cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
  for i in range(120):
   x=np.linspace(-.15,.7,80);k=rng.uniform(-4,4);path=np.c_[x,.5*k*x*x+rng.uniform(-.08,.08)]
   c=SteeringControl(RoadPolicy(),[-.055,0.],[.075,.105,.085],.3,calibration=cal if i%2 else None)
   theta=rng.uniform(-.1,.1);co,si=math.cos(theta),math.sin(theta)
   transform=(np.array([[co,si],[-si,co]]),np.array([-.02,.01])) if i%3==0 else None
   args=(path,k,.01,.3);kwargs=dict(observed_path=path[i%25:],speed_m_s=.15,sensor_transform=transform,lookahead_m=.24)
   a=c.envelope(*args,**kwargs);b=envelope_reference(c,*args,**kwargs)
   self.assertEqual(a.keys(),b.keys())
   for key in a:
    if isinstance(a[key],(str,type(None))):self.assertEqual(a[key],b[key])
    else:self.assertAlmostEqual(a[key],b[key],delta=1e-10)
