import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/curve_follow'))
from road_policy import RoadPolicy
from steering_control import SteeringControl
class FeasibleTargetTests(unittest.TestCase):
 def scene(self):
  c=SteeringControl(RoadPolicy(),[0.,0.],[.075,.105,.085],.3)
  p=np.c_[np.linspace(.08,.6,50),np.linspace(.0,.08,50)]
  return c,dict(valid=True,state='OBSERVED',confidence=.9,control_target_m=p[0],control_path_m=p,path=p,path_observed_mask=[True]*len(p),path_curvature_m_inv=0.,road_model=dict(sigma_m=.005))
 def test_other_supported_goal_is_tried_before_zero_or_search(self):
  c,r=self.scene()
  blocked=dict(speed_scale=0.,first_unsafe_m=0.)
  clear=dict(speed_scale=1.,first_unsafe_m=None)
  with patch.object(c,'envelope',side_effect=[blocked,clear]) as checked:
   command=c.command(r)
  self.assertIsNotNone(command);self.assertTrue(any(command));self.assertEqual(checked.call_count,2)
  self.assertEqual(c.status['target_selection'],'FEASIBLE_PURSUIT')
 def test_no_goal_can_override_an_unsafe_body_or_unconfirmed_geometry(self):
  c,r=self.scene()
  with patch.object(c,'envelope',return_value=dict(speed_scale=0.,first_unsafe_m=0.)):
   self.assertIsNone(c.command(r))
  r['state']='CONFIRMING'
  with patch.object(c,'envelope') as checked:
   self.assertIsNone(c.command(r));checked.assert_not_called()
if __name__=='__main__':unittest.main()
