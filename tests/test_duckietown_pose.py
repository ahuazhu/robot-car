import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/duckietown_test'))
from lane_pose import estimate_lane_pose

def boundaries(d=0.,phi=0.,width=.30):
    result=[]
    for side,sign in [('left',1),('right',-1)]:
        c=sign*width/2-d
        for a,b in [(.22,.35),(.36,.49),(.50,.63)]:
            points=[[x,(c-math.sin(phi)*x)/math.cos(phi)] for x in (a,b)]
            result.append(dict(side=side,ground=points))
    return result

class PoseTests(unittest.TestCase):
    def test_offset_heading_and_endpoint_direction(self):
        for d,phi in [(0,0),(.05,.18),(-.06,-.22)]:
            lines=boundaries(d,phi)
            for line in lines[::2]:line['ground'].reverse()
            r=estimate_lane_pose(lines)
            self.assertTrue(r['valid'])
            self.assertAlmostEqual(r['d_m'],d,places=6)
            self.assertAlmostEqual(r['phi_rad'],phi,places=6)
    def test_missing_or_single_side_cannot_be_valid(self):
        for lines in ([],boundaries()[:3]):
            r=estimate_lane_pose(lines)
            self.assertFalse(r['valid']);self.assertIsNone(r['d_m'])
    def test_far_clutter_cannot_change_near_pose(self):
        lines=boundaries(.03,.12)
        lines += [dict(side='left',ground=[[.8,.01],[1.2,.3]]) for _ in range(100)]
        r=estimate_lane_pose(lines)
        self.assertTrue(r['valid']);self.assertAlmostEqual(r['d_m'],.03,places=6)
    def test_wrong_lane_width_not_accepted(self):
        self.assertFalse(estimate_lane_pose(boundaries(width=.10))['valid'])
    def test_curved_lane_target_follows_bend(self):
        for direction in (-1,1):
            lines=[]
            for side,sign in [('left',1),('right',-1)]:
                for a,b in [(.22,.30),(.30,.38),(.38,.46),(.46,.54),(.54,.62)]:
                    points=[]
                    for x in (a,b):
                        center=direction*(x-.2)**2
                        slope=direction*2*(x-.2)
                        points.append([x,center+sign*.15*math.sqrt(1+slope*slope)])
                    lines.append(dict(side=side,ground=points))
            result=estimate_lane_pose(lines)
            self.assertTrue(result['valid']);self.assertEqual(result.get('mode'),'CURVE')
            x,y=result['target_m']
            self.assertAlmostEqual(y,direction*(x-.2)**2,delta=.01)
            self.assertGreater(direction*y,0)
    def test_single_side_opt_in_normal_offset(self):
        for side,sign in [('left',1),('right',-1)]:
            for slope in (-.3,.3):
                intercept=sign*.15*math.sqrt(1+slope*slope)
                lines=[dict(side=side,ground=[[x,slope*x+intercept] for x in (a,a+.06)]) for a in (.22,.29,.36,.43,.50,.57)]
                r=estimate_lane_pose(lines,allow_single_side=True)
                self.assertTrue(r['valid']);self.assertEqual(r['single_side'],side)
                x,y=r['target_m'];self.assertAlmostEqual(y,slope*x,places=6)
    def test_single_side_rejects_short_or_far_evidence(self):
        for start,step in [(.22,.02),(.43,.04)]:
            lines=[dict(side='right',ground=[[x,-.15] for x in (start+i*step,start+(i+1)*step)]) for i in range(6)]
            self.assertFalse(estimate_lane_pose(lines,allow_single_side=True)['valid'])
    def test_single_side_does_not_override_wrong_width(self):
        self.assertFalse(estimate_lane_pose(boundaries(width=.10),allow_single_side=True)['valid'])
    def test_no_stale_estimate(self):
        self.assertTrue(estimate_lane_pose(boundaries())['valid'])
        self.assertFalse(estimate_lane_pose([])['valid'])
if __name__=='__main__':unittest.main()
