import unittest,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/duckietown_drive'))
from controller import drive_command,CommandRamp,SingleSideGuard

def pose(d=0,phi=0,**other):return dict(valid=True,d_m=d,phi_rad=phi,width_m=.30,**other)
class DriveTests(unittest.TestCase):
    def test_centered_straight_has_equal_forward_wheels(self):
        l,r=drive_command(pose());self.assertEqual(l,r);self.assertGreater(l,0)
    def test_car_left_of_center_turns_right(self):
        l,r=drive_command(pose(.05));self.assertGreater(l,r)
    def test_car_right_of_center_turns_left(self):
        l,r=drive_command(pose(-.05));self.assertLess(l,r)
    def test_heading_sign_corrects_toward_lane(self):
        l,r=drive_command(pose(0,.2));self.assertGreater(l,r)
        l,r=drive_command(pose(0,-.2));self.assertLess(l,r)
    def test_missing_invalid_or_outside_stops(self):
        for p in (None,dict(valid=False),pose(.11),pose(0,.61),pose(float('nan')),pose(0,float('inf'))):
            self.assertIsNone(drive_command(p))
    def test_limits_across_valid_range(self):
        for d in [-.1,-.05,0,.05,.1]:
            for phi in [-.6,-.3,0,.3,.6]:
                l,r=drive_command(pose(d,phi))
                self.assertGreaterEqual(min(l,r),42);self.assertLessEqual(max(l,r),66)
    def test_curve_target_controls_turn_instead_of_tangent_intercept(self):
        p=pose(.08,-.1,mode='CURVE',near_center_m=.03,target_m=[.45,.08])
        l,r=drive_command(p);self.assertLess(l,r)
        p['target_m']=[.45,-.08];l,r=drive_command(p);self.assertGreater(l,r)
    def test_startup_preserves_target_turn_ratio(self):
        ramp=CommandRamp()
        for i in range(8):
            l,r=ramp.apply((50,62),i*.05)
            self.assertLessEqual(abs(r-l),12*sum((l,r))/112+1)
    def test_single_side_requires_recent_pair_and_continuity(self):
        g=SingleSideGuard();p=pose(mode='CURVE',near_center_m=0,target_m=[.4,.02],single_side='right')
        self.assertFalse(g.accept(p,0))
        pair=dict(p);pair.pop('single_side')
        self.assertTrue(g.accept(pair,1));self.assertTrue(g.accept(p,1.2))
        self.assertFalse(g.accept(dict(p,target_m=[.4,.10]),1.3))
        self.assertFalse(g.accept(p,2.6))
    def test_single_side_lowers_forward_speed(self):
        p=pose(mode='CURVE',near_center_m=0,target_m=[.4,.02])
        self.assertLess(sum(drive_command(dict(p,single_side='right'))),sum(drive_command(p)))
    def test_stop_bypasses_ramp(self):
        ramp=CommandRamp()
        for i in range(10):ramp.apply((60,60),i*.05)
        self.assertEqual(ramp.apply(None,.51),(0,0))
        self.assertEqual(ramp.apply((0,0),.52),(0,0))
if __name__=='__main__':unittest.main()
