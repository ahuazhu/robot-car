import importlib.util
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
# Isolate from other versions' tracker.py in the shared unittest process.
spec=importlib.util.spec_from_file_location('curve_tracker',ROOT/'maixcam2/curve_follow/tracker.py')
tracker=importlib.util.module_from_spec(spec);spec.loader.exec_module(tracker)
from geometry import fit_curve,inward_center,merge_centers,choose_target,arc
from controller import target_command,motion_from_pwm
from ground_motion import apply_ground_motion,command_motion
from road_policy import RoadPolicy

from geometry import nearest_forward_point

CAL=json.loads((ROOT/'maixcam2/duckietown_test/calibration.json').read_text())


def straight(y=0.):
    return fit_curve(np.column_stack((np.linspace(.12,.65,80),np.full(80,y))))


def scene(side='both',offset=0.):
    yy,xx=np.indices((240,320))
    p=np.stack((xx/320,yy/240,np.ones_like(xx)),axis=-1)@np.array(CAL['H_normalized']).T
    x=p[:,:,0]/p[:,:,2];y=p[:,:,1]/p[:,:,2]-offset
    valid=(x>.08)&(x<1.)
    img=np.full((240,320,3),120,np.uint8)
    if side in ('both','left'):
        img[(y>.15)&valid]=(35,180,35);img[(y>=.15)&(y<.17)&valid]=240
    if side in ('both','right'):
        img[(y<-.15)&valid]=(35,180,35);img[(y<=-.15)&(y>-.17)&valid]=240
    return img


class GeometryTests(unittest.TestCase):
    def test_normal_offset_straight_both_sides(self):
        for side,y in [('left',.15),('right',-.15)]:
            center=inward_center(straight(y),side)
            self.assertIsNotNone(center)
            self.assertLess(abs(center['path'][:,1]).max(),1e-8)
    def test_circle_beyond_ninety_degrees_is_supported(self):
        theta=np.linspace(0,2.0,100);radius=.45
        edge=fit_curve(np.column_stack((.12+radius*np.sin(theta),radius*(1-np.cos(theta))-.15)))
        self.assertIsNotNone(edge)
        # x turns back; y(x) could not represent this portion.
        self.assertTrue((np.diff(edge['path'][:,0])<0).any())
        center=inward_center(edge,'right')
        self.assertIsNotNone(center)
        radii=np.linalg.norm(center['path']-[.12,.30],axis=1)
        np.testing.assert_allclose(radii,.30,atol=.01)
    def test_offset_cusp_rejected(self):
        theta=np.linspace(.1,2.,90);r=.10
        edge=fit_curve(np.column_stack((.20+r*np.sin(theta),r*(1-np.cos(theta)))))
        self.assertIsNone(inward_center(edge,'right'))
    def test_dual_curves_merge_and_wrong_width_rejected(self):
        self.assertIsNotNone(merge_centers(straight(.01),straight(-.01)))
        self.assertIsNone(merge_centers(straight(.10),straight(-.10)))
    def test_no_target_beyond_maximum_range_and_nearest_on_control_curve(self):
        self.assertIsNone(choose_target(straight()['path']+[.7,0]))
        target=choose_target(straight()['path'])
        expected,_=nearest_forward_point(target['reference_path'])
        np.testing.assert_allclose(target['point'],expected,atol=1e-10)
    def test_curvature_describes_road_without_changing_nearest_selection_rule(self):
        a=choose_target(straight()['path'])
        theta=np.linspace(0,1.5,80);r=.30
        p=np.column_stack((.10+r*np.sin(theta),r*(1-np.cos(theta))))
        b=choose_target(fit_curve(p)['path'])
        self.assertGreater(b['curvature'],a['curvature'])
        for target in (a,b):
            nearest,_=nearest_forward_point(target['reference_path'])
            np.testing.assert_allclose(target['point'],nearest,atol=1e-10)
    def test_recorded_right_bend_keeps_curve_and_uses_its_nearest_forward_point(self):
        cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
        image=cv2.imread(str(ROOT/'.maixpy/root-cause/preferred-mask-driving/capture-11.jpg'))[:240,:320]
        pixels=tracker.CurvePixels(cal);fits,_=pixels.extract(image,white_value_min=210)
        path=fits['left']['center']['path'];near=path[np.argmin(np.linalg.norm(path,axis=1))]
        self.assertGreater(near[1],.09)
        target=choose_target(path)
        self.assertLess(path[-1,1],0.) # Fitted bend is preserved, independently of selected target.
        nearest,_=nearest_forward_point(target['reference_path'])
        np.testing.assert_allclose(target['point'],nearest,atol=1e-10)
        self.assertAlmostEqual(target['distance_m'],np.linalg.norm(target['point']))
        mirror=choose_target(path*np.array([1.,-1.]))
        np.testing.assert_allclose(mirror['point'],target['point']*[1.,-1.],atol=1e-8)
    def test_target_is_nearest_on_supported_and_bounded_predicted_curve(self):
        p=fit_curve(np.column_stack((np.linspace(.08,.6,80),np.linspace(.04,.20,80))))['path']
        target=choose_target(p)
        nearest,_=nearest_forward_point(target['reference_path'])
        np.testing.assert_allclose(target['point'],nearest,atol=1e-10)
        self.assertAlmostEqual(np.linalg.norm(target['point']),target['distance_m'])
        self.assertLessEqual(target['reference_sigma_m'],.05)
    def test_near_support_crossing_nominal_circle_has_no_target_jump(self):
        paths=[np.column_stack((np.linspace(start,.65,80),np.zeros(80)))
               for start in (.299,.301)]
        a,b=[choose_target(path)['point'] for path in paths]
        # A two-millimetre crop change must not cause the old 12cm jump.
        self.assertLess(np.linalg.norm(b-a),.003)

    def test_short_supported_curve_predicts_near_approach_then_selects_nearest(self):
        path=np.column_stack((np.linspace(.1,.18,30),np.zeros(30)))
        target=choose_target(path)
        np.testing.assert_array_equal(target['reference_path'],path)
        np.testing.assert_allclose(target['reference_path'][-len(path):],path,atol=1e-10)
        expected,_=nearest_forward_point(target['reference_path'])
        np.testing.assert_allclose(target['point'],expected,atol=1e-10)

    def test_curve_agreement_ignores_crop_but_rejects_reverse_and_no_overlap(self):
        from geometry import curve_agreement
        p=straight()['path'];cropped=p[8:-2]
        match=curve_agreement(p,cropped)
        self.assertLess(match['distance_m'],1e-6)
        self.assertGreater(match['heading_cos'],.99)
        reversed_match=curve_agreement(p,cropped[::-1])
        self.assertLess(reversed_match['heading_cos'],-.99)
        self.assertIsNone(curve_agreement(p,p+[.8,0]))

    def test_noise_does_not_destroy_curve(self):
        rng=np.random.default_rng(42);p=straight()['path']
        q=fit_curve(p+rng.normal(0,.002,p.shape))
        self.assertIsNotNone(q);self.assertLess(abs(q['path'][:,1]).max(),.006)
    def test_invalid_geometry_rejected(self):
        self.assertIsNone(fit_curve([[0,0],[0,0],[0,0]]))
        self.assertIsNone(fit_curve([[0,0],[1,float('nan')],[2,0]]))


class VisionTests(unittest.TestCase):
    def test_white_inner_edge_offset_accuracy(self):
        for side in ('left','right','both'):
            for offset in (-.03,0.,.03):
                r,d=tracker.TargetTracker(CAL).update(scene(side,offset),0.)
                self.assertTrue(r['valid'],(side,offset,r))
                self.assertEqual(r['mode'],'TWO EDGES' if side=='both' else 'ONE '+side.upper())
                self.assertAlmostEqual(np.interp(.3,np.asarray(r['path'])[:,0],np.asarray(r['path'])[:,1]),offset,delta=.012)
    def test_current_pose_roi_preserves_visible_boundary(self):
        base=ROOT/'.maixpy/calibrations/20261023-fixed-pose'
        cal=json.loads((base/'calibration.json').read_text())
        img=cv2.imread(str(base/'clean-road.jpg'))
        r,_=tracker.TargetTracker(cal).update(img,0.)
        self.assertTrue(r['observed'],r)
        self.assertGreater(r['confidence'],.70)
        self.assertLess(abs(r['target_m'][1]),.02)
        self.assertGreaterEqual(r['control_target_m'][0],.06)
        self.assertEqual(r['target_selection'],'NEAREST_CURVE')

    def test_blank_and_horizontal_stop_line_do_not_make_lane(self):
        for white in (False,True):
            img=np.full((240,320,3),120,np.uint8)
            if white:cv2.line(img,(0,180),(319,180),(240,240,240),8)
            r,_=tracker.TargetTracker(CAL).update(img,0)
            self.assertFalse(r['valid'])
    def test_historical_bend_consistency(self):
        paths=sorted((ROOT/'.maixpy/runs/20261021-211932-duck_single_stationary/artifacts').glob('raw*.jpg'))
        self.assertEqual(len(paths),3)
        targets=[]
        for p in paths:
            r,_=tracker.TargetTracker(CAL).update(cv2.imread(str(p)),0)
            self.assertTrue(r['valid'],r);self.assertEqual(r['mode'],'ONE RIGHT')
            targets.append(r['target_m'])
        self.assertLess(np.ptp(targets,axis=0).max(),.02)
    def test_prediction_expires_without_refresh(self):
        t=tracker.TargetTracker(CAL);r,_=t.update(scene('right'),0)
        self.assertTrue(r['valid'])
        blank=np.zeros((240,320,3),np.uint8)
        r,_=t.update(blank,.1,.05,0);self.assertEqual(r['mode'],'PREDICT')
        r,_=t.update(blank,.7,.05,0);self.assertFalse(r['valid'])
        self.assertEqual(r['invalid_reason'],'ROAD SUPPORT EXPIRED')
    def test_live_one_side_continues(self):
        t=tracker.TargetTracker(CAL);img=scene('right')
        for now in (0,.1,.5,1.,3.,5.):
            r,_=t.update(img,now);self.assertTrue(r['observed'])
    def test_bend_reduces_speed_and_mirrors_steering(self):
        r=dict(valid=True,state='OBSERVED',confidence=1.,target_m=[.15,.15],mode='ONE RIGHT',path_curvature_m_inv=7.)
        l=target_command(r)
        right=target_command(dict(r,target_m=[.15,-.15]))
        straight_cmd=target_command(dict(r,target_m=[.3,0],path_curvature_m_inv=0.))
        self.assertEqual(l,right[::-1]);self.assertLess(l[0],l[1]);self.assertLess(sum(l),sum(straight_cmd))
        self.assertIsNone(target_command(dict(r,valid=False)))

class BoundaryHandoffTests(unittest.TestCase):
    def test_opposite_side_fresh_confirmations_complete_while_stationary(self):
        from controller import VisionRecovery
        t=tracker.TargetTracker(CAL)
        initial=np.column_stack((np.linspace(.12,.24,30),np.zeros(30)))
        replacement=np.column_stack((np.linspace(.34,.65,40),np.zeros(40)))
        observations=iter([dict(path=initial,mode='ONE LEFT',confidence=.82)]+
                          [dict(path=replacement,mode='ONE RIGHT',confidence=.82)]*4)
        t.pixels.measure=lambda *args:(next(observations),{})
        r,_=t.update(None,0);self.assertTrue(r['observed'])
        r,_=t.update(None,.70);self.assertFalse(r['observed'])
        r,_=t.update(None,.90)
        self.assertFalse(r['valid']) # The old path memory is still expired.
        self.assertEqual(r['state'],'CONFIRMING')
        self.assertIsNone(VisionRecovery().update(r,None,.9,170,280))
        self.assertIsNone(target_command(r)) # No blind driving.
        r,_=t.update(None,1.10);self.assertEqual(r['state'],'CONFIRMING')
        r,_=t.update(None,1.30)
        self.assertTrue(r['observed']);self.assertEqual(r['mode'],'ONE RIGHT')
        self.assertEqual(r['state'],'OBSERVED')
    def test_new_opposite_side_candidate_stops_old_prediction_immediately(self):
        t=tracker.TargetTracker(CAL)
        a=np.column_stack((np.linspace(.12,.24,30),np.zeros(30)))
        b=np.column_stack((np.linspace(.34,.65,40),np.zeros(40)))
        observations=iter([dict(path=a,mode='ONE LEFT',confidence=.82),
                           dict(path=b,mode='ONE RIGHT',confidence=.82)])
        t.pixels.measure=lambda *args:(next(observations),{})
        t.update(None,0)
        r,_=t.update(None,.1,.04,0)
        self.assertTrue(r['valid']) # Old memory has not expired.
        self.assertEqual(r['mode'],'PREDICT')
        self.assertEqual(r['state'],'CONFIRMING')
        self.assertEqual(r['pending_frames'],1) # Only one observation so far.
        self.assertIsNone(target_command(r)) # Command zero now.

    def test_same_side_and_dual_side_support_switches_stop_and_reacquire(self):
        from controller import VisionRecovery
        a=np.column_stack((np.linspace(.12,.24,30),np.zeros(30)))
        b=np.column_stack((np.linspace(.34,.65,40),np.zeros(40)))
        for next_mode,confidence in (('ONE LEFT',.82),('TWO EDGES',.69)):
            t=tracker.TargetTracker(CAL)
            observations=iter([dict(path=a,mode='ONE LEFT',confidence=.82)]+
                              [dict(path=b,mode=next_mode,confidence=confidence)]*4)
            t.pixels.measure=lambda *args:(next(observations),{})
            t.update(None,0)
            r,_=t.update(None,.1)
            self.assertTrue(r['valid']);self.assertEqual(r['state'],'CONFIRMING')
            self.assertIsNone(target_command(r))
            r,_=t.update(None,.3)
            self.assertIsNone(VisionRecovery().update(r,None,.3,160,280))
            r,_=t.update(None,.5)
            self.assertEqual(r['state'],'CONFIRMING')
            self.assertIsNone(VisionRecovery().update(r,None,.3,160,280))
            r,_=t.update(None,.7)
            self.assertTrue(r['observed']);self.assertEqual(r['mode'],next_mode)
            self.assertIsNotNone(target_command(r))



class BrightRoadFallbackTests(unittest.TestCase):
    def test_shadowed_exterior_edge_wins_over_interior_specular_highlight(self):
        cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
        raw=cv2.imread(str(ROOT/'tests/fixtures/curve_shadow_glare.jpg'))[:240,:320]
        pixels=tracker.CurvePixels(cal);measurement,details=pixels.measure(raw)
        self.assertIsNotNone(measurement)
        self.assertEqual(measurement['mode'],'ONE LEFT')
        self.assertGreater(measurement['confidence'],.65)
        self.assertTrue(details['exposure_normalized'])
        self.assertIsNone(details['fits']['right'])
        # Crosswalk keeps original camera luminance, not normalized contrast.
        bird=details['bird'];raw_value=cv2.cvtColor(bird,cv2.COLOR_BGR2HSV)[:,:,2]
        np.testing.assert_array_equal(details['value'],raw_value)

    def test_recorded_bend_keeps_visible_boundary_on_bright_gray_cloth(self):
        cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
        for name in ('current-not-ready.jpg','resume-driving/capture-11.jpg',
                     'resume-driving/capture-stopped.jpg'):
            image=cv2.imread(str(ROOT/'.maixpy/root-cause'/name))[:240,:320]
            pixels=tracker.CurvePixels(cal)
            regular,_=pixels.extract(image)
            self.assertTrue(all(f is None for f in regular.values()),name)
            r,_=tracker.TargetTracker(cal).update(image,0)
            self.assertTrue(r['observed'],(name,r))
            self.assertEqual(r['mode'],'ONE LEFT')
            self.assertGreater(r['confidence'],.65)
            self.assertEqual(r['white_value_min'],210)
            self.assertLess(np.asarray(r['path'])[-1,1],0.)
            target=choose_target(r['path']);nearest,_=nearest_forward_point(target['reference_path'])
            np.testing.assert_allclose(target['point'],nearest,atol=1e-10)
    def test_preferred_mask_keeps_full_bend_instead_of_short_fragment(self):
        cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
        image=cv2.imread(str(ROOT/'.maixpy/root-cause/bright-fix-driving/capture-05.jpg'))[:240,:320]
        pixels=tracker.CurvePixels(cal)
        permissive,_=pixels.extract(image,white_value_min=180)
        strict,_=pixels.extract(image,white_value_min=210)
        self.assertLess(permissive['left']['length'],.20)
        self.assertGreater(strict['left']['length'],.45)
        result,details=pixels.measure(image)
        self.assertIsNotNone(result)
        self.assertEqual(details['white_value_min'],210)
        self.assertGreater(result['confidence'],.75)
        # Preserve the full observed rightward continuation independently of
        # control preview length; a nearer physical target can precede the bend.
        self.assertLess(result['path'][-1,1],-.10)
        target=choose_target(result['path'])
        nearest,_=nearest_forward_point(target['reference_path'])
        np.testing.assert_allclose(target['point'],nearest,atol=1e-10)

    def test_fallback_still_rejects_blank_and_horizontal_stop_line(self):
        cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
        for value in (150,195,205):
            image=np.full((240,320,3),value,np.uint8)
            for stop_line in (False,True):
                sample=image.copy()
                if stop_line:cv2.line(sample,(0,180),(319,180),(245,245,245),8)
                r,_=tracker.TargetTracker(cal).update(sample,0)
                self.assertFalse(r['valid'],(value,stop_line,r))
    def test_contrast_recovery_reuses_projection_without_changing_geometry_rules(self):
        pixels=tracker.CurvePixels(CAL)
        image=np.zeros((240,320,3),np.uint8)
        calls=[];original=pixels.prepare
        def counted(bgr):
            calls.append(1)
            return original(bgr)
        pixels.prepare=counted
        measurement,details=pixels.measure(image)
        self.assertIsNone(measurement)
        self.assertEqual(calls,[1])

    def test_preferred_valid_mask_does_not_invoke_second_extraction(self):
        cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
        image=cv2.imread(str(ROOT/'.maixpy/calibrations/20261023-fixed-pose/clean-road.jpg'))
        pixels=tracker.CurvePixels(cal);calls=[];original=pixels.extract
        def counted(*args,**kwargs):
            calls.append(kwargs.get('white_value_min',180))
            return original(*args,**kwargs)
        pixels.extract=counted
        result,details=pixels.measure(image)
        self.assertIsNotNone(result);self.assertEqual(calls,[210])


class BoundaryEligibilityTests(unittest.TestCase):
    def test_recorded_weak_fragment_cannot_veto_usable_left_boundary(self):
        cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
        raw=cv2.imread(str(ROOT/'.maixpy/readiness-check-20261023-100130/capture.jpg'))[:240,:320]
        # Preserve the original 5mm fixture that produces the weak fragment.
        t=tracker.TargetTracker(dict(cal,vision_grid_step_m=.005))
        fits,_=t.pixels.extract(raw)
        self.assertGreater(.85*fits['left']['confidence'],.55)
        self.assertLess(.85*fits['right']['confidence'],.55)
        r,_=t.update(raw,0)
        self.assertTrue(r['observed'],r)
        # A weak selected fragment cannot veto the usable left road. The new
        # candidate association may find a different, coherent right fragment.
        self.assertIn(r['mode'],('ONE LEFT','TWO EDGES'))
        # Compare overlapping road geometry, not adaptive lookahead points:
        # retaining nearer support intentionally changes the selected point.
        from geometry import curve_agreement
        agreement=curve_agreement(fits['left']['center']['path'],np.asarray(r['path']))
        self.assertIsNotNone(agreement)
        self.assertLess(agreement['distance_m'],.055)
        self.assertGreater(agreement['heading_cos'],.9)
        if r['mode']=='ONE LEFT':
            self.assertEqual(r['boundary_fallback'],'IGNORED INELIGIBLE RIGHT')
    def test_current_sampling_retains_recorded_left_road_geometry(self):
        cal=json.loads((ROOT/'maixcam2/curve_follow/calibration.json').read_text())
        raw=cv2.imread(str(ROOT/'.maixpy/readiness-check-20261023-100130/capture.jpg'))[:240,:320]
        baseline=tracker.TargetTracker(dict(cal,vision_grid_step_m=.005))
        current=tracker.TargetTracker(cal)
        reference,_=baseline.update(raw,0)
        result,_=current.update(raw,0)
        self.assertTrue(result['observed'],result)
        from geometry import curve_agreement
        agreement=curve_agreement(np.asarray(reference['path']),np.asarray(result['path']))
        self.assertIsNotNone(agreement)
        self.assertLess(agreement['distance_m'],.02)
        self.assertGreater(agreement['heading_cos'],.95)
    def measure(self,left_conf,right_conf,right_offset):
        p=tracker.CurvePixels(CAL)
        fits={side:dict(side=side,confidence=conf,center=straight(offset)) for side,conf,offset in
              [('left',left_conf,0),('right',right_conf,right_offset)]}
        return p.measure_fits(fits,{})
    def test_two_eligible_contradictory_edges_are_still_rejected(self):
        result,details=self.measure(.95,.90,.15)
        self.assertIsNone(result);self.assertEqual(details['measurement_reason'],'EDGES DISAGREE')
    def test_two_weak_but_coherent_edges_can_still_support_joint_measurement(self):
        result,_=self.measure(.62,.62,0)
        self.assertIsNotNone(result);self.assertEqual(result['mode'],'TWO EDGES')
        self.assertGreaterEqual(result['confidence'],.55)
    def test_two_ineligible_contradictory_edges_remain_rejected(self):
        result,_=self.measure(.52,.52,.15);self.assertIsNone(result)


class MovingReacquisitionTests(unittest.TestCase):
    def make_tracker(self):
        t=tracker.TargetTracker(CAL)
        t.pixels=type('Pixels',(),{})()
        self.measurement=dict(path=straight()['path'],mode='ONE LEFT',confidence=.9)
        t.pixels.measure=lambda image,prior=None:(self.measurement,dict(counts={},measurement_reason='NO EDGE'))
        self.assertTrue(t.update(None,0)[0]['observed'])
        return t
    def test_two_strong_consistent_measurements_override_motion_estimate(self):
        t=self.make_tracker();self.measurement=dict(self.measurement,path=straight(.08)['path'])
        self.assertFalse(t.update(None,.1,.05,0)[0]['observed'])
        self.assertTrue(t.update(None,.2,.05,0)[0]['observed'])
    def test_one_frame_outlier_and_large_jump_remain_rejected(self):
        t=self.make_tracker();self.measurement=dict(self.measurement,path=straight(.20)['path'])
        self.assertFalse(t.update(None,.1,.05,0)[0]['observed'])
        self.assertFalse(t.update(None,.2,.05,0)[0]['observed'])
    def test_visible_support_changes_do_not_break_curve_confirmation(self):
        t=self.make_tracker()
        self.measurement=dict(self.measurement,path=straight(.08)['path'])
        first,_=t.update(None,.1,.05,0)
        self.assertFalse(first['observed'])
        # Same centre curve, but only its farther support is visible now.
        # The old pursuit-point gate rejected this >3.5cm target shift.
        clipped=fit_curve(np.column_stack((np.linspace(.36,.65,50),np.full(50,.08))))['path']
        first_point=choose_target(self.measurement['path'])['point']
        self.assertGreater(np.linalg.norm(choose_target(clipped)['point']-first_point),.2)
        # Control points now follow available support; road association must
        # remain independent of that legitimate crop-induced point change.
        self.measurement=dict(self.measurement,path=clipped)
        second,_=t.update(None,.2,.05,0)
        self.assertTrue(second['observed'],second)
        self.assertTrue(second['reacquired'])
    def test_moderate_confidence_requires_three_consistent_curves(self):
        t=self.make_tracker()
        self.measurement=dict(self.measurement,path=straight(.10)['path'],confidence=.70)
        for now in (.1,.2):self.assertFalse(t.update(None,now,.05,0)[0]['observed'])
        r,_=t.update(None,.3,.05,0)
        self.assertTrue(r['observed'],r);self.assertTrue(r['reacquired'])
    def test_confirmation_is_independent_of_boundary_provenance(self):
        for mode in ('ONE LEFT','ONE RIGHT','TWO EDGES','RENAMED DISPLAY LABEL'):
            with self.subTest(mode=mode):
                t=self.make_tracker()
                self.measurement=dict(self.measurement,path=straight(.08)['path'],mode=mode)
                first,_=t.update(None,.1,.05,0)
                self.assertEqual(first['state'],'CONFIRMING')
                self.assertIsNone(target_command(first))
                second,_=t.update(None,.2,.05,0)
                self.assertTrue(second['observed'],second)
                self.assertEqual(second['mode'],mode)
    def test_pending_curve_is_aligned_for_robot_motion(self):
        t=self.make_tracker();shifted=straight(.09)['path']
        self.measurement=dict(self.measurement,path=apply_ground_motion(shifted,command_motion(.05,.3,.1,RoadPolicy())))
        self.assertFalse(t.update(None,.1,.05,.3)[0]['observed'])
        self.measurement=dict(self.measurement,path=apply_ground_motion(shifted,command_motion(.05,.3,.2,RoadPolicy())))
        r,_=t.update(None,.2,.05,.3)
        self.assertTrue(r['observed'],r)
        self.assertLess(r['confirmation_alignment']['distance_m'],.001)
    def test_weak_frame_does_not_confirm_or_poison_following_strong_frames(self):
        t=self.make_tracker()
        self.measurement=dict(self.measurement,path=straight(.10)['path'],confidence=.60)
        r,_=t.update(None,.1,.05,0)
        self.assertFalse(r['observed']);self.assertEqual(r['pending_frames'],0)
        self.measurement=dict(self.measurement,confidence=.90)
        self.assertFalse(t.update(None,.2,.05,0)[0]['observed'])
        self.assertTrue(t.update(None,.3,.05,0)[0]['observed'])
    def test_changing_false_candidates_do_not_accumulate_confirmation(self):
        t=self.make_tracker()
        for i in range(1,9):
            self.measurement=dict(self.measurement,path=straight(.14 if i%2 else -.14)['path'])
            r,_=t.update(None,i*.1,.05,0)
            self.assertFalse(r['observed'],r)
        self.assertFalse(r['valid']);self.assertEqual(r['invalid_reason'],'ROAD SUPPORT EXPIRED')

    def test_missing_observation_breaks_consecutive_confirmation(self):
        t=self.make_tracker();shifted=dict(self.measurement,path=straight(.08)['path'])
        self.measurement=shifted;t.update(None,.1,.05,0)
        self.measurement=None;t.update(None,.2,.05,0)
        self.measurement=shifted
        self.assertFalse(t.update(None,.3,.05,0)[0]['observed'])
        self.assertTrue(t.update(None,.4,.05,0)[0]['observed'])


class SteeringTests(unittest.TestCase):
    def test_wheel_ratio_follows_target_curvature(self):
        r=dict(valid=True,state='OBSERVED',confidence=1.,target_m=[.30,.07],mode='ONE LEFT',path_curvature_m_inv=0.)
        l,rr=target_command(r)
        expected=2*.07/(.30**2+.07**2)
        actual=(rr-l)/(.15*(rr+l))
        self.assertAlmostEqual(actual,expected,delta=.10)
    def test_sharp_right_has_more_turn_and_less_forward_than_old_limit(self):
        r=dict(valid=True,state='OBSERVED',confidence=1.,target_m=[.158,-.176],mode='ONE LEFT',path_curvature_m_inv=7.)
        left,right=target_command(r)
        self.assertGreater(left-right,50)
        self.assertLess(left+right,86)
        self.assertGreaterEqual(right,0)
    def test_prediction_slows_without_straightening_curve(self):
        r=dict(valid=True,state='OBSERVED',confidence=1.,target_m=[.3,.07],mode='ONE LEFT',path_curvature_m_inv=0.)
        a=target_command(r);b=target_command(dict(r,state='PREDICTED',confidence=.5))
        self.assertLess(sum(b),sum(a))
        self.assertAlmostEqual((a[1]-a[0])/sum(a),(b[1]-b[0])/sum(b),delta=.02)
    def test_saturated_bend_stops_inner_wheels_after_ramp_and_compensation(self):
        from controller import CommandRamp,compensate_pwm
        result=dict(valid=True,state='OBSERVED',confidence=1.,target_m=[.104,-.059],mode='ONE LEFT',path_curvature_m_inv=7.)
        for sign in (-1,1):
            r=dict(result,target_m=[.104,sign*.059])
            command=target_command(r,steering_gain=1.2)
            ramp=CommandRamp()
            for i in range(20):actual=compensate_pwm(ramp.apply(command,i*.1))
            inner=0 if sign>0 else 1
            self.assertEqual(actual[inner],0)
            self.assertGreater(actual[1-inner],90)
            self.assertLessEqual(max(actual),120)
            self.assertGreaterEqual(min(actual),0)
    def test_steering_gain_increases_bend_ratio_without_changing_straight(self):
        r=dict(valid=True,state='OBSERVED',confidence=1.,target_m=[.30,.07],mode='ONE LEFT',path_curvature_m_inv=0.)
        a=target_command(r);b=target_command(r,steering_gain=1.2)
        self.assertGreater((b[1]-b[0])/sum(b),(a[1]-a[0])/sum(a))
        predicted=target_command(dict(r,state='PREDICTED',confidence=.5),steering_gain=1.2)
        self.assertLess(sum(predicted),sum(b))
        self.assertAlmostEqual((predicted[1]-predicted[0])/sum(predicted),(b[1]-b[0])/sum(b),delta=.03)
        straight=dict(r,target_m=[.3,0.])
        self.assertEqual(target_command(straight),target_command(straight,steering_gain=1.2))

    def test_motion_estimate_uses_matching_kinematics(self):
        cmd=(30,50);v,omega=motion_from_pwm(cmd)
        self.assertAlmostEqual(omega/v,(cmd[1]-cmd[0])/(.15*sum(cmd)))


class GeometryDrivenControlTests(unittest.TestCase):
    def result(self, **changes):
        r=dict(valid=True,state='OBSERVED',confidence=.85,target_m=[.3,.07],
               path_curvature_m_inv=0.,mode='ONE LEFT')
        r.update(changes)
        return r

    def test_identical_centre_and_quality_get_identical_wheels_for_all_sources(self):
        expected=target_command(self.result())
        for label in ('ONE LEFT','ONE RIGHT','TWO EDGES','PREDICT','arbitrary label'):
            self.assertEqual(target_command(self.result(mode=label)),expected)

    def test_confidence_slows_continuously_without_changing_steering_direction(self):
        commands=[target_command(self.result(confidence=q)) for q in (.95,.8,.6,.3)]
        self.assertTrue(all(sum(a)>sum(b) for a,b in zip(commands,commands[1:])))
        ratios=[(r-l)/(r+l) for l,r in commands]
        self.assertLess(np.ptp(ratios),.04)

    def test_bend_speed_depends_on_geometric_curvature_without_scene_thresholds(self):
        sums=[sum(target_command(self.result(path_curvature_m_inv=k)))
              for k in (2.,3.,4.,5.,6.)]
        self.assertTrue(all(a>b for a,b in zip(sums,sums[1:])),sums)
        for k in np.linspace(0.,10.,41):
            left=target_command(self.result(target_m=[.2,.1],path_curvature_m_inv=k))
            right=target_command(self.result(target_m=[.2,-.1],path_curvature_m_inv=k))
            self.assertEqual(left,right[::-1])
            self.assertGreaterEqual(min(left),0)
            self.assertLessEqual(max(left),60)

    def test_unconfirmed_or_invalid_data_cannot_command_motion(self):
        for state in ('CONFIRMING','LOST','UNKNOWN'):
            self.assertIsNone(target_command(self.result(state=state)))
        for point in ([0.,0.],[-.2,.1],[float('nan'),.1],[.3,float('inf')],None):
            self.assertIsNone(target_command(self.result(target_m=point)))
        self.assertIsNone(target_command(self.result(confidence=0)))
        self.assertIsNone(target_command(self.result(confidence=float('nan'))))
        missing=self.result();del missing['confidence']
        self.assertIsNone(target_command(missing))

    def test_predicted_curve_is_rotation_equivariant_but_target_respects_vehicle_forward(self):
        p=straight()['path'];original=choose_target(p)
        theta=.8;c,s=np.cos(theta),np.sin(theta);rotation=np.array([[c,-s],[s,c]])
        rotated=choose_target(p@rotation.T)
        np.testing.assert_allclose(rotated['reference_path'],original['reference_path']@rotation.T,atol=1e-8)
        expected,_=nearest_forward_point(rotated['reference_path'])
        np.testing.assert_allclose(rotated['point'],expected,atol=1e-8)
        self.assertGreaterEqual(rotated['point'][0],.06)

    def test_nearest_geometry_parameters_must_be_finite_and_positive(self):
        for config in (dict(forward_min_m=-1),dict(maximum_target_m=0),
                       dict(curve_analysis_window_m=0),dict(curve_analysis_window_m=float('nan'))):
            with self.assertRaises(ValueError):choose_target(straight()['path'],**config)


class DriveGateTests(unittest.TestCase):
    def test_unverified_pose_cannot_enable_drive(self):
        spec=importlib.util.spec_from_file_location('curve_controller',ROOT/'maixcam2/curve_follow/controller.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        gate=module.can_enable_drive
        config={'allow_drive':True,'preview_only':False}
        cal={'geometry_confirmed':True,'camera_pose_verified':True}
        self.assertFalse(gate({},config))
        self.assertFalse(gate(dict(cal,camera_pose_verified=False),config))
        self.assertFalse(gate(cal,{}))
        self.assertFalse(gate(cal,config,True))
        self.assertTrue(gate(cal,config))
        # Strings such as "false" must not accidentally become truthy approvals.
        self.assertFalse(gate(dict(cal,camera_pose_verified='false'),config))

if __name__=='__main__':unittest.main()
