"""Duckietown detection experiment; no motor commands.

Actual upstream LineDetector / GroundProjector; miniAuto-specific mask/side
classification below. Four-point raw-image projection is approximate until
lens distortion is calibrated. It is diagnostic, never an automatic drive gate.
"""
import json
import math
from types import SimpleNamespace
from pathlib import Path
import cv2
import numpy as np
from duckietown_core import LineDetector, ColorRange, GroundProjector, NormalizedImagePoint

from lane_pose import estimate_lane_pose

WIDTH, HEIGHT = 320, 240
ROAD_WIDTH_M = 0.30


class LaneVision:
    def __init__(self, calibration=None, detection_scale=1.0, allow_single_side=False):
        self.allow_single_side=allow_single_side
        self.detection_scale=detection_scale
        self.detector = LineDetector(canny_thresholds=[40,100], hough_threshold=max(2,round(6*detection_scale)),
                                    hough_min_line_length=max(3,round(6*detection_scale)), hough_max_line_gap=max(1,round(4*detection_scale)))
        self.white = ColorRange.fromDict({'low':[0,0,110], 'high':[179,100,255]})
        self.calibration = calibration
        self.projector = None
        if calibration:
            self.projector = GroundProjector(SimpleNamespace(H=np.asarray(calibration['H_normalized'],float)))

    def detect(self, bgr):
        if bgr.shape[:2] != (HEIGHT, WIDTH):
            raise ValueError('Expected 320x240 image; calibration must match resolution')
        # Duckietown extracts short colored edge segments, rather than requiring
        # a full-width road interval in every scan row.
        crop_y=HEIGHT//2
        roi=bgr[crop_y:]
        if self.detection_scale != 1:
            roi=cv2.resize(roi,None,fx=self.detection_scale,fy=self.detection_scale,interpolation=cv2.INTER_AREA)
        detected=self.detector.detect(roi,[self.white])[0]
        hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
        green=cv2.inRange(hsv,np.array([32,55,30]),np.array([95,255,255]))
        lines=[]; rejected=[]
        raw_lines=np.asarray(detected.lines,float).reshape((-1,4))/self.detection_scale
        if len(raw_lines):
            raw_lines[:,[1,3]]+=crop_y
            deltas=raw_lines[:,2:]-raw_lines[:,:2]
            lengths=np.linalg.norm(deltas,axis=1)
            eligible=(lengths>=6)&(np.abs(deltas[:,1])>=lengths*.15)
            rejected.extend(raw_lines[~eligible].tolist())
            raw_lines=raw_lines[eligible];deltas=deltas[eligible];lengths=lengths[eligible]
        if len(raw_lines):
            t=np.linspace(.15,.85,7)
            centers=raw_lines[:,:2,None]+deltas[:,:,None]*t
            normals=np.stack((-deltas[:,1],deltas[:,0]),axis=1)/lengths[:,None]
            scores=[]
            for sign in (-1,1):
                samples=centers[:,:,:,None]+sign*normals[:,:,None,None]*np.array([5,9,14])
                xx=np.rint(samples[:,0]).astype(int);yy=np.rint(samples[:,1]).astype(int)
                valid=(xx>=0)&(xx<WIDTH)&(yy>=crop_y)&(yy<HEIGHT)
                green_samples=green[np.clip(yy,0,HEIGHT-1),np.clip(xx,0,WIDTH-1)]>0
                scores.append(np.sum(green_samples&valid,axis=(1,2))/np.maximum(1,valid.sum(axis=(1,2))))
            score_pairs=np.array(scores).T
        else:
            score_pairs=np.empty((0,2));normals=np.empty((0,2))
        for raw,scores,normal in zip(raw_lines,score_pairs,normals):
            x1,y1,x2,y2=raw.tolist()
            nx,ny=normal
            if max(scores)<0.35 or abs(scores[0]-scores[1])<0.20:
                rejected.append([x1,y1,x2,y2]);continue
            # Outward normal points into green. Positive image x = right edge.
            sign=(-1,1)[int(scores[1]>scores[0])]
            side='right' if sign*nx>0 else 'left'
            line={'pixels':[x1,y1,x2,y2], 'side':side, 'green_support':float(max(scores))}
            if self.projector:
                ground=[]
                for x,y in ((x1,y1),(x2,y2)):
                    p=self.projector.vector2ground(NormalizedImagePoint(x/WIDTH,y/HEIGHT))
                    ground.append([float(p.x),float(p.y)])
                if np.isfinite(ground).all() and all(0.05<p[0]<1.5 and abs(p[1])<1 for p in ground):
                    line['ground']=ground
            lines.append(line)
        counts={side:sum(v['side']==side for v in lines) for side in ('left','right')}
        pose=estimate_lane_pose(lines, self.calibration.get('road_width_m',ROAD_WIDTH_M),allow_single_side=self.allow_single_side) if self.calibration else None
        return dict(lines=lines,rejected=rejected,counts=counts,pose=pose,
                    status='SEGMENTS ONLY - NO CALIBRATION' if not self.projector else ('POSE PREVIEW - MOTORS OFF' if pose['valid'] else 'POSE UNAVAILABLE - MOTORS OFF'),
                    calibrated=self.projector is not None, motor_enabled=False)


def make_calibration(pixels, near_m, far_m, width_m=ROAD_WIDTH_M):
    """Points: near-left, near-right, far-right, far-left; metres forward/left.
    Vehicle centered and aligned on a straight road; camera rigidly fixed.
    Origin is camera's ground footprint, NOT wheel axle (no drive use).
    """
    pts=np.asarray(pixels,np.float32)
    if pts.shape!=(4,2) or not np.isfinite(pts).all():raise ValueError('Four finite image points required')
    if not (0.05<=near_m<far_m<=2 and far_m-near_m>=0.1 and 0.1<=width_m<=1):raise ValueError('Invalid measured dimensions')
    if (pts[:,0]<0).any() or (pts[:,0]>=WIDTH).any() or (pts[:,1]<0).any() or (pts[:,1]>=HEIGHT).any():raise ValueError('Points outside image')
    if not (pts[0,0]<pts[1,0] and pts[3,0]<pts[2,0] and min(pts[:2,1])>max(pts[2:,1])+5):raise ValueError('Incorrect near/far corner order')
    if not cv2.isContourConvex(pts.astype(np.int32)) or abs(cv2.contourArea(pts))<200:raise ValueError('Degenerate calibration quadrilateral')
    dst=np.array([[near_m,width_m/2],[near_m,-width_m/2],[far_m,-width_m/2],[far_m,width_m/2]],np.float32)
    H=cv2.getPerspectiveTransform(pts/np.array([WIDTH,HEIGHT],np.float32),dst)
    if not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-10:raise ValueError('Invalid homography')
    return dict(H_normalized=H.tolist(),image_size=[WIDTH,HEIGHT],pixels=pts.tolist(),near_m=near_m,far_m=far_m,road_width_m=width_m,origin='camera ground footprint',lens_rectified=False,drive_validated=False)


def annotate(bgr,result):
    out=bgr.copy()
    for line in result['lines']:
        p=np.rint(line['pixels']).astype(int)
        color=(0,220,255) if line['side']=='left' else (255,180,0)
        cv2.line(out,tuple(p[:2]),tuple(p[2:]),color,2)
    cv2.rectangle(out,(0,0),(319,37),(0,0,0),-1)
    cv2.putText(out,'DUCKIETOWN SEGMENTS | MOTORS OFF',(4,14),0,.39,(255,255,255),1)
    cv2.putText(out,'left=%d right=%d | %s'%(result['counts']['left'],result['counts']['right'],'PROJECTED' if result['calibrated'] else 'NEED CALIBRATION'),(4,30),0,.36,(0,255,255),1)
    pose=result.get('pose')
    if pose and pose['valid']:
        cv2.putText(out,'d=%+.1fcm phi=%+.1fdeg'%(100*pose['d_m'],math.degrees(pose['phi_rad'])),(4,51),0,.40,(0,255,0),1)
    return out


def bird_view(result):
    out=np.zeros((240,320,3),np.uint8)
    for y in range(20,240,30):cv2.line(out,(0,y),(319,y),(45,45,45),1)
    cv2.line(out,(160,0),(160,239),(90,90,90),1)
    pose=result.get('pose')
    for index,line in enumerate(result['lines']):
        if 'ground' not in line:continue
        pts=[(round(160-p[1]*350),round(240-p[0]*220)) for p in line['ground']]
        color=(0,220,255) if line['side']=='left' else (255,180,0)
        selected=pose and pose['valid'] and index in pose['inliers']
        cv2.line(out,pts[0],pts[1],color if selected else (65,65,65),2 if selected else 1)
    if pose and pose['valid']:
        d=pose['d_m'];phi=pose['phi_rad']
        if 'path_coefficients' in pose:
            c=pose['path_coefficients'];ground=[(x,c[0]+c[1]*x+c[2]*x*x) for x in np.linspace(*pose['path_range_m'],20)]
        else:
            ground=[(x,(-d-math.sin(phi)*x)/math.cos(phi)) for x in (.20,.65)]
        pts=[(round(160-y*350),round(240-x*220)) for x,y in ground]
        cv2.polylines(out,[np.array(pts,np.int32)],False,(0,255,0),2)
        cv2.putText(out,'NEAR LANE POSE',(4,14),0,.45,(0,255,0),1)
    else:
        cv2.putText(out,'NO VALID POSE',(4,14),0,.45,(0,180,255),1)
    if not result['calibrated']:cv2.putText(out,'CALIBRATION REQUIRED',(20,120),0,.6,(0,180,255),1)
    return out
