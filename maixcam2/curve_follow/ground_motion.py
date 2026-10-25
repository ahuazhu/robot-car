"""Sparse ground motion in metric coordinates; no global map or SLAM.

Transform is OLD points -> CURRENT points: q = p @ rotation.T + translation.
An image receive timestamp is not an exposure timestamp; uncertainty therefore
includes model error and this estimator reports its input interval explicitly.
"""
import math
import cv2
import numpy as np
from road_policy import RoadPolicy


def command_motion(v, omega, dt, policy):
    theta = omega*dt
    dx = v*dt if abs(omega) < 1e-6 else v*math.sin(theta)/omega
    dy = 0. if abs(omega) < 1e-6 else v*(1-math.cos(theta))/omega
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c,s],[-s,c]])
    translation = -np.array([dx,dy]) @ rotation.T
    # Unknown actuation scale/coasting: PWM is only a weak prior.
    sigma = policy.prediction_noise_m_s*dt + .5*abs(v)*dt
    return dict(rotation=rotation,translation=translation,sigma_m=sigma,
                valid=False,source='COMMAND PRIOR',reason='NO VISUAL MOTION',
                interval_s=dt,travel_m=abs(v)*dt,inliers=0,residual_m=None,
                speed_m_s=None,yaw_rad_s=None)


def apply_ground_motion(points, motion):
    return np.asarray(points) @ motion['rotation'].T + motion['translation']


class GroundMotion:
    def __init__(self, calibration, policy=None):
        self.policy = policy or RoadPolicy()
        self.H = np.asarray(calibration['H_normalized'], float)
        self.roi_top = int(calibration.get('vision_roi_top_px',100))
        self.previous = self.previous_mask = self.previous_time = None
        self.ground_visible = None

    def project(self, uv):
        p = np.column_stack((np.asarray(uv)/[320.,240.],np.ones(len(uv)))) @ self.H.T
        with np.errstate(divide='ignore',invalid='ignore'):
            return p[:,:2]/p[:,2,None]

    def road_mask(self, bgr):
        h,s,v = cv2.split(cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV))
        # Ground candidates on gray/white road; do not track colored signs.
        mask = ((s < 120) & (v > 30)).astype(np.uint8)*255
        if self.ground_visible is not None:
            mask[~self.ground_visible]=0
            return mask
        yy,xx = np.indices(mask.shape)
        ground = self.project(np.column_stack((xx.ravel(),yy.ravel()))).reshape(240,320,2)
        good = (np.isfinite(ground).all(axis=2) & (ground[:,:,0] >= .08)
                & (ground[:,:,0] <= .80) & (abs(ground[:,:,1]) <= .45)
                & (yy >= self.roi_top) & (yy < 236))
        self.ground_visible=good
        mask[~good] = 0
        return mask

    def update(self, bgr, now, v=0., omega=0.):
        p = self.policy
        dt = 0. if self.previous_time is None else max(0.,now-self.previous_time)
        fallback = command_motion(v,omega,dt,p)
        if bgr is None:
            self.previous = self.previous_mask = None
            self.previous_time = now
            return fallback
        gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        mask = self.road_mask(bgr)
        previous, old_mask = self.previous, self.previous_mask
        self.previous, self.previous_mask, self.previous_time = gray,mask,now
        if previous is None or not 0 < dt <= p.motion_max_interval_s:
            fallback['reason']='FIRST FRAME' if previous is None else 'FRAME INTERVAL'
            return fallback
        features = cv2.goodFeaturesToTrack(previous,p.motion_max_features,.015,7,mask=old_mask,blockSize=5)
        if features is None or len(features) < p.motion_min_features:
            fallback['reason']='INSUFFICIENT FEATURES'
            return fallback
        current, status, _ = cv2.calcOpticalFlowPyrLK(previous,gray,features,None,winSize=(15,15),maxLevel=2)
        if current is None:
            return fallback
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray,previous,current,None,winSize=(15,15),maxLevel=2)
        if back is None:
            return fallback
        uv0,uv1 = features[:,0],current[:,0]
        rounded=np.rint(np.clip(uv1,[0,0],[319,239])).astype(int)
        ok = (status.ravel()!=0)&(back_status.ravel()!=0)
        ok &= np.linalg.norm(back[:,0]-uv0,axis=1) <= p.motion_fb_error_px
        ok &= (uv1[:,0]>=0)&(uv1[:,0]<320)&(uv1[:,1]>=0)&(uv1[:,1]<240)
        ok &= mask[rounded[:,1],rounded[:,0]] != 0
        old,new = self.project(uv0[ok]),self.project(uv1[ok])
        finite = np.isfinite(old).all(axis=1)&np.isfinite(new).all(axis=1)
        old,new=old[finite],new[finite]
        if len(old) < p.motion_min_features:
            fallback['reason']='FLOW CHECK FAILED'
            return fallback
        affine,inliers = cv2.estimateAffinePartial2D(old,new,method=cv2.RANSAC,
                ransacReprojThreshold=p.motion_inlier_m,maxIters=100,confidence=.99,refineIters=5)
        if affine is None:
            fallback['reason']='MOTION FIT FAILED'
            return fallback
        keep = inliers.ravel().astype(bool)
        if keep.sum()<p.motion_min_features or keep.mean()<p.motion_min_inlier_fraction:
            fallback['reason']='INSUFFICIENT INLIERS'
            return fallback
        # Reject scale changes (non-planarity/mount movement), then refit rigid
        # SE(2), never apply the similarity transform's scale to road geometry.
        scale = float(np.linalg.norm(affine[:,0]))
        a,b = old[keep],new[keep]
        if not .95 <= scale <= 1.05 or min(np.ptp(a,axis=0)) < p.motion_min_span_m:
            fallback['reason']='GROUND GEOMETRY DEGENERATE'
            return fallback
        ca,cb=a.mean(axis=0),b.mean(axis=0)
        u,_,vt=np.linalg.svd((a-ca).T@(b-cb))
        rotation=vt.T@u.T
        if np.linalg.det(rotation)<0:
            vt[-1]*=-1;rotation=vt.T@u.T
        translation=cb-ca@rotation.T
        residual=np.linalg.norm(a@rotation.T+translation-b,axis=1)
        error=float(np.quantile(residual,.8))
        yaw=math.atan2(rotation[1,0],rotation[0,0])
        travel=float(np.linalg.norm(translation))
        if (error>p.motion_max_residual_m or travel>p.motion_max_speed_m_s*dt+p.motion_inlier_m
                or abs(yaw)>p.motion_max_yaw_rad_s*dt):
            fallback['reason']='MOTION LIMIT OR RESIDUAL'
            return fallback
        # Camera-frame motion is useful even with unmeasured body lever arm.
        # Do not label translation as measured wheel speed.
        return dict(rotation=rotation,translation=translation,
                    sigma_m=max(.0015,error)+.002*dt,valid=True,source='GROUND FLOW',reason='',
                    interval_s=dt,travel_m=travel,inliers=int(keep.sum()),residual_m=error,
                    speed_m_s=travel/dt,yaw_rad_s=-yaw/dt)


def motion_summary(motion):
    return {key:(round(value,5) if isinstance(value,float) else value)
            for key,value in motion.items() if key not in ('rotation','translation')}
