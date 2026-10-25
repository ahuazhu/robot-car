import math
from statistics import median
import cv2
import numpy as np
def detect_reference(self, details, road, now):
    p=self.policy;grid=self.pixels
    bird=details.get('bird');path=np.asarray(road.get('path',[]),float)
    if bird is None or len(path)<2 or road.get('confidence',1.)<.55: return None
    # A road control target may start beyond a close crossing. Extend only
    # the semantic search corridor along the measured near tangent; this
    # does not create lane support or authorize driving into an unseen area.
    direction=path[1]-path[0];length=np.linalg.norm(direction)
    if length>1e-6:
        path=np.vstack((path[0]-self.pixels.width*direction/length,path))
    tangent=np.gradient(path,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-8)
    normals=np.c_[-tangent[:,1],tangent[:,0]]
    half=grid.width/2-p.road_margin_m
    polygon=np.vstack((path+half*normals,(path-half*normals)[::-1]))
    uv=np.rint(np.c_[(grid.ys[0]-polygon[:,1])/grid.step,(polygon[:,0]-grid.xs[0])/grid.step]).astype(np.int32)
    corridor=np.zeros(bird.shape[:2],np.uint8);cv2.fillPoly(corridor,[uv],255)
    if 'saturation' in details:
        saturation,value=details['saturation'],details['value']
    else:
        hsv=cv2.cvtColor(bird,cv2.COLOR_BGR2HSV);saturation,value=hsv[:,:,1],hsv[:,:,2]
    # Same calibrated white levels as lane extraction; no route-specific colors.
    best=None
    for level in grid.white_value_levels:
        mask=((saturation<=105)&(value>=level)&grid.visible&(corridor!=0)).astype(np.uint8)*255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        bars=[]
        for contour in contours:
            if len(contour)<4 or cv2.contourArea(contour)<3: continue
            rc=contour[:,0,:]
            points=np.c_[grid.xs[rc[:,1]],grid.ys[rc[:,0]]].astype(np.float32)
            rect=cv2.minAreaRect(points);box=cv2.boxPoints(rect).astype(float)
            sides=np.roll(box,-1,axis=0)-box;lengths=np.linalg.norm(sides,axis=1)
            major=int(np.argmax(lengths));long=float(lengths[major]);short=float(np.min(lengths))
            if not (p.bar_length_min_m<=long<=p.bar_length_max_m and
                    p.bar_width_min_m<=short<=p.bar_width_max_m and long>=2.3*short): continue
            center=box.mean(axis=0);idx=int(np.argmin(np.linalg.norm(path-center,axis=1)))
            direction=sides[major]/long
            if np.dot(direction,tangent[idx])<0: direction=-direction
            if np.dot(direction,tangent[idx])<.85: continue
            bars.append(dict(center=center,direction=direction,length=long,width=short,box=box))
        centers=np.asarray([b['center'] for b in bars]);directions=np.asarray([b['direction'] for b in bars])
        evaluated=set()
        for anchor in bars:
            direction=anchor['direction'];normal=np.array([-direction[1],direction[0]])
            ids=np.flatnonzero((abs((centers-anchor['center'])@direction)<=p.row_alignment_m)&(directions@direction>=.94))
            key=tuple(ids.tolist())
            if key in evaluated:continue
            evaluated.add(key)
            group=[bars[int(i)] for i in ids]
            if len(group)<p.minimum_bars: continue
            group.sort(key=lambda b:np.dot(b['center'],normal))
            lateral=np.array([np.dot(b['center'],normal) for b in group]);spacing=np.diff(lateral)
            pitch=np.rint(spacing/p.bar_pitch_m)
            good=(pitch>=1)&(pitch<=2)&(abs(spacing-pitch*p.bar_pitch_m)<=p.pitch_tolerance_m)
            # Split nonuniform clusters rather than joining unrelated reflections.
            chunks=np.split(np.arange(len(group)),np.flatnonzero(~good)+1)
            for ids in chunks:
                if len(ids)<p.minimum_bars: continue
                selected=[group[int(i)] for i in ids]
                t=np.mean([b['direction'] for b in selected],axis=0);t/=np.linalg.norm(t)
                if t[0]<.35: continue
                corners=np.vstack([b['box'] for b in selected]);near=float(np.min(corners@t));far=float(np.max(corners@t))
                length=float(median([b['length'] for b in selected]));width=float(median([b['width'] for b in selected]))
                # Require the rule's bar/gap scale as a repeated pattern.
                quality=min(1.,len(selected)/7)*max(0.,1-abs(length-.10)/.12)
                candidate=dict(normal=t.tolist(),near_m=near,far_m=far,observed_at=now,
                               sigma_m=p.distance_error_m+grid.step,bar_count=len(selected),
                               confidence=float(quality),bar_length_m=length,bar_width_m=width,
                               corners=corners.tolist())
                if best is None or candidate['confidence']>best['confidence']: best=candidate
        # A complete seven-bar observation needs no second photometric pass.
        if best and best['bar_count']==7 and best['confidence']>=.85:return best
    return best
