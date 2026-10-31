"""Full-frame lamp search, learned appearance and one-shot starting gate.

The source can clip to white. Candidate locations come from the entire image;
matching its calibrated emitting appearance rejects ordinary red clothing and
bright sky. Pixel coordinates in telemetry are detections, never a fixed ROI.
"""
import math
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np


class StartLight:
    def __init__(self, config=None, template=None):
        c=config or {}
        self.enabled=c.get('enabled',False)
        if not isinstance(self.enabled,bool):raise ValueError('start_light.enabled must be boolean')
        self.armed=self.enabled and c.get('auto_arm_on_launch',False)
        if not isinstance(c.get('auto_arm_on_launch',False),bool):raise ValueError('Invalid start-light auto arm')
        self.fired=False;self.phase='WAIT_LIGHT' if self.armed else 'DISARMED'
        self.since=None;self.last=None;self.elapsed=0.;self.observation=dict(state='UNKNOWN');self.roi=None;self.previous_roi=None
        if not self.enabled:return
        self.size=tuple(c['image_size'])
        if self.size!=(320,240):raise ValueError('Start-light calibration image size mismatch')
        extras=c.get('extra_templates',[])
        if not isinstance(extras,list) or len(extras)>3 or any(not isinstance(e,dict) or set(e)!={'template_file','core_roi'} for e in extras):raise ValueError('Invalid extra lamp appearances')
        self.templates=[self.load_appearance(c,template)]+[self.load_appearance(dict(c,**entry)) for entry in extras]
        self.template,self.core=self.templates[0]
        self.preferred_appearance=0
        self.appearances=[cv2.GaussianBlur(cv2.cvtColor(t,cv2.COLOR_BGR2GRAY),(3,3),0) for t,_ in self.templates]
        self.match_min=float(c['match_score_min']);self.minimum=float(c['minimum_luma']);self.contrast_min=float(c['minimum_contrast'])
        self.confirm=float(c.get('confirm_s',.35));self.gap=float(c.get('maximum_frame_gap_s',.35))
        self.halo_min=float(c.get('minimum_halo_contrast',5.))
        if not all(math.isfinite(v) for v in (self.match_min,self.minimum,self.contrast_min,self.confirm,self.gap,self.halo_min)):raise ValueError('Nonfinite lamp configuration')
        if not 0<self.halo_min<255:raise ValueError('Invalid lamp halo calibration')
        if not 0<self.match_min<1 or not 0<self.minimum<=255 or not 0<self.contrast_min<255 or not 0<self.confirm<=2 or not 0<self.gap<=1:raise ValueError('Invalid lamp calibration limits')

    @staticmethod
    def load_appearance(config,template=None):
        if template is None:
            filename=config['template_file']
            if Path(filename).name!=filename:raise ValueError('Lamp template must be an app file')
            template=cv2.imread(str(Path(__file__).with_name(filename)))
        if template is None or template.ndim!=3 or template.shape[2]!=3 or min(template.shape[:2])<8 or float(template.std())<5:raise ValueError('Invalid lamp appearance template')
        core=tuple(config['core_roi'])
        if len(core)!=4 or any(isinstance(v,bool) or not isinstance(v,int) for v in core):raise ValueError('Invalid lamp core')
        x,y,w,h=core
        if min(x,y)<0 or min(w,h)<3 or x+w>template.shape[1] or y+h>template.shape[0]:raise ValueError('Lamp core outside template')
        return template,core

    def arm(self):
        if self.enabled:
            self.armed=True;self.fired=False;self.since=self.last=None;self.elapsed=0.;self.phase='WAIT_LIGHT';self.roi=self.previous_roi=None;self.observation=dict(state='UNKNOWN',reason='WAITING_FOR_FRAME')

    def cancel(self):
        self.armed=False;self.since=self.last=None;self.elapsed=0.;self.phase='DISARMED';self.roi=self.previous_roi=None;self.observation=dict(state='UNKNOWN',reason='DISARMED')

    def started(self):
        self.armed=False;self.fired=True;self.phase='RUNNING'

    @lru_cache(maxsize=512)
    def _resized(self,index,width,height):
        template=cv2.resize(self.appearances[index],(width,height))
        return template

    def _proposals(self,bgr,gray):
        hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(bgr,(225,225,225),(255,255,255))
        mask|=cv2.inRange(hsv,(0,80,180),(11,255,255))
        mask|=cv2.inRange(hsv,(171,80,180),(179,255,255))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        height,width=gray.shape
        components=[a for a in stats[1:] if 9<=a[4]<=.03*width*height and min(a[2:4])>=3 and .4<=a[2]/a[3]<=3.]
        if len(components)>64:return None
        proposals=[];integral=cv2.integral(gray)
        def resolvable(w,h):
            return any(.4<=w/core[2]<=3. and .4<=h/core[3]<=3. for _,core in self.templates)
        for x,y,w,h,area in components:
            if not resolvable(w,h):continue
            # Opening at the component scale removes thin wires connected to
            # a blooming lamp, while retaining joined cells of an LED array.
            kernel=max(1,int(min(w,h)*.2)//2*2+1)
            crop=cv2.morphologyEx(mask[y:y+h,x:x+w],cv2.MORPH_OPEN,
                                 np.ones((kernel,kernel),np.uint8),
                                 borderType=cv2.BORDER_CONSTANT,borderValue=0)
            _,_,refined,_=cv2.connectedComponentsWithStats(crop)
            for dx,dy,rw,rh,count in refined[1:]:
                if count<9 or min(rw,rh)<3 or count/(rw*rh)<.5 or not resolvable(rw,rh):continue
                box=tuple(int(v) for v in (x+dx,y+dy,rw,rh))
                bx,by,bw,bh=box;core=gray[by:by+bh,bx:bx+bw]
                # The median threshold is a pixel-count test. Avoid Python
                # percentile/ring allocations for every unrelated bright spot;
                # exact photometry is reserved for matched lamp appearances.
                bright=cv2.countNonZero(cv2.inRange(core,math.ceil(self.minimum),255))
                if bright*2>=core.size and self._halo_drop(gray,box,integral)>=self.halo_min:proposals.append(box)
        return proposals

    @staticmethod
    def _photometry(gray,box,outer=None):
        x,y,w,h=box;height,width=gray.shape
        a=max(0,x);b=max(0,y);c=min(width,x+w);e=min(height,y+h)
        core=gray[b:e,a:c]
        if not core.size:return 0.,0.
        if outer is None:
            margin=max(2,round(min(w,h)*.3));outer=(x-margin,y-margin,w+2*margin,h+2*margin)
        ox,oy,ow,oh=outer
        x0=max(0,ox);y0=max(0,oy);x1=min(width,ox+ow);y1=min(height,oy+oh)
        ring=np.ones((y1-y0,x1-x0),bool)
        ring[max(0,b-y0):min(y1-y0,e-y0),max(0,a-x0):min(x1-x0,c-x0)]=False
        if not ring.any():return 0.,0.
        brightness=float(np.median(core))
        return brightness,brightness-float(np.median(gray[y0:y1,x0:x1][ring]))

    @staticmethod
    def _halo_drop(gray,box,integral=None):
        # An emitting lamp has a soft falloff; a flat white rectangle does not.
        # Integral sums give the same two band means without allocating masks.
        if integral is None:integral=cv2.integral(gray)
        x,y,w,h=box;height,width=gray.shape
        near=max(1,round(min(w,h)*.15));far=max(near+1,round(min(w,h)*.5))
        def rectangle(margin):
            a=max(0,x-margin);b=max(0,y-margin);c=min(width,x+w+margin);e=min(height,y+h+margin)
            total=int(integral[e,c])-int(integral[b,c])-int(integral[e,a])+int(integral[b,a])
            return total,(c-a)*(e-b)
        center,n0=rectangle(0);inner,n1=rectangle(near);outer,n2=rectangle(far)
        if n1<=n0 or n2<=n1:return 0.
        return (inner-center)/(n1-n0)-(outer-inner)/(n2-n1)

    def _match(self,image,box,gray):
        height,width=image.shape;x,y,w,h=box;best=None;models=[]
        for index in sorted(range(len(self.templates)),key=lambda i:i!=self.preferred_appearance):
            _,core=self.templates[index]
            cx,cy,cw,ch=core;sx=w/cw;sy=h/ch
            # Bound perspective/exposure changes: arbitrary stretching would
            # turn a vertical white object into the learned lamp shape.
            if not .4<=sx<=3. or not .4<=sy<=3. or not 1/1.6<=sx/sy<=1.6:continue
            models.append(dict(index=index,core=core,sx=sx,sy=sy,tried=set(),best=None,factors=None))

        def evaluate(model,fx,fy):
            nonlocal best
            index=model['index'];cx,cy,cw,ch=model['core']
            rx=model['sx']*fx;ry=model['sy']*fy
            if not 1/1.6<=rx/ry<=1.6:return
            ah,aw=self.appearances[index].shape
            tw=max(1,round(aw*rx));th=max(1,round(ah*ry))
            if (tw,th) in model['tried']:return
            model['tried'].add((tw,th));rx=tw/aw;ry=th/ah
            px=round(x+w/2-(cx+cw/2)*rx);py=round(y+h/2-(cy+ch/2)*ry)
            x0=max(0,px-3);y0=max(0,py-3);x1=min(width,px+tw+3);y1=min(height,py+th+3)
            tx=max(0,-px);ty=max(0,-py);ex=tw-max(0,px+tw-width);ey=th-max(0,py+th-height)
            if (ex-tx)*(ey-ty)<.6*tw*th:return
            template=self._resized(index,tw,th)[ty:ey,tx:ex];region=image[y0:y1,x0:x1]
            if not template.size or region.shape[0]<template.shape[0] or region.shape[1]<template.shape[1]:return
            if cv2.meanStdDev(template)[1][0,0]<5:return
            _,score,_,pos=cv2.minMaxLoc(cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED))
            if not math.isfinite(score):return
            bx=x0+pos[0]-tx;by=y0+pos[1]-ty
            a=round(bx+cx*rx);b=round(by+cy*ry);c=round(bx+(cx+cw)*rx);e=round(by+(cy+ch)*ry)
            match=dict(score=score,roi=[bx,by,tw,th],core_box=(a,b,c-a,e-b))
            if score>=self.match_min:
                lamp,contrast=self._photometry(gray,match['core_box'],match['roi'])
                if lamp>=self.minimum and contrast>=self.contrast_min:
                    self.preferred_appearance=index
                    match.update(lamp_luma=lamp,contrast=contrast)
                    return match
            if model['best'] is None or score>model['best']['score']:model.update(best=match,factors=(fx,fy))
            if best is None or score>best['score']:best=match

        # Search a coarse scale/aspect grid first. Refine only when necessary,
        # around each model's best hypothesis, instead of exhausting 49 scales
        # for every bright object on every frame. Resized models are cached.
        # A confirmed appearance needs no maximum-score search. Try the last
        # successful appearance first, while still searching the whole frame.
        for model in models:
            for fx in (1.,.7,1.3):
                for fy in (1.,.7,1.3):
                    accepted=evaluate(model,fx,fy)
                    if accepted is not None:return accepted
        for model in models:
            if model['factors'] is None:continue
            fx,fy=model['factors']
            for dx in (-.1,0.,.1):
                for dy in (-.1,0.,.1):
                    accepted=evaluate(model,fx+dx,fy+dy)
                    if accepted is not None:return accepted
        return best

    def detect(self,bgr):
        if not self.enabled:return dict(state='DISABLED')
        if bgr is None or bgr.shape!=(self.size[1],self.size[0],3):return dict(state='UNKNOWN',reason='IMAGE_SIZE')
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        proposals=self._proposals(bgr,gray)
        if proposals is None:return dict(state='UNKNOWN',reason='TOO_MANY_LIGHT_CANDIDATES')
        # Mild smoothing compares the lamp envelope instead of individual LED
        # cells, which disappear when the same lamp clips to white. Colour is
        # used for proposals, luminance for appearance and actual brightness.
        match_image=cv2.GaussianBlur(gray,(3,3),0);found=[];peak=0.
        for box in proposals:
            best=self._match(match_image,box,gray)
            if best is None:continue
            peak=max(peak,best['score'])
            if best['score']<self.match_min:continue
            if 'lamp_luma' in best:lamp,contrast=best['lamp_luma'],best['contrast']
            else:lamp,contrast=self._photometry(gray,best['core_box'],best['roi'])
            if lamp<self.minimum or contrast<self.contrast_min:continue
            bx,by,bw,bh=best['roi'];center=np.array([bx+bw/2,by+bh/2])
            if any(np.linalg.norm(center-f['center'])<min(bw,bh)/2 for f in found):continue
            found.append(dict(best,center=center,lamp_luma=lamp,contrast=contrast))
        if len(found)!=1:
            self.roi=self.previous_roi=None
            return dict(state='OFF' if not found else 'UNKNOWN',reason='NO_MATCHED_LAMP' if not found else 'MULTIPLE_MATCHED_LAMPS',best_match=round(peak,3),candidates=len(proposals))
        best=found[0];bx,by,bw,bh=best['roi'];height,width=gray.shape
        # Publish only the visible rectangle; an edge detection never invents
        # pixels outside the frame for brightness or temporal confirmation.
        self.roi=[max(0,bx),max(0,by),min(width,bx+bw)-max(0,bx),min(height,by+bh)-max(0,by)]
        previous=self.previous_roi;self.previous_roi=list(self.roi)
        moved=previous is not None and np.linalg.norm(np.array(self.roi[:2])+np.array(self.roi[2:])/2-np.array(previous[:2])-np.array(previous[2:])/2)>min(self.roi[2:])/2
        return dict(state='UNKNOWN' if moved else 'ON',reason='CANDIDATE_MOVED' if moved else 'MATCHED_LAMP',roi=list(self.roi),match_score=round(best['score'],3),lamp_luma=best['lamp_luma'],contrast=round(best['contrast'],2),candidates=len(proposals))

    def update(self,observation,now,*,active=False,timely=True):
        self.observation=observation
        if not self.enabled:return False
        if active:self.started();return False
        if not self.armed:return False
        valid_time=math.isfinite(now) and (self.last is None or 0<now-self.last<=self.gap)
        if not valid_time or not timely or observation.get('state')!='ON':self.since=None
        if timely and math.isfinite(now) and observation.get('state')=='ON':
            if self.since is None:self.since=now
        self.last=now if math.isfinite(now) else None
        self.elapsed=0. if self.since is None else max(0.,now-self.since)
        confirmed=timely and observation.get('state')=='ON' and self.elapsed>=self.confirm
        self.phase='WAIT_FRESH_FRAME' if observation.get('state')=='ON' and not timely else 'SIGNAL_READY' if confirmed else 'CONFIRM_LIGHT' if self.since is not None else 'WAIT_LIGHT'
        return confirmed

    def snapshot(self):
        result=dict(enabled=self.enabled,armed=bool(self.armed),phase=self.phase,fired=self.fired,
                    lamp=self.observation,confirmed_s=round(self.elapsed,3),search_scope='FULL_FRAME')
        if self.enabled:result.update(required_s=self.confirm,roi=self.roi,match_score_min=self.match_min,appearance_count=len(self.templates))
        return result
