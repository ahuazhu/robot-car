"""Metric local target tracking. x=forward, y=left, origin=camera footprint.
Motion prediction uses approximate commanded velocity, NOT wheel odometry.
"""
import math
import cv2
import numpy as np

LOOKAHEAD=.30
ROAD_WIDTH=.30
BLIND_SECONDS=.60


def transform_path(points, v, omega, dt):
    """Move fixed ground points into the new robot frame (constant twist)."""
    theta=omega*dt
    dx=v*dt if abs(omega)<1e-6 else v*math.sin(theta)/omega
    dy=0. if abs(omega)<1e-6 else v*(1-math.cos(theta))/omega
    p=np.asarray(points,float)-[dx,dy]
    c,s=math.cos(theta),math.sin(theta)
    return p@np.array([[c,-s],[s,c]])


def target_at(path,x=None):
    """Choose a supported centerline point; never extend the fitted path.
    Forward lookahead contracts with heading, bend curvature and target bearing.
    The inferred center point need not project inside the camera image.
    """
    p=np.asarray(path,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():return None
    if np.any(np.diff(p[:,0])<=1e-6):return None
    lo=max(.10,float(p[0,0]));hi=min(.38,float(p[-1,0]))
    if lo>hi:return None
    if x is None:
        probe=float(np.clip(LOOKAHEAD,lo,hi))
        slopes=np.gradient(p[:,1],p[:,0])
        slope=float(np.interp(probe,p[:,0],slopes))
        second=np.gradient(slopes,p[:,0])
        curvature=abs(float(np.interp(probe,p[:,0],second)))/(1+slope*slope)**1.5
        y=float(np.interp(probe,p[:,0],p[:,1]))
        severity=min(1.,max(abs(math.atan(slope))/1.0,curvature/5.,abs(math.atan2(y,probe))/.8))
        x=float(np.clip(LOOKAHEAD-.18*severity,lo,hi))
    if x<lo or x>hi:return None
    target=np.array([x,np.interp(x,p[:,0],p[:,1])])
    return target if np.linalg.norm(target)<=.65 else None


def path_difference(old,new):
    """Compare paths at the SAME forward coordinates despite adaptive targets."""
    if old is None:return None
    if np.any(np.diff(old[:,0])<=0) or np.any(np.diff(new[:,0])<=0):return None
    lo=max(.10,old[0,0],new[0,0]);hi=min(.38,old[-1,0],new[-1,0])
    if hi-lo<.015:return None
    xs=np.linspace(lo,hi,8)
    delta=np.interp(xs,old[:,0],old[:,1])-np.interp(xs,new[:,0],new[:,1])
    return float(np.median(delta)),float(np.max(np.abs(delta)))


class RoadPixels:
    def __init__(self,cal):
        self.H=np.asarray(cal['H_normalized'],float)
        self.Hinv=np.linalg.inv(self.H)
        self.width=float(cal.get('road_width_m',ROAD_WIDTH))
        self.xs=np.arange(.10,.711,.01)
        self.ys=np.arange(.60,-.601,-.005)
        xx,yy=np.meshgrid(self.xs,self.ys,indexing='ij')
        p=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@self.Hinv.T
        u=p[:,:,0]/p[:,:,2]*320;v=p[:,:,1]/p[:,:,2]*240
        self.visible=(u>=1)&(u<319)&(v>=120)&(v<239)&np.isfinite(u)&np.isfinite(v)
        self.mapx=u.astype(np.float32);self.mapy=v.astype(np.float32)

    def extract(self,bgr,prior=None,search=.25):
        bird=cv2.remap(bgr,self.mapx,self.mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        hsv=cv2.cvtColor(bird,cv2.COLOR_BGR2HSV)
        green=cv2.inRange(hsv,np.array([32,55,25]),np.array([95,255,255]))>0
        white=cv2.inRange(hsv,np.array([0,0,180]),np.array([179,105,255]))>0
        white &= self.visible
        candidates={'left':[],'right':[]}
        transitions=np.diff(np.pad(white.astype(np.int8),((0,0),(1,1))),axis=1)
        rows,starts=np.where(transitions==1);end_rows,ends=np.where(transitions==-1)
        use=(ends-starts<=20)&(starts>=7)&(ends+7<len(self.ys))
        rows=rows[use];starts=starts[use];ends=ends[use]
        prefix=np.pad(np.cumsum(green&self.visible,axis=1),((0,0),(1,0)))
        gl=(prefix[rows,starts-1]-prefix[rows,starts-7])/6
        gr=(prefix[rows,ends+7]-prefix[rows,ends+1])/6
        use=(np.maximum(gl,gr)>=.5)&(np.abs(gl-gr)>=.33)
        rows=rows[use];starts=starts[use];ends=ends[use];gl=gl[use];gr=gr[use]
        left=gl>gr;xs=self.xs[rows];ys=self.ys[np.where(left,ends-1,starts)]
        keep=np.ones(len(xs),bool)
        if prior is not None and np.all(np.diff(prior[:,0])>1e-6):
            center=np.interp(xs,prior[:,0],prior[:,1])
            slope=np.interp(xs,prior[:,0],np.gradient(prior[:,1],prior[:,0]))
            expected=center+np.where(left,1,-1)*self.width/2*np.sqrt(1+slope*slope)
            keep=np.abs(ys-expected)<=search
        values=np.column_stack((xs,ys,np.maximum(gl,gr)))
        candidates['left']=values[keep&left].tolist()
        candidates['right']=values[keep&~left].tolist()
        return candidates,bird

    @staticmethod
    def trace(points):
        """Continuous scanline chain, followed by robust local fitting."""
        if len(points)<6:return None
        p=np.asarray(points,float);order=np.lexsort((p[:,1],p[:,0]));p=p[order]
        score=np.zeros(len(p));prev=np.full(len(p),-1,int)
        for i in range(len(p)):
            score[i]=p[i,2]
            for j in range(i-1,-1,-1):
                dx=p[i,0]-p[j,0]
                if dx>.055:break
                if dx<.005:continue
                slope=(p[i,1]-p[j,1])/dx
                if abs(slope)>2.0:continue
                cost=.10*abs(slope)
                if prev[j]>=0:
                    k=prev[j];old=(p[j,1]-p[k,1])/(p[j,0]-p[k,0])
                    cost+=.3*min(3.,abs(slope-old))
                value=score[j]+p[i,2]-cost
                if value>score[i]:score[i]=value;prev[i]=j
        i=int(np.argmax(score));idx=[]
        while i>=0:idx.append(i);i=int(prev[i])
        q=p[idx[::-1]]
        if len(q)<7 or np.ptp(q[:,0])<.10:return None
        degree=2 if np.ptp(q[:,0])>=.18 else 1
        design=np.column_stack([q[:,0]**k for k in range(degree+1)])
        w=q[:,2].copy();coeff=np.zeros(degree+1)
        for _ in range(4):
            coeff=np.linalg.lstsq(design*np.sqrt(w[:,None]),q[:,1]*np.sqrt(w),rcond=None)[0]
            residual=np.abs(design@coeff-q[:,1]);w=q[:,2]*np.minimum(1,.008/np.maximum(residual,1e-6))
        inside=residual<.025
        if inside.mean()<.75 or np.median(residual)>.012:return None
        lo=float(q[inside,0].min());hi=float(q[inside,0].max())
        xs=np.linspace(lo,hi,40);ys=np.polynomial.polynomial.polyval(xs,coeff)
        slopes=np.polynomial.polynomial.polyval(xs,np.polynomial.polynomial.polyder(coeff))
        if max(abs(slopes))>2.0:return None
        confidence=float(min(1,len(q)/15)*min(1,(hi-lo)/.18)*np.mean(q[:,2])*(1-min(.4,np.median(residual)/.03)))
        return dict(path=np.column_stack((xs,ys)),slope=slopes,confidence=confidence,points=q[:,:2])

    def measure(self,bgr,prior=None,search=.25):
        candidates,bird=self.extract(bgr,prior,search)
        fits={s:self.trace(p) for s,p in candidates.items()}
        centers={}
        for side,f in fits.items():
            if f is None:continue
            sign=1 if side=='right' else -1
            n=np.sqrt(1+f['slope']**2)
            path=f['path']+sign*self.width/2*np.column_stack((-f['slope']/n,1/n))
            if target_at(path) is not None:centers[side]=path
        measurement=None
        if len(centers)==2:
            l,r=centers['left'],centers['right']
            lo=max(l[0,0],r[0,0]);hi=min(l[-1,0],r[-1,0])
            xs=np.linspace(lo,hi,32)
            yl=np.interp(xs,l[:,0],l[:,1]);yr=np.interp(xs,r[:,0],r[:,1])
            near=(xs>=.10)&(xs<=.38)
            if hi>lo and near.any() and np.max(np.abs(yl[near]-yr[near]))<.065:
                path=np.column_stack((xs,(yl+yr)/2))
                measurement=dict(path=path,mode='TWO EDGES',confidence=min(f['confidence'] for f in fits.values()))
        elif len(centers)==1:
            side=next(iter(centers));other='right' if side=='left' else 'left'
            # A second long but inconsistent boundary is not silently ignored.
            if fits[other] is None or fits[other]['confidence']<.55:
                measurement=dict(path=centers[side],mode='ONE '+side.upper(),confidence=fits[side]['confidence']*.85)
        return measurement,dict(candidates=candidates,fits=fits,bird=bird)


class TargetTracker:
    def __init__(self,cal):
        self.pixels=RoadPixels(cal);self.path=None;self.last_time=None
        self.last_seen=None;self.confidence=0.;self.blind_distance=0.;self.mode='LOST'
        self.pending=None;self.pending_count=0

    def update(self,bgr,now,v=0.,omega=0.):
        dt=0 if self.last_time is None else max(0.,min(.5,now-self.last_time));self.last_time=now
        if self.path is not None:self.path=transform_path(self.path,v,omega,dt)
        age=1e3 if self.last_seen is None else now-self.last_seen
        if self.path is not None:self.blind_distance+=abs(v)*dt
        prior=self.path if age<1.0 else None
        search=min(.30,.10+.25*age)
        m,details=self.pixels.measure(bgr,prior,search)
        accepted=False;reject_reason='NO EDGE'
        if m is not None:
            reject_reason='LOW CONFIDENCE'
        if m is not None and m['confidence']>=.55:
            target=target_at(m['path'])
            delta=path_difference(self.path,m['path'])
            continuous=self.path is None or (delta is not None and delta[1]<=.05+.08*min(age,1))
            reject_reason='PATH JUMP' if target is not None else 'NO SUPPORTED TARGET'
            if target is not None and not continuous:
                if self.pending is not None and np.linalg.norm(target-self.pending)<.025:self.pending_count+=1
                else:self.pending_count=1
                self.pending=target.copy()
                # Stationary reacquisition only; do not chase a new lane while moving.
                continuous=abs(v)<.005 and abs(omega)<.02 and self.pending_count>=4
            if target is not None and continuous:
                path=m['path'].copy()
                if delta is not None and age<.3:
                    path[:,1]+=.25*delta[0]
                self.path=path;self.last_seen=now;self.confidence=m['confidence'];self.blind_distance=0
                self.mode=m['mode'];self.pending=None;self.pending_count=0;accepted=True
        age=1e3 if self.last_seen is None else now-self.last_seen
        target=target_at(self.path) if self.path is not None else None
        if accepted:confidence=self.confidence;mode=self.mode
        else:confidence=self.confidence*math.exp(-age/.45);mode='PREDICT'
        invalid_reason=''
        if target is None:invalid_reason='NO SUPPORTED TARGET'
        elif not accepted and (age>BLIND_SECONDS or self.blind_distance>.05):invalid_reason='MEMORY EXPIRED'
        elif confidence<.22:invalid_reason='LOW CONFIDENCE'
        valid=not invalid_reason
        if not valid:mode='LOST'
        result=dict(valid=bool(valid),mode=mode,confidence=float(confidence),age_s=round(age,3),
                    target_m=target.tolist() if target is not None else None,
                    observed=accepted,invalid_reason=invalid_reason,measurement_reject=None if accepted else reject_reason,
                    blind_distance_m=self.blind_distance,
                    path=self.path.tolist() if self.path is not None else [],
                    counts={s:len(p) for s,p in details['candidates'].items()})
        return result,details


def target_command(result):
    if not result['valid']:return None
    x,y=result['target_m'];curvature=2*y/(x*x+y*y)
    bearing=abs(math.atan2(y,x))
    severity=min(1.,max(bearing/.8,abs(curvature)/7.))
    # Forward motion slows in bends while wheel difference can still increase.
    base=60-17*severity
    if result['mode'].startswith('ONE'):base=min(base,53.)
    if result['mode']=='PREDICT':base=min(base,42.)
    turn=float(np.clip(2.5*curvature,-14,14))
    if result['mode']=='PREDICT':turn=float(np.clip(turn,-10,10))
    if abs(turn)<.6:turn=0.
    return round(base-turn),round(base+turn)


def motion_from_pwm(pwm):
    # Conservative nominal estimates until measured wheel/visual odometry exists.
    left,right=pwm
    v=max(0.,(left+right)/2-30)*.003
    omega=(right-left)*.025
    return v,omega


def render_target(bgr,result,details,cal):
    out=bgr.copy();Hinv=np.linalg.inv(np.asarray(cal['H_normalized']))
    def image_points(xy):
        p=np.column_stack((np.asarray(xy),np.ones(len(xy))))@Hinv.T
        return np.rint(p[:,:2]/p[:,2,None]*[320,240]).astype(np.int32)
    for side,f in details['fits'].items():
        if f is None:continue
        pts=image_points(f['path'])
        cv2.polylines(out,[pts],False,(0,220,255) if side=='left' else (255,180,0),2)
    if result['valid']:
        pts=image_points(result['path'])
        cv2.polylines(out,[pts],False,(0,230,0),1)
        target=tuple(image_points([result['target_m']])[0])
        inimage=0<=target[0]<320 and 0<=target[1]<240
        onscreen=inimage and 38<=target[1]<209
        marker=(int(np.clip(target[0],8,311)),int(np.clip(target[1],62,195)))
        cv2.drawMarker(out,marker,(0,0,255),cv2.MARKER_CROSS if onscreen else cv2.MARKER_TRIANGLE_UP,16,2)
        label='%dcm %s'%(round(result['target_m'][0]*100),'TARGET' if onscreen else 'LOW TARGET' if inimage else 'OFF VIEW')
        cv2.putText(out,label,(max(0,min(170,marker[0]-40)),max(65,marker[1]-10)),0,.37,(0,0,255),1)
    cv2.putText(out,'%s C=%.2f'%(result['mode'],result['confidence']),(4,51),0,.42,(0,255,255),1)
    bird=cv2.resize(cv2.flip(details['bird'],0),(320,240))
    return out,bird
