"""Near-ground robust boundary fit for miniAuto's two white borders.

Duckietown LineDetector and GroundProjector supply segments. This adapter
fits each side's endpoint cloud before using the segment-normal / half-width
lane-pose geometry. It is not Duckietown's encoder-based Bayesian filter.
Coordinates: x forward, y left; d vehicle left offset, phi vehicle heading
relative to lane. No previous pose survives a missing/inconsistent measurement.
"""
import math
import numpy as np


def fit_boundary(segments):
    if not segments:return None
    points=np.array([p for s in segments for p in s['ground']],float)
    weights=np.repeat([min(s['length'],.12)/2 for s in segments],2)
    # Generate hypotheses from spatially separated endpoints, including pairs
    # across fragments. Tiny segment angles are deliberately not used as votes.
    i,j=np.triu_indices(len(points),1)
    delta=points[j]-points[i]
    use=np.abs(delta[:,0])>=.075
    if not use.any():return None
    i=i[use];delta=delta[use]
    slopes=delta[:,1]/delta[:,0]
    use=np.abs(slopes)<math.tan(math.radians(65))
    if not use.any():return None
    i=i[use];slopes=slopes[use]
    intercepts=points[i,1]-slopes*points[i,0]
    residual=np.abs(points[None,:,1]-slopes[:,None]*points[None,:,0]-intercepts[:,None])/np.sqrt(1+slopes[:,None]**2)
    scores=np.exp(-.5*(residual/.012)**2)@weights
    best=int(np.argmax(scores));a=slopes[best];b=intercepts[best]
    for _ in range(3):
        residual=np.abs(points[:,1]-a*points[:,0]-b)/math.sqrt(1+a*a)
        inside=residual<.025
        if inside.sum()<2 or np.ptp(points[inside,0])<.075:return None
        w=weights[inside]*np.minimum(1,.010/np.maximum(residual[inside],1e-6))
        design=np.column_stack((points[inside,0],np.ones(inside.sum())))
        a,b=np.linalg.lstsq(design*np.sqrt(w[:,None]),points[inside,1]*np.sqrt(w),rcond=None)[0]
    residual=np.abs(points[:,1]-a*points[:,0]-b)/math.sqrt(1+a*a)
    inside=residual<.025
    support=float(weights[inside].sum()/weights.sum())
    if support<.55 or abs(a)>math.tan(math.radians(65)):return None
    chosen=[s['index'] for k,s in enumerate(segments) if inside[2*k:2*k+2].all()]
    if not chosen:return None
    return dict(a=float(a),b=float(b),inliers=chosen,support=support,
                x_min=float(points[inside,0].min()),x_max=float(points[inside,0].max()))


def fit_curve(segments):
    if not segments:return None
    p=np.array([q for s in segments for q in s['ground']],float)
    if len(p)<6 or np.ptp(p[:,0])<.18:return None
    w=np.repeat([min(s['length'],.12)/2 for s in segments],2)
    z=(p[:,0]-.4)/.25;design=np.column_stack((np.ones(len(p)),z,z*z))
    rng=np.random.default_rng(17)
    picks=rng.integers(0,len(p),size=(128,3))
    zz=z[picks];yy=p[picks,1]
    use=(np.abs(zz[:,1]-zz[:,0])>.1)&(np.abs(zz[:,2]-zz[:,1])>.1)&(np.abs(zz[:,2]-zz[:,0])>.1)
    zz=zz[use];yy=yy[use]
    if not len(zz):return None
    first=(yy[:,1]-yy[:,0])/(zz[:,1]-zz[:,0])
    second=(yy[:,2]-yy[:,1])/(zz[:,2]-zz[:,1])
    c2=(second-first)/(zz[:,2]-zz[:,0])
    c1=first-c2*(zz[:,0]+zz[:,1])
    c0=yy[:,0]-c1*zz[:,0]-c2*zz[:,0]**2
    coeff=np.column_stack((c0,c1,c2));coeff=coeff[np.abs(coeff[:,2])<.35]
    if not len(coeff):return None
    residual=np.abs(coeff[None,:,0]+z[:,None]*coeff[None,:,1]+z[:,None]**2*coeff[None,:,2]-p[:,1,None])
    scores=(np.exp(-.5*(residual/.018)**2)*w[:,None]).sum(axis=0)
    c=coeff[int(np.argmax(scores))]
    for _ in range(4):
        residual=np.abs(design@c-p[:,1]);inside=residual<.035
        if inside.sum()<5 or np.ptp(p[inside,0])<.18:return None
        rw=w[inside]*np.minimum(1.,.012/np.maximum(residual[inside],1e-6))
        c=np.linalg.lstsq(design[inside]*np.sqrt(rw[:,None]),p[inside,1]*np.sqrt(rw),rcond=None)[0]
    if abs(c[2])>.35 or w[inside].sum()/w.sum()<.6:return None
    chosen=[s['index'] for k,s in enumerate(segments) if inside[2*k:2*k+2].all()]
    actual=[float(c[0]-1.6*c[1]+2.56*c[2]),float(4*c[1]-12.8*c[2]),float(16*c[2])]
    return dict(coeff=actual,inliers=chosen,x_min=float(p[inside,0].min()),x_max=float(p[inside,0].max()))

def curve_pose(sides,width):
    fits={side:fit_curve(v) for side,v in sides.items()}
    if not all(fits.values()):return None
    l,r=fits['left'],fits['right'];lo=max(l['x_min'],r['x_min']);hi=min(l['x_max'],r['x_max'])
    if hi-lo<.16:return None
    lc=np.array(l['coeff']);rc=np.array(r['coeff']);c=(lc+rc)/2
    xs=np.linspace(lo,hi,5);design=np.column_stack((np.ones(5),xs,xs*xs))
    slopes=c[1]+2*c[2]*xs
    widths=(design@(lc-rc))/np.sqrt(1+slopes*slopes)
    if not all((widths>width-.065)&(widths<width+.085)):return None
    target_x=min(max(.40,lo+.03),hi-.02)
    if target_x<.30:return None
    target_y=float(c[0]+c[1]*target_x+c[2]*target_x*target_x)
    near_y=float(c[0]+c[1]*lo+c[2]*lo*lo)
    near_slope=float(c[1]+2*c[2]*lo)
    tangent_intercept=near_y-near_slope*lo
    return dict(valid=True,status='CURVED PATH',mode='CURVE',d_m=-tangent_intercept/math.sqrt(1+near_slope**2),
                phi_rad=-math.atan(c[1]+2*c[2]*target_x),target_m=[target_x,target_y],near_center_m=near_y,
                path_coefficients=c.tolist(),path_range_m=[lo,hi],width_m=float(np.mean(widths)),coverage_m=hi-lo,
                support={s:len(f['inliers']) for s,f in fits.items()},inliers=l['inliers']+r['inliers'],votes=sum(map(len,sides.values())))


def single_boundary_pose(sides,width):
    """Normal-offset a well-supported boundary, never extrapolate its range."""
    fits={side:fit_curve(v) for side,v in sides.items()}
    available=[side for side,f in fits.items() if f is not None]
    if len(available)!=1:return None
    side=available[0];other='left' if side=='right' else 'right'
    # Conflicting second boundary must not be silently discarded.
    opposite=fit_boundary(sides[other])
    if opposite is not None and opposite['x_max']-opposite['x_min']>=.18:return None
    f=fits[side]
    if len(f['inliers'])<5 or f['x_max']-f['x_min']<.25 or f['x_min']>.40:return None
    c=np.array(f['coeff']);xs=np.linspace(f['x_min'],f['x_max'],24)
    ys=c[0]+c[1]*xs+c[2]*xs*xs;slope=c[1]+2*c[2]*xs
    norm=np.sqrt(1+slope*slope)
    curvature=2*c[2]/norm**3
    if np.max(np.abs(curvature))*width/2>.65:return None
    inward=1 if side=='right' else -1
    cx=xs-inward*width/2*slope/norm;cy=ys+inward*width/2/norm
    if np.any(np.diff(cx)<=0) or cx[-1]-cx[0]<.18:return None
    target_x=.40
    if not cx[0]+.02<=target_x<=cx[-1]-.02:return None
    target_y=float(np.interp(target_x,cx,cy))
    cc=np.polynomial.polynomial.polyfit(cx,cy,2)
    if np.max(np.abs(np.polynomial.polynomial.polyval(cx,cc)-cy))>.01:return None
    near_y=float(cy[0]);near_slope=float(cc[1]+2*cc[2]*cx[0])
    return dict(valid=True,status='SINGLE '+side.upper(),mode='CURVE',single_side=side,
                width_assumed=True,width_m=width,d_m=-(near_y-near_slope*cx[0])/math.sqrt(1+near_slope**2),
                phi_rad=-math.atan(cc[1]+2*cc[2]*target_x),target_m=[target_x,target_y],near_center_m=near_y,
                path_coefficients=cc.tolist(),path_range_m=[float(cx[0]),float(cx[-1])],coverage_m=float(cx[-1]-cx[0]),
                support={side:len(f['inliers']),other:0},inliers=f['inliers'],votes=sum(map(len,sides.values())))


def estimate_lane_pose(lines, width=.30, allow_single_side=False):
    sides={'left':[],'right':[]}
    for index,line in enumerate(lines):
        if 'ground' not in line or line['side'] not in sides:continue
        p=np.asarray(line['ground'],float)
        if p.shape!=(2,2) or not np.isfinite(p).all():continue
        p=p[np.argsort(p[:,0])];delta=p[1]-p[0]
        if delta[0]<.012:continue
        lo=max(.20,p[0,0]);hi=min(.65,p[1,0])
        if hi-lo<.012:continue
        q=p[0]+np.array([lo-p[0,0],hi-p[0,0]])[:,None]/delta[0]*delta
        length=float(np.linalg.norm(q[1]-q[0]))
        if length<.025:continue
        sides[line['side']].append(dict(index=index,ground=q.tolist(),length=length))
    invalid=dict(valid=False,status='NO CONSISTENT PAIR',d_m=None,phi_rad=None,
                 support={'left':0,'right':0},inliers=[],votes=sum(map(len,sides.values())))
    curved=curve_pose(sides,width)
    if curved is not None:return curved
    single=single_boundary_pose(sides,width) if allow_single_side else None
    if single is not None:return single
    fits={side:fit_boundary(segments) for side,segments in sides.items()}
    if not all(fits.values()):return invalid
    left,right=fits['left'],fits['right']
    # Require common near-ground coverage, compatible headings and measured
    # separation. No forced 30-cm corridor when the visible edges disagree.
    lo=max(left['x_min'],right['x_min']);hi=min(left['x_max'],right['x_max'])
    if hi-lo<.05:return invalid
    angles=[math.atan(f['a']) for f in (left,right)]
    if abs(angles[0]-angles[1])>.25:return invalid
    heading=sum(angles)/2
    widths=[((left['a']-right['a'])*x+left['b']-right['b'])*math.cos(heading) for x in (lo,hi)]
    if not all(abs(w-width)<=.065 for w in widths):return invalid
    distances=[width/2-left['b']/math.sqrt(1+left['a']**2),
               -width/2-right['b']/math.sqrt(1+right['a']**2)]
    d=sum(distances)/2
    if abs(d)>.30:return invalid
    return dict(valid=True,status='POSE PREVIEW',d_m=d,phi_rad=-heading,
                support={s:len(f['inliers']) for s,f in fits.items()},
                inliers=left['inliers']+right['inliers'],votes=invalid['votes'],
                width_m=sum(widths)/2,coverage_m=hi-lo)
