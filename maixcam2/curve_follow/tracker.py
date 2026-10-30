"""White-boundary extraction, parametric centre curves and bounded prediction."""
import math
import time
import cv2
import numpy as np
from geometry import (arc, fit_curve, inward_center, merge_centers, choose_target,
                      curve_distance, contiguous_runs, curve_agreement, geometry_frame,arc)
from native_geometry import native_boundary_chains
from road_policy import RoadPolicy
from ground_motion import GroundMotion,motion_summary
from road_model import LocalRoadModel,BoundaryTrackBank


class CurvePixels:
    def __init__(self, cal, policy=None):
        self.policy=policy or RoadPolicy.from_config(cal)
        self.exterior_distances=np.array([.025,.035,.045])
        self.boundary_prior={'left':[],'right':[]}
        self.H = np.asarray(cal['H_normalized'], float)
        self.Hinv = np.linalg.inv(self.H)
        self.width = float(cal.get('road_width_m', .30))
        # Photometric calibration, shared by every frame and curve source.
        self.white_value_levels = tuple(cal.get('white_value_levels', (210, 180)))
        if (not self.white_value_levels or
                any(not 0 <= value <= 255 for value in self.white_value_levels)):
            raise ValueError('Invalid white boundary brightness levels')
        self.step = float(np.clip(cal.get('vision_grid_step_m', .005), .005, .01))
        self.xs = np.arange(.06, .805, self.step)
        self.ys = np.arange(.65, -.655, -self.step)
        xx, yy = np.meshgrid(self.xs, self.ys, indexing='ij')
        p = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ self.Hinv.T
        u = p[:, :, 0]/p[:, :, 2]*320
        v = p[:, :, 1]/p[:, :, 2]*240
        # ROI belongs to the calibrated camera pose; fixed old crops truncate
        # otherwise usable boundaries after a mount-angle change.
        roi_top = int(np.clip(cal.get('vision_roi_top_px', 100), 0, 220))
        self.visible = (u >= 2) & (u < 318) & (v >= roi_top) & (v < 238) & np.isfinite(u) & np.isfinite(v)
        self.visible_u8=self.visible.astype(np.uint8)
        rows,cols=np.where(self.visible)
        self.visible_near_m=float(np.hypot(self.xs[rows],self.ys[cols]).min())
        self.mapx = u.astype(np.float32)
        self.mapy = v.astype(np.float32)

    def sample(self, mask, points):
        row = np.rint((points[:, 0]-self.xs[0])/self.step).astype(int)
        col = np.rint((self.ys[0]-points[:, 1])/self.step).astype(int)
        valid = (row >= 0) & (row < len(self.xs)) & (col >= 0) & (col < len(self.ys))
        r = np.clip(row, 0, len(self.xs)-1); c = np.clip(col, 0, len(self.ys)-1)
        valid &= self.visible[r, c]
        return mask[r, c].astype(float), valid

    def prepare(self, bgr):
        bird = cv2.remap(bgr, self.mapx, self.mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
        h, s, v = cv2.split(cv2.cvtColor(bird, cv2.COLOR_BGR2HSV))
        green = (h >= 32) & (h <= 95) & (s > 55) & (v > 25) & self.visible
        # Normalize local luminance in ground metric space. A white stripe in
        # shadow must not lose to a specular road highlight solely because of
        # absolute exposure. A quarter-lane window spans paint and its context.
        vf=v.astype(np.float32);window=max(3,int(self.width/self.step/4)|1)
        mean=cv2.blur(vf,(window,window))
        deviation=np.sqrt(np.maximum(0.,cv2.blur(vf*vf,(window,window))-mean*mean))
        contrast=np.clip(180+60*(vf-mean)/np.maximum(deviation,1.),0,255).astype(np.uint8)
        contrast[v<35]=0
        gray=(s<110)&(contrast>=35)&(contrast<210)&self.visible&~green
        # Necessary condition for the existing normal-probe exterior test.
        # Reject only entire contours with no possible green sample, never
        # clip contours or require green at every point of a valid chain.
        radius=int(math.ceil(float(self.exterior_distances.max())/self.step))+1
        vicinity=cv2.dilate(green.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8))
        return bird,s,v,(s<110)&(v>=35)&(v<210)&self.visible,green,contrast,gray,vicinity

    def contour_chains_reference(self,pixels,region_codes,gray,minimum_samples):
        chains=[]
        p = np.column_stack((self.xs[pixels[:, 1]], self.ys[pixels[:, 0]]))
        tangent = np.roll(p, -4, axis=0)-np.roll(p, 4, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        t = tangent/np.maximum(norm[:, None], 1e-8)
        normal = np.column_stack((-t[:, 1], t[:, 0]))
        # Sample both sides, all three distances and both region masks in
        # one indexed pass; repeated tiny numpy calls dominate on ARM.
        offsets = np.array([1.,-1.])[:,None,None,None] * self.exterior_distances[None,:,None,None]
        samples = p[None,None,:,:] + offsets*normal[None,None,:,:]
        # White paint counts as road only with green on the opposite
        # side of the same normal sample. Without that exterior evidence,
        # preserve the original gray-road test. Keep gray-only immediate
        # facing checks below so the outer stripe edge is still rejected.
        codes, checks = self.sample(region_codes, samples.reshape(-1,2))
        codes=codes.astype(np.uint8).reshape(2,3,len(p))
        outside_bits=(codes & 2)>0
        roads=(((codes & 1)>0) | (((codes & 4)>0) & outside_bits[::-1])).mean(axis=1)
        outsides=outside_bits.mean(axis=1)
        visibility=checks.reshape(2,3,len(p)).all(axis=1)
        # Gray on road side and contrasting exterior: stops/crosswalks with gray
        # on BOTH sides are deliberately not interpreted as lane boundaries.
        evidence = roads[0]-roads[1]+.7*(outsides[1]-outsides[0])
        for inward_sign in (1, -1):
            road_index = 0 if inward_sign == 1 else 1
            good = (inward_sign*evidence > .65) & (roads[road_index] >= .66)
            good &= visibility[0] & visibility[1] & (norm > .012)
            # A cyclic contour can have a valid chain across its array seam.
            if good.all():
                continue
            start = int(np.flatnonzero(~good)[0])
            pp = np.roll(p, -start, axis=0)
            nn = np.roll(normal*inward_sign, -start, axis=0)
            gg = np.roll(good, -start)
            for ids in contiguous_runs(gg):
                if len(ids) < minimum_samples:
                    continue
                chain, inward = pp[ids], nn[ids]
                # Require exterior evidence over a majority of this whole
                # chain, while allowing paint joins and short occlusions.
                # A reflected stripe entirely inside the road cannot pass.
                exterior=np.roll(outsides[1-road_index],-start)[ids]>0
                if float(exterior.mean())<.5:continue
                if arc(chain)[-1] < self.policy.minimum_segment_m:
                    continue
                if np.linalg.norm(chain[-1]) < np.linalg.norm(chain[0]):
                    chain = chain[::-1]; inward = inward[::-1]
                # Select the road-facing edge of the white stripe, not its
                # outer edge (otherwise half-width inference is biased 2cm).
                immediate, visible = self.sample(gray, chain+.012*inward)
                facing = float(np.mean(immediate[visible])) if visible.any() else 0.
                if facing < .45:
                    continue
                chains.append((chain,inward,facing))
        return chains

    def extract(self, bgr, prior=None, white_value_min=180, prepared=None, normalized=False):
        bird,s,raw_value,gray,green,contrast,normalized_gray,vicinity=self.prepare(bgr) if prepared is None else prepared
        v=contrast if normalized else raw_value
        if normalized:gray=normalized_gray
        white = (s <= 105) & (v >= white_value_min) & self.visible
        contours, _ = cv2.findContours(white.astype(np.uint8)*255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        fits = {'left': [], 'right': []}
        region_codes=gray.astype(np.uint8)+2*green.astype(np.uint8)+4*white.astype(np.uint8)
        gray_u8=gray.astype(np.uint8)
        minimum_samples=max(5,int(math.ceil(self.policy.minimum_segment_m/self.step)))
        for contour in contours:
            pixels = contour[:, 0, :]
            if len(pixels) < 2*minimum_samples or not vicinity[pixels[:,1],pixels[:,0]].any():
                continue
            chains=native_boundary_chains(pixels,region_codes,gray_u8,self.visible_u8,self.xs,self.ys,
                                          self.step,self.exterior_distances,minimum_samples,self.policy.minimum_segment_m)
            if chains is None:chains=self.contour_chains_reference(pixels,region_codes,gray,minimum_samples)
            for chain,inward,facing in chains:
                edge = fit_curve(chain)
                if edge is None or edge['length'] < self.policy.minimum_segment_m:
                    continue
                direction = chain[-1]-chain[0]
                left_normal = np.array([-direction[1], direction[0]])
                side = 'right' if np.dot(left_normal, np.mean(inward, axis=0)) > 0 else 'left'
                center = inward_center(edge, side, self.width)
                if center is None:
                    continue
                confidence = min(1., edge['length']/.18)*(1-min(.3, edge['residual']/.025))
                score = confidence*(.5+.5*facing)
                # Observation relevance is continuous in distance; a strong
                # far fragment must not eclipse a usable near boundary.
                near=float(np.linalg.norm(center['path'],axis=1).min())
                score *= self.width/(self.width+near)
                if prior is not None:
                    diff = curve_distance(prior, center['path'])
                    if diff is not None:
                        score *= math.exp(-diff/.12)
                edge.update(side=side, center=center, confidence=confidence, score=score)
                # Associate only with this side's established boundary tracks.
                # Unmatched observations remain candidates for reacquisition.
                agreements=[curve_agreement(track['path'],edge['path'])
                            for track in self.boundary_prior[side]]
                agreements=[a for a in agreements if a is not None
                            and a['heading_cos']>self.policy.association_heading_cos]
                if agreements:
                    score *= math.exp(-min(a['distance_m'] for a in agreements)/self.width)
                    edge['score']=score
                fits[side].append(edge)
        # White strips have two contours. Choose a single supported chain per side;
        # matching uses normal offsets and tangent agreement, not image left/right.
        selected = {s: max(fs, key=lambda f:f['score']) if fs else None for s, fs in fits.items()}
        return selected, dict(bird=bird, mask=white, saturation=s,value=raw_value,contrast_value=contrast,exposure_normalized=normalized, fits=selected, candidates=fits,
                             counts={s:len(fs) for s,fs in fits.items()})

    def measure(self, bgr, prior=None):
        # Contrast levels are calibration data, not scene-dependent driving
        # modes. Every level uses the same boundary and curve validity checks.
        # Cache the ground projection and region masks once for this frame.
        prepared = self.prepare(bgr)
        best = None
        best_score = -1.
        first_failure = None
        # Coverage is a sensor/road observation property, not target policy.
        visible_near=self.visible_near_m
        for passes,(normalized,cutoff) in enumerate([(n,c) for n in (True,False) for c in self.white_value_levels],1):
            fits, details = self.extract(bgr, prior, white_value_min=cutoff, prepared=prepared,normalized=normalized)
            details['white_value_min'] = cutoff
            details['photometric_passes']=passes
            measurement, details = self.measure_fits(fits, details, prior)
            if first_failure is None:
                first_failure = (measurement, details)
            if measurement is None or measurement['confidence'] < self.policy.measurement_quality:
                continue
            near=float(np.linalg.norm(measurement['path'],axis=1).min())
            score=measurement['confidence']*self.width/(self.width+near)
            if score>best_score:
                best,best_score=(measurement,details),score
            # A far-only strict mask may hide a continuous darker near edge.
            # Try the other calibrated levels before declaring it the road.
            if near<=max(visible_near,self.policy.observation_search_near_m):
                return measurement,details
        selected=best if best is not None else first_failure
        selected[1]['photometric_passes']=passes
        return selected

    def measure_fits(self, fits, details, prior=None):
        left, right = fits['left'], fits['right']
        curve = None; mode = 'LOST'; confidence = 0.; reason = 'NO EDGE'
        if left is not None and right is not None:
            curve = merge_centers(left['center'], right['center'],left['confidence'],right['confidence'],
                                  self.policy.association_distance_m,self.policy.association_heading_cos,
                                  self.policy.minimum_overlap_m)
            reason = 'EDGES DISAGREE'
            if curve is None:
                # Do not commit independently to one contour on each side.
                # Compare a bounded set of alternatives before rejecting a pair.
                options=[]
                for lf in sorted(details.get('candidates',{}).get('left',[]),key=lambda f:f['score'],reverse=True)[:3]:
                    for rf in sorted(details.get('candidates',{}).get('right',[]),key=lambda f:f['score'],reverse=True)[:3]:
                        if lf is left and rf is right:continue
                        merged=merge_centers(lf['center'],rf['center'],lf['confidence'],rf['confidence'],
                                             self.policy.association_distance_m,self.policy.association_heading_cos,
                                             self.policy.minimum_overlap_m)
                        if merged is not None:
                            options.append((lf['score']+rf['score'],merged,lf,rf))
                if options:
                    _,curve,left,right=max(options,key=lambda option:option[0])
                    details['fits']=dict(left=left,right=right)
            if curve is not None:
                mode = 'TWO EDGES'; confidence = min(left['confidence'], right['confidence'])
            else:
                # Apply the existing tracker acceptance threshold to each
                # independent single-edge measurement. An individually unusable
                # fragment must not veto the other, usable boundary. If both are
                # usable but contradictory, retain EDGES DISAGREE rejection.
                eligible=[f for f in (left,right) if
                          self.policy.single_edge_quality_scale*f['confidence']>=self.policy.measurement_quality]
                if len(eligible)==1:
                    f=eligible[0];curve=f['center'];mode='ONE '+f['side'].upper()
                    confidence=self.policy.single_edge_quality_scale*f['confidence']
                    details['boundary_fallback']='IGNORED INELIGIBLE '+('RIGHT' if f is left else 'LEFT')
                elif len(eligible)==2 and prior is not None:
                    # A far conflicting fragment cannot veto a uniquely
                    # associated near boundary. If both/neither agree, retain
                    # the conflict; never choose a side arbitrarily.
                    associated=[]
                    for f in eligible:
                        agreement=curve_agreement(prior,f['center']['path'])
                        if (agreement is not None and agreement['distance_m']<self.policy.association_distance_m
                                and agreement['heading_cos']>self.policy.association_heading_cos):
                            associated.append(f)
                    if len(associated)==1:
                        f=associated[0];curve=f['center'];mode='ONE '+f['side'].upper()
                        confidence=self.policy.single_edge_quality_scale*f['confidence']
                        details['boundary_fallback']='ASSOCIATED '+f['side'].upper()
        elif left is not None or right is not None:
            f = left if left is not None else right
            curve = f['center']; mode = 'ONE '+f['side'].upper()
            confidence = self.policy.single_edge_quality_scale*f['confidence']
        details['measurement_reason'] = reason if curve is None else ''
        if curve is None:
            return None, details
        return dict(path=curve['path'], mode=mode, confidence=confidence,
                    boundaries={side:[dict(path=f['path'],confidence=f['confidence'])
                                for f in details.get('candidates',{}).get(side,[])]
                                for side in ('left','right')}), details



class TargetTracker:
    """Adapter: observe -> compensate motion -> update road -> select target.

    LocalRoadModel owns association, partial freshness and confirmation.
    Controller readiness cannot suppress otherwise valid road observations.
    """
    def __init__(self, cal):
        self.policy=RoadPolicy.from_config(cal)
        frame=cal.get('control_frame',{})
        self.reference_origin=np.asarray(frame.get('origin_m',[0.,0.]),float)
        self.control_body=frame.get('body_m') if frame.get('geometry_verified') else None
        if self.control_body is not None and (len(self.control_body)!=3 or not all(math.isfinite(v) and v>0 for v in self.control_body)):raise ValueError('Invalid body geometry')
        if self.reference_origin.shape!=(2,) or not np.isfinite(self.reference_origin).all():raise ValueError('Invalid control frame')
        self.pixels=CurvePixels(cal,self.policy)
        self.model=LocalRoadModel(self.policy)
        self.boundaries=BoundaryTrackBank(self.policy)
        self.motion=GroundMotion(cal,self.policy)
        self.last_time=None
        # Legacy snapshot attributes remain readable for recorded replay tools.
        self.path=None;self.last_seen=None;self.confidence=0.;self.mode='LOST'
        self.blind_distance=0.
        self.pending=None;self.pending_count=0;self.pending_confidence=0.

    def update(self, bgr, now, v=0., omega=0., *, require_current=False):
        with geometry_frame():
            return self._update(bgr, now, v, omega, require_current=require_current)

    def _update(self, bgr, now, v=0., omega=0., *, require_current=False):
        p=self.policy
        dt=0. if self.last_time is None else max(0.,now-self.last_time)
        self.last_time=now
        if self.model.path is None and self.path is not None:
            self.model.initialize(self.path,self.last_seen if self.last_seen is not None else now,
                                  self.confidence,self.mode)
        tick=time.perf_counter()
        motion=self.motion.update(bgr,now,v,omega)
        motion_ms=(time.perf_counter()-tick)*1000
        tick=time.perf_counter()
        self.model.predict(motion,dt)
        self.boundaries.predict(motion,now)
        self.pixels.boundary_prior=self.boundaries.tracks
        prior,_=self.model.support(now)
        measurement,details=self.pixels.measure(bgr,self.model.pending if self.model.pending is not None else prior)
        perception_ms=(time.perf_counter()-tick)*1000
        tick=time.perf_counter()
        self.boundaries.update(details.get('candidates',{}),now)
        # Actual observed scene motion wins over a newly published zero command.
        stationary=(motion['speed_m_s']<.005 and abs(motion['yaw_rad_s'])<.02
                    if motion['valid'] else abs(v)<.005 and abs(omega)<.02)
        continuation_margin=None
        if self.control_body is not None and measurement is not None and motion['valid'] and not require_current:
            _,old_support=self.model.support(now)
            if not old_support['reason']:
                continuation_margin=max(0.,self.pixels.width/2-float(self.control_body[2])
                    -float(old_support['sigma_m'])-p.observation_sigma_m/max(measurement['confidence'],.1))
        event=self.model.update(measurement,now,stationary,allow_handoff=bool(motion['valid'] and not require_current),
                                continuation_clearance_m=continuation_margin)
        path,support=self.model.support(now,current_only=require_current)
        model_ms=(time.perf_counter()-tick)*1000
        tick=time.perf_counter()
        target=choose_target(path,forward_min_m=p.forward_min_m,maximum_target_m=p.maximum_target_m,
                             reference_origin_m=self.reference_origin,
                             observation_sigma_m=float(support.get('sigma_m') or p.observation_sigma_m),
                             maximum_reference_sigma_m=p.maximum_sigma_m,
                             curve_analysis_window_m=p.curve_analysis_window_m) if path is not None else None
        observed_mask=support.get('observed_mask',[])
        target_observed=False
        if target is not None:
            position=target['position_m'];positions=arc(target['reference_path'])
            index=min(len(positions)-1,int(np.searchsorted(positions,position)))
            target_observed=all(observed_mask[max(0,index-1):index+1])
        reason=support['reason'] or ('NO FORWARD CURVE POINT' if target is None else '')
        transition_pending=bool(self.model.pending_count and event['continuation_allowed'] and target is not None and not reason and not require_current)
        state=('PREDICTED' if transition_pending else 'CONFIRMING' if self.model.pending_count else
               'LOST' if reason else 'OBSERVED' if event['observed'] else 'PREDICTED')
        self.path=self.model.path;self.last_seen=self.model.last_seen
        self.confidence=self.model.confidence;self.mode=self.model.mode
        self.pending=self.model.pending;self.pending_count=self.model.pending_count
        self.pending_confidence=self.model.pending_confidence
        self.blind_distance=support['blind_distance_m']
        confidence=support.get('confidence',0.)
        # Position uncertainty and track existence are separate from quality.
        # Freshness remains per sample even after a distant accepted observation.
        evidence=('OBSERVED' if event['observed'] else 'CANDIDATE' if event['measurement_usable'] else
                  'AMBIGUOUS' if measurement is not None or any(details.get('counts',{}).values()) else 'ABSENT')
        result=dict(valid=not bool(reason),state=state,road_evidence=evidence,
                    mode=self.mode if event['observed'] else 'LOST' if reason else 'PREDICT',
                    confidence=float(confidence),observed=event['observed'],invalid_reason=reason,
                    observation_confidence=float(measurement['confidence']) if event['observed'] and measurement is not None else 0.,
                    measurement_reject=None if event['observed'] else
                        (event['reject'] if measurement is not None else details.get('measurement_reason','NO EDGE')),
                    age_s=round(support['age_s'],3),reacquired=event['reacquired'],
                    pending_frames=self.pending_count,pending_confidence=self.pending_confidence,
                    transition_pending=transition_pending,association_kind=event['association_kind'],
                    handoff_evidence=event['handoff_evidence'],history_continuity_evidence=event['history_continuity_evidence'] if self.model.pending_count else None,
                    association_prior_path_m=event['prior_path'] if self.model.pending_count else None,
                    candidate_path_m=self.model.pending.tolist() if self.model.pending_count else None,
                    candidate_mode=measurement['mode'] if measurement is not None else None,
                    candidate_confidence=measurement['confidence'] if measurement is not None else 0.,
                    confirmation_required_frames=event['required'] if self.pending_count else 0,
                    measurement_alignment=event['alignment'],confirmation_alignment=event['confirmation_alignment'],
                    blind_distance_m=self.blind_distance,path=path.tolist() if path is not None else [],
                    target_m=(target['point']+self.reference_origin).tolist() if target else None,
                    control_target_m=target['point'].tolist() if target else None,
                    control_reference_origin_m=self.reference_origin.tolist(),
                    control_path_m=target['reference_path'].tolist() if target else [],
                    reference_sigma_m=target['reference_sigma_m'] if target else None,
                    reference_inferred=bool(observed_mask and not all(observed_mask)),
                    target_inferred=bool(target is not None and not target_observed),
                    path_observed_mask=observed_mask,
                    control_target_source=('OBSERVED' if target_observed else 'HISTORY PREDICTION') if target else None,
                    target_distance_m=target['distance_m'] if target else None,
                    target_selection=target['target_selection'] if target else None,
                    target_speed_scale=target['target_speed_scale'] if target else 0.,
                    path_heading_rad=target['path_heading_rad'] if target else None,
                    path_curvature_m_inv=target['curvature'] if target else 0.,
                    near_support_m=target['near_support_m'] if target else None,
                    reliable_length_m=target['reliable_length_m'] if target else 0.,
                    road_model=dict(support,version=2,support_scope='CURRENT' if require_current else 'HISTORY_ALLOWED',
                                    geometry_observed=event['observed'],
                                    model_age_s=None if self.last_seen is None else round(now-self.last_seen,3)),
                    timings_ms=dict(motion=round(motion_ms,1),perception=round(perception_ms,1),
                                    model=round(model_ms,1),target=round((time.perf_counter()-tick)*1000,1)),
                    boundary_tracks=self.boundaries.summary(now),
                    motion=motion_summary(motion),counts=details.get('counts',{}),
                    boundary_fallback=details.get('boundary_fallback'),white_value_min=details.get('white_value_min'),
                    photometric_passes=details.get('photometric_passes',0),
                    exposure_normalized=details.get('exposure_normalized',False))
        details['measured_road']=measurement
        details['ground_motion']=motion
        return result,details


def render_target(bgr,result,details,cal):
    out=bgr.copy(); inverse=np.linalg.inv(np.asarray(cal['H_normalized']))
    def project(path):
        q=np.column_stack((path,np.ones(len(path))))@inverse.T
        good=np.abs(q[:,2])>1e-6
        uv=np.zeros((len(q),2));uv[good]=q[good,:2]/q[good,2,None]*[320,240]
        good &= np.isfinite(uv).all(axis=1) & (abs(uv).max(axis=1)<10000)
        return np.rint(np.clip(uv,-10000,10000)).astype(np.int32),good
    def draw(path,color,thickness):
        pts,valid=project(path)
        for ids in contiguous_runs(valid):
            if len(ids)>1:cv2.polylines(out,[pts[ids]],False,color,thickness)
    for side,f in details['fits'].items():
        if f is not None:draw(f['path'],(0,220,255) if side=='left' else (255,180,0),2)
    path=np.asarray(result.get('path',[]))
    if len(path)>=2:
        fresh=np.asarray(result.get('path_observed_mask',np.ones(len(path),bool)),bool)
        if len(fresh)!=len(path):fresh=np.ones(len(path),bool)
        # An orange segment has actual past observation support. It is not a
        # polynomial drawn outside the local road model's geometric support.
        for ids in contiguous_runs(fresh):
            if len(ids)>=2:draw(path[ids],(0,230,0),2)
        for i in range(0,len(path)-1,2):
            if not (fresh[i] and fresh[i+1]):draw(path[i:i+2],(0,165,255),2)
    if result.get('target_m') is not None:
        pts,good=project(np.asarray([result['target_m']]))
        if good[0]:
            x,y=pts[0]; inside=0<=x<320 and 38<=y<209
            mark=(int(np.clip(x,8,311)),int(np.clip(y,65,195)))
            cv2.drawMarker(out,mark,(0,0,255),cv2.MARKER_CROSS if inside else cv2.MARKER_TRIANGLE_UP,16,2)
            cv2.putText(out,'TARGET' if inside else 'OFF VIEW',(max(0,min(230,mark[0]-30)),max(65,mark[1]-10)),0,.4,(0,0,255),1)
    bird=cv2.resize(cv2.flip(details['bird'],0),(320,240))
    return out,bird
