"""Stateless nonlinear motor-output adapter; no velocity regulator or timers.

Speed knots and paired response factors are measured optical-coordinate
references. The continuous model ends at PWM200; the UNO protocol permits255.
Unmeasured201..255 are not extrapolated as known motor speeds.
"""
import math
import numpy as np
from robot.control.controller import compensate_pwm

class BendResponse:
    PWM=np.array([50,60,70,80,90,120,150,180,200],float)
    SPEED=np.array([.060709456,.085130098,.125667112,.156915681,.186321590,
                    .226128018,.254343242,.254343242,.277048724])
    RIGHT=((120,60,1.18,.738),(140,70,1.139,.808),(160,70,1.145,.878),
           (160,100,1.065,.867),(140,50,1.334,.909))
    LEFT=((120,60,1.16,.66),(140,70,1.09,.821),(160,70,1.09,.871))
    HARDWARE_MAX=255
    MODEL_MAX=200

    def __init__(self,lever,boost):
        self.lever,self.boost=float(lever),float(boost)
        if not math.isfinite(self.lever) or self.lever<=0 or not 0<=self.boost<=60:
            raise ValueError('Invalid actuator coordinates')
        left,right=np.meshgrid(np.arange(60,self.MODEL_MAX+1),
                               np.arange(60,self.MODEL_MAX+1),indexing='ij')
        self.pairs=np.c_[left.ravel(),right.ravel()]
        self.twists=self._predict(self.pairs)
        self.last_reason='NOT_EVALUATED'

    def _predict(self, pairs):
        p=np.asarray(pairs,float).reshape(-1,2)
        speeds=np.interp(p,self.PWM,self.SPEED)
        speeds=np.where(p>0,speeds,0.)
        v=np.mean(speeds,axis=1)
        w=(speeds[:,1]-speeds[:,0])/(2*self.lever)
        outside=np.max(p,axis=1);inside=np.min(p,axis=1)
        for direction,bank in ((-1,self.RIGHT),(1,self.LEFT)):
            mask=(w*direction)>0
            if not np.any(mask):continue
            q=np.c_[np.clip(outside[mask],120,160),
                     np.clip(inside[mask],50 if direction<0 else 60,100)]
            points=np.asarray(bank,float)
            distance=np.sum((q[:,None,:]-points[None,:,:2])**2,axis=2)
            weight=1/np.maximum(1e-6,distance)
            factors=(weight@points[:,2:])/weight.sum(axis=1)[:,None]
            # Blend the low-duty paired fit into the high-duty paired records.
            high=np.clip((outside[mask]-90)/30,0,1)
            differential=np.clip((outside[mask]-inside[mask])/20,0,1)
            v[mask]*=1+high*differential*(factors[:,0]-1)
            low_gain=.6902793 if direction<0 else 1.0198228
            w[mask]*=(1-high)*low_gain+high*factors[:,1]
        return np.c_[v,w]

    def predict(self, wire):
        if not any(wire):return (0.,0.)
        return tuple(float(x) for x in self._predict([wire])[0])

    def curvature(self,wire):
        if min(wire)<50 or max(wire)>self.MODEL_MAX:return None
        v,w=self.predict(wire)
        return w/v if v>0 else 0.

    def allocate(self,wheels,curvature,*,allowed,ceiling=None):
        self.last_reason='BASELINE_GATE'
        wire=compensate_pwm(wheels,self.boost)
        if (not allowed or not math.isfinite(curvature) or abs(curvature)<1.
                or min(wire)<60 or (wire[1]-wire[0])*curvature<=0):return None
        if max(wire)>self.MODEL_MAX:
            self.last_reason='OUTSIDE_MEASURED_PWM_DOMAIN';return None
        # The existing speed regulator still owns forward effort. Translate
        # that effort into equivalent body speed, then solve the two outputs
        # jointly for the selected DWPP arc. Do not add a new speed integrator.
        linear,_=self.predict(wire)
        target=min(linear,float(ceiling)) if ceiling is not None else linear
        return self.solve(wheels,wire,target,curvature,ceiling)

    def solve(self,base_wheels,base_wire,linear,curvature,ceiling,*,angular_window=None):
        angular=linear*curvature
        valid=self.twists[:,1]*angular>=0
        if ceiling is not None:valid &= self.twists[:,0]<=ceiling+1e-9
        if angular_window is not None:
            low,high=angular_window
            valid &= (self.twists[:,1]>=low-1e-9) & (self.twists[:,1]<=high+1e-9)
        if not np.any(valid):
            self.last_reason='NO_CONTINUOUS_PAIR_WITHIN_SPEED_CEILING';return None
        cost=(self.twists[:,0]-linear)**2+(.15*(self.twists[:,1]-angular))**2
        # Only break equal tracking errors with duty changes; no fixed bonus
        # window, maximum addition, controller gain or velocity bias is used.
        cost+=1e-11*np.sum((self.pairs-np.array(base_wire))**2,axis=1)
        index=int(np.argmin(np.where(valid,cost,np.inf)))
        pair=tuple(int(x) for x in self.pairs[index]);v,w=map(float,self.twists[index])
        peak=max(pair);logical=tuple(p*(peak-self.boost)/peak for p in pair)
        k=w/v
        limited=abs(v-linear)>.015 or abs(k-curvature)>.20
        self.last_reason='MODEL_DOMAIN_LIMIT' if limited else 'SOLVED'
        outside=1 if curvature>0 else 0
        return dict(wheels=logical,base_wheels=tuple(base_wheels),
            base_mean=sum(base_wheels)/2,wire=list(pair),base_wire=list(base_wire),
            target_speed_m_s=linear,target_yaw_rad_s=angular,
            estimated_speed_m_s=v,estimated_yaw_rad_s=w,
            estimated_curvature_m_inv=k,
            nominal_curvature_m_inv=(pair[1]-pair[0])/(self.lever*sum(pair)),
            outside_added_pwm=pair[outside]-base_wire[outside],
            inside_preserved=pair[1-outside]==base_wire[1-outside],
            hardware_limit_pwm=self.HARDWARE_MAX,model_limit_pwm=self.MODEL_MAX,
            limited=limited,reason=self.last_reason,
            outside_training_domain=max(pair)>160,
            source='NONLINEAR_PAIRED_RESPONSE_ESTIMATE')
