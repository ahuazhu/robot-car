"""Bounded outside-wheel trial, in actual UNO PWM coordinates.

Paired body-curvature observations are not wheel RPM measurements. Local
interpolation and its rolling adjustment are an estimate; duties above160
are explicitly marked outside the paired training domain. This first field
trial adds at most20 to the outside wheel and never lowers the inside wheel.
"""
import math

from robot.control.controller import compensate_pwm


class BendResponse:
    # (outside PWM, inside PWM, observed curvature / PWM-ratio curvature).
    # Source: docs/data/outer-wheel-simple-20261112.json.
    RIGHT = ((120.,60.,.913), (140.,70.,.738),
             (160.,70.,.714), (160.,100.,.455))
    LEFT = ((120.,60.,1.086), (140.,70.,1.029), (160.,70.,.979))
    ROLLING = (.726 / 1.029, .684 / .738)  # left, right at70/140.
    MAXIMUM_ADDITION = 20.
    MAXIMUM_OUTSIDE = 200.

    def __init__(self, lever, boost):
        self.lever, self.boost = float(lever), float(boost)
        if (not math.isfinite(self.lever) or self.lever<=0
                or not math.isfinite(self.boost) or not 0<=self.boost<=60):
            raise ValueError('Invalid bend actuator coordinates')

    def curvature(self, wire):
        left,right = wire
        total=left+right
        if total<=0 or min(wire)<=0:
            return None
        if abs(right-left)<1e-9:
            return 0.
        direction=0 if right>left else 1
        outer,inner=max(wire),min(wire)
        # Avoid extrapolating the gain surface itself. Its frozen edge remains
        # an uncertain estimate when the command exceeds the observed domain.
        query=(min(160.,max(120.,outer)),min(100.,max(60.,inner)))
        points=self.LEFT if direction==0 else self.RIGHT
        weighted=[]
        for out,inside,gain in points:
            distance=(query[0]-out)**2+(query[1]-inside)**2
            if distance<1e-9:
                factor=gain
                break
            weighted.append((1./distance,gain))
        else:
            factor=sum(w*g for w,g in weighted)/sum(w for w,g in weighted)
        factor*=self.ROLLING[direction]
        return factor*(right-left)/(self.lever*total)

    def allocate(self, wheels, curvature, *, allowed):
        wire=compensate_pwm(wheels,self.boost)
        if (not allowed or not math.isfinite(curvature) or abs(curvature)<1.
                or min(wire)<60 or max(wire)<100 or max(wire)>=self.MAXIMUM_OUTSIDE
                or (wire[1]-wire[0])*curvature<=0):
            return None
        outside=1 if curvature>0 else 0
        estimated=self.curvature(wire)
        if abs(estimated)>=abs(curvature):
            return None
        maximum=min(self.MAXIMUM_OUTSIDE,wire[outside]+self.MAXIMUM_ADDITION)
        choices=[]
        for duty in range(wire[outside]+1,int(maximum)+1):
            candidate=list(wire);candidate[outside]=duty
            actual=self.curvature(candidate)
            choices.append((abs(actual-curvature),duty,tuple(candidate),actual))
        if not choices:
            return None
        _,_,delivered,actual=min(choices)
        if abs(actual-curvature)>=abs(estimated-curvature):
            return None
        peak=max(delivered)
        # Undo the common peak boost exactly once. Session/ramp remains the
        # sole publisher, so preview, logs and UART all see the same command.
        logical=tuple(p*(peak-self.boost)/peak for p in delivered)
        nominal=(delivered[1]-delivered[0])/(self.lever*sum(delivered))
        return dict(wheels=logical,base_wheels=tuple(wheels),base_mean=sum(wheels)/2,
            wire=list(delivered),base_wire=list(wire),estimated_curvature_m_inv=actual,
            nominal_curvature_m_inv=nominal,
            outside_added_pwm=delivered[outside]-wire[outside],
            outside_training_domain=not (120<=max(delivered)<=160 and 60<=min(delivered)<=100),
            source='LOCAL_PAIRED_RESPONSE_ESTIMATE',inside_preserved=True)
