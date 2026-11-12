"""Measured paired-PWM response; invert only the outside wheel.

The table describes optical body motion, not independent wheel RPM. Between
measured combinations it is an interpolation estimate; outside its triangles
the original command is retained. No speed integrator, floor or pulse is added.
"""
import json
import math
from pathlib import Path

import numpy as np
from robot.control.controller import compensate_pwm


class BendResponse:
    HARDWARE_MAX = 255

    def __init__(self, lever, boost):
        self.lever, self.boost = float(lever), float(boost)
        if not math.isfinite(self.lever) or self.lever <= 0 or not 0 <= self.boost <= 60:
            raise ValueError('Invalid actuator coordinates')
        data = json.loads(Path(__file__).with_name('bend_response_data.json').read_text())
        self.banks = {}
        for direction, name in ((-1, 'right'), (1, 'left')):
            bank = data[name]
            nodes = np.asarray([p['value'] for p in bank['nodes']], float)
            triangles = np.asarray(bank['triangles'], int)
            coordinates = nodes[triangles, :2]
            matrices = np.concatenate((coordinates, np.ones((*coordinates.shape[:2], 1))), axis=2)
            self.banks[direction] = (np.linalg.inv(matrices), nodes[triangles, 2:],
                                     int(max(nodes[:, 0])))
        self.last_reason = 'NOT_EVALUATED'
        self.last_base_wire = None
        self.last_base_motion = None

    def _predict(self, pairs, direction):
        pairs = np.asarray(pairs, float).reshape(-1, 2)
        outside = 0 if direction < 0 else 1
        points = np.c_[pairs[:, outside], pairs[:, 1-outside], np.ones(len(pairs))]
        inverse, values, _ = self.banks[direction]
        weights = np.einsum('ni,tij->ntj', points, inverse)
        contains = np.all(weights >= -1e-9, axis=2) & np.all(weights <= 1+1e-9, axis=2)
        covered = np.any(contains, axis=1)
        chosen = np.argmax(contains, axis=1)
        result = np.einsum('ni,nij->nj', weights[np.arange(len(pairs)), chosen], values[chosen])
        result[~covered] = np.nan
        return result

    def predict(self, wire):
        if not any(wire):
            return (0., 0.)
        direction = -1 if wire[0] >= wire[1] else 1
        result = self._predict([wire], direction)[0]
        return tuple(map(float, result)) if np.isfinite(result).all() else None

    def curvature(self, wire):
        motion = self.predict(wire)
        return motion[1]/motion[0] if motion is not None and motion[0] > 0 else None

    def allocate(self, wheels, curvature, *, allowed, ceiling=None, effort_limits=None,
                 lateral_acceleration=None):
        self.last_reason = 'BASELINE_GATE'
        wire = compensate_pwm(wheels, self.boost)
        self.last_base_wire = wire
        self.last_base_motion = None
        if (not allowed or not math.isfinite(curvature) or abs(curvature) < 1.
                or min(wire) <= 0 or (wire[1]-wire[0])*curvature <= 0):
            return None
        base_motion = self.predict(wire)
        self.last_base_motion = base_motion
        if base_motion is None:
            self.last_reason = 'OUTSIDE_MEASURED_PAIR_DOMAIN'
            return None
        direction = 1 if curvature > 0 else -1
        outside = 1 if direction > 0 else 0
        maximum = self.banks[direction][2]
        base_k = base_motion[1]/base_motion[0]
        if direction*base_k >= abs(curvature):
            self.last_reason = 'NO_UNDERSTEER_IN_MODEL'
            return None
        duties = np.arange(wire[outside]+1, maximum+1)
        if not len(duties):
            self.last_reason = 'MEASURED_OUTSIDE_LIMIT'
            return None
        pairs = np.tile(wire, (len(duties), 1))
        pairs[:, outside] = duties
        motion = self._predict(pairs, direction)
        k = motion[:, 1]/motion[:, 0]
        logical = pairs*(duties-self.boost)[:, None]/duties[:, None]
        valid = np.isfinite(motion).all(axis=1) & (motion[:, 0] > 0)
        valid &= direction*k > direction*base_k+1e-9
        if ceiling is not None:
            valid &= motion[:, 0] <= ceiling+1e-9
        if lateral_acceleration is not None:
            valid &= motion[:, 0]*np.abs(motion[:, 1]) <= lateral_acceleration+1e-9
        if effort_limits is not None:
            valid &= np.max(logical, axis=1) <= effort_limits['peak']+1e-9
            valid &= np.sum(logical, axis=1) <= effort_limits['sum']+1e-9
        error = np.abs(k-curvature)
        valid &= error < abs(base_k-curvature)-1e-9
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            self.last_reason = 'NO_FEASIBLE_OUTSIDE_IMPROVEMENT'
            return None
        # Stop at the first reachable crossing. A later, weaker high-PWM
        # branch must not win merely through an integer-rounding coincidence.
        crossing = candidates[direction*k[candidates] >= abs(curvature)]
        if len(crossing):
            candidates = candidates[candidates <= crossing[0]]
        index = int(candidates[np.argmin(error[candidates])])
        pair = tuple(map(int, pairs[index]))
        v, w = map(float, motion[index])
        limited = not len(crossing)
        self.last_reason = 'BEST_FEASIBLE_OUTSIDE_ONLY' if limited else 'SOLVED_OUTSIDE_ONLY'
        return dict(wheels=tuple(map(float, logical[index])), base_wheels=tuple(wheels),
            base_mean=sum(wheels)/2, wire=list(pair), base_wire=list(wire),
            target_speed_m_s=base_motion[0], target_yaw_rad_s=base_motion[0]*curvature,
            target_curvature_m_inv=curvature, base_curvature_m_inv=base_k,
            estimated_speed_m_s=v, estimated_yaw_rad_s=w, estimated_curvature_m_inv=w/v,
            nominal_curvature_m_inv=(pair[1]-pair[0])/(self.lever*sum(pair)),
            outside_added_pwm=pair[outside]-wire[outside], inside_preserved=True,
            hardware_limit_pwm=self.HARDWARE_MAX, model_limit_pwm=maximum,
            limited=limited, reason=self.last_reason, outside_training_domain=False,
            calibration_scope='INNER_70_ROW' if pair[1-outside] == 70 else 'PAIR_INTERPOLATION',
            source='MEASURED_PAIRED_PWM_RESPONSE_20261112')
