"""Measured paired-PWM response, with opt-in direct pair inversion.

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
        self._joint_bounds = {}
        self._joint_cache = {}
        for direction, name in ((-1, 'right'), (1, 'left')):
            bank = data[name]
            nodes = np.asarray([p['value'] for p in bank['nodes']], float)
            triangles = np.asarray(bank['triangles'], int)
            coordinates = nodes[triangles, :2]
            matrices = np.concatenate((coordinates, np.ones((*coordinates.shape[:2], 1))), axis=2)
            self.banks[direction] = (np.linalg.inv(matrices), nodes[triangles, 2:],
                                     int(max(nodes[:, 0])))
            self._joint_bounds[direction] = (
                int(min(nodes[:, 0])), int(max(nodes[:, 0])),
                int(min(nodes[:, 1])), int(max(nodes[:, 1])))
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

    def prepare_inner_reduction(self):
        """Precompute on a parked flag change, never in the normal frame loop.

        Bounds come from the accepted measurements, including asymmetric low
        inside support. Chunking limits temporary triangle/weight allocations.
        No unmeasured floor, extrapolation or independent wheel-speed model.
        """
        for direction in (-1, 1):
            if direction in self._joint_cache:
                continue
            low_out, high_out, low_in, high_in = self._joint_bounds[direction]
            high_out, high_in = min(high_out, self.HARDWARE_MAX), min(high_in, self.HARDWARE_MAX)
            outside, inside = np.meshgrid(np.arange(low_out, high_out+1),
                                         np.arange(low_in, high_in+1))
            pairs = np.c_[outside.ravel(), inside.ravel()]
            pairs = pairs[pairs[:, 0] > pairs[:, 1]]
            if direction > 0:
                pairs = pairs[:, ::-1]
            chunks = [self._predict(pairs[i:i+1024], direction)
                      for i in range(0, len(pairs), 1024)]
            motion = np.concatenate(chunks)
            supported = np.isfinite(motion).all(axis=1) & (motion[:, 0] > 0)
            pairs, motion = pairs[supported], motion[supported]
            peak = pairs.max(axis=1)
            logical = pairs*(peak-self.boost)[:, None]/peak[:, None]
            # Neighboring integer outputs on either PWM axis can bracket a
            # desired curvature. Build these edges once while parked; the
            # frame loop must not re-evaluate the triangulated model or sort
            # thousands of candidates.
            grid = np.full((self.HARDWARE_MAX+1, self.HARDWARE_MAX+1), -1, np.int32)
            grid[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))
            edges = []
            for axis in (0, 1):
                lower = grid[:-1, :] if axis == 0 else grid[:, :-1]
                upper = grid[1:, :] if axis == 0 else grid[:, 1:]
                adjacent = (lower >= 0) & (upper >= 0)
                edges.append(np.c_[lower[adjacent], upper[adjacent]])
            strength = direction*motion[:, 1]/motion[:, 0]
            edges = np.concatenate(edges)
            swap = strength[edges[:, 0]] > strength[edges[:, 1]]
            edges[swap] = edges[swap, ::-1]
            self._joint_cache[direction] = dict(
                pairs=pairs, motion=motion, logical=logical,
                curvature=motion[:, 1]/motion[:, 0],
                strength=strength,
                effort_peak=logical.max(axis=1), effort_sum=logical.sum(axis=1),
                lateral=motion[:, 0]*np.abs(motion[:, 1]),
                edges=edges, edge_min=strength[edges[:, 0]],
                edge_mid=(strength[edges[:, 0]]+strength[edges[:, 1]])/2.)

    @staticmethod
    def _feasible_pairs(table, indices, ceiling, effort_limits, lateral_acceleration):
        valid = table['strength'][indices] > 0
        if ceiling is not None:
            valid &= table['motion'][indices, 0] <= ceiling+1e-9
        if lateral_acceleration is not None:
            valid &= table['lateral'][indices] <= lateral_acceleration+1e-9
        if effort_limits is not None:
            valid &= table['effort_peak'][indices] <= effort_limits['peak']+1e-9
            valid &= table['effort_sum'][indices] <= effort_limits['sum']+1e-9
        return valid

    def allocate(self, wheels, curvature, *, allowed, ceiling=None, effort_limits=None,
                 lateral_acceleration=None, allow_inner_reduction=False,
                 reference_speed=None):
        if not allow_inner_reduction:
            return self._allocate_outside_only(wheels, curvature, allowed=allowed,
                ceiling=ceiling, effort_limits=effort_limits,
                lateral_acceleration=lateral_acceleration)
        self.last_reason = 'BASELINE_GATE'
        wire = compensate_pwm(wheels, self.boost)
        self.last_base_wire, self.last_base_motion = wire, None
        if (not allowed or not math.isfinite(curvature) or abs(curvature) < 1.
                or min(wire) <= 0 or (wire[1]-wire[0])*curvature <= 0):
            return None
        base_motion = self.predict(wire)
        self.last_base_motion = base_motion
        if reference_speed is None and base_motion is not None:
            reference_speed = base_motion[0]
        if (reference_speed is None or not math.isfinite(reference_speed)
                or reference_speed <= 0):
            self.last_reason = 'REFERENCE_SPEED_UNAVAILABLE'
            return None
        self.prepare_inner_reduction()
        direction = 1 if curvature > 0 else -1
        outside = 1 if direction > 0 else 0
        inside = 1-outside
        table = self._joint_cache[direction]
        pairs, motion = table['pairs'], table['motion']
        logical, k = table['logical'], table['curvature']
        strength, target = table['strength'], abs(curvature)
        # A physically feasible neighbor on each side proves that a target is
        # reachable to integer-PWM precision. Never jump across an excluded
        # domain/limit hole. Publish only the weaker side, so quantization
        # cannot turn a speed constraint into an oversteering command.
        # Only the nearest integer end is a quantized solution. Otherwise a
        # steep one-PWM edge could make its badly understeering end win merely
        # because that end happens to match the speed reference.
        crossing = ((table['edge_min'] <= target)
                    & (target <= table['edge_mid']+5e-13))
        edges = table['edges'][crossing]
        feasible = self._feasible_pairs(table, edges, ceiling, effort_limits,
                                       lateral_acceleration).all(axis=1)
        candidates = edges[feasible, 0]
        exact = np.flatnonzero(strength == target)
        if len(exact):
            exact = exact[self._feasible_pairs(table, exact, ceiling, effort_limits,
                                               lateral_acceleration)]
            candidates = np.concatenate((candidates, exact))
        limited = not len(candidates)
        if limited:
            valid = self._feasible_pairs(table, slice(None), ceiling, effort_limits,
                                        lateral_acceleration)
            candidates = np.flatnonzero(valid & (strength <= target))
        if not len(candidates):
            self.last_reason = 'NO_FEASIBLE_PAIRED_RESPONSE'
            return None
        # Among quantized target solutions, track the requested body speed.
        # Curvature decimals cannot buy a large unintended slowdown. If no
        # bracket exists, first minimize understeer and report the limitation.
        error = target-strength[candidates]
        speed_error = np.abs(motion[candidates, 0]-reference_speed)
        priorities = (error, speed_error) if limited else (speed_error, error)
        selected = np.arange(len(candidates))
        for priority in priorities:
            best = priority[selected].min()
            selected = selected[priority[selected] <= best+1e-12]
        if len(selected) > 1:
            movement = np.abs(pairs[candidates[selected]]-wire).sum(axis=1)
            selected = selected[[int(np.argmin(movement))]]
        index = int(candidates[selected[0]])
        pair = tuple(map(int, pairs[index]))
        v, w = map(float, motion[index])
        self.last_reason = ('BEST_FEASIBLE_PAIRED_INVERSE' if limited
                            else 'SOLVED_PAIRED_INVERSE')
        base_k = None if base_motion is None else base_motion[1]/base_motion[0]
        return dict(wheels=tuple(map(float, logical[index])), base_wheels=tuple(wheels),
            base_mean=sum(wheels)/2, wire=list(pair), base_wire=list(wire),
            target_speed_m_s=reference_speed, target_yaw_rad_s=reference_speed*curvature,
            target_curvature_m_inv=curvature, base_curvature_m_inv=base_k,
            estimated_speed_m_s=v, estimated_yaw_rad_s=w, estimated_curvature_m_inv=w/v,
            speed_error_m_s=v-reference_speed, base_model_available=base_motion is not None,
            nominal_curvature_m_inv=(pair[1]-pair[0])/(self.lever*sum(pair)),
            outside_added_pwm=pair[outside]-wire[outside], inside_preserved=pair[inside]==wire[inside],
            inside_reduced_pwm=wire[inside]-pair[inside],
            pwm_delta=[pair[0]-wire[0], pair[1]-wire[1]],
            allocation_mode='PAIRED_MODEL_INVERSE',
            hardware_limit_pwm=self.HARDWARE_MAX, model_limit_pwm=self.banks[direction][2],
            limited=limited, reason=self.last_reason, outside_training_domain=False,
            calibration_scope='INNER_70_ROW' if pair[inside] == 70 else 'PAIR_INTERPOLATION',
            source='MEASURED_PAIRED_PWM_RESPONSE_20261112')

    def _allocate_outside_only(self, wheels, curvature, *, allowed, ceiling=None,
                               effort_limits=None, lateral_acceleration=None):
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
