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
        self._cold_banks = {}
        self._joint_bounds = {}
        self._joint_cache = {}
        self._pair_cache = None
        for direction, name in ((-1, 'right'), (1, 'left')):
            bank = data[name]
            nodes = np.asarray([p['value'] for p in bank['nodes']], float)
            triangles = np.asarray(bank['triangles'], int)
            coordinates = nodes[triangles, :2]
            matrices = np.concatenate((coordinates, np.ones((*coordinates.shape[:2], 1))), axis=2)
            inverse, values = np.linalg.inv(matrices), nodes[triangles, 2:]
            self.banks[direction] = (inverse, values,
                                     int(max(nodes[:, 0])))
            warm = np.asarray([p.get('warmup_s', 0.) > 0 for p in bank['nodes']], bool)
            cold = ~np.any(warm[triangles], axis=1)
            self._cold_banks[direction] = (inverse[cold], values[cold])
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

    def _cold_coverage(self, pairs, direction, motion):
        """Does the same prediction have support without a rolling-only node?"""
        inverse, values = self._cold_banks[direction]
        if not len(inverse):
            return np.zeros(len(pairs), bool)
        outside = 0 if direction < 0 else 1
        points = np.c_[pairs[:, outside], pairs[:, 1-outside], np.ones(len(pairs))]
        weights = np.einsum('ni,tij->ntj', points, inverse)
        contains = np.all(weights >= -1e-9, axis=2) & np.all(weights <= 1+1e-9, axis=2)
        predictions = np.einsum('nti,tij->ntj', weights, values)
        agrees = np.all(np.isclose(predictions, motion[:, None, :], rtol=1e-9, atol=1e-10), axis=2)
        return np.any(contains & agrees, axis=1)

    def curvature(self, wire):
        motion = self.predict(wire)
        return motion[1]/motion[0] if motion is not None and motion[0] > 0 else None

    def prepare_inner_reduction(self):
        """Precompute on a parked flag change, never in the normal frame loop.

        Bounds come from the accepted measurements, including asymmetric low
        inside support. Chunking limits temporary triangle/weight allocations.
        No unmeasured floor, extrapolation or independent wheel-speed model.
        """
        if self._pair_cache is not None:
            return
        for direction in (-1, 1):
            if direction in self._joint_cache:
                continue
            low_out, high_out, low_in, high_in = self._joint_bounds[direction]
            high_out, high_in = min(high_out, self.HARDWARE_MAX), min(high_in, self.HARDWARE_MAX)
            outside, inside = np.meshgrid(np.arange(low_out, high_out+1),
                                         np.arange(low_in, high_in+1))
            pairs = np.c_[outside.ravel(), inside.ravel()]
            pairs = pairs[pairs[:, 0] >= pairs[:, 1]]
            if direction > 0:
                pairs = pairs[:, ::-1]
            chunks = [self._predict(pairs[i:i+1024], direction)
                      for i in range(0, len(pairs), 1024)]
            motion = np.concatenate(chunks)
            supported = np.isfinite(motion).all(axis=1) & (motion[:, 0] > 0)
            pairs, motion = pairs[supported], motion[supported]
            cold = np.concatenate([self._cold_coverage(pairs[i:i+1024], direction,
                                                       motion[i:i+1024])
                                   for i in range(0, len(pairs), 1024)])
            peak = pairs.max(axis=1)
            logical = pairs*(peak-self.boost)[:, None]/peak[:, None]
            self._joint_cache[direction] = dict(
                pairs=pairs, motion=motion, logical=logical,
                curvature=motion[:, 1]/motion[:, 0], requires_warmup=~cold)
        # PWM side and actual yaw sign differ near straight travel because
        # the motors are asymmetric. Search both measured banks, including
        # equal-PWM anchors, and keep their shared diagonal only once.
        right, left = self._joint_cache[-1], self._joint_cache[1]
        off_diagonal = left['pairs'][:, 0] != left['pairs'][:, 1]
        pairs = np.concatenate((right['pairs'], left['pairs'][off_diagonal]))
        motion = np.concatenate((right['motion'], left['motion'][off_diagonal]))
        logical = np.concatenate((right['logical'], left['logical'][off_diagonal]))
        requires_warmup = np.concatenate((right['requires_warmup'],
                                         left['requires_warmup'][off_diagonal]))
        k = motion[:, 1]/motion[:, 0]
        grid = np.full((self.HARDWARE_MAX+1, self.HARDWARE_MAX+1), -1, np.int32)
        grid[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))
        adjacent_edges = []
        for axis in (0, 1):
            lower = grid[:-1, :] if axis == 0 else grid[:, :-1]
            upper = grid[1:, :] if axis == 0 else grid[:, 1:]
            adjacent = (lower >= 0) & (upper >= 0)
            adjacent_edges.append(np.c_[lower[adjacent], upper[adjacent]])
        edges = np.concatenate(adjacent_edges)
        swap = k[edges[:, 0]] > k[edges[:, 1]]
        edges[swap] = edges[swap, ::-1]
        common = dict(pairs=pairs, motion=motion, logical=logical, curvature=k,
            effort_peak=logical.max(axis=1), effort_sum=logical.sum(axis=1),
            lateral=motion[:, 0]*np.abs(motion[:, 1]), requires_warmup=requires_warmup)
        self._pair_cache = {}
        for direction in (-1, 1):
            ordered = edges if direction > 0 else edges[:, ::-1]
            strength = direction*k
            self._pair_cache[direction] = dict(common, strength=strength,
                edges=ordered, edge_min=strength[ordered[:, 0]],
                edge_mid=(strength[ordered[:, 0]]+strength[ordered[:, 1]])/2.)
        # A zero-curvature request has no turn sign. Its candidate must still
        # be the nearest end of a real feasible zero crossing; an arbitrary
        # low-speed sharp-turn point is not a straight-drive fallback.
        zero_edges = edges[(k[edges[:, 0]] <= 0) & (k[edges[:, 1]] >= 0)].copy()
        swap = np.abs(k[zero_edges[:, 0]]) > np.abs(k[zero_edges[:, 1]])
        zero_edges[swap] = zero_edges[swap, ::-1]
        self._pair_cache[0] = dict(common, strength=np.ones(len(k)), edges=zero_edges)

    @staticmethod
    def _feasible_pairs(table, indices, ceiling, effort_limits, lateral_acceleration,
                        allow_warmup=True):
        valid = table['strength'][indices] > 0
        if not allow_warmup:
            valid &= ~table['requires_warmup'][indices]
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
                 reference_speed=None, reference_effort=None, motion_requested=None,
                 allow_warmup=True):
        if motion_requested is False:
            self.last_reason = 'STOP_REQUESTED'
            self.last_base_wire, self.last_base_motion = compensate_pwm(wheels, self.boost), None
            return None
        if not allow_inner_reduction:
            return self._allocate_outside_only(wheels, curvature, allowed=allowed,
                ceiling=ceiling, effort_limits=effort_limits,
                lateral_acceleration=lateral_acceleration)
        self.last_reason = 'BASELINE_GATE'
        wire = compensate_pwm(wheels, self.boost)
        self.last_base_wire, self.last_base_motion = wire, None
        if motion_requested is None:
            motion_requested = any(wire)
        if not motion_requested:
            self.last_reason = 'STOP_REQUESTED'
            return None
        if not allowed or not math.isfinite(curvature):
            return None
        base_motion = self.predict(wire)
        if base_motion is not None and base_motion[0] <= 0:
            base_motion = None
        self.last_base_motion = base_motion
        if reference_speed is not None and (isinstance(reference_speed, bool)
                                            or not math.isfinite(reference_speed)
                                            or reference_speed <= 0):
            self.last_reason = 'REFERENCE_SPEED_UNAVAILABLE'
            return None
        if reference_effort is None:
            reference_effort = sum(wheels)/2.
        if (isinstance(reference_effort, bool) or not math.isfinite(reference_effort)
                or reference_effort < 0):
            self.last_reason = 'REFERENCE_EFFORT_UNAVAILABLE'
            return None
        self.prepare_inner_reduction()
        direction = 1 if curvature > 0 else -1 if curvature < 0 else 0
        outside = 1 if direction > 0 else 0
        inside = 1-outside
        table = self._pair_cache[direction]
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
        if direction:
            crossing = ((table['edge_min'] <= target)
                        & (target <= table['edge_mid']+5e-13))
            edges = table['edges'][crossing]
        else:
            edges = table['edges']
        feasible = self._feasible_pairs(table, edges, ceiling, effort_limits,
                                       lateral_acceleration, allow_warmup).all(axis=1)
        candidates = edges[feasible, 0]
        exact = np.flatnonzero(k == curvature)
        if len(exact):
            exact = exact[self._feasible_pairs(table, exact, ceiling, effort_limits,
                                               lateral_acceleration, allow_warmup)]
            candidates = np.concatenate((candidates, exact))
        limited = not len(candidates)
        if limited and direction:
            valid = self._feasible_pairs(table, slice(None), ceiling, effort_limits,
                                        lateral_acceleration, allow_warmup)
            candidates = np.flatnonzero(valid & (strength <= target))
        if not len(candidates):
            self.last_reason = ('NO_FEASIBLE_PAIRED_RESPONSE' if direction
                                else 'NO_FEASIBLE_STRAIGHT_RESPONSE')
            return None
        # Among quantized target solutions, track the requested body speed.
        # Curvature decimals cannot buy a large unintended slowdown. If no
        # bracket exists, first minimize understeer and report the limitation.
        error = np.abs(k[candidates]-curvature)
        if limited:
            closest = error <= error.min()+1e-12
            candidates, error = candidates[closest], error[closest]
        reference_values = (motion[candidates, 0] if reference_speed is not None
                            else table['effort_sum'][candidates]/2.)
        requested_reference = reference_speed if reference_speed is not None else reference_effort
        reference_error = np.abs(reference_values-requested_reference)
        reachable_reference = (float(reference_values.min()), float(reference_values.max()))
        limited_reference = not (reachable_reference[0]-1e-9 <= requested_reference
                                 <= reachable_reference[1]+1e-9)
        priorities = (reference_error, error)
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
                            else 'REFERENCE_LIMITED_PAIRED_INVERSE' if limited_reference
                            else 'SOLVED_PAIRED_INVERSE')
        base_k = None if base_motion is None or base_motion[0] <= 0 else base_motion[1]/base_motion[0]
        pair_direction = -1 if pair[0] >= pair[1] else 1
        return dict(wheels=tuple(map(float, logical[index])), base_wheels=tuple(wheels),
            base_mean=sum(wheels)/2, wire=list(pair), base_wire=list(wire),
            target_speed_m_s=reference_speed,
            target_yaw_rad_s=None if reference_speed is None else reference_speed*curvature,
            target_effort_mean_pwm=reference_effort,
            effort_error_mean_pwm=float(table['effort_sum'][index]/2.-reference_effort),
            reference_kind='SPEED' if reference_speed is not None else 'EFFORT',
            feasible_reference_range=list(reachable_reference), limited_reference=limited_reference,
            limited_speed=limited_reference if reference_speed is not None else None,
            target_curvature_m_inv=curvature, base_curvature_m_inv=base_k,
            estimated_speed_m_s=v, estimated_yaw_rad_s=w, estimated_curvature_m_inv=w/v,
            speed_error_m_s=None if reference_speed is None else v-reference_speed,
            base_model_available=base_motion is not None and base_motion[0] > 0,
            nominal_curvature_m_inv=(pair[1]-pair[0])/(self.lever*sum(pair)),
            outside_added_pwm=pair[outside]-wire[outside], inside_preserved=pair[inside]==wire[inside],
            inside_reduced_pwm=wire[inside]-pair[inside],
            pwm_delta=[pair[0]-wire[0], pair[1]-wire[1]],
            allocation_mode='PAIRED_MODEL_INVERSE',
            hardware_limit_pwm=self.HARDWARE_MAX, model_limit_pwm=self.banks[pair_direction][2],
            limited=limited, limited_curvature=limited, reason=self.last_reason,
            motion_requested=True, outside_training_domain=False,
            requires_warmup=bool(table['requires_warmup'][index]),
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
