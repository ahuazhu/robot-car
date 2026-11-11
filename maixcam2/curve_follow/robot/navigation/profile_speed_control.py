"""Current supported-path speed planning and measured body-speed feedback.

This scalar regulator cannot authorize a path or bypass mission supervision.
Steering supplies an already accepted pursuit and body envelope; actual applied
effort is fed back after task limits and the motor ramp to prevent windup.
"""

import math

from robot.core.geometry import arc, describe_curve
from robot.navigation.supported_path import supported_segment
from robot.navigation.speed_profile import SpeedLimits, plan_speed_profile
from robot.core.route_confidence import confidence_speed_factor


class ProfileSpeedControl:
    def __init__(self, limits, maximum_effort, response_s, maximum_gap_s=.30,
                 *, high_confidence_gain=1.):
        self.limits = limits
        self.maximum_effort = float(maximum_effort)
        self.response_s = float(response_s)
        self.maximum_gap_s = float(maximum_gap_s)
        if (isinstance(high_confidence_gain, bool)
                or not isinstance(high_confidence_gain, (int, float))
                or not math.isfinite(high_confidence_gain) or not 1 <= high_confidence_gain <= 2):
            raise ValueError('High-confidence tracking gain must be within 1..2')
        self.high_confidence_gain = float(high_confidence_gain)
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.maximum_effort, self.response_s, self.maximum_gap_s)):
            raise ValueError('Invalid speed feedback settings')
        self.reset()

    @classmethod
    def from_settings(cls, settings):
        config = settings.runtime.get('speed_profile', {})
        if not isinstance(config, dict):
            raise ValueError('speed_profile must be an object')
        unknown = set(config) - {'mode', 'maximum_m_s', 'lateral_acceleration_m_s2',
                                 'acceleration_m_s2', 'high_confidence_gain'}
        if unknown or config.get('mode', 'LEGACY') not in ('LEGACY', 'ACTIVE'):
            raise ValueError('Invalid speed_profile configuration')
        if config.get('mode', 'LEGACY') == 'LEGACY':
            return None
        maximum = config.get('maximum_m_s', .30)
        for name in ('maximum_m_s', 'lateral_acceleration_m_s2', 'acceleration_m_s2'):
            value = config.get(name, dict(maximum_m_s=.30,
                lateral_acceleration_m_s2=.15, acceleration_m_s2=.30)[name])
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError('Invalid speed_profile ' + name)
        if maximum > .30:
            raise ValueError('Active speed must be within the measured 0.30m/s ceiling')
        p = settings.crosswalk_policy
        return cls(SpeedLimits(
            maximum, config.get('lateral_acceleration_m_s2', .15),
            config.get('acceleration_m_s2', .30),
            float(p.brake_deceleration_m_s2 or .08), p.command_latency_s,
            settings.turn_lever,
        ), 200 - settings.pwm_boost, p.speed_response_s,
            high_confidence_gain=config.get('high_confidence_gain', 1.))

    def reset(self):
        self.applied_at = None
        self.applied_mean = 0.
        self.reference = None
        self.updated_at = None
        self.status = dict(mode='ACTIVE', selected=False, reason='NOT_EVALUATED')
        self.confidence_status = {}

    def applied(self, now, wheels):
        self.applied_at = float(now)
        self.applied_mean = max(0., sum(wheels) / 2.)
        if not any(wheels):
            self.reference = self.updated_at = None

    def fallback(self, reason):
        self.reference = self.updated_at = None
        self.status = dict(mode='ACTIVE', selected=False, reason=reason,**self.confidence_status)
        return None

    def limit_fallback(self, command):
        """Use the same normalized envelope when measured-speed feedback is unavailable."""
        if not self.confidence_status.get('confidence_valid',False):
            return (0,0)
        peak=max(abs(v) for v in command)
        ceiling=self.maximum_effort*self.confidence_status['envelope_speed_factor']
        scale=min(1.,ceiling/peak) if peak else 0.
        return tuple(round(v*scale) for v in command)

    def command(self, target, path, now, measured_speed, curvature, sigma, front_margin,
                eligible, high_confidence=False, *, clearance_limit_m_s=None):
        try:
            confidence=float(target.get('control_confidence',target.get('confidence',1.)))
            length=(target.get('road_model') or {}).get('length_m')
            if length is None:length=arc(path)[-1]
            length=float(length)
            confidence*=min(1.,max(0.,length/.30))
            factor=confidence_speed_factor(confidence)
        except (ValueError,TypeError,IndexError):
            self.confidence_status={}
            return self.fallback('INVALID_CENTERLINE_CONFIDENCE')
        self.confidence_status=dict(centerline_confidence=confidence,confidence_speed_factor=factor,
                                   confidence_ceiling_m_s=self.limits.maximum_m_s*factor,confidence_valid=True)
        if clearance_limit_m_s is not None:
            if (isinstance(clearance_limit_m_s,bool)
                    or not isinstance(clearance_limit_m_s,(int,float))
                    or not math.isfinite(clearance_limit_m_s) or clearance_limit_m_s<0):
                self.confidence_status={}
                return self.fallback('INVALID_CLEARANCE_SPEED_LIMIT')
            if (target.get('road_model') or {}).get('persistent_history'):
                # Finish-priority recovery retains a crawl instead of waiting
                # indefinitely when the chassis already intersects a boundary.
                clearance_limit_m_s=max(clearance_limit_m_s,
                    self.limits.maximum_m_s*confidence_speed_factor(0.))
        envelope_factor=min(factor,1. if clearance_limit_m_s is None else
                            clearance_limit_m_s/self.limits.maximum_m_s)
        self.confidence_status.update(clearance_ceiling_m_s=clearance_limit_m_s,
                                      envelope_speed_factor=envelope_factor)
        if not eligible:
            return self.fallback('EXISTING_PATH_OR_BODY_GATE')
        if (measured_speed is None or not math.isfinite(measured_speed)
                or measured_speed < -.006):
            return self.fallback('MEASURED_FORWARD_SPEED_REQUIRED')
        if not all(math.isfinite(v) for v in (now, curvature, sigma, front_margin)) or sigma < 0:
            return self.fallback('INVALID_GEOMETRY')
        reference_sigma = target.get('reference_sigma_m')
        if reference_sigma is None or not math.isfinite(reference_sigma) or reference_sigma < 0:
            return self.fallback('MEASURED_PATH_UNCERTAINTY_REQUIRED')
        sigma = max(sigma, reference_sigma)
        segment, reason = supported_segment(dict(target, control_path_m=path),
                                            skip_inferred_prefix=True,include_history=True)
        if segment is None:
            return self.fallback(reason)
        curve = describe_curve(segment)
        if curve is None:
            return self.fallback('INVALID_SUPPORTED_CURVE')
        positions = arc(curve['path'])
        plan = plan_speed_profile(
            positions, curve['curvature'], self.limits,
            distance_margin_m=front_margin + 2 * sigma,
            initial_speed_m_s=max(0., measured_speed),
            centerline_confidence=confidence,
            clearance_limit_m_s=clearance_limit_m_s,
        )
        ratio = self.limits.turn_lever_m * curvature
        if abs(ratio) >= 1 or plan.entry_limit_m_s <= 0:
            return self.fallback('FORWARD_PURSUIT_NOT_FEASIBLE')
        # Use the supported-segment entry envelope as an upper bound now. The
        # unseen distance to its first point grants NO extra braking distance;
        # path authorization still belongs to steering's original body checks.
        ceiling = min(plan.entry_limit_m_s,
                      self.limits.maximum_m_s / (1 + abs(ratio)))
        dt = .05 if self.updated_at is None else now - self.updated_at
        if not 0 < dt < self.maximum_gap_s:
            return self.fallback('CONTROL_GAP')
        fast_path = bool(high_confidence and confidence >= .8 and positions[-1] >= .40)
        reference_acceleration = self.limits.acceleration_m_s2 * (
            self.high_confidence_gain if fast_path else 1.)
        previous_reference = max(0., measured_speed) if self.reference is None else self.reference
        reference = min(ceiling, previous_reference + reference_acceleration * dt)
        actual = (self.applied_mean if self.applied_at is not None
                  and 0 <= now - self.applied_at < self.maximum_gap_s else 0.)
        # Extra effort closes a positive speed error on an independently
        # confirmed gentle path. It changes neither the reference envelope nor
        # the response to overspeed; both wheel limits remain authoritative.
        tracking_gain = (self.high_confidence_gain if fast_path and measured_speed < reference else 1.)
        gain = self.maximum_effort / (self.limits.maximum_m_s * (self.response_s + dt))
        gain *= tracking_gain
        mean = max(0., min(self.maximum_effort / (1 + abs(ratio)),
                          actual + gain * (reference - measured_speed) * dt))
        if measured_speed > ceiling and actual > 0:
            mean = min(mean, actual * ceiling / measured_speed)
        wheels = (round(mean * (1 - ratio)), round(mean * (1 + ratio)))
        self.reference, self.updated_at = reference, now
        self.status = dict(
            mode='ACTIVE', selected=True, reason=plan.entry_reason,
            target_speed_m_s=reference, ceiling_m_s=ceiling,
            measured_speed_m_s=float(measured_speed), initial_overspeed=plan.initial_overspeed,
            supported_length_m=float(positions[-1]),
            usable_length_m=plan.stations_m[-1], limiting_station_m=plan.limiting_station_m,
            first_supported_point_m=segment[0].tolist(),
            effort_mean_pwm=mean, applied_mean_pwm=actual,
            requested_wheels=list(wheels),
            tracking_gain=tracking_gain, high_confidence_boost=tracking_gain > 1.,
            reference_acceleration_m_s2=reference_acceleration,
            **self.confidence_status,
        )
        return wheels
