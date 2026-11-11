"""Supported-path speed envelope, optional DWPP, and forward-speed feedback.

This scalar regulator cannot authorize a path or bypass mission supervision.
Steering supplies an already accepted pursuit and body envelope; actual applied
effort is fed back after task limits and the motor ramp to prevent windup.
"""

import math

from robot.core.geometry import arc, describe_curve
from robot.control.controller import MAX_MOTOR_PWM
from robot.navigation.supported_path import supported_segment
from robot.navigation.speed_profile import SpeedLimits, plan_speed_profile
from robot.core.route_confidence import confidence_speed_factor
from robot.navigation.dwpp import plan_twist


class ProfileSpeedControl:
    def __init__(self, limits, maximum_effort, response_s, maximum_gap_s=.30,
                 *, high_confidence_gain=1., controller='PURE_PURSUIT',
                 angular_acceleration=2., angular_deceleration=2.):
        self.limits = limits
        self.maximum_effort = float(maximum_effort)
        self.response_s = float(response_s)
        self.maximum_gap_s = float(maximum_gap_s)
        if (isinstance(high_confidence_gain, bool)
                or not isinstance(high_confidence_gain, (int, float))
                or not math.isfinite(high_confidence_gain) or not 1 <= high_confidence_gain <= 2):
            raise ValueError('High-confidence tracking gain must be within 1..2')
        self.high_confidence_gain = float(high_confidence_gain)
        if controller not in ('PURE_PURSUIT','DWPP'):
            raise ValueError('Unknown path controller')
        if any(isinstance(v,bool) or not isinstance(v,(int,float))
               or not math.isfinite(v) or v<=0
               for v in (angular_acceleration,angular_deceleration)):
            raise ValueError('Angular acceleration limits must be positive and finite')
        self.controller=controller
        self.angular_acceleration=float(angular_acceleration)
        self.angular_deceleration=float(angular_deceleration)
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
                                 'acceleration_m_s2', 'high_confidence_gain','controller',
                                 'angular_acceleration_rad_s2','angular_deceleration_rad_s2'}
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
        ), MAX_MOTOR_PWM - settings.pwm_boost, p.speed_response_s,
            high_confidence_gain=config.get('high_confidence_gain', 1.),
            controller=config.get('controller','PURE_PURSUIT'),
            angular_acceleration=config.get('angular_acceleration_rad_s2',2.),
            angular_deceleration=config.get('angular_deceleration_rad_s2',2.))

    @classmethod
    def for_driving(cls, settings):
        """The deployed app has one motion planner; baseline controllers are offline only."""
        control = cls.from_settings(settings)
        if control is None or control.controller != 'DWPP':
            raise ValueError('Driving requires ACTIVE DWPP; legacy entry is disabled')
        return control

    def reset(self):
        self.applied_at = None
        self.applied_mean = 0.
        self.reference = None
        self.updated_at = None
        self.status = dict(mode='ACTIVE',controller=self.controller,selected=False,reason='NOT_EVALUATED')
        self.confidence_status = {}
        self.dwpp_state = None
        self.dwpp_proposal = None

    def applied(self, now, wheels, *, delivered_wheels=None):
        self.applied_at = float(now)
        self.status.pop('applied_curvature_m_inv',None)
        # Keep feedback effort in the pre-boost PWM domain. The final integer
        # wire command supplies its direction/ratio only: a +37 starting boost
        # is not 37 additional units of this regulator's integrated effort.
        self.applied_mean = max(0., sum(wheels) / 2.)
        if self.controller=='DWPP':
            proposal=self.dwpp_proposal
            self.dwpp_proposal=None
            self.dwpp_state=None
            if proposal is not None and any(wheels):
                requested,v,w=proposal
                norm=sum(x*x for x in requested)
                scale=sum(a*b for a,b in zip(wheels,requested))/norm if norm else 0.
                if (0<scale<=1.02 and max(abs(a-scale*b) for a,b in zip(wheels,requested))<=1.5):
                    # Command-domain estimate after the shared motor ramp or
                    # task scaling. It is not measured wheel/body velocity.
                    delivered = wheels if delivered_wheels is None else delivered_wheels
                    total = sum(delivered)
                    if total > 0 and min(delivered) >= 0:
                        curvature = ((delivered[1]-delivered[0]) /
                                     (self.limits.turn_lever_m*total))
                        linear = v*min(1.,scale)
                        self.dwpp_state=(float(now),linear,linear*curvature)
                        self.status['applied_curvature_m_inv']=curvature
        if not any(wheels):
            self.reference = self.updated_at = None

    def fallback(self, reason):
        self.reference = self.updated_at = None
        self.dwpp_proposal=self.dwpp_state=None
        self.status = dict(mode='ACTIVE',controller=self.controller,selected=False,
                           reason=reason,**self.confidence_status)
        return None

    def constrain_clearance(self, command, ceiling):
        """Brake a chosen arc after its actual curvature's body sweep is checked.

        A common scale keeps that checked arc unchanged. As with any emergency
        speed reduction, braking can override the nominal acceleration window;
        the exception is explicit in telemetry and cannot authorize motion.
        """
        if self.controller!='DWPP' or self.dwpp_proposal is None:return command
        requested,v,w=self.dwpp_proposal
        if v<=ceiling:return command
        scale=max(0.,ceiling/v)
        wheels=tuple(x*scale for x in command)
        self.reference=v*scale
        self.dwpp_proposal=(wheels,v*scale,w*scale)
        self.status.update(reason='BODY_CLEARANCE',ceiling_m_s=ceiling,
                           target_speed_m_s=v*scale,reference_angular_rad_s=w*scale,
                           requested_wheels=list(wheels),effort_mean_pwm=sum(wheels)/2.)
        self.status['dwpp']['overrides'].append('BODY_SWEEP_BRAKING')
        return wheels

    def limit_fallback(self, command):
        """Use the same normalized envelope when measured-speed feedback is unavailable."""
        if not self.confidence_status.get('confidence_valid',False):
            return (0,0)
        peak=max(abs(v) for v in command)
        ceiling=self.maximum_effort*self.confidence_status['envelope_speed_factor']
        scale=min(1.,ceiling/peak) if peak else 0.
        return tuple(v*scale for v in command)

    def command(self, target, path, now, measured_speed, curvature, sigma, front_margin,
                eligible, high_confidence=False, *, clearance_limit_m_s=None,
                measured_yaw_rate=None, motion_feedback=True, task_limit_m_s=None):
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
        feedback = bool(motion_feedback and measured_speed is not None
                        and math.isfinite(measured_speed) and measured_speed >= -.006
                        and measured_yaw_rate is not None and math.isfinite(measured_yaw_rate))
        state_source = 'MEASURED'
        if self.controller == 'DWPP' and not feedback:
            # This is command state, never odometry. Do not integrate an error
            # against an invented speed or use this to transport a road/crossing.
            recent = (self.dwpp_state is not None
                      and 0 <= now-self.dwpp_state[0] < self.maximum_gap_s)
            measured_speed, measured_yaw_rate = self.dwpp_state[1:] if recent else (0., 0.)
            state_source = 'APPLIED_ESTIMATE' if recent else 'CRAWL_SEED'
        elif measured_speed is None or not math.isfinite(measured_speed) or measured_speed < -.006:
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
            exact_curvature_required=self.controller!='DWPP',
        )
        ratio = self.limits.turn_lever_m * curvature
        if ((self.controller!='DWPP' and abs(ratio) >= 1)
                or plan.entry_limit_m_s <= 0):
            return self.fallback('FORWARD_PURSUIT_NOT_FEASIBLE')
        # Use the supported-segment entry envelope as an upper bound now. The
        # unseen distance to its first point grants NO extra braking distance;
        # path authorization still belongs to steering's original body checks.
        ceiling = min(plan.entry_limit_m_s,
                      self.limits.maximum_m_s / (1 + abs(ratio)))
        if task_limit_m_s is not None:
            if not math.isfinite(task_limit_m_s) or task_limit_m_s < 0:
                return self.fallback('INVALID_TASK_SPEED_LIMIT')
            ceiling = min(ceiling, task_limit_m_s)
        if self.controller == 'DWPP' and not feedback:
            ceiling = min(ceiling, self.limits.maximum_m_s * .25)
        if self.controller=='DWPP':
            if measured_yaw_rate is None or not math.isfinite(measured_yaw_rate):
                return self.fallback('DWPP_MEASURED_YAW_REQUIRED')
            # Reacquisition after PWM fallback still has a real control period.
            # A fixed 50ms seed window repeatedly suppresses turn corrections
            # on this camera's 150..220ms cycle. Unsent proposals own no time.
            previous_at=self.applied_at if self.dwpp_state is None else self.dwpp_state[0]
            dt=.05 if previous_at is None else now-previous_at
        else:
            dt = .05 if self.updated_at is None else now - self.updated_at
        if not 0 < dt < self.maximum_gap_s:
            if self.controller != 'DWPP' or dt <= 0:
                return self.fallback('CONTROL_GAP')
            # A long command gap is a new bounded crawl seed, not a giant
            # acceleration window and not an invitation to switch algorithms.
            self.dwpp_state = None
            dt = .05
            ceiling = min(ceiling, self.limits.maximum_m_s * .25)
            feedback = False
            measured_speed = measured_yaw_rate = 0.
            state_source = 'GAP_CRAWL_SEED'
        fast_path = bool(high_confidence and confidence >= .8 and positions[-1] >= .40)
        reference_acceleration = self.limits.acceleration_m_s2 * (
            self.high_confidence_gain if fast_path else 1.)
        previous_reference = max(0., measured_speed) if self.reference is None else self.reference
        dwpp_status=None
        if self.controller=='DWPP':
            source='APPLIED_REFERENCE' if self.dwpp_state is not None else 'MEASURED_SEED'
            current=(self.dwpp_state[1:] if self.dwpp_state is not None else
                     (min(self.limits.maximum_m_s,max(0.,measured_speed)),
                      max(-self.limits.maximum_m_s/self.limits.turn_lever_m,
                          min(self.limits.maximum_m_s/self.limits.turn_lever_m,measured_yaw_rate))))
            solution=plan_twist(*current,max(-1/self.limits.turn_lever_m,
                min(1/self.limits.turn_lever_m,curvature)),ceiling,dt,
                maximum=self.limits.maximum_m_s,lever=self.limits.turn_lever_m,
                acceleration=reference_acceleration,deceleration=self.limits.deceleration_m_s2,
                angular_acceleration=self.angular_acceleration,angular_deceleration=self.angular_deceleration)
            reference=solution.linear
            ratio=self.limits.turn_lever_m*solution.angular/reference if reference > 1e-10 else 0.
            # Reference/effort conversion preserves the selected wheel ratio.
            ratio=max(-1.,min(1.,ratio))
            dwpp_status=dict(source=source,current_linear_m_s=current[0],current_angular_rad_s=current[1],
                dt_s=dt,window=dict(v_min=solution.window.v_min,v_max=solution.window.v_max,
                    w_min=solution.window.w_min,w_max=solution.window.w_max),
                intersection=solution.intersection,overrides=list(solution.overrides),
                requested_curvature_m_inv=curvature,selected_curvature_m_inv=solution.curvature)
            if abs(curvature)*self.limits.turn_lever_m > 1:
                dwpp_status['overrides'].append('FORWARD_CURVATURE_PROJECTION')
        else:
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
        if self.controller == 'DWPP' and not feedback:
            # Bounded feedforward during missing/noisy flow. Never wind up the
            # measured-speed integrator while the sensor cannot observe motion.
            mean = min(self.maximum_effort / (1 + abs(ratio)),
                       self.maximum_effort * reference / self.limits.maximum_m_s)
        if self.controller == "DWPP" and reference <= 1e-10:
            mean = 0.  # Hard zero also clears previously accumulated effort at measured rest.
        wheels = (mean * (1 - ratio), mean * (1 + ratio))
        self.reference, self.updated_at = reference, now
        self.status = dict(
            mode='ACTIVE',controller=self.controller,selected=True,reason=plan.entry_reason,
            target_speed_m_s=reference, ceiling_m_s=ceiling,
            measured_speed_m_s=float(measured_speed) if feedback else None,
            motion_feedback=feedback, motion_state_source=state_source, task_ceiling_m_s=task_limit_m_s,
            initial_overspeed=plan.initial_overspeed,
            supported_length_m=float(positions[-1]),
            usable_length_m=plan.stations_m[-1], limiting_station_m=plan.limiting_station_m,
            first_supported_point_m=segment[0].tolist(),
            effort_mean_pwm=mean, applied_mean_pwm=actual,
            requested_wheels=list(wheels),
            tracking_gain=tracking_gain, high_confidence_boost=tracking_gain > 1.,
            reference_acceleration_m_s2=reference_acceleration,
            **self.confidence_status,
        )
        if dwpp_status is not None:
            self.status.update(dwpp=dwpp_status,reference_angular_rad_s=solution.angular,
                               effective_curvature_m_inv=solution.curvature)
            self.dwpp_proposal=(wheels,solution.linear,solution.angular)
        return wheels
